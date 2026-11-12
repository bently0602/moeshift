# Neural Engine (ANE) review, second round

A second review of [ANE.md](ANE.md), written on 2026-11-10. Claude Opus 5.5 reviewed the
proposals, Claude Fable 5.2 (the author of `ANE.md`) responded, and Opus replied. This file
records where the two reviews ended up. The only new data here is a dump of the GGUF's dense
tensor types and a look at the Metal kernels. **Nothing was benchmarked.** Every gain below is
an estimate from the measurements in [MACM5MAX36GB.md](MACM5MAX36GB.md).

## Summary

- **Proposal 2 (2-row kernel for the quantized dense weights) moves to first.** It is not an
  ANE idea, but it is now the strongest item. About 2.1 GB of dense weights per token are in
  types the 2-row path skips. For the K-quants the kernels already exist and only a dispatch
  check keeps them out.
- **Proposal 1 (ANE drafter):** two corrections to the break-even arithmetic. The 1.6-step
  check cost covers back-to-back checks but leaves out the cost of rejections. Confidence gating
  should keep rejections rare, so the risk is lower than the first reply to the review said, but
  only the GPU run settles it.
- **Proposal 3 (side-request lane):** only short requests such as titles. Compaction stays on
  the main model.
- **The ANE's bandwidth was estimated wrong in `ANE.md`** (and again in the first reply to the
  review). It is about 200 GB/s with a fixed cost of several milliseconds per step. The
  conclusions do not change.
- **The order of work is unchanged by the discussion:** the K-quant gate, then the IQ kernels,
  then the GPU model-draft test, and only then any ANE work.

## Corrections to ANE.md

| `ANE.md` says | Corrected | Source |
|---|---|---|
| Break-even acceptance at a 1.6-step check: 62% | 60%: a check yields `1 + a` tokens for 1.6 steps, so it pays when `a > 0.6` | Opus |
| Break-even at a 1.4-step check: 43% | 40% | Opus |
| Attention and shared-expert weight types: not checked | Checked; see the table below | Opus |
| ANE reads weights at about 105 GB/s, so serial work is about 3.5× slower per byte | About 200 GB/s plus a fixed cost of 3.5–5.5 ms per step; see below | Fable, confirmed by Opus |
| The Core ML runtime-weights limit is "the stronger blocker" for an ANE prefill backend | Moot: the 13% ceiling already rules the backend out | Opus |

## Proposal 2: dense weight types

From the first GGUF shard, the dense tensors excluding the routed experts, largest first
(`gguf-py`, `GGUFReader`):

| Weights | Types | Size | 2-row small-batch kernel (`mul_mv_ext`) |
|---|---|---:|---|
| Output head | Q5_K | 417 MiB | Kernel exists; dispatch only allows it from 4 rows |
| `attn_qkv`, `attn_gate`, `attn_q`, `attn_output`, `ssm_out` | Q6_K, Q4_K, Q5_K | most of about 1.7 GB | Same: kernel exists, allowed only from 4 rows |
| The rest of attention and `ssm_out`; shared-expert gate and up | IQ4_XS, IQ3_S | | No kernel |
| Shared-expert down | IQ4_NL | 35 MiB | Covered from 2 rows |
| Hyper-connection projections, router, PLE key, indexer | BF16 | 1.4 GB | Covered from 2 rows |
| Per-layer token embedding | IQ4_NL | 27.5 GB | Not a mat-vec (row lookup) |

The quantized dense weights come to about 2.1 GB per token, which matches the 7.40 ms of
"quantized dense mat-vecs" in the generation trace. A 2-row check today reads all of it twice.

- **K-quants:** `kernel_mul_mv_ext_q{4,5,6}_K_f32_r1_2` are already instantiated in
  `ggml-metal.metal`. Only the `ne11 >= 4 && ne11 <= 8` condition for K-quants in
  `ggml_metal_op_mul_mat` (`ggml-metal-ops.cpp`, around line 2425) keeps them out of 2- and
  3-row steps. Upstream probably chose that limit from measurements on older chips. Time it on
  the M5 Max at 2 rows; B1's `mv_bench.cpp` (`/Volumes/Data/moeshift-scratch/b1/`) runs these
  shapes.
- **IQ types:** `dequantize_iq4_xs` and `dequantize_iq3_s` have the same signature as
  `dequantize_q4_K` (`(block, short il, thread type4x4 &)`, 256 values per block). They should
  drop into the existing `kernel_mul_mv_ext_q4x4_f32_disp` template as new instantiations, plus
  the dispatch condition. Not compiled yet.
- **Estimated gain:** a 2-row check would fall from about 1.6 to about 1.42 single steps if the
  second row shares one pass over the weights. That makes the 2-row n-gram drafting already in
  `config-source.json` faster on copy-heavy output, and lowers the break-even for any drafter to
  about 40% acceptance.
- **The budget is consistent but does not pin the gain.** The extra 0.6 step of a 2-row check is
  about 23 ms. The second row's expert reads (up to 10–13 ms), routed-expert kernels (4.2 ms) and
  quantized dense mat-vecs (7.4 ms) add up to about that, which fits the dense work doubling.
  But copies, delta-net and attention also scale with rows and were not counted, so the second
  row's reads are probably less than 10–13 ms (its experts overlap with the first row's). About
  7 ms is the upper bound for this proposal, not the expected gain; the microbenchmark settles
  it.
- **Output:** a different kernel rounds differently, so 2-row checks may change at near-ties.
  AGENTS.md requires an output comparison against today's 2-row build.

## Proposal 1: what the 1.6-step check cost covers

The first review argued that 1.6 was measured where over 99% of drafts were accepted, and that
phase D's model-draft checks cost about twice as much as n-gram checks (218 ms against 115 ms),
so 1.6 might not apply. If a model-drafted 2-row check cost 1.9 steps, break-even would rise to
about 90% acceptance.

Fable answered that the engine cannot tell where a draft came from, so an accepted model draft
costs the same as an accepted n-gram draft, and attributed much of phase D's gap to the draft
model's memory.

Where it ended up:

- **1.6 already includes back-to-back checks.** In the whole-file run almost every step was a
  check. A 2-row step uses about 20 experts per layer, as many as the window holds, so 1.6 was
  measured with the windows churning at the worst check frequency.
- **Memory does not explain phase D.** It peaked at about 11 GB against about 10 GB for n-gram
  speculation. At about 2.5% of generation per GB of file cache, that extra gigabyte accounts for
  2–3% of a gap of nearly 90%.
- **Rejection probably does.** At 40–50% acceptance, a rejected 4-row check reads about 30
  experts per layer for discarded rows and pushes useful experts out of the 44 slots. The
  whole-file 4-row run fits this: at nearly full acceptance its checks cost about 124 ms
  (32.3 tok/s at about 4 tokens per check), close to the n-gram figure, not phase D's.
- **What this means for gated 1-token drafts:** rejections should be rare, and a rejected 2-row
  check wastes only one row of reads, so the cost should stay near 1.6. This rests on gating
  keeping acceptance high on prose, which is not measured.

So the offline threshold sweep in `ANE.md` (step 1) still cannot decide Proposal 1 on its own.
Run it at check costs of 1.6, 1.8 and 2.0. The GPU run (step 2: `spec-type: draft-simple`,
`spec-draft-n-max: 1`, `LLAMA_MOE_DECODE_ROWS: 2`, the best `p-min`) measures the real cost.
Cap the draft context size in that run, since the server estimated 7.4 GB for the draft context
in phase D, and log rejections separately to see what one costs.

**The decision point:** if the model-drafted 2-row check costs about 1.6 steps or less (after
Proposal 2), the ANE drafter is worth building. If it is near 1.9, Proposal 1 is dead.

The ANE pipelining in `ANE.md` holds: two draft steps fit easily inside the roughly 60 ms
check.

## The ANE's speed

`ANE.md` estimated about 105 GB/s from the fp16 step alone. The first review then argued from
the 4-bit step (about 41 GB/s) that the ANE is mostly limited by overhead. Both were wrong. A
fit through two points, the same model at fp16 and at 4-bit:

| Runs | fp16 | 4-bit | Weight bandwidth | Fixed cost per step |
|---|---:|---:|---:|---:|
| Stage 3a, random weights, 1k KV cache | 6.78 ms, 716 MB | 4.36 ms, 180 MB | about 221 GB/s | about 3.5 ms |
| Stage 3b, real weights, 4k KV cache | 8.77 ms | 5.99 ms | about 193 GB/s | about 5.5 ms |

The fit assumes the 4-bit model reads its compressed weights directly, with no extra decoding
cost, and the larger fixed cost in stage 3b includes the 4k KV cache. Treat both numbers as
rough.

- **The ANE reads weights at roughly 60% of the GPU's 355–384 GB/s.** The bandwidth gap is
  narrower than `ANE.md` said.
- **The fixed cost per step is what rules out serial work.** The output head at 4-bit (about
  640 MB) would take about 3 ms plus the fixed cost on the ANE, against about 1.5 ms on the GPU.
  Every "do not put it on the ANE" conclusion in `ANE.md` stands, for this reason.
- **For the drafter it is good news.** A full Qwen3.5-0.8B step with its output head (about
  127 MB at 4-bit) should take about 8 ms at short contexts. Two steps per check fit within a
  roughly 60 ms 2-row check.

## Proposal 3: side-request lane

Agreed with one refinement: only short requests such as titles belong in the lane. Compaction
sends the whole conversation, which the small model would be slow to read and poor at
summarizing. M2's cheap-first eviction already protects the main conversation's prompt cache,
so what the lane would save is the main conversation waiting behind side requests. S1 (the
agent-client soak test) should show whether that happens often enough to matter, before
anything is built.

## Still disputed

Only why phase D's checks were slow: rejection (Opus) or partly memory (Fable). It does not
change what to try next, and the GPU run in step 3 below, with rejections logged and the draft
context capped, answers it.

## Order of work

**Step 1 result (K1 in `TODO.md`):** the gate saves about 1.0 ms per 2-row check, not the 7 ms
upper bound above. The plain K-quant kernels already run 2 rows at 1.15–1.65× one row. End to end
+1.2–1.4% on copy-heavy output, identical output. It also found and fixed a half-precision bug in
the Q4_K small-batch kernel. A 2-row check now costs about 1.58 single steps instead of 1.6, so
the break-even for Proposal 1 barely moves.

1. **K-quant 2-row gate.** Change the dispatch condition, microbenchmark the K-quant `r1_2`
   kernels at the real shapes against the plain mat-vec, then run interleaved tests on the
   copy-heavy N2 tasks against the current build, with an output comparison.
2. **IQ4_XS and IQ3_S 2-row kernels.** New `mul_mv_ext` instantiations and dispatch entries;
   same tests.
3. **GPU model-draft test.** Confidence-gated 1-token drafts with a capped draft context; check
   cost and rejection cost logged. Run the offline threshold sweep first, at 1.6, 1.8 and 2.0
   steps, to choose `p-min`.
4. **ANE drafter**, only if step 3 meets the decision point above.
5. **S1**, counting side requests, before any work on Proposal 3.

**Steps 2 and 3 result:** step 2 (K2) was built and not kept: the IQ4_XS small-batch kernel was
slower, and IQ3_S's saved about 0.17 ms per check but changed output. Step 3: the offline sweep
passed at 1.6 steps per check (+4.5–11% on prose at 93–98% acceptance). On the GPU the draft model
lost 13–22% end to end, but a control run (draft model loaded, no drafts checked) showed why: its
context takes about 7.5 GB system-wide from the file cache, which slows single steps 9–33% without
any check, and drafting costs about 4 ms per call. With that held equal, a gated 2-row check cost
about 1.7 single steps and the big model's time per token fell 5.5–13%. That settles the dispute
above mostly on the memory side. With the draft context's batch capped (1.7 GB instead of 7.5), a
gated check costs 1.59–1.66 single steps like for like, and the big model's time per token is 1–9%
below the baseline's even with the draft model on the GPU; end to end the GPU drafter still loses
3–15% to its own serial GPU time, which is the part the ANE takes over. Step 3 meets the decision
point, so step 4 is open to build (`MACM5MAX36GB.md`, confidence-gated draft model).

**Step 4, first two parts:** the full 0.8B model (24 layers, norm, tied output head in 16 parts)
runs on the Neural Engine at 4-bit: 379 MB, 10.1 ms per step. Its top token agrees with Flash
Next's at 64.9%, against 71.8% for the Q8_0 model (fp16 arithmetic, 4-bit weights and the 4k
attention window each cost a few points), and running it alongside costs the big model about
1–2%. Net estimate: about +1% on chat prose, +5–6% on explanations and summaries, +10–15% on
code; not measured end to end.

**Long-context check:** on a 25k-token coding conversation the 4-bit model's agreement falls to
53.6%, against 74.3% for the Q8_0 model with the full context. The 4k attention window costs 11
points, 4-bit weights 8, fp16 arithmetic 1.5. The net estimate becomes about +5–6% on new code
and nothing on prose, so the drafter is parked; a longer window and 6- or 8-bit weights are the
untested ways back.
