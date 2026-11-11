# Neural Engine (ANE) review and next experiments

A review of the Neural Engine work recorded in [MACM5MAX36GB.md](MACM5MAX36GB.md) and
[TODO.md](TODO.md), written on 2026-11-10, with proposals for what to try next. It is a reading
of the notes plus one look at the Metal mat-vec dispatch in
`ggml/src/ggml-metal/ggml-metal-ops.cpp`. **Nothing here was measured for this review.** Every
gain below is an estimate from the existing measurements, and each proposal names the cheapest
test that would confirm or kill it.

Sections reviewed: "Implications for an ANE prefill backend", "Draft-model acceptance (N5
feasibility)", "Speculative decoding (phases A and B)", "Draft-model speculation (phase D,
parked)", "n-gram drafting on agent-style output (N2 measurement)", "Expert predictor on the
Neural Engine (N7)" stages 1–4, "Cache bounds for expert reads", "Learned file-cache policy (N8)"
and "What a GB of file cache is worth".

## Summary

- The two conclusions already reached hold: expert prediction on the ANE (N7) and expert compute
  on the ANE (prefill backend) are dead ends.
- One parked conclusion is worth reopening: the draft model on the ANE (N5). It was judged with
  3-token drafts and 4-row checks. With 1-token, confidence-gated drafts and 2-row checks the
  arithmetic is different, and most of the parts already exist.
- One new use: the small model on the ANE as a second lane for agent side requests.
- One related change that is not about the ANE: a 2-row shared-weight mat-vec kernel for the
  quantized dense weights, which would make every 2-row check cheaper.

## Review of the existing conclusions

### ANE prefill backend: conclusion right, one reason outdated

- The ceiling stands. Routed-expert work is 1.32 s of 7.07 s of GPU time, so free expert compute
  saves at most about 13%, and prompts under about 5k tokens are limited by the 40 GB expert read,
  which the ANE cannot speed up.
- The stated reason "the ANE cannot use these quantization formats, so 16-bit copies would be
  about 6× larger" is contradicted by N7 stage 3a, where 4-bit palettized weights kept all 1,364
  operations on the ANE. A 4-bit palettized expert copy would be about 1.3× the IQ3_XXS size, not
  6×.
- The stronger blocker is probably that Core ML bakes weights into the compiled model. Streamed
  experts would have to be runtime inputs, which the ANE is not known to run well. This is from
  general knowledge of Core ML, not checked here. If the prefill section is ever revised, this is
  the reason to state.

### Nothing serial belongs on the ANE

Stage 3a gives a usable rule. The fp16 small-model step read 716 MB of weights in 6.78 ms, about
105 GB/s. The GPU's mat-vec kernels reach 355–384 GB/s (B1). Work moved from the GPU to the ANE
in series therefore runs about 3.5× slower, before the 0.1–0.3 ms dispatch. Only work that runs
in parallel with the GPU can gain.

- A decode layer is about 0.46 ms of GPU time (22 ms over 48 layers), about the size of one
  dispatch, so no per-layer split can pay. "Neural Engine work during generation: considered, not
  kept" is correct.
- The output head is the one block that is not per-layer, but at fp16 it is about 1.27 GB
  (248,320 × 2,560), roughly 12 ms on the ANE against an estimated 1.5 ms on the GPU.
- During long prompts a layer takes about 150 ms, so dispatch cost stops mattering, but the ANE
  would add perhaps 10–15% of the GPU's throughput to the dense ops only. That agrees with the
  "few percent, only at 5k tokens or more" estimate for the shared expert.

### N7 (expert predictor): the stopping point is sound

- The linear fit in stage 4 (+0.16% per miss caught, −0.14% per wasted read) reproduces both
  measured rows it was fitted to: exact guesses from layer 8 (92 caught, +15%) and 8 of 10 right
  (58 caught, 80 wasted, −2%). Note that 8 of 10 right caught 58 misses, not the 74 that 80% of 92
  would give: wrong reads also push useful experts out of the 20-slot window.
- The "85% right" bar is in terms of all guesses, including experts already resident. In terms of
  reads actually issued, the fit puts break-even near one miss caught per wasted read.
- The top-2 small-model heads show the real problem. About 80 guesses per token at 72% precision
  caught 9 of 165 misses. The predictor mostly names experts that are already in the window; the
  misses are the context-driven experts that a token-only model predicts worst (25–35% decided by
  the token in the middle layers).
- The ceiling is lower than stage 1 suggests. Skipping reads gives +45–48%, but exact guesses at
  step start give +20%, and a fully warm file cache still costs about 8 ms per token over skipped
  reads. More than half of the "read cost" is copying and per-layer handshakes, which no predictor
  removes.
- The bandwidth explanation in stage 3a checks out: 716 MB × 25 steps/s is about 18 GB/s.

**One untested gap.** Every head was trained to name the experts a token uses, and scored by
window replay afterwards. None was trained on the misses themselves (used and not resident), with
the window contents or recency as an input. `n7i/patch_resid.py` already dumps the resident sets
and demand misses per step and layer, so this is an offline retrain. Expected to fail, for the
reason above, but it is the only cheap check left before closing N7 for good.

### N5 (draft model): the parked conclusion depends on the draft length

"Drafting itself took only 12.6 ms per check, so moving it to the ANE matters only after that"
was true for the configuration measured: 3-token drafts, 4-row checks of about 218 ms, 40–50%
acceptance. It does not carry over to 1-token drafts. See the next section.

## Proposal 1: confidence-gated 1-token drafts from the small model on the ANE

### Why the arithmetic changes

- A 2-row check costs about 1.6 single steps (N2, whole-file run). An accepted check yields 2
  tokens for 1.6 steps; a rejected one yields 1 token for 1.6 steps. Break-even acceptance is
  about 62%.
- Ungated, Qwen3.5-0.8B's top token agreed with Flash Next's at 72% of positions (54% on the
  story, 92% on code editing). Drafting only when the draft model is confident should raise the
  acceptance of the drafts that are made well above that, at the price of covering fewer tokens.
- On the GPU a drafted token cost about 4 ms (12.6 ms for 3), around 7% of a 2-row check, taken
  from the same GPU. On the ANE the measured cost to the big model was 0.4–0.7%.
- The ANE model can draft ahead. A check of `[t, d1]` returns the true token after `t` and, if
  `d1` was right, the true token after `d1`; the next check then needs a draft one token further
  on. While the big model runs the check (about 60 ms), the ANE takes two steps along its own
  greedy path: `d2` after `d1`, then `d3` after `d2` (about 12 ms). If the check accepts `d1` and
  its second row equals `d2`, `d3` is ready as the next draft and drafting adds no latency.
  Otherwise the ANE restarts from the true token (about 6 ms), or the engine takes one undrafted
  step while it catches up.

### Estimate (not measured)

If confident drafts produce 40–60% of the tokens at 95% acceptance, generation is about 7–11%
faster on prose and thinking, where n-gram drafting gains nothing today. n-gram drafting stays
first choice where it fires; the model draft fills the steps where it does not.

### What exists and what is missing

| Exists | Missing |
|---|---|
| Qwen3.5-0.8B with real weights on the ANE, 4-bit, 4k ring buffer, layers 0–17, 5.9–6 ms per step (stages 3b, 3d; `/Volumes/Data/moeshift-scratch/ane/small/`) | The remaining layers and the output head (248,320 × 1,024: about 127 MB at 4-bit), and their step time |
| 2-row decode steps and rollback under LES (phases A, B); enabled in `config-source.json` | `--spec-draft-p-min` accepted in the profile (`moeshift_model_options`) |
| `spec-type: draft-simple` working end to end with the draft-context rollback fix (phase D) | A drafter interface that takes tokens from an out-of-process or Core ML model instead of a llama context |
| N5 agreement tool and six reference answers (`/Volumes/Data/moeshift-scratch/n5`) | The draft's top probability per position |

### Test order, cheapest first

1. **Offline, no engine change.** Extend the N5 tool to record the draft model's top-token
   probability at every position. For thresholds from 0.5 to 0.99, report coverage (share of
   positions drafted) and acceptance (share of those that match). Compute the speedup with a
   1.6-step check: with `C` drafted checks at acceptance `a` over `N` tokens, cost is
   `1.6·C + (N − C·(1 + a))` steps. **Stop if no threshold gives about 5% on the prose and
   explanation samples**, roughly coverage under 30% at 90% acceptance.
2. **On the GPU, existing code.** `spec-type: draft-simple`, `spec-draft-n-max: 1`,
   `LLAMA_MOE_DECODE_ROWS: 2`, the best `p-min` from step 1, interleaved against
   `config-source.json` on the 12 prompts used for B2. This includes the GPU drafting cost, so it
   is a lower bound for the ANE version. Subtract the logged drafting time to estimate the ANE
   result.
3. **Agreement of the 4-bit ANE model.** Stage 3b measured cosines of 0.91–0.94 against fp32 at
   4-bit; whether its top token still agrees as often is untested. Repeat step 1 with the Core ML
   model's logits. fp16 (716 MB for 18 layers) is the fallback.
4. **Only then** build the ANE drafter into the server.

### Costs and risks

- **Output changes.** 2-row checks round differently from 1-row steps, so greedy output differs at
  near-ties, as it already does with 2-row n-gram checks on 2 of 12 prompts. Hundreds of checks
  per reply instead of a few make this more frequent (phase D: five of six prompts). AGENTS.md
  requires a measured quality comparison for this.
- **Memory.** About 250–350 MB for the 4-bit model, head and ring buffer, worth about 0.6–0.9% of
  generation at the measured 2.3–2.9% per GB of file cache.
- **Prompts.** The small model must process each prompt too. Token by token at 6 ms that is 36 s
  for 6k tokens, so it needs a batched Core ML function or the handover described in stage 3b
  (prefill by llama.cpp on the GPU, state passed to the ANE). Phase D measured +0.9 s on the long
  prompt for a GPU draft prefill.
- **SSD.** Core ML compiled into the internal SSD's temporary folder before compilation was
  pointed at `/Volumes/Data` (stage 3a). Keep it there.

## Proposal 2: a 2-row shared-weight kernel for the quantized dense weights (not ANE)

This lowers the 1.6-step check cost, which helps n-gram drafting today and lowers the break-even
for proposal 1.

In `ggml_metal_op_mul_mat` (`ggml/src/ggml-metal/ggml-metal-ops.cpp`, around line 2396) the
small-batch kernel (`mul_mv_ext`), which reads each weight row once for several input rows, is
used for f32, f16, bf16, Q4_0–Q8_0 and IQ4_NL from 2 rows, for the K-quants only from 4 rows, and
for no other IQ type. Everything else with fewer than 9 rows falls to the plain mat-vec kernel,
which handles each input row on its own.

- The output head is Q5_K (417 MiB), so a 2-row check reads it twice.
- If the attention and shared-expert weights are IQ3 or K-quant types, the same applies to them.
  **Their types were not checked for this review**; `gguf-dump` on the first GGUF shard settles it.
- "Quantized dense mat-vecs (attention, shared expert, output head)" are 7.40 ms of the 23.4 ms of
  kernel time per token, and the cost is the bytes read (B1). If the second row could share one
  pass, a 2-row check might fall from about 1.6 to about 1.4 steps. That would raise an
  all-accepted n-gram run from +23% to roughly +40%, and lower the draft break-even from 62% to
  about 43%. Estimated, not measured.
- First step: a GPU trace or per-kernel timer of a 2-row check against a 1-row step, to see which
  kernel categories double. The routed experts will (each row has its own experts); the question
  is how much of the rest does.
- A changed kernel may round differently, so check that 2-row output stays identical to today's
  2-row output.

## Proposal 3: the small model on the ANE as a side-request lane

Agent clients send small side requests between turns (titles, summaries, compaction). M2 added
cheap-first eviction because of them, with 12 side requests between two turns in the scripted
session. On Flash Next each one costs a prefill of 2–6 s, expert reads from the SSD, prompt-cache
pressure, and it blocks the main conversation, which has one slot.

A small model served on the ANE under a second model alias would take those requests while the
main model keeps generating, at the measured 0.4–0.7% cost. Clients with a separate small-model
setting (opencode has `small_model`) can be pointed at it without guessing which requests are
side requests.

- **First:** S1 (the agent-client soak test) should count side requests and their prompt sizes. If
  a real session sends few, this is not worth building.
- **Needs:** a batched prefill function for the small model, the output head, sampling, and a
  route in the server. Much of this is shared with proposal 1.
- **Costs:** small-model quality on those requests (0.8B or 2B), and the same memory as
  proposal 1 if both use one loaded model.
- A GPU-hosted small model through llama.cpp is the simpler baseline to compare against: no Core
  ML, but it takes GPU time and about 0.8 GB.

## Not recommended

| Idea | Why not |
|---|---|
| Image encoding on the ANE | A known good use of the ANE, but only relevant if the model is served with images |
| Shared expert on the ANE during long prompts | Agreed with the existing estimate: a few percent, only at 5k tokens or more, for the memory of a second copy of the weights |
| Next-token expert prefetch (draft the next token, read its layer 0–7 experts from a token table) | Draft agreement (72%) times the token's share of those experts (73% at layer 0, 45–50% at layers 1–7) gives 35–50% precision, at or below break-even; stage 2c already measured next-token prediction from the small model at 36% recall with 8 guesses |
| Output head or any other serial block on the ANE | About 3.5× slower than the GPU per byte of weights (see above) |
| More work on expert prediction | Closed, apart from the one offline check under N7 above |

## Suggested order

1. Proposal 1, step 1 (offline threshold sweep with the N5 tool). An hour or two, and it decides
   whether proposal 1 continues.
2. Proposal 2, first step (which kernels double in a 2-row check). Independent of the ANE and
   useful for the n-gram drafting already enabled.
3. S1, counting side requests, before any work on proposal 3.
4. The N7 miss-targeted retrain, if a definite close-out of N7 is wanted.
