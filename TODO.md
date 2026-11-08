# Performance TODO

Remaining performance work for Flash Next on the Mac Studio (M5 Max, 36 GB). Measurements
behind these notes are in [MACM5MAX36GB.md](MACM5MAX36GB.md).

Current state, with `config-source.json`:

- The 6,060-token prompt takes 7.41 s. Prompts of 1–4k tokens take 6.4–6.6 s.
- Generation runs about 25.8 tok/s after a short prompt and 22.5 tok/s after the long one (with C1
  and P4; measured in a period when this machine ran about 6% slower than earlier, so compare only
  interleaved runs).
- Peak process memory is about 9.4 GB (12 GB before the phase-switch memory fix, when a long prompt
  follows a generation).

## Remaining

| # | Name | What it does | Predicted benefit | Extra RAM |
|---|---|---|---|---|
| N6 | Better expert guesses | **N6a built (uncommitted, off by default):** late prefetch routes each next layer on the previous layer's output and reads its guesses during its attention (`LLAMA_MOE_LATE_PREFETCH=8`, `LLAMA_MOE_LATE_READ=8`; see [late prefetch](docs/les/late-prefetch.md)). It cuts demand misses by 42% but gains only +1.8% after a short prompt and +0.7% after the long one, since the guess's GPU time and the slow tail of extra reads take most of it. Possible next steps: a quantized router copy just for the guess (it is about 10 of 28 µs per layer), and reading by confidence instead of a fixed count. N6b (trained predictor) and N6c (history) not started | Measured +1.8% / +0.7%; identical output | About 4 MB |
| P0 | Protected cache off | `LLAMA_MOE_PROTECTED_SLOTS=0`: the page cache already holds what the 24 protected slots per layer do. **Measured (A/B):** +1.1% after a short prompt, +0.9% after the long one, identical output, 16% less SSD reading, 1.9 GB less memory while generating (peak unchanged, set by prefill). **Applied** to `config-source.json` and the Flash Next profile in `configs/moeshift.json` | +1% / +0.9%; identical output | 1.9 GB **less** while generating |
| B2 | Q8_0 hyper-connection and router weights | Stores the four bf16 hyper-connection projections per layer and the router as Q8_0 in a copy of the GGUF, about halving the 1.45 GB of bf16 weights read per token | About +4–5% generation (estimated from measured kernel speeds). Output changes and needs a measured quality comparison (teacher-forced agreement against the bf16 model). Lower priority: Qwen's FP8 release, NVIDIA's NVFP4 release and the ISTA-DASLab GSQ-RCO GGUF all keep exactly these tensors in bf16, and no release found quantizes them, so they are considered sensitive (router errors change expert choice; hyper-connection errors affect every layer's stream mixing) | About 0.7 GB **less** |
| P3 | Merged expert sum | Merges the GPU steps that add up each token's 10 expert outputs into one kernel | Small: the adds are already partly merged by ggml-metal's add fusion; that whole category is 0.63 ms per token (GPU trace), so about 0.2–0.3 ms (under 1%). P4 measured about 1.5 µs per removed kernel, so merging the adds alone gains little | none |
| P5 | Merged state update and norm | Merges the delta-net state update with the normalization after it | At most about 0.2 ms per token; changes the large shared delta-net kernel, which also writes rollback snapshots. Probably not worth it | none |
| N2 | Draft words from the text so far | When the answer repeats text from the prompt or earlier output, drafts several words and checks them in one pass | Large on repetitive text such as code edits, summaries and quotes (possibly +30–100%); about zero on free-form writing | Small: delta-net state copies, roughly tens of MB (estimate) |
| N4 | MTP draft model | A small extra layer predicts the next word, which the main model then checks | Unlikely: N5 showed checks are limited by expert reads, not drafting, and another engine found MTP slower on a 32 GB Mac that streams experts | +1.2 GB, plus a ~5 GB download (not in the GGUF) |

The predicted benefits are estimates from the GPU trace and earlier measurements. They have not
been tested.

- **P3–P5:** prompts of 1–4k tokens are limited by the SSD's read speed, so the kernel merges
  speed up only prompts of about 5k tokens or more, where the GPU is the limit, and generation.
- **N2 prerequisites:**
  - The decode path handles one token at a time (`window_max_rows = 1`). A multi-token check
    switches to the prefill phase, which clears the decode expert cache.
  - Rejected drafts need the delta-net state saved and restored.
  - Output may not be identical to today's, so N2 needs a measured quality comparison.
- **N2, N4 and N5 groundwork** (phases A and B, uncommitted): decode steps of up to 4 tokens and
  speculative decoding under LES work, with identical output and no cost when no draft is made
  (see [speculative decoding](docs/les/speculative-decoding.md)). n-gram drafting (N2) works
  today: editing code ran 8.5% faster.
- **Check cost:** a 4-token check costs about 2.45 single steps, because drafted tokens miss the
  expert cache like ordinary tokens. Drafting pays only above about 2.45 accepted tokens per
  check. Prefetching guessed experts for all drafted tokens before the check, or longer drafts
  with more rows (which needs wider windows and more memory), could lower the threshold.

## Done

| # | Name | What it does | Measured | Extra RAM |
|---|---|---|---|---|
| C1 | Fewer copies | **Done (C1a and C1c in commit `0731c56`, C1b in `81d0038`, its memory-reuse fix in `e52e51d`).** C1a: the one-token delta-net kernel writes its new state straight into the recurrent cache (+6.4% / +5.0% generation after a short / the long prompt). C1c: 131 small `cont`s per token removed (+1.7% / +4.1%). C1b: the one-token kernel reads its state from the cache row, skipping the `get_rows` gather (+2.8% / +1.2%; limited to one-token steps, since prompt batches were about 1% slower with it). All keep output identical | Done | none |
| P4 | Merged routing top-k | **Done (commit `81d0038`):** one Metal kernel computes the router's argsort, the top-10 weights, their sum, clamp and division (5 kernels into 1, in all 48 layers). The softmax stays separate, and the sort is the same bitonic sort, so ties are ordered identically | Measured +0.9% after a short prompt, +0.7% after the long one (about 0.3 ms per token, a third of the estimate: about 1.5 µs per removed kernel); identical output | none |

## Parked

| # | Idea | Status | To revisit |
|---|---|---|---|
| N5 | Small draft model (Qwen3.5-0.8B), later on the ANE | Works end to end under LES (phase D): `spec-type: draft-simple` with the draft model in `/Volumes/Data/models/n5`. Measured 9.6–21.5 tok/s against 21.2–25.7 on five prompts, break-even on code editing; output differed at near-ties on five of six prompts. The draft-context rollback fix it needed is kept | A 4-token check costs 2.5–4.5 single steps because every drafted row reads its own experts, and rejected rows waste those reads. It needs cheaper checks first: prefetching the experts of drafted tokens, a more accurate drafter, or confidence-limited drafts (`--spec-draft-p-min`, not yet accepted in the profile; add it to `moeshift_model_options`) so likely rejections are not checked. Drafting itself took only 12.6 ms per check, so moving it to the ANE matters only after that |

## Tried and not kept

| # | Idea | What happened | Why it was not kept |
|---|---|---|---|
| P1 | Read only the experts predicted for each prompt batch | A probe found the guesses accurate (95–97% of used experts with 20 guesses per token), but the guessed set is 20–30% larger than the set used | Estimated −0.9 s at 1.2k tokens, −0.4 s at 4k and nothing below 1k tokens. Not enough for the added code and graph changes |
| P2 | Start the first layer's expert read before the PLE lookups | Implemented and measured. No change on any prompt (±0.05 s), with identical output | With two staging buffers the reader can run only two layers ahead. It loses the head start later in the prompt, and 1–4k prompts are limited by the SSD throughout |
| — | Third prefill staging buffer | Considered, not tried | +0.85 GB during prefill for an estimated 0.1–0.2 s, and only on prompts of about 6k tokens |
| N3 | Prefetch two layers ahead during generation | A probe found 86% of misses are experts the current guess never picks; only 14% are guessed but late. Two layers ahead, 8 guesses cover 50% of used experts, against 58% one layer ahead | Read timing is not the problem. Catching more misses needs about 16 guesses per layer, and N1 showed extra guesses cost more than they save |
| R2a | Persistent read helpers for the lookahead prefetch | Implemented and measured. Replacing about 150–180 thread starts and joins per token with persistent helpers changed nothing: 24.00 against 24.01 tok/s after a short prompt, 20.60 against 20.67 after the long one | The thread starts run on the background worker and were not a measurable cost. Reverted. A persistent pool for PLE page reads (estimated ≤1%) was not built for the same reason |
| R2b | Start next-layer prefetch without waiting for demand reads | Implemented as an option, with 2 or 4 prefetch read threads while demand reads run. Generation was 6–9% slower | Prefetch reads take SSD bandwidth from the demand reads the GPU is waiting for. Waiting for demand reads, as today, is kept. Reverted |
| B1 | Faster bf16 mat-vec kernel | Measured (microbenchmark at the real shapes): the bf16 kernel reaches 355–384 GB/s, the fastest of the kernels measured (f16 370, q8_0 320, q4_K 259 GB/s on a large mat-vec) | Little headroom: about 0.3–0.5 ms per token at best. The 5.4 ms per token is the 1.45 GB of bf16 weights, not the kernel; see B2 |
| I1 | Cache the pooled indexer keys | Measured (microbenchmark of the pooling chain, 12 layers, one token): pooling only the newest block instead of all of them saves 0.26 ms per token at the long prompt's 6.6k context (about 0.6%), 1.0 ms at 16k, 2.1 ms at 32k and 4.5 ms at 64k. C1c had already cut this chain's cost by more than half (13.6 to 5.1 ms per token at 64k) | Under 1% below about 10k tokens of context. It needs persistent state that must be invalidated on rollback, sequence removal, shifts and state loads, a reusable graph shape, and about 96 MB at 64k context. Worth revisiting only for regular use at 16k tokens or more |
| G6 | One command buffer per token | Prototype built: the expert step ran on a host thread inside the Metal graph, handed off by shared events (2 splits per token instead of 98); identical output. +0.8–1.2% after a short prompt, −0.5% after the long one; with reads skipped it saved 0.5 of the 5.1 ms of idle GPU | The host step took 19 µs; the rest of the ~100 µs per layer is on the GPU side (resuming after event waits, encoder barriers), which one command buffer does not remove. Reverted (patch in `/Volumes/Data/moeshift-scratch/g6`) |
| Z1 | Zero-copy experts from the page cache | Prototype (no-copy buffer per expert over its covering pages, address table in `kernel_mul_mv_id`): identical output, only 3–6% of expert uses missed the page cache, up to 3 GB less memory. All layers zero-copy: 32% slower (binding experts being read stalls encoding; missing pages fault in one by one). Hybrid (only page-cache hits in place, the rest through the windows): 10% slower | The copies it removes were mostly hidden behind GPU work already; residency checks (1.8 ms per token) and binding about 480 new file-backed buffers per token (1–2.4 ms of encoding) cost more. Patches in `/Volumes/Data/moeshift-scratch/zc` |
| L0 | Guess layer 0's experts | Probed: from the token embedding (layer 0's input before attention), 2–6% of its experts, about chance; from the previous step's router ranks 11–26, 5–6% of its misses. An LRU replay found 407–418 distinct experts in 511 steps | Nothing predicts layer 0 before its attention; only a much larger window for that layer helps (128 slots, about +156 MB, would save about 3 of ~100 misses per token) |
| — | Neural Engine work during generation | Considered | Layers run strictly in order, a Core ML dispatch takes 0.1–0.3 ms, and generation is limited by reads and handshakes that the ANE cannot speed up. Only image encoding and the shared expert during long prompts could run on it in parallel |
| N1 | Prefetch depth 12 or 16 | Measured: 24.8 and 23.8 tok/s after a short prompt, against 25.0 at depth 8 | Slower than depth 8, with 10–12 GB peak memory and far more SSD reads. Depth 8 is kept |
| G2 | Larger protected decode cache (48 or 96 slots) | At most +1.5% after a short prompt and +3% after the long one | +2–7 GB of memory, and more SSD reads, because the bigger cache takes memory from the macOS file cache. The 72-token prompt slowed from 1.13 s to 1.85 s |
| G5 | Encode the next Metal split while the GPU is busy | Implemented. Encoding left the critical path, but GPU wait time did not shrink | Generation was 1.5% slower (21.61 → 21.28 tok/s). Reverted |
| — | 32 parallel PLE readers instead of 16 | Measured: 8.93 s against 8.97 s on the long prompt | 0.04 s is within noise. 16 is kept |
| — | Two GPU-side expert banks for prefill | Considered, not built | It would remove only the 0.27 s bank copy, add about 1 GB during generation and change the model loader. Double-buffered host staging was built instead |
| — | Shared expert on the ANE during long prompts | Considered, not built | Real parallel work, but at most a few percent and only on prompts of about 5k tokens or more, for the memory of 16-bit shared-expert weights |
| — | Neural Engine (ANE) prefill backend | Evaluated from the GPU trace | Routed-expert work is 1.32 s of a 7.07 s GPU prompt, so even free expert work would save at most about 13%. The ANE cannot use these quantization formats, and 16-bit copies would be about 6× larger |

The P1 and N3 probe patches, the R2 patch and their results are kept in
`/Volumes/Data/moeshift-scratch/p1probe`, `/Volumes/Data/moeshift-scratch/n3probe` and
`/Volumes/Data/moeshift-scratch/r2`.
