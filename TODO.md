# Performance TODO

Remaining performance work for Flash Next on the Mac Studio (M5 Max, 36 GB). Measurements
behind these notes are in [MACM5MAX36GB.md](MACM5MAX36GB.md).

Current state, with `config-source.json`:

- The 6,060-token prompt takes 7.41 s. Prompts of 1–4k tokens take 6.4–6.6 s.
- Generation runs about 25 tok/s after a short prompt and 21.8 tok/s after the long one.
- Peak process memory is about 9.4 GB, with the uncommitted phase-switch memory fix (12 GB without it
  when a long prompt follows a generation).

## Remaining

| # | Name | What it does | Predicted benefit | Extra RAM |
|---|---|---|---|---|
| P3 | Merged expert sum | Merges the GPU steps that add up each token's 10 expert outputs into one kernel | −0.1–0.2 s on long prompts, +1–2% generation | none |
| P4 | Merged routing top-k | Merges the router's softmax, top-10 selection and weight normalization into one kernel | −0.1–0.2 s on long prompts, +1–2% generation | none |
| P5 | Merged state update and norm | Merges the delta-net state update with the normalization after it | −0.1–0.2 s on long prompts, +1–3% generation | none |
| N2 | Draft words from the text so far | When the answer repeats text from the prompt or earlier output, drafts several words and checks them in one pass | Large on repetitive text such as code edits, summaries and quotes (possibly +30–100%); about zero on free-form writing | Small: delta-net state copies, roughly tens of MB (estimate) |
| N4 | MTP draft model | A small extra layer predicts the next word, which the main model then checks | Uncertain: slower on another engine's 32 GB Mac that streams experts | +1.2 GB, plus a ~5 GB download (not in the GGUF) |

The predicted benefits are estimates from the GPU trace and earlier measurements. They have not
been tested.

- **P3–P5:** prompts of 1–4k tokens are limited by the SSD's read speed, so the kernel merges
  speed up only prompts of about 5k tokens or more, where the GPU is the limit, and generation.
- **N2 prerequisites:**
  - The decode path handles one token at a time (`window_max_rows = 1`). A multi-token check
    switches to the prefill phase, which clears the decode expert cache.
  - Rejected drafts need the delta-net state saved and restored.
  - Output may not be identical to today's, so N2 needs a measured quality comparison.

## Tried and not kept

| # | Idea | What happened | Why it was not kept |
|---|---|---|---|
| P1 | Read only the experts predicted for each prompt batch | A probe found the guesses accurate (95–97% of used experts with 20 guesses per token), but the guessed set is 20–30% larger than the set used | Estimated −0.9 s at 1.2k tokens, −0.4 s at 4k and nothing below 1k tokens. Not enough for the added code and graph changes |
| P2 | Start the first layer's expert read before the PLE lookups | Implemented and measured. No change on any prompt (±0.05 s), with identical output | With two staging buffers the reader can run only two layers ahead. It loses the head start later in the prompt, and 1–4k prompts are limited by the SSD throughout |
| — | Third prefill staging buffer | Considered, not tried | +0.85 GB during prefill for an estimated 0.1–0.2 s, and only on prompts of about 6k tokens |
| N3 | Prefetch two layers ahead during generation | A probe found 86% of misses are experts the current guess never picks; only 14% are guessed but late. Two layers ahead, 8 guesses cover 50% of used experts, against 58% one layer ahead | Read timing is not the problem. Catching more misses needs about 16 guesses per layer, and N1 showed extra guesses cost more than they save |
| R2a | Persistent read helpers for the lookahead prefetch | Implemented and measured. Replacing about 150–180 thread starts and joins per token with persistent helpers changed nothing: 24.00 against 24.01 tok/s after a short prompt, 20.60 against 20.67 after the long one | The thread starts run on the background worker and were not a measurable cost. Reverted. A persistent pool for PLE page reads (estimated ≤1%) was not built for the same reason |
| R2b | Start next-layer prefetch without waiting for demand reads | Implemented as an option, with 2 or 4 prefetch read threads while demand reads run. Generation was 6–9% slower | Prefetch reads take SSD bandwidth from the demand reads the GPU is waiting for. Waiting for demand reads, as today, is kept. Reverted |
| N1 | Prefetch depth 12 or 16 | Measured: 24.8 and 23.8 tok/s after a short prompt, against 25.0 at depth 8 | Slower than depth 8, with 10–12 GB peak memory and far more SSD reads. Depth 8 is kept |
| G2 | Larger protected decode cache (48 or 96 slots) | At most +1.5% after a short prompt and +3% after the long one | +2–7 GB of memory, and more SSD reads, because the bigger cache takes memory from the macOS file cache. The 72-token prompt slowed from 1.13 s to 1.85 s |
| G5 | Encode the next Metal split while the GPU is busy | Implemented. Encoding left the critical path, but GPU wait time did not shrink | Generation was 1.5% slower (21.61 → 21.28 tok/s). Reverted |
| — | 32 parallel PLE readers instead of 16 | Measured: 8.93 s against 8.97 s on the long prompt | 0.04 s is within noise. 16 is kept |
| — | Two GPU-side expert banks for prefill | Considered, not built | It would remove only the 0.27 s bank copy, add about 1 GB during generation and change the model loader. Double-buffered host staging was built instead |
| — | Neural Engine (ANE) prefill backend | Evaluated from the GPU trace | Routed-expert work is 1.32 s of a 7.07 s GPU prompt, so even free expert work would save at most about 13%. The ANE cannot use these quantization formats, and 16-bit copies would be about 6× larger |

The P1 and N3 probe patches, the R2 patch and their results are kept in
`/Volumes/Data/moeshift-scratch/p1probe`, `/Volumes/Data/moeshift-scratch/n3probe` and
`/Volumes/Data/moeshift-scratch/r2`.
