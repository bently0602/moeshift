# MoEShift on a Mac with M5 Max and 36 GB

Measurements and configuration notes for Qwen 3.8 Flash Next on one machine,
collected on 2026-11-06. Results apply to this hardware, model and prompt; they
are not general benchmarks.

## Summary

- Raising `args.prefill-max-tokens` from `3072` to `6144` made long-prompt
  processing **33% faster** (15.47 s → 10.38 s for 6,060 tokens; 391.7 → 584.0
  tok/s). Peak process memory rose by 0.3 GB.
- The gain comes mostly from reading the 40 GB of prepared experts once per prompt
  instead of twice, and from much less time waiting on expert loading.
- With 3,072-token batches, about 40% of prompt-processing time was spent loading
  experts rather than computing. That limits what any faster compute unit, such
  as the Apple Neural Engine, could gain.
- A GPU trace of one 6,060-token prompt (at 6,144) found the GPU busy for 7.07 s.
  Routed-expert matrix work took 1.32 s of that (19%). Element-wise operations,
  norms and copies took the most time: 2.10 s (30%).
- A CPU profile found that each long prompt starts with about 2.1 s of serial,
  uncached 16 KiB reads from the 27 GB per-layer embedding table. The GPU has no
  work during that time. A change to `src/llama-ssd-rows.cpp` (commit
  `d38cb01`) reads these pages up to 16 at a time. It made prompt processing 12.8% faster
  (10.29 s → 8.97 s), with identical output.
- An `f16` KV cache made prompt processing 3.2% faster at a cost of about
  0.7 GiB more memory at full context. Turning off the GPU keepalive made
  generation 22% slower.
- MoEShift wrote nothing measurable to disk while running: the internal SSD wrote
  54–92 MB in total during each 4-request run, for the whole system.
- Copying each layer's experts with 8 threads instead of one (`d38cb01`)
  saved another 0.37 s (4.1%).
- Double-buffered prefill reads (`6de789b`) let one reader stream layers
  back to back. The wait for expert reads fell from 0.55 s to 0.13 s per
  prompt, saving another 0.37 s (4.5%).
- Five graph changes in `src/models/qwen4exp.cpp` (`6de789b`) avoid full
  passes over the 4-stream hyper-connection state. They saved another 0.32 s
  (4.0%), with identical output. Generation after a short prompt was 3.0%
  faster in an interleaved benchmark.
- A fused Metal kernel for the end of the hyper-connection mix (`6de789b`,
  based on Strata's `gr_epi_up`) saved another 0.16 s (2.1%). Generation was 2.0%
  faster, with identical output.
- Persistent reader threads for decode-time expert misses, with each expert's
  matrices read in parallel (`6de789b`), made generation 4.3% faster.
- Combined, with `f16` KV, the long prompt takes 7.44 s, against 10.29 s on the
  original build: 28% faster. It took 15.47 s with the original 3,072-token
  batches. A 12,121-token prompt (two batches) took 16.6 s, against 21.8 s on
  the Homebrew build, with identical output.
- Reading only predicted experts during prefill (P1) and starting the first
  layer's read earlier (P2) were measured and not kept. Prompts of 1–4k tokens
  are limited by the SSD's read speed.
- A long prompt sent after a generation peaked at 12 GB, because macOS
  compressed released decode windows instead of discarding them. Replacing
  released ranges with fresh memory (uncommitted) brings the peak to 9.4 GB,
  with identical output, prompts 2–15% faster and generation 1.5–2.5% faster.
- Tool calling works through the OpenAI-compatible API, including with
  streaming.
- The user configuration now uses `prefill-max-tokens: 6144` and an `f16` KV
  cache.
- Later (X1): with an `f16` KV cache the sparse-attention indexer's cache
  overflowed at long context, raising perplexity 2.5–4% at 32k tokens. The
  indexer cache is now always bf16, and the configuration is back on `q8_0` KV
  for half the KV memory. A single 120k-token prompt ran in about 6 minutes at
  about 19 GB peak memory, without swap (L1).
- Later (L2, uncommitted): generation slowed with context because the per-layer
  embedding input visited every cached cell for each token, and the indexer pooled
  every cached block for each token. With both fixed, generation is 12% faster at
  32k tokens of context, 20% at 64k and 30% at 120k (14.4 → 18.6 tok/s), with
  identical output and about 1.5 KiB more memory per token of context.

## Recommended configuration for this Mac

The Flash Next profile in `~/.config/moeshift/config.json`, as used for the
final measurements. JSON does not expand `~`, so the two paths must be
absolute.

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "models": {
    "qwen-3.8-flash-next-rco-gsq-iq3-xxs": {
      "args": {
        "model": "/Users/<user>/.moeshift/models/qwen-3.8-flash-next-rco-gsq-iq3-xxs/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
        "moe-experts": "/Users/<user>/.moeshift/models/qwen-3.8-flash-next-rco-gsq-iq3-xxs/experts",
        "ctx-size": 65536,
        "ctx-checkpoints": 1,
        "prefill-max-tokens": 6144,
        "moe-prefill-bank": 512,
        "moe-decode-bank": 0,
        "n-gpu-layers": 99,
        "flash-attn": "on",
        "cache-type-k": "f16",
        "cache-type-v": "f16",
        "threads": 10,
        "no-mmap": true,
        "alias": "qwen-3.8-flash-next-rco-gsq-iq3-xxs"
      },
      "env": {
        "LLAMA_MOE_QUIET": 1,
        "LLAMA_GPU_KEEPALIVE": 1,
        "LLAMA_MOE_DECODE_WINDOW": 12,
        "LLAMA_MOE_LOOKAHEAD_PREFETCH": 1,
        "LLAMA_MOE_PREFETCH_READ": 2,
        "LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH": 3,
        "LLAMA_MOE_PREFILL_PIPELINE": 1,
        "LLAMA_MOE_PROTECTED_SLOTS": 24
      }
    }
  }
}
```

Only two settings differ from the shipped profile: `prefill-max-tokens` and
the two `cache-type` entries. Everything else is the shipped default. Settings
marked "not tested" were left at the default and not compared with other
values on this Mac. See `configs/README.md` for the full reference.

| Setting | Value here | Shipped default | Effect on this Mac |
|---|---|---|---|
| `args.prefill-max-tokens` | **`6144`** | `3072` | **Measured.** Prompt processing was 33% faster for a 6,060-token prompt, because the 40 GB of experts are read once per prompt instead of twice. Peak process memory was 0.3 GB higher. Prompts up to 6,144 tokens are processed in one batch; longer prompts are split. Larger values were not tested. |
| `args.cache-type-k`, `args.cache-type-v` | **`f16`** | `q8_0` | **Superseded: `config-source.json` uses `q8_0` again** ([X1](#kv-cache-type-and-the-indexer-cache-x1): with `f16` the indexer cache overflowed at long context; `q8_0` halves the KV memory). **Measured.** Prompt processing was 3.2% faster; generation did not change. The KV cache uses 24 KiB per token instead of about 12.8 KiB: about 70 MiB more for a 6,000-token prompt, and about 0.7 GiB more at the full 65,536-token context. Output differs slightly from `q8_0`; quality was not compared. |
| `env.LLAMA_GPU_KEEPALIVE` | `1` | `1` | **Measured.** With `0`, prompt processing did not change, but generation was 22% slower. Keep `1`. It runs a small GPU job about 148 times per second while MoEShift runs, including when idle. |
| `env.LLAMA_MOE_QUIET` | `1` | `1` | `0` prints expert-loading diagnostics (`prefill #…`, `pmgmt`, `pread_wall`) to the terminal. Use it for investigation only; diagnostics are not written to files. |
| `args.moe-prefill-bank` | `512` | `512` | Not tested. The model has 512 experts per layer, and a long prompt uses almost all of them, so the bank holds a whole layer. |
| `args.moe-decode-bank` | `0` | `0` | Not tested. |
| `args.ctx-size`, `args.ctx-checkpoints` | `65536`, `1` | same | Not tested near the limit. The KV cache starts small and grows as needed, so a large `ctx-size` costs memory only when it is used. |
| `args.threads` | `10` | `10` | Not tested. The M5 Max has 18 CPU cores. Most prompt-processing work runs on the GPU. |
| `args.n-gpu-layers`, `args.flash-attn` | `99`, `on` | same | Required: expert execution needs Metal. |
| `args.no-mmap` | `true` | `true` | Not tested. Weights are read explicitly instead of memory-mapped. |
| `env.LLAMA_MOE_PREFILL_PIPELINE` | `1` | `1` | Not tested with `0`. Reads the next layer's experts while the GPU computes the current layer. |
| `env.LLAMA_MOE_DECODE_WINDOW`, `env.LLAMA_MOE_PROTECTED_SLOTS` | `12`, `24` | same | Not tested. Expert slots kept per layer during generation. Larger values use more RAM and can reduce SSD reads. |
| `env.LLAMA_MOE_LOOKAHEAD_PREFETCH`, `…_DEPTH`, `env.LLAMA_MOE_PREFETCH_READ` | `1`, `3`, `2` | same | Not tested. Predictive expert reads during generation. |

With this configuration on the original build, the long prompt takes about
9.96 s. The code changes bring it to 7.44 s (see
[Combined result](#combined-result)). None of them adds a setting, and the
Homebrew build does not include them.

Memory on this 36 GB Mac: peak process footprint is about 9.4 GB with this
configuration, and swap did not grow during inference. Before the phase-switch
memory fix, a long prompt sent after a generation could peak at 12 GB (see
[Phase-switch memory release](#phase-switch-memory-release-uncommitted)). The main risk to SSD
life is macOS swapping under memory pressure. Leave headroom before raising
`prefill-max-tokens`, `ctx-size` or the slot counts.

## System

| Item | Value |
|---|---|
| Chip | Apple M5 Max (18 CPU cores) |
| Memory | 36 GB unified memory |
| GPU | Metal 4 (`MTLGPUFamilyApple10`), `has tensor = true` |
| Storage | 500 GB internal SSD (model files are stored here) |
| macOS | 27.2 (build 26B123) |
| MoEShift | Source build of commit `2b1b3e8` (`v0.1.0-alpha.3`), built with `scripts/build-moeshift.sh` |
| Model | Qwen 3.8 Flash Next, ISTA-DASLab GSQ-RCO IQ3_XXS |
| Model files | GGUF 71 GB (two shards) + prepared experts 40 GB, in `~/.moeshift/models/` |

Model shape, from the expert manifest: 48 MoE layers, 512 experts per layer,
10 experts per token, hidden size 2,560, expert width 640. Expert tensors use
mixed quantization that varies by layer: gate and up use IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS or IQ3_S; down uses Q2_0 or IQ4_NL.

## Method

- Prompt: `benchmarks/long.json` (6,060 templated input tokens, 512 output
  tokens, `temperature: 0`, `seed: 42`, prompt cache off).
- Procedure: the manual alternative in `benchmarks/README.md`. One fresh server,
  one warmup request (includes model loading), then three measured requests
  sent with `benchmarks/long.sh`.
- Configuration: a copy of the user configuration with `env.LLAMA_MOE_QUIET: 0`
  for diagnostics. The second run also set `args.prefill-max-tokens: 6144`.
  Everything else matched the shipped Flash Next profile.
- Memory: `footprint -p` on the model worker after the runs (lifetime peak).
- Disk: internal-SSD byte counters from `ioreg` (`IOBlockStorageDriver`) before
  and after each run, plus `sysctl vm.swapusage`.
- Diagnostic logging may cost a little speed in both runs.

## Results

Medians of the three measured requests. All requests produced 512 output tokens
(`finish_reason: length`) with no cached prompt tokens.

| Measurement | `prefill-max-tokens` 3072 | `prefill-max-tokens` 6144 | Change |
|---|---:|---:|---:|
| Prompt processing time | 15.47 s | **10.38 s** | −33% |
| Prompt throughput | 391.7 tok/s | **584.0 tok/s** | +49% |
| Generation | 16.54 tok/s | 15.47 tok/s | −6% (within run-to-run variation) |
| Request wall time | 47 s | 43 s | −4 s |
| Peak process memory | 8,942 MB | 9,239 MB | +0.3 GB |
| Swap used | 1.88 MB | 1.88 MB | unchanged |

Individual requests:

| Run | 3072: prompt | 3072: generation | 6144: prompt | 6144: generation |
|---|---:|---:|---:|---:|
| Warmup | 16.25 s / 372.9 tok/s | 16.66 tok/s | 10.55 s / 574.6 tok/s | 15.21 tok/s |
| 1 | 15.37 s / 394.4 tok/s | 16.86 tok/s | 10.35 s / 585.3 tok/s | 15.30 tok/s |
| 2 | 15.47 s / 391.7 tok/s | 16.54 tok/s | 10.38 s / 584.0 tok/s | 16.40 tok/s |
| 3 | 15.52 s / 390.5 tok/s | 15.82 tok/s | 10.45 s / 579.7 tok/s | 15.47 tok/s |

The README's historical Flash Next figure for a Mac mini M4 Pro with 24 GB is
160.8 tok/s prompt processing. That run used 256 output tokens and a slightly
different prompt count, so it is only a rough comparison.

## Where prompt-processing time goes

With 6,060 tokens and 10 of 512 experts per token, a batch of a few thousand
tokens uses virtually every expert in every layer. Each batch therefore reads
the full 40 GB of prepared experts (40,925 MiB) from the SSD.

| Per prompt | 3072 (two batches: 2,988 + 3,068 tokens) | 6144 (one batch: 6,056 tokens) |
|---|---|---|
| Expert data read | 2 × 40,925 MiB | 1 × 40,925 MiB |
| Read time (`pread_wall`) | about 3.0 s per batch | 1.08–1.23 s |
| Reported read rate | 12,962–13,933 MiB/s | 33,166–37,991 MiB/s* |
| Expert-management time (`pmgmt`) | 2.94–3.10 s per batch, about 6 s total | 1.10–1.22 s |
| Internal SSD reads per request | about 133 GB | about 86 GB |

\* Faster than expected from the SSD; the per-read timing may be inaccurate in
this mode. The server's overall prompt time is the reliable number.

With 3,072-token batches, each batch took about 7.5 s: about 3 s of expert
reads and about 4.5 s of everything else (GPU work for attention, routing and
experts, plus copying). The time spent in expert management was about equal to
the read time, which suggests the prefill pipeline hides little of the reads
behind GPU work. The cause has not been confirmed.

## GPU trace of one long prompt

A Metal System Trace (Xcode Instruments, Shader Timeline enabled) of one
request with `benchmarks/long.json` and `max_tokens: 1`, using the user
configuration (`prefill-max-tokens: 6144`, quiet logging). The server had
already processed one warmup request. Tracing slowed the prompt to 11.77 s
(10.44 s for the untraced warmup), so absolute times are a little high.

| Measurement | Time |
|---|---:|
| Server prompt time | 11.77 s |
| First to last inference kernel on the GPU | 8.94 s |
| GPU busy with inference kernels | **7.07 s** |
| GPU idle between inference kernels | 1.87 s |

GPU busy time by kernel type. When kernels ran at the same time, their shared
time is divided between them.

| Kernel type | Time | Share of busy |
|---|---:|---:|
| Element-wise operations, norms and copies | 2.10 s | 29.6% |
| Dense matrix multiplications (attention projections, shared expert, router, output) | 1.42 s | 20.0% |
| **Routed experts** (`MUL_MAT_ID`, 144 kernels: gate, up and down for 48 layers) | **1.32 s** | **18.7%** |
| Full attention (flash attention, 12 layers) | 0.96 s | 13.6% |
| Linear attention (gated delta net and convolution, 36 layers) | 0.84 s | 11.8% |
| Routing (top-k sort, expert maps) | 0.44 s | 6.2% |

- Routed experts took about 27.6 ms per layer. That is about 28.6 TFLOP of
  expert matrix work in 1.32 s, or about 21.6 TFLOPS on the GPU's matrix
  hardware.
- Each full-attention layer took about 80 ms on its own. Unlike the other
  layers, its cost grows with the square of the prompt length.
- For about 4.7 s of the server's prompt time (11.77 s − 7.07 s), no inference
  kernel was running: 1.87 s of gaps between kernels, plus 2.83 s before the
  first kernel or after the last. The CPU profile below shows that most of it
  is the per-layer embedding lookup.
- Almost all of the 1.87 s of gaps came at one point in each layer: just before
  the routed-expert stage (47 gaps, 1.64 s in total, median 23 ms). At that
  point the CPU waits for the next layer's experts to finish reading, then
  copies them into the expert bank on a single thread
  (`install_prefill_pipeline` in `src/llama-moe-slot.cpp`). The diagnostic run
  reported the same cost as `pmgmt` (1.22 s).
- A synthetic copy test on this Mac with the same sizes (443 experts of
  1.67 MiB per layer) took 11.1 ms per layer with one thread and 6.9 ms with
  eight. So parallel copying alone would save about 0.2 s per prompt. Most of
  the per-layer gap is something else, such as waiting for the read to finish
  or synchronizing with the GPU.
- `LLAMA_GPU_KEEPALIVE` keeps a small spinning GPU job (one 32-thread group,
  about 6.5 ms per command buffer, about 148 per second) running the whole time
  MoEShift is running, including when it is idle. It uses little GPU capacity,
  but it keeps the GPU from resting.

Kernel times come from the trace's per-shader GPU intervals, mapped to kernel
names by program address. The summarized Shader Timeline view missed some
kernels (for example, it showed 5 of the 12 flash-attention kernels), so it was
not used. The encoder-level table for this trace disagreed with the per-shader
intervals (4.6 s busy) and was not used either. An earlier trace without shader
profiling measured 6.96 s busy, which agrees with 7.07 s.

## CPU profile of one long prompt

A Time Profiler recording, with the GPU and Metal Application instruments, of
the same request after a warmup. The server reported 11.67 s of prompt time.
Time Profiler samples running threads only, about once per millisecond. A
thread blocked on I/O or waiting on another thread does not appear.

**Before the GPU starts: 2.2 s of per-layer embedding (PLE) reads.** The
request arrived about 2.2 s before the first inference kernel. Almost all of
that time the main thread was in `llama_ssd_rows::read_rows`
(`src/llama-ssd-rows.cpp`, called from `src/models/qwen4exp.cpp`), mostly in
`pread`.

- The PLE table is `per_layer_token_embd.weight`: IQ4_NL, 160 values per row,
  320 million rows, 27 GB. It makes up almost all of the second GGUF shard and
  is read from the SSD, never loaded into memory.
- Each token looks up 16 rows, picked by hashing its 2-gram and 3-gram context,
  so the rows are spread at random. That is 96,960 lookups for this prompt.
- Each missed lookup is one uncached 16 KiB `pread` (`F_NOCACHE`, read-ahead
  off) into a 32 MiB LRU cache. The reads are issued one at a time, under a
  lock, on the thread that prepares the batch. Nothing else runs until they
  finish.

A synthetic test on this Mac read random 16 KiB pages from the PLE shard with
the same uncached settings:

| Threads | Per read (16 KiB) | Per read (4 KiB) |
|---:|---:|---:|
| 1 | 70.4 µs | 67.7 µs |
| 4 | 20.2 µs | 18.0 µs |
| 8 | 11.9 µs | 9.9 µs |
| 16 | 7.8 µs | 6.1 µs |
| 32 | 5.5 µs | 4.2 µs |

At 70 µs per read, 2.1 s means about 30,000 lookups missed the cache; the rest
hit pages already read. That count is estimated, because the reader's
`bytes_read()` counter is not logged. Issuing the batch's missing pages in
parallel (16–32 at a time) would take roughly 0.15–0.25 s instead of about
2.1 s. That would save about 1.9 s, roughly 18% of a 10.3 s prompt. Reads stay
uncached, so memory use and SSD writes would not change.

### Parallel PLE reads (commit `d38cb01`)

`llama_ssd_rows::read_rows` now gathers the pages for as many of the batch's
rows as fit in the 32 MiB cache, reads the missing pages with up to 16 threads,
then decodes the rows as before. A row too large for the whole cache still
uses the sequential path. The cache size, uncached reads and error handling
are unchanged. `tests/test-ssd-rows.cpp` adds a check that one call can read
more pages than the cache holds, and a check of the sequential path.

Same procedure and baseline as the KV cache and keepalive comparison below
(user configuration, fresh server, one warmup, medians of three requests):

| Build | Prompt time | Prompt throughput | Generation | Peak process memory |
|---|---:|---:|---:|---:|
| Before (sequential reads) | 10.29 s | 589.0 tok/s | 14.71 tok/s | 9,211 MB |
| After (up to 16 concurrent reads) | **8.97 s** | **675.6 tok/s** | 16.18 tok/s | 9,210 MB |
| After, up to 32 concurrent reads (not kept) | 8.93 s | 678.3 tok/s | 14.59 tok/s | 9,210 MB |

Prompt processing was 12.8% faster (1.32 s less), somewhat less than the 1.9 s
estimate. All measured responses from both reader counts matched the baseline
responses exactly. The seven focused native checks passed.

- 32 readers gained only 0.04 s over 16, so the change keeps 16.
- Generation speed differences here are run-to-run variation, not an effect of
  the change. Each new token needs at most 16 page reads, so generation runs
  the same code with 16 or 32 readers. Yet it measured 16.18 tok/s in one run
  and 14.59 tok/s in the other. Generation was not shown to change.

**During prefill: 0.63 s of single-threaded expert copying.** In the GPU
phase, the main thread spent 625 of its 699 running samples in `memmove`
inside `install_prefill_pipeline`, about 13 ms per layer. This matches the
synthetic copy test above (11.1 ms per layer with one thread). The rest of the
per-layer GPU gap is time the thread spends waiting, which sampling cannot
show. Comparing it with the `pmgmt` total suggests about 12 ms per layer of
waiting for the next layer's read to finish.

### Threaded expert copy (commit `d38cb01`)

`install_prefill_pipeline` (`src/llama-moe-slot.cpp`) now copies the layer's
experts from the staging buffer into their slots with 8 threads instead of one
(`kPrefillCopyThreads`), using the same helper-thread pattern as the prefetch
reader.

| Build (`q8_0` KV, otherwise the user configuration) | Prompt time | Prompt throughput | Generation | Peak process memory |
|---|---:|---:|---:|---:|
| Parallel PLE reads | 8.97 s | 675.6 tok/s | 16.18 tok/s | 9,210 MB |
| Parallel PLE reads + threaded copy | **8.60 s** | **704.3 tok/s** | 14.89 tok/s | 9,210 MB |

Prompt processing was 0.37 s (4.1%) faster, more than the 0.2 s estimate from
the synthetic copy test. All three measured responses matched the baseline
exactly. Generation is within the run-to-run variation described above.

### Remaining time before the GPU starts

A second CPU profile, with parallel PLE reads, `f16` KV and no threaded copy
yet, found about 0.9 s between the request and the first inference kernel
(previously about 2.2 s):

- About 0.15 s: request parsing, the chat template, and filling the attention
  mask on one thread (`llama_kv_cache::set_input_kq_mask`, about 40 ms).
- About 0.4 s: PLE reads, now parallel. That is slower than the synthetic
  estimate of 0.15–0.25 s.
- About 0.25 s: reading and copying the first layer's experts, which nothing
  overlaps. The staging buffer (about 850 MiB) is unmapped after each prompt
  so that generation has the memory, and is mapped again at the next prompt.
  Each prompt therefore faults it in again. The first layer's copy took about
  85 ms, against about 13 ms for later layers, probably because the bank pages
  were also being touched for the first time.

## KV cache type and GPU keepalive

Three configurations, each with a fresh server, one warmup and three measured
`benchmarks/long.sh` requests. The baseline is the user configuration
(`prefill-max-tokens: 6144`, quiet logging). Each of the other two changes one
setting. Medians of the measured requests:

| Configuration | Prompt time | Prompt throughput | Generation | Peak process memory |
|---|---:|---:|---:|---:|
| Baseline (`q8_0` KV, keepalive on) | 10.29 s | 589.0 tok/s | 14.71 tok/s | 9,211 MB |
| `cache-type-k`/`cache-type-v`: `f16` | **9.96 s** | **607.3 tok/s** | 14.89 tok/s | 9,362 MB |
| `LLAMA_GPU_KEEPALIVE: 0` | 10.34 s | 586.4 tok/s | **11.52 tok/s** | 9,210 MB |

- **`f16` KV cache:** prompt processing was 3.2% faster (0.33 s). Generation
  speed did not change beyond run-to-run variation. The model keeps KV only for
  its 12 full-attention layers (2 KV heads of 256 dimensions), so `f16` uses
  24 KiB per token, against about 12.8 KiB for `q8_0`. That is about 0.7 GiB
  more at the full 65,536-token context, and about 70 MiB more for this prompt.
  The generated text differed from the `q8_0` run after the opening sentence,
  which is expected when the cache precision changes. Output quality was not
  compared here; see
  [KV cache type and the indexer cache](#kv-cache-type-and-the-indexer-cache-x1)
  for the later comparison, which found the `f16` setting worse at long context
  and moved the configuration back to `q8_0`.
- **Keepalive off:** prompt processing did not change, but generation was 22%
  slower (11.52 tok/s against 14.71 tok/s), and requests took 51–60 s against
  43–46 s. Keep `LLAMA_GPU_KEEPALIVE: 1`.
- System swap was already about 1.16 GB before these runs, against 1.88 MB in
  the earlier runs. Swap use did not grow during the runs. The increase came
  from other activity between the sessions, probably Xcode, Instruments and the
  trace exports.

## Memory

| Measurement | 3072 | 6144 |
|---|---:|---:|
| Metal memory allocated during prefill (`metal_total`) | 10.85 GB | 14.34 GB |
| Batch scratch memory released before generation | 3.65 GB | 7.32 GB (first request) |
| Peak physical process footprint | 8,942 MB | 9,239 MB |

Allocated Metal memory rose by about 3.5 GB, but the physical footprint macOS
charged to the process peaked only 0.3 GB higher, and swap did not change.
Larger batches allocate more temporary memory per batch, so watch memory on a
Mac with less RAM.

## Disk writes

Reads do not meaningfully wear an SSD; writes do.

| Run (4 requests) | Internal SSD read | Internal SSD written |
|---|---:|---:|
| 3072 | 532.9 GB | 54.5 MB |
| 6144 | 345.1 GB | 92.0 MB |

Written bytes are system-wide and include macOS background activity and the
diagnostic logs from these runs. The code review agrees: during inference,
MoEShift keeps the KV cache in RAM, writes logs only to Terminal, and accepts
no profile option that writes files. The main wear risk is macOS swap under
memory pressure. Swap stayed at 1.88 MB throughout.

## Implications for an ANE prefill backend

A proposed experiment would compute routed experts on the Apple Neural Engine
during prefill while the GPU prepares weights. On this Mac and model:

- At 6,144-token batches, routed-expert matrix work took 1.32 s of 7.07 s of
  GPU time, for a prompt that takes about 10.4 s untraced. Making expert work
  free and fully parallel would save at most about 1.3 s (about 13%). A
  realistic ANE offload would save less.
- At 3,072-token batches, about 40% of prompt-processing time was expert
  loading rather than compute.
- The Metal backend already uses the GPU's matrix hardware here
  (`mpp::tensor_ops` with `has tensor = true`).
- The ANE cannot use these quantization formats directly. Expanding experts to 16-bit is
  about 6× larger (roughly 240 GB per batch versus 40 GB stored), adding about
  480 GB of memory traffic per batch and extra buffer memory.
- Each expert sees about 60 tokens per 3,072-token batch, with uneven counts
  that change every batch.
- Reading experts once per prompt (larger batches) and better overlap of reads
  with GPU work are cheaper ways to speed things up, and the first already gave
  +49% prompt throughput here. The trace points to other targets: element-wise
  operations and copies (2.10 s, which kernel fusion could reduce), time with no
  kernel running (about 4.7 s), and full attention (0.96 s).

## Configuration change

`~/.config/moeshift/config.json`, profile `qwen-3.8-flash-next-rco-gsq-iq3-xxs`:

```diff
-        "prefill-max-tokens": 3072,
+        "prefill-max-tokens": 6144,
```

The previous file is saved as `~/.config/moeshift/config.json.bak-3072`. To
revert:

```sh
cp ~/.config/moeshift/config.json.bak-3072 ~/.config/moeshift/config.json
```

Later, the KV cache was switched to `f16`:

```diff
-        "cache-type-k": "q8_0",
-        "cache-type-v": "q8_0",
+        "cache-type-k": "f16",
+        "cache-type-v": "f16",
```

The file from before this change is saved as
`~/.config/moeshift/config.json.bak-q8_0-kv`.

### Source-build configuration

`~/.config/moeshift/config-source.json` is a copy of the main configuration
with prefetch depth 8 and read 8 (see [Prefetch depth](#prefetch-depth-n1)) and no protected
slots (see [Protected decode slots](#protected-decode-slots-p0); the previous file is
`config-source.json.bak-protected24`):

```diff
-        "LLAMA_MOE_PREFETCH_READ": 2,
-        "LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH": 3,
+        "LLAMA_MOE_PREFETCH_READ": 8,
+        "LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH": 8,
-        "LLAMA_MOE_PROTECTED_SLOTS": 24
+        "LLAMA_MOE_PROTECTED_SLOTS": 0
```

It also sets `"cache-ram": 4096` (see [Prompt cache limit](#prompt-cache-limit-m2-uncommitted);
the previous file is `config-source.json.bak-cram8192`).
Since N2 it also drafts from repeated text, with 2-row checks
(`"spec-type": "ngram-simple"`, `"spec-draft-n-max": 1`, `env.LLAMA_MOE_DECODE_ROWS: 2`; see
[n-gram drafting](#n-gram-drafting-on-agent-style-output-n2-measurement); the previous file is
`config-source.json.bak-nospec`).
Since B2 it also sets `"hc-type": "q8_0"`, which changes output (see
[Q8_0 hyper-connection weights](#q8_0-hyper-connection-weights-b2-hc-type-uncommitted); the previous
file is `config-source.json.bak-hcbf16`).

The Homebrew build rejects this file, because its validator still requires
depth 3 when protected slots are on. The main `config.json` stays compatible
with Homebrew, for comparisons. To run the source build with the faster
settings:

```sh
scripts/build-moeshift.sh --run --config ~/.config/moeshift/config-source.json
```

Both files started successfully with their builds on 2026-11-07.

### Could enable: late prefetch (not enabled)

The source build can also read each layer's experts from a later, better guess
(see [Late prefetch](#late-prefetch-n6a)). It is not in either file yet. To try
it, add to the Flash Next profile's `env` in `config-source.json`:

```diff
+        "LLAMA_MOE_LATE_PREFETCH": 8,
+        "LLAMA_MOE_LATE_READ": 8,
```

- Measured +1.8% generation after a short prompt and +0.7% after the long one,
  with identical output and about 4 MB more memory, with 24 protected slots. With the
  protected tier off (the current setting) it gained nothing measurable in one run (see
  [R1](#cache-settings-without-the-protected-tier-r1)), so it is not worth enabling now.
- Costs: one CPU thread spins while the GPU is generating (it sleeps when idle),
  and about a quarter of the extra expert reads go unused.
- The code is new (two output-changing bugs were found and fixed while
  building it), so it stays off until it has had more use. Removing the two
  lines turns it off completely.
- It needs the current source build; the Homebrew build rejects these keys.

Restart MoEShift after changing the configuration.

## Double-buffered prefill reads (commit `6de789b`)

Temporary timers in `install_prefill_pipeline` (removed afterwards) split the
per-layer expert stall, summed over one long prompt with the threaded copy:

| Part | Time per prompt |
|---|---:|
| Waiting for the layer's read to finish | 0.55 s |
| Copying into the bank (8 threads) | 0.27 s |
| Reading, mostly hidden behind GPU work | 6.0 s (about 125 ms per layer) |

A synthetic test read whole expert-layer files at about 7.2 GB/s with 1, 2, 4
or 8 threads, so the SSD's sequential read speed is the limit. Reading 40 GB
takes at least about 5.7 s, against about 7 s of GPU work per prompt.

Two GPU-side expert banks, the change first proposed, would remove only the
0.27 s copy. They would also hold about 1 GB more memory during generation and
change the model loader. Instead, the prefill pipeline
(`src/llama-moe-slot.cpp`) now has two host staging buffers and one reader
thread per batch. The reader reads enabled layers back to back into
alternating buffers and waits only while both buffers are held. Each layer's
install waits for its buffer, copies it into the bank, and frees it. Before,
the next read could start only after the current layer's copy. Both buffers
are still released before generation.

With the same temporary timers, the read wait fell from 0.55 s to 0.13 s per
prompt. Most of what remains is the first layer's read, which nothing
overlaps. The copy time did not change.

## Hyper-connection graph changes (commit `6de789b`)

The model keeps its residual as 4 parallel streams (`dsv4_hc_mult`). For a
6,056-token batch that is a 248 MB f32 tensor. Each element-wise pass over it
reads and writes the whole tensor, taking 1.3–2 ms, close to memory bandwidth.
Element-wise kernels took 2.1 s of the GPU's 7.07 s. About 0.85 s of that came
from `build_hc_mix` (twice per layer), about 0.36 s extra from the 12 PLE
layers, and the rest from the expert-output sum and the hyper-connection merges.
The only gain available is running fewer passes.

Five changes in `src/models/qwen4exp.cpp` keep the same arithmetic in the same
order:

| # | Change | Where |
|---|---|---|
| D1 | Add the four stream views directly instead of copying stream 0 first (`ggml_cont`). Metal's binary kernels accept strided rows, and the three adds still fuse. | `build_hc_mix` |
| D2 | Multiply by the norm weight before the reshape, with the weight viewed as `[n_embd, hc]`, so Metal's fused RMS norm + multiply kernel applies. The reshape had kept the fusion from matching. | `build_hc_mix` |
| D3 | The same for the three grouped norms. | `build_ple` |
| D4 | Transpose the padded convolution history once and take each of the 4 taps as a row range of it, instead of a transposed copy per tap. | `build_ple` |
| D5 | Drop a `ggml_cont` of a tensor that is already contiguous. | `build_ple` |

The estimate was about 0.4 s; the measured saving was 0.32 s (4.0%, table
below). All three long-prompt responses matched the previous build exactly,
and both two-batch responses matched the Homebrew build. The seven focused
native checks passed.

Not done: a faster expert-output sum (3.5 ms per layer, in the MoE code shared
with Qwen3.6).

## Fused hyper-connection mix tail (F1, commit `6de789b`)

Strata's prompt path (`metal/kernels/prompt.metal`, `gr_epi_up`, MIT license)
computes the end of the hyper-connection read in one pass:
`mixed = mean_c xn[c] * sigmoid(G[c])`. MoEShift ran this as separate kernels
for the sigmoid, the multiply, the stream adds and the scale. Each one wrote
and re-read a 248 MB intermediate.

MoEShift already fuses `REPEAT + MUL + ADD` inside the Metal backend, without a
new ggml operation. F1 does the same in `ggml/src/ggml-metal/ggml-metal-ops.cpp`:

- When a `SIGMOID` is followed by `MUL(xn, sigmoid)`, an `ADD` chain over the
  stream views and a `SCALE` with zero bias, Metal runs
  `kernel_hc_mix_tail_f32` instead. That is the node sequence
  `build_hc_mix` produces after D1. The intermediates must have no other users
  and must not be outputs.
- The kernel (`ggml-metal.metal`) reads `G` and `xn` once and writes only
  `mixed`. It uses the separate kernels' float4 expressions in the same order,
  with contraction and reassociation off, so the result is bitwise identical.
- No ggml operation was added, the graph code is unchanged, and other
  backends are unaffected. Without a match, the separate kernels run as before.
- The kernel is new MoEShift code: the idea comes from Strata, not its source.

`tests/test-metal-hc-fusion.cpp` now compares the fused result with the
separate kernels bit for bit, at widths 2560, 40 and 37 and 1 to 1,536 tokens.
Exposing the intermediate as an output prevents the fusion, which gives the
separate kernels' result. All cases matched. At 2560 × 1,536 tokens, a graph of
just these operations took 0.69 ms fused, against 1.39 ms separate.

| Measurement | D1–D5 | + F1 |
|---|---:|---:|
| Long prompt (median of 3) | 7.60 s | **7.44 s** |
| Prompt throughput | 796.9 tok/s | **815.0 tok/s** |
| Generation, `short.json`, 6 interleaved runs | 20.75–20.98 tok/s (median 20.88) | 21.20–21.37 tok/s (median **21.29**) |
| Two-batch prompt, 12,121 tokens | 17.1 s | **16.6 s** |

- **Prompt:** 0.16 s faster (2.1%), about half of the 0.3 s estimated.
- **Generation:** 2.0% faster. The ranges do not overlap.
- **Output:** every response matched the previous build exactly: three long
  prompts, two two-batch prompts and 18 generation runs. The generation output
  was also the same as before D1–D5.
- **Checks:** the seven focused native checks passed.

## Generation profile (G1)

Measured with the F1 build and the user configuration, with diagnostics on
(`LLAMA_MOE_QUIET: 0`). One server: a warmup, two `short.json` requests and
one `long.json` request, each generating 512 tokens. SSD bytes are the
internal disk's read counter around each request.

| Request | Generation | Time per token | SSD read during the request |
|---|---:|---:|---:|
| Short prompt (72 tokens) + 512 tokens | 21.2–21.4 tok/s | 47 ms | 29.6 GB (58 MB per token) |
| Long prompt (6,060 tokens) + 512 tokens | 18.8 tok/s | 53 ms | 74.9 GB (about 40 GB for the prompt) |

**Expert cache.** Each token uses 10 experts in each of 48 layers: 480 uses.
In steady state, 96 to 240 of them missed the per-layer cache (12 recent +
3 prefetch + 24 protected slots), a hit rate of about 50–80%. That is about
150 misses and 250 MB of expert reads per token. Only about 58 MB per token
came from the SSD; macOS's file cache served the rest, because expert reads do
not bypass it by default. Bypassing the file cache (`LLAMA_MOE_F_NOCACHE`)
would therefore slow generation. (Prefill now bypasses it; see
[File cache](#file-cache-prefill-bypasses-it).)

**Prefetch accounting.** The final log reports `lookahead-prefetch: hits=0`.
That counter is only updated outside window mode, and generation always runs in
window mode. The prefetcher writes predictions straight into the layer's cache
window, where a correct prediction later counts as an ordinary cache hit. It
read about 44 experts per token (about 70 MB), and nothing currently measures
how many of them were used.

**GPU timeline.** A Metal System Trace of 10 s of steady generation, with the
Time Profiler (tracing slowed generation from 21 to 15.7 tok/s, so absolute
times are high):

- 19,450 inference command buffers, about 124 per token (about 2.6 per
  layer), each with one encoder. Median encoder time was 131 µs.
- The GPU was busy 5.14 s of 10.34 s (50%). Most idle time was 16,313 gaps of
  0.1–1 ms (4.2 s in total), between consecutive command buffers. There were
  also 562 gaps of 1–5 ms (0.8 s in total), consistent with waiting for missed
  experts.
- The CPU side of each gap is the graph split at every MoE layer: the CPU
  waits for the GPU, reads the router's choice, finds or loads the experts,
  then encodes and submits the next part. The main thread spent most of its
  samples encoding (inflated by the trace's own instrumentation). Three
  threads submitted command buffers, and the expert reader spent its samples
  in `preadv`.

**Conclusion.** Generation is limited mainly by per-layer CPU–GPU round trips,
about 124 per token, rather than by SSD bandwidth. Waiting for missed experts
comes second. Strata's Metal work points the same way: its docs describe "a
fixed hand-shake per layer" as the cost of streaming experts. Its probes test
two replacements for per-layer command-buffer round trips: a GPU kernel that
polls a host-written flag, and `MTLSharedEvent` waits inside one committed
command buffer (`metal/probe/mtl_doorbell.cpp`, `busy_flag_probe.mm`).

## Prefetch use, cache size and handshake cost (G1a, G2, G5 probe)

**Prefetch use (G1a, commit `6de789b`).** Window mode now counts prefetched experts
that the router later used, and those evicted unused
(`lookahead-prefetch window:` in the final diagnostics). For each layer slot,
`src/llama-moe-slot.cpp` records whether a prefetched entry has been used yet.
Over one server run (a warmup, two short requests and one long request):

| Protected slots | Prefetched experts published | Used | Evicted unused |
|---:|---:|---:|---:|
| 24 | 88,488 | 69,528 (78.6%) | 17,850 |
| 48 | 65,179 | 48,222 (74.0%) | 15,376 |
| 96 | 37,751 | 23,743 (62.9%) | 11,340 |

With the shipped 24 slots, about four in five prefetched experts are used, so
the prefetch reads pay off.

**Cache size (G2).** The same requests with 24, 48 and 96 protected slots per
layer, each on a fresh server with diagnostics on:

| Protected slots | Short prompt, generation | Long prompt, generation | SSD read per short request | Peak process memory |
|---:|---:|---:|---:|---:|
| 24 (shipped) | 22.20–22.33 tok/s | 19.28 tok/s | 16.8 GB | 9.2 GB |
| 48 | 22.02–22.35 tok/s | 19.90 tok/s | 22.9 GB | 11 GB |
| 96 | 22.42–22.58 tok/s | 19.91 tok/s | 37.3 GB | 16 GB |

- More protected slots gave at most 1.5% after a short prompt and 3% after the
  long one, for 2–7 GB more memory. Not worth it; the configuration keeps 24.
- SSD reads rose with the cache size. macOS's file cache is already a large
  second-level expert cache, and a bigger in-process cache takes that memory
  away, so more misses reach the SSD.
- The 72-token prompt also got slower (1.13 s → 1.85 s).
- All responses were identical to the 24-slot run.

**Handshake cost (G5 probe).** A standalone Metal program
(`handshake.mm`, not in the repository) timed 2,000 tiny GPU stages, each
followed by a CPU response:

| Handshake | Idle GPU | With a keepalive-style kernel on a second queue |
|---|---:|---:|
| A: one command buffer per stage, `waitUntilCompleted` (as ggml's graph splits) | 128 µs | **54 µs** |
| D: as A, the host polls the status | — | 50 µs |
| C: one command buffer, `MTLSharedEvent` signal/wait, host notified | 141 µs | 55 µs |
| B: one command buffer, `MTLSharedEvent` signal/wait, host spins | 88 µs | **27 µs** |

- The handshake itself costs about 54 µs with the GPU kept awake. Event
  ping-pong inside one command buffer, with a spinning host, halves that.
  About 124 handshakes per token puts the saving at about 3.4 ms per token,
  around 7%.
- Without tracing, a token takes about 47 ms, and the GPU is busy for an
  estimated 31–33 ms of it. That leaves about 130 µs of idle GPU per split,
  so most of each split is CPU work rather than the handshake: reading the
  router output, assigning slots, and encoding and submitting the next part.
- A single pre-encoded command buffer per token would take the encoding off
  the critical path as well. The split timing below measures how much more
  that could save.

**Per-split timing.** Temporary timers in `ggml_backend_sched_compute_splits`
(`ggml/src/ggml-backend.cpp`), removed afterwards, during `short.json`
generation without tracing (22.1–22.4 tok/s, 44.6–45.2 ms per token).
Averages per token:

| Part | Per token | Per layer |
|---|---:|---:|
| Splits | 49 Metal + 49 CPU | — |
| Metal split: encode and commit (asynchronous) | 6.7 ms | about 137 µs |
| CPU split: waiting for the GPU before reading its inputs | 33.5 ms | about 684 µs |
| CPU split: compute (router output, slot assignment, starting reads) | 1.4 ms | about 29 µs |
| Scheduler total | 41.5 ms | — |
| Outside the scheduler (graph setup, inputs, sampling, final sync) | about 3.2 ms | — |

At every layer, the GPU idles while the CPU encodes the next Metal split
(about 137 µs), during the CPU split itself (about 29 µs) and during the
wake-up handshake (about 54 µs). That is about 220 µs per layer, or about
10.8 ms per token, so the GPU is busy about 31 ms of each 45 ms token. This
agrees with the trace.

Encoding is the largest part, and it does not depend on the CPU split's
result: the router's choice reaches the GPU as data, not as graph structure.
A scheduler could encode and commit the next Metal split before running the
CPU split, with the split waiting on an `MTLSharedEvent` that the CPU signals
when it finishes. That would hide the encoding, and with a spinning host it
would also halve the handshake. Per layer, the idle time would fall from about
220 µs to about 60 µs, saving an estimated 8 ms per token: about 45 → 37 ms,
or roughly +20% generation speed. This is an estimate and is not implemented.

**Pre-encoding tried and reverted (G5).** An implementation encoded the next
Metal split while the GPU was still busy:

- A `graph_prepare` hook in the backend interface, called by
  `ggml_backend_sched_compute_splits` before a CPU split.
- Metal command buffers built without being enqueued.
- The second command buffer gated on a shared event, opened by the host or by
  a small command buffer that waits for the expert reads, with the read waits
  looked up at commit time.

Encoding moved off the critical path, as intended: the commit took 0.4 ms per
token instead of 6.7 ms. But the time spent waiting for the GPU did not shrink
(33.5 → 34.0 ms per token), and generation was 1.5% slower:

| Build | Generation, `short.json`, 6 interleaved runs |
|---|---|
| F1 | 21.43–21.72 tok/s (median 21.61) |
| F1 + pre-encoding | 21.12–21.35 tok/s (median 21.28) |

Output was identical everywhere, and the long prompt was unchanged (7.41 s).
The change was reverted.

**The real limit: missing-expert reads.** Temporary timers showed why. In
about 90% of layers, about 3.5 experts miss the cache. Their reads take about
0.36 ms from the CPU split to the shared-event signal. The second Metal
command buffer of each layer waits for that signal. Encoding (about 0.14 ms)
was already hidden inside the read time.

Skipping the reads (`LLAMA_MOE_SKIP_READ=1`, timing only, output invalid,
`ignore_eos` to keep 512 tokens) measured the cost:

| Generation, `short.json` with `ignore_eos` | tok/s | ms per token |
|---|---:|---:|
| Normal | 21.70–21.80 | 45.9–46.1 |
| Expert reads skipped | 29.20–29.29 | 34.1–34.3 |

Missing-expert reads put about 12 ms per token on the critical path. Faster
reads could raise generation by up to about 35%.

How the reads are issued today:

- Each layer with misses creates a new coordinator thread (`start_demand`),
  which creates up to four reader threads (`install_packed_wave`). That is
  about 220 thread creations per token.
- Each expert (about 1.7 MB) is read by one thread in one `preadv`.

Reusing persistent threads, and splitting each expert's three matrices across
threads, are the next things to try.

**Persistent split demand reads (R1, commit `6de789b`).** In `src/llama-moe-slot.cpp`,
`start_demand` now queues one read per matrix of each missing expert, about ten
reads per layer, to a pool of 12 persistent reader threads
(`kDemandReadThreads`, QoS user-interactive). It no longer creates a
coordinator thread and up to four reader threads per layer.

- `read_expert_directv` takes an optional family index and reads that
  matrix's sub-range of the packed expert. A split expert still counts as one
  miss.
- The last read of a layer's batch adds the time, signals the layer's shared
  event and arms the next layer's prefetch, as the coordinator thread did. A
  new batch for the same layer waits until the previous batch has finished.

| Build | Generation, `short.json`, 6 interleaved runs | Generation after the long prompt (median of 3) |
|---|---|---:|
| F1 | 21.36–21.87 tok/s (median 21.47) | 19.08 tok/s |
| F1 + R1 | 22.24–23.03 tok/s (median **22.39**) | **19.90 tok/s** |

- **Generation:** 4.3% faster after a short prompt, and 4.3% after the long
  one. The ranges do not overlap. That is less than the 10–15% estimated.
- **Thread count:** 6, 12 and 24 reader threads gave about the same speed in a
  quick check (about 22.7–23.1 tok/s each).
- **Unchanged:** prompt time (7.41 s), peak memory (9,563 MB) and swap.
- **Output:** every response was identical to the previous build: 18
  generation runs, three long prompts and both two-batch prompts.
- **Checks:** the seven focused native checks passed.

## Prefetch depth (N1)

During generation, the lookahead predicts the next layer's experts from the
current layer's hidden state and reads them in the background. The shipped
profile predicts 3 (`LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH`) and reads 2
(`LLAMA_MOE_PREFETCH_READ`). Positive protected slots used to require depth 3.
That rule now accepts depths 1–16. The protected cache's recent group is the
decode window plus the depth (commit `6de789b`: `src/llama-moe-tensor-loader.h`,
`src/llama-moe-slot.cpp`, `tools/server/server-les-router.cpp`,
`configs/README.md`, `docs/les/protected-decode-cache.md`).

Each setting ran on a fresh server with diagnostics on and the R1 build: a
warmup, two `short.json` requests and one `long.json` request, each generating
512 tokens.

| Depth / read | Short prompt, generation | Long prompt, generation | Predictions used | SSD read per short request | Peak process memory |
|---|---:|---:|---:|---:|---:|
| 3 / 2 (shipped) | 22.23–22.44 tok/s | 19.82 tok/s | 78.2% | 24.0 GB | 9,186 MB |
| 3 / 3 | 22.33–22.47 tok/s | 19.87 tok/s | 77.3% | 24.3 GB | 9,186 MB |
| 5 / 5 | 23.80–23.91 tok/s | 20.90 tok/s | 69.9% | 26.8 GB | 9,346 MB |
| **8 / 8** | **24.94–25.03 tok/s** | **21.76 tok/s** | 58.9% | 33.0 GB | 9,446 MB |
| 12 / 12 | 24.82–24.86 tok/s | 20.82 tok/s | 45.2% | 45.3 GB | 10 GB |
| 16 / 16 | 23.79–23.82 tok/s | 20.20 tok/s | 35.1% | 59.9 GB | 12 GB |

- **Best setting:** depth 8 was fastest. It made generation 12% faster after a
  short prompt and 10% faster after the long one, for 260 MB more peak memory.
- **Deeper than 8:** more predictions go unused, and the extra reads and
  memory cost more than they save.
- **SSD:** depth 8 reads about 38% more from the SSD per request. These are
  reads, not writes, so they do not wear the SSD. Swap did not change.
- **Output:** identical at every depth: the routing and weights are the same,
  and only which experts are already loaded changes.
- **Prompt time:** grew slightly with depth (1.30 → 1.39 s for the short
  prompt, 7.40 → 7.48 s for the long one).

## Combined result

User configuration (`prefill-max-tokens: 6144`, `f16` KV) with all three code
changes, compared with the `q8_0` baseline on the original build. Medians of
three measured requests:

| Setup | Prompt time | Prompt throughput | Generation | Peak process memory |
|---|---:|---:|---:|---:|
| Original build, `q8_0` KV | 10.29 s | 589.0 tok/s | 14.71 tok/s | 9,211 MB |
| Parallel PLE reads + threaded copy, `f16` KV | 8.29 s | 730.9 tok/s | 15.29 tok/s | 9,362 MB |
| + double-buffered prefill reads | 7.92 s | 764.7 tok/s | 16.08 tok/s | 9,587 MB |
| + hyper-connection graph changes (D1–D5) | 7.60 s | 796.9 tok/s | 18.87 tok/s | 9,591 MB |
| + fused hyper-connection mix tail (F1) | **7.44 s** | **815.0 tok/s** | 19.08 tok/s | 9,591 MB |

Prompt processing was 28% faster than the original build (2.85 s less), and
throughput 38% higher. Before any of these changes, at
`prefill-max-tokens: 3072`, the same prompt took 15.47 s. All measured
responses matched the corresponding earlier runs exactly. Peak process memory
rose by 225 MB with double buffering, less than the size of the second buffer
(about 0.9 GiB). The peak may come from a different point in the request. The
cause was not investigated.

A 12,121-token prompt (`benchmarks/long.json` content twice, 64 output tokens)
is split into two batches. That exercises the reader restarting between
batches. It took 17.5 s with the source build before the hyper-connection
changes, 17.1 s after them and 16.6 s with F1, against 21.8 s with the Homebrew build (same
configuration, none of the code changes). Every run gave the same output as the
Homebrew build.

Generation measured 18.7–19.3 tok/s in this run, higher than in any earlier
run (14.5–17.0 tok/s). Most of that difference is run-to-run variation, as the
dedicated benchmark below shows.

**Generation benchmark.** Two builds that differ only in D1–D5, with
`benchmarks/short.json` (512 generated tokens, greedy). They were interleaved
over three rounds; each round used a fresh server, one warmup and two measured
requests.

| Build | Generation, 6 measured runs | Median |
|---|---|---:|
| Before D1–D5 | 20.13–20.57 tok/s | 20.33 tok/s |
| With D1–D5 | 20.80–21.05 tok/s | **20.94 tok/s** |

Generation was 3.0% faster with D1–D5; the two ranges do not overlap. All 18
responses, warmups included, were identical. The larger differences between the
long-prompt runs above came from variation between runs, not from these
changes.

During every benchmark run, the internal SSD wrote 18–139 MB in total, for
the whole system, and swap use did not grow. Swap grew between runs, while
Instruments recorded and exported traces.

## Prompt-read ideas measured and not kept (P1, P2)

Prompts of 1,209–3,996 tokens all took 6.4–6.6 s. Each batch of 1,024 tokens or
more reads all 40.9 GB of experts with the pipelined reader, so these prompts are
limited by the SSD's sequential read speed (about 6.2 s per prompt here). Below
1,024 tokens the batch reads only the experts it uses, but after routing, without
overlap.

| Prompt | Prompt time | Experts used per layer (of 512) | Expert data read |
|---:|---:|---:|---:|
| 313 tokens | 3.94 s | 279 | 22.6 GB, not overlapped |
| 711 tokens | 4.94 s | 331 | 26.5 GB, not overlapped |
| 1,209 tokens | 6.41 s | 361 | 40.9 GB, pipelined |
| 2,499 tokens | 6.45 s | 397 | 40.9 GB, pipelined |
| 3,996 tokens | 6.56 s | 427 | 40.9 GB, pipelined |

**P1: read only predicted experts.** The idea was to predict each layer's experts
during prefill from the previous layer's FFN input, using the next layer's router
(the method decode prefetch uses), and read only those. A temporary probe
compared the union of each token's top-k predictions with the experts the layer
actually used (47 layers per prompt, 10 experts per token):

| Prompt | Top 10 per token: predicted / missed | Top 20: predicted / missed | Top 40: predicted / missed |
|---:|---:|---:|---:|
| 313 tokens | 276 / 43 (85% covered) | 361 / 14 (95%) | 436 / 4 (98.5%) |
| 711 tokens | 323 / 46 (86%) | 403 / 14 (96%) | 461 / 4 (98.9%) |
| 1,209 tokens | 350 / 47 (87%) | 425 / 14 (96%) | 475 / 4 (98.9%) |
| 2,499 tokens | 386 / 43 (89%) | 449 / 13 (97%) | 486 / 3 (99.1%) |
| 3,996 tokens | 414 / 40 (91%) | 466 / 12 (97%) | 494 / 3 (99.3%) |
| 6,056 tokens | 429 / 38 (91%) | 475 / 11 (97.5%) | 498 / 2 (99.5%) |

The predictions are accurate, but 95% coverage needs a predicted set 20–30%
larger than the set used. Estimated from read volume at 6.3 GB/s (not measured),
reading the top-20 set plus the misses would save about 0.9 s at 1.2k tokens,
0.6 s at 2.5k and 0.4 s at 4k. Below 1,000 tokens it would save nothing or be
slower, because those prompts already read only the experts they use. The
6k-token prompt is limited by the GPU. P1 was not implemented. The probe patch
and output are in `/Volumes/Data/moeshift-scratch/p1probe`.

**P2: start the first layer's read earlier.** Before, the reader started at the
first layer's expert step, after the PLE lookups and the first layer's attention.
In the change, `process_ubatch` started it before the batch's inputs were set.
Timestamps confirmed the first layer's read was complete when the GPU reached it
(no wait there), but the times did not change. Interleaved A/B, fresh server per
round, medians of three rounds:

| Prompt | Before | P2 |
|---|---:|---:|
| 1,209 tokens | 6.43 s | 6.37 s |
| 2,499 tokens | 6.47 s | 6.45 s |
| 3,996 tokens | 6.54 s | 6.53 s |
| 6,060 tokens | 7.40 s | 7.42 s |
| 12,121 tokens (two batches) | 16.58 s | 16.57 s |

Outputs were identical, and the 12k-token output matched the Homebrew build.
Peak process memory was the same (9.6 GB). With two staging buffers, the reader
can run only two layers ahead. It filled both during the PLE lookups, then lost
the lead later in the prompt, where reads are about as slow as the GPU's work
per layer (about 130 ms). Up to 4k tokens, the prompt is limited by the SSD from
start to finish. A third buffer might keep the lead on the 6k-token prompt
(estimated 0.1–0.2 s, not measured), but would hold 0.85 GB more during prefill.
P2 was reverted.

## Generation reader ideas measured and not kept (N3, R2)

During generation, the lookahead guesses the next layer's experts and reads them in the
background (depth 8, read 8 in `config-source.json`). A prefetch that is not ready in time
does not block anything: the expert counts as a miss and is read on demand.

**N3: prefetch two layers ahead.** A temporary probe compared, for every expert a layer used
during two 512-token generations, whether it was already loaded, whether the one-layer-ahead
guess picked it, and whether a guess made two layers ahead (layer L's FFN input through
layer L+2's router) would have:

| Guess | Covers of used experts | Covers of misses | Covers of misses the 1-ahead guess missed |
|---|---:|---:|---:|
| 1 layer ahead, 8 guesses (shipped) | 58.4% | 14% | — |
| 2 layers ahead, 3 guesses | 23.5% | 7.7% | 3.5% |
| 2 layers ahead, 8 guesses | 50.1% | 25% | 18% |
| 2 layers ahead, 16 guesses | 69.4% | 48% | 42% |

There were about 2.05 misses per layer at 23.4 tok/s. Only 14% of misses were guessed but not
loaded in time; the other 86% were never guessed, so read timing is not the main limit.
Guessing two layers ahead is less accurate, and catching a useful share of misses would need
about 16 guesses per layer. Raising the one-layer-ahead depth past 8 already made generation
flat or slower (see the N1 section). N3 was not implemented. The probe ran after short
prompts only.

**R2: reader threads and prefetch timing.** Two changes in `src/llama-moe-slot.cpp`, measured
with interleaved runs (fresh server each, `config-source.json`, 512 generated tokens):

- **(a)** The prefetch worker kept its read helpers alive instead of starting and joining up
  to 4 threads per job, about 150–180 per token.
- **(b)** An option armed the next layer's prefetch as soon as the current layer's demand
  reads were queued, instead of after they finished, with at most 2 or 4 prefetch read
  threads while demand reads ran.

| Build | After a short prompt | After the long prompt |
|---|---:|---:|
| Base | 24.01 tok/s | 20.67 tok/s |
| (a) | 24.00 tok/s | 20.60 tok/s |
| (a) + (b), 2 threads | 22.49 tok/s | 19.31 tok/s |
| (a) + (b), 4 threads | 21.93 tok/s | 18.99 tok/s |

Short-prompt values are medians of six runs per build. Long-prompt values for the base build
and (a) are medians of a separate four-round A/B, because a first three-round run measured
the base build low (19.79 tok/s). Long-prompt values for (b) are medians of three runs.
Outputs were identical in every run.

- (a) made no difference. The thread starts run on the background worker, not on the GPU's
  path. A persistent pool for PLE page reads, estimated at 1% or less, was not built.
- (b) was 6–9% slower. Prefetch reads take SSD bandwidth from the demand reads the GPU is
  waiting for, so the shipped behavior, waiting for demand reads first, is kept.

Both were reverted. The patch is in `/Volumes/Data/moeshift-scratch/r2`.

## Phase-switch memory release (uncommitted)

A long prompt sent after a generation peaked at 12 GB (`phys_footprint_peak`), against
9.1–9.7 GB when the previous request only processed a prompt. Every build tested showed it,
including builds from before D1–D5, so it was not caused by the changes above. Earlier peak
measurements in this document (9.2–9.6 GB) came from request orders that missed it.

Temporary counters in the worker (`task_info`, and `mincore` over the expert bank) traced it.
The expert bank holds the 512-slot prefill bank (1.19 GB) and the per-layer decode windows
and protected cache (3.27 GB). At each phase switch,
`ggml_metal_buffer_active_ranges` (`ggml/src/ggml-metal/ggml-metal-device.m`) marked the
whole bank `MADV_FREE_REUSABLE` and reclaimed only the new phase's ranges. The footprint fell
from 8.4 to 5.8 GB at the switch to prefill. During the prompt, the 40 GB of expert reads
filled the file cache, and macOS compressed the reusable decode windows instead of discarding
them. Compressed memory counts in the footprint:

| Point in a 6,056-token prompt after a generation | Footprint | Reusable | Compressed |
|---|---:|---:|---:|
| Switch to prefill | 5.9 GB | 3.1 GB | 0.9 GB |
| Layer 8 | 9.9 GB | 1.4 GB | 2.8 GB |
| Layer 24 | 10.9 GB | 0.9 GB | 3.3 GB |
| Layer 40 | 12.2 GB | 0.1 GB | 4.1 GB |

How much was compressed varied between runs (peaks of 9.7–12.3 GB for the same requests).
Every switch already invalidates the expert mappings in the inactive ranges. The change
therefore replaces those ranges with fresh zero-fill memory (`vm_allocate` with
`VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE`) instead of marking them reusable. Their pages leave the
process at once, and the active ranges are left as they are. With the change, compressed
memory stayed at 0.58 GB through the same prompt, and the peak was 9.2 GB.

Interleaved A/B, `config-source.json`, fresh server per round, three rounds. Each round was a
warmup and two `short.json` requests, then `long.json`, each generating 512 tokens. Medians:

| Build | Short prompt (72 tokens) | Generation after it | Long prompt | Generation after it | Peak footprint |
|---|---:|---:|---:|---:|---:|
| Base | 1.55 s | 24.08 tok/s | 7.56 s | 20.66 tok/s | 12 GB |
| Fresh-memory release | **1.32 s** | **24.45 tok/s** | **7.43 s** | **21.17 tok/s** | **9,445 MB** |

The ranges of the two builds did not overlap on any measure. All responses matched the base
build and earlier runs exactly. Swap stayed at 0.64 GB. The short prompt is 15% faster and
the long prompt 1.8% faster. Generation after them is 1.5% and 2.5% faster. The speedup was
not investigated further; less work for the memory compressor is a likely cause. Not tested:
`config.json` (depth 3) in the A/B, though the traces used it, and prompts of other lengths.

## Draft-model acceptance (N5 feasibility)

Speculative decoding with a small draft model would let one pass of Flash Next, with its expert
reads and per-layer round trips, produce several tokens. Qwen3.5-0.8B and Qwen3.5-2B (dense,
Apache 2.0, unsloth Q8_0 GGUFs of 0.8 and 2.0 GB) use exactly Flash Next's tokenizer: the
248,320 tokens and 247,587 merges in the GGUFs are identical. Draft models built for Flash
Next (DFlash, DFlash2, an MTP head) need the target's hidden states and server support that
llama.cpp does not have, so they were not tested.

Method: six greedy Flash Next answers (`config-source.json`, thinking off, 400–512 tokens
each) from `benchmarks/short.json`, `benchmarks/long.json`, and four new prompts: writing code,
a technical explanation, a short story and editing code. A small tool linked against this
build's `libllama` fed each prompt and answer to the draft model in one pass. At every answer
position it compared the draft's top token with Flash Next's next token. The prompts
tokenized to exactly the token counts Flash Next reported. With greedy checking, a draft is
accepted up to its first mismatch, so these results give the accepted lengths exactly; one
check also yields one token from Flash Next itself.

| Sample | 0.8B agreement | Tokens per check, 3-token draft | 8-token draft | 2B agreement | Tokens per check, 3-token draft | 8-token draft |
|---|---:|---:|---:|---:|---:|---:|
| Story | 54.2% | 1.95 | 2.19 | 62.5% | 2.25 | 2.65 |
| `short.json` | 59.2% | 2.19 | 2.42 | 61.1% | 2.21 | 2.55 |
| Explanation | 70.2% | 2.44 | 3.03 | 76.0% | 2.74 | 3.60 |
| `long.json` (summary) | 73.8% | 2.55 | 3.51 | 77.3% | 2.68 | 3.71 |
| Writing code | 84.5% | 3.20 | 4.94 | 89.0% | 3.51 | 5.97 |
| Editing code | 91.8% | 3.45 | 6.15 | 91.8% | 3.45 | 6.25 |
| **All** | **71.8%** | **2.61** | **3.64** | **75.7%** | **2.77** | **4.04** |

- Even free writing reaches about two tokens per check; prompt lookup (N2) would give almost
  nothing there. Code editing does well with either.
- The 2B model agrees about 4 points more often, at 2.5 times the drafting work and memory, so
  the 0.8B model is the first choice.
- The speedup depends on how much a check of 2–9 tokens costs compared with one token. That
  cannot be measured yet: decode windows take one row (`window_max_rows = 1`), and any wider
  batch switches the engine to the prefill phase. In earlier logs, 4-token batches used about
  19.5 distinct experts per layer, against 10 for one token. Estimated speedup from that, not
  measured: 1.5–1.9 times with 3-token drafts.

The tool, references and results are in `/Volumes/Data/moeshift-scratch/n5`; the models are in
`/Volumes/Data/models/n5`.

## Speculative decoding (phases A and B)

The server already implements speculative decoding, with draft models, n-gram drafting and
recurrent-state rollback for Flash Next's architecture. The worker disabled it. Phases A and B
(uncommitted) make it work under LES; see
[speculative decoding](docs/les/speculative-decoding.md):

- **Phase A:** decode steps of several tokens (`LLAMA_MOE_DECODE_ROWS`, default 1, at most 4
  with the 44-slot windows), including the protected-cache fallback and the multi-row
  resident/missing split.
- **Phase B:** the worker keeps `--spec-*` options, with drafts capped at rows − 1 tokens.
  Multi-row batches stay in decode only while decode is active, so prompt tails keep the prefill
  path. Rollback snapshots are written only by multi-token ubatches.

Validation (`config-source.json` plus `LLAMA_MOE_DECODE_ROWS: 4`, against non-speculative
greedy references from the same build family):

| Request | Reference | Default (1 row), no speculation | 4 rows, `spec-type: ngram-simple` | Accepted drafts |
|---|---:|---:|---:|---:|
| `short.json` | 24.37 tok/s | 24.24 tok/s | 23.99 tok/s | none made |
| Writing code | 22.61 tok/s | — | 22.60 tok/s | none made |
| Editing code | 20.00 tok/s | — | **21.70 tok/s** | 47 of 48 |
| `long.json` | 21.18 tok/s | 21.09 tok/s | 20.55 tok/s | none made |

- Every response, with and without speculation and with 1 or 4 rows, matched the reference
  exactly, including the editing run, which rolled back one rejected draft. The multi-row
  expert kernels run one mat-vec per row and expert, like single-token decode.
- With 1 row, output, speed and peak memory (9,445 MB) are unchanged. With speculation, the peak
  was about 10 GB: snapshot rows widen the recurrent state.
- Writing all four rollback snapshots on every step, as the code did first, made generation
  about 15% slower (20.5 against 24.2 tok/s). Writing two cost 11%. Single tokens now write
  none; a temporary build with rollback disabled showed the same speed as the reference.
- n-gram drafting (`size_n` 12) proposed drafts only on the editing prompt: 16 checks in 400
  tokens. The long-prompt request was 3% slower, possibly from the n-gram search over the 6k
  context at every step. Not investigated.

**Cost of a check.** Per-step GPU and CPU timings (`LES_BLOCK_DIAG`-style counters, temporary)
during the editing request:

| Step | Steps | Time per step |
|---|---:|---:|
| 1 token | 336 | 46.7–47.0 ms |
| 4 tokens (3 drafted) | 17 | 113.7–115.9 ms |

A 4-token check costs about 2.45 single steps. Its rows used about 22 distinct experts per layer
and missed about 14 per layer, roughly four times a single token's misses. Drafted tokens need
experts that are no more likely to be cached than the next tokens of ordinary decode, so batching
saves the fixed per-step work but not the reads. Waiting for the prefetch worker before a check
was not the cause (113.7 ms without the wait). With a 3-token draft, a check pays off only above
about 2.45 accepted tokens per check: the editing prompt (3.9) clears it, while the draft-model
average measured in N5 (2.6) barely does, before the cost of drafting.

## Draft-model speculation (phase D, parked)

`spec-type: draft-simple` with Qwen3.5-0.8B Q8_0 (all layers on the GPU), 3-token drafts and
`LLAMA_MOE_DECODE_ROWS: 4`, on the six N5 prompts. One run each:

| Prompt | Reference | With the draft model | Drafted tokens accepted |
|---|---:|---:|---:|
| `short.json` | 24.37 tok/s | 9.57 tok/s | 279 of 696 (40%) |
| `long.json` | 21.18 tok/s | 10.11 tok/s | 292 of 654 (45%) |
| Explanation | 23.27 tok/s | 16.82 tok/s | 238 of 480 (50%) |
| Story | 25.67 tok/s | 13.85 tok/s | 193 of 612 (32%) |
| Writing code | 22.61 tok/s | 21.48 tok/s | 276 of 369 (75%) |
| Editing code | 20.00 tok/s | 20.16 tok/s | 283 of 345 (82%) |

- **Slower everywhere except editing, which broke even.** Acceptance matched the N5
  prediction: 40% of drafted tokens on `short.json` is 1.15 per check, as expected from 59%
  per-position agreement. But a check cycle took about 230 ms on `short.json` (53.5 s over 232
  checks), of which drafting was 12.6 ms. A check of the target cost about 218 ms here,
  against 115 ms for the mostly accepted n-gram checks. Every drafted row reads its own experts,
  and rejected rows read experts and take cache slots for tokens that are thrown away.
- **Output differed** on five of six prompts, at near-ties ("that kitchen" against "the
  kitchen", "Key Characteristics" against "Advantages"). The text stayed coherent. Hundreds of
  4-row checks evaluate positions with multi-row kernels, whose rounding differs slightly from
  single-token decode; the n-gram runs made only 16 checks. Not separately verified.
- **Prompts were slower** (`long.json` 8.30 s against 7.39 s), because the draft model also
  processes the prompt. Peak memory was 11 GB; the server estimated 7.4 GB for the draft
  context at the full context size and prefill batch, which was not reduced.
- **Bug found and fixed:** with snapshot rollback on the target, the server also removed rejected
  tokens from the draft context with a partial `seq_rm`, but created the draft context without
  snapshots. A recurrent draft model such as Qwen3.5 then aborted on the first rejection
  (`failed to remove sequence 0`). The draft context now gets snapshots too, and keeps the
  drafting checkpoint for its own single-token draft steps.

On this machine, speculative decoding is limited by expert streaming, not by drafting. A drafted
token costs nearly as much SSD reading as a generated one, and a rejected token costs as much for
nothing. This matches the report that MTP drafting was slower on a 32 GB Mac that streams
experts (N4). n-gram drafting (N2) helps only on text that repeats its context, such as code
edits, where almost every draft is accepted.

## GPU time per generated token (trace for P3–P5, B1)

A Metal System Trace with the shader timeline (`MST-shaders` template, `config-source.json`,
current build) of 10 s of steady generation with `short.json`. Shader capture worked only with
Instruments.app open; captures of prompt processing stayed empty, so this covers generation only.
The traced request ran at 21.9 tok/s. The keepalive kernel and window-server work are excluded.
196 tokens were traced (7,056 delta-net calls over 36 layers):

| ms per token | Share | Calls per token | Kernels |
|---:|---:|---:|---|
| 7.40 | 31.7% | 260 | Quantized dense mat-vecs (attention, shared expert, output head) |
| 5.43 | 23.3% | 344 | bf16 mat-vecs: hyper-connection projections, router (read twice), PLE, indexer |
| 4.17 | 17.8% | 192 | Routed experts (`MUL_MAT_ID`) |
| 1.78 | 7.6% | 270 | Copies (`cpy`) |
| 0.98 | 4.2% | 178 | Routing: argsort, softmax, sum_rows |
| 0.83 | 3.5% | 139 | `get_rows` (routing weights, state gathers) |
| 0.63 | 2.7% | 284 | Fused adds and muls (expert sum and other chains) |
| 0.58 | 2.5% | 117 | Delta-net (state update, convolution, l2 norm) |
| 1.36 | 5.8% | 793 | Unary ops, norms, other elementwise, full attention |
| **23.4** | | **2,578** | **Total kernel time** (about 45 ms per traced token) |

The GPU kernels take about half of each token's time; the rest is CPU round trips and waiting
for expert reads. The planned merges are small against this: the expert sum (P3) is already
partly fused by ggml-metal, routing (P4) is about 1 ms, and the delta-net norm (P5) at most about
0.2 ms.

**bf16 weights (B1).** The GGUF stores these dense matrices as bf16: per layer the four
hyper-connection projections `hc_attn_down/up` and `hc_ffn_down/up` ([10240, 320] and [320,
10240], 1.2 GB over 48 layers) and the router `ffn_gate_inp` (120 MiB total, read twice per token
with prefetch), plus PLE, indexer and `ssm_alpha/beta` matrices. A microbenchmark of the Metal
mat-vec (`/Volumes/Data/moeshift-scratch/b1/mv_bench.cpp`) at these shapes:

| Mat-vec | Time | Weight bandwidth |
|---|---:|---:|
| bf16 [10240, 320] (`down`), 96 in a graph | 17.1–17.8 µs | 367–384 GB/s |
| bf16 [320, 10240] (`up`), 96 in a graph | 17.3–17.8 µs | 369–379 GB/s |
| bf16 router [2560, 512], 96 in a graph | 8.5 µs | 306 GB/s |
| Dependent `down` → `up` chain, 96 pairs (one token's projections) | 3.55 ms per pass | 353 GB/s |
| Large 8192 × 8192: bf16 / f16 / q8_0 / q4_K | | 374 / 370 / 320 / 259 GB/s |

The bf16 kernel is the fastest measured, at Flash Next's shapes as well as large ones; a 256 MiB
copy reached only 218 GB/s (read plus write). A better kernel would gain about 10% at most. The
cost is the bytes: about 1.45 GB of bf16 weights per token. Storing them as Q8_0 would roughly
halve that (estimated 4.4 → 2.4 ms per token and about 0.7 GB less memory), but it changes the
output and needs a quality comparison (B2 in `TODO.md`).

## Delta-net state written into the cache (C1a, uncommitted)

A dump of the one-token decode graph found 371 copy nodes per step (about 270 dispatched; empty
ones are skipped). By bytes, one kind dominated: after each of the 36 delta-net layers, the 3.1 MB
recurrent state went from the kernel's output buffer to the recurrent cache with a separate copy
(113 MB per token, read and written). The rest are small: 47 `prefetch_topk` conts of 8 ints, 36
convolution-tail conts, 48 attention-indexer key conts and a few per attention layer.

The change, all in the Metal backend:

- `ggml_graph_optimize` (`ggml-metal-common.cpp`) keeps a one-token `GATED_DELTA_NET` next to the
  `CPY` of its new-state view into the cache, when no node in between touches either. The graph
  order put two unrelated ops between them.
- `ggml_metal_op_gated_delta_net` binds the copy's destination as the kernel's state output and
  skips the copy. Otherwise it binds the old place, after the attention output.
- The kernel's one-token path (`K == 1`) writes the state through that pointer. The snapshot
  path for multi-token rollback is unchanged.

`tests/test-metal-hc-fusion` compares the merged path against a separate copy, bit for bit, for
the cache row and the attention output; with `GGML_METAL_FUSION_DEBUG=1` it logs the merge.

Three-way interleaved A/B, `config-source.json`, fresh server per round, three rounds. Each round
was a warmup and two `short.json` requests, then `long.json`, each generating 512 tokens. Medians:

| Build | After a short prompt | After the long prompt | Long prompt |
|---|---:|---:|---:|
| Memory fix (commit `5335427`) | 22.85 tok/s | 20.51 tok/s | 7.42 s |
| Phases A and B (1 row) | 22.84 tok/s | 20.36 tok/s | 7.40 s |
| **With C1a** | **24.31 tok/s (+6.4%)** | **21.54 tok/s (+5.0%)** | 7.39 s |

The C1a short-prompt runs (24.12–24.82 tok/s) did not overlap the others (at most 23.42). Every
response matched across builds and the earlier references; peak memory stayed 9,446 MB. The gain
is larger than the bytes alone suggest (about 1 ms at the measured copy speed), probably because
the kernel, its barrier and the GPU's wait for it also go away; not measured separately. All three
builds ran about 6% slower here than in earlier runs of the same code, so only the interleaved
comparison is meaningful. The phase A and B code costs nothing at its default.

## Small copies removed from the graph (C1c, uncommitted)

Three groups of `cont` in `src/models/qwen4exp.cpp` copied a strided view only so that the next
op got a contiguous input. The next ops read strided input directly:

- **Prefetch guesses** (47 per token): the next layer's top-k stays a view of the argsort. The
  expert runtime reads it row by row (`nb[1]`), which also covers multi-row decode steps.
- **Convolution tails** (36 delta-net layers and the PLE convolution): the copy into the
  recurrent cache reads the tail view directly.
- **Indexer key pooling** (48, four per attention layer): the adds read the four strided member
  slices, in the same order. Without the copies between them, the three adds are adjacent and
  eligible for Metal's add fusion (not confirmed with the fusion log). These copies grow with
  the context: each one holds every block's key.

Interleaved A/B against C1a, same setup as above (three rounds, fresh server each). Medians:

| Build | After a short prompt | After the long prompt | Long prompt |
|---|---:|---:|---:|
| C1a | 24.78 tok/s | 21.46 tok/s | 7.43 s |
| **C1a and C1c** | **25.20 tok/s (+1.7%)** | **22.35 tok/s (+4.1%)** | 7.48 s |

The ranges did not overlap: after the long prompt 21.42–21.49 against 22.35–22.36 tok/s, after a
short one 24.50–24.90 against 25.01–25.34 tok/s. The long prompt took 0.05 s longer (7.43–7.46
against 7.45–7.48 s); that may be noise and was not resolved.

The larger gain after the long prompt comes partly from the indexer copies, which grow with the
context. A microbenchmark of the pooling chain for one token, 12 layers
(`/Volumes/Data/moeshift-scratch/i1/i1_bench.cpp`):

| Cells in the indexer cache | With the copies | Without (C1c) |
|---:|---:|---:|
| 6,656 (the long prompt) | 1.55 ms | 0.61 ms |
| 16,384 | 3.13 ms | 1.38 ms |
| 32,768 | 6.61 ms | 2.54 ms |
| 65,536 (the configured context) | 13.60 ms | 5.10 ms |

At the long prompt, that is 0.94 ms of the 1.9 ms per token gained; the other copies make up the
rest.

- Every response matched C1a and the earlier references.
- With n-gram speculation (4-row steps; the code edit accepted 47 of 48 drafted tokens), all four
  requests matched the non-speculative references.
- Peak memory was 9,446 MB (C1a 9,445 MB), and about 10 GB with speculation, as before. The
  reserved compute buffer shrank from 6,696 to 6,120 MiB. It is reserved, not resident, so the
  peak does not change.

The indexer still pools, normalizes and rotates every cached block's key on every token. In the
same microbenchmark, pooling only the newest block (as a cache of pooled keys, I1 in `TODO.md`,
would) saves 0.26 ms per token at 6,656 cells, 1.0 ms at 16,384, 2.1 ms at 32,768 and 4.5 ms at
65,536. That cache (I1) was not built: under 1% below about 10k tokens of context, for persistent
state that every rollback, removal, shift and state load must invalidate, and about 96 MB at 64k.
It was built later, with long-context use in view: see [L2](#long-context-generation-l2).

## Delta-net state read from the cache (C1b, uncommitted)

Each delta-net layer gathered its 3.1 MB recurrent state from the cache with `get_rows` before the
kernel (`build_rs`; 113 MB per token over 36 layers). For a one-token step of one sequence, the
Metal kernel now reads the cache row itself, through the same row index, and the `get_rows` is
skipped:

- Both ops use one test (`ggml_metal_op_gdn_state_rows`): the state comes from a single-use
  `get_rows` of one row through reshapes, and no node in between writes the cache or the index.
- The index stays in use, so a rollback (which reads a snapshot row) is handled too. With C1a, the
  kernel then reads and writes the same row. Each thread reads its state values before it writes
  them.
- The kernel's reads of the cache and index are tracked for barriers in place of the skipped copy.

`tests/test-metal-hc-fusion` compares the row read in place and a different row (as after a
rollback) against the separate gather, bit for bit.

Applied to prompt batches as well, it made the long prompt about 1% slower (7.51 against 7.44 s,
also with the build order reversed; tracking only the real reads did not help). The cause was not
found, so the skip is limited to one-token steps; prompt time is then unchanged.

Interleaved A/B against commit `0731c56` (C1a and C1c), same setup as above. Medians:

| Build | After a short prompt | After the long prompt | Long prompt |
|---|---:|---:|---:|
| `0731c56` | 25.09 tok/s | 22.15 tok/s | 7.45 s |
| **With C1b** | **25.79 tok/s (+2.8%)** | **22.41 tok/s (+1.2%)** | 7.46 s |

After a short prompt the ranges did not overlap (25.01–25.30 against 25.62–26.15). After the long
prompt they did (21.68–22.99 against 22.38–23.01), but C1b was ahead in each round. An earlier run
of the version that also covered prompt batches measured +2.0% and +1.1%. Every response matched,
and so did the four n-gram speculation requests; the peak with speculation was about 10 GB, as
before.

A later probe found that C1b could read a clobbered row index when the graph allocator reuses the
index's memory; it is fixed with an address check (see the N6a section).

A note on C1a's grouping: `ggml_graph_optimize` records only a group's last node as written, so for
the delta net plus its state copy, the kernel's own output is not recorded. Its consumers stay
after it only because they read it through a separate view node, whose range blocks the reorder.
A probe with that pattern (`/Volumes/Data/moeshift-scratch/c1b/reorder_test.cpp`) kept the order.

## Routing top-k in one kernel (P4, uncommitted)

After the router's softmax, `build_moe_ffn` ran five kernels per layer: argsort of the 512 expert
probabilities, `get_rows` of the top 10, `sum_rows`, `clamp` and `div`. One Metal kernel
(`kernel_argsort_topk_norm_f32`) now computes all five:

- `ggml_graph_optimize` keeps the five nodes (and the views between them) together, so the
  reorder cannot separate them; `ggml_metal_op_argsort` checks the pattern and encodes the kernel.
- The sort is the argsort kernel's bitonic network in one 512-thread threadgroup, so ties are ordered
  the same. The full index row is still written, for the expert runtime and the expert kernels.
- The 10 weights are summed with one `simd_sum`, as `sum_rows` does with 10 threads, then clamped
  and divided.
- The softmax stays a separate kernel: merging it would mean reproducing its reduction order.

`tests/test-metal-hc-fusion` compares ids and weights against the five kernels, bit for bit, for 1,
4 and 2,048 rows, including rows with many tied probabilities. A counter in the real decode graph
found the merge in all 48 layers (the other 47 argsorts per token are prefetch guesses).

Interleaved A/B against the C1b build, same setup as above. Medians:

| Build | After a short prompt | After the long prompt | Long prompt |
|---|---:|---:|---:|
| C1b | 25.56 tok/s | 22.34 tok/s | 7.46 s |
| **With P4** | **25.79 tok/s (+0.9%)** | **22.50 tok/s (+0.7%)** | 7.48 s |

After the long prompt the ranges did not overlap (22.31–22.35 against 22.41–22.54); after a short
one they did (25.45–25.62 against 25.58–26.02). With the build order reversed, prompt times
matched run for run within 0.05 s, except one cold first run. Every response matched; peak memory
was 9,446 MB for both.

That is about 0.3 ms per token for 192 fewer kernels, about 1.5 µs each: less than estimated. C1a
and C1c gained more per kernel because they also removed copies of megabytes.

For prompt batches, P4 now declines when the allocator gave its output the probabilities' memory
(see the N6a section); one-token steps are unaffected.

## Later expert guesses (N6a probe)

Decode prefetch guesses layer L+1's experts by running L+1's router on layer L's FFN input (top 8),
about one layer ahead. The N3 probe found 86% of the remaining misses were never guessed. N6a asked
whether a later, closer signal guesses better. A temporary probe
(`/Volumes/Data/moeshift-scratch/n6a/probe.patch`, reverted) computed, at the end of layer L,
guesses for L+1 from:

- **V1:** L+1's own FFN mix (`build_hc_mix` with L+1's weights) of the residual after layer L, that
  is L+1's router input without L+1's attention output.
- **V2:** L+1's attention input.
- **V3:** V1 without the gate: L+1's norm and the stream mean only, no projections.
- **V4:** V3 gated with layer L's FFN mix gate.

For every expert each layer used (46 layers per step, 10 experts per token, config as in
`config-source.json`), it recorded whether each guess named it and whether it was a miss:

| Guess for layer L+1 | Covers used, short prompts | Covers used, long prompt | Misses the shipped guess missed, long |
|---|---:|---:|---:|
| Shipped: L+1's router on L's FFN input, top 8 | 57.8% | 55.0% | — |
| V1, top 8 | 72.0% | 69.3% | 43% |
| **V1, top 16** | **93.6%** | **90.8%** | **81%** |
| V2, top 8 / 16 | 54.2% / 75.7% | — | — |
| V3, top 8 | 64.8% | 61.3% | 34% |
| V3, top 16 | 85.9% | 81.9% | 65% |
| V4, top 8 / 16 | 63.3% / 84.1% | 60.1% / 80.3% | 20% / 58% |

Misses per layer were 2.0 after short prompts and 2.36 after the long one. The residual after
layer L predicts L+1's experts much better than L's FFN input does, once L+1's own mix is applied:
V1 with 16 guesses names 91–94% of the experts used. V2 (L+1's attention input) is worse than the
shipped guess.

**Delivery test.** The late guess exists on the GPU in the middle of a command buffer, while the
CPU normally sees results only at each layer's split. A standalone Metal program
(`/Volumes/Data/moeshift-scratch/n6a/delivery.mm`) ran a tiny "guess" kernel followed by a
kernel of about 1 ms in one command buffer, with a CPU thread waiting for the guess:

| How the CPU learns of the guess | Seen after the buffer's GPU start | Lead before the buffer ends |
|---|---:|---:|
| A flag the kernel writes to shared memory (one or two encoders) | 1,100–1,330 µs | none: visible only at completion |
| `MTLSharedEvent` signalled between two encoders, CPU thread spinning | 16–18 µs | 1,060–1,310 µs |

The ids written by the guess kernel were visible when the event was seen. Ending the encoder and
signalling cost the GPU about 1.5 µs.

**Lead in the real engine.** A temporary probe (`/Volumes/Data/moeshift-scratch/n6a/lead-probe.patch`,
reverted) put a one-element marker after each layer's final combine, where a late guess would be
ready. The Metal encoder ended its encoder after the marker, signalled a shared event and opened a
new one; a spinning thread stamped each signal, and the expert runtime compared it with the start
of the next layer's step:

| After | Lead p10 | p50 | p90 | Late |
|---|---:|---:|---:|---:|
| Short prompt | 322 µs | 356 µs | 415 µs | 0 of 12,032 |
| Long prompt | 324 µs | 357 µs | 537 µs | 0 of 12,032 |

Output was unchanged, and splitting the encoder in every layer did not slow generation (single
runs).

**What reads cost now.** With reads skipped (timing only, output invalid, `ignore_eos`, 512
tokens), a token took 27.3 ms instead of 37.9 ms after a short prompt and 29.0 ms instead of
42.7 ms after the long one: reads put 10.6–13.7 ms per token on the critical path, 28–32%. With
`LLAMA_MOE_QUIET=0`, decode missed about 2.2 experts per layer (72–116 per token, 115–196 MiB),
about 0.36 ms of `pread` thread time per expert. The internal SSD read only about 1.2 GB/s during
decode (`iostat`); the page cache served most misses, so the SSD has headroom for extra reads.

**Estimate, not measured** (built afterwards; see the late prefetch section). A layer's demand reads (an expert's matrices split over 12 threads)
mostly take less than the 356 µs lead, so a covered miss read from the late guess would usually
be in place in time. V1 with 16 guesses covers 81–86% of the misses today's guess leaves; if
about half to two thirds of the read time went away, that is 5–7 ms per token (about +15–20%),
minus V1's own cost (about 600 MB more bf16 reads, about 1.6 ms) and contention from extra reads
(R2b). V3 is cheaper but covers 65–74%. Building it means: the guess and an event in the graph
after each layer's combine, a spinning CPU thread that hands the guess to the prefetch worker,
and demand reads that wait for a late read already in flight instead of reading again. Output
cannot change, since prefetch only moves reads earlier.

**The probe exposed a C1b bug (fixed, uncommitted).** One probe variant changed the short-prompt
answer, deterministically. It was not F1: F1 switched off everywhere gave the reference answer,
and so did the probe graph with graph reordering or kernel merging off. Switching the merges off
one at a time pointed at C1b, and only at layer 46.

- C1b's kernel reads the row index (an input copy managed by the graph allocator) after the
  skipped `get_rows`, which the graph lists as the index's last reader. `ggml-alloc` may then
  give the index's memory to a later tensor. In the probe graph, layer 46's convolution output
  got it, so the kernel read a convolution value as the row number and used the wrong state.
- The shipped graph happened not to reuse that memory (counted: 0 of 420,000 checks in short and
  long requests), so outputs so far were right, but any change in tensor sizes or graph shape
  could trigger it.
- The fix compares real addresses: C1b declines if any node from the `get_rows` through the
  kernel was given memory that overlaps the index. With it, the probe graph gives the reference
  answer.

The same class of hazard (a merged kernel reading an input after its last listed reader, or
writing a later node's output early) was checked in the other merges (`ggml_metal_op_overlap`):

- **P4:** the `div` output may be given the probabilities' memory (it was, in a `ggml_gallocr`
  test). One row is safe, since its threadgroup reads everything before writing. With several
  rows another threadgroup may still be reading, so P4 now declines there on overlap.
- **F1:** in the final head mix (`result_norm`), the output was given the start of `xn` in about
  0.5% of mixes. For one token that is safe (each thread reads its own element there before
  writing it), so it is allowed; any other overlap declines.
- **Hyper-connection combine (REPEAT + MUL + ADD):** declines if the output overlaps `x` or `w`;
  no overlaps were counted.
- **Upstream add chains and norm chains (A):** temporary counters over a short and a long
  request (decode and prefill) found the output sharing an input's memory in 136 of 20,000 add
  chains and none of 80,000 norm chains, always at the same start and with the same layout. That
  is safe, since each element is written by the thread that read it (the norm kernel reads the
  whole row, behind barriers, before writing). No other overlap occurred. Both now fall back to
  separate kernels on any other overlap (`ggml_metal_op_clobbers`); `test-backend-ops` passes
  for `ADD` (60/60) and `RMS_NORM_MUL_ADD` (30/30), and the real outputs are unchanged.

`tests/test-metal-hc-fusion` now allocates two cases with `ggml_gallocr`: a computed row index
whose memory goes to the next tensor (it fails without the fix), and the routing chain with the
probabilities' memory reused.

Interleaved A/B of the fix against commit `81d0038`, same setup as above: 26.34 and 26.33 tok/s
after a short prompt, 23.24 and 23.26 tok/s after the long one, 7.45 s for the long prompt in
both. Every response matched, the four n-gram speculation requests matched their references, and
peak memory was 9,446 MB (about 10 GB with speculation). This machine ran about 3% faster than
during the P4 measurements.


## Late prefetch (N6a)

Built from the N6a probes, off by default (`LLAMA_MOE_LATE_PREFETCH`, `LLAMA_MOE_LATE_READ`; see
[late prefetch](docs/les/late-prefetch.md)). At the end of each decode layer the graph routes the
next layer on the new residual (its grouped norm, each stream routed and summed), the Metal
encoder signals a shared event at its next barrier, a spinning thread hands the guess to the
expert runtime, and the reads overlap the next layer's attention. Output is unchanged.

How it got there, all on `short.json` with diagnostics on:

- **First version: slower** (23.6 tok/s against about 26.2). With 16 guesses read, 70% of the
  ~6 late reads per layer went unused, and the steps waited 460 µs on average for reads still in
  flight. The readers also took the worst guess first (their queue is last in, first out).
- **A race changed the output** in one run: a guess that arrived just after the layer's step
  could evict experts the step had given to the GPU. The guess now checks freshness under the
  layer's lock, under which the step counts itself; and no prefetch may evict the latest step's
  experts. Repeated runs were identical afterwards.
- **The guess cost 1.3–2 ms per token of GPU time** with no reads at all (24.9 against 26.1
  tok/s). A microbenchmark of the chain put it at about 28 µs per layer, the router mat-vec
  (2.6 MB of bf16 weights) about 10 µs of it. Routing the 4 streams in one mat-mul, deferring the
  signal to the next barrier, and dropping the lookahead's own guess (the late one replaces it)
  cut the net cost.
- **Waiting on the CPU was the wrong place.** Reads took 185–193 µs after the callback at the
  median, but 577–607 µs at p90 and over 1 ms at p99, against a lead of about 350 µs. A step that
  waited on the CPU for such a read also held back the GPU's resident experts. Attaching the read
  to the step's demand batch, with a negative slot id like a miss, lets the resident experts run
  at once; an earlier attempt without the negative ids let the GPU read unfinished slots, and the
  output changed until it was fixed.

Sweep after those changes (single runs, baseline about 26.0–26.2 tok/s):

| Guesses / read | Generation | Demand misses per token | Late reads used |
|---|---:|---:|---:|
| 8 / 8 | 26.28 tok/s | 78.7 (baseline 135.7) | 73.7% |
| 16 / 6 | 25.73 tok/s | 64.8 | 54.6% |
| 16 / 8 | 25.31 tok/s | 50.0 | 49.7% |

More guesses cut more misses but cost more in unused reads (copies that compete for memory
bandwidth) and in slow reads the GPU then waits for.

Interleaved A/B, `config-source.json`, three rounds, fresh server each. Medians:

| Configuration | After a short prompt | After the long prompt | Long prompt |
|---|---:|---:|---:|
| Off | 26.17 tok/s | 22.85 tok/s | 7.47 s |
| **8 guesses, read 8** | **26.63 tok/s (+1.8%)** | **23.01 tok/s (+0.7%)** | 7.49 s |
| 8 guesses, read 6 | 26.67 tok/s (+1.9%) | 22.97 tok/s (+0.5%) | 7.48 s |

After a short prompt the ranges did not overlap (26.07–26.42 against 26.54–26.72); after the long
one they nearly did. Every response matched, the n-gram speculation suite (4-row steps, late
prefetch on) matched its references, and peak memory was 9,450 MB against 9,446 MB. The gain is far
below the 15–20% estimated from the read cost: the guess's GPU time and the slow tail of the
extra reads take most of what the 42% fewer demand misses save.

**Layer 0.** Layer 0 misses the most (7.1–7.4 experts per token against about 2 elsewhere) and gets
no guess, since no layer comes before it. A temporary probe found nothing to guess from:

- Its input before attention is the token embedding alone (PLE starts at layer 1). Routing that
  through layer 0's FFN mix or norm named 2–6% of the experts it used, about chance (16 of 512 is
  3.1%), so its routing depends on its attention over the context.
- The previous step's router ranks 11–26 held 5–6% of layer 0's misses (19–20% for other layers).
- An LRU replay of its expert ids (407–418 distinct experts in 511 steps) gave 8.5–8.8 misses per
  step with 36 slots and 4.3–5.6 with 128: a 128-slot window for layer 0 alone (about +156 MB)
  would save about 3 misses per token out of about 100.

The CPU has 2.5–3 ms between a token's last layer step and the next token's layer-0 step, but no
predictor for it.

## Where a generated token's time goes now (G6 measurement)

Temporary timers in `ggml_backend_sched_compute_splits` and a completion handler on every Metal
command buffer (removed afterwards), `short.json` with `ignore_eos`, 512 tokens, averages per
decode graph:

| Per decode graph | Normal | Expert reads skipped (timing only, output invalid) |
|---|---:|---:|
| Total | 37.12 ms | 27.05 ms |
| GPU busy (command buffers, start to end) | 23.72 ms | 21.96 ms |
| GPU idle (total minus busy) | 13.4 ms | 5.1 ms |
| Metal split: encode and commit | 5.53 ms | 3.49 ms |
| CPU split: waiting for the GPU | 28.88 ms | 21.82 ms |
| CPU split: expert step | 1.19 ms | 0.48 ms |
| Splits | 98 | 98 |
| Generation | 26.2 tok/s | 36.5 tok/s |

With late prefetch on, the normal run was much the same (36.25 ms per graph, 24.53 ms GPU busy,
4.96 ms encoding, 1.09 ms expert step).

- The GPU computes for about 22 ms of each 38 ms token.
- Expert reads put about 10 ms on the critical path, as measured before.
- Handshakes, encoding and the expert step cost about 5.1 ms when no reads hide them, about
  104 µs for each of the 49 CPU splits. That is what one command buffer per token (G6) could
  recover: at most about 5 ms (+15%). In normal runs the layers with misses already hide their
  encoding behind the reads (which is why G5's pre-encoding did not help), so a realistic gain is
  perhaps 2–4 ms per token, about +5–10%; not measured.

## One command buffer per token (G6 prototype, reverted)

A prototype (`LLAMA_MOE_ONE_BUFFER=1`; patch in `/Volumes/Data/moeshift-scratch/g6/g6-prototype.patch`)
kept each layer's expert step inside the Metal graph instead of ending a graph split there:

- Metal claimed the LES custom op of one-token decode graphs, so the scheduler kept the whole
  token in one Metal graph (2 splits instead of 98).
- For step v, the encoder ended its encoder, signalled a shared event with v and waited for a
  second event to reach 2v − 1 (slot ids written); before each layer's missing-expert kernels it
  waited for 2v (reads done). A spinning host thread ran the custom op on host-visible memory, and
  the demand batch's last read signalled 2v.
- With one command buffer per graph split (`n_cb` 1) the GPU waited for the whole token to be
  encoded, about 5 ms, so `n_cb` was raised to 2 or 8.

Output was identical everywhere. Interleaved A/B, three rounds, as above:

| Configuration | After a short prompt | After the long prompt | Long prompt |
|---|---:|---:|---:|
| Splits (baseline) | 26.19 tok/s | 23.14 tok/s | 7.46 s |
| One graph, `n_cb` 2 | 26.39 tok/s (+0.8%) | 23.02 tok/s (−0.5%) | 7.46 s |
| One graph, `n_cb` 8 | 26.51 tok/s (+1.2%) | 23.01 tok/s (−0.6%) | 7.47 s |

With expert reads skipped it saved only about 0.5 ms per token (27.25 → 26.75 ms) of the 5.1 ms
of idle GPU measured above. The host side was fast (the step took about 19 µs on the host
thread), so most of the roughly 100 µs per layer is on the GPU side: resuming after an event wait
and the barriers of the extra encoder splits, which one command buffer does not remove. Peak
memory rose by about 20 MB. Reverted.

## Zero-copy experts from the page cache (feasibility)

[ANE Fieldfare](https://github.com/bently0602/ane-fieldfare) (Apache-2.0) serves routed experts
without an expert cache of its own: each layer file is mapped read-only, every expert is a no-copy
`MTLBuffer` over its page-aligned slice, `mincore` checks residency, misses are read in with
parallel `pread`s, and each command buffer binds only the experts it uses (`useResources`).
Replacing its own 48-slot cache with the OS file cache made its decode about 15% faster. In
MoEShift about 75% of decode misses are already served by the page cache, so most of the 10–14 ms
per token of reads is copying into the decode windows.

Two standalone Metal tests (`/Volumes/Data/moeshift-scratch/zc/`), with a 1 GB file written
uncached to the external drive for the cold cases:

| Test | Result |
|---|---|
| One no-copy buffer over a whole 1 GB file, kernel reads 4 KB | **Metal paged in the whole file** (65,536 of 65,536 pages, 3.2 s from the USB drive); on the half-cached layer 0 file, 153 ms. Whole-file buffers do not work |
| Unaligned slices through a mapping (also layer 0's experts 5 and 300) | Correct sums |
| A page-cached slice, through the mapping and in an ordinary buffer | 0.18 ms each; copying the same 1.87 MB out of the page cache took 0.23 ms |
| 512 no-copy buffers, one per expert over the pages that cover it (neighbours share an edge page) | Created in 0.95 ms, nothing paged in |
| Binding one cold expert's buffer | Paged in exactly its 115 pages; its neighbour, sharing a page, read correctly |

The prepared expert files are not page aligned (layer 0's stride is 1,868,800 bytes, a multiple
of 1 KB), but per-expert buffers over the covering pages make repacking unnecessary. Cold reads in
the tests came from the USB drive and are not representative of the internal SSD.

**Prototype (reverted, `LLAMA_MOE_ZC_LAYER=all|L|a-b`; patch in
`/Volumes/Data/moeshift-scratch/zc/zc-prototype.patch`).** The expert runtime maps each chosen
layer's expert file and creates a no-copy buffer per expert over its covering pages, with a table
of GPU addresses (`bridge-cpu.mm`). `kernel_mul_mv_id` takes an expert's weights from the table
when the encoder finds the weight tensor registered, and the encoder binds the step's experts with
`useResources`. A decode step's slot ids become expert ids; an expert counts as resident when 8
sampled pages are (`mincore`), and a miss is read into the page cache by the demand readers, with
the GPU's missing phase waiting for it as before. The decode windows of those layers are not used.

| `short.json`, single runs | Generation | Expert uses that needed a read | Metal encode and commit per token | GPU busy per token | Peak memory |
|---|---:|---:|---:|---:|---:|
| Windows (baseline) | 26.0–26.4 tok/s | (about 100+ demand misses per token) | 5.45 ms | 23.72 ms | 8.6 GB |
| Zero-copy, all layers | 17.6–17.7 tok/s | 5.7% (about 27 per token) | 11.28 ms | 22.39 ms | 5.6 GB |
| Zero-copy, residency check off (no reads) | 21.8 tok/s | — | 3.44 ms | 22.11 ms | — |

Output was identical in every run, and layer 24 alone (its file fully cached) needed no reads in
511 steps. The page cache served far more experts than the windows do, and the process used 3 GB
less. But generation was 32% slower:

- With reads in flight, encoding the next split took twice as long: binding (`useResources`) an
  expert whose pages are still being read appears to wait for the read, which puts the reads on
  the encoding thread.
- Even without reads, a token took about 8 ms longer, with neither encoding nor GPU busy time
  higher. That points to Metal making the bound file-backed buffers resident for every command
  buffer (about 60 buffers per layer bound in 6 encoders), between commit and GPU start; not
  measured directly.

**Binding cost, measured alone** (`zc_test4.mm`): 48 command buffers per "token", each a small
kernel reading 10 experts through the table. Binding 10 warm file-backed buffers with
`useResources` cost nothing measurable (5.99 ms per token, the same as a residency set holding
them). A residency set of 2,048 experts wired 2.7 GB, so it would pin its experts like an
app-managed cache. The test reused the same few experts every token.

**Hybrid** (reverted; patch in `/Volumes/Data/moeshift-scratch/zc/zc-hybrid.patch`): only experts in the
page cache (first, middle and last page resident) are read in place, through a flagged slot id
(`GGML_MOE_ZC_FLAG`); every other expert goes through the decode window as before, so no
file-backed buffer is ever bound while it is being read, and prefetch skips experts already in
the page cache. Single runs, `short.json`:

| Configuration | Generation | Peak memory | Expert uses not in the page cache | Encode and commit per token | Residency checks per token |
|---|---:|---:|---:|---:|---:|
| Windows, 24 protected slots (baseline) | 26.22 tok/s | 8.6 GB | — | 5.6 ms | — |
| Hybrid, 24 protected slots | 23.49–24.30 tok/s | 8.65 GB | 3.9% | 6.62 ms | 1.80 ms |
| Windows, protected slots 0 | 26.44 tok/s | 6.7 GB | — | 5.7 ms | — |
| Hybrid, protected slots 0 | 23.66 tok/s | 6.7 GB | 2.9% | 8.06 ms | — |

Output was identical everywhere. The hybrid stayed about 10% slower: the residency checks
(`mincore`) cost about 1.8 ms per token and encoding 1–2.4 ms more, probably because each token
binds about 480 different file-backed buffers, which Metal maps at commit (the standalone test
reused the same ones). What zero-copy removes, copying page-cache hits into the windows, was
mostly hidden behind the GPU's work on resident experts already; the reads on the critical path
are the SSD misses, and those did not change. Not kept.

**Side result: the protected cache** (see the P0 section below).

## Protected decode slots (P0)

The zero-copy tests suggested that macOS's file cache already holds what the 24 protected decode
slots per layer do. Interleaved A/B on the current build (commit after `d81e49f`), same setup as
above: three rounds, fresh server each, warmup + 2 short requests + the long one, 512 tokens
each; `config-source.json` with only `LLAMA_MOE_PROTECTED_SLOTS` changed. Medians; peak and final
footprint per server; internal-SSD transfer per server run (`iostat`, reads and writes, almost
all reads):

| Protected slots | After a short prompt | After the long prompt | Long prompt | Peak | Footprint after generating | SSD transfer per run |
|---:|---:|---:|---:|---:|---:|---:|
| 24 (configured) | 26.40 tok/s | 23.21 tok/s | 7.46 s | 9,446 MB | 9,320 MB | 189 GB |
| 12 | 26.58 tok/s (+0.7%) | 23.38 tok/s (+0.7%) | 7.46 s | 9,181 MB | 8,360 MB | 173 GB |
| **0** | **26.68 tok/s (+1.1%)** | **23.41 tok/s (+0.9%)** | 7.46 s | 9,180 MB | **7,400 MB** | **159 GB (−16%)** |

- After the long prompt the ranges did not overlap (23.16–23.21 against 23.39–23.42 tok/s);
  after a short one they touched (25.99–26.56 against 26.56–26.73).
- Every response matched.
- With no protected slots the decode windows take 1.9 GB less, which the page cache uses. The
  SSD read 16% less over the same requests: an expert in a protected slot is usually also in the
  page cache, so the slots held second copies. The peak hardly changes, because prefill's expert
  bank sets it.
- The protected cache was tuned before the page cache was measured as an expert cache (G2 found
  the same effect for larger caches).

Applied: `config-source.json` and the Flash Next profile in `configs/moeshift.json` now set
`LLAMA_MOE_PROTECTED_SLOTS` to `0` (the Qwen3.6 profile keeps 24; not measured). The main
`config.json` keeps 24. Without protected slots Flash Next's windows have 20 slots
(`config-source.json`, depth 8), which allows decode steps of at most 2 rows for speculation.

## Cache settings without the protected tier (R1)

With protected slots at 0 the page cache is the second-level expert cache, so the decode window,
the prefetch depth and late prefetch were checked again. Single runs (fresh server, warmup, then
`short.json` and `long.json`, 512 tokens each), `config-source.json` with one change each:

| Change | After a short prompt | After the long prompt | Long prompt | Footprint after generating | SSD transfer per run |
|---|---:|---:|---:|---:|---:|
| None (window 12, depth 8, read 8) | 26.28–26.35 tok/s | 22.81–23.05 tok/s | 7.49–7.50 s | 7,400 MB | 143–148 GB |
| Window 10 | 26.35 | 23.10 | 7.47 s | 7,240 MB | 142 GB |
| Window 16 | 26.47 | 23.04 | 7.47 s | 7,720 MB | 148 GB |
| Depth 3, read 3 | 22.95 | 20.36 | 7.47 s | 7,000 MB | 124 GB |
| Depth 6, read 6 | 25.97 | 22.89 | 7.47 s | 7,240 MB | 134 GB |
| Depth 10, read 10 | 25.93 | 22.17 | 7.48 s | 7,560 MB | 166 GB |
| Depth 12, read 12 | 25.53 | 21.82 | 8.08 s | 7,720 MB | 180 GB |
| Late prefetch 8 / 8 | 26.50 | 23.05 | 7.47 s | 7,400 MB | 145 GB |

Every response matched. The current settings are the best measured: the window (10–16 slots,
of which at least 10 are needed per token) makes no difference beyond about 1%; prefetch depth 8
is clearly better than 3 (13% faster) and somewhat better than 6, 10 or 12. Late prefetch, which
gained 1.8% with the protected tier on, now gains nothing measurable (+0.6% and 0 in one run):
with the file cache holding more experts, most of its guesses are no longer misses. No setting
was changed.

## Prefill peak memory (R2, uncommitted)

### What the footprint does not count

Metal shared buffers are host memory wrapped without a copy. A test wrapped a 1 GiB
`vm_allocate` buffer the way ggml-metal does and had the GPU write every page. The process
footprint rose by 89 MiB, and `resident_size`, `internal` and `vmmap` showed nothing. System-wide
internal + wired + compressed memory rose by 1,131 MiB. The pages appeared in the process
(+1,024 MiB) only once the CPU read them.

So pages that only the GPU writes, which is most of the graph's compute buffer, are missing from
`phys_footprint`, `footprint` and `vmmap`. Pages the CPU writes are counted: the graph inputs (KQ
mask, indexer bias) and the buffers it clears (the KV cache). The peak footprints elsewhere in this
document undercount prefill by the compute buffer.

The figures below are system-wide instead: the change in internal + wired + compressed pages from
just before the worker started, taking the highest 1-second rolling median
(`/Volumes/Data/moeshift-scratch/r2m/tools/memwatch.c`). The figure includes other processes;
repeated runs differed by up to 0.3 GB.

### Where prefill memory goes

`config-source.json`, a 6,144-token batch, at about 42k tokens of context:

| Part | Size |
|---|---:|
| Non-expert weights (Metal) and token embedding (CPU) | 3,665 + 260 MiB |
| Prefill staging: two host buffers, each the largest layer pack (the pipeline reads whole layers) | 2 × 1,137 MiB |
| Prefill bank: 512 slots of the largest layer, into which the used experts are copied from staging | 1,138 MiB |
| KV cache (grows; 49,152 cells here) and indexer keys | 1,152 + 144 MiB |
| Recurrent state, one context checkpoint | 113 + 113 MiB |
| Compute buffer, allocated per graph (allocator high-water) | 2,653 MiB at short context, 5,806 MiB at 42k |

The compute buffer grows with context through the indexer's `[n_kv, rows]` tensors: scores,
expanded scores, the f32 copy of the mask and, largest, the top-k scratch at 8 bytes per score
(3 GiB for 6,144 rows at 64k cells). The buffer reserved at load is 6,120 MiB, mostly logits for
all 6,144 rows (5,820 MiB), although a prompt needs only the last row's logits. LES already
releases compute buffers before each decode phase, so later prefills allocate only what their
graph needs. The first prompt after a reserve, however, ran in the reserved buffer, and Metal made
all of it resident on first use: the 72-token warm-up prompt held 14.4–14.6 GB.

### Changes

- **Release the reserved compute buffers** (`sched_reserve` in `src/llama-context.cpp`): for
  LES models the scheduler is recreated after reserving, as before each decode phase. The
  first prompt after load then allocates like every later one.
- **Sparse attention in chunks of query rows** (`build_qsa_top_k` and `build_attn_qsa` in
  `src/models/qwen4exp.cpp`): keys, K/V stores and the indexer queries are computed once; scores,
  top-k, the attention mask and flash attention run per chunk, and the outputs are concatenated.
  Chunks are used only when the whole batch's top-k scratch would exceed 512 MiB (above about
  10.9k cells for 6,144 rows), sized to that budget, equal and rounded up to 32 rows. Every op
  involved works row by row, so the output is unchanged. In a test build with 1,024-row chunks, the
  compute high-water during the 42k prompt stayed at 2,653–3,484 MiB, against 2,653–5,806 MiB.

Interleaved A/B, two rounds, fresh worker each. Each round sent `short.json`, a 3,996-token
prompt, `long.json` and a 42,389-token prompt (the long prompt's text seven times, 16 tokens
generated):

| Request | R1 system peak | R2 system peak | R1 prompt | R2 prompt |
|---|---:|---:|---:|---:|
| First prompt (72 tokens) | 14.39 / 14.59 GB | **8.17 / 8.15 GB** | 2.10 / 2.09 s | **1.69 / 1.68 s** |
| 3,996 tokens | 11.16 / 11.45 GB | 11.71 / 11.54 GB | 6.26 / 6.30 s | 6.25 / 6.29 s |
| 6,060 tokens (`long.json`) | 12.26 / 12.61 GB | 12.77 / 12.72 GB | 7.45 / 7.57 s | 7.46 / 7.45 s |
| 42,389 tokens | 17.26 / 17.58 GB | **16.18 / 16.46 GB** | 88.2 / 88.0 s | 88.1 / 88.2 s |

Every response matched the R1 build: 512 tokens after the short and long prompts, and the 16 after
the 42k prompt (generated with chunked attention). Generation was 26.03–26.36 tok/s (R1) and
26.66–26.67 (R2) after the short prompt, and 23.39–23.48 and 23.42–23.66 after the long one; the
difference is within noise. At 4k and 6k tokens both builds run the same graph, so the differences
there are run-to-run noise. The chunked graph raises the process footprint at 42k (13.1 → 14.4 GB),
because the CPU-written inputs land on different compute pages from batch to batch; the
system-wide figure is lower.

### Tried and not kept

- **Smaller prefill batches** (`prefill-max-tokens` 4096 or 3072): the long prompt took 14.32 and
  12.91 s instead of 7.47 s, and the 4k prompt 6.39 and 10.75 s instead of 6.32 s, since every
  batch reads all experts again. (The footprint peaks measured, 9,106 and 9,045 MB against
  9,273 MB, also miss the smaller compute buffer.)
- **A 256-slot prefill bank**: the worker aborts; a 6,144-token batch uses more than 256 of a
  layer's 512 experts.
- **No prompt cache** (`LLAMA_ARG_CACHE_RAM=0`): 0.25 GB less on the long prompt. Kept: it lets a
  client switch between conversations without reprocessing them. Its limit is now 4,096 MiB (see
  [Prompt cache limit](#prompt-cache-limit-m2-uncommitted)).

### Not tested

- Contexts above 42k tokens, where chunking should save more (the unchunked top-k scratch reaches
  3 GiB at 64k cells).
- Prefill straight from the staging buffers, which would remove the 1,138 MiB bank copy (see
  TODO).
- The Homebrew build.

## Prompt cache limit (M2, uncommitted)

The worker has one slot. When a request does not continue the conversation in it, the server saves
the slot's state (KV cache, recurrent state and its context checkpoint) to the prompt cache and
restores the closest saved state for the new request, if there is one. When the cache exceeds
`--cache-ram` (8,192 MiB by default), entries are dropped (upstream: oldest first; now cheap ones
first, see [Eviction order](#eviction-order)), but the newest is always kept. A restored entry
leaves the cache. Agent clients rely on this: title requests, subagents and
summaries between turns each replace the main conversation in the slot.

### Entry size

From the server's log (`config-source.json`, `f16` KV): 240.5 MiB for 583 tokens, 398.7 MiB for
6,571 and 1,346 MiB for 42,389. That is about 225 MiB fixed (113 MiB recurrent state and a
113 MiB checkpoint) plus 27 KiB per token, about 1.9 GiB at 64k. 8 GiB holds about 34 short
requests, so ordinary side requests fill it.

### Agent-style session

Twenty requests on a fresh worker: a 42,389-token conversation continued three times with its own
replies (M1–M4), with side requests between turns: one title request before M2; a 3,996-token
"subagent" prompt and two title requests before M3; 12 title requests before M4. Prompt time of the
main turns:

| `cache-ram` | M2 | M3 | M4 | Cache before M4 |
|---|---:|---:|---:|---|
| 8192 | 1.74 s | 2.59 s | 3.33 s | 16 entries, 5,023 MiB |
| **4096** | **1.74 s** | **2.56 s** | **3.24 s** | 12 entries, 4,067 MiB (the four oldest side entries dropped) |
| 2048 | 1.72 s | 2.54 s | 89.64 s | 8 entries, 1,910 MiB (main conversation dropped) |
| 1024 | 1.64 s | 89.02 s | 89.62 s | 4 entries, 955 MiB |
| 0 | 88.51 s | 88.49 s | 88.68 s | — |

All 20 responses were identical at every limit, so restoring from the cache gives the same output
as reprocessing. Saving or restoring an entry took 14–170 ms. With oldest-first eviction (these
runs), the main conversation is lost when the side requests saved between two of its turns exceed
the limit minus its own size.

### Freed entries stayed in the process

With 4096, the process footprint at the end was 14.6 GB, against 8.5 GB without the cache,
although the cache held 2.9 GB. `vmmap` showed why: macOS malloc keeps freed large blocks (each
restored, dropped or replaced entry) as 128 MiB chunks for reuse, still dirty. They took 4.4 GB:
2.7 GB resident and 1.8 GB already compressed. Under memory pressure they would be swapped like
live data. `malloc_zone_pressure_relief` after each cache update released none of them.

Context checkpoints and prompt-cache states now use `common_state_buffer`
(`common/common.h`), a vector whose pages come from `mmap` and return with `munmap`. Same
session with 4096:

| | malloc | Page-backed |
|---|---:|---:|
| Footprint at the end | 14.6 GB | **10.8 GB** |
| Freed large blocks kept | 4.4 GB | 0.7 GB (as with the 6,571-token session without a cache) |
| System-wide, last four requests | 15.0–15.7 GB | 11.4–12.6 GB |
| Restoring the 42k entry | 170 ms | 79 ms |

Responses were identical, and prompt times were the same.

### 64k conversation

The same session with a 63,286-token main conversation (63,429 by M4), page-backed build, 4096:

| | Oldest first | Cheap first |
|---|---:|---:|
| M1, read from scratch | 159.9 s | 159.0 s |
| M2, M3 | 1.74 s, 2.45 s | 1.73 s, 2.43 s |
| M4, after 12 side requests | **160.9 s** (dropped at the 11th) | **2.93 s** |
| Entries dropped | five short, the subagent, then the main conversation | five short, the subagent |

The 64k entry took 1,900 MiB, leaving room for about nine short entries. Peak memory during the
63k prompt was 16.8 and 17.9 GB system-wide in the two runs (42k: 16.2–16.5 GB), and 17.9 GB
when the main conversation was reprocessed with 2.9 GB in the cache. A 64k prompt with a full
4 GiB cache should stay near 21 GB (not measured). The footprint at the end was 11.5 GB. All
20 responses matched between the two runs, including M4, which one run restored and the other
reprocessed. Swap did not grow (806 → 817 MB, from other activity).

### Eviction order

Upstream drops the oldest entry first. Each side request is saved after the conversation it
interrupted, so a burst of them pushes that conversation out, although it is by far the most
expensive to reprocess (160 s against about 4 s for a title request).

`server_prompt_cache::evict_one` (`tools/server/server-task.cpp`) now drops the oldest entry with
fewer than a quarter of the largest entry's tokens, and only then the oldest of the rest. Short
side requests and small subagent prompts go first; long conversations replace each other oldest
first, so an unused one leaves as soon as another long one needs the room. In a simulation with
the measured entry sizes, a 42k or 64k conversation survived 40 consecutive short side requests at
4096. A greedy-dual-size order (prompt tokens per MiB, aged by evictions) was tried first and
dropped: it ranked a stale 64k conversation above a recently used 42k one, and aged so slowly that
an unused long conversation stayed indefinitely. The 42k session with cheap-first eviction
behaved as before (M2–M4 1.63, 2.31 and 2.82 s; identical responses).

### Setting

`cache-ram` is now a profile option. `config-source.json` and the Flash Next profile in
`configs/moeshift.json` use 4096: with cheap-first eviction it kept the main conversation through
all side requests above at 42k and 64k, while halving the worst case to 4 GiB. Several long conversations in turn need more; raise it, or
lower it to trade reprocessing for memory. `config.json` stays without it, because the Homebrew
build rejects the option.

### Not tested

- Real agent clients; the session above imitates their side requests.
- `q8_0` KV (smaller entries).
- Several long conversations in turn with cheap-first eviction (only simulated).
- The Homebrew build, which keeps the 8 GiB default, oldest-first eviction and the malloc
  behavior.

## Prefill from the staging buffers (M1, uncommitted)

M1 would run prefill's expert matmuls straight from the two staging buffers that hold whole layer
packs, instead of copying the experts into the 512-slot bank (1,138 MiB). Two probes checked its
risks before building it.

### Strided expert views

In a layer pack each expert's gate, up and down weights lie back to back, so prefill would use
views with the pack stride between experts instead of contiguous `[ne0, ne1, 512]` tensors. The
Metal `mul_mat_id` kernels address experts as `i02*nb02`, and no fusion or support check requires
contiguous weights. A standalone test (`/Volumes/Data/moeshift-scratch/m1/probe1.cpp`) read one real
layer file per layout (10 layouts: gate/up in IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS or IQ3_S, down in
IQ4_NL or Q2_0) and ran gate, up, SiLU·up and down with 10 random experts per token, from a
contiguous copy and from views into the pack:

| Rows | Gate/up and down results | Time, contiguous → strided |
|---|---|---|
| 6,144 | bit-identical on all 10 layouts | 28.8–33.1 ms, −0.1% to +0.1% |
| 1,024 | bit-identical on all 10 layouts | 6.3–7.4 ms, −0.2% to +0.3% |

### Pipeline timing

A probe build logged per layer when the reader got a buffer and finished its read, and when each
install started, had its layer ready and finished the copy (`config-source.json`, 1,209-, 3,996-,
6,060- and 42,389-token prompts; 11 batches). A replay from those times reproduced every batch, then
applied M1's rules: no copy, and a staging buffer held until the GPU has run that layer's expert
matmuls (the probe's 7–33 ms after the install):

| Prompt | Reader idle | GPU waiting for reads | Copy | Today | M1 (replay) |
|---|---:|---:|---:|---:|---:|
| 1,209 tokens | 0 s | 4.03 s | 0.26 s | 5.47 s | 5.46 s (−0.1%) |
| 3,996 | 0 s | 1.89 s | 0.29 s | 6.06 s | 6.05 s (−0.1%) |
| 6,060 | 0.64 s | 0.18 s | 0.28 s | 6.77 s | 6.65 s (−1.9%) |
| 42,389 (7 batches) | 0.98–9.97 s per batch | 0.09–0.15 s per batch | 0.28–0.39 s per batch | 84.39 s | 82.51 s (−2.2%) |

Times run from the first to the last install of each batch. Up to 4k tokens the SSD sets the
pace: the GPU finishes a layer long before the next read, so holding its buffer costs nothing.
From 6k tokens the GPU sets it and the reader is already idle, so the lost read-ahead does not
matter either; what remains is the copy, about 6 ms per layer. Holding each buffer until the next
install instead (an upper bound) would cost up to 4.5% at 6k tokens; releasing it at once (a lower
bound) would save 2.4%.

### Implementation

- **Staging buffer** (`init_stage` in `src/llama-moe-slot.cpp`): one Metal buffer with two halves
  of the largest layer pack (2 × 1,137.5 MiB), allocated at load from the expert buffers' type and
  kept at one page outside pipelined prefill. Views of each layer's gate, up and down weights
  (144) point into the half that layer always uses (even or odd among the MoE layers), with the
  pack stride between experts.
- **Third phase**: batches of 1,024 tokens or more run in a *staged* phase, in which the bank and
  the decode windows are released and the staging buffer is active (`set_adaptive_phase`).
  Smaller prefill batches, such as the last 4 tokens the server splits off a prompt, keep the
  bank phase; decode is unchanged. `build_qwen_moe_ffn` uses the views in the staged phase, and
  the slot-id op passes the expert ids through after waiting for its layer's read.
- **Release**: a half is freed by a CPU op on one element of its layer's expert output, so the
  reader starts the read two layers ahead as soon as the GPU has finished those matmuls. The
  first version freed it at the next layer's slot-id op, after that layer's attention too, and
  was 2.5% slower at 4k tokens, 5.5% at 6k and 1.8% at 42k (the probe's upper bound).
- The copy into the bank is gone. If the staging buffer cannot be allocated or released, the
  worker logs it and uses the bank for every batch, as with `LLAMA_MOE_PREFILL_PIPELINE=0`.

Interleaved A/B against the committed build (`09a6193`), two rounds, fresh worker each,
`config-source.json` settings, the same four requests as R2:

| Request | Footprint peak | System peak | Prompt |
|---|---:|---:|---:|
| 72 tokens | 6.99 → 6.99 GB | 8.10–8.18 → 7.60–8.44 GB | 1.69–1.82 → 1.70–1.74 s |
| 3,996 tokens | 9.24 → **8.18 GB** | 11.19–11.25 → **9.29–10.11 GB** | 6.23 → 6.16–6.20 s |
| 6,060 tokens | 9.66 → **8.56 GB** | 12.44–12.48 → **10.29–11.18 GB** | 7.43–7.45 → 7.38–7.40 s |
| 42,389 tokens | 14.71 → **13.58–13.61 GB** | 16.20–16.32 → **15.08–15.74 GB** | 88.3–88.4 → **85.9 s** |

All responses were identical to the committed build's, and generation speed was unchanged
(23.27–23.40 against 23.30–23.40 tok/s after the long prompt). The footprint, which counts the
CPU-written bank and staging pages, fell by 1.06–1.13 GB, as the bank's 1,138 MiB predicts; the
system-wide figures vary more between runs. The 42k prompt was 2.8% faster, close to the replay's
2.2%; at 4–6k tokens the gain was under 1%. With the pipeline off, the same build answered
identically from the bank (4k: 8.76 s, 6k: 12.11 s, as before M1).

### Not tested

- Contexts above 42k tokens.
- Qwen3.6, whose profile also uses the pipeline.
- The Homebrew build, which keeps the copy into the bank.

## n-gram drafting on agent-style output (N2 measurement)

N2 drafts tokens by looking up the last few generated tokens in the context and proposing what
followed them there; one multi-row decode step then checks the draft. The server's n-gram
drafters and multi-row checks under LES already work (phases A and B). Since P0 removed the
protected slots, the 20-slot decode windows allow checks of 2 rows (1 drafted token).

### How much output can be drafted

Greedy reference outputs (`config-source.json`, thinking off unless noted) for eight tasks with
real repository files as context: an edit tool call (`tools` field), the same with thinking on,
rewriting a Python function, reproducing a 169-line C++ file with comments added, explaining code
with quotes, editing a JSON config, `short.json` (a story) and a summary of `long.json`. A harness
(`/Volumes/Data/moeshift-scratch/n2/replay.cpp`) replayed the server's own drafters over the
reference tokens the way the server drives them; for greedy decoding this gives exactly the
drafts and acceptances a real run would see. Predicted speedups, with a check of k drafted tokens
costing 1 + 0.483·k single steps (from the earlier 4-row measurement):

| Task | Draft 1 token | Draft 3 | Draft 7 |
|---|---:|---:|---:|
| Reproduce a whole file | 1.31× | 1.54× | 1.69× |
| Edit a JSON config | 1.31× | 1.53× | 1.67× |
| Rewrite a function | 1.22× | 1.35× | 1.38× |
| Edit tool call | 1.08× | 1.07× | 1.03× |
| Thinking before an edit | 1.05× | 1.04× | 1.04× |
| Explain code with quotes | 1.04× | 1.03× | 1.02× |
| Story, summary | 1.00× | 1.00× | 1.00× |

Best drafter per cell; `ngram-simple` with its default 12-token key was best or close everywhere,
and never drafted on prose. Short keys (3 tokens) drafted on prose and would have slowed it by up
to 36%. The edit task gains little overall because most of it is prose: the model reasoned for
390 tokens before its 187-token tool call. The call itself drafts well, since this model's chat
template writes arguments as XML-style parameters with the code unescaped
(`<parameter=old_string>` followed by the code as it appears in the file). Within the call,
`ngram-simple` with a 12-token key gives 1.12× (1 drafted token) and 1.16× (3); a 6-token key
1.17× and 1.24×, because it resyncs sooner after each changed line, but it also drafts on prose.
Drafting through JSON escaping, considered for tool calls, would gain nothing with this model.

### End-to-end

`spec-type: ngram-simple`, against the same build without speculation (two interleaved rounds for
2 rows, one for 3 and 4):

| Task | No speculation | 2 rows | 3 rows (10 protected slots) | 4 rows (24 protected slots) |
|---|---:|---:|---:|---:|
| Reproduce a whole file | 20.63 / 20.68 tok/s | **25.39 / 25.48** | 26.91 | **32.31** |
| Edit a JSON config | 22.65 / 22.45 | **27.86 / 27.95** | 29.61 | **35.77** |
| Rewrite a function | 19.99 / 20.04 | **22.34 / 22.72** | 22.98 | **26.11** |
| Edit tool call | 21.09 / 20.50 | 21.44 / 20.36 | 20.90 | 22.20 |
| Story | 24.46 / 24.58 | 24.38 / 23.44 | 23.59 | 24.46 |
| Summary (6k prompt) | 20.62 / 21.43 | 21.38 / 21.40 | 20.69 | 21.53 |

Every response matched the reference (tool calls apart from their random ids). Over 99% of
2-row drafts were accepted on the copy-heavy tasks. A 2-row check cost about 1.6 single steps
(whole-file run). The 4-row runs gained more than predicted, probably because the protected
slots also hold experts the drafted rows need. Drafting took 2.3 ms in total over 1,034 steps.

Memory (system-wide and footprint peaks, `mem.sh` sequence, one run each):

| Setting | 72 tokens + 512 generated | 4k | 6k + 512 generated | 42k |
|---|---:|---:|---:|---:|
| No speculation | 6.99 GB fp | 8.18 GB | 8.56 GB | 13.61 GB |
| 2 rows, `spec-draft-n-max: 1` | 7.10 GB | 8.30 GB | 8.80 GB | 13.70 GB |
| 4 rows, 24 protected slots | 9.35 GB | 9.35 GB | 10.07 GB | 13.93 GB |

With 2 rows the cost is one recurrent-state snapshot (`spec-draft-n-max: 1`; the default 3
keeps three). The 24 protected slots add about 2.4 GB while generating, which now sets the peak
for prompts up to about 6k tokens.

### Setting

`config-source.json` now uses the 2-row option. Through the router, the config-editing task ran
at 28.93 tok/s with 365 of 373 drafts accepted and the reference output. The shipped profile is
unchanged: its prefetch depth of 3 gives 15-slot windows, which allow only 1 row, so speculation
would be disabled with a message.

### Prefetching a copied token's experts (probe, not kept)

The idea: a token the drafter copies from the context might use the experts its source position
used during prefill, which could be read ahead for the drafted row. A temporary build dumped every
routing decision; for the three copy-heavy tasks, without speculation, each generated token whose
preceding 12 tokens occurred earlier (with the same next token) was compared with that source
position:

| Task | Copied tokens | Same experts as the source | As the previous token | Chance |
|---|---:|---:|---:|---:|
| Reproduce a whole file | 1,095 of 1,200 | 24.7% | 22.8% | 5.9% |
| Edit a JSON config | 717 of 778 | 29.6% | 26.3% | 5.2% |
| Rewrite a function | 437 of 700 | 27.7% | 21.3% | 5.3% |

Only the first layers route mostly by the token itself: about 90% overlap at layer 0, 60–80% at
layers 1–3 and about 50% at layers 4–7, then 10–20% through the middle and 30–50% in the last
layers, which depend on the context. Reading the source's experts for drafted rows would cover a
quarter of their experts, about what the existing one-layer lookahead already predicts, so it was
not built. (`/Volumes/Data/moeshift-scratch/n2p`)

### Not tested

- Real agent sessions; how much of an agent's output is copy-heavy decides the overall gain.
- Thinking output beyond the one task (no gain there), and `q8_0` KV.

## Per-turn prompt latency (uncommitted)

In an agent session each turn appends new text (tool output, a file) to a cached conversation,
and the wait before the reply is mostly its prefill. Two growing conversations
(`/Volumes/Data/moeshift-scratch/turns/turns.py`) appended messages of about 64 to 4,000 tokens
of C++ source in shuffled order, with `config-source.json` settings (2-row n-gram drafting) and
16-token replies.

### What a turn costs

- **Expert reads set a floor of about 4.5–6 s from a few hundred tokens up.** A batch of 1,024
  tokens or more reads all 40 GB of experts (about 8.8–9.3 GB/s here, SSD plus file cache; the SSD
  alone reads about 7.2 GB/s). Smaller batches read only the experts they use, but by about 500
  tokens that is most of them, read slot by slot. A 79-token turn took 2.2 s, 485 tokens 5.6 s,
  1,200 tokens 6.7 s, 4,400 tokens 7.3 s. There is no step at 1,024 tokens.
- **Extra small batches.** The server split each turn at the start of the latest user message (a
  batch of about 9 tokens) and before the last 4 tokens. With one checkpoint kept, the first split
  is useless: the checkpoint before the last 4 tokens replaces it in the same prompt.
- **Reprocessing after short messages.** Checkpoints had to be at least 256 tokens apart, so after
  a shorter message no checkpoint was taken near its end, and the next turn restored the older one
  and processed the short message again: 310 tokens in 7.39 s for a 64-token turn, 3,488 tokens in
  11.0 s for a 3,000-token turn.
- **Replies that stay in the prefill phase.** A 2-token speculative check right after the prompt
  counted as prefill (multi-token batches were decode only once decode was active), so when every
  step drafted, a whole reply ran as 2-token prefill batches reading experts through the bank.

### Changes

- `server-les.cpp` sets `checkpoint_min_step` to 0: the checkpoint 4 tokens before each prompt's
  end is always taken (it replaces the previous one; creating it took a few milliseconds).
- `server-context.cpp` does not split the batch before the latest user message when only one
  checkpoint is kept and there is no minimum spacing.
- `decode_rows` in `src/llama-moe-slot.cpp`: every batch of at most `LLAMA_MOE_DECODE_ROWS` tokens
  runs in the decode phase.

Same two conversations, committed build against these changes:

| Turn | Before | After |
|---|---:|---:|
| 3,000 tokens after a 128-token turn | 3,488 processed, 10.42 s | 3,292 processed, **6.98 s** |
| 64 tokens after a 128-token turn | 310 processed, 7.39 s | 134 processed, **3.55 s** |
| 768 tokens after a 64-token turn | 680 processed, 6.32 s | 605 processed, **5.61 s** |
| Other turns | | 0.2–0.45 s faster each (one 9-token batch fewer), some within ±0.3 s |
| Total prompt time, conversation A / B | 67.6 / 69.0 s | **61.2 / 64.7 s** |

All 22 replies matched the committed build. The n-gram speed tests (N2) gave the same speeds and
outputs with these changes: in those tasks drafting did not start on the first step.

### File cache: prefill bypasses it

Expert reads went through macOS's file cache in every phase. SSD traffic (`iostat`) and expert
residency (`mincore` on the layer files, `/Volumes/Data/moeshift-scratch/pc`) around single
requests showed:

- About 14–15 GB of experts fit in the file cache next to the worker.
- **Prefill hit almost nothing:** a staged prefill reads all 48 layers in order, 40.9 GB through a
  14 GB cache, so each layer was evicted before it was needed again (40.5 and 38.9 GB read from the
  SSD). Afterwards only layers 34–47 were cached; generation then started with a cold cache.
- Generation read about 75–95 MB per token from the SSD and kept 300–400 MB per layer cached.

The worker now sets `LLAMA_MOE_F_NOCACHE=auto` (it forced `0` before): prefill reads use
`F_NOCACHE`, decode reads go through the cache (`process()` in `src/llama-moe-slot.cpp` switches
the expert files at each phase change, now also before staged batches). Prefill still finds what
decode left cached, and no longer replaces it. Bypassing the cache in every phase (`1`) would slow
generation, as measured in G1.

| Request | Cached prefill: SSD, prompt | Prefill bypass: SSD, prompt | First 16 layers kept cached in prefill |
|---|---:|---:|---:|
| 4k prompt, first | 39.5 GB, 6.45 s | 34.3 GB, 5.85 s | 37.3 GB, 6.20 s |
| 6k prompt + 512 tokens | 83.3 GB, 7.39 s | 78.4 GB, 7.04 s | 73.8 GB, 7.35 s |
| 4k prompt again | 38.7 GB, 6.38 s | **27.9 GB, 4.86 s** | 37.1 GB, 6.22 s |
| 72 tokens + 512 tokens | 43.3 GB, 2.12 s | 42.8 GB, 1.79 s | 44.8 GB, 2.01 s |

Generation ran at 22.0–22.1 and 24.5–24.9 tok/s in all three; outputs were identical. Keeping a
fixed set of layers cached during prefill did not hold: decode reads evicted them between prompts.

The two conversations, the per-turn changes alone against both:

| | Per-turn changes | + prefill bypass |
|---|---:|---:|
| Total prompt time, conversation A / B | 61.6 / 62.2 s | **52.7 / 53.4 s** |
| Each turn | | 0.4–1.1 s faster |

All replies matched. With 16-token replies, generation warmed the cache only a little between
turns; longer replies should leave more for the next prefill (not measured).

### Not tested

- Thinking turns, whose re-rendered history drops the reasoning and may need longer rollbacks.
- Clients that edit or regenerate the last user message, which used the checkpoint before it.

## Expert predictor on the Neural Engine (N7, stage 1: the ceiling)

The idea, tried before in other forms: a small model that predicts which experts each layer will
need early enough to read them before the layer runs, on the Neural Engine so that it costs no GPU
time. Earlier guesses were accurate but late: N6a named 91–94% of the next layer's experts with
16 guesses, but only about 356 µs ahead and with GPU work of its own, and gained 1.8% (nothing
after P0). The same-token data from the N2 probe shows the early layers are largely decided by
the token itself (73% of layer 0's experts, 45–50% at layers 1–7, for the same token after a
different previous token), the middle layers by context (25–35%).

**Stage 1: what reads cost now.** `config-source.json` settings without speculation, 512 tokens
with `ignore_eos`, normal against `LLAMA_MOE_SKIP_READ=1` (timing only, output invalid), two
interleaved rounds:

| After | Normal | Reads skipped | Reads on the critical path |
|---|---:|---:|---:|
| `short.json` | 24.45 / 24.88 tok/s (40.2–40.9 ms) | 36.07 / 36.10 tok/s (27.7 ms) | 12.5–13.2 ms (31–32%) |
| `long.json` | 22.26 / 22.56 tok/s (44.3–44.9 ms) | 33.35 / 33.36 tok/s (30.0 ms) | 14.3–14.9 ms (32–33%) |

Hiding every expert read would make generation about 45–48% faster (35% in G5, before the GPU-side
changes). Normal generation read about 85–95 MB per token from the SSD (`iostat`), the file cache
serving the rest of about 100 misses per token. Read 27.7 ms apart, those bytes need about
3.1 GB/s, within the SSD's 7.2 GB/s: bandwidth would not limit a good predictor, lead time and
accuracy would.

## Expert predictor on the Neural Engine (N7, stage 2: offline accuracy)

A temporary build dumped each layer's router input (2,560 values, fp16) and the experts it chose,
while prefilling 16 documents: about 17,700 tokens to train on (engine C++ and Python sources,
these notes, the long prose prompt, model-generated code) and 6,000 held out (other sources,
LES docs, generated prose, the TODO). Routing does not depend on whether a token is prefilled or
generated. For every layer pair up to 4 apart, predictors of layer L+d's experts from layer L's
router input were scored on the held-out tokens: the share of the 10 experts a token used that
are among the top 8 or 16 guesses (`/Volumes/Data/moeshift-scratch/ane/stage2.py`; PyTorch 2.8 on
the GPU, 146 s for all of them).

| Layers ahead | L+d's router on L's input, 8 / 16 guesses | Trained linear, 8 / 16 | Two-layer net (512 hidden), 8 / 16 |
|---:|---:|---:|---:|
| 1 | 55.7 / 76.3% | 58.3 / 80.4% | 59.3 / 81.3% |
| 2 | 47.4 / 65.9% | 55.6 / 77.3% | 56.8 / 78.6% |
| 3 | 43.7 / 61.1% | 53.8 / 75.3% | — |
| 4 | 41.7 / 58.5% | 52.9 / 74.2% | — |

(Averages over target layers; per layer, the linear model with 16 guesses ranged 64–88%.) A
token → experts table from the training rows named 69% of layer 0's experts with 16 guesses
(52% with 8) and 46–52% at layers 1–7; 82% of held-out tokens occurred in the training rows.

- Training adds 11–16 points over the router heuristic from 2 layers ahead, and accuracy falls
  slowly with distance (80% → 74% from 1 to 4 layers ahead), so lead time is cheap in accuracy.
- The stop point set for this stage, about 85% with 16 guesses at 2 or more layers ahead, was not
  reached (77–79%).
- The input was limited: layer L's router input, before its expert output is added. N6a's late
  guess, from the residual after layer L with L+1's own mix, named 91–94% one layer ahead, against
  80% here, so a richer input should raise every row. The training set was also small.
- Recall of all experts used overstates what matters: experts already in the decode window are
  the easy ones, and every wrong guess costs a read (about 2 MB). The deciding measure is how many
  of today's misses a predictor catches within a fixed read budget, which a replay of the decode
  windows over these routing sequences can estimate.

## Expert predictor on the Neural Engine (N7, stage 2b: richer input, window replay)

A second dump over a larger corpus: 56,200 training tokens (24 C++ and Python sources, 7 Markdown
files, model-written prose and reasoning from ten prompts, the long prose prompt) and 10,200
held out (other sources and prose). Besides each layer's router input, it saved the residual after
layers 3, 7, …, 43 (all 4 hyper-connection streams, 10,240 values). Predictors of layer L+d's
experts were trained per sampled L and scored on the held-out tokens, by recall and by a replay of
the decode windows: a 20-slot LRU window per layer, reset at each document like a new reply,
filled per token with up to 8 read-ahead guesses and then the 10 experts used
(`/Volumes/Data/moeshift-scratch/ane/stage2b.py`, 413 s).

| Layers ahead | Today's guess (T's router on T−1's input), 8 | Linear on L's router input, 8 / 16 | Linear on L's residual, 8 / 16 | Net on L's residual (1,024 hidden), 8 / 16 |
|---:|---:|---:|---:|---:|
| 1 | 59.0% | 59.0 / 81.5% | 55.5 / 76.4% | 60.7 / 82.3% |
| 2 | 58.7% | 56.5 / 78.9% | 53.8 / 74.7% | 59.2 / 81.1% |
| 3 | 55.2% | 54.8 / 76.7% | 54.8 / 76.1% | — |
| 4 | 51.3% | 49.4 / 69.9% | 47.6 / 67.0% | — |

Window replay, misses per layer and token (8 reads ahead per layer):

| Layers ahead | No read-ahead | Today's guess | Best predictor |
|---:|---:|---:|---:|
| 1 | 5.16 | 2.66 | 2.37 |
| 2 | 5.40 | 2.71 | 2.59 |
| 3 | 5.93 | 3.05 | 3.00 |
| 4 | 5.06 | 2.95 | 3.08 |

- The residual did not help a linear model, and a network on it gained 1–2 points: with 8 guesses
  every predictor named about 59% of the experts, the same as today's one-layer-ahead guess.
- Two layers ahead, the best predictor removed about 4% of the misses today's guess leaves (the
  replay ignores timing). Its remaining advantage is arriving on time: N3 found 14% of misses
  were guessed but late. Both together might be worth about 2 ms per token (about 5%), before
  the cost of running a model on the Neural Engine every token and copying the residual off the GPU.
- **Not continued.** Two or more layers ahead there is not enough information in this layer's
  state to name more of the next layers' experts within 8 reads. A different input (routing
  history over several tokens) or more expert cache (memory) would be needed; the LRU window and
  the file cache already capture much of both.

## Expert predictor from a small model (N7, stage 2c)

Qwen3.5-0.8B tokenizes the corpus exactly like Flash Next, so its states line up with the stage 2b
routing token by token. A small tool (`/Volumes/Data/moeshift-scratch/ane/small/hid.cpp`, on the GPU
for this study) saved the outputs of its layers 5, 11 and 17 (3 × 1,024 values per token); MLPs
(1,024 hidden) were trained per big-model layer on the same train/held-out split
(`stage3s.py`, 467 s):

| Big-model layers | Today's guess, 8 | Small model, same token, 8 / 16 | Small model, next token, 8 / 16 | Small model + big model's router input of T−2, 8 / 16 |
|---|---:|---:|---:|---:|
| All | 54.9% | 45.6 / 65.1% | 35.6 / 52.8% | 58.1 / 79.7% |
| 0–7 | 48.4% (none for layer 0) | 58.7 / 80.9% | 45.3 / 65.2% | 60.9 / 83.0% |
| 8–39 | 56.1% | 43.5 / 62.7% | 34.3 / 51.4% | 57.6 / 79.3% |
| 40–47 | 55.6% | 40.6 / 58.7% | 30.7 / 46.2% | 57.9 / 78.7% |

Window replay (20 slots, 8 reads ahead), misses per token over all layers: no read-ahead 265,
today's guess 147, the small model alone 165, the small model with the big model's state two
layers back 127 (−14%).

- The two are complementary: the small model alone is weaker than today's guess past layer 7,
  but with the big model's router input from two layers back it names more experts than today's
  one-layer-ahead guess, and early enough not to be late.
- Layer 0, which nothing predicted before (see L0), gets 68% of its experts with 8 guesses (91%
  with 16) from the small model's state for the same token: 8.9 → 2.9 misses per token.
- Predicting the next token's experts from the small model alone is weak (36% with 8).
- Not yet known: whether the small model can run fast enough and cheaply enough per token (it must
  finish token t before the big model's layers need the guesses), and the memory of the heads (48
  MLPs of 1,024 hidden are about 300 M parameters; smaller heads untested).

### On the Neural Engine (N7, stage 3a)

A PyTorch decode step with Qwen3.5-0.8B's shapes for layers 0–17 (the predictor uses the outputs of
5, 11 and 17; random weights): gated delta-net layers with their 16 × 128 × 128 recurrent and
convolution state, a gated full-attention layer every fourth layer with a 1,024-token KV cache,
SwiGLU FFNs; all state in Core ML states (one per layer, written by slice assignment). Converted
with coremltools 9 (fp16, macOS 15 target, `/Volumes/Data/moeshift-scratch/ane/small/ane_probe.py`):

- The compute plan for CPU + Neural Engine puts all 1,364 non-constant operations on the Neural
  Engine, including the state reads and writes.
- One step (100 runs after warm-up): 6.78 ms median (p90 7.30) on CPU + Neural Engine, 7.15 ms
  with all units, 4.20 ms on CPU + GPU; CPU only failed to load (error −14, probably the fp16
  states). Loading took 2.5 s.

**Running alongside the big model** (`ane_loop.py`, `conc.sh` in the same folder): 512 tokens
with `ignore_eos` after `short.json` and `long.json`, the step looping in a separate process on
CPU + Neural Engine, two interleaved rounds:

| Neural Engine load | After `short.json` | After `long.json` | `long.json` prompt | Small-model step |
|---|---:|---:|---:|---:|
| Off | 25.78 / 26.63 tok/s | 25.02 / 24.98 tok/s | 7.03 / 7.04 s | — |
| One step every 40 ms (one per token) | 26.11 / 26.22 tok/s | 24.47 / 24.60 tok/s | 7.14 / 7.17 s | 7.33–7.35 ms (p99 7.8) |
| As fast as possible (about 140 steps/s) | 23.88 / 23.90 tok/s | 22.61 / 22.55 tok/s | 7.68 / 7.70 s | 7.08 ms |

At one step per token the big model lost 0–2% (none after the short prompt, 1.9% after the long
one; its prompt 1.6%); flat out 9–10%. The shared resource is most likely memory bandwidth: the
fp16 weights add about 19 GB/s at 25 steps per second. The paced loop also ran during prompts and
between requests, which a real integration would not.

**4-bit palettized weights** (`palettize.py`: k-means, a lookup table per group of 16 channels,
`coremltools.optimize.coreml.palettize_weights`): the compiled model shrank from 716 to 180 MB, all
1,364 operations stayed on the Neural Engine (the 140 lookup-table expansions are constants), and a
step took 4.36 ms alone (p90 4.41). Alongside the big model, same test:

| Neural Engine load, 4-bit | After `short.json` | After `long.json` | `long.json` prompt | Small-model step |
|---|---:|---:|---:|---:|
| Off | 26.18 / 26.24 tok/s | 24.79 / 24.76 tok/s | 7.08 / 7.10 s | — |
| One step every 40 ms | 26.07 / 26.12 tok/s | 24.67 / 24.55 tok/s | 7.08 / 7.09 s | 4.76–4.78 ms (p99 5.0) |
| As fast as possible (about 220 steps/s) | 25.37 / 25.21 tok/s | 23.98 / 23.82 tok/s | 7.31 / 7.35 s | 4.57 ms |

At one step per token the big model lost 0.4–0.7%, its prompts nothing. The weights are random,
so the 4-bit model's hidden states (and how well the predictor works on them) are untested; a
predictor would have to be trained on the 4-bit model's own outputs.

Still open: predictions
arrive about 7 ms after a token is sampled, in time two layers ahead only from about layer 10,
unless the step is split so that layer 5's output comes earlier; fp16 weights take about 0.76 GB
(4-bit palettization untested); the KV cache grows with context (about 0.6 GB at 60k tokens);
prompts need a batched prefill path. Core ML compiled into the internal SSD's temporary folder
twice before compilation was pointed at `/Volumes/Data` (about 1.5 GB written).

### Real weights on the Neural Engine (N7, stage 3b)

The PyTorch step with Qwen3.5-0.8B's real weights (dequantized from the Q8_0 GGUF,
`/Volumes/Data/moeshift-scratch/ane/small/real_step.py`) matched llama.cpp's layer 5/11/17
outputs (cosine 1.0000 over 96 tokens). Converted to Core ML with a 4,096-token KV cache
(`convert_real.py`) and run on CPU + Neural Engine:

| Version | Step | Cosine with fp32, layers 5 / 11 / 17 (median, first 96 tokens) | Lowest |
|---|---:|---:|---:|
| fp16 | 8.77 ms | 0.983 / 0.967 / 0.979 | 0.92 |
| 4-bit palettized | 5.99 ms | 0.941 / 0.912 / 0.936 | 0.61 |

The whole corpus, token by token from a fresh state per document (`run_corpus.py`; one
6,246-token training document restarted its state every 4,096 tokens), took 396 s at 4-bit and
581 s at fp16. Agreement with llama.cpp's outputs by position (`drift.py`, median cosine):

| Layer | fp16, positions 0–64 / 64–256 / 256–1,024 / 1,024–4,096 | 4-bit, same |
|---|---|---|
| 5 | 0.982 / 0.984 / 0.984 / 0.983 | 0.943 / 0.950 / 0.952 / 0.949 |
| 11 | 0.972 / 0.963 / 0.956 / 0.947 | 0.927 / 0.914 / 0.902 / 0.888 |
| 17 | 0.982 / 0.975 / 0.970 / 0.964 | 0.946 / 0.938 / 0.931 / 0.922 |

Layer 5 stays steady; layers 11 and 17, which carry the recurrent and attention state of earlier
layers, drift further along the document (fp16 rounding accumulating in that state). Heads
retrained on each version's own outputs (`heads.py`) predicted as well as on llama.cpp's:

| States from | Small model alone, 8 guesses | Small + big model's input of T−2, 8 guesses | Misses per token, small + big (today's guess 147) | Layer 0, 8 guesses |
|---|---:|---:|---:|---:|
| llama.cpp (Q8) | 45.6% | 58.1% | 127 | 67.9% |
| Core ML fp16 | 44.9% | 58.0% | 128 | 68.3% |
| Core ML 4-bit | 44.3% | 57.9% | 128 | 68.8% |

At up to 4,096 tokens the drift costs nothing measurable. Untested: contexts of 16–64k, where it
keeps growing; a turn could start from an exact state if the small model's prompt is processed by
llama.cpp on the GPU and its state handed over, so drift would accumulate only over each reply.

### Smaller heads (N7, stage 3c)

The heads above (1,024 hidden, inputs 3,072 + 2,560) have about 297 M parameters (594 MB at fp16).
Retrained on the 4-bit Neural Engine model's outputs, with the same window replay
(`/Volumes/Data/moeshift-scratch/ane/small/shrink.py`); "PCA 256" compresses the small model's
3,072 values once per token (64% of their variance); "router(T)" feeds layer T's own router applied
to the big model's input of T−2 (512 scores, no new weights) instead of that 2,560-value input:

| Heads | Misses per token (today's guess 147) | Recall with 8 guesses | Parameters | fp16 |
|---|---:|---:|---:|---:|
| Small 3,072 + big 2,560, 1,024 hidden (above) | 128.0 | 57.9% | 297 M | 594 MB |
| Small 3,072 + big 2,560, 256 hidden | 133.7 | 55.9% | 74 M | 148 MB |
| Small PCA 256 + big 2,560, 256 hidden | 132.2 | 56.3% | 40 M | 81 MB |
| Small PCA 256 + router(T), 256 hidden | 129.2 | 58.0% | 16 M | 33 MB |
| Small PCA 256 + router(T), linear | 128.4 | 57.6% | 19 M | 38 MB |
| Small PCA 256 + router(T), 128 hidden | 133.1 | 56.7% | 9 M | 17 MB |

Feeding the big model's state through the target layer's existing router keeps the whole gain with
a fraction of the weights: a linear head of 768 → 512 per layer (38 MB at fp16, about 19 MB at
8-bit, untested) matched the full heads. With the 4-bit small model (180 MB) and its cache, the
predictor would take about 250 MB at short contexts.

### Long context (N7, stage 3d)

A 16,643-token document (the start of these notes, in a chat prompt) prefilled once by the dump
build for its routing and router inputs, llama.cpp's small-model states as the exact reference,
and two Neural Engine versions of the 4-bit step: one with a 20,480-token KV cache, and the 4k
step used as a ring buffer (attention over the last 4,096 tokens, rotary positions absolute, the
delta-net state over the whole document). The compact linear heads, trained on the corpus
(documents under 4k tokens), were applied to the long document; the window replay ran through it
without a reset (`/Volumes/Data/moeshift-scratch/ane/small/run_long.py`, `run_window.py`,
`long_eval.py`).

| | 20k cache | 4k ring buffer |
|---|---:|---:|
| Step, at every position | 15.27 ms | 5.94 ms |
| KV cache of the small model | about 170 MB | about 32 MB |

The full cache costs the same at every position because the step reads all of it.

Median cosine with llama.cpp's states by position (the ring buffer also differs by design past 4k):

| | Layer | 0–1k | 2–4k | 4–8k | 8–12k | 12–16.6k |
|---|---|---:|---:|---:|---:|---:|
| 20k cache | 5 | 0.947 | 0.949 | 0.951 | 0.942 | 0.936 |
| | 11 | 0.900 | 0.881 | 0.877 | 0.863 | 0.839 |
| | 17 | 0.932 | 0.921 | 0.916 | 0.908 | 0.893 |
| Ring buffer | 11 | 0.900 | 0.881 | 0.841 | 0.812 | 0.795 |
| | 17 | 0.932 | 0.921 | 0.894 | 0.872 | 0.856 |

Misses per token in the window replay:

| Position | 0–1k | 1–2k | 2–4k | 4–8k | 8–12k | 12–16.6k |
|---|---:|---:|---:|---:|---:|---:|
| Heads, 20k cache | 127.8 | 128.3 | 139.5 | 140.0 | 139.1 | 147.8 |
| Heads, 4k ring buffer | 127.9 | 128.1 | 139.7 | 140.2 | 139.7 | 148.5 |
| Today's guess | 143.0 | 144.6 | 153.1 | 151.6 | 151.3 | 156.6 |

The ring buffer predicts as well as the full cache at a constant 6 ms step and 32 MB. Against today's
guess the reduction shrinks along the document, from 11% to 5–6%; the heads never saw positions past
4k in training, which training on long documents would address (untested).

### Heads trained on long documents (N7, stage 3e)

Five more training documents of 12–18k tokens (the first 52,000 characters of
`src/llama-moe-slot.cpp`, `tools/server/server-context.cpp`, `ggml-metal-ops.cpp`,
`src/llama-graph.cpp` and the other Markdown docs; 75,433 tokens) and a second held-out long
document (`src/llama-context.cpp`, 14,528 tokens), dumped like the others; the small model's states
from the 4k ring buffer on the Neural Engine (537 s for all six). Compact linear heads trained on the
short corpus only, or on the short corpus plus the five long documents
(`/Volumes/Data/moeshift-scratch/ane/small/long_train.py`). Misses per token in the window replay:

| Held-out | Position | Short-trained | Short + long-trained | Today's guess |
|---|---|---:|---:|---:|
| Notes, 16.6k tokens | 0–1k | 129.5 | 123.0 | 143.0 |
| | 1–2k | 130.1 | 124.2 | 144.6 |
| | 2–4k | 141.7 | 134.1 | 153.1 |
| | 4–8k | 142.1 | 133.3 | 151.6 |
| | 8–12k | 141.6 | 132.2 | 151.3 |
| | 12–16.6k | 150.6 | 140.2 | 156.6 |
| `llama-context.cpp`, 14.5k tokens | 0–1k | 125.6 | 116.1 | 145.7 |
| | 1–2k | 137.6 | 128.3 | 154.9 |
| | 2–4k | 152.1 | 140.7 | 169.6 |
| | 4–8k | 154.8 | 142.5 | 172.0 |
| | 8–12k | 162.7 | 149.1 | 178.6 |
| | 12–14.5k | 151.1 | 140.4 | 171.2 |
| Short held-out set | all | 130.3 | 124.6 | 146.5 |

With long documents in training, the heads cut misses against today's guess by 10–14% on the notes
(the smallest gain at the end), 17–20% on the long source file, and 15% on the short held-out set
(11% when trained on the short corpus alone): more training data helped at every length.

### In the engine (N7, stage 4: guesses injected from a file, not kept)

Before building the Neural Engine path into the engine, the heads' guesses were computed offline for
a fixed greedy run and fed to the engine, to measure the real gain. A scratch build (worktree
`/Volumes/Data/moeshift-scratch/n7i/wt`, patch `n7i/patch_n7i.py`, not in the repository) dumped
the tokens, routing and router inputs of `short-ie.json` and `long-ie.json` (512 tokens each,
`config-source.json` settings without speculation, one server for both). `n7i/make_guesses.py` ran
the 4-bit small model with its 4k ring buffer over each request's tokens on the Neural Engine and
trained the stage 3e heads (short + long documents). It wrote the heads' guesses per decode step
and layer, and today's guesses as a control. The engine then replaced the graph's guess for layer
L+1 with the file's (same timing as today: read once layer L's demand reads finish). A check against
the recorded layer-0 routing found every step aligned, and every response matched the dump run.

On the model's own output, the window replay gave the heads 10% fewer misses than today's guess
after the short prompt and 6% after the long one. Starting at layer 8, roughly where the small
model's output would arrive in time (about 6 ms into a 38–40 ms step), it gave 7% and 4%. Three
interleaved rounds, fresh server each; medians, with demand misses per token from the probe's
counter:

| Guesses | After `short-ie.json` | After `long-ie.json` |
|---|---:|---:|
| Graph (baseline) | 26.22 tok/s, 160.9 misses | 24.89 tok/s, 180.2 misses |
| Today's, from the file (control) | 26.46 tok/s, 159.8 misses | 24.82 tok/s, 180.3 misses |
| Heads, layers 1–47 | 26.52 tok/s, 150.3 misses | 24.62 tok/s, 175.0 misses |
| Heads from layer 8, today's before | 26.41 tok/s, 154.6 misses | 24.76 tok/s, 177.7 misses |

The heads cut demand misses by 6% and 3% but gained nothing; after the long prompt they were
0.8% slower in all three rounds (ranges 24.51–24.67 against 24.78–24.84). With diagnostics on,
the control read about 174k experts on demand and 205k by prefetch (65% of them used) over the run.
The heads read about 166k and 210k (67.5% used): they moved about 2% of the reads from demand to
prefetch, and the total stayed within 1%.

**Ceiling.** The experts each layer actually used, injected the same way (two rounds each):

| Guesses | After `short-ie.json` | After `long-ie.json` | Demand misses, first 768 steps |
|---|---:|---:|---:|
| Today's, from the file | 26.49–26.67 tok/s | 24.87 tok/s | 125k |
| Exact, 8 of the 10 read | 28.65–28.97 tok/s (+8–9%) | 26.96–27.11 tok/s (+9%) | 55k |
| Exact, all 10 read | 27.63–28.20 tok/s | 25.56–26.67 tok/s | 55–57k |

At today's timing, even exact guesses gain only about 9%, for 56% fewer demand misses. The heads
remove about a tenth of those misses, worth about 1% at most, and the Neural Engine model would cost
0.4–0.7% and about 250 MB. The integration was stopped here. A predictor would have to remove about
half of the demand misses at this timing to approach 9%; a larger gain needs guesses well ahead of
today's one layer.

**Early guesses: what accuracy pays.** To decide whether fine-tuning the small model could be worth
it, a second scratch patch (`n7i/patch_early.py`) read guesses early. At the start of each decode
step, the guesses for layers F and up go into a queue in layer order. After an optional delay,
four threads read them into each layer's window through the read-ahead install path, which never
evicts the latest step's experts and drops its copy if a demand read got there first. Guesses for
a layer the step has already reached are dropped. Today's guess for the next layer stays on. F = 8
with a 6 ms delay approximates the Neural Engine model: its output arrives about 6 ms into the step,
when the big model is near layer 8. Exact guesses (the 10 experts used) and copies with some
replaced by random unused experts, all 10 read; two interleaved rounds, every response identical.
The baseline ran about 4% slower that session than in the table above.

| Early guesses | After `short-ie.json` | After `long-ie.json` | Demand misses per token, short |
|---|---:|---:|---:|
| None (baseline) | 25.16–25.23 tok/s | 23.28–23.52 tok/s | 165 |
| Exact, layers 1–47, at step start | 30.16–30.19 (+20%) | 28.31–28.45 (+21%) | 46 |
| Exact, layers 8–47, 6 ms in | 28.89–28.99 (+15%) | 27.16–27.24 (+16%) | 73 |
| 8 of 10 right, layers 8–47, 6 ms in | 24.66–24.69 (−2%) | 22.54–22.61 (−3%) | 107 |
| 6 of 10 right | 20.02–20.06 (−20%) | 18.63–18.71 (−20%) | 151 |
| 4 of 10 right | 16.90 (−33%) | 15.29–15.93 (−33%) | 194 |

Early, exact guesses would be worth about 15% at the Neural Engine's timing, but a wrong guess costs
a full expert read and can push a useful expert out of the window: below about 85% right it is a
loss. Random wrong guesses are the worst case, since they are never already in the window. Real
predictions err on plausible experts and can be limited to the confident ones, so heads on the
small model's state alone (`n7i/small_only.py`, PCA 256, linear, trained as stage 3e; precision of
the top 2 / 4 / 8 guesses 72% / 65% / 53%, recall 14% / 26% / 43%) were read the same way. Two
rounds, every response identical:

| Small-model heads, layers 8–47, 6 ms in | After `short-ie.json` | After `long-ie.json` | Demand misses per token, short / long |
|---|---:|---:|---:|
| None (baseline) | 25.14–25.30 tok/s | 23.44–23.66 tok/s | 165 / 186 |
| Top 2 read | 25.37–25.43 (+0.5%) | 23.25–23.48 (−0.8%) | 156 / 179 |
| Top 4 read | 25.37–25.43 (+0.5%) | 23.03–23.27 (−1.7%) | 148 / 173 |
| Top 8 read | 25.05–25.06 (−0.7%) | 22.46 (−4%) | 142 / 171 |

Fine-tuning was not tried. To pay, it would have to take the small model from 53% to about 85% precision
with most of the used experts named, from the tokens alone. Earlier stages found the token decides
most of layer 0's experts but only 25–35% of the middle layers'. The big model's own next-layer
signal, the best guess found (N6a), names 72% of the experts with 8 guesses.

**The big model's residual, k layers ahead.** Whether N6a's accurate late guess keeps its accuracy
further ahead. The scratch build dumped, for the same two requests (responses identical), the
residual streams after every layer, and for each decode step and layer the experts resident when
the routing arrived and the demand misses (`n7i/patch_resid.py`). `n7i/ahead.py` guessed layer T
from the residual after layer T−k. V1 applies layer T's full hyper-connection mix, then its router
(N6a's V1). V3 applies layer T's grouped norm only, each stream routed and summed (the late
prefetch's default form). A guess catches a miss if it names a demand miss and wastes a read if it
names an expert neither used nor resident. The gain is estimated from the early-read curve above,
which these two numbers fit within a few points: about +0.16% per miss caught and −0.14% per
wasted read. Layers 4–47 (145 of the 169 demand misses per token), top 4 guesses per layer:

| Guess | Lead before the layer's experts | Guesses used | Misses caught per token | Wasted reads per token | Reads that were needed | Estimated gain |
|---|---|---:|---:|---:|---:|---:|
| V1, k = 1 | about 0.35 ms (layer T's attention) | 97% | 27.5 | 2.7 | 91% | +4.1% |
| V1, k = 2 | about 1.1 ms | 88% | 25.4 | 11.6 | 69% | +2.5% |
| V1, k = 3 | about 1.9 ms | 78% | 24.9 | 27.4 | 48% | +0.1% |
| V1, k = 4 | about 2.6 ms | 69% | 23.7 | 42.0 | 36% | −2.2% |
| V3, k = 1 / 2 / 3 | | 92% / 86% / 81% | 24.8 / 23.4 / 23.8 | 7.0 / 13.2 / 19.7 | 78% / 64% / 55% | +3.0% / +1.9% / +1.0% |
| Today's guess (k = 1) | | 84% | 10.5 | 3.5 | 75% | +1.2% |

The best of any width: V1 k = 1 with 8 guesses, +9.1%; V1 k = 2 with 6 guesses, +3.1%; V3 k = 2 with 4
guesses, +1.9%. Accuracy falls off quickly with lead: the share of reads that were needed drops
from 91% to 69%, 48% and 36% for k = 1 to 4. Where it is high, the lead is too short for the
reads. N6a measured late reads at 185–193 µs at the median and 577–607 µs at p90, and its k = 1
version gained 1.8% before P0 and nothing after. These estimates assume every read finishes in
time, and they leave out the guesses' GPU cost: V1 reads about 13 MB of mix weights per layer, and
N6a's cheaper V3 chain cost 1.3–2 ms per token before it was trimmed. Two layers ahead, the best
estimate (+2.5–3%) is about what the guess costs to compute. No guess source found reaches the
accuracy that early reads need, so expert prediction was left here.

## Cache bounds for expert reads (after N7)

What any keep-or-evict policy could save, from the held-out stage 2b routing in token order
(`/Volumes/Data/moeshift-scratch/ane/bounds.py`, `l2bounds.py`; no read-ahead, so absolute
numbers are higher than in generation).

Decode window, per layer, reset per document:

| Policy | Misses per layer and token | Per token |
|---|---:|---:|
| LRU, 20 slots (today) | 5.53 | 265 |
| Belady's optimal, 20 slots | 4.11 | 197 |
| LRU, 40 / 64 slots | 4.11 / 3.01 | 198 / 144 |

A perfect replacement policy would cut window misses by 26%, as much as doubling the window
(about +1.7 GB). Most window misses are served by the file cache, though.

File cache, all window misses of the 66,400-token corpus in order through a cache shared by all
layers and kept across documents (about 14 GB holds roughly 7,000 of the 24,576 experts):

| Capacity | LRU | Admit only experts missed twice in the last 20,000 misses | Belady's optimal |
|---|---:|---:|---:|
| 7,000 experts | 56.9 SSD reads per token | 52.5 | 24.3 |
| 5,000 experts | 94.7 | 83.6 | 41.5 |

The simulated LRU figure is close to the 40–45 experts per token generation reads from the SSD
(85–95 MB). An optimal file-cache policy would read 57% fewer experts from the SSD; a simple
frequency rule recovers 8–12% of that. The engine controls admission (a second, `F_NOCACHE`
descriptor per file for reads that should not be cached), not the OS's eviction order.

**What an SSD read costs.** The same greedy 256-token generation after `short.json`, in one
worker, with the expert pages first pushed out of the file cache (reading 16 GB of unrelated
data) and then repeated with them cached; the SSD counter read when the first token streamed in
separates prefill from generation (`/Volumes/Data/moeshift-scratch/ane/ssdsplit.sh`):

| Run | Prefill: SSD, time | Generation: SSD per token | Generation |
|---|---:|---:|---:|
| Cold | 13.1 / 12.3 GB, 2.51 / 2.41 s | 75.9 / 74.0 MB | 41.13 / 41.17 ms per token |
| Warm | 3.5 / 3.6 GB, 1.10 / 1.12 s | 22.0 / 22.3 MB | 35.51 / 35.79 ms per token |

All four generated the same text. About 53 MB per token (roughly 30 experts) read from the file
cache instead of the SSD saved about 5.5 ms per token: about 0.1 ms per MB, 0.18 ms per expert.
Even fully warm, a token took 35.6 ms against 27.7 ms with reads skipped, so about 8 ms per token
is the copy from the file cache and the per-layer handshakes, not the SSD.

At about 85–95 MB per token from the SSD in a session's steady state, the optimal file-cache
policy above (57% fewer SSD reads) would be worth about 5 ms per token (+12–14%); a policy that
closes half the gap to it about +6–7%.

## Learned file-cache policy (N8, offline, not kept)

The file-cache access stream of decoding the stage 2b corpus, with today's read-ahead (8 guesses
per layer from the next layer's router), through a 7,000-expert cache; models trained on the first
60% of the stream, scored on the last 40% (`/Volumes/Data/moeshift-scratch/ane/l2learn.py`,
`l2evict.py`, `l2ra.py`). The stream had 371 accesses per token, 59% of them read-ahead.

| Policy | SSD reads per token: demand + read-ahead |
|---|---:|
| LRU, admit everything (today) | 37.3 + 42.0 = 79.3 |
| Admit if accessed in the last 10,000 accesses | 35.2 + 38.7 = 73.9 |
| Learned admission (MLP on recency, counts over 1k–1M accesses, layer, kind), best of 9 settings | 34.8 + 37.1 = 71.9 |
| Learned eviction: furthest predicted next use (MLP regression of reuse distance) | 50.7 + 53.6 = 104.3 |
| LFU-style eviction (accesses in the last 100,000) | 49.9 + 57.2 = 107.2 |
| Read-ahead misses not cached | 30.1 + 69.6 = 99.6 |
| Belady's optimum | 16.6 + 17.2 = 33.8 |

- With the OS keeping its LRU-like eviction order, admission recovers 9% of the SSD reads (about
  1.3 ms per token, 3%). The gap to the optimum is in eviction order.
- Owning the cache (for example in purgeable memory, which the kernel discards instead of swapping)
  would allow any eviction order, but eviction learned from these history features was worse than
  LRU: the access pattern is dominated by recency, which LRU already uses, and the optimum's
  advantage comes from knowing the future, which the history does not carry.
- Not caching read-ahead misses cut demand SSD reads by 19% but raised read-ahead reads by 66%,
  because wrong guesses were read again. Measured in the engine (a temporary build reading
  read-ahead through a second, `F_NOCACHE` descriptor; 512 tokens after `short.json`, `long.json`
  and `short.json` again per worker, two interleaved rounds): generation was 2–3% slower in steady
  state (26.49 / 26.50 against 27.35 / 27.41 tok/s after the second short prompt, 24.78 / 24.71
  against 25.29 / 25.32 after the long one) and read 32–45% more from the SSD; outputs identical.
  Not kept.

## What a GB of file cache is worth

Routed experts take 40 GB on disk and the file cache holds roughly 14–17 GB of them, so any memory
the worker or other programs hold comes out of the expert cache. Expert use is spread out: in the
66,400-token corpus, the 1,000 most-used experts (1.6 GB) served 24% of expert uses, the top 7,000
(11.4 GB) 70% and the top 15,000 (24.4 GB) 92%. Only 2 of the 24,576 experts served more than half
of the tokens and 451 more than 10%. The hot set also moves with the content: the 7,000 experts
most used on code covered 44% of the uses on other text (that text's own top 7,000: 79%), and the
corpus's top 7,000 covered 47% of the two benchmark generations. A fixed pinned set would do worse
than the recency-ordered file cache.

To measure the value of cache memory, a helper (`/Volumes/Data/moeshift-scratch/memval/balloon.c`)
filled 0, 2, 4 or 6 GB with random data and locked it (`mlock`, so it was neither compressed nor
swapped), and a fresh server then ran `short-ie.json`, `long-ie.json`, `short-ie.json`,
`long-ie.json` (512 tokens each, `config-source.json` settings without speculation). Two
interleaved rounds; swap use did not change. Generation of the second pair, both rounds, SSD
reads per request (`iostat`, prefill included) and demand misses per generated token:

| Memory locked | Second short request | Second long request | SSD per request (short / long) | Demand misses (short / long) |
|---:|---:|---:|---:|---:|
| 0 GB | 25.42 / 25.38 tok/s | 22.56 / 23.36 tok/s | 34 / 59 GB | 165 / 187 |
| 2 GB | 24.45 / 24.00 (−4.6%) | 21.88 / 21.36 (−5.8%) | 43 / 76 GB | 169 / 193 |
| 4 GB | 23.10 / 23.16 (−8.9%) | 20.72 / 21.10 (−8.9%) | 54 / 87 GB | 173 / 197 |
| 6 GB | 21.56 / 21.61 (−15%) | 19.66 / 19.73 (−14%) | 71 / 103 GB | 180 / 202 |

The first pair of requests showed the same trend (after the short prompt 25.46, 24.77, 23.62 and
22.45 tok/s; after the long one 24.00, 22.27, 21.18 and 19.68). Each GB taken from the file cache
cost about 2.3–2.9% of generation (about 1 ms per token), close to linearly up to 6 GB. Demand
misses hardly changed (+6–9%); the extra misses went to the SSD instead of the file cache. By the
same slope, a GB the worker or other programs give back is worth roughly +2.5%, until the cache
holds most of what a session uses.

This prices the memory the configuration spends: the prompt cache (`cache-ram`, up to 4,096 MiB:
about 10% of generation when full; it held 640 MiB in these runs), the decode windows (12 cache
plus 8 read-ahead slots per layer, 1,027 MiB, mostly second copies of file-cache pages), and
other applications.

**Where the worker's footprint goes** (`footprint` after one `short-ie.json` request, repository
build, `config-source.json` settings): 6,663 MB, of which 5,808 MB untagged (Metal buffers: the
non-expert weights, the decode windows, caches) and 679 MB of large `malloc` blocks. `vmmap` counted
433 MB of those as freed but still dirty ("Malloc Large (empty)"): memory the process no longer
uses but macOS still charges to it, as the prompt cache's freed entries were before M2 (fixed in
M4 below). With
`--ctx-size` 16,384, 32,768 or 65,536 the footprint was identical to the megabyte: the context's
KV cache costs memory only as it fills, so a smaller context saves nothing. The token embedding
is 260 MiB (IQ3_S) and the output layer 417 MiB (Q5_K); the bf16 hyper-connection projections
total 1.2 GB (B2).

## Freed load and graph buffers (M4, uncommitted)

Two large host buffers stayed charged to the worker for its whole life:

- **The model loader's read buffer.** With `--no-mmap`, `load_all_data` reads each tensor into one
  buffer and copies it to the GPU; the buffer grows to the largest tensor, Flash Next's output layer
  (417 MiB). `malloc_history` on the freed region traced it there. macOS's allocator kept the freed
  block as dirty memory: neither `shrink_to_fit` nor `malloc_zone_pressure_relief` returned it, and
  under memory pressure it was compressed rather than released. The buffer now comes from
  `ggml_aligned_malloc` (`vm_allocate` on macOS), whose pages go back to the system when freed.
- **The graph metadata buffers.** Each `llm_graph_result` sizes its metadata for `max_nodes`
  (245,760 for 6,144-token batches: about 100 MB), which a `std::vector` zero-filled, while Flash
  Next's graphs have 8,274 nodes for one token and 10,449 for 6,144. The buffer also comes from
  `ggml_aligned_malloc` now, so only the pages a graph uses are backed.

Footprint (`footprint`, repository build, `config-source.json` settings without speculation):

| Build | After load | After `short-ie.json` and `long-ie.json` | Freed but charged after load |
|---|---:|---:|---:|
| Before | 4,955 MB | 7,461 MB | 436 MB |
| M4 | 4,356 MB (−599 MB) | 6,862 MB (−599 MB) | 19 MB |

Three interleaved rounds, fresh server each, short, long, short and long requests (512 tokens each);
medians, SSD reads per request; every response identical:

| Build | First short / long | Second short / long | SSD per request (second pair) |
|---|---:|---:|---:|
| Before | 24.66 / 22.27 tok/s | 24.37 / 21.81 tok/s | 47 / 78 GB |
| M4 | 24.71 / 22.41 tok/s | 24.61 / 22.11 tok/s (+1.0% / +1.4%) | 43 / 73 GB |

About +1% once the file cache has settled, with 4–8% fewer SSD reads, as the measured price of a
GB of file cache predicts (0.6 GB, about +1.5%); the ranges overlap. Also tried and dropped: turning
off `--fit` for LES (it builds and frees a trial context; startup 0.75 s faster, footprint unchanged)
and a `max_nodes` bound for Flash Next's architecture without the per-token term (smaller virtual
reservations, footprint unchanged).

## Q8_0 hyper-connection weights (B2, `hc-type`, uncommitted)

A dev-only loader patch (worktree in `/Volumes/Data/moeshift-scratch/b2`, not in the repository)
converts chosen bf16 tensors to another type in place after loading. A smaller type fits the bf16
allocation, so that type's own Metal kernels run, with no GGUF copy. Quality was measured against
today's bf16 build with `llama-perplexity --kl-divergence`:

- **Text:** 11 chunks of 4,096 tokens (22.5k scored tokens): the model's own earlier answers in
  chat format, this repository's documentation, `src/llama-moe-slot.cpp` and Python tools.
- **Batches:** prompt batches of 2,048 rows, so this measures the matrix-matrix kernels.
- **Reference perplexity:** 2.9495.

| Change (against bf16) | Mean KLD | 99th pct KLD | Same top token | Perplexity |
|---|---:|---:|---:|---:|
| None, batches of 1,024 rows (identical arithmetic) | 0.00000 | 0.00004 | 99.996% | ±0 |
| hc weights stored as f16 (identical values, other kernel) | 0.0044 | 0.051 | 97.7% | +0.25% |
| hc and router as f16 | 0.0041 | 0.050 | 97.9% | +0.05% |
| **Q8_0 hc** (`hc_{attn,ffn}_{down,up}`, 1.2 GB) | 0.0104 | 0.116 | 96.6% | +0.04% |
| Q8_0 `hc_attn_down` only | 0.0052 | 0.057 | 97.6% | +0.06% |
| Q8_0 `hc_attn_up` only | 0.0086 | 0.102 | 96.9% | +0.24% |
| Q8_0 `hc_ffn_down` only | 0.0063 | 0.071 | 97.4% | +0.21% |
| Q8_0 `hc_ffn_up` only | 0.0095 | 0.107 | 96.5% | +0.42% |
| Q8_0 router (`ffn_gate_inp`, 120 MB) | 0.0074 | 0.085 | 97.1% | −0.03% |
| Q8_0 hc and router (B2 as planned) | 0.0147 | 0.207 | 96.2% | **+1.0% ± 0.14** |
| Q4_0 hc (scale check) | 0.126 | 1.31 | 88.5% | +2.7% |

This model's next token is sensitive to arithmetic. Storing the same values as f16 changes only the
kernel's rounding, and that alone flips 2.3% of top tokens, at about 40% of Q8_0's KLD. Each quarter
of the hc weights alone does more than a quarter of the damage, since it flips the same near-ties.
The rounding error itself is ordinary for Q8_0 (0.6–0.7% RMS in every tensor, block max/RMS about
2.5, no outlier blocks), so another 8-bit layout would not help. KLD against bf16 therefore mixes
damage with near-tie flips. Perplexity, scored against the actual text, separates them:

- **Q8_0 hc:** within the f16 control's range.
- **Q8_0 router:** saves only 60 MB per token for a KLD 1.7 times the control's.
- **Both together:** +1.0% perplexity, the one change clearly above noise.

Greedy generations on the 12 server prompts (the N5 set plus the N2 tasks), with the decode
kernels:

- **Identical to bf16:** 4 for Q8_0 hc, 3 for the f16 control.
- **Divergence:** the rest diverge at similar points for both.
- **The two scoreable tool-call edits:**
  - Without thinking, all three builds produced the same edit.
  - With thinking, only the Q8_0 build finished within the 1,500-token budget, with the intended
    edit; the bf16 and f16 builds were still thinking when the budget ran out. That is one sample,
    not evidence for Q8_0.

### The option

`hc-type: q8_0` in a profile (`--hc-type q8_0`; default `bf16`):

- **At creation:** the loader creates the four hyper-connection projections of each layer as
  Q8_0, so the bf16 size is never allocated. The router stays bf16.
- **While loading:** it converts them from the GGUF's bf16, 256k values at a time.
- **Constraint:** requires `no-mmap`; the router rejects other values and `q8_0` without it.

Its quality matches the probe exactly (perplexity 2.950766, mean KLD 0.010375). Measured with the
repository build and `config-source.json` settings without speculation:

| | `bf16` | `q8_0` |
|---|---:|---:|
| Footprint after load | 4,356 MB | 3,792 MB (−564 MB) |
| Load time | 1.74 s | 1.96 s |

Generation speed, on the 12 prompts of the generation test, two interleaved rounds per build with a
fresh server each:

| Prompts where both builds wrote identical text | `bf16` (two rounds) | `q8_0` (two rounds) | Change of means |
|---|---|---|---:|
| config | 22.80 / 21.57 tok/s | 22.82 / 23.52 tok/s | +4.4% |
| edit | 20.34 / 20.07 | 21.09 / 21.05 | +4.3% |
| rewrite-fn | 18.82 / 17.86 | 19.04 / 19.68 | +5.6% |
| whole-file | 18.02 / 17.73 | 18.44 / 19.05 | +4.9% |

- **Identical text:** every `q8_0` run was faster than every `bf16` run. This is the like-for-like
  comparison.
- **All 12 prompts:** 20.95 → 21.51 tok/s (+2.7%), but there the two builds decode different text.
- **Short and long A/B:** three rounds of M4's short and long prompts gave medians of +4.6% and
  +3.1% for the first pair and +2.2% and +0.4% for the second. One `q8_0` round read about 30% more
  from the SSD on the same prompts and was slow throughout.
- **Repeatability:** every build's 12 outputs are identical between fresh servers, apart from
  the random ids of tool calls. The default `bf16` build's outputs are also identical to those of the
  build without the option.

With your `config-source.json` speculation (`ngram-simple`, `spec-draft-n-max: 1`,
`LLAMA_MOE_DECODE_ROWS: 2`), same 12 prompts:

| | `bf16` | `q8_0` |
|---|---:|---:|
| All 12 prompts (two rounds each) | 24.39 / 24.24 tok/s | 25.48 / 25.58 tok/s (+5%) |
| The 4 prompts whose output stayed identical | 24.64 tok/s | 25.94 tok/s (+5.3%) |
| Drafts accepted | 1,332 of 1,353 | 1,449 of 1,474 |

- **Gain from speculation:** +16% for `bf16`, from 20.95 tok/s without it.
- **Identical-text prompts:** config +11.8% (one `bf16` round was low), edit +4.1%, rewrite-fn
  +4.1%, whole-file +2.7%.
- **Excluded round:** a first `q8_0` round, 20.82 tok/s, was left out. Its last four prompts ran
  at half speed (`story` 12.6 against 28.5 tok/s) with identical output, and a rerun matched the
  other round.
- **Output changes from speculation:** speculation changes the output of 2 of the 12 prompts,
  `edit-think` and `explain-quote`, with both builds and identically in every round. The 2-row
  checks round differently from 1-row steps. N2's identical-output measurement did not include
  these two prompts.

**Not tested:**
- **Ground truth.** Task accuracy on a benchmark large enough to resolve 1%.
- **Perplexity bias.** A quarter of the scored text is the bf16 model's own output, which favours
  bf16 in perplexity.

## 2-row checks for K-quant weights (K1)

A 2-row speculative check (`LLAMA_MOE_DECODE_ROWS: 2`) runs every dense mat-vec with two input
rows. ggml-metal's small-batch kernel (`mul_mv_ext`), which reads each weight row once for several
input rows, was used for bf16, Q8_0 and IQ4_NL from 2 rows but for the K-quants only from 4. Most
of this model's quantized dense weights are K-quants (`gguf-py` dump of the first shard):

| Weights | Types | Small-batch kernel at 2 rows, before |
|---|---|---|
| Output head (417 MiB) | Q5_K | No (from 4 rows) |
| `attn_qkv`, `attn_gate`, `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ssm_out`, some shared-expert gate and up | Q4_K, Q5_K, Q6_K | No (from 4 rows) |
| The other attention and `ssm_out` tensors, most shared-expert gate and up | IQ4_XS, IQ3_S | No kernel |
| Shared-expert down | IQ4_NL | Yes |
| Hyper-connection projections (Q8_0 with `hc-type: q8_0`), router, PLE key, indexer | Q8_0, BF16 | Yes |

**Microbenchmark** (`/Volumes/Data/moeshift-scratch/kq/kq_bench.cpp`): Gaussian weights quantized
with ggml's quantizer, 1–3 input rows, enough copies of each matrix (up to 768 MB) that the
weights stream from memory; 200 iterations, both settings in alternation twice. 2-row time against
1 row:

| Shapes (k × n) | Plain kernel (before) | Small-batch kernel |
|---|---:|---:|
| Q4_K, 2560 × 6144–12288 | 1.22–1.28× | 1.13–1.15× |
| Q5_K, 2560 × 6144–12288 | 1.63–1.65× | 1.16–1.18× |
| Q6_K, all shapes | 1.08–1.19× | 1.01–1.03× |
| Q5_K output head, 2560 × 248320 | 1.55× (1,874 µs) | 1.15× (1,377 µs) |
| 512- and 640-row matrices | within noise (4–6 µs either way) | |

The plain kernels already do much better than twice the 1-row time, apparently because the second
row's pass over the weights partly hits the GPU caches. Weighted by the model's tensor counts, the
small-batch kernel saves about 1.0 ms per 2-row check (Q5_K 0.64 ms, most of it the output head;
Q6_K 0.34 ms; Q4_K 0.06 ms), against about 7 ms if the second row had cost a full pass. For the
untouched types, at 2 rows: IQ3_S 1.74–1.86× (it does read the weights twice), IQ4_XS 1.19–1.26×,
IQ4_NL (small-batch kernel) 1.24–1.31×.

**Q4_K accuracy bug.** Against exact results (dequantized weights, double-precision dot products,
`kq_ref.cpp`), the Q4_K small-batch kernel was off by 4.5e-3 relative; the plain Q4_K kernel and the
Q5_K and Q6_K small-batch kernels by 1–3e-7. `dequantize_q4_K` divides the block scale for the high
nibbles by 16 in half precision (`xb->d / 16.h`); with typical weight scales the result is a half
subnormal and loses precision. `dequantize_q5_K` divides in float. The ggml CPU backend is no
reference here: it rounds the input to Q8_K for K-quant dot products (6–8e-3 for all three types).

**Changes** (`ggml-metal-ops.cpp`, `ggml-metal.metal`):

- Q4_K, Q5_K and Q6_K use the small-batch kernel from 2 rows. Q2_K and Q3_K keep 4 rows (not
  measured).
- The Q4_K small-batch kernels use `dequantize_q4_K_f`, a copy of `dequantize_q4_K` that divides in
  float: 1.6e-7 against exact results. This also changes batches of 4–8 tokens, which used the
  inaccurate kernel before.
- The prompt matrix kernels (`mul_mm`, `mul_mm_id`) and `get_rows` keep `dequantize_q4_K`, so prompt
  processing is unchanged. They have the same half-precision division; fixing it would change
  prompt results and greedy output, which needs its own quality comparison (not done).

**End to end** (`kq/e2e.sh`): `config-source.json` settings (2-row n-gram drafting, `hc-type:
q8_0`), the N2 tasks, a fresh server per round, in the order old binary, new with the old gate,
new with the 2-row gate, then the reverse:

| Task (2-row checks / tokens) | Old binary | Old gate | 2-row gate | 2-row against old gate |
|---|---:|---:|---:|---:|
| Whole file (550 / 1,200) | 26.25 tok/s | 26.49 | 26.86 | +1.4% |
| Config (373 / 778) | 28.92 | 29.11 | 29.46 | +1.2% |
| Rewrite a function (222 / 700) | 24.33 | 24.38 | 24.39 | 0% |
| Story (no drafts, control) | 25.86 | 25.72 | 25.74 | 0% |

- **Output:** every response in all six rounds was identical to the old binary's.
- **Gain:** +1.2–1.4% where most steps are checks, as predicted from the microbenchmark (about
  1 ms over 550 checks in a 45 s reply); nothing on the story, which makes no checks.
- **Only the first three rounds are used.** In the reverse rounds the machine ran 3–5% slower
  (story control 24.4–25.1 tok/s, prompts up to 45% slower), with Apple's on-device inference
  service and `aned` using about 38% CPU each for part of the time. Those rounds show the same
  direction (+2–8% for the 2-row gate against the old gate run after it), more than the change can
  explain. An earlier attempt was stopped when an iOS Simulator app took CPU and memory.
- **Memory:** none.

**Not tested:** IQ4_XS and IQ3_S small-batch kernels (about 0.5 ms more per check if they reach
about 1.05×); 3-row steps end to end; Q2_K and Q3_K.

## A 120k-token context (L1)

The model's GGUF declares a trained context of 262,144 tokens (`qwen4exp.context_length`); the
profiles' 65,536 is only the configured `ctx-size`, and nothing in the router or worker caps it.
Only 12 of the 48 layers have full attention (2 KV heads of 256 dimensions), so the KV cache takes
24 KiB per token at `f16` and 12.8 KiB at `q8_0`, and an indexer chooses 2,048 positions per query
for those layers.

**Test** (`/Volumes/Data/moeshift-scratch/ctx128/`): `config-source.json` settings with
`--ctx-size 131072`, one request of 119,928 prompt tokens built from 283 repository source and
Markdown files (`build_prompt.py`, sized with `llama-tokenize`), with three access codes planted at
2%, 50% and 95% depth. The request asks for the codes and a 200-word description of the first file
(300 tokens at most). Memory sampled every 200 ms with r2m's `memwatch` (footprint, and system-wide
memory above the level before the worker started); a watchdog would have stopped the worker if swap
grew by more than 1 GB. Run before the indexer-cache fix below, so the `f16` run had an `f16`
indexer cache and the `q8_0` run a `q8_0` one.

| KV cache | Prompt | Generation | Peak footprint | Peak system-wide | Swap | Recall |
|---|---:|---:|---:|---:|---:|---|
| `f16` | 353.4 s (339 tok/s) | 14.23 tok/s (279 tokens) | 17.1 GB | 19.2 GB (15–16.6 sustained) | unchanged | 3 of 3 |
| `q8_0` | 385.2 s (311 tok/s) | 14.31 tok/s (295 tokens) | 15.4 GB | 19.0 GB (17–18.5 sustained) | unchanged | 3 of 3 |

- **It fits:** about 19 GB at the peak, no swap growth, every code recalled, and both descriptions
  of the first file (`src/llama-adapter.cpp`) accurate. The two texts diverge after a few sentences,
  as expected when the cache type changes.
- **Prompt:** about 6 minutes, faster than the extrapolation from the 6k and 42k prompts (about 8).
  Every 6,144-token batch reads all experts, and attention and the indexer cost more per batch as
  the context grows.
- **Memory over the `f16` prompt:** footprint 9 → 17.1 GB, about 67 KB per token, more than the KV
  cache's 24 KiB, so other buffers grow with the context too (not broken down; the attention mask
  and per-batch scratch scale with context times batch). The system-wide peak of 19.2 GB was a
  single spike at about 150 s. While generating: 9.8 GB footprint.
- **Generation:** about 70 ms per token, against about 40 ms after a short prompt. The KV cache's
  2.75 GiB taken from the file cache accounts for about 3 ms (2.5% per GB, see above), the indexer
  key pooling for about 8 ms (I1 measured 4.5 ms at 64k, growing linearly); about 19 ms is not
  accounted for, probably the indexer scoring every cached key per token. Not traced. Traced and
  mostly fixed later: see [long-context generation (L2)](#long-context-generation-l2).
- **Prompt cache:** an entry for this conversation is about 3.3 GB at `f16`, close to the whole
  4,096 MiB `cache-ram`, so it would push out every other saved conversation.

For comparison, anemll-forge 0.2.1 runs a dense Qwen3.8-27B to 192K on the Neural Engine of an M6
with 32 GB: INT8 KV of 32.25 KiB per token, fixed context entries (the move from 128K to 192K holds
both caches, +4.3 GB), and attention tiled above 64K because the ANE compiler limits one tensor
axis to 65,536 (the reason its release 0.2 stopped at 64K). It reports 95 tok/s prefill and
25.1 tok/s decode at 128K, and calls 128K the comfortable maximum on 32 GB.

**Not tested:** `ctx-size` above 131,072; prompts of 120k tokens through the router with the prompt
cache filling; a GPU trace of generation at long context.

## KV cache type and the indexer cache (X1)

The notes measured `f16` KV 3.2% faster than `q8_0` on prompts (see "KV cache type and GPU
keepalive") without comparing quality. Compared now with `llama-perplexity --kl-divergence` against
an `f16` reference (`/Volumes/Data/moeshift-scratch/kvq/kld.sh`), `hc-type` bf16 in all runs:

- **4k context:** B2's corpus and reference logits (11 chunks of 4,096, 22.5k scored tokens).
- **32k context:** the first 250,000 characters of L1's prompt (repository sources and docs,
  74,517 tokens), 2 chunks of 32,768, 32.8k scored tokens; a new `f16` reference.

| 4k context | Mean KLD | 99th pct KLD | Same top token | Perplexity |
|---|---:|---:|---:|---:|
| `f16` again (control) | 0.00000 | 0.00004 | 99.996% | ±0 |
| `q8_0` | 0.0058 | 0.071 | 97.4% | −0.01% ± 0.09 |
| `f16`, indexer cache `bf16` | 0.0027 | 0.034 | 98.3% | +0.12% ± 0.06 |

At 4k, `q8_0` flips about as many near-tied tokens as other arithmetic changes on this model (B2:
storing the hc weights as f16 gave 0.0044 and 97.7%), with perplexity unchanged.

At 32k the comparison looked different: `q8_0` gave a mean KLD of 0.124 with a median of 0.0001
(most tokens agree, a few disagree completely: 99.9th percentile 12.6), and perplexity **2.5%
lower** than the `f16` reference. bf16 KV, which has f32's exponent range, behaved like `q8_0`.
The `f16` control was identical to the reference (KLD 0, 100% same top token), so the runs are
deterministic. The indexer stores its keys in a cache of its own, created with `cache-type-k`
(`llama-memory-hybrid-idx.cpp`), and caches them raw, before their RMS norm; a temporary switch
gave that cache its own type:

| 32k context, against `f16` / `f16` | Attention cache | Indexer cache | Perplexity | Mean KLD | 99th pct KLD | Same top token |
|---|---|---|---:|---:|---:|---:|
| Reference (the old `f16` setting) | `f16` | `f16` | 4.331 | — | — | — |
| Control | `f16` | `f16` | 4.331 (±0) | 0.0000 | 0.00003 | 100% |
| | `bf16` | `f16` | 4.445 (+2.9%) | 0.131 | 3.55 | 94.0% |
| The old `q8_0` setting | `q8_0` | `q8_0` | 4.214 (−2.5%) | 0.124 | 3.17 | 93.7% |
| | `bf16` | `bf16` | 4.180 (−3.3%) | 0.135 | 3.69 | 93.7% |
| | `f16` | `bf16` | 4.142 (−4.1%) | 0.097 | 1.93 | 94.0% |
| | `f16` | `f32` | 4.206 (−2.7%) | 0.082 | 1.72 | 95.1% |
| **Now with `q8_0`** | `q8_0` | `bf16` | 4.273 (−1.1%) | 0.122 | 3.26 | 93.7% |

Perplexity standard errors are about ±0.5–0.6% against the reference.

- **An `f16` indexer cache is the fault.** The two runs with one are the worst; every other
  indexer type improves perplexity by 1–4%, whatever the attention cache uses.
- **Range, not precision:** bf16 has fewer mantissa bits than f16 but f32's exponent range, and it
  fixes the problem. The raw indexer keys most likely exceed f16's largest value (65,504) on some
  tokens, which spoils the block scores, so those queries attend to the wrong positions. Not
  confirmed by dumping the keys.
- **Why only at long context:** the indexer selects 2,048 positions per query, so below about 2k
  tokens of context it selects everything, and at 4k about half; the longer the context, the more
  its choices decide the result.
- **Among the fixes the differences are within noise.** The spread is about 1.5% and the order is
  not consistent (f32, the exact type, scored worse than bf16) on two chunks of one kind of text.
  The runs with an `f16` attention cache average about 4.17, those with `q8_0` about 4.24, which
  looked like a cost of up to 1.5% for `q8_0`. Every row here is measured against the old `f16`
  setting, whose indexer was broken, so the KLD tails are distances from a broken reference. X2
  below compares against the fixed `f16` setting and finds no cost.
- **The chunk-1 perplexity is high** (about 14 for the `f16` reference, 13 for the others): the
  first half of the 32k text is harder for every setting; chunk 2 is about 1.3.

**Changes:**

- `llama-memory-hybrid-idx.cpp` creates the indexer cache as bf16 whatever `cache-type-k` says.
  bf16 takes the same memory as f16 (3 KiB per token over the 12 layers). **Changes output** above
  a few thousand tokens of context with `f16` KV (the fix) and slightly with `q8_0` KV.
- `config-source.json` uses `q8_0` KV again, for the memory: 12.8 against 24 KiB per token (about
  0.7 GiB less at 64k tokens, 1.4 GiB at 120k), worth about 2.5% of generation per GB returned to
  the file cache at long context. Costs: prompts 3.2% slower at 6k tokens and 9% at 120k (the
  flash-attention kernel unpacks every key and value); generation the same, since the sparse
  attention reads only about 48 MiB of KV per token at `f16`. The shipped Flash Next profile in
  `configs/moeshift.json` already used `q8_0`.

**Verified** with the fixed build (no override): `q8_0` at 32k reproduced the `q8_0` / bf16 row
exactly (perplexity 4.273238, mean KLD 0.121807). The N2 tasks through a server with
`config-source.json` and `q8_0` KV (one round): whole file 27.43, config 30.10, rewrite a function
27.02, story 27.36 tok/s, with the same drafts accepted as before (548 of 550, 365 of 373, 220 of
222). The three drafting tasks wrote the same text as K1's `f16` run, the story differed. Every
task, the story included, ran 2–11% faster than K1's `f16` round, although generation should not
depend on the KV type; a single round at another time, so this is not a speed result.

**Prose at 32k (X2).** Wilkie Collins's *The Moonstone* from Project Gutenberg (header and licence
removed, 266,404 tokens), 8 chunks of 32,768 tokens, 131k scored tokens; the reference is the
fixed build with `f16` KV (attention cache `f16`, indexer cache bf16), perplexity 6.4376:

| 32k context, prose | Perplexity | Mean KLD | Median KLD | 99.9th pct KLD | Same top token |
|---|---:|---:|---:|---:|---:|
| `q8_0` against `f16` (both with the bf16 indexer cache) | 6.4331 (**−0.05% ± 0.04**) | 0.0064 | 0.0026 | 0.15 | 96.4% |

- **No cost at 32k on prose:** perplexity is unchanged to within 0.1%, every chunk within ±0.15%
  with no trend from the first to the eighth.
- **No heavy tail:** the KLD is the size of `q8_0` at 4k (0.0058), rounding that flips near-ties.
- So `q8_0` KV saves its memory for nothing measurable in quality. The prompt cost (3–9%) remains.
- The reference logits are 65 GB, in `/Volumes/Data/moeshift-scratch/x2/` (deletable).

**Not tested:** a chat transcript or code against the fixed `f16` setting; contexts of 64–128k in this comparison;
the key magnitudes themselves; a speed comparison of the bf16 indexer cache (it takes part in a
`get_rows` and a mat-vec per layer, so no measurable change is expected).

## Long-context generation (L2)

L1 found generation at 120k tokens of context taking about 70 ms per token, and most of the growth
unexplained. Uncommitted changes, identical output.

**Method** (`/Volumes/Data/moeshift-scratch/l2/`): one worker per run with `config-source.json`
settings, `--ctx-size 131072`, no speculative decoding (one token per step) and no prompt cache;
prompts of 8k, 32k, 64k and 120k tokens built from repository files as in L1, plus a 29-token
story prompt; 320 tokens generated with `ignore_eos`, timed from their arrival over tokens 40–320.
Runs of the same build agreed within 0.5 ms per token. A first sweep that also ran macOS `sample`
during generation was about 25% slower throughout and is not used for timings; its profile pointed
at the right function, but at the wrong cause.

**Where the time went.** Temporary timers (the `LES_BLOCK_DIAG` phase log, a timer per graph input,
and the GPU time of every Metal command buffer), per decode step:

| Per step, PLE fix not yet in | 8k | 64k |
|---|---:|---:|
| Input setup on the CPU, before the GPU starts | 1.2 ms | 6.4 ms |
| of which the per-layer embedding (PLE) input | 1.0 ms | 5.5–7.1 ms |
| GPU busy (command buffers) | 25.5 ms | 33.0 ms |
| Expert misses / bytes read | 172 / 291 MiB | 164 / 275 MiB |

- **PLE input.** The n-gram embedding needs the two tokens before each new one.
  `llama_kv_cache::get_prev_tokens` found them by visiting every used cell and testing each of
  its 256 sequence bits, so its cost grew with the context.
- **GPU.** A microbenchmark of one token's attention-layer work (`l2_bench.cpp`, 12 layers, `q8_0`
  KV, random keys and selections), in ms per token:

  | Cells | Pool keys | Score, expand | Top-k | Mask | Attention | Total |
  |---:|---:|---:|---:|---:|---:|---:|
  | 8k | 0.96 | 0.13 | 0.51 | 0.06 | 2.11 | 3.8 |
  | 32k | 1.88 | 0.72 | 0.90 | 0.18 | 4.40 | 8.1 |
  | 64k | 4.04 | 1.11 | 1.73 | 0.00 | 5.10 | 12.0 |
  | 128k | 8.42 | 2.11 | 3.68 | 0.23 | 5.35 | 19.8 |

  The indexer pooled, normed and rotated every cached block's key on every token (I1, see
  C1c above). Attention skips 32-cell chunks with no selected cell, but scattered selections touch
  many chunks: 5.5 ms at 128k for random selections, 1.35 ms if the 2,051 cells were adjacent.
- **Not the cause:** expert reads per step did not change with context.

**Changes:**

- **PLE window** (`src/llama-kv-cache.cpp`, `src/llama-kv-cells.h`): `get_prev_tokens` visits only
  the cells with positions in the window it looks up, `[p_min − 2, p_max]`. The cells before it are
  visited only when a lookup finds no cell in the window (M-RoPE position gaps). `for_each_token_in`
  scans the cell array instead of walking the set of used cells, in the same order. PLE input:
  0.33–0.49 ms per step at both 8k and 64k.
- **Pooled indexer keys (I1)** (`src/llama-memory-hybrid-idx.*`, `src/models/qwen4exp.cpp`): an F32
  row per block of 4 positions and attention layer, as the scores read it (pooled, normed, rotated).
  Each ubatch pools only the blocks its tokens can touch (one per decode step, two for a 2-row
  check, 1,537 for a 6,144-token prompt batch) and writes them with `set_rows`; the scores read
  every row from the cache. Every op of the pooling chain works row by row, so a block pools to
  the same bits alone as among all of them.
  - Rows are indexed by position, so the cache exists only with one sequence (the worker's case).
  - Rollbacks need nothing: an incomplete block is already masked, and its row is recomputed when
    its last cell is written.
  - State loads, sequence copies, shifts and clears mark the rows stale; the next graph pools every
    block once. Rows are not saved with the state.
  - The worker's KV cache grows (`LLAMA_KV_GROW`, starting at 2,048 cells). The rows follow the
    indexer cache's size: a growth reallocates them (no copy) and marks them stale. A first version
    sized them for a fixed cache only, and was silently off in the worker.
  - Memory: 512 bytes per block per attention layer, 1.5 KiB per token of context: 96 MiB at 64k
    cells, 192 MiB at 128k.

**Result.** Interleaved A/B, two rounds in opposite orders (HEAD, the PLE fix alone, both), steady
ms per generated token, means of the two rounds (each pair within 0.5 ms):

| Context | HEAD | PLE window | PLE window + I1 | Change |
|---|---:|---:|---:|---:|
| 29 tokens | 35.8 | 35.9 | 35.7 | none |
| 8k | 41.2 | 39.8 | 39.6 | −4% |
| 32k | 47.2 | 45.1 | 42.1 | −11% (20.4 → 22.9 tok/s) |
| 64k | 54.5 | 52.3 | 45.4 | −17% (17.7 → 21.2 tok/s) |
| 120k (one run each) | 66.6 | | 51.4 | −23% (14.4 → 18.6 tok/s) |

- Generation now loses about 6 ms per token from 8k to 64k (39.6 → 45.4 ms), against 13 ms
  before. With I1 the GPU time per step at 64k fell from 33.0 to 28.6 ms.
- The PLE fix moved part of its time into waiting, since the next step's expert reads take as long
  as before: input setup fell by 5.7 ms at 64k, the step by 2.2 ms.
- Prompts: unchanged (64k: 152.8 s against 151.7 s).
- Peak footprint: +100 MB at 64k (12.21 → 12.31 GB), +230 MB at 120k (15.42 → 15.65 GB).

**Verification:** output identical to HEAD in every request:

- all 5 prompts above, in both rounds and at 120k;
- the 9 n-gram tasks with 2-row checks (`config-source.json` settings, prompt cache on), with the
  same drafts accepted;
- a prompt-cache restore (an 8k request, a short one, the 8k request again with 8,015 tokens
  restored, then a second turn on it);
- perplexity at 32k context (X1's `q8_0` run): 4.273238, as before.

On the 9 tasks (1–5k tokens of context) the two builds were level: whichever ran second was 0.2–8%
slower, in both orders.

**What is left** (L3 in `TODO.md`): from 8k to 64k the GPU still takes 3.4 ms more per step,
mostly attention over scattered chunks, the top-k over every cell and the scoring. Attending only
the gathered 2,051 cells and selecting 512 blocks instead of 2,051 cells would remove most of it,
but both change rounding or the boundary cells, so they need a quality comparison.

**Not tested:** more than one sequence (the pooled rows are then off); context shifts; images,
whose M-RoPE gaps take the PLE fallback scan.

## Tool calling

MoEShift forwards `/v1/chat/completions` request bodies unchanged to the
llama.cpp server worker. That worker uses Jinja chat templates by default, and
the Flash Next template supports tools. A test with one `get_weather` function
(`temperature: 0`, thinking off) found:

- The model returned a `tool_calls` entry with arguments
  `{"city":"Paris","unit":"celsius"}` and `finish_reason: tool_calls`.
- After a `tool` message with the result, it answered "The current weather in
  Paris is **14°C** with **light rain**." (`finish_reason: stop`).
- With `stream: true`, the function name and arguments arrived in streamed
  `tool_calls` deltas, ending with `finish_reason: tool_calls`.

Multiple tools, parallel calls and `tool_choice` were not tested.

## Not tested

- Prompt lengths other than 313–6,060 and 12,121 tokens (see the P1 and P2
  section). Prompts under 3,072 tokens fit in one batch either way; prompts over
  6,144 tokens still split, into half as many batches.
- Long conversations or contexts near the 65,536-token limit (a single 120k-token
  prompt was measured later, see [L1](#a-120k-token-context-l1)).
- Output agreement between the two settings.
- The Homebrew build (the source build at the same commit was used).
