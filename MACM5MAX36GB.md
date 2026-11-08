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
| `args.cache-type-k`, `args.cache-type-v` | **`f16`** | `q8_0` | **Measured.** Prompt processing was 3.2% faster; generation did not change. The KV cache uses 24 KiB per token instead of about 12.8 KiB: about 70 MiB more for a 6,000-token prompt, and about 0.7 GiB more at the full 65,536-token context. Output differs slightly from `q8_0`; quality was not compared. |
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
  compared.
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
  with identical output and about 4 MB more memory.
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
would therefore slow generation.

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
- Long conversations or contexts near the 65,536-token limit.
- Output agreement between the two settings.
- The Homebrew build (the source build at the same commit was used).
