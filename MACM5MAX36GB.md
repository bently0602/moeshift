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
  work during that time. An uncommitted change to `src/llama-ssd-rows.cpp`
  reads these pages up to 16 at a time. It made prompt processing 12.8% faster
  (10.29 s → 8.97 s), with identical output.
- An `f16` KV cache made prompt processing 3.2% faster at a cost of about
  0.7 GiB more memory at full context. Turning off the GPU keepalive made
  generation 22% slower.
- MoEShift wrote nothing measurable to disk while running: the internal SSD wrote
  54–92 MB in total during each 4-request run, for the whole system.
- Copying each layer's experts with 8 threads instead of one (uncommitted)
  saved another 0.37 s (4.1%).
- Combined, with `f16` KV, the long prompt takes 8.29 s, against 10.29 s on the
  original build: 19% faster. It took 15.47 s with the original 3,072-token
  batches.
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
9.96 s. Two uncommitted code changes bring it to 8.29 s (see
[Combined result](#combined-result)). Neither change adds a setting, and the
Homebrew build does not include them.

Memory on this 36 GB Mac: peak process footprint was 9.4 GB with this
configuration, and swap did not grow during inference. The main risk to SSD
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

### Parallel PLE reads (uncommitted change)

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

### Threaded expert copy (uncommitted change)

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

Restart MoEShift after changing the configuration.

## Combined result

User configuration (`prefill-max-tokens: 6144`, `f16` KV) with both
uncommitted changes, compared with the `q8_0` baseline on the original build.
Medians of three measured requests:

| Setup | Prompt time | Prompt throughput | Generation | Peak process memory |
|---|---:|---:|---:|---:|
| Original build, `q8_0` KV | 10.29 s | 589.0 tok/s | 14.71 tok/s | 9,211 MB |
| Parallel PLE reads + threaded copy, `f16` KV | **8.29 s** | **730.9 tok/s** | 15.29 tok/s | 9,362 MB |

Prompt processing was 19.4% faster (2.00 s less), and throughput 24% higher.
Before any of these changes, at `prefill-max-tokens: 3072`, the same prompt
took 15.47 s. All three measured responses matched the `f16` run on the
original build exactly.

During every benchmark run, the internal SSD wrote 18–139 MB in total, for
the whole system, and swap use did not grow. Swap grew between runs, while
Instruments recorded and exported traces.

## Not tested

- Prompts of other lengths. Prompts under 3,072 tokens fit in one batch either
  way; prompts over 6,144 tokens still split, into half as many batches.
- Long conversations or contexts near the 65,536-token limit.
- Output agreement between the two settings.
- The Homebrew build (the source build at the same commit was used).
