# Adaptive prefill

MoEShift uses different expert-memory budgets for reading a prompt (prefill)
and generating a reply (decode). Both [supported architectures](../architecture.md#supported-llm-architectures)
use this path on Apple Silicon with Metal. Inference parallelism is fixed to one.

## Configuration

Edit the model profile in `~/.config/moeshift/config.json`:

| Profile entry | Default | Effect |
|---|---:|---|
| `args.prefill-max-tokens` | `3072` | Maximum logical batch and physical microbatch size. Smaller batches reduce peak working memory. |
| `args.moe-prefill-bank` | `512` / `256` | Shared expert slots used by prefill batches that do not use the pipeline (under 1,024 tokens, or with the pipeline off). A larger bank can reduce SSD reads, using more memory. |
| `args.moe-decode-bank` | `0` | Shared expert slots retained during decode; must be smaller than the prefill bank. Per-layer decode windows remain allocated separately. |
| `env.LLAMA_MOE_PREFILL_PIPELINE` | `1` | Read whole layers ahead into a staging buffer and compute batches of 1,024 tokens or more from it. `0` uses the shared bank for every batch. |

The three `args` entries also accept CLI overrides with the corresponding
`--` names. Runtime tuning is read from JSON, not inherited shell exports.
Both shipped profiles use window 12, prefetch depth 3 and 24 protected
slots; see [protected decode cache](protected-decode-cache.md).

## Execution and phase transitions

The server fills each batch with available prompt tokens up to the configured
limit. Context checkpoints split it further: the last 4 tokens of a prompt form
their own batch, so that the checkpoint taken before them can be restored by the
next request, whose re-rendered history differs from this prompt only in those
tokens. The worker keeps one checkpoint, takes it whatever the distance to the
previous one (`checkpoint_min_step` 0), and does not split the batch before the
latest user message, whose checkpoint the end checkpoint would replace anyway. Batches of 1,024 tokens or more use the prefill
pipeline (below). Smaller ones read only the experts they use into the shared
bank, which is reused across layers: 512 slots cover Flash Next's 512 experts;
256 cover Qwen3.6's 256 experts.

There are three expert memory phases: decode (per-layer windows), prefill (the
shared bank) and staged prefill (the pipeline's staging buffer). Before
switching, the runtime synchronizes GPU work and drains expert readers. It
invalidates cached mappings and sets the active Metal buffer ranges for the new
phase. During decode, the unused shared-bank ranges and the staging buffer are
released; the per-layer decode windows and protected cache remain available.
During prefill, the decode windows and the staging buffer are released the same
way, and during staged prefill the bank and the windows. Each switch replaces the
inactive ranges with fresh zero-fill memory, so their pages leave the process at
once while the virtual addresses stay reserved. (Marking them
`MADV_FREE_REUSABLE` instead let macOS compress them under file-cache pressure,
which counted them in the footprint again.) The next phase's ranges are
zero-filled on first use, and nothing relies on their earlier contents.

Expert execution uses the common Metal phase identifiers in
`ggml/src/ggml-metal/ggml-metal-impl.h`: `ALL` (0), `RESIDENT` (1), and
`MISSING` (2). The resident/missing path requires Metal placement. Predictions
schedule reads; the actual router determines which experts are computed.

Graph scratch (the compute buffers) follows the same phases. Before a decode
phase, `src/llama-context.cpp` releases the prefill compute buffers, so each
later prefill allocates only what its graph needs. The buffers reserved at load
or after a settings change are released the same way: they are sized for logits
on every row at full context, and Metal makes a whole buffer resident when it is
first used.

Pages of these buffers that only the GPU writes do not appear in the process
footprint (`phys_footprint`, `footprint`, `vmmap`), although they take physical
memory. To measure peak memory, use system-wide internal + wired + compressed
pages.

Set `env.LLAMA_MOE_QUIET` to `0` for diagnostics such as `adaptive-batch`,
`adaptive phase=prefill/decode`, `phase-scratch`, and `protected-cache`.

The pipeline reads whole layer packs (all experts of a layer, as stored in the
expert files) into the two halves of one Metal staging buffer, each the size of
the largest layer. One reader thread reads the layers in order, alternating
halves. The graph's expert matmuls read views of the half holding their layer,
with the pack stride between experts, so nothing is copied and the shared bank
is not used. A small CPU op after each layer's experts frees its half for the
read two layers ahead while the next layer's attention runs. Models whose layers
exceed 1,200 MiB use bounded slot reads into the bank instead.

Prefill reads bypass macOS's file cache (`F_NOCACHE`); decode reads go through
it. A prefill reads every expert once, so caching its reads would replace the
experts decode keeps cached, which serve most decode misses and part of the next
prefill. The worker sets `LLAMA_MOE_F_NOCACHE=auto`; the expert files switch at
each phase change.

## Code and verification

The server sets batch and bank budgets in `tools/server/server-les.cpp`.
`src/llama-moe-slot.cpp` handles phase transitions and expert readers;
`src/llama-context.cpp` coordinates the graph and phase scratch memory.

With tests enabled in the build:

```sh
cmake --build build --target moeshift test-les-active-ranges test-metal-moe-phases -j 8
ctest --test-dir build -R '^(test-les-active-ranges|test-metal-moe-phases)$' --output-on-failure
```

These checks cover Metal active-range shrink/restore and shared phase constants.
They are not end-to-end performance benchmarks; see the repository README for
measured inference results.
