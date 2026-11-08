# Adaptive prefill

MoEShift uses different expert-memory budgets for reading a prompt (prefill)
and generating a reply (decode). Both [supported architectures](../architecture.md#supported-llm-architectures)
use this path on Apple Silicon with Metal. Inference parallelism is fixed to one.

## Configuration

Edit the model profile in `~/.config/moeshift/config.json`:

| Profile entry | Default | Effect |
|---|---:|---|
| `args.prefill-max-tokens` | `3072` | Maximum logical batch and physical microbatch size. Smaller batches reduce peak working memory. |
| `args.moe-prefill-bank` | `512` / `256` | Shared expert slots used during prefill. A larger bank can reduce SSD reads, using more memory. |
| `args.moe-decode-bank` | `0` | Shared expert slots retained during decode; must be smaller than the prefill bank. Per-layer decode windows remain allocated separately. |
| `env.LLAMA_MOE_PREFILL_PIPELINE` | `1` | Overlap prefill expert reads with computation. |

The three `args` entries also accept CLI overrides with the corresponding
`--` names. Runtime tuning is read from JSON, not inherited shell exports.
Both shipped profiles use window 12, prefetch depth 3 and 24 protected
slots; see [protected decode cache](protected-decode-cache.md).

## Execution and phase transitions

The server fills each batch with available prompt tokens up to the configured
limit. Context checkpoints can split it further. The shared bank is reused across
layers: 512 slots cover Flash Next's 512 experts;
256 cover Qwen3.6's 256 experts.

Before switching phases, the runtime synchronizes GPU work and drains expert
readers. It invalidates cached mappings and sets the active Metal buffer ranges
for the new phase. During decode, the unused shared-bank ranges are released;
the per-layer decode windows and protected cache remain available. During
prefill, the decode windows are released the same way. Each switch replaces the
inactive ranges with fresh zero-fill memory, so their pages leave the process at
once while the virtual addresses stay reserved. (Marking them
`MADV_FREE_REUSABLE` instead let macOS compress them under file-cache pressure,
which counted them in the footprint again.) The next phase's ranges are
zero-filled on first use, and nothing relies on their earlier contents.

Expert execution uses the common Metal phase identifiers in
`ggml/src/ggml-metal/ggml-metal-impl.h`: `ALL` (0), `RESIDENT` (1), and
`MISSING` (2). The resident/missing path requires Metal placement. Predictions
schedule reads; the actual router determines which experts are computed.

Set `env.LLAMA_MOE_QUIET` to `0` for diagnostics such as `adaptive-batch`,
`adaptive phase=prefill/decode`, `phase-scratch`, and `protected-cache`.

The pipeline stages at most 1,200 MiB of expert data per layer; larger layers
use bounded slot reads instead. One reader thread reads layers in order into two
host staging buffers, so the next layer's read does not wait for the current
layer's copy into the shared bank. Both buffers hold the largest layer and are
released before decoding.

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
