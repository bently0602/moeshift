# MoEShift configuration

The [standard template](moeshift.json) includes both tested model profiles:

| Model ID and alias | Prefill slots | Protected slots per layer |
|---|---:|---:|
| `qwen-3.8-flash-next-rco-gsq-iq3-xxs` | 512 | 0 |
| `qwen-3.6-35b-a3b-q6-xl` | 256 | 24 |

## Manual setup

1. Download the GGUF for your model, keeping all shards together. If you
   already prepared experts for that exact GGUF, skip to step 3.

   | Tested model | GGUF downloads |
   |---|---|
   | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF · IQ3_XXS | [Shard 1](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf?download=true) and [shard 2](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf?download=true) |
   | unsloth/Qwen3.6-35B-A3B-GGUF · UD-Q6_K_XL | [Single GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf?download=true) |

2. Prepare and verify the experts. For a Homebrew install, the example below
   assumes the Qwen3.6 GGUF is in Downloads. Adjust `MODEL` for your file
   (use the first shard for a split model) and choose a new or empty `EXPERTS`
   directory. Keep both the GGUF and the prepared files.

   ```sh
   MODEL="$HOME/Downloads/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"
   EXPERTS="$HOME/.moeshift/models/qwen-3.6-35b-a3b-q6-xl/experts"
   PREFIX="$(brew --prefix moeshift)"
   TOOL="$PREFIX/share/moeshift/tools/moe/experts.py"
   "$PREFIX/libexec/bin/python" "$TOOL" extract --model "$MODEL" --out-dir "$EXPERTS"
   "$PREFIX/libexec/bin/python" "$TOOL" verify --model "$MODEL" --experts "$EXPERTS"
   ```

3. Open the configuration file printed by the wizard. For the default or XDG location:

   ```sh
   open -e "${XDG_CONFIG_HOME:-$HOME/.config}/moeshift/config.json"
   ```

   Under `models`, find your model ID from the table above. In its `args`,
   replace `model` with the full GGUF path and `moe-experts` with the full path
   to the folder containing `manifest.json`. Use actual paths in JSON:
   `~` and `$HOME` are shell shortcuts and are not expanded there.
   Keep the existing alias and tuning values. You can remove profiles you do not use.

4. Save the file and run `moeshift`. Use the displayed API URL and the model
   ID in your client. The first inference request loads the model.

## Configuration reference

Lookup order: explicit `--config FILE`, `moeshift.json` in the current directory,
`$XDG_CONFIG_HOME/moeshift/config.json`
(or `~/.config/moeshift/config.json`). When none exists, an interactive launch opens the setup wizard.
Profile paths resolve relative to the configuration file.

CLI profile overrides apply to every configured model. The JSON profile is the
single source of runtime tuning: shell variables starting with `LLAMA_`, `GGML_`,
`LES_` and `MB_` are not inherited by inference workers. Parallelism is fixed to
1; external hosts and speculative-decoding options are rejected.

Both shipped LRFU profiles use a 12-slot decode window, prediction depth 3 and
24 protected slots per layer. Positive protected slots require window 12,
enabled prefetch and a prediction depth of 1–16. The policy scores actual router demand with a 32-step
half-life; it does not change routing top-k, expert weights or precision.

| Profile setting | Default and purpose |
|---|---|
| `host`, `port` | `127.0.0.1`, `8080`; localhost only |
| `args.model`, `args.moe-experts` | First GGUF shard and matching expert directory/manifest; required paths |
| `args.alias` | Model ID; must equal its key under `models` |
| `args.ctx-size`, `args.ctx-checkpoints` | `131072` tokens for Flash Next, `65536` for Qwen3.6; `1` checkpoint. The worker grows its caches as a conversation does, so a larger limit costs nothing until used: on the M5 Max 36 GB, 131072 against 65536 changed neither memory after load nor speed, and a 120k-token conversation through the router peaked at 17.6 GB system-wide, with later turns restored from the prompt cache in 3–5 s ([measurements](../MACM5MAX36GB.md#a-128k-context-limit-l4)) |
| `args.cache-ram` | `4096` MiB for Flash Next; unset (`8192`) for Qwen3.6. Saved states of earlier conversations, restored when a request continues one instead of reprocessing it. Above the limit, the oldest entry under a quarter of the largest entry's tokens is dropped first (short side requests before the long conversation they interrupt), then the oldest; the newest is always kept. `0` disables it. A Flash Next entry takes about 225 MiB plus 27 KiB per token with `f16` KV (about 16 KiB per token with `q8_0`, estimated from the cache sizes) ([measurements](../MACM5MAX36GB.md#prompt-cache-limit-m2-uncommitted)) |
| `args.threads` | `10` CPU threads |
| `args.prefill-max-tokens` | `3072` tokens per prefill microbatch |
| `args.moe-prefill-bank`, `args.moe-decode-bank` | `512` / `256` shared expert slots for prefill batches under 1,024 tokens, or all batches with the pipeline off (Flash Next / Qwen3.6); larger banks use more RAM; `0` shared slots retained during decode |
| `args.n-gpu-layers`, `args.flash-attn` | `99`, `on`; expert execution requires Metal |
| `args.cache-type-k`, `args.cache-type-v` | `q8_0` KV quantization. The Qwen4Exp indexer stores keys only, always as `bf16` whatever these say: its keys are cached before their norm and outgrow `f16`'s range, which raised perplexity 2.5–4% at 32k context. On the M5 Max 36 GB, `q8_0` against `f16`: half the KV memory (12.8 against 24 KiB per token, about 1.4 GB less at 120k tokens), prompts 3% (6k tokens) to 9% (120k) slower, generation unchanged, perplexity unchanged at 4k context and at 32k on prose (−0.05% ± 0.04) ([measurements](../MACM5MAX36GB.md#kv-cache-type-and-the-indexer-cache-x1)) |
| `args.no-mmap` | `true`; explicit reads for Qwen weights |
| `args.hc-type` | Unset (`bf16`). Flash Next only: `q8_0` converts the bf16 hyper-connection weights (1.2 GB) to Q8_0 while loading, so each token reads about 0.56 GB less and that memory goes to the file cache. **Changes output:** against `bf16`, mean KL divergence 0.0104 and 96.6% of next tokens the same, where storing the same values in another format already gives 0.0044 and 97.7%; perplexity unchanged within noise. On the M5 Max 36 GB: 4–6% faster generation where the output stayed the same, 564 MB less memory ([measurements](../MACM5MAX36GB.md#q8_0-hyper-connection-weights-b2-hc-type-uncommitted)). Requires `no-mmap` |
| `env.LLAMA_MOE_DECODE_WINDOW` | `12` decode slots per layer; with 3 prefetch slots that is 15 for Flash Next, plus 24 protected slots (39) for Qwen3.6 |
| `env.LLAMA_MOE_PROTECTED_SLOTS` | `0` for Flash Next, `24` for Qwen3.6: additional protected slots per layer; `0` disables the protected tier. On the M5 Max 36 GB, `0` made Flash Next generate about 1% faster, read 16% less from the SSD and use 1.9 GB less memory while generating, because macOS's file cache already held the protected experts ([measurements](../MACM5MAX36GB.md#protected-decode-slots-p0)) |
| `env.LLAMA_MOE_LOOKAHEAD_PREFETCH` | `1`; enable predictive expert reads |
| `env.LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH`, `env.LLAMA_MOE_PREFETCH_READ` | Predict `3` experts, read `2` |
| `env.LLAMA_MOE_LATE_PREFETCH`, `env.LLAMA_MOE_LATE_READ` | Unset (off). Flash Next only: route each next layer on the previous layer's output and read up to `LATE_READ` (default all) of its `LATE_PREFETCH` best guesses while its attention runs; `8` and `8` measured +1.8% generation after a short prompt, with identical output. Replaces the lookahead guess for one-token steps (see [late prefetch](../docs/les/late-prefetch.md)) |
| `env.LLAMA_MOE_LATE_MIX` | `0`; `1` makes late guesses with the next layer's full hyper-connection mix (more accurate, more GPU work) |
| `env.LLAMA_MOE_DECODE_ROWS` | `1` token per decode step; up to the narrowest window / 10 lets batches of a few tokens, such as speculative checks, stay in the decode phase. Flash Next's 15-slot window allows only 1 (with prefetch depth 8, 20 slots allow 2); protected slots raise it |
| `args.spec-type`, `args.spec-draft-model`, `args.spec-draft-n-max`, `args.spec-draft-n-min`, `args.spec-draft-p-min`, `args.spec-draft-ngl` | Unset; `spec-draft-p-min` drafts only tokens the draft model gives at least that probability (over its top 10 candidates). Speculative decoding needs `LLAMA_MOE_DECODE_ROWS` of 2 or more, and drafts are capped at rows − 1 tokens (see [speculative decoding](../docs/les/speculative-decoding.md)) |
| `env.LLAMA_MOE_PREFILL_PIPELINE` | `1`; batches of 1,024 tokens or more read whole layers ahead into a staging buffer (two layers' size) and compute from it, overlapping reads with computation; `0` uses the shared bank, which is slower for long prompts |
| `env.LLAMA_GPU_KEEPALIVE` | `1`; keep the GPU active while the worker exists; `0` disables it |
| `env.LLAMA_MOE_QUIET` | `1`; suppress expert diagnostics; `0` enables them |

Use the argument names as CLI overrides (`--ctx-size`, `--threads`, etc.).
`--no-mmap true/false` overrides its JSON boolean. Cache overrides are
`--moe-decode-window` and `--moe-protected-slots`. Other environment entries are
set in the profile. Stop with Ctrl-C. The worker, its readers and the GPU
keepalive terminate with the server, including after a forced server exit.

The installed template is `share/moeshift/moeshift.json`. Guided setup writes only the selected profile. Manual setup copies both example profiles and prints the config location. File paths in the generated profile are absolute. Downloaded GGUFs and experts live under `~/.moeshift/models/<model-id>/`; local GGUFs stay in place.

An invalid user configuration can be reconfigured interactively. MoEShift reports the reason and saves the original next to the replacement. For unattended startup, supply an existing configuration or pass `--config FILE`.
