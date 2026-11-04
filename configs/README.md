# MoEShift configuration

The [standard template](moeshift.json) includes both tested model profiles:

| Model ID and alias | Prefill slots | Protected slots per layer |
|---|---:|---:|
| `qwen-3.8-flash-next-rco-gsq-iq3-xxs` | 512 | 24 |
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
24 protected slots per layer. Positive protected slots require that combination
and enabled prefetch. The policy scores actual router demand with a 32-step
half-life; it does not change routing top-k, expert weights or precision.

| Profile setting | Default and purpose |
|---|---|
| `host`, `port` | `127.0.0.1`, `8080`; localhost only |
| `args.model`, `args.moe-experts` | First GGUF shard and matching expert directory/manifest; required paths |
| `args.alias` | Model ID; must equal its key under `models` |
| `args.ctx-size`, `args.ctx-checkpoints` | `65536` tokens; `1` checkpoint |
| `args.threads` | `10` CPU threads |
| `args.prefill-max-tokens` | `3072` tokens per prefill microbatch |
| `args.moe-prefill-bank`, `args.moe-decode-bank` | `512` / `256` shared expert slots during prefill (Flash Next / Qwen3.6); larger banks use more RAM; `0` shared slots retained during decode |
| `args.n-gpu-layers`, `args.flash-attn` | `99`, `on`; expert execution requires Metal |
| `args.cache-type-k`, `args.cache-type-v` | `q8_0` KV quantization; the Qwen4Exp indexer stores keys only |
| `args.no-mmap` | `true`; explicit reads for Qwen weights |
| `env.LLAMA_MOE_DECODE_WINDOW` | `12` decode slots per layer; combined with 3 prefetch slots and 24 protected slots, the default total is 39 |
| `env.LLAMA_MOE_PROTECTED_SLOTS` | `24` additional protected slots; larger values use more RAM and can reduce SSD reads; `0` disables the protected tier |
| `env.LLAMA_MOE_LOOKAHEAD_PREFETCH` | `1`; enable predictive expert reads |
| `env.LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH`, `env.LLAMA_MOE_PREFETCH_READ` | Predict `3` experts, read `2` |
| `env.LLAMA_MOE_PREFILL_PIPELINE` | `1`; overlap prefill reads with computation |
| `env.LLAMA_GPU_KEEPALIVE` | `1`; keep the GPU active while the worker exists; `0` disables it |
| `env.LLAMA_MOE_QUIET` | `1`; suppress expert diagnostics; `0` enables them |

Use the argument names as CLI overrides (`--ctx-size`, `--threads`, etc.).
`--no-mmap true/false` overrides its JSON boolean. Cache overrides are
`--moe-decode-window` and `--moe-protected-slots`. Other environment entries are
set in the profile. Stop with Ctrl-C. The worker, its readers and the GPU
keepalive terminate with the server, including after a forced server exit.

The installed template is `share/moeshift/moeshift.json`. Guided setup writes only the selected profile. Manual setup copies both example profiles and prints the config location. File paths in the generated profile are absolute. Downloaded GGUFs and experts live under `~/.moeshift/models/<model-id>/`; local GGUFs stay in place.

An invalid user configuration can be reconfigured interactively. MoEShift reports the reason and saves the original next to the replacement. For unattended startup, supply an existing configuration or pass `--config FILE`.
