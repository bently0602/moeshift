# MoEShift apps

Two small Mac apps, one per model. Each downloads and prepares its model, then starts and stops
the engine with the settings measured on a Mac Studio with M5 Max and 36 GB
([measurements](../MACM5MAX36GB.md)):

| App | Model | Download | Kept on disk | Default port |
|---|---|---:|---:|---:|
| MoEShift Flash Next | Qwen 3.8 Flash Next, ISTA-DASLab GSQ-RCO IQ3_XXS | 75.8 GB | 75.8 GB | 8080 |
| MoEShift A3B | Qwen3.6 35B A3B, unsloth UD-Q6_K_XL | 31.8 GB | 31.8 GB | 8081 |

## Build

Requires Apple Silicon, macOS 14 or later, the Apple Command Line Tools (`xcode-select --install`)
and CMake (`brew install cmake`). Xcode is not needed. From the repository root:

```sh
apps/build.sh
```

This builds the engine (`scripts/build-moeshift.sh`), then both apps into `build/apps/`. Each app
bundle contains the `moeshift` binary and a copy of [configs/moeshift.json](../configs/moeshift.json),
the tuned profiles. The engine is built for the Mac that builds it (`GGML_NATIVE`), and the apps
are signed ad hoc, for that Mac only. Open them from `build/apps/` or copy them to `/Applications`.
`apps/build.sh --skip-engine` reuses an existing `build/bin/moeshift`.

## First start

The app asks for two folders:

- **Model files (kept):** the prepared experts, which the engine reads from the SSD for every
  token, and the rest of the model in one file. Default `~/.moeshift/models/<model>`, which the
  `moeshift` command line uses too, so models prepared there before are found.
- **Download (temporary):** the GGUF from Hugging Face. By default the external drive with the most
  free space, so that the download is not written to the internal SSD as well; otherwise
  `~/Downloads/MoEShift`. A stopped download resumes. Each file is checked against Hugging Face's
  SHA-256 before it is used.

Then it runs `moeshift experts extract`, `verify` and `strip` (see
[configuration](../configs/README.md#manual-setup)), and deletes the download if you leave that
option on. Nothing but the download is ever deleted.

Files already on disk are reused:

- A model folder with `experts/` and the stripped GGUF, or `experts/` and the original GGUF (as
  the `moeshift` command line prepares it), opens straight to the main window.
- Before downloading, setup looks for the complete original GGUF in the model folder and where the
  experts were extracted from (recorded in `experts/manifest.json`), and prepares from it instead
  ("Prepare from the GGUF on disk"). Existing experts are verified, not extracted again.
- "Use a GGUF I already downloaded…" prepares from a GGUF anywhere else.

Files found or chosen this way are never deleted. To shrink a model folder that holds the original
GGUF, run setup from it and delete the original shards once the stripped GGUF is written.

## Using it

- **Start / Stop.** Start also loads the model, so the first request does not wait for it.
  Quitting the app stops the engine. If the engine or the model fails to load, the app stops the
  engine and shows the error; the Log has the engine's output.
- **Connect a client** with the API URL (`http://127.0.0.1:<port>/v1`, OpenAI-compatible) and the
  model ID shown; a client that sends another model ID, or none, gets this model. The engine serves OpenAI chat completions, completions and Responses, and
  Anthropic Messages. A client that asks for another endpoint gets `File Not Found`, and the Log
  shows the path it asked for (`no endpoint for POST …`).
- **Port** (1024–65535) is saved as you type, and the field says if a port is in use by another
  program. While the engine runs, "Restart on port N" moves it to the new port (the model loads
  again); until then it keeps serving on the old one, which the API URL shows.
- **Share on the local network** serves on every network interface and requires the API key shown
  (`Authorization: Bearer <key>`). Turning off **Require the API key** lets anyone on the network
  use the model without one. macOS may ask whether `moeshift` may accept incoming connections.
- **Thinking** limits how long the model may think before it answers: Off, Low (1,024 tokens),
  Medium (4,096) or High (no limit). The default is Medium for Flash Next, which gave the unlimited
  output on all 16 measured problems while capping runaway thinking
  ([measurements](../MACM5MAX36GB.md#reasoning-effort)), and High for A3B. A client that sends its own reasoning effort
  (`reasoning_effort`, Responses `reasoning.effort`, Anthropic `thinking` or `output_config.effort`)
  or thinking budget overrides it. Change it while the engine is stopped.
- Everything else is fixed: context 131,072 tokens, n-gram drafting on repeated text, and the
  cache settings of each profile.

Settings and the generated engine configuration are in
`~/Library/Application Support/MoEShift/<model>/` (mode 0600, since they hold the API key). The
engine's output is kept in memory for the Log view, not written to disk.

Running both apps at once is possible on different ports, but both models then compete for memory
and the file cache.

## Self-tests

From a terminal, without a window:

```sh
"build/apps/MoEShift A3B.app/Contents/MacOS/A3BApp" --selftest-engine [--port N] [--lan [--no-key]] [--effort LEVEL] [--model-folder DIR] [--switch-port M]
"build/apps/MoEShift A3B.app/Contents/MacOS/A3BApp" --selftest-prepare MODEL_DIR DOWNLOAD_DIR [GGUF]
```

The first starts the engine with the saved settings (or the given port, network sharing, with or
without the key, thinking effort or model folder), waits until the model is loaded and stops it, or prints the
error the window would show. When sharing, it checks that a request without a key on a network
address is refused only if the key is required. With `--switch-port` it first restarts the engine
on port M, as the port field does, and checks that only M answers.
The second runs the setup into the given folders and keeps the download. Neither changes the
saved settings.
