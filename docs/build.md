# Build MoEShift from source

Requires Apple Silicon, Apple Command Line Tools and CMake. Python 3.10+ is
needed for expert preparation and the Python checks. Use the existing GGUF
package dependencies; see `gguf-py/pyproject.toml`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=ON \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build --target moeshift -j 8
cmake --install build --component moeshift
```

`scripts/build-moeshift.sh` runs the same configuration and build. Add
`--test` to run the focused checks below, `--install [PREFIX]` to install
(default `$HOME/.local`), or `--run [ARGS...]` to start `build/bin/moeshift`
afterwards with the remaining arguments. The router check uses `PYTHON` when set, otherwise
the first Python 3.10+ on `PATH`.

This installs the executable, default configuration and expert preparation
sources. Homebrew installs the wizard's Python environment automatically.
For a source install, activate a Python environment with the GGUF dependencies
before the first run. Run `~/.local/bin/moeshift` to choose a model and start
the server. Future launches use the saved configuration directly.

For preparation from the source checkout:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install ./gguf-py
.venv/bin/python tools/moe/experts.py extract --model "$MODEL" --out-dir "$EXPERTS"
.venv/bin/python tools/moe/experts.py verify --model "$MODEL" --experts "$EXPERTS"
```

`MODEL` must name the first GGUF shard and `EXPERTS` a new output directory.
Keep the original shards alongside the generated files.

## Focused native checks

```sh
cmake --build build --target test-les-protected-cache test-les-active-ranges \
  test-metal-moe-phases test-indexer-keys test-kv-grow test-metal-hc-fusion test-ssd-rows -j 8
ctest --test-dir build -R '^(test-les-protected-cache|test-les-active-ranges|test-metal-moe-phases|test-indexer-keys|test-kv-grow|test-metal-hc-fusion|test-ssd-rows)$' --output-on-failure
PYTHONDONTWRITEBYTECODE=1 python3 tests/test-moeshift-router.py build/bin/moeshift
.venv/bin/python tests/test-moeshift-onboarding.py
```

Real-model validation is separate from these focused checks. The parent-lifetime
check takes the executable and a model configuration:

```sh
python3 tests/test-moeshift-parent.py build/bin/moeshift "$HOME/.config/moeshift/config.json"
```
