#!/bin/sh
# Build MoEShift from source with the flags documented in docs/build.md.
#
# Usage: scripts/build-moeshift.sh [--test] [--install [PREFIX]] [--run [ARGS...]]
#   --test     also build and run the focused native checks and the router check
#   --install  install to PREFIX (default: $HOME/.local)
#   --run      then start build/bin/moeshift from the current directory; later
#              arguments are passed to it
# The router check uses $PYTHON, or the first Python 3.10+ found on PATH.
set -eu

usage() {
  echo "Usage: $0 [--test] [--install [PREFIX]] [--run [ARGS...]]" >&2
  exit 2
}

test=0
install=0
run=0
prefix="$HOME/.local"
while [ $# -gt 0 ]; do
  case "$1" in
    --test) test=1 ;;
    --install)
      install=1
      case "${2-}" in
        ''|-*) ;;
        *) prefix="$2"; shift ;;
      esac
      ;;
    --run) run=1; shift; break ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
  shift
done

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "MoEShift requires macOS on Apple Silicon" >&2
  exit 1
fi
command -v cmake >/dev/null || { echo "CMake not found; install it with: brew install cmake" >&2; exit 1; }
xcode-select -p >/dev/null 2>&1 || { echo "Apple Command Line Tools not found; install them with: xcode-select --install" >&2; exit 1; }

supported_python() {
  command -v "$1" >/dev/null 2>&1 &&
    "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null
}

if [ "$test" = 1 ]; then
  if [ -n "${PYTHON-}" ]; then
    supported_python "$PYTHON" || { echo "PYTHON=$PYTHON is not Python 3.10 or newer" >&2; exit 1; }
    python="$PYTHON"
  else
    python=
    for candidate in python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
      if supported_python "$candidate"; then python="$candidate"; break; fi
    done
    [ -n "$python" ] || { echo "Python 3.10 or newer not found; install it with: brew install python" >&2; exit 1; }
  fi
fi

start="$(pwd)"
cd "$(dirname "$0")/.."
root="$(pwd)"
jobs="$(sysctl -n hw.ncpu)"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=ON \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX="$prefix"
cmake --build build --target moeshift -j "$jobs"

if [ "$test" = 1 ]; then
  checks="test-les-protected-cache test-les-active-ranges test-metal-moe-phases test-indexer-keys test-kv-grow test-metal-hc-fusion test-ssd-rows"
  # shellcheck disable=SC2086
  cmake --build build --target $checks -j "$jobs"
  ctest --test-dir build -R "^($(echo $checks | tr ' ' '|'))\$" --output-on-failure
  PYTHONDONTWRITEBYTECODE=1 "$python" tests/test-moeshift-router.py build/bin/moeshift
fi

if [ "$install" = 1 ]; then
  cmake --install build --component moeshift
  echo "Installed $prefix/bin/moeshift"
else
  echo "Built build/bin/moeshift"
fi

if [ "$run" = 1 ]; then
  cd "$start"
  exec "$root/build/bin/moeshift" "$@"
fi
