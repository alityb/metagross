#!/usr/bin/env bash
set -euo pipefail

# Build an isolated Gen9 engine with production external root priors plus
# root-centered learned MLP leaf evaluation. Never overwrite .venv-fp-priors.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/engine/pe_v3_learned_priors"
PATCH="$ROOT/engine/patches/poke-engine-0.0.47-gen9-learned-priors.patch"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-pe-gen9-priors/bin/python}"

rm -rf "$BUILD"
cp -R "$ROOT/engine/pe_v2" "$BUILD"
patch -d "$BUILD" -p1 < "$PATCH"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/.cargo-gen9-learned-priors}" \
  "$PYTHON_BIN" -m pip install -v --force-reinstall --no-cache-dir "$BUILD" \
  --config-settings="build-args=--no-default-features --features poke-engine/gen9,poke-engine/terastallization"
