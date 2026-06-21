#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_TARBALL="$ROOT_DIR/external/source-dist/poke_engine-0.0.47.tar.gz"
BUILD_DIR="$ROOT_DIR/external/poke_engine_learned_eval"
PATCH_FILE="$ROOT_DIR/engine/patches/poke-engine-0.0.47-learned-eval.patch"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-foul-play/bin/python}"

if [[ ! -f "$SRC_TARBALL" ]]; then
  echo "missing source tarball: $SRC_TARBALL" >&2
  echo "run scripts/install_foul_play.sh first" >&2
  exit 1
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
tar -xzf "$SRC_TARBALL" -C "$BUILD_DIR" --strip-components=1
patch -d "$BUILD_DIR" -p1 < "$PATCH_FILE"
"$PYTHON_BIN" -m pip install -v --force-reinstall --no-cache-dir "$BUILD_DIR" \
  --config-settings="build-args=--features poke-engine/terastallization --no-default-features"
