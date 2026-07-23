#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOUL_PLAY_DIR="$ROOT_DIR/external/foul-play"
FOUL_PLAY_REPO="https://github.com/pmariglia/foul-play.git"
FOUL_PLAY_COMMIT="${FOUL_PLAY_COMMIT:-e1e2ca650598621e85c3b6ab751c66e625489934}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
POKE_ENGINE_SOURCE_DIR="$ROOT_DIR/external/source-dist/poke_engine-0.0.47"

mkdir -p "$ROOT_DIR/external"

if [[ ! -d "$FOUL_PLAY_DIR/.git" ]]; then
  git clone "$FOUL_PLAY_REPO" "$FOUL_PLAY_DIR"
fi

git -C "$FOUL_PLAY_DIR" fetch --quiet origin "$FOUL_PLAY_COMMIT"
git -C "$FOUL_PLAY_DIR" checkout --quiet "$FOUL_PLAY_COMMIT"

"$PYTHON_BIN" -m venv "$ROOT_DIR/.venv-foul-play"
"$ROOT_DIR/.venv-foul-play/bin/python" -m pip install --upgrade pip
"$ROOT_DIR/.venv-foul-play/bin/python" -m pip install -r "$FOUL_PLAY_DIR/requirements.txt"

mkdir -p "$ROOT_DIR/external/source-dist" "$POKE_ENGINE_SOURCE_DIR"
"$ROOT_DIR/.venv-foul-play/bin/python" -m pip download \
  --no-binary :all: \
  --no-deps \
  poke-engine==0.0.47 \
  -d "$ROOT_DIR/external/source-dist"
tar -xzf "$ROOT_DIR/external/source-dist/poke_engine-0.0.47.tar.gz" \
  -C "$POKE_ENGINE_SOURCE_DIR" \
  --strip-components=1

git -C "$FOUL_PLAY_DIR" rev-parse HEAD
