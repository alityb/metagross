#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOWDOWN_DIR="$ROOT_DIR/external/pokemon-showdown"
SHOWDOWN_REPO="https://github.com/smogon/pokemon-showdown.git"

mkdir -p "$ROOT_DIR/external"

if [[ ! -d "$SHOWDOWN_DIR/.git" ]]; then
  git clone "$SHOWDOWN_REPO" "$SHOWDOWN_DIR"
fi

npm install --prefix "$SHOWDOWN_DIR"

if [[ ! -f "$SHOWDOWN_DIR/config/config.js" ]]; then
  cp "$SHOWDOWN_DIR/config/config-example.js" "$SHOWDOWN_DIR/config/config.js"
fi

git -C "$SHOWDOWN_DIR" rev-parse HEAD
