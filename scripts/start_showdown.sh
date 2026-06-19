#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOWDOWN_DIR="$ROOT_DIR/external/pokemon-showdown"
PORT="${1:-8000}"

exec node "$SHOWDOWN_DIR/pokemon-showdown" start --no-security "$PORT"
