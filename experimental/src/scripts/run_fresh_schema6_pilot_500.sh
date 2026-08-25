#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PILOT_DIR="${1:?usage: run_fresh_schema6_pilot_500.sh PILOT_DIR [BASE_PORT] [fresh|resume]}"
BASE_PORT="${2:-8040}"
RUN_MODE="${3:-fresh}"
if [[ "$RUN_MODE" != fresh && "$RUN_MODE" != resume ]]; then
  echo "RUN_MODE must be fresh or resume" >&2
  exit 2
fi
if [[ "$RUN_MODE" == fresh && -e "$PILOT_DIR" ]] && find "$PILOT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "PILOT_DIR must be absent or empty: $PILOT_DIR" >&2
  exit 2
fi
mkdir -p "$PILOT_DIR"
PILOT_DIR="$(cd "$PILOT_DIR" && pwd)"

stratum_mode() {
  local directory="$1"
  if [[ "$RUN_MODE" == resume && -f "$directory/result.json.progress.json" ]]; then
    echo resume
  else
    echo fresh
  fi
}

"$ROOT_DIR/experimental/src/scripts/run_fresh_schema6_capture.sh" \
  "$PILOT_DIR/peer" 300 "$BASE_PORT" peer 1 "$(stratum_mode "$PILOT_DIR/peer")"
"$ROOT_DIR/experimental/src/scripts/run_fresh_schema6_capture.sh" \
  "$PILOT_DIR/direct_r1" 100 "$BASE_PORT" direct_r1 1 "$(stratum_mode "$PILOT_DIR/direct_r1")"
"$ROOT_DIR/experimental/src/scripts/run_fresh_schema6_capture.sh" \
  "$PILOT_DIR/unguided" 100 "$BASE_PORT" unguided 1 "$(stratum_mode "$PILOT_DIR/unguided")"

"$ROOT_DIR/.venv-metamon/bin/python" \
  "$ROOT_DIR/experimental/src/scripts/summarize_schema6_capture_pilot.py" \
  --root "$PILOT_DIR" \
  --output "$PILOT_DIR/summary.json"
