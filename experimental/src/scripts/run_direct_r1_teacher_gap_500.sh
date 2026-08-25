#!/bin/sh
# Frozen 250-pair/500-game direct-r1 versus r1-search teacher-gap evaluation.
# Prerequisites: local Showdown was started with METAGROSS_EVAL_PAIR_DIR equal
# to PAIR_DIR, and the frozen r1 prior server is healthy at PRIOR_SERVER_URL.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=${ROOT:-"$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)"}
RUN_ID=${RUN_ID:-direct_r1_vs_r1_search_teacher_gap_500_v1}
N_GAMES=${N_GAMES:-500}
MIRROR_SEED=${MIRROR_SEED:-2026072901}
OUT_DIR=${OUT_DIR:-"$ROOT/experimental/runs/$RUN_ID"}
PAIR_DIR=${PAIR_DIR:-"$OUT_DIR/pair-registrations"}
PRIOR_SERVER_URL=${PRIOR_SERVER_URL:-http://127.0.0.1:8977}
WEBSOCKET_URI=${WEBSOCKET_URI:-ws://localhost:8000/showdown/websocket}

mkdir -p "$OUT_DIR/logs" "$PAIR_DIR"
PYTHONPATH="$ROOT/experimental/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
exec "$ROOT/.venv-metamon/bin/python" -m eval.run \
  --mode h2h --server local --format gen9randombattle \
  --websocket-uri "$WEBSOCKET_URI" \
  --agent-a direct_r1 --agent-b foul_play_root_priors \
  --n-games "$N_GAMES" --paired --mirrored-pairs --mirror-seed "$MIRROR_SEED" \
  --foul-play-python "$ROOT/.venv-fp-priors/bin/python" \
  --foul-play-search-time-ms 500 --foul-play-search-parallelism 8 \
  --foul-play-search-threads 1 --cpuct 2.0 \
  --agent-b-prior-server-url "$PRIOR_SERVER_URL" --agent-b-require-priors \
  --showdown-dir "$ROOT/external/pokemon-showdown" \
  --mirrored-team-generator "$ROOT/experimental/src/scripts/generate_mirrored_randbats_pair.cjs" \
  --pair-registration-dir "$PAIR_DIR" --log-dir "$OUT_DIR/logs" \
  --json-out "$OUT_DIR/results.json" --run-id "$RUN_ID" --fail-fast
