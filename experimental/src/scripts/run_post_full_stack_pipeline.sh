#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FULLSTACK_PID="${FULLSTACK_PID:?set FULLSTACK_PID to the active full-stack supervisor}"
FULLSTACK_ROOT="${FULLSTACK_ROOT:-$ROOT/experimental/runs/g4_full_stack_20260730}"
TEACHER_ROOT="${TEACHER_ROOT:-$ROOT/experimental/runs/direct_r1_vs_r1_search_teacher_gap_500_v1}"
SCALED_ROOT="${SCALED_ROOT:-$ROOT/experimental/runs/online_rl_scaled_replay_5k_20260730}"
SHOWDOWN_PORT="${SHOWDOWN_PORT:-8013}"
PRIOR_PORT="${PRIOR_PORT:-8999}"
PYTHON="$ROOT/.venv-metamon/bin/python"

mkdir -p "$TEACHER_ROOT/pair-registrations" "$TEACHER_ROOT/logs" "$SCALED_ROOT"

while kill -0 "$FULLSTACK_PID" 2>/dev/null; do
  sleep 60
done

"$PYTHON" -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); data=json.loads(p.read_text()); assert data.get("summary", {}).get("completed_games") == 1000' \
  "$FULLSTACK_ROOT/ab/result.json"

showdown_pid=""
prior_pid=""
cleanup() {
  for pid in "$prior_pid" "$showdown_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

METAGROSS_EVAL_PAIR_DIR="$TEACHER_ROOT/pair-registrations" \
SHOWDOWN_DIR="$ROOT/external/pokemon-showdown" \
bash "$ROOT/experimental/src/scripts/start_showdown.sh" "$SHOWDOWN_PORT" \
  >"$TEACHER_ROOT/showdown.log" 2>&1 &
showdown_pid=$!

METAMON_CACHE_DIR="$ROOT/srcs/runtime/metamon-cache" \
TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true WANDB_MODE=disabled \
"$PYTHON" -u "$ROOT/experimental/src/scripts/prior_server.py" \
  --local-run-dir "$ROOT/srcs/models" \
  --local-run-name randbats_exit_r1 \
  --local-base-model Kakuna \
  --checkpoint 5 \
  --username r1search \
  --host 127.0.0.1 \
  --port "$PRIOR_PORT" \
  >"$TEACHER_ROOT/prior.log" 2>&1 &
prior_pid=$!

ready=0
for _ in $(seq 1 240); do
  if curl --silent --fail "http://127.0.0.1:$PRIOR_PORT/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
[[ "$ready" == "1" ]]

ROOT="$ROOT" \
RUN_ID=direct_r1_vs_r1_search_teacher_gap_smoke_2_v1 \
N_GAMES=2 \
MIRROR_SEED=2026073003 \
OUT_DIR="$TEACHER_ROOT/smoke" \
PAIR_DIR="$TEACHER_ROOT/pair-registrations" \
PRIOR_SERVER_URL="http://127.0.0.1:$PRIOR_PORT" \
WEBSOCKET_URI="ws://localhost:$SHOWDOWN_PORT/showdown/websocket" \
"$ROOT/experimental/src/scripts/run_direct_r1_teacher_gap_500.sh"

ROOT="$ROOT" \
OUT_DIR="$TEACHER_ROOT" \
PAIR_DIR="$TEACHER_ROOT/pair-registrations" \
PRIOR_SERVER_URL="http://127.0.0.1:$PRIOR_PORT" \
WEBSOCKET_URI="ws://localhost:$SHOWDOWN_PORT/showdown/websocket" \
"$ROOT/experimental/src/scripts/run_direct_r1_teacher_gap_500.sh"

cleanup
showdown_pid=""
prior_pid=""
trap - EXIT INT TERM

exec "$PYTHON" "$ROOT/experimental/src/scripts/online_rl_controller.py" \
  --config "$ROOT/experimental/configs/online_rl_scaled_replay_5k.json" \
  --run-dir "$SCALED_ROOT"
