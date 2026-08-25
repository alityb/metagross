#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="${1:?usage: run_terminal_mcts_h2h.sh RUN_DIR [N_GAMES]}"
N_GAMES="${2:-10}"
BASE_PORT="${METAGROSS_TERMINAL_H2H_SHOWDOWN_PORT:-8765}"
PRIOR_A_PORT="${METAGROSS_TERMINAL_H2H_PRIOR_A_PORT:-8977}"
PRIOR_B_PORT="${METAGROSS_TERMINAL_H2H_PRIOR_B_PORT:-8978}"
CHECKPOINT_SHA256="c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
MIRROR_SEED="${METAGROSS_TERMINAL_H2H_MIRROR_SEED:-2026101501}"
PRODUCTION_SEED="${METAGROSS_TERMINAL_H2H_PRODUCTION_SEED:-1515151515151515151515151515151515151515151515151515151515151515}"
USERNAME_PREFIX="${METAGROSS_TERMINAL_H2H_USERNAME_PREFIX:-tmc1}"
RUN_ID="${METAGROSS_TERMINAL_H2H_RUN_ID:-terminal-mcts-cycle1-canary}"

if [[ ! "$N_GAMES" =~ ^[0-9]+$ ]] || ((N_GAMES < 2 || N_GAMES % 2 != 0)); then
  echo "N_GAMES must be an even integer >=2" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "run directory must be absent or empty: $RUN_DIR" >&2
  exit 2
fi
for port in "$BASE_PORT" "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null; then
    echo "port already occupied: $port" >&2
    exit 2
  fi
done

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/registrations"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${PIDS[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

env CUDA_VISIBLE_DEVICES='' \
  "$ROOT_DIR/experimental/src/scripts/start_showdown.sh" "$BASE_PORT" \
  >>"$RUN_DIR/showdown.log" 2>&1 &
PIDS+=("$!")

for side in a b; do
  if [[ "$side" == a ]]; then port="$PRIOR_A_PORT"; else port="$PRIOR_B_PORT"; fi
  env METAMON_CACHE_DIR="$ROOT_DIR/external/metamon_cache" \
    TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' \
    "$ROOT_DIR/.venv-metamon/bin/python" -u \
    "$ROOT_DIR/srcs/metagross/prior_server.py" \
    --local-run-dir "$ROOT_DIR/srcs/models" \
    --local-run-name randbats_exit_r1 \
    --checkpoint 5 \
    --checkpoint-sha256 "$CHECKPOINT_SHA256" \
    --trajectory-mode causal-history \
    --username "term${side}" \
    --port "$port" \
    --decision-dump "$RUN_DIR/prior-${side}.jsonl" \
    >>"$RUN_DIR/prior-${side}.log" 2>&1 &
  PIDS+=("$!")
done

for _attempt in {1..180}; do
  ready=1
  for port in "$BASE_PORT" "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null || ready=0
  done
  ((ready == 1)) && break
  sleep 1
done
for port in "$BASE_PORT" "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null || {
    echo "local infrastructure failed readiness on port $port" >&2
    exit 1
  }
done

env METAMON_CACHE_DIR="$ROOT_DIR/external/metamon_cache" \
  TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' \
  METAGROSS_TERMINAL_MCTS_TEACHER_SLOT=agent_a \
  METAGROSS_TERMINAL_MCTS_PYTHON="$ROOT_DIR/.venv-metamon/bin/python" \
  METAGROSS_TERMINAL_MCTS_SCRIPT="$ROOT_DIR/experimental/src/scripts/terminal_mcts_live_decision.py" \
  METAGROSS_TERMINAL_MCTS_PYTHONPATH="$ROOT_DIR/experimental/engine/pe_v3_learned_priors/poke-engine-py/python:$ROOT_DIR/experimental/src" \
  METAGROSS_TERMINAL_MCTS_WORKERS="${METAGROSS_TERMINAL_MCTS_WORKERS:-8}" \
  METAGROSS_TERMINAL_MCTS_TIMEOUT_SECONDS=60 \
  PYTHONPATH="$ROOT_DIR/experimental/src:$ROOT_DIR" \
  "$ROOT_DIR/.venv-metamon/bin/python" -u "$ROOT_DIR/experimental/src/eval/run.py" \
  --mode h2h \
  --server local \
  --format gen9randombattle \
  --websocket-uri "ws://127.0.0.1:${BASE_PORT}/showdown/websocket" \
  --paired \
  --mirrored-pairs \
  --mirror-seed "$MIRROR_SEED" \
  --showdown-dir "$ROOT_DIR/external/pokemon-showdown" \
  --mirrored-team-generator "$ROOT_DIR/experimental/src/scripts/generate_mirrored_randbats_pair.cjs" \
  --pair-registration-dir "$RUN_DIR/registrations" \
  --agent-a production_r1_search_first \
  --agent-b production_r1_search_first \
  --agent-a-prior-server-url "http://127.0.0.1:${PRIOR_A_PORT}" \
  --agent-b-prior-server-url "http://127.0.0.1:${PRIOR_B_PORT}" \
  --agent-a-require-priors \
  --agent-b-require-priors \
  --strict-isolated-priors \
  --agent-a-decision-log "$RUN_DIR/agent-a-decisions.jsonl" \
  --agent-b-decision-log "$RUN_DIR/agent-b-decisions.jsonl" \
  --foul-play-python "$ROOT_DIR/.venv-foul-play/bin/python" \
  --foul-play-search-time-ms 500 \
  --foul-play-search-parallelism 8 \
  --foul-play-search-threads 1 \
  --cpuct 2.0 \
  --production-run-seed "$PRODUCTION_SEED" \
  --concurrent-games 1 \
  --fail-fast \
  --game-timeout-seconds 900 \
  --n-games "$N_GAMES" \
  --username-prefix "$USERNAME_PREFIX" \
  --run-id "$RUN_ID" \
  --json-out "$RUN_DIR/result.json" \
  --log-dir "$RUN_DIR/logs"
