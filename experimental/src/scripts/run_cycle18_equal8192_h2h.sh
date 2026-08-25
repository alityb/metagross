#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN="$ROOT/experimental/runs/search_native_v2_cycle18_h2h_protocol_20260815"
MANIFEST="$RUN/PREMEASUREMENT_MANIFEST.json"
PAIR_MANIFEST="$RUN/preflight-result.json.pairs.json"
PAIR_SHA="$(shasum -a 256 "$PAIR_MANIFEST" | awk '{print $1}')"
BASE_PORT=8895
PRIOR_A_PORT=9177
PRIOR_B_PORT=9178
CHECKPOINT_SHA256=c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93
MIRROR_SEED=202618081501
PRODUCTION_SEED=1818181818181818181818181818181818181818181818181818181818181818
USERNAME_PREFIX=c18e8k
RUN_ID=cycle18-equal8192-canary

if [[ "${METAGROSS_CYCLE18_EXECUTE:-}" != "FROZEN_20_GAME_CANARY" ]]; then
  echo "set METAGROSS_CYCLE18_EXECUTE=FROZEN_20_GAME_CANARY" >&2; exit 2
fi
[[ -f "$MANIFEST" && -f "$PAIR_MANIFEST" ]] || { echo "frozen artifacts missing" >&2; exit 2; }
PYTHONPATH="$ROOT" "$ROOT/.venv-metamon/bin/python" \
  "$ROOT/experimental/src/scripts/verify_cycle18_h2h_freeze.py" "$MANIFEST" >/dev/null
[[ ! -e "$RUN/preflight-result.json" && ! -e "$RUN/prior-a.jsonl" && ! -e "$RUN/prior-b.jsonl" ]] || {
  echo "Cycle18 outcome files already exist" >&2; exit 2;
}
for port in "$BASE_PORT" "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null && { echo "port occupied: $port" >&2; exit 2; }
done
PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait "${PIDS[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

env CUDA_VISIBLE_DEVICES='' "$ROOT/experimental/src/scripts/start_showdown.sh" "$BASE_PORT" \
  >>"$RUN/showdown.log" 2>&1 & PIDS+=("$!")
for side in a b; do
  if [[ "$side" == a ]]; then port="$PRIOR_A_PORT"; else port="$PRIOR_B_PORT"; fi
  env METAMON_CACHE_DIR="$ROOT/external/metamon_cache" TORCHDYNAMO_DISABLE=1 \
    ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' \
    "$ROOT/.venv-metamon/bin/python" -u "$ROOT/srcs/metagross/prior_server.py" \
    --local-run-dir "$ROOT/srcs/models" --local-run-name randbats_exit_r1 \
    --checkpoint 5 --checkpoint-sha256 "$CHECKPOINT_SHA256" \
    --trajectory-mode causal-history --username "c18${side}" --port "$port" \
    --decision-dump "$RUN/prior-${side}.jsonl" >>"$RUN/prior-${side}.log" 2>&1 &
  PIDS+=("$!")
done
for _ in {1..180}; do
  ready=1
  for port in "$BASE_PORT" "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null || ready=0
  done
  ((ready == 1)) && break
  sleep 1
done
for port in "$BASE_PORT" "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null || { echo "service failed readiness" >&2; exit 1; }
done

env METAMON_CACHE_DIR="$ROOT/external/metamon_cache" TORCHDYNAMO_DISABLE=1 \
  ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' \
  METAGROSS_TERMINAL_MCTS_TEACHER_SLOT=agent_a \
  METAGROSS_TERMINAL_MCTS_MODE=cycle18_equal8192 \
  METAGROSS_TERMINAL_MCTS_PYTHON="$ROOT/.venv-metamon/bin/python" \
  METAGROSS_TERMINAL_MCTS_SCRIPT="$ROOT/experimental/src/scripts/cycle18_equal8192_live_decision.py" \
  METAGROSS_TERMINAL_MCTS_PYTHONPATH="$ROOT/experimental/runs/search_native_v2_cycle17_teacher_stability_20260815/engine-binding/unpacked:$ROOT/experimental/src:$ROOT" \
  METAGROSS_TERMINAL_MCTS_WORKERS=8 METAGROSS_TERMINAL_MCTS_TIMEOUT_SECONDS=30 \
  PYTHONPATH="$ROOT/experimental/src:$ROOT" \
  "$ROOT/.venv-metamon/bin/python" -u "$ROOT/experimental/src/eval/run.py" \
  --mode h2h --server local --format gen9randombattle \
  --websocket-uri "ws://127.0.0.1:${BASE_PORT}/showdown/websocket" \
  --paired --mirrored-pairs --mirror-seed "$MIRROR_SEED" \
  --showdown-dir "$ROOT/external/pokemon-showdown" \
  --mirrored-team-generator "$ROOT/experimental/src/scripts/generate_mirrored_randbats_pair.cjs" \
  --pair-registration-dir "$RUN/registrations" \
  --agent-a production_r1_search_first --agent-b production_r1_search_first \
  --agent-a-prior-server-url "http://127.0.0.1:${PRIOR_A_PORT}" \
  --agent-b-prior-server-url "http://127.0.0.1:${PRIOR_B_PORT}" \
  --agent-a-require-priors --agent-b-require-priors --strict-isolated-priors \
  --agent-a-decision-log "$RUN/agent-a-decisions.jsonl" \
  --agent-b-decision-log "$RUN/agent-b-decisions.jsonl" \
  --foul-play-python "$ROOT/.venv-foul-play/bin/python" \
  --foul-play-search-time-ms 500 --foul-play-search-parallelism 8 \
  --foul-play-search-threads 1 --cpuct 2.0 \
  --production-run-seed "$PRODUCTION_SEED" --concurrent-games 1 --fail-fast \
  --game-timeout-seconds 900 --n-games 20 --username-prefix "$USERNAME_PREFIX" \
  --run-id "$RUN_ID" --pair-manifest-sha256 "$PAIR_SHA" \
  --json-out "$RUN/preflight-result.json" --log-dir "$RUN/preflight-logs"
