#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN="$ROOT/experimental/runs/search_native_v2_cycle20_ability_lineage_20260815"
MANIFEST="$RUN/PRESMOKE_MANIFEST.json"
PAIR="$RUN/smoke-result.json.pairs.json"
PAIR_SHA="356174a7f863164733ea44f5062b00337c9c905785a779ae41532607ff667d4c"
ENGINE_ROOT="$ROOT/experimental/runs/search_native_v2_cycle17_teacher_stability_20260815/engine-binding/unpacked"
ENGINE_SHA="ece46434a7bd6dc831b4737c9abecc05918b9c188a2f64c7cb69e8a30a6b41e0"
BASE_PORT=8898
PRIOR_A_PORT=9185
PRIOR_B_PORT=9186
CHECKPOINT_SHA256=c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93
PIDS=()

if [[ "${METAGROSS_CYCLE20_EXECUTE:-}" != "FROZEN_FORM_TRANSITION_SMOKE" ]]; then
  echo "set METAGROSS_CYCLE20_EXECUTE=FROZEN_FORM_TRANSITION_SMOKE" >&2
  exit 2
fi
PYTHONPATH="$ROOT/experimental/src:$ROOT" "$ROOT/.venv-metamon/bin/python" \
  "$ROOT/experimental/src/scripts/verify_cycle20_presmoke.py" "$MANIFEST" >/dev/null
[[ "$(shasum -a 256 "$PAIR" | awk '{print $1}')" == "$PAIR_SHA" ]] || exit 2
[[ ! -e "$RUN/SMOKE_RESULT.json" && ! -e "$RUN/prior-a.jsonl" && ! -e "$RUN/prior-b.jsonl" ]] || {
  echo "Cycle20 smoke outputs already exist" >&2
  exit 2
}
for port in "$BASE_PORT" "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null && { echo "occupied port $port" >&2; exit 2; }
done

cleanup() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait "${PIDS[@]:-}" 2>/dev/null || true
  while read -r pid; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done < <(pgrep -f 'srcs.metagross.run_foul_play.*c20smk' || true)
}
trap cleanup EXIT INT TERM

env CUDA_VISIBLE_DEVICES='' "$ROOT/experimental/src/scripts/start_showdown.sh" "$BASE_PORT" \
  >>"$RUN/smoke-showdown.log" 2>&1 & PIDS+=("$!")
for side in a b; do
  if [[ "$side" == a ]]; then port="$PRIOR_A_PORT"; else port="$PRIOR_B_PORT"; fi
  env METAMON_CACHE_DIR="$ROOT/external/metamon_cache" TORCHDYNAMO_DISABLE=1 \
    ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' \
    "$ROOT/.venv-metamon/bin/python" -u "$ROOT/srcs/metagross/prior_server.py" \
    --local-run-dir "$ROOT/srcs/models" --local-run-name randbats_exit_r1 \
    --checkpoint 5 --checkpoint-sha256 "$CHECKPOINT_SHA256" \
    --trajectory-mode causal-history --username "c20smk${side}" --port "$port" \
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
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null || { echo "service readiness failed" >&2; exit 1; }
done

env METAMON_CACHE_DIR="$ROOT/external/metamon_cache" TORCHDYNAMO_DISABLE=1 \
  ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' \
  METAGROSS_PINNED_ENGINE_IMPORT_ROOT="$ENGINE_ROOT" \
  METAGROSS_PINNED_ENGINE_SHA256="$ENGINE_SHA" \
  METAGROSS_PINNED_ENGINE_RECEIPT_DIR="$RUN/engine-receipts" \
  METAGROSS_TERMINAL_MCTS_TEACHER_SLOT=agent_a \
  METAGROSS_TERMINAL_MCTS_MODE=cycle19_equal8192 \
  METAGROSS_TERMINAL_MCTS_PYTHON="$ROOT/.venv-metamon/bin/python" \
  METAGROSS_TERMINAL_MCTS_SCRIPT="$ROOT/experimental/src/scripts/cycle19_equal8192_live_decision.py" \
  METAGROSS_TERMINAL_MCTS_PYTHONPATH="$ENGINE_ROOT:$ROOT/experimental/src:$ROOT" \
  METAGROSS_TERMINAL_MCTS_WORKERS=8 METAGROSS_TERMINAL_MCTS_TIMEOUT_SECONDS=30 \
  PYTHONPATH="$ENGINE_ROOT:$ROOT/experimental/src:$ROOT" \
  "$ROOT/.venv-metamon/bin/python" -u "$ROOT/experimental/src/eval/run.py" \
  --mode h2h --server local --format gen9randombattle \
  --websocket-uri "ws://127.0.0.1:${BASE_PORT}/showdown/websocket" \
  --paired --mirrored-pairs --mirror-seed 202620000142 \
  --showdown-dir "$ROOT/external/pokemon-showdown" \
  --mirrored-team-generator "$ROOT/experimental/src/scripts/generate_mirrored_randbats_pair.cjs" \
  --pair-registration-dir "$RUN/smoke-registrations" \
  --agent-a production_r1_search_first --agent-b production_r1_search_first \
  --agent-a-prior-server-url "http://127.0.0.1:${PRIOR_A_PORT}" \
  --agent-b-prior-server-url "http://127.0.0.1:${PRIOR_B_PORT}" \
  --agent-a-require-priors --agent-b-require-priors --strict-isolated-priors \
  --foul-play-python "$ROOT/.venv-foul-play/bin/python" \
  --foul-play-search-time-ms 500 --foul-play-search-parallelism 8 \
  --foul-play-search-threads 1 --cpuct 2.0 \
  --production-run-seed 2020202020202020202020202020202020202020202020202020202020202020 \
  --concurrent-games 1 --fail-fast --game-timeout-seconds 900 --n-games 2 \
  --username-prefix c20smk --run-id cycle20-ability-lineage-smoke \
  --pair-manifest-sha256 "$PAIR_SHA" --json-out "$RUN/smoke-result.json" \
  --log-dir "$RUN/smoke-logs" >>"$RUN/smoke-runner.log" 2>&1 &
EVAL_PID="$!"
PIDS+=("$EVAL_PID")

PYTHONPATH="$ENGINE_ROOT:$ROOT/srcs/vendor/foul-play:$ROOT/experimental/src:$ROOT" \
  "$ROOT/.venv-metamon/bin/python" \
  "$ROOT/experimental/src/scripts/monitor_cycle20_form_smoke.py" \
  --run "$RUN" --expected-engine-sha256 "$ENGINE_SHA" --timeout-seconds 300 \
  --output "$RUN/SMOKE_RESULT.json"

kill "$EVAL_PID" 2>/dev/null || true
wait "$EVAL_PID" 2>/dev/null || true
