#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN="$ROOT/experimental/runs/search_native_v2_cycle30_dynamic_boundary_20260815"
MANIFEST="$RUN/PRESMOKE_MANIFEST.json"
CANONICAL="$RUN/CANONICAL_SMOKE_ARGV.json"
PAIR="$RUN/smoke-result.json.pairs.json"
PAIR_SHA="e8ec95f59a24ed92a3f95a500123c7f00390a06742abcb984deaf90fa037e49b"
ENGINE_ROOT="$ROOT/experimental/runs/search_native_v2_cycle22_certified_ability_install_20260815/engine-binding/unpacked"
ENGINE_SHA="c8ba2bdf6854c943b60c2f35d0f9869895222aa77f85b9eea15507bb3b144055"
PRIOR_A_PORT=9535
PRIOR_B_PORT=9536
CHECKPOINT_SHA256=c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93
PIDS=()

if [[ "${METAGROSS_CYCLE30_EXECUTE:-}" != "FROZEN_DYNAMIC_CAUSAL_BOUNDARY_SMOKE" ]]; then
  echo "set METAGROSS_CYCLE30_EXECUTE=FROZEN_DYNAMIC_CAUSAL_BOUNDARY_SMOKE" >&2; exit 2
fi
PYTHONPATH="$ROOT/experimental/src:$ROOT" "$ROOT/.venv-metamon/bin/python" \
  "$ROOT/experimental/src/scripts/verify_cycle30_presmoke.py" "$MANIFEST" >/dev/null
PYTHONPATH="$ROOT/experimental/src:$ROOT" "$ROOT/.venv-metamon/bin/python" \
  "$ROOT/experimental/src/scripts/cycle30_canonical_smoke.py" --canonical "$CANONICAL" \
  --phase verify --pair-manifest "$PAIR" --pair-sha256 "$PAIR_SHA" >/dev/null
[[ ! -e "$RUN/SMOKE_RESULT.json" && ! -e "$RUN/REGISTRATION_CONSUMPTION.json" \
   && ! -e "$RUN/SHOWDOWN_LAUNCH.json" && ! -e "$RUN/smoke-showdown.log" \
   && ! -e "$RUN/PUBLIC_EXECUTION_BOUNDARY.json" ]] || {
  echo "Cycle30 live outputs already exist" >&2; exit 2;
}
for port in 8010 "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null && { echo "occupied port $port" >&2; exit 2; }
done
cleanup() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait "${PIDS[@]:-}" 2>/dev/null || true
  while read -r pid; do [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true; done \
    < <(pgrep -f 'srcs.metagross.run_foul_play.*c30smk' || true)
}
trap cleanup EXIT INT TERM

PYTHONPATH="$ROOT/experimental/src:$ROOT" "$ROOT/.venv-metamon/bin/python" -u \
  "$ROOT/srcs/metagross/showdown_runtime_server.py" \
  --runtime-manifest "$RUN/SHOWDOWN_RUNTIME_MANIFEST.json" \
  --pair-directory "$RUN/smoke-registrations" \
  --launch-record "$RUN/SHOWDOWN_LAUNCH.json" --server-log "$RUN/smoke-showdown.log" \
  >>"$RUN/showdown-supervisor.log" 2>&1 & PIDS+=("$!")
for side in a b; do
  if [[ "$side" == a ]]; then port="$PRIOR_A_PORT"; else port="$PRIOR_B_PORT"; fi
  env METAMON_CACHE_DIR="$ROOT/external/metamon_cache" TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' \
    "$ROOT/.venv-metamon/bin/python" -u "$ROOT/srcs/metagross/prior_server.py" \
    --local-run-dir "$ROOT/srcs/models" --local-run-name randbats_exit_r1 \
    --checkpoint 5 --checkpoint-sha256 "$CHECKPOINT_SHA256" --trajectory-mode causal-history \
    --username "c30smk${side}" --port "$port" --decision-dump "$RUN/prior-${side}.jsonl" \
    >>"$RUN/prior-${side}.log" 2>&1 & PIDS+=("$!")
done
for _ in {1..180}; do
  ready=1
  for port in 8010 "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null || ready=0; done
  ((ready == 1)) && [[ -e "$RUN/SHOWDOWN_LAUNCH.json" ]] && break
  sleep 1
done
for port in 8010 "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null || { echo "service readiness failed" >&2; exit 1; }
done

PYTHONPATH="$ROOT/experimental/src:$ROOT" "$ROOT/.venv-metamon/bin/python" -u \
  "$ROOT/experimental/src/scripts/watch_cycle30_registrations.py" \
  --directory "$RUN/smoke-registrations" --pair-manifest "$PAIR" --timeout-seconds 600 \
  --output "$RUN/REGISTRATION_CONSUMPTION.json" >>"$RUN/registration-watcher.log" 2>&1 &
WATCH_PID="$!"; PIDS+=("$WATCH_PID")

env METAGROSS_CYCLE30_BOUNDARY_RECEIPT="$RUN/PUBLIC_EXECUTION_BOUNDARY.json" \
  METAGROSS_CYCLE30_MAX_DECISION_INDEX=5 METAGROSS_CYCLE30_MAX_BATTLE_TURN=6 \
  METAGROSS_CAUSAL_MOVE_RECEIPT_DIR="$RUN/move-receipts" \
  METAMON_CACHE_DIR="$ROOT/external/metamon_cache" TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' \
  METAGROSS_PINNED_ENGINE_IMPORT_ROOT="$ENGINE_ROOT" METAGROSS_PINNED_ENGINE_SHA256="$ENGINE_SHA" \
  METAGROSS_PINNED_ENGINE_RECEIPT_DIR="$RUN/engine-receipts" METAGROSS_TERMINAL_MCTS_TEACHER_SLOT=agent_a \
  METAGROSS_TERMINAL_MCTS_MODE=cycle19_equal8192 METAGROSS_TERMINAL_MCTS_PYTHON="$ROOT/.venv-metamon/bin/python" \
  METAGROSS_TERMINAL_MCTS_SCRIPT="$ROOT/experimental/src/scripts/cycle19_equal8192_live_decision.py" \
  METAGROSS_TERMINAL_MCTS_PYTHONPATH="$ENGINE_ROOT:$ROOT/experimental/src:$ROOT" \
  METAGROSS_TERMINAL_MCTS_WORKERS=8 METAGROSS_TERMINAL_MCTS_TIMEOUT_SECONDS=30 \
  PYTHONPATH="$ENGINE_ROOT:$ROOT/experimental/src:$ROOT" "$ROOT/.venv-metamon/bin/python" -u \
  "$ROOT/experimental/src/scripts/cycle30_canonical_smoke.py" --canonical "$CANONICAL" \
  --phase live --pair-sha256 "$PAIR_SHA" >>"$RUN/smoke-runner.log" 2>&1 &
EVAL_PID="$!"; PIDS+=("$EVAL_PID")

PYTHONPATH="$ENGINE_ROOT:$ROOT/srcs/vendor/foul-play:$ROOT/experimental/src:$ROOT" \
  "$ROOT/.venv-metamon/bin/python" "$ROOT/experimental/src/scripts/monitor_cycle30_dynamic_boundary_smoke.py" \
  --run "$RUN" --pair-manifest "$PAIR" --expected-engine-sha256 "$ENGINE_SHA" \
  --timeout-seconds 600 --output "$RUN/SMOKE_RESULT.json"
kill "$EVAL_PID" 2>/dev/null || true
wait "$EVAL_PID" 2>/dev/null || true
wait "$WATCH_PID"
