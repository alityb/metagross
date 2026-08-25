#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN="$ROOT/experimental/runs/search_native_v2_cycle40_integrated_h2h_20260816"
MANIFEST="$RUN/H2H_PREMEASUREMENT_MANIFEST.json"
CANONICAL="$RUN/CANONICAL_H2H_ARGV.json"
PAIR="$RUN/h2h-result.json.pairs.json"
PAIR_SHA="$(shasum -a 256 "$PAIR" | awk '{print $1}')"
RUNTIME_MANIFEST="$ROOT/experimental/runs/search_native_v2_cycle30_dynamic_boundary_20260815/SHOWDOWN_RUNTIME_MANIFEST.json"
ENGINE_ROOT="$ROOT/experimental/runs/search_native_v2_cycle22_certified_ability_install_20260815/engine-binding/unpacked"
ENGINE_SHA="c8ba2bdf6854c943b60c2f35d0f9869895222aa77f85b9eea15507bb3b144055"
PRIOR_A_PORT=9741
PRIOR_B_PORT=9742
CHECKPOINT_SHA256=c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93
PIDS=()

[[ "${METAGROSS_CYCLE40_EXECUTE:-}" == "FROZEN_20_GAME_GATE" ]] || {
  echo "set METAGROSS_CYCLE40_EXECUTE=FROZEN_20_GAME_GATE" >&2
  exit 2
}
PYTHONPATH="$ROOT/experimental/src:$ROOT" "$ROOT/.venv-metamon/bin/python" \
  "$ROOT/experimental/src/scripts/verify_cycle33_h2h_freeze.py" "$MANIFEST" >/dev/null
PYTHONPATH="$ROOT/experimental/src:$ROOT" "$ROOT/.venv-metamon/bin/python" \
  "$ROOT/experimental/src/scripts/cycle33_canonical_h2h.py" --canonical "$CANONICAL" \
  --phase verify --pair-manifest "$PAIR" --pair-sha256 "$PAIR_SHA" >/dev/null
[[ ! -e "$RUN/h2h-result.json" && ! -e "$RUN/H2H_RESULT_REPORT.json" \
   && ! -e "$RUN/REGISTRATION_CONSUMPTION.json" && ! -e "$RUN/SHOWDOWN_LAUNCH.json" ]] || {
  echo "Cycle40 scored outputs already exist" >&2
  exit 2
}
for port in 8010 "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null && {
    echo "occupied port $port" >&2
    exit 2
  }
done
cleanup() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait "${PIDS[@]:-}" 2>/dev/null || true
  while read -r pid; do [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true; done \
    < <(pgrep -f 'srcs.metagross.run_foul_play.*c40h2h' || true)
}
trap cleanup EXIT INT TERM

PYTHONPATH="$ROOT/experimental/src:$ROOT" "$ROOT/.venv-metamon/bin/python" -u \
  "$ROOT/srcs/metagross/showdown_runtime_server.py" --runtime-manifest "$RUNTIME_MANIFEST" \
  --pair-directory "$RUN/h2h-registrations" --launch-record "$RUN/SHOWDOWN_LAUNCH.json" \
  --server-log "$RUN/h2h-showdown.log" >>"$RUN/showdown-supervisor.log" 2>&1 &
PIDS+=("$!")
for side in a b; do
  [[ "$side" == a ]] && port="$PRIOR_A_PORT" || port="$PRIOR_B_PORT"
  env METAMON_CACHE_DIR="$ROOT/external/metamon_cache" TORCHDYNAMO_DISABLE=1 \
    ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' "$ROOT/.venv-metamon/bin/python" -u \
    "$ROOT/srcs/metagross/prior_server.py" --local-run-dir "$ROOT/srcs/models" \
    --local-run-name randbats_exit_r1 --checkpoint 5 --checkpoint-sha256 "$CHECKPOINT_SHA256" \
    --trajectory-mode causal-history --username "c40h2h${side}" --port "$port" \
    --decision-dump "$RUN/h2h-prior-${side}.jsonl" >>"$RUN/h2h-prior-${side}.log" 2>&1 &
  PIDS+=("$!")
done
for _ in {1..180}; do
  ready=1
  for port in 8010 "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null || ready=0
  done
  ((ready == 1)) && [[ -e "$RUN/SHOWDOWN_LAUNCH.json" ]] && break
  sleep 1
done
for port in 8010 "$PRIOR_A_PORT" "$PRIOR_B_PORT"; do
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null || {
    echo "service readiness failed" >&2
    exit 1
  }
done

PYTHONPATH="$ROOT/experimental/src:$ROOT" "$ROOT/.venv-metamon/bin/python" -u \
  "$ROOT/experimental/src/scripts/watch_cycle40_registrations.py" \
  --directory "$RUN/h2h-registrations" --pair-manifest "$PAIR" --timeout-seconds 7200 \
  --output "$RUN/REGISTRATION_CONSUMPTION.json" >>"$RUN/registration-watcher.log" 2>&1 &
WATCH_PID="$!"
PIDS+=("$WATCH_PID")

env METAGROSS_CAUSAL_MOVE_RECEIPT_DIR="$RUN/move-receipts" \
  METAGROSS_CAUSAL_ABILITY_RECEIPT_DIR="$RUN/ability-receipts" \
  METAMON_CACHE_DIR="$ROOT/external/metamon_cache" TORCHDYNAMO_DISABLE=1 \
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
  "$ROOT/.venv-metamon/bin/python" -u \
  "$ROOT/experimental/src/scripts/cycle33_canonical_h2h.py" --canonical "$CANONICAL" \
  --phase live --pair-sha256 "$PAIR_SHA" >>"$RUN/h2h-runner.log" 2>&1
wait "$WATCH_PID"
PYTHONPATH="$ENGINE_ROOT:$ROOT/srcs/vendor/foul-play:$ROOT/experimental/src:$ROOT" \
  "$ROOT/.venv-metamon/bin/python" "$ROOT/experimental/src/scripts/summarize_cycle40_h2h.py" \
  --run "$RUN" --manifest "$MANIFEST" --output "$RUN/H2H_RESULT_REPORT.json"
