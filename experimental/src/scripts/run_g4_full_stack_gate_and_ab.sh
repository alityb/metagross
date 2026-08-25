#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/experimental/runs/g4_full_stack_20260730}"
SHOWDOWN_PORT="${SHOWDOWN_PORT:-8012}"
PRIOR_A_PORT="${PRIOR_A_PORT:-8997}"
PRIOR_B_PORT="${PRIOR_B_PORT:-8998}"
PYTHON="$ROOT/.venv-metamon/bin/python"
FP_PYTHON="$ROOT/.venv-fp-priors/bin/python"
R1_RUN="randbats_exit_r1"
R1_CHECKPOINT="5"
R1_SHA="c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
G4_RUN="randbats_online_g4_autonomous_freshfix_20260729"
G4_CHECKPOINT="1"
G4_SHA="cd21dbb22bfc7a92836c7c75c6287ffc1b79c58e0a2dca3d402b76b001ff7505"

mkdir -p "$RUN_ROOT/registrations" "$RUN_ROOT/scorer/logs" "$RUN_ROOT/ab/logs"
mkdir -p \
  "$ROOT/external/pokemon-showdown/databases" \
  "$ROOT/external/pokemon-showdown/logs/repl" \
  "$ROOT/external/pokemon-showdown/logs/modlog" \
  "$ROOT/external/pokemon-showdown/logs/ladder"

sha256() {
  shasum -a 256 "$1" | cut -d ' ' -f 1
}

R1_PATH="$ROOT/srcs/models/$R1_RUN/ckpts/policy_weights/policy_epoch_${R1_CHECKPOINT}.pt"
G4_PATH="$ROOT/srcs/models/$G4_RUN/ckpts/policy_weights/policy_epoch_${G4_CHECKPOINT}.pt"
[[ "$(sha256 "$R1_PATH")" == "$R1_SHA" ]]
[[ "$(sha256 "$G4_PATH")" == "$G4_SHA" ]]

unset METAGROSS_VALUE_MODEL METAGROSS_RANDBATS_POOL
unset METAGROSS_ACTION_CONDITIONED_BELIEF METAGROSS_SHARED_ROOT_SEARCH
unset METAGROSS_OPP_PRIORS_ONLY METAGROSS_PRIOR_NAMESPACE METAGROSS_PRIOR_DUMP

showdown_pid=""
prior_a_pid=""
prior_b_pid=""

cleanup() {
  for pid in "$prior_a_pid" "$prior_b_pid" "$showdown_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

METAGROSS_EVAL_PAIR_DIR="$RUN_ROOT/registrations" \
SHOWDOWN_DIR="$ROOT/external/pokemon-showdown" \
bash "$ROOT/experimental/src/scripts/start_showdown.sh" "$SHOWDOWN_PORT" \
  >"$RUN_ROOT/showdown.log" 2>&1 &
showdown_pid=$!

start_prior_a() {
  local run_name="$1"
  local checkpoint="$2"
  local label="$3"
  METAMON_CACHE_DIR="$ROOT/srcs/runtime/metamon-cache" \
  TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true WANDB_MODE=disabled \
  "$PYTHON" -u "$ROOT/experimental/src/scripts/prior_server.py" \
    --local-run-dir "$ROOT/srcs/models" \
    --local-run-name "$run_name" \
    --local-base-model Kakuna \
    --checkpoint "$checkpoint" \
    --username "$label" \
    --host 127.0.0.1 \
    --port "$PRIOR_A_PORT" \
    >"$RUN_ROOT/prior-a-${label}.log" 2>&1 &
  prior_a_pid=$!
}

start_prior_a "$R1_RUN" "$R1_CHECKPOINT" r1a
METAMON_CACHE_DIR="$ROOT/srcs/runtime/metamon-cache" \
TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true WANDB_MODE=disabled \
"$PYTHON" -u "$ROOT/experimental/src/scripts/prior_server.py" \
  --local-run-dir "$ROOT/srcs/models" \
  --local-run-name "$R1_RUN" \
  --local-base-model Kakuna \
  --checkpoint "$R1_CHECKPOINT" \
  --username r1b \
  --host 127.0.0.1 \
  --port "$PRIOR_B_PORT" \
  >"$RUN_ROOT/prior-b-r1.log" 2>&1 &
prior_b_pid=$!

for url in \
  "http://127.0.0.1:$PRIOR_A_PORT/health" \
  "http://127.0.0.1:$PRIOR_B_PORT/health"; do
  ready=0
  for _ in $(seq 1 240); do
    if curl --silent --fail "$url" >/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  [[ "$ready" == "1" ]]
done

COMMON_ARGS=(
  --mode h2h
  --server local
  --format gen9randombattle
  --websocket-uri "ws://localhost:$SHOWDOWN_PORT/showdown/websocket"
  --paired
  --mirrored-pairs
  --showdown-dir "$ROOT/external/pokemon-showdown"
  --mirrored-team-generator "$ROOT/experimental/src/scripts/generate_mirrored_randbats_pair.cjs"
  --pair-registration-dir "$RUN_ROOT/registrations"
  --agent-a foul_play_root_priors_opp
  --agent-b foul_play_root_priors_opp
  --agent-a-prior-server-url "http://127.0.0.1:$PRIOR_A_PORT"
  --agent-b-prior-server-url "http://127.0.0.1:$PRIOR_B_PORT"
  --agent-a-require-priors
  --agent-b-require-priors
  --strict-isolated-priors
  --foul-play-python "$FP_PYTHON"
  --foul-play-search-time-ms 500
  --foul-play-search-parallelism 8
  --foul-play-search-threads 1
  --cpuct 2.0
  --concurrent-games 1
  --fail-fast
  --game-timeout-seconds 900
)

"$PYTHON" "$ROOT/experimental/src/eval/run.py" \
  "${COMMON_ARGS[@]}" \
  --n-games 200 \
  --mirror-seed 2026073001 \
  --username-prefix r1gate \
  --run-id r1_full_stack_scorer_gate_200_20260730 \
  --scorer-gate \
  --json-out "$RUN_ROOT/scorer/result.json" \
  --log-dir "$RUN_ROOT/scorer/logs"

"$PYTHON" -c '
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))["summary"]
if not summary["scorer_gate_passed"]:
    raise SystemExit(f"scorer gate failed: {summary['"'"'scorer_gate_message'"'"']}")
' "$RUN_ROOT/scorer/result.json"

kill "$prior_a_pid"
wait "$prior_a_pid" 2>/dev/null || true
prior_a_pid=""
start_prior_a "$G4_RUN" "$G4_CHECKPOINT" g4a

ready=0
for _ in $(seq 1 240); do
  if curl --silent --fail "http://127.0.0.1:$PRIOR_A_PORT/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
[[ "$ready" == "1" ]]

"$PYTHON" "$ROOT/experimental/src/eval/run.py" \
  "${COMMON_ARGS[@]}" \
  --n-games 1000 \
  --mirror-seed 2026073002 \
  --username-prefix g4r1 \
  --run-id g4_vs_r1_full_stack_mirrored_1000_20260730 \
  --json-out "$RUN_ROOT/ab/result.json" \
  --log-dir "$RUN_ROOT/ab/logs"
