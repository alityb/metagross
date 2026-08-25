#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/experimental/runs/g4_counterbalanced_pilot_20260730}"
SHOWDOWN_PORT="${SHOWDOWN_PORT:-8012}"
G4_PRIOR_PORT="${G4_PRIOR_PORT:-8997}"
R1_PRIOR_PORT="${R1_PRIOR_PORT:-8998}"
GAMES_PER_ORIENTATION="${GAMES_PER_ORIENTATION:-50}"
MIRROR_SEED="${MIRROR_SEED:-2026073004}"
PYTHON="$ROOT/.venv-metamon/bin/python"
FP_PYTHON="$ROOT/.venv-fp-priors/bin/python"
R1_RUN="randbats_exit_r1"
R1_CHECKPOINT="5"
R1_SHA="c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
G4_RUN="randbats_online_g4_autonomous_freshfix_20260729"
G4_CHECKPOINT="1"
G4_SHA="cd21dbb22bfc7a92836c7c75c6287ffc1b79c58e0a2dca3d402b76b001ff7505"

if (( GAMES_PER_ORIENTATION <= 0 || GAMES_PER_ORIENTATION % 2 != 0 )); then
  echo "GAMES_PER_ORIENTATION must be a positive even integer" >&2
  exit 2
fi

mkdir -p \
  "$RUN_ROOT/registrations" \
  "$RUN_ROOT/g4-as-a/logs" \
  "$RUN_ROOT/g4-as-b/logs" \
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
g4_prior_pid=""
r1_prior_pid=""

cleanup() {
  for pid in "$g4_prior_pid" "$r1_prior_pid" "$showdown_pid"; do
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

start_prior() {
  local run_name="$1"
  local checkpoint="$2"
  local label="$3"
  local port="$4"
  local log="$5"
  METAMON_CACHE_DIR="$ROOT/srcs/runtime/metamon-cache" \
  TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true WANDB_MODE=disabled \
  "$PYTHON" -u "$ROOT/experimental/src/scripts/prior_server.py" \
    --local-run-dir "$ROOT/srcs/models" \
    --local-run-name "$run_name" \
    --local-base-model Kakuna \
    --checkpoint "$checkpoint" \
    --username "$label" \
    --host 127.0.0.1 \
    --port "$port" \
    >"$log" 2>&1 &
  STARTED_PRIOR_PID=$!
}

start_prior "$G4_RUN" "$G4_CHECKPOINT" g4 "$G4_PRIOR_PORT" "$RUN_ROOT/prior-g4.log"
g4_prior_pid=$STARTED_PRIOR_PID
start_prior "$R1_RUN" "$R1_CHECKPOINT" r1 "$R1_PRIOR_PORT" "$RUN_ROOT/prior-r1.log"
r1_prior_pid=$STARTED_PRIOR_PID

for url in \
  "http://127.0.0.1:$G4_PRIOR_PORT/health" \
  "http://127.0.0.1:$R1_PRIOR_PORT/health"; do
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
  --n-games "$GAMES_PER_ORIENTATION"
  --mirror-seed "$MIRROR_SEED"
)

run_orientation() {
  local orientation="$1"
  local agent_a_url="$2"
  local agent_b_url="$3"
  local username_prefix="$4"
  local output="$RUN_ROOT/$orientation/result.json"
  local resume=()
  if [[ -f "$output" ]]; then
    return
  fi
  if [[ -f "$output.progress.json" ]]; then
    resume=(--resume)
  fi
  "$PYTHON" "$ROOT/experimental/src/eval/run.py" \
    "${COMMON_ARGS[@]}" \
    --agent-a-prior-server-url "$agent_a_url" \
    --agent-b-prior-server-url "$agent_b_url" \
    --username-prefix "$username_prefix" \
    --run-id "g4_counterbalanced_${orientation}_20260730" \
    --json-out "$output" \
    --log-dir "$RUN_ROOT/$orientation/logs" \
    "${resume[@]}"
}

run_orientation \
  g4-as-a \
  "http://127.0.0.1:$G4_PRIOR_PORT" \
  "http://127.0.0.1:$R1_PRIOR_PORT" \
  cba
run_orientation \
  g4-as-b \
  "http://127.0.0.1:$R1_PRIOR_PORT" \
  "http://127.0.0.1:$G4_PRIOR_PORT" \
  cbb

"$PYTHON" "$ROOT/experimental/src/scripts/summarize_counterbalanced_ab.py" \
  --g4-as-a "$RUN_ROOT/g4-as-a/result.json" \
  --g4-as-b "$RUN_ROOT/g4-as-b/result.json" \
  --games-per-orientation "$GAMES_PER_ORIENTATION" \
  --out "$RUN_ROOT/SUMMARY.json"
