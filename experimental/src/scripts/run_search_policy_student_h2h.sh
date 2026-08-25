#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/experimental/runs/search_policy_h2h_20260813/powered_pipeline}"
SHOWDOWN_PORT="${SHOWDOWN_PORT:-8013}"
PRIOR_A_PORT="${PRIOR_A_PORT:-9001}"
PRIOR_B_PORT="${PRIOR_B_PORT:-9002}"
SCREEN_GAMES_PER_ORIENTATION="${SCREEN_GAMES_PER_ORIENTATION:-10}"
GATE_GAMES_PER_ORIENTATION="${GATE_GAMES_PER_ORIENTATION:-250}"
PYTHON="$ROOT/.venv-metamon/bin/python"
FP_PYTHON="$ROOT/.venv-foul-play/bin/python"
POLICY_ROOT="$ROOT/experimental/releases/search_policy_student"
R1_ROOT="$ROOT/srcs/models"

ACTION_RUN="search_policy_action_1k_seed20260812"
ACTION_SHA="efeda02ece3c4fd2f28172450b3ea0d6488bc33447308a7d0140e24962422d6f"
VISITS_RUN="search_policy_visits_1k_seed20260812"
VISITS_SHA="a2b27bebcabea97a9ec01a84a3f371c7d7902e64b1eb302668a05f27b39d3c34"
R1_RUN="randbats_exit_r1"
R1_CHECKPOINT="5"
R1_SHA="c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
PRODUCTION_RUN_SEED="2026081320260813202608132026081320260813202608132026081320260813"

for games in "$SCREEN_GAMES_PER_ORIENTATION" "$GATE_GAMES_PER_ORIENTATION"; do
  if (( games <= 0 || games % 2 != 0 )); then
    echo "games per orientation must be a positive even integer" >&2
    exit 2
  fi
done

mkdir -p \
  "$RUN_ROOT/registrations" \
  "$RUN_ROOT/screen/action-as-a/logs" \
  "$RUN_ROOT/screen/action-as-b/logs" \
  "$RUN_ROOT/gate/candidate-as-a/logs" \
  "$RUN_ROOT/gate/candidate-as-b/logs" \
  "$ROOT/external/pokemon-showdown/databases" \
  "$ROOT/external/pokemon-showdown/logs/repl" \
  "$ROOT/external/pokemon-showdown/logs/modlog" \
  "$ROOT/external/pokemon-showdown/logs/ladder"

sha256() {
  shasum -a 256 "$1" | cut -d ' ' -f 1
}

checkpoint_path() {
  local root="$1"
  local run_name="$2"
  local checkpoint="$3"
  echo "$root/$run_name/ckpts/policy_weights/policy_epoch_${checkpoint}.pt"
}

[[ "$(sha256 "$(checkpoint_path "$POLICY_ROOT" "$ACTION_RUN" 1)")" == "$ACTION_SHA" ]]
[[ "$(sha256 "$(checkpoint_path "$POLICY_ROOT" "$VISITS_RUN" 1)")" == "$VISITS_SHA" ]]
[[ "$(sha256 "$(checkpoint_path "$R1_ROOT" "$R1_RUN" "$R1_CHECKPOINT")")" == "$R1_SHA" ]]

showdown_pid=""
prior_a_pid=""
prior_b_pid=""

stop_pid() {
  local pid="$1"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  stop_pid "$prior_a_pid"
  stop_pid "$prior_b_pid"
  stop_pid "$showdown_pid"
}
trap cleanup EXIT INT TERM

METAGROSS_EVAL_PAIR_DIR="$RUN_ROOT/registrations" \
SHOWDOWN_DIR="$ROOT/external/pokemon-showdown" \
bash "$ROOT/experimental/src/scripts/start_showdown.sh" "$SHOWDOWN_PORT" \
  >"$RUN_ROOT/showdown.log" 2>&1 &
showdown_pid=$!

start_prior() {
  local root="$1"
  local run_name="$2"
  local checkpoint="$3"
  local digest="$4"
  local label="$5"
  local port="$6"
  local log="$7"
  METAMON_CACHE_DIR="$ROOT/srcs/runtime/metamon-cache" \
  TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true WANDB_MODE=disabled \
  "$PYTHON" -u "$ROOT/srcs/metagross/prior_server.py" \
    --local-run-dir "$root" \
    --local-run-name "$run_name" \
    --local-base-model Kakuna \
    --checkpoint "$checkpoint" \
    --checkpoint-sha256 "$digest" \
    --username "$label" \
    --host 127.0.0.1 \
    --port "$port" \
    >"$log" 2>&1 &
  STARTED_PRIOR_PID=$!
}

await_prior() {
  local port="$1"
  for _ in $(seq 1 240); do
    if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
      return
    fi
    sleep 1
  done
  echo "prior server on port $port did not become healthy" >&2
  exit 1
}

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
  --agent-a production_r1_search_first
  --agent-b production_r1_search_first
  --agent-a-require-priors
  --agent-b-require-priors
  --strict-isolated-priors
  --foul-play-python "$FP_PYTHON"
  --foul-play-search-time-ms 500
  --foul-play-search-parallelism 8
  --foul-play-search-threads 1
  --cpuct 2.0
  --production-run-seed "$PRODUCTION_RUN_SEED"
  --concurrent-games 1
  --fail-fast
  --game-timeout-seconds 900
)

run_orientation() {
  local output="$1"
  local log_dir="$2"
  local games="$3"
  local mirror_seed="$4"
  local agent_a_url="$5"
  local agent_b_url="$6"
  local username_prefix="$7"
  local run_id="$8"
  local resume=()
  if [[ -f "$output" ]]; then
    return
  fi
  if [[ -f "$output.progress.json" ]]; then
    resume=(--resume)
  fi
  "$PYTHON" "$ROOT/experimental/src/eval/run.py" \
    "${COMMON_ARGS[@]}" \
    --n-games "$games" \
    --mirror-seed "$mirror_seed" \
    --agent-a-prior-server-url "$agent_a_url" \
    --agent-b-prior-server-url "$agent_b_url" \
    --username-prefix "$username_prefix" \
    --run-id "$run_id" \
    --json-out "$output" \
    --log-dir "$log_dir" \
    "${resume[@]}"
}

start_prior "$POLICY_ROOT" "$ACTION_RUN" 1 "$ACTION_SHA" action "$PRIOR_A_PORT" "$RUN_ROOT/prior-action.log"
prior_a_pid=$STARTED_PRIOR_PID
start_prior "$POLICY_ROOT" "$VISITS_RUN" 1 "$VISITS_SHA" visits "$PRIOR_B_PORT" "$RUN_ROOT/prior-visits.log"
prior_b_pid=$STARTED_PRIOR_PID
await_prior "$PRIOR_A_PORT"
await_prior "$PRIOR_B_PORT"

run_orientation \
  "$RUN_ROOT/screen/action-as-a/result.json" \
  "$RUN_ROOT/screen/action-as-a/logs" \
  "$SCREEN_GAMES_PER_ORIENTATION" 2026081302 \
  "http://127.0.0.1:$PRIOR_A_PORT" "http://127.0.0.1:$PRIOR_B_PORT" \
  spaa search_policy_screen_action_as_a_20260813
run_orientation \
  "$RUN_ROOT/screen/action-as-b/result.json" \
  "$RUN_ROOT/screen/action-as-b/logs" \
  "$SCREEN_GAMES_PER_ORIENTATION" 2026081302 \
  "http://127.0.0.1:$PRIOR_B_PORT" "http://127.0.0.1:$PRIOR_A_PORT" \
  spab search_policy_screen_action_as_b_20260813

"$PYTHON" "$ROOT/experimental/src/scripts/summarize_counterbalanced_ab.py" \
  --g4-as-a "$RUN_ROOT/screen/action-as-a/result.json" \
  --g4-as-b "$RUN_ROOT/screen/action-as-b/result.json" \
  --games-per-orientation "$SCREEN_GAMES_PER_ORIENTATION" \
  --out "$RUN_ROOT/screen/SUMMARY.json"

action_wins="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["g4_wins"])' "$RUN_ROOT/screen/SUMMARY.json")"
total_screen_games=$((SCREEN_GAMES_PER_ORIENTATION * 2))
if (( action_wins * 2 >= total_screen_games )); then
  candidate_run="$ACTION_RUN"
  candidate_sha="$ACTION_SHA"
  candidate_label="action"
else
  candidate_run="$VISITS_RUN"
  candidate_sha="$VISITS_SHA"
  candidate_label="visits"
fi
printf '%s\n' "$candidate_label" >"$RUN_ROOT/SELECTED_CANDIDATE.txt"

stop_pid "$prior_a_pid"
stop_pid "$prior_b_pid"
prior_a_pid=""
prior_b_pid=""

start_prior "$POLICY_ROOT" "$candidate_run" 1 "$candidate_sha" candidate "$PRIOR_A_PORT" "$RUN_ROOT/prior-candidate.log"
prior_a_pid=$STARTED_PRIOR_PID
start_prior "$R1_ROOT" "$R1_RUN" "$R1_CHECKPOINT" "$R1_SHA" r1 "$PRIOR_B_PORT" "$RUN_ROOT/prior-r1.log"
prior_b_pid=$STARTED_PRIOR_PID
await_prior "$PRIOR_A_PORT"
await_prior "$PRIOR_B_PORT"

run_orientation \
  "$RUN_ROOT/gate/candidate-as-a/result.json" \
  "$RUN_ROOT/gate/candidate-as-a/logs" \
  "$GATE_GAMES_PER_ORIENTATION" 2026081303 \
  "http://127.0.0.1:$PRIOR_A_PORT" "http://127.0.0.1:$PRIOR_B_PORT" \
  spca search_policy_candidate_as_a_vs_r1_20260813
run_orientation \
  "$RUN_ROOT/gate/candidate-as-b/result.json" \
  "$RUN_ROOT/gate/candidate-as-b/logs" \
  "$GATE_GAMES_PER_ORIENTATION" 2026081303 \
  "http://127.0.0.1:$PRIOR_B_PORT" "http://127.0.0.1:$PRIOR_A_PORT" \
  spcb search_policy_candidate_as_b_vs_r1_20260813

"$PYTHON" "$ROOT/experimental/src/scripts/summarize_counterbalanced_ab.py" \
  --g4-as-a "$RUN_ROOT/gate/candidate-as-a/result.json" \
  --g4-as-b "$RUN_ROOT/gate/candidate-as-b/result.json" \
  --games-per-orientation "$GATE_GAMES_PER_ORIENTATION" \
  --out "$RUN_ROOT/gate/SUMMARY.json"
