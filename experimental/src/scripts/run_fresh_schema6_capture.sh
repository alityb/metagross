#!/usr/bin/env bash
set -euo pipefail

# Local, fail-closed fresh-corpus capture for the outcome-residual program.
# The output directory must not already contain data. Infrastructure is scoped
# to this invocation and is stopped on every exit path.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="${1:?usage: run_fresh_schema6_capture.sh RUN_DIR [N_GAMES] [BASE_PORT] [peer|direct_r1|unguided] [MINIMUM_ELIGIBLE_GROUPS] [fresh|resume]}"
N_GAMES="${2:-2}"
BASE_PORT="${3:-8040}"
PROFILE="${4:-peer}"
MINIMUM_ELIGIBLE_GROUPS="${5:-}"
RUN_MODE="${6:-fresh}"
PRIOR_A_PORT="$((BASE_PORT + 1))"
PRIOR_B_PORT="$((BASE_PORT + 2))"
CHECKPOINT_SHA256="c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
CAPTURE_ENGINE_SOURCE_SHA256="ccc5dd035c25fbf8e6b98301198cdaa35ce1012ae6f45ab81851965b0b05a0a3"
CAPTURE_ENGINE_BINARY_SHA256="3910185bb7f5e5f0283781b0b2292664f4c980126f320325143ba5970d4aba35"

if [[ ! "$N_GAMES" =~ ^[0-9]+$ ]] || ((N_GAMES < 2 || N_GAMES % 2 != 0)); then
  echo "N_GAMES must be a positive even integer of at least 2" >&2
  exit 2
fi
if [[ ! "$BASE_PORT" =~ ^[0-9]+$ ]] || ((BASE_PORT < 1024 || BASE_PORT > 65532)); then
  echo "BASE_PORT must be in [1024, 65532]" >&2
  exit 2
fi
actual_engine_source="$(cd "$ROOT_DIR" && { PYTHONPATH="$ROOT_DIR" "$ROOT_DIR/.venv-metamon/bin/python" -c \
  'from pathlib import Path; from srcs.metagross.mcts_contract import compute_engine_source_sha256; print(compute_engine_source_sha256(Path("srcs/vendor/poke-engine")))'; } 2>/dev/null)"
if [[ "$actual_engine_source" != "$CAPTURE_ENGINE_SOURCE_SHA256" ]]; then
  echo "capture engine source hash mismatch: $actual_engine_source" >&2
  exit 2
fi
engine_binary="$("$ROOT_DIR/.venv-foul-play/bin/python" -c \
  'import poke_engine.poke_engine as core; print(core.__file__)')"
actual_engine_binary="$(shasum -a 256 "$engine_binary" | awk '{print $1}')"
if [[ "$actual_engine_binary" != "$CAPTURE_ENGINE_BINARY_SHA256" ]]; then
  echo "capture engine binary hash mismatch: $actual_engine_binary" >&2
  exit 2
fi
case "$PROFILE" in
  peer)
    AGENT_B="production_r1_search_first"
    PRIOR_SIDES=(a b)
    EXPECTED_GROUPS="$((N_GAMES * 2))"
    DEFAULT_MIRROR_SEED=2026081451
    DEFAULT_PRODUCTION_SEED=736368656d613663617074757265736368656d61366361707475726573636865
    ;;
  direct_r1)
    AGENT_B="direct_r1"
    PRIOR_SIDES=(a)
    EXPECTED_GROUPS="$N_GAMES"
    DEFAULT_MIRROR_SEED=2026081452
    DEFAULT_PRODUCTION_SEED=736368656d61366469726563747231736368656d613664697265637472317363
    ;;
  unguided)
    AGENT_B="foul_play"
    PRIOR_SIDES=(a)
    EXPECTED_GROUPS="$N_GAMES"
    DEFAULT_MIRROR_SEED=2026081453
    DEFAULT_PRODUCTION_SEED=736368656d6136756e677569646564736368656d6136756e6775696465647363
    ;;
  *)
    echo "PROFILE must be peer, direct_r1, or unguided" >&2
    exit 2
    ;;
esac
MIRROR_SEED="${METAGROSS_CAPTURE_MIRROR_SEED:-$DEFAULT_MIRROR_SEED}"
PRODUCTION_SEED="${METAGROSS_CAPTURE_PRODUCTION_SEED:-$DEFAULT_PRODUCTION_SEED}"
USERNAME_PREFIX="${METAGROSS_CAPTURE_USERNAME_PREFIX:-s6c}"
if [[ ! "$MIRROR_SEED" =~ ^[0-9]+$ ]]; then
  echo "METAGROSS_CAPTURE_MIRROR_SEED must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "$PRODUCTION_SEED" =~ ^[0-9a-f]{64}$ ]]; then
  echo "METAGROSS_CAPTURE_PRODUCTION_SEED must be exactly 64 lowercase hex characters" >&2
  exit 2
fi
if [[ ! "$USERNAME_PREFIX" =~ ^[A-Za-z0-9]{1,8}$ ]]; then
  echo "METAGROSS_CAPTURE_USERNAME_PREFIX must be 1-8 alphanumeric characters" >&2
  exit 2
fi
MINIMUM_COMPLETE_GROUPS="$(((EXPECTED_GROUPS * 95 + 99) / 100))"
if [[ -z "$MINIMUM_ELIGIBLE_GROUPS" ]]; then
  MINIMUM_ELIGIBLE_GROUPS="$((EXPECTED_GROUPS / ${#PRIOR_SIDES[@]}))"
fi
if [[ ! "$MINIMUM_ELIGIBLE_GROUPS" =~ ^[0-9]+$ ]] || ((MINIMUM_ELIGIBLE_GROUPS < 1)); then
  echo "MINIMUM_ELIGIBLE_GROUPS must be a positive integer" >&2
  exit 2
fi
if [[ "$RUN_MODE" != fresh && "$RUN_MODE" != resume ]]; then
  echo "RUN_MODE must be fresh or resume" >&2
  exit 2
fi
if [[ "$RUN_MODE" == fresh && -e "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "RUN_DIR must be absent or empty: $RUN_DIR" >&2
  exit 2
fi
if [[ "$RUN_MODE" == resume && ! -f "$RUN_DIR/result.json.progress.json" ]]; then
  echo "resume requires an existing atomic eval progress snapshot in $RUN_DIR" >&2
  exit 2
fi
PORTS=("$BASE_PORT" "$PRIOR_A_PORT")
[[ "$PROFILE" == peer ]] && PORTS+=("$PRIOR_B_PORT")
for port in "${PORTS[@]}"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null; then
    echo "port $port is already occupied" >&2
    exit 2
  fi
done

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/registrations"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait "${PIDS[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

env CUDA_VISIBLE_DEVICES='' \
  "$ROOT_DIR/experimental/src/scripts/start_showdown.sh" "$BASE_PORT" \
  >>"$RUN_DIR/showdown.log" 2>&1 &
PIDS+=("$!")

for side in "${PRIOR_SIDES[@]}"; do
  if [[ "$side" == a ]]; then
    prior_port="$PRIOR_A_PORT"
  else
    prior_port="$PRIOR_B_PORT"
  fi
  env METAMON_CACHE_DIR="$ROOT_DIR/external/metamon_cache" \
    TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' \
    METAGROSS_DUAL_R1_CAPTURE=1 \
    "$ROOT_DIR/.venv-metamon/bin/python" -u \
    "$ROOT_DIR/srcs/metagross/prior_server.py" \
    --local-run-dir "$ROOT_DIR/srcs/models" \
    --local-run-name randbats_exit_r1 \
    --checkpoint 5 \
    --checkpoint-sha256 "$CHECKPOINT_SHA256" \
    --trajectory-mode causal-history \
    --username "schema6${side}" \
    --port "$prior_port" \
    --decision-dump "$RUN_DIR/prior-${side}.jsonl" \
    >>"$RUN_DIR/prior-${side}.log" 2>&1 &
  PIDS+=("$!")
done

for _attempt in {1..180}; do
  ready=1
  for port in "${PORTS[@]}"; do
    if ! lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null; then
      ready=0
    fi
  done
  ((ready == 1)) && break
  sleep 1
done
for port in "${PORTS[@]}"; do
  if ! lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null; then
    echo "local infrastructure did not become ready; inspect $RUN_DIR/*.log" >&2
    exit 1
  fi
done

EVAL_PRIOR_ARGS=(
  --agent-a-prior-server-url "http://127.0.0.1:${PRIOR_A_PORT}"
  --agent-a-require-priors
  --agent-a-decision-log "$RUN_DIR/agent-a-decisions.jsonl"
)
RESUME_ARGS=()
[[ "$RUN_MODE" == resume ]] && RESUME_ARGS+=(--resume)
if [[ "$PROFILE" == peer ]]; then
  EVAL_PRIOR_ARGS+=(
    --agent-b-prior-server-url "http://127.0.0.1:${PRIOR_B_PORT}"
    --agent-b-require-priors
    --strict-isolated-priors
    --agent-b-decision-log "$RUN_DIR/agent-b-decisions.jsonl"
  )
fi

env METAMON_CACHE_DIR="$ROOT_DIR/external/metamon_cache" \
  TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' \
  METAGROSS_DUAL_R1_CAPTURE=1 \
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
  --agent-b "$AGENT_B" \
  "${EVAL_PRIOR_ARGS[@]}" \
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
  --run-id "$(basename "$RUN_DIR")" \
  --json-out "$RUN_DIR/result.json" \
  --log-dir "$RUN_DIR/logs" \
  "${RESUME_ARGS[@]}"

DECISION_ARGS=(
  --decision-log "$RUN_DIR/agent-a-decisions.jsonl.dual-r1-roots.jsonl"
)
SNAPSHOT_ARGS=(--prior-snapshot "$RUN_DIR/prior-a.jsonl")
if [[ "$PROFILE" == peer ]]; then
  DECISION_ARGS+=(--decision-log "$RUN_DIR/agent-b-decisions.jsonl.dual-r1-roots.jsonl")
  SNAPSHOT_ARGS+=(--prior-snapshot "$RUN_DIR/prior-b.jsonl")
fi

env PYTHONPATH="$ROOT_DIR/experimental/src:$ROOT_DIR" \
  "$ROOT_DIR/.venv-metamon/bin/python" \
  "$ROOT_DIR/experimental/src/scripts/audit_schema6_capture.py" \
  "${DECISION_ARGS[@]}" \
  "${SNAPSHOT_ARGS[@]}" \
  --h2h-result "$RUN_DIR/result.json" \
  --minimum-battles "$MINIMUM_COMPLETE_GROUPS" \
  --minimum-capture-rate 0.95 \
  --output "$RUN_DIR/schema6-capture-audit.json"

env METAMON_CACHE_DIR="$ROOT_DIR/external/metamon_cache" \
  TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH="$ROOT_DIR/experimental/engine/pe_v3_learned_priors/poke-engine-py/python:$ROOT_DIR/experimental/src:$ROOT_DIR" \
  "$ROOT_DIR/.venv-metamon/bin/python" \
  "$ROOT_DIR/experimental/src/scripts/audit_schema6_panel_bridge.py" \
  "${DECISION_ARGS[@]}" \
  "${SNAPSHOT_ARGS[@]}" \
  --minimum-groups "$MINIMUM_ELIGIBLE_GROUPS" \
  --output "$RUN_DIR/schema6-panel-bridge-audit.json"
