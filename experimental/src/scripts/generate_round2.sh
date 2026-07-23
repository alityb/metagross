#!/usr/bin/env bash
set -euo pipefail

# Round-2 self-play generation: 500ms/P8 with Kakuna priors + replay capture.
# Runs on a game-worker instance (c6i). Connects to a prior server on another
# instance.  Both sides are our agent (foul_play_root_priors_opp).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_ROOT="$REPO_ROOT/src"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-foul-play/bin/python}"
N_GAMES="${N_GAMES:-50000}"
SEARCH_TIME_MS="${SEARCH_TIME_MS:-500}"
SEARCH_PARALLELISM="${SEARCH_PARALLELISM:-8}"
FORMAT="${FORMAT:-gen9randombattle}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data/selfplay_round2}"
SHOWDOWN_URI="${SHOWDOWN_URI:-ws://localhost:8000/showdown/websocket}"
PRIOR_SERVER_URL="${PRIOR_SERVER_URL:-http://127.0.0.1:8977}"
CONCURRENCY="${CONCURRENCY:-8}"
REPLAY_DIR="${REPLAY_DIR:-$OUTPUT_DIR/replays}"

mkdir -p "$OUTPUT_DIR" "$REPLAY_DIR"
ABS_OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
ABS_REPLAY_DIR="$(cd "$REPLAY_DIR" && pwd)"
ACCEPTOR_LOG="$ABS_OUTPUT_DIR/acceptor_decisions.jsonl"
CHALLENGER_LOG="$ABS_OUTPUT_DIR/challenger_decisions.jsonl"

declare -a ACTIVE_PIDS=()
games_started=0

wait_for_slot() {
    while [[ ${#ACTIVE_PIDS[@]} -ge $CONCURRENCY ]]; do
        local new_pids=()
        for pid in "${ACTIVE_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then new_pids+=("$pid"); fi
        done
        ACTIVE_PIDS=("${new_pids[@]+"${new_pids[@]}"}")
        if [[ ${#ACTIVE_PIDS[@]} -ge $CONCURRENCY ]]; then sleep 1; fi
    done
}

for i in $(seq 1 "$N_GAMES"); do
    wait_for_slot
    # Metamon parsed-replay filenames use `_` as a field separator. Keep
    # generated usernames underscore-free so both POV trajectories survive
    # indexing as distinct files.
    ACCEPTOR_USER="r2A$(printf '%05d' $i)$(openssl rand -hex 2)"
    CHALLENGER_USER="r2C$(printf '%05d' $i)$(openssl rand -hex 2)"
    (
        env METAGROSS_DECISION_LOG="$ACCEPTOR_LOG" \
            METAGROSS_REPLAY_DIR="$ABS_REPLAY_DIR" \
            METAGROSS_PRIOR_SERVER="$PRIOR_SERVER_URL" \
            METAGROSS_CPUCT="2.0" \
            METAGROSS_REQUIRE_PRIORS="1" \
            "$PYTHON_BIN" "$SRC_ROOT/scripts/run_foul_play.py" \
            --websocket-uri "$SHOWDOWN_URI" \
            --ps-username "$ACCEPTOR_USER" \
            --bot-mode accept_challenge \
            --pokemon-format "$FORMAT" \
            --run-count 1 \
            --search-time-ms "$SEARCH_TIME_MS" \
            --search-parallelism "$SEARCH_PARALLELISM" \
            --search-threads 1 \
            --log-level WARNING &
        ACC=$!
        sleep 5
        env METAGROSS_DECISION_LOG="$CHALLENGER_LOG" \
            METAGROSS_PRIOR_SERVER="$PRIOR_SERVER_URL" \
            METAGROSS_CPUCT="2.0" \
            METAGROSS_REQUIRE_PRIORS="1" \
            "$PYTHON_BIN" "$SRC_ROOT/scripts/run_foul_play.py" \
            --websocket-uri "$SHOWDOWN_URI" \
            --ps-username "$CHALLENGER_USER" \
            --bot-mode challenge_user \
            --user-to-challenge "$ACCEPTOR_USER" \
            --pokemon-format "$FORMAT" \
            --run-count 1 \
            --search-time-ms "$SEARCH_TIME_MS" \
            --search-parallelism "$SEARCH_PARALLELISM" \
            --search-threads 1 \
            --log-level WARNING &
        CHAL=$!
        wait $ACC $CHAL || true
    ) &
    ACTIVE_PIDS+=($!)
    games_started=$((games_started + 1))
    echo "game=$i launched (active=$(( ${#ACTIVE_PIDS[@]} )))"
done

for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do wait "$pid" || true; done
echo "all $N_GAMES games done"

SELFPLAY_OUT="$ABS_OUTPUT_DIR/selfplay_decisions.jsonl"
cat "$ACCEPTOR_LOG" "$CHALLENGER_LOG" > "$SELFPLAY_OUT" 2>/dev/null || true
TOTAL_ROWS=$(wc -l < "$SELFPLAY_OUT")
REPLAY_COUNT=$(ls "$ABS_REPLAY_DIR"/*.json 2>/dev/null | wc -l)
echo "total rows: $TOTAL_ROWS | replays: $REPLAY_COUNT"

if [[ $games_started -gt 0 && $TOTAL_ROWS -eq 0 ]]; then
    echo "FATAL: $games_started games but 0 decision rows" >&2; exit 1
fi
if [[ $games_started -gt 0 && $REPLAY_COUNT -eq 0 ]]; then
    echo "FATAL: $games_started games but 0 replays" >&2; exit 1
fi
