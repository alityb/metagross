#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-foul-play/bin/python}"
N_GAMES="${N_GAMES:-100}"
SEARCH_TIME_MS="${SEARCH_TIME_MS:-100}"
FORMAT="${FORMAT:-gen9randombattle}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/external/selfplay_data}"
SHOWDOWN_URI="${SHOWDOWN_URI:-ws://localhost:8000/showdown/websocket}"
# METAGROSS_VALUE_MODEL may be set externally to inject a learned eval
# into the generating policy. If not set, both sides use stock eval.
VALUE_MODEL="${METAGROSS_VALUE_MODEL:-}"
# Max games running concurrently = floor(physical_cores / 2) so each
# pair of search processes gets a dedicated core. Override with CONCURRENCY.
CONCURRENCY="${CONCURRENCY:-5}"

mkdir -p "$OUTPUT_DIR"
# Use absolute paths so that os.chdir inside run_foul_play.py doesn't redirect writes.
ABS_OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
ACCEPTOR_LOG="$ABS_OUTPUT_DIR/acceptor_decisions.jsonl"
CHALLENGER_LOG="$ABS_OUTPUT_DIR/challenger_decisions.jsonl"

# Track active background job pids for concurrency control.
declare -a ACTIVE_PIDS=()
games_started=0

wait_for_slot() {
    while [[ ${#ACTIVE_PIDS[@]} -ge $CONCURRENCY ]]; do
        # Wait for any one job to finish.
        local new_pids=()
        for pid in "${ACTIVE_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        ACTIVE_PIDS=("${new_pids[@]+"${new_pids[@]}"}")
        if [[ ${#ACTIVE_PIDS[@]} -ge $CONCURRENCY ]]; then
            sleep 1
        fi
    done
}

for i in $(seq 1 "$N_GAMES"); do
    wait_for_slot

    ACCEPTOR_USER="spA$(printf '%04d' $i)$(openssl rand -hex 2)"
    CHALLENGER_USER="spC$(printf '%04d' $i)$(openssl rand -hex 2)"

    # Build env prefix: always log decisions; optionally inject value model.
    ENV_PREFIX="METAGROSS_DECISION_LOG"
    if [[ -n "$VALUE_MODEL" ]]; then
        (
            env METAGROSS_DECISION_LOG="$ACCEPTOR_LOG" \
                METAGROSS_VALUE_MODEL="$VALUE_MODEL" \
                "$PYTHON_BIN" "$ROOT_DIR/scripts/run_foul_play.py" \
                --websocket-uri "$SHOWDOWN_URI" \
                --ps-username "$ACCEPTOR_USER" \
                --bot-mode accept_challenge \
                --pokemon-format "$FORMAT" \
                --run-count 1 \
                --search-time-ms "$SEARCH_TIME_MS" \
                --search-parallelism 1 \
                --search-threads 1 \
                --log-level WARNING &
            ACC=$!
            sleep 5
            env METAGROSS_DECISION_LOG="$CHALLENGER_LOG" \
                METAGROSS_VALUE_MODEL="$VALUE_MODEL" \
                "$PYTHON_BIN" "$ROOT_DIR/scripts/run_foul_play.py" \
                --websocket-uri "$SHOWDOWN_URI" \
                --ps-username "$CHALLENGER_USER" \
                --bot-mode challenge_user \
                --user-to-challenge "$ACCEPTOR_USER" \
                --pokemon-format "$FORMAT" \
                --run-count 1 \
                --search-time-ms "$SEARCH_TIME_MS" \
                --search-parallelism 1 \
                --search-threads 1 \
                --log-level WARNING &
            CHAL=$!
            wait $ACC $CHAL || true
        ) &
    else
        (
            env METAGROSS_DECISION_LOG="$ACCEPTOR_LOG" \
                "$PYTHON_BIN" "$ROOT_DIR/scripts/run_foul_play.py" \
                --websocket-uri "$SHOWDOWN_URI" \
                --ps-username "$ACCEPTOR_USER" \
                --bot-mode accept_challenge \
                --pokemon-format "$FORMAT" \
                --run-count 1 \
                --search-time-ms "$SEARCH_TIME_MS" \
                --search-parallelism 1 \
                --search-threads 1 \
                --log-level WARNING &
            ACC=$!
            sleep 5
            env METAGROSS_DECISION_LOG="$CHALLENGER_LOG" \
                "$PYTHON_BIN" "$ROOT_DIR/scripts/run_foul_play.py" \
                --websocket-uri "$SHOWDOWN_URI" \
                --ps-username "$CHALLENGER_USER" \
                --bot-mode challenge_user \
                --user-to-challenge "$ACCEPTOR_USER" \
                --pokemon-format "$FORMAT" \
                --run-count 1 \
                --search-time-ms "$SEARCH_TIME_MS" \
                --search-parallelism 1 \
                --search-threads 1 \
                --log-level WARNING &
            CHAL=$!
            wait $ACC $CHAL || true
        ) &
    fi
    ACTIVE_PIDS+=($!)
    games_started=$((games_started + 1))
    echo "game=$i launched (active=$(( ${#ACTIVE_PIDS[@]} )))"
done

# Wait for all remaining games.
for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
    wait "$pid" || true
done
echo "all $N_GAMES games done"

# Merge into single file.
SELFPLAY_OUT="$ABS_OUTPUT_DIR/selfplay_decisions.jsonl"
cat "$ACCEPTOR_LOG" "$CHALLENGER_LOG" > "$SELFPLAY_OUT" 2>/dev/null || true
TOTAL_ROWS=$(wc -l < "$SELFPLAY_OUT")
echo "total rows: $TOTAL_ROWS"

# FATAL ASSERT: if games completed but we got zero rows, something is
# systemically broken (path resolution, env propagation, etc.). Halt loudly.
if [[ $games_started -gt 0 && $TOTAL_ROWS -eq 0 ]]; then
    echo "FATAL: $games_started games completed but selfplay_decisions.jsonl has 0 rows." >&2
    echo "FATAL: Decision logging is broken. Check METAGROSS_DECISION_LOG path resolution," >&2
    echo "FATAL: env propagation to subprocesses, and that run_foul_play.py patch_decision_logging() activates." >&2
    exit 1
fi
