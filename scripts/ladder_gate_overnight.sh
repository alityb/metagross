#!/usr/bin/env bash
# Overnight ladder gate: run the fine-tuned policy on the live human ladder.
# Crash-resilient: relaunches the runner until the target game count is reached
# (results CSV is append-only; ratings.jsonl polls are append-only).
# Run under caffeinate so the Mac stays awake.
#
# Usage:
#   source ~/.metagross_ladder_secret
#   METAGROSS_SHOWDOWN_PASSWORD=$METAGROSS_LADDER_PASSWORD \
#     nohup bash scripts/ladder_gate_overnight.sh > experiments/ladder_gate.out 2>&1 &
set -u

USERNAME="${LADDER_USERNAME:-metagrossrb1}"
TARGET_GAMES="${TARGET_GAMES:-130}"
OUT_DIR="${OUT_DIR:-experiments/ladder_gate_${USERNAME}}"
RUN_DIR="${RUN_DIR:-$PWD/nets/checkpoints/randbats_full}"
RUN_NAME="${RUN_NAME:-randbats_base}"
CKPT="${CKPT:-2}"
MAX_RESTARTS=25

mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/battle_log_${USERNAME}_gen9randombattle.csv"

log() { echo "$(date -u +%FT%TZ) $*"; }

count_games() {
  if [ -f "$CSV" ]; then echo $(( $(wc -l < "$CSV") - 1 )); else echo 0; fi
}

restarts=0
while true; do
  done_games=$(count_games)
  if [ "$done_games" -ge "$TARGET_GAMES" ]; then
    log "TARGET REACHED: $done_games games"
    break
  fi
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    log "FATAL: exceeded $MAX_RESTARTS restarts at $done_games games; giving up"
    break
  fi
  remaining=$(( TARGET_GAMES - done_games ))
  log "LAUNCH attempt=$((restarts+1)) done=$done_games remaining=$remaining"
  METAMON_CACHE_DIR="$PWD/external/metamon_cache" WANDB_MODE=disabled \
  TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true \
  caffeinate -i .venv-metamon/bin/python scripts/run_policy_public_ladder.py \
    --local-run-dir "$RUN_DIR" --local-run-name "$RUN_NAME" --checkpoint "$CKPT" \
    --username "$USERNAME" \
    --battle-format gen9randombattle \
    --total-battles "$remaining" \
    --out-dir "$OUT_DIR" >> "$OUT_DIR/runner.out" 2>&1 &
  runner_pid=$!
  # Stall watchdog: poke-env's ladder loop can die silently (swallowed
  # exception in POKE_LOOP) leaving a healthy-looking process that never
  # searches. If no new game for STALL_MINUTES, kill and relaunch.
  STALL_MINUTES="${STALL_MINUTES:-12}"
  last_count=$(count_games)
  last_change=$(date +%s)
  while kill -0 "$runner_pid" 2>/dev/null; do
    sleep 60
    now_count=$(count_games)
    if [ "$now_count" -ne "$last_count" ]; then
      last_count=$now_count
      last_change=$(date +%s)
    elif [ $(( $(date +%s) - last_change )) -ge $(( STALL_MINUTES * 60 )) ]; then
      log "FATAL stall: no new game in ${STALL_MINUTES}m at $now_count games; killing runner"
      kill "$runner_pid" 2>/dev/null
      sleep 10
      kill -9 "$runner_pid" 2>/dev/null
      break
    fi
  done
  wait "$runner_pid" 2>/dev/null
  rc=$?
  log "runner exited rc=$rc at $(count_games) games; backing off"
  restarts=$((restarts+1))
  sleep $(( 30 * restarts > 300 ? 300 : 30 * restarts ))
done

log "FINAL ratings tail:"
tail -3 "$OUT_DIR/ratings.jsonl" 2>/dev/null
