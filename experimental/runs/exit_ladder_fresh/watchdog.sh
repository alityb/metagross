#!/bin/zsh
# Self-healing watchdog for the metaexitr1 ExIt ladder run.
# - Restarts the ladder client if it stalls (log silent > STALL s) or dies.
# - Restarts the prior server (ExIt ckpt) if its /health goes down.
# - Preserves every run's game log under logs/run_<ts>/ (no clobbering).
# - Touch experiments/exit_ladder_fresh/STOP to shut down gracefully.
set -u

REPO=/Users/alityb/projects/metagross
cd "$REPO" || exit 1
LOGDIR="$REPO/experiments/exit_ladder_fresh"
WLOG="$LOGDIR/watchdog.out"
STOP="$LOGDIR/STOP"
PORT=8977
STALL=300          # seconds of log silence => hung
CHECK=30           # poll interval
CUR_CLILOG=""

log(){ echo "$(date '+%F %T') $*" >> "$WLOG" }

prior_up(){ curl -s --max-time 6 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok": true' }

start_prior(){
  log "starting prior server (randbats_exit_r1 ckpt5) on :$PORT"
  nohup env METAMON_CACHE_DIR="$REPO/external/metamon_cache" TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true PYTHONPATH=src \
    .venv-metamon/bin/python -u src/scripts/prior_server.py \
      --local-run-dir "$REPO/src/nets/checkpoints/randbats_full" \
      --local-run-name randbats_exit_r1 --checkpoint 5 --port $PORT --username metaexitr1 \
      >> "$LOGDIR/prior_server.out" 2>&1 &
  disown
  for i in {1..40}; do sleep 5; prior_up && { log "prior ready"; return 0 }; done
  log "prior FAILED to become ready"; return 1
}

kill_client(){
  pkill -f "eval.run.*metaexitr1" 2>/dev/null
  pkill -f "run_foul_play.py.*metaexitr1" 2>/dev/null
  sleep 5
}

client_running(){ pgrep -f "run_foul_play.py.*metaexitr1" >/dev/null 2>&1 }

start_client(){
  local sub="$LOGDIR/logs/run_$(date '+%Y%m%dT%H%M%S')"
  mkdir -p "$sub"
  CUR_CLILOG="$sub/metaexitr1.log"
  log "starting ladder client; log=$CUR_CLILOG"
  nohup env PYTHONPATH=src \
    .venv-metamon/bin/python -u -m eval.run \
      --mode ladder --format gen9randombattle --server live \
      --agent foul_play_root_priors_opp \
      --username metaexitr1 --password "$(cat "$REPO/.ladder_exit_r1_credentials")" \
      --foul-play-python .venv-fp-priors/bin/python \
      --foul-play-search-time-ms 500 --foul-play-search-parallelism 8 --foul-play-search-threads 1 \
      --prior-server-url "http://127.0.0.1:$PORT" \
      --n-games 200 --log-dir "$sub" \
      >> "$LOGDIR/run.out" 2>&1 &
  disown
}

BACKOFF=60          # current relaunch delay (grows on consecutive failures)
BACKOFF_MAX=600     # cap at 10 min
FAIL_STREAK=0       # consecutive launch failures
LAST_GAMES=0        # games seen last cycle (to detect progress)

count_games(){ grep -aca 'Winner:' "$CUR_CLILOG" 2>/dev/null || echo 0 }

log "=== watchdog start (pid $$) ==="
prior_up || start_prior
kill_client
start_client
sleep 90

while true; do
  if [[ -f "$STOP" ]]; then
    log "STOP present -> killing client, exiting"; kill_client; log "watchdog exit"; rm -f "$STOP"; exit 0
  fi
  prior_up || { log "prior DOWN -> restart"; start_prior }
  if ! client_running; then
    FAIL_STREAK=$((FAIL_STREAK+1))
    BACKOFF=$(( BACKOFF * 2 > BACKOFF_MAX ? BACKOFF_MAX : BACKOFF * 2 ))
    log "client not running (streak=$FAIL_STREAK, backoff=${BACKOFF}s) -> relaunch"
    kill_client; sleep $BACKOFF; start_client; sleep 90; continue
  fi
  if [[ -n "$CUR_CLILOG" && -f "$CUR_CLILOG" ]]; then
    now=$(date +%s); m=$(stat -f %m "$CUR_CLILOG"); age=$(( now - m ))
    cur_games=$(count_games)
    if (( cur_games > LAST_GAMES )); then
      FAIL_STREAK=0; BACKOFF=60; LAST_GAMES=$cur_games
    fi
    if (( age > STALL )); then
      FAIL_STREAK=$((FAIL_STREAK+1))
      BACKOFF=$(( BACKOFF * 2 > BACKOFF_MAX ? BACKOFF_MAX : BACKOFF * 2 ))
      log "STALL ${age}s (>$STALL, streak=$FAIL_STREAK, backoff=${BACKOFF}s) -> restart"
      kill_client; sleep $BACKOFF; start_client; sleep 90; continue
    fi
  fi
  sleep $CHECK
done
