#!/usr/bin/env bash
# Bounded peak-Elo push on the stateless arm (tophfan32). Observational —
# no gate, no RD stop; inner loop restarts the supervisor until the
# deadline. No cron (crontab writes hang on this host).
set -u
ROOT="/Users/alityb/projects/metagross"
RUN="$ROOT/experimental/runs/peak_push_20260830"
PY="$ROOT/.venv-metamon/bin/python"
DEADLINE=$(( $(date +%s) + 16200 ))   # 4.5h
cd "$ROOT"
ENGINE_ROOT="$ROOT/.venv-fp-priors/lib/python3.11/site-packages"
ENGINE_SO="$(ls "$ENGINE_ROOT"/poke_engine/poke_engine*.so | head -1)"
export METAGROSS_PINNED_ENGINE_IMPORT_ROOT="$ENGINE_ROOT"
export METAGROSS_PINNED_ENGINE_SHA256="$(shasum -a 256 "$ENGINE_SO" | awk '{print $1}')"
export PYTHONPATH="$ROOT"
export METAGROSS_SEARCH_ITERATIONS_PER_500MS="472000"
USER_B="$(sed -n 1p "$HOME/.metagross_ladder_pair_b")"
PASS_B="$(sed -n 2p "$HOME/.metagross_ladder_pair_b")"
launches=0
while [ "$(date +%s)" -lt "$DEADLINE" ] && [ "$launches" -lt 40 ]; do
  launches=$((launches+1))
  echo "=== $(date -u +%FT%TZ) launch #$launches" >>"$RUN/push.log"
  env METAGROSS_TRAJECTORY_MODE="legacy-stateless" \
      METAGROSS_SHOWDOWN_PASSWORD="$PASS_B" \
    "$PY" "$ROOT/srcs/metagross/ladder_supervisor.py" \
      --r1-username "$USER_B" --block-games 12 --cycles 1 \
      --search-parallelism 8 --port 8977 \
      --output-root "$RUN/supervisor" >>"$RUN/push.log" 2>&1
  sleep 10
done
pkill -f "prior_server.py" 2>/dev/null
echo "=== $(date -u +%FT%TZ) push done ($launches launches)" >>"$RUN/push.log"
