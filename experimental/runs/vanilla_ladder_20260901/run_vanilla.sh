#!/usr/bin/env bash
set -u
ROOT="/Users/alityb/projects/metagross"
RUN="$ROOT/experimental/runs/vanilla_ladder_20260901"
PY="$ROOT/.venv-metamon/bin/python"
DEADLINE=$(( $(date +%s) + 50400 ))
cd "$ROOT"
export PYTHONPATH="$ROOT/experimental/src"
export METAGROSS_PINNED_ENGINE_IMPORT_ROOT="$ROOT/.venv-foul-play/lib/python3.11/site-packages"
export METAGROSS_PINNED_ENGINE_SHA256="79bea0e467b32e2958bd5d39595fd728a3068be2950085f7b18fa69943f30d71"
USER_C="$(sed -n 1p "$HOME/.metagross_ladder_pair_c")"
PASS_C="$(sed -n 2p "$HOME/.metagross_ladder_pair_c")"
launches=0
while [ "$(date +%s)" -lt "$DEADLINE" ] && [ "$launches" -lt 20 ]; do
  launches=$((launches+1))
  echo "=== $(date -u +%FT%TZ) launch #$launches" >>"$RUN/run.log"
  "$PY" -m eval.run --mode ladder --server live --format gen9randombattle \
    --agent foul_play --username "$USER_C" --password "$PASS_C" \
    --foul-play-python "$ROOT/.venv-foul-play/bin/python" \
    --foul-play-search-time-ms 500 --foul-play-search-parallelism 8 \
    --foul-play-search-threads 1 --n-games 25 --cpuct 2.0 \
    --game-timeout-seconds 900 \
    --log-dir "$RUN/logs" >>"$RUN/run.log" 2>&1
  curl -s -m 15 "https://pokemonshowdown.com/users/$USER_C.json" >> "$RUN/ratings_poll.jsonl"; echo >> "$RUN/ratings_poll.jsonl"
  sleep 15
done
echo "=== $(date -u +%FT%TZ) done ($launches launches)" >>"$RUN/run.log"
