#!/usr/bin/env bash
set -u
ROOT="/Users/alityb/projects/metagross"
RUN="$ROOT/experimental/runs/replications_20260903"
PY="$ROOT/.venv-metamon/bin/python"
HOOK="$(cat "$HOME/.metagross_discord_webhook" 2>/dev/null)"
ping() { [ -n "$HOOK" ] && curl -s -m 15 -o /dev/null -H "Content-Type: application/json" -X POST "$HOOK" -d "{\"content\":\"$1\"}"; }
DEADLINE=$(( $(date +%s) + 190000 ))
for stage in r1_gumbel r2_flatten; do
  attempts=0
  while [ ! -f "$RUN/$stage/league_report.json" ] && [ "$(date +%s)" -lt "$DEADLINE" ] && [ "$attempts" -lt 30 ]; do
    attempts=$((attempts+1))
    echo "=== $(date -u +%FT%TZ) $stage attempt $attempts" >>"$RUN/chain.log"
    # recycle showdown so METAGROSS_EVAL_PAIR_DIR points at this stage's dir
    /usr/sbin/lsof -ti tcp:8022 2>/dev/null | xargs kill -9 2>/dev/null; sleep 2
    "$PY" "$ROOT/experimental/src/scripts/league.py" \
      --config "$RUN/$stage.json" --out "$RUN/$stage" >>"$RUN/chain.log" 2>&1 &
    LP=$!
    # Fail-closed activation enforcement: within 25 min the candidate arm
    # must show its ACTIVE line (GUMBEL_ROOT for r1, temperature for r2),
    # else kill the stage and alert rather than burn 200 inert games.
    if [ "$stage" = r1_gumbel ]; then PAT="GUMBEL_ROOT ACTIVE"; else PAT="schedule ACTIVE"; fi
    for _ in $(seq 1 25); do
      sleep 60
      kill -0 "$LP" 2>/dev/null || break
      if grep -rql "$PAT" "$RUN/$stage"/m*/logs 2>/dev/null; then PAT=""; break; fi
    done
    if [ -n "$PAT" ] && kill -0 "$LP" 2>/dev/null; then
      ping "🛑 $stage: NO activation ('$PAT') within 25 min — stage killed, needs investigation"
      kill "$LP" 2>/dev/null; sleep 3; pkill -f "eval[.]run"; pkill -f "prior_server.py"
      echo "=== $(date -u +%FT%TZ) $stage KILLED: inert" >>"$RUN/chain.log"
      break 2
    fi
    wait "$LP" 2>/dev/null
    sleep 15
  done
  [ -f "$RUN/$stage/league_report.json" ] && ping "🔁 replication $stage COMPLETE: $(cat "$RUN/$stage/league_report.md" | tail -2 | head -1)" || ping "⚠️ replication $stage gave up"
done
pkill -f "prior_server.py" 2>/dev/null
echo "=== $(date -u +%FT%TZ) chain done" >>"$RUN/chain.log"
