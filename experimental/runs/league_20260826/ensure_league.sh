#!/usr/bin/env bash
# Cron guardian for the baseline league run. Idempotent (league.py skips
# completed matchups); self-removes when the report exists.
set -u
ROOT="/Users/alityb/projects/metagross"
RUN="$ROOT/experimental/runs/league_20260826"
if [ -f "$RUN/baseline/league_report.json" ]; then
  crontab -l 2>/dev/null | grep -v "ensure_league.sh" | crontab -
  curl -s -X POST -H "Content-Type: application/json" \
    -d '{"content":"🏟️ **Baseline league COMPLETE** — reference vector written (league_report.md)"}' \
    "$(cat ~/.metagross_discord_webhook)" >/dev/null 2>&1
  exit 0
fi
if pgrep -f "scripts/league.py" >/dev/null; then
  # Stall watchdog: a hung game leaves the process alive but the newest
  # eval.log frozen. If no eval.log under baseline/ changed in 40 min,
  # kill the stack; the resume machinery continues from banked games.
  # Any activity log counts as freshness: prior servers write during the
  # multi-minute boot phase, so a fresh matchup with no eval.log yet is NOT
  # a stall (previous check kill-looped every boot).
  newest=$(find "$RUN/baseline" "$RUN/league.log" -name "*.log" -newermt "-40 minutes" 2>/dev/null | head -1)
  if [ -n "$newest" ]; then exit 0; fi
  echo "$(date -u +%FT%TZ) STALL detected — killing for resume" >>"$RUN/guardian.log"
  pkill -f "scripts/league.py"; sleep 2
  pkill -f "eval.run"; pkill -f "prior_server.py"; sleep 3
  pkill -9 -f "scripts/league.py|eval.run|prior_server.py" 2>/dev/null
fi
echo "$(date -u +%FT%TZ) league dead — (re)launching" >>"$RUN/guardian.log"
nohup caffeinate -i "$ROOT/.venv-metamon/bin/python" \
  "$ROOT/experimental/src/scripts/league.py" \
  --config "$RUN/league_baseline.json" \
  --out "$RUN/baseline" >>"$RUN/league.log" 2>&1 &
