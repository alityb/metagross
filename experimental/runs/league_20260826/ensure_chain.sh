#!/usr/bin/env bash
# Overnight chain guardian: (1) re-arm the baseline league when the
# root-cause session releases the machine (or at the fallback deadline if
# the machine is quiet), (2) after the baseline report, run the
# preregistered retrodiction league with the same liveness+stall pattern,
# (3) self-remove when the retrodiction report exists.
set -u
ROOT="/Users/alityb/projects/metagross"
export PATH="/Users/alityb/.nvm/versions/node/v22.19.0/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"  # cron PATH lacks node (nvm)
RUN="$ROOT/experimental/runs/league_20260826"
HOOK="$(cat "$HOME/.metagross_discord_webhook" 2>/dev/null)"
ping() { [ -n "$HOOK" ] && curl -s -X POST -H "Content-Type: application/json" -d "{\"content\":\"$1\"}" "$HOOK" >/dev/null 2>&1; }

PH="$ROOT/experimental/runs/priors_h2h_20260828"
# Stage D done -> teardown
if [ -f "$PH/run/league_report.json" ]; then
  crontab -l 2>/dev/null | grep -v "ensure_chain.sh" | crontab -
  ping "🏁 **Priors-vs-vanilla H2H COMPLETE** — chain finished; all three results on disk."
  exit 0
fi
# Stage D: retrodiction done -> drive the powered priors H2H
if [ -f "$RUN/flattened/league_report.json" ]; then
  if pgrep -f "scripts/league.py" >/dev/null; then
    newest=$(find "$PH/run" "$PH/league_ph.log" -name "*.log" -newermt "-40 minutes" 2>/dev/null | head -1)
    [ -n "$newest" ] && exit 0
    echo "$(date -u +%FT%TZ) priors-h2h STALL — killing for resume" >>"$RUN/guardian.log"
    pkill -f "scripts/league.py"; sleep 2; pkill -f "eval.run"; pkill -f "prior_server.py"; sleep 3
    pkill -9 -f "scripts/league.py|eval.run|prior_server.py" 2>/dev/null
  fi
  if [ ! -d "$PH/run" ]; then ping "🏟️ retrodiction done — **powered priors-vs-vanilla H2H starting** (200 games, prereg'd)"; fi
  echo "$(date -u +%FT%TZ) priors-h2h (re)launch" >>"$RUN/guardian.log"
  nohup caffeinate -i "$ROOT/.venv-metamon/bin/python"     "$ROOT/experimental/src/scripts/league.py"     --config "$PH/priors_h2h.json"     --out "$PH/run" >>"$PH/league_ph.log" 2>&1 &
  exit 0
fi

# Stage A: baseline still gated by the root-cause session's disable line?
if grep -q "^exit 0" "$RUN/ensure_league.sh" 2>/dev/null; then
  # fallback: past the deadline AND machine quiet -> re-arm ourselves
  now=$(date -u +%s)
  if [ "$now" -ge "1787910996" ] && ! pgrep -f "eval[.]run|prior_[s]erver" >/dev/null; then
    sed -i '' '/^# TEMPORARILY DISABLED/,/^exit 0/d' "$RUN/ensure_league.sh"
    ping "⚙️ chain fallback: root-cause session quiet past deadline — baseline league re-armed"
  fi
  exit 0
fi

# Stage B: baseline running/pending (its own ensure_league.sh cron drives it)
if [ ! -f "$RUN/baseline/league_report.json" ]; then exit 0; fi

# Stage C: baseline done -> drive the retrodiction league
if ! crontab -l 2>/dev/null | grep -q "ensure_league.sh"; then :; fi
if pgrep -f "scripts/league.py" >/dev/null; then
  newest=$(find "$RUN/flattened" "$RUN/league_retro.log" -name "*.log" -newermt "-40 minutes" 2>/dev/null | head -1)
  [ -n "$newest" ] && exit 0
  echo "$(date -u +%FT%TZ) retro STALL — killing for resume" >>"$RUN/guardian.log"
  pkill -f "scripts/league.py"; sleep 2; pkill -f "eval.run"; pkill -f "prior_server.py"; sleep 3
  pkill -9 -f "scripts/league.py|eval.run|prior_server.py" 2>/dev/null
fi
if [ ! -d "$RUN/flattened" ]; then ping "🏟️ baseline done — **retrodiction league starting** (flattened vs frozen pool, paired seeds)"; fi
echo "$(date -u +%FT%TZ) retro (re)launch" >>"$RUN/guardian.log"
nohup caffeinate -i "$ROOT/.venv-metamon/bin/python" \
  "$ROOT/experimental/src/scripts/league.py" \
  --config "$RUN/league_retro.json" \
  --out "$RUN/flattened" >>"$RUN/league_retro.log" 2>&1 &
