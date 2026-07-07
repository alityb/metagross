#!/usr/bin/env python3
"""Self-play data generation for expert iteration.

Runs FP vs FP (or FP vs policy) games on the local Showdown server,
capturing ALL data per game:
  - Showdown replay JSON (protocol log + inputlog) for the metamon parser
  - FP decision log (state strings, MCTS visit distributions, chosen moves)
  - Game result (winner, turns, duration)

Output directory structure:
  data/selfplay/<run_id>/
    replays/          # Showdown replay JSONs (parser input)
    decisions/        # Per-turn decision logs (policy targets)
    games.jsonl       # Per-game metadata

Usage:
  METAGROSS_PRIOR_SERVER=http://127.0.0.1:8977 \
  METAGROSS_DECISION_LOG=data/selfplay/run1/decisions/decisions.jsonl \
  METAGROSS_REPLAY_DIR=data/selfplay/run1/replays \
  .venv-metamon/bin/python -m eval.run --mode h2h --format gen9randombattle \
    --server local --n-games 100 --paired \
    --agent-a foul_play_root_priors --agent-b foul_play \
    --agent-a-python .venv-fp-priors/bin/python \
    --foul-play-search-time-ms 250 \
    --run-id selfplay_run1 --phase exit --change-name selfplay_pilot \
    --json-out data/selfplay/run1/result.json

The replay capture is handled by patching run_foul_play.py to save
the full protocol log on game end.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
