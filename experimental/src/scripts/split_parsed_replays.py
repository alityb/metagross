#!/usr/bin/env python3
"""Deterministic game-level train/held-out split for parsed randbats trajectories.

Split is BY GAME (both POVs of a game go to the same side) via a stable hash of
the gameid, so re-running after new parses routes consistently and training can
never see a held-out game. Held-out files are physically moved to a sibling dir.

Usage:
  .venv-metamon/bin/python scripts/split_parsed_replays.py \
      --parsed-dir data/parsed_replays/gen9randombattle \
      --heldout-dir data/parsed_replays_heldout/gen9randombattle \
      --heldout-frac 0.10
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def heldout(gameid: str, frac: float) -> bool:
    h = int(hashlib.sha256(gameid.encode()).hexdigest(), 16)
    return (h % 10_000) < int(frac * 10_000)


def rating_of(filename: str) -> int | None:
    parts = filename.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None  # "Unrated"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-dir", default="data/parsed_replays/gen9randombattle")
    parser.add_argument("--heldout-dir", default="data/parsed_replays_heldout/gen9randombattle")
    parser.add_argument("--heldout-frac", type=float, default=0.10)
    args = parser.parse_args()

    parsed = Path(args.parsed_dir)
    held = Path(args.heldout_dir)
    held.mkdir(parents=True, exist_ok=True)

    moved = 0
    kept = 0
    ratings = Counter()
    for f in sorted(parsed.glob("*.json.lz4")):
        gameid = f.name.split("_")[0]
        r = rating_of(f.name)
        ratings[(r // 100 * 100) if r else "unrated"] += 1
        if heldout(gameid, args.heldout_frac):
            f.rename(held / f.name)
            moved += 1
        else:
            kept += 1

    summary = {
        "train_files": kept,
        "heldout_files_moved_now": moved,
        "heldout_total": len(list(held.glob("*.json.lz4"))),
        "rating_histogram": {str(k): v for k, v in sorted(ratings.items(), key=str)},
    }
    print(json.dumps(summary, indent=2))
    (held.parent / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
