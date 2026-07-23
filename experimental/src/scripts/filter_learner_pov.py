#!/usr/bin/env python3
"""Keep only the capture-side POVs from parsed PFSP replays.

The replay parser emits both player perspectives. PFSP collection captures raw
replays from the learner side, recorded as ``_our_name`` in every raw replay;
this script retains only that learner trajectory.
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import shutil
from pathlib import Path


def filter_learner_povs(
    raw_dir: Path,
    parsed_dir: Path,
    out_dir: Path,
    index_path: Path | None = None,
) -> dict[str, int]:
    """Link or copy only capture-side trajectories from one flat shard."""
    learner_povs: set[tuple[str, str]] = set()
    for raw_path in raw_dir.glob("*.json"):
        try:
            raw = json.loads(raw_path.read_text())
        except json.JSONDecodeError:
            continue
        battle_id = str(raw.get("id") or raw_path.stem)
        learner = raw.get("_our_name")
        if learner:
            learner_povs.add((battle_id, str(learner)))

    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    malformed_names = 0
    for parsed_path in parsed_dir.glob("*.lz4"):
        name = parsed_path.name
        try:
            battle_id, _, replay_fields = name.split("_", 2)
            pov = replay_fields.removeprefix("Unrated_").split("_vs_", 1)[0]
        except ValueError:
            malformed_names += 1
            continue
        if (battle_id, pov) not in learner_povs:
            continue
        destination = out_dir / name
        if destination.exists():
            if not filecmp.cmp(parsed_path, destination, shallow=False):
                raise ValueError(f"existing learner trajectory differs from source: {destination}")
        else:
            try:
                os.link(parsed_path, destination)
            except OSError:
                shutil.copy2(parsed_path, destination)
        kept.append(f"gen9randombattle/{name}")

    if index_path is not None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename"])
            writer.writerows([[name] for name in sorted(set(kept))])
    return {
        "raw_learner_povs": len(learner_povs),
        "parsed_input": len(list(parsed_dir.glob("*.lz4"))),
        "learner_trajectories": len(set(kept)),
        "malformed_parsed_names": malformed_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--parsed-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(filter_learner_povs(args.raw_dir, args.parsed_dir, args.out_dir, args.out_dir.parent / "index.csv"))


if __name__ == "__main__":
    main()
