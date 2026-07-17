#!/usr/bin/env python3
"""Finalize a trainable schema-v3 snapshot while collection continues.

Merges per-shard target files from build_mcts_v3_dataset.py and retains only
the matching learner POV trajectories from the corresponding parsed shard.
Targets without a parsed trajectory remain useful to the stateless v3
auxiliary loss; they are counted explicitly rather than silently discarded.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def pov_from_filename(path: Path) -> tuple[str, str] | None:
    name = path.name.removesuffix(".json.lz4")
    try:
        battle_tag, _, fields = name.split("_", 2)
        username = fields.removeprefix("Unrated_").split("_vs_", 1)[0]
    except ValueError:
        return None
    return battle_tag.removeprefix("battle-"), username


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def finalize(targets_root: Path, parsed_root: Path, output_targets: Path, learner_root: Path) -> dict:
    target_paths = sorted(targets_root.rglob("*.jsonl"))
    if not target_paths:
        raise ValueError("no target JSONL files found")
    if output_targets.exists() or learner_root.exists():
        raise ValueError("output paths must be fresh")

    report: dict = {
        "target_files": len(target_paths), "targets": 0,
        "target_groups": 0, "learner_trajectories": 0,
        "target_groups_without_parsed_trajectory": 0,
        "parse_only_trajectories": 0, "errors": [],
    }
    output_targets.parent.mkdir(parents=True, exist_ok=True)
    with output_targets.open("w", encoding="utf-8") as output:
        for target_path in target_paths:
            relative = target_path.relative_to(targets_root).with_suffix("")
            parsed_dir = parsed_root / relative
            groups: set[tuple[str, str]] = set()
            rows = []
            for line_no, line in enumerate(target_path.read_text().splitlines(), 1):
                row = json.loads(line)
                if row.get("schema") != 3:
                    raise ValueError(f"{target_path}:{line_no}: wrong schema")
                key = row.get("battle_tag"), row.get("username")
                if not all(isinstance(value, str) and value for value in key):
                    raise ValueError(f"{target_path}:{line_no}: invalid group key")
                groups.add(key)
                rows.append(row)
            for row in rows:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
            report["targets"] += len(rows)
            report["target_groups"] += len(groups)

            parsed = {}
            if parsed_dir.is_dir():
                for path in parsed_dir.glob("*.json.lz4"):
                    key = pov_from_filename(path)
                    if key is not None and key not in parsed:
                        parsed[key] = path
            for key in groups:
                source = parsed.pop(key, None)
                if source is None:
                    report["target_groups_without_parsed_trajectory"] += 1
                    continue
                link_or_copy(source, learner_root / relative / source.name)
                report["learner_trajectories"] += 1
            report["parse_only_trajectories"] += len(parsed)
    if report["targets"] == 0 or report["learner_trajectories"] == 0:
        raise ValueError("snapshot contains no usable targets or learner trajectories")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-root", required=True, type=Path)
    parser.add_argument("--parsed-root", required=True, type=Path)
    parser.add_argument("--output-targets", required=True, type=Path)
    parser.add_argument("--learner-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = finalize(args.targets_root, args.parsed_root, args.output_targets, args.learner_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
