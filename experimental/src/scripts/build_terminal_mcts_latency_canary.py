#!/usr/bin/env python3
"""Freeze one opened top-two root for the terminal-MCTS latency canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    eligible = sorted(
        (row for row in rows if len(row.get("candidate_actions", [])) >= 2),
        key=lambda row: (str(row["root_id"]), str(row["battle_id"])),
    )
    if not eligible:
        raise ValueError("source has no top-two root")
    row = eligible[0]
    frozen = {
        **row,
        "candidate_actions": list(row["candidate_actions"][:2]),
        "teacher_action": row["baseline_action"],
        "selection": {
            **row.get("selection", {}),
            "purpose": "opened_development_latency_canary",
            "selection_rule": "lexicographically_first_root_top_two_prefix",
        },
    }
    if frozen["candidate_actions"][0] != frozen["baseline_action"]:
        raise ValueError("source top-two prefix does not begin with baseline")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, sort_keys=True, separators=(",", ":")) + "\n")
    report = {
        "schema": "metagross-terminal-mcts-latency-canary-panel/v1",
        "source_sha256": sha256(args.source),
        "output_sha256": sha256(args.output),
        "source_roots": len(rows),
        "roots": 1,
        "root_id": frozen["root_id"],
        "battle_id": frozen["battle_id"],
        "candidate_actions": frozen["candidate_actions"],
        "confirmation_rows_read": 0,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
