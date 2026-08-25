#!/usr/bin/env python3
"""Apply a termination-only prefilter without observing outcome direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.collect_outcome_grounded_continuations import read_panel, sha256, write_private
from train.outcome_grounded import RESULT_SCHEMA


def filter_panel(args: argparse.Namespace) -> dict:
    panel = read_panel(args.panel)
    probe_panel = read_panel(args.probe_panel)
    source_by_root = {row["root_id"]: row for row in panel}
    if set(source_by_root) != {row["root_id"] for row in probe_panel}:
        raise ValueError("probe panel root set differs from source panel")
    if any(
        row["candidate_actions"] != [source_by_root[row["root_id"]]["baseline_action"]]
        for row in probe_panel
    ):
        raise ValueError("probe panel is not baseline-only")
    results = [json.loads(line) for line in args.probe_results.read_text().splitlines() if line.strip()]
    by_root: dict[str, list[dict]] = {}
    for row in results:
        if row.get("schema") != RESULT_SCHEMA or row.get("root_id") not in source_by_root:
            raise ValueError("invalid termination probe result")
        if row.get("candidate_actions") != [row.get("baseline_action")]:
            raise ValueError("termination probe evaluated a non-baseline action")
        by_root.setdefault(row["root_id"], []).append(row)
    kept = []
    rejected = []
    for root_id, root in source_by_root.items():
        rows = by_root.get(root_id, [])
        if len(rows) != 2 or {int(row["schedule_id"]) for row in rows} != {0, 1}:
            raise ValueError(f"termination probe is incomplete for root {root_id}")
        samples = [
            sample
            for row in rows
            for sample in row["action_outcomes"][root["baseline_action"]]
        ]
        if len(samples) != 16:
            raise ValueError(f"termination probe sample count changed for root {root_id}")
        terminal = sum(sample.get("outcome") is not None for sample in samples)
        if terminal == len(samples):
            kept.append(root)
        else:
            rejected.append({"root_id": root_id, "terminal": terminal, "samples": len(samples)})
    if len(kept) < args.minimum_roots:
        raise ValueError(f"termination prefilter retained only {len(kept)} roots")
    write_private(args.output, kept)
    report = {
        "schema": "metagross-outcome-termination-prefilter-report/v1",
        "source_panel_sha256": sha256(args.panel),
        "probe_panel_sha256": sha256(args.probe_panel),
        "probe_results_sha256": sha256(args.probe_results),
        "output_sha256": sha256(args.output),
        "source_roots": len(panel),
        "retained_roots": len(kept),
        "rejected_roots": len(rejected),
        "criterion": "16_of_16_baseline_probe_trajectories_terminal",
        "rejections": rejected,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--probe-panel", type=Path, required=True)
    parser.add_argument("--probe-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-roots", type=int, default=50)
    print(json.dumps(filter_panel(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
