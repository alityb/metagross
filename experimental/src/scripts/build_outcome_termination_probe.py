#!/usr/bin/env python3
"""Build a baseline-only panel for preregistered termination feasibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.collect_outcome_grounded_continuations import read_panel, sha256, write_private


def build(args: argparse.Namespace) -> dict:
    rows = read_panel(args.panel)
    probe = [
        {
            **row,
            "candidate_actions": [row["baseline_action"]],
            "termination_probe": {
                "source_panel_sha256": sha256(args.panel),
                "selection_uses_outcomes": False,
            },
        }
        for row in rows
    ]
    write_private(args.output, probe)
    report = {
        "schema": "metagross-outcome-termination-probe-panel-report/v1",
        "roots": len(probe),
        "source_panel_sha256": sha256(args.panel),
        "probe_panel_sha256": sha256(args.output),
        "candidate_actions_per_root": 1,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
