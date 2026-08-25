#!/usr/bin/env python3
"""Read-only completeness audit for frozen Cycle 17 raw search rows."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from experimental.src.scripts import run_cycle15_teacher_stability as c15


def modal(cells):
    return c15._modal(cells)


def root_topk(cells, k=3):
    actions = sorted({a for cell in cells for a in cell["policy"]})
    means = {a: statistics.fmean(cell["policy"].get(a, 0.0) for cell in cells) for a in actions}
    return sorted(actions, key=lambda a: (-means[a], a))[:k]


def world_top(world):
    return min(world["result"]["side_one"], key=lambda row: (-row["N"], row["action"]))["action"]


def main(run_dir: Path) -> None:
    roots = [json.loads((run_dir / "measurement/workers" / f"{i:03d}.json").read_text()) for i in range(40)]
    names = ["production_exact", "production_paired", *c15.ARMS]
    detailed = {}
    integrity_failures = []
    for name in names:
        top3_contains_20k = []
        world_disagreement = []
        for root_index, root in enumerate(roots):
            cells = root[name] if name.startswith("production") else root["arms"][name]
            reference = modal(root["arms"]["equal_20000"])
            top3_contains_20k.append(reference in root_topk(cells))
            for cell in cells:
                world_disagreement.append(statistics.fmean(
                    world_top(world) != cell["top1"] for world in cell["worlds"]
                ))
                for world in cell["worlds"]:
                    side_one = world["result"]["side_one"]
                    if {row["action"] for row in side_one} != set(root["request_actions"]):
                        integrity_failures.append([root_index, name, "legal_support"])
                    for row in [*side_one, *world["result"]["side_two"]]:
                        if (row["N"] == 0) != (row["Q"] is None):
                            integrity_failures.append([root_index, name, "missing_q"])
                        if row["N"] and not math.isfinite(row["Q"]):
                            integrity_failures.append([root_index, name, "nonfinite_q"])
        detailed[name] = {
            "root_top3_contains_equal_20k_top1": statistics.fmean(top3_contains_20k),
            "mean_within_cell_world_top1_disagreement": statistics.fmean(world_disagreement),
        }
    agreement = 0; stable_agreement = 0
    for root in roots:
        eight, twenty, prod = root["arms"]["equal_8192"], root["arms"]["equal_20000"], root["production_exact"]
        agrees_differs = modal(eight) == modal(twenty) and modal(eight) != modal(prod)
        agreement += agrees_differs
        stable_agreement += agrees_differs and all(len({cell["top1"] for cell in rows}) == 1 for rows in (eight, twenty, prod))
    report = {
        "schema": "metagross-cycle17-completeness-audit/v1",
        "status": "pass" if not integrity_failures else "fail",
        "integrity_failures": integrity_failures,
        "detailed_metrics": detailed,
        "equal_8k_and_20k_agree_while_P_exact_differs": agreement,
        "all_three_all_cell_stable_subset": stable_agreement,
        "P_paired_modal_top1_agreement_with_P_exact": statistics.fmean(
            modal(root["production_paired"]) == modal(root["production_exact"]) for root in roots
        ),
        "raw_worker_files": 40,
        "claim": "descriptive completeness audit only; frozen admission decision unchanged",
    }
    path = run_dir / "measurement/COMPLETENESS_AUDIT.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True)
    main(parser.parse_args().run_dir.resolve())
