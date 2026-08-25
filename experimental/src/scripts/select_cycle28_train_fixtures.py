#!/usr/bin/env python3
"""Select label-blind Cycle28 TRAIN fixtures from opened Cycle27 mechanics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def key(row: dict) -> tuple[str, str, int]:
    return row["dependency_cluster_id"], row["role"], row["request_index"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle27-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.cycle27_run.resolve()
    selection = [
        json.loads(line)
        for line in (source / "selection-200.jsonl").read_text().splitlines()
    ]
    results = {
        key(row): row
        for row in map(
            json.loads,
            (source / "mechanics-audit/root-results.jsonl").read_text().splitlines(),
        )
        if row.get("status") == "pass"
    }
    classes = (
        (
            "opening_empty",
            4,
            lambda row: row["intrinsic_revealed_move_count"] == 0
            and row["derived_execution_count"] == 0,
        ),
        ("derived", 7, lambda row: row["derived_execution_count"] > 0),
        (
            "later_intrinsic",
            5,
            lambda row: row["intrinsic_revealed_move_count"] > 0
            and row["derived_execution_count"] == 0,
        ),
    )
    selected = []
    for class_name, count, predicate in classes:
        candidates = [
            (row["state_rank_sha256"], row, results[key(row)])
            for row in selection
            if key(row) in results and predicate(results[key(row)])
        ]
        candidates.sort(key=lambda item: item[0])
        if len(candidates) < count:
            raise RuntimeError(f"insufficient {class_name} TRAIN fixtures")
        for _rank, row, result in candidates[:count]:
            frozen = dict(row)
            frozen.update(
                {
                    "cycle28_fixture_class": class_name,
                    "cycle27_intrinsic_count": result[
                        "intrinsic_revealed_move_count"
                    ],
                    "cycle27_derived_count": result["derived_execution_count"],
                }
            )
            selected.append(frozen)
    if (
        len(selected) != 16
        or len({row["dependency_cluster_id"] for row in selected}) != 16
        or any(row["split"] != "train" for row in selected)
    ):
        raise RuntimeError("Cycle28 fixture selection violates frozen contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "status": "written",
                "fixtures": len(selected),
                "classes": {
                    name: sum(
                        row["cycle28_fixture_class"] == name for row in selected
                    )
                    for name, _count, _predicate in classes
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
