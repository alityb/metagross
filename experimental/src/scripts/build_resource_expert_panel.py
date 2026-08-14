#!/usr/bin/env python3
"""Freeze battle-disjoint, resource-stratified development and holdout panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from scripts.run_public_mcts_leaf_gate import PANEL_SCHEMA, _load_panel, _sha256


SCHEMA = "metagross-resource-expert-panel-report/v1"


def _priority(seed: int, split: str, root_id: str) -> str:
    return hashlib.sha256(f"{seed}:{split}:{root_id}".encode("ascii")).hexdigest()


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_map(paths: list[Path]) -> dict[str, Path]:
    output = {}
    for path in paths:
        digest = _sha256(path)
        if digest in output:
            raise ValueError("duplicate source decision-log hash")
        output[digest] = path
    return output


def _source_rows(
    panel: list[dict[str, Any]], sources: dict[str, Path]
) -> dict[tuple[str, int], dict[str, Any]]:
    needed: dict[str, set[int]] = {}
    for row in panel:
        digest = str(row["source_file_sha256"])
        if digest in sources:
            needed.setdefault(digest, set()).add(int(row["source_line"]))
    rows = {}
    for digest, line_numbers in needed.items():
        with sources[digest].open(encoding="utf-8", errors="replace") as handle:
            for index, raw in enumerate(handle, 1):
                if index in line_numbers:
                    rows[(digest, index)] = json.loads(raw)
                    if len(rows) == sum(len(values) for values in needed.values()):
                        break
    missing = {
        (digest, line_number)
        for digest, line_numbers in needed.items()
        for line_number in line_numbers
        if (digest, line_number) not in rows
    }
    if missing:
        raise ValueError(f"source decision rows are absent: {sorted(missing)[:3]}")
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    import poke_engine

    if args.battles < 50 or args.battles % 2:
        raise ValueError("each resource panel requires an even count of at least 50 battles")
    panel, input_hash = _load_panel(args.input_panel)
    sources = _source_map(args.decision_log)
    source_rows = _source_rows(panel, sources)
    enriched = []
    for row in panel:
        source = sources.get(str(row["source_file_sha256"]))
        if source is None:
            continue
        source_row = source_rows[(str(row["source_file_sha256"]), int(row["source_line"]))]
        historical = source_row.get("selected_action")
        if not isinstance(historical, str) or not historical:
            continue
        supports = []
        tera = False
        for schedule in row["schedules"]:
            for world in schedule["worlds"]:
                own, _ = poke_engine.root_options(poke_engine.State.from_string(world["state"]))
                supports.append(set(own))
                tera = tera or any(action.endswith("-tera") for action in own)
        common = set.intersection(*supports)
        if historical not in common:
            continue
        enriched.append(
            {
                **row,
                "historical_500ms_action": historical,
                "historical_source": {
                    "battle_tag": source_row.get("battle_tag"),
                    "username": source_row.get("username"),
                    "turn": source_row.get("turn"),
                },
                "resource_stratum": "tera_available" if tera else "tera_spent_or_forbidden",
            }
        )

    selected: dict[str, list[dict[str, Any]]] = {}
    unavailable = set()
    per_stratum = args.battles // 2
    for split in ("development", "holdout"):
        rows = []
        for stratum in ("tera_available", "tera_spent_or_forbidden"):
            candidates = [
                row
                for row in enriched
                if row["battle_id"] not in unavailable and row["resource_stratum"] == stratum
            ]
            candidates.sort(key=lambda row: _priority(args.seed, split, row["root_id"]))
            if len(candidates) < per_stratum:
                raise ValueError(f"insufficient {stratum} roots for {split}")
            rows.extend(candidates[:per_stratum])
        rows.sort(key=lambda row: _priority(args.seed, split, row["root_id"]))
        selected[split] = rows
        unavailable.update(row["battle_id"] for row in rows)

    _write(args.development_output, selected["development"])
    _write(args.holdout_output, selected["holdout"])
    report = {
        "schema": SCHEMA,
        "status": "frozen_before_teacher_execution",
        "input_panel": {"path": str(args.input_panel), "sha256": input_hash},
        "source_decision_logs": [
            {"path": str(path), "sha256": _sha256(path)} for path in args.decision_log
        ],
        "selection_seed": args.seed,
        "split_contract": "one_root_per_battle; development/holdout battle-disjoint; 50/50 Tera stratum",
        "development": {
            "path": str(args.development_output),
            "sha256": _sha256(args.development_output),
            "battles": len(selected["development"]),
        },
        "holdout": {
            "path": str(args.holdout_output),
            "sha256": _sha256(args.holdout_output),
            "battles": len(selected["holdout"]),
        },
        "overlap_battles": len(
            {row["battle_id"] for row in selected["development"]}
            & {row["battle_id"] for row in selected["holdout"]}
        ),
        "eligible_roots": len(enriched),
        "panel_schema": PANEL_SCHEMA,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-panel", type=Path, required=True)
    parser.add_argument("--decision-log", type=Path, action="append", required=True)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--battles", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    print(json.dumps(build(args), sort_keys=True))


if __name__ == "__main__":
    main()
