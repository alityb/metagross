#!/usr/bin/env python3
"""Build a deterministic, one-root-per-battle public MCTS panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from belief.randbats_determinize import RandbatsDeterminizer
from scripts.collect_gen9_mcts_leaf_samples import determinize_leaf_state


SCHEMA = "metagross-public-mcts-root-panel/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _source_groups(paths: list[Path], seed: int) -> list[dict[str, Any]]:
    chosen: dict[tuple[str, str, str], tuple[str, dict[str, Any]]] = {}
    for path in paths:
        source_hash = _sha256(path)
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("record_type") != "decision" or not isinstance(row.get("state"), str):
                    continue
                try:
                    turn = int(row.get("turn", 0))
                except (TypeError, ValueError):
                    turn = 0
                if turn < 3:
                    continue
                group = (source_hash, str(row.get("battle_tag")), str(row.get("username")))
                priority = _hash([seed, *group, line_number])
                candidate = {
                    "source_file_sha256": source_hash,
                    "source_line": line_number,
                    "battle_tag": group[1],
                    "username": group[2],
                    "state": row["state"],
                    "turn": turn,
                }
                if group not in chosen or priority < chosen[group][0]:
                    chosen[group] = (priority, candidate)
    return [item[1] for item in chosen.values()]


def _excluded_groups(training_metrics: Path) -> set[tuple[str, str]]:
    report = json.loads(training_metrics.read_text())
    excluded: set[tuple[str, str]] = set()
    for source in report["provenance"]["source_files"]:
        path = Path(source["path"])
        if _sha256(path) != source["sha256"]:
            raise ValueError(f"training source hash changed: {path}")
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("record_type") == "battle_result":
                    excluded.add((str(row.get("battle_tag")), str(row.get("username"))))
    return excluded


def build(args: argparse.Namespace) -> dict[str, Any]:
    import poke_engine

    if args.battles < 50:
        raise ValueError("promotion panels require at least 50 source battles")
    if args.purpose == "evaluation" and args.training_metrics is None:
        raise ValueError("evaluation panels require --training-metrics")
    excluded = (
        _excluded_groups(args.training_metrics)
        if args.training_metrics is not None
        else set()
    )
    excluded_battle_ids = {
        str(row["battle_id"])
        for path in args.exclude_panel
        for row in (
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        )
    }
    candidates = [
        row
        for row in _source_groups(args.decision_log, args.seed)
        if (row["battle_tag"], row["username"]) not in excluded
    ]
    candidates.sort(
        key=lambda row: _hash(
            [args.seed, row["source_file_sha256"], row["battle_tag"], row["username"]]
        )
    )
    rows = []
    failures: dict[str, int] = {}
    determinizer = RandbatsDeterminizer(args.pool, seed=args.seed)
    for candidate in candidates:
        if len(rows) >= args.battles:
            break
        try:
            public_state = poke_engine.State.from_string(candidate["state"])
            public_features = list(poke_engine.compute_public_value_features(public_state))
        except Exception:
            failures["invalid_public_state"] = failures.get("invalid_public_state", 0) + 1
            continue
        battle_id = _hash(
            {
                "source_file_sha256": candidate["source_file_sha256"],
                "battle_tag": candidate["battle_tag"],
                "username": candidate["username"],
            }
        )
        if battle_id in excluded_battle_ids:
            failures["previous_panel_battle"] = failures.get("previous_panel_battle", 0) + 1
            continue
        root_id = _hash(
            {
                "battle_id": battle_id,
                "source_line": candidate["source_line"],
                "public_state_sha256": hashlib.sha256(candidate["state"].encode()).hexdigest(),
            }
        )
        schedules = []
        valid = True
        for schedule_id in range(args.schedules):
            schedule_seed = int.from_bytes(
                hashlib.sha256(f"{args.seed}:{root_id}:{schedule_id}".encode()).digest()[:8],
                "big",
            )
            determinizer.reseed(schedule_seed)
            worlds = []
            for world_index in range(args.worlds):
                completed = determinize_leaf_state(
                    poke_engine.State.from_string(candidate["state"]),
                    poke_engine,
                    determinizer=determinizer,
                )
                if completed is None:
                    valid = False
                    break
                completed_features = list(poke_engine.compute_public_value_features(completed))
                if completed_features != public_features:
                    raise ValueError("public feature contract changed under determinization")
                state_text = completed.to_string()
                worlds.append(
                    {
                        "world_index": world_index,
                        "weight": 1.0 / args.worlds,
                        "state_sha256": hashlib.sha256(state_text.encode()).hexdigest(),
                        "state": state_text,
                    }
                )
            if not valid:
                break
            schedules.append(
                {"schedule_id": schedule_id, "seed": schedule_seed, "worlds": worlds}
            )
        if not valid:
            failures["determinization_failed"] = failures.get("determinization_failed", 0) + 1
            continue
        rows.append(
            {
                "schema": SCHEMA,
                "battle_id": battle_id,
                "root_id": root_id,
                "source_file_sha256": candidate["source_file_sha256"],
                "source_line": candidate["source_line"],
                "battle_turn": candidate["turn"],
                "public_state_sha256": hashlib.sha256(candidate["state"].encode()).hexdigest(),
                "public_features": public_features,
                "schedules": schedules,
            }
        )
    if len(rows) != args.battles:
        raise ValueError(
            f"only built {len(rows)} of {args.battles} required battles; "
            f"candidates={len(candidates)} failures={failures}"
        )
    if len({row["battle_id"] for row in rows}) != len(rows):
        raise ValueError("panel source battles are not unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "schema": "metagross-public-mcts-root-panel-report/v1",
        "battles": len(rows),
        "roots": len(rows),
        "schedules": len(rows) * args.schedules,
        "worlds": len(rows) * args.schedules * args.worlds,
        "feature_contract": "metagross-public-information-value-features/v1",
        "training_overlap_groups": 0,
        "previous_panel_overlap_battles": 0,
        "selection_seed": args.seed,
        "failures": failures,
        "panel_sha256": _sha256(args.output),
        "purpose": args.purpose,
        "training_metrics_sha256": (
            _sha256(args.training_metrics) if args.training_metrics is not None else None
        ),
        "excluded_panel_sha256": [_sha256(path) for path in args.exclude_panel],
        "source_file_sha256": [_sha256(path) for path in args.decision_log],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-log", type=Path, action="append", required=True)
    parser.add_argument("--purpose", choices=("training", "evaluation"), default="evaluation")
    parser.add_argument("--training-metrics", type=Path)
    parser.add_argument("--exclude-panel", type=Path, action="append", default=[])
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--battles", type=int, default=60)
    parser.add_argument("--schedules", type=int, default=2)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    print(json.dumps(build(args), sort_keys=True))


if __name__ == "__main__":
    main()
