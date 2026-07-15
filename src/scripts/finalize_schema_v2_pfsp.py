#!/usr/bin/env python3
"""Fail-closed finalization of nested schema-v2 high-budget PFSP shards.

This is intentionally post-collection only. It validates, parses, filters, and
indexes files already on disk; it never invokes collection, training, or a
shell command.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_mcts_policy_sidecar import build_sidecar
from scripts.build_mcts_trajectory_index import build_trajectory_index
from scripts.filter_learner_pov import filter_learner_povs
from scripts.parse_randbats_replays import parse_replay_dir
from scripts.validate_strict_shard import validate_strict_shard


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _temporary_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        return Path(handle.name)


def discover_shards(raw_root: Path) -> list[Path]:
    """Return every directory that looks like a schema-v2 collection shard."""
    candidates = {path.parent for path in raw_root.rglob("agent_a_decisions.jsonl")}
    candidates.update(path.parent for path in raw_root.rglob("replays") if path.is_dir())
    return sorted(candidates, key=lambda path: path.relative_to(raw_root).as_posix())


def _record_error(errors: list[str], stage: str, shard: Path | None, exc: BaseException | str) -> None:
    location = f" {shard}" if shard is not None else ""
    errors.append(f"{stage}{location}: {exc}")


def _schema_v2_decisions(path: Path) -> tuple[int, int]:
    """Count decision rows and rows that do not meet the schema-v2 contract."""
    decisions = 0
    invalid = 0
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if row.get("record_type") != "decision":
            continue
        decisions += 1
        if row.get("mcts_schema_version") != 2 or row.get("learner_pov") != row.get("username"):
            invalid += 1
    return decisions, invalid


def finalize(
    raw_root: Path,
    parsed_root: Path,
    learner_only_root: Path,
    trajectory_index: Path,
    sidecar: Path,
    pool_path: Path,
    report_path: Path,
    min_decisions: int = 1,
    min_opponent_prior_coverage: float | None = None,
) -> dict[str, Any]:
    """Finalize all nested shards and return a report, even on failure."""
    raw_root = raw_root.resolve()
    parsed_root = parsed_root.resolve()
    learner_only_root = learner_only_root.resolve()
    errors: list[str] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "raw_root": str(raw_root),
            "parsed_root": str(parsed_root),
            "learner_only_root": str(learner_only_root),
            "trajectory_index": str(trajectory_index.resolve()),
            "sidecar": str(sidecar.resolve()),
            "pool_path": str(pool_path.resolve()),
        },
        "stages": {
            "discovery": {"shards_discovered": 0},
            "validation": {"passed": 0, "failed": 0, "decision_records": 0, "shards": {}},
            "schema_v2": {"decision_records": 0, "invalid": 0, "shards": {}},
            "parse": {"parsed_ok": 0, "failed": 0, "total_pov_trajectories": 0, "shards": {}},
            "learner_filter": {"learner_trajectories": 0, "shards": {}},
            "trajectory_index": {"trajectories": 0},
            "sidecar": {"accepted": 0, "rejected": 0, "invalid_rows": 0},
        },
        "integrity_errors": errors,
        "ok": False,
    }
    try:
        if not raw_root.is_dir():
            _record_error(errors, "preflight", None, f"raw root is missing: {raw_root}")
        if not pool_path.is_file():
            _record_error(errors, "preflight", None, f"exact RandBats pool is missing: {pool_path}")
        if parsed_root == learner_only_root:
            _record_error(errors, "preflight", None, "parsed and learner-only roots must differ")
        if not errors:
            shards = discover_shards(raw_root)
            report["stages"]["discovery"]["shards_discovered"] = len(shards)
            if not shards:
                _record_error(errors, "discovery", None, "no nested shard directories found")
            valid_shards: list[Path] = []
            for shard in shards:
                relative = shard.relative_to(raw_root).as_posix()
                if not (shard / "replays").is_dir() or not (shard / "agent_a_decisions.jsonl").is_file():
                    report["stages"]["validation"]["failed"] += 1
                    _record_error(errors, "validation", shard, "requires replays/ and agent_a_decisions.jsonl")
                    continue
                try:
                    manifest = validate_strict_shard(
                        shard,
                        min_decisions,
                        min_opponent_prior_coverage,
                    )
                except (Exception, SystemExit) as exc:
                    report["stages"]["validation"]["failed"] += 1
                    _record_error(errors, "validation", shard, exc)
                    continue
                report["stages"]["validation"]["passed"] += 1
                report["stages"]["validation"]["decision_records"] += manifest["decision_records"]
                report["stages"]["validation"]["shards"][relative] = manifest
                schema_decisions, schema_invalid = _schema_v2_decisions(shard / "agent_a_decisions.jsonl")
                report["stages"]["schema_v2"]["decision_records"] += schema_decisions
                report["stages"]["schema_v2"]["invalid"] += schema_invalid
                report["stages"]["schema_v2"]["shards"][relative] = {
                    "decision_records": schema_decisions,
                    "invalid": schema_invalid,
                }
                if schema_invalid:
                    _record_error(errors, "schema_v2", shard, f"{schema_invalid} invalid decision records")
                valid_shards.append(shard)

            # Do not turn an invalid collection into partially finalized data.
            if not errors:
                decision_logs: list[Path] = []
                for shard in valid_shards:
                    relative = shard.relative_to(raw_root)
                    parsed_dir = parsed_root / relative
                    learner_dir = learner_only_root / relative
                    try:
                        parsed = parse_replay_dir(shard / "replays", parsed_dir, pool_path, workers=1)
                        report["stages"]["parse"]["shards"][relative.as_posix()] = parsed
                        for key in ("parsed_ok", "failed", "total_pov_trajectories"):
                            report["stages"]["parse"][key] += parsed[key]
                        expected_povs = 2 * report["stages"]["validation"]["shards"][relative.as_posix()]["raw_replay_files"]
                        if parsed["failed"] or parsed["total_pov_trajectories"] != expected_povs:
                            _record_error(
                                errors,
                                "parse",
                                shard,
                                f"expected {expected_povs} POVs, found {parsed['total_pov_trajectories']}",
                            )
                            continue
                        filtered = filter_learner_povs(shard / "replays", parsed_dir, learner_dir)
                        report["stages"]["learner_filter"]["shards"][relative.as_posix()] = filtered
                        report["stages"]["learner_filter"]["learner_trajectories"] += filtered[
                            "learner_trajectories"
                        ]
                        if (
                            filtered["malformed_parsed_names"]
                            or filtered["raw_learner_povs"] != report["stages"]["validation"]["shards"][relative.as_posix()]["raw_replay_files"]
                            or filtered["learner_trajectories"] != filtered["raw_learner_povs"]
                        ):
                            _record_error(errors, "learner_filter", shard, "incomplete learner POV coverage")
                            continue
                        decision_logs.append(shard / "agent_a_decisions.jsonl")
                    except Exception as exc:
                        _record_error(errors, "parse_or_filter", shard, exc)

                if not errors:
                    temporary_index = _temporary_output(trajectory_index)
                    temporary_sidecar = _temporary_output(sidecar)
                    try:
                        indexed = build_trajectory_index(learner_only_root, temporary_index)
                        report["stages"]["trajectory_index"] = indexed
                        if indexed["trajectories"] != report["stages"]["learner_filter"]["learner_trajectories"]:
                            _record_error(errors, "trajectory_index", None, "ambiguous or stale learner trajectory identities")
                        else:
                            sidecar_result = build_sidecar(
                                decision_logs,
                                learner_only_root,
                                temporary_sidecar,
                                temporary_index,
                            )
                            report["stages"]["sidecar"] = sidecar_result
                            if (
                                sidecar_result["accepted"] == 0
                                or sidecar_result["rejected"]
                                or sidecar_result["invalid_rows"]
                            ):
                                _record_error(errors, "sidecar", None, "not every decision produced a verified target")
                            else:
                                os.replace(temporary_index, trajectory_index)
                                os.replace(temporary_sidecar, sidecar)
                    except Exception as exc:
                        _record_error(errors, "index_or_sidecar", None, exc)
                    finally:
                        for temporary in (temporary_index, temporary_sidecar):
                            if temporary.exists():
                                temporary.unlink()
    finally:
        report["ok"] = not errors and report["stages"]["sidecar"]["accepted"] > 0
        _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--parsed-root", required=True, type=Path)
    parser.add_argument("--learner-only-root", required=True, type=Path)
    parser.add_argument("--trajectory-index", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--pool-path", required=True, type=Path, help="Exact RandBats generator pool used for parsing.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--min-decisions", type=int, default=1)
    parser.add_argument("--min-opponent-prior-coverage", type=float, default=None)
    args = parser.parse_args()
    report = finalize(
        args.raw_root,
        args.parsed_root,
        args.learner_only_root,
        args.trajectory_index,
        args.sidecar,
        args.pool_path,
        args.report,
        args.min_decisions,
        args.min_opponent_prior_coverage,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit("PFSP finalization rejected; see the JSON report")


if __name__ == "__main__":
    main()
