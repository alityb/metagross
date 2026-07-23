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
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
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
        if row.get("mcts_schema_version") not in (2, 3) or row.get("learner_pov") != row.get("username"):
            invalid += 1
    return decisions, invalid


def _parsed_pov(path: Path) -> str | None:
    """Extract the parser POV from its stable replay filename format."""
    try:
        _, _, replay_fields = path.name[:-9].split("_", 2)
        return replay_fields.removeprefix("Unrated_").split("_vs_", 1)[0]
    except ValueError:
        return None


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _fresh_replay_admission_outputs(
    parsed_root: Path,
    learner_only_root: Path,
    trajectory_index: Path,
    sidecar: Path,
) -> list[Path]:
    """Replay admission never appends to, links into, or replaces prior output."""
    return [path for path in (parsed_root, learner_only_root, trajectory_index, sidecar) if path.exists()]


def _write_decision_stream(
    path: Path,
    rows_by_pov: dict[tuple[str, str], list[dict[str, Any]]],
    retained: set[tuple[str, str]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pov in sorted(retained):
            for row in rows_by_pov[pov]:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _finalize_replay_admission(
    raw_root: Path,
    valid_shards: list[Path],
    parsed_root: Path,
    learner_only_root: Path,
    trajectory_index: Path,
    sidecar: Path,
    pool_path: Path,
    report: dict[str, Any],
    errors: list[str],
) -> None:
    """Admit only whole, independently verified learner replay trajectories."""
    admission: dict[str, Any] = {
        "raw": 0,
        "parser_valid": 0,
        "learner_valid": 0,
        "target_valid": 0,
        "parser_valid_replays": [],
        "learner_valid_replays": [],
        "target_valid_replays": [],
        "exclusions": [],
        "exclusions_by_reason": {},
        "candidate_decision_records": 0,
        "filtered_decision_records": 0,
    }
    exclusions: Counter[str] = Counter()

    def exclude(relative: Path, battle_tag: str, reason: str, learner_pov: str | None = None) -> None:
        entry: dict[str, str] = {
            "shard": relative.as_posix(),
            "battle_tag": battle_tag,
            "reason": reason,
        }
        if learner_pov is not None:
            entry["learner_pov"] = learner_pov
        admission["exclusions"].append(entry)
        exclusions[reason] += 1

    candidates: list[dict[str, Any]] = []
    rows_by_pov: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for shard in valid_shards:
        relative = shard.relative_to(raw_root)
        relative_text = relative.as_posix()
        parsed_dir = parsed_root / relative
        try:
            parsed = parse_replay_dir(shard / "replays", parsed_dir, pool_path, workers=1)
            report["stages"]["parse"]["shards"][relative_text] = parsed
            for key in ("parsed_ok", "failed", "total_pov_trajectories"):
                report["stages"]["parse"][key] += parsed[key]
        except Exception as exc:
            _record_error(errors, "parse", shard, exc)
            continue

        for raw_path in sorted((shard / "replays").glob("*.json")):
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            battle_tag = str(raw.get("id") or raw_path.stem)
            admission["raw"] += 1
            pov_paths = [path for path in parsed_dir.glob("*.json.lz4") if path.name.startswith(f"{battle_tag}_")]
            if len(pov_paths) != 2:
                exclude(relative, battle_tag, "parser_output_count")
                continue
            admission["parser_valid"] += 1
            admission["parser_valid_replays"].append({"shard": relative_text, "battle_tag": battle_tag})
            learner = raw.get("_our_name")
            if not isinstance(learner, str) or not learner:
                exclude(relative, battle_tag, "missing_raw_learner_pov")
                continue
            learner_paths = [path for path in pov_paths if _parsed_pov(path) == learner]
            if len(learner_paths) != 1:
                exclude(relative, battle_tag, "missing_or_ambiguous_learner_pov", learner)
                continue
            pov = (battle_tag, learner)
            candidate = {
                "shard": relative,
                "battle_tag": battle_tag,
                "learner_pov": learner,
                "source": learner_paths[0],
                "pov": pov,
            }
            candidates.append(candidate)
            admission["learner_valid"] += 1
            admission["learner_valid_replays"].append(
                {"shard": relative_text, "battle_tag": battle_tag, "learner_pov": learner}
            )

        for line_number, line in enumerate((shard / "agent_a_decisions.jsonl").read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                _record_error(errors, "decision_stream", shard, f"line {line_number}: invalid JSON")
                continue
            if row.get("record_type") != "decision":
                continue
            battle_tag, username = row.get("battle_tag"), row.get("username")
            if isinstance(battle_tag, str) and isinstance(username, str):
                rows_by_pov[(battle_tag, username)].append(row)

    if errors:
        admission["exclusions_by_reason"] = dict(sorted(exclusions.items()))
        report["stages"]["admission"] = admission
        return

    pov_counts = Counter(candidate["pov"] for candidate in candidates)
    duplicate_povs = {pov for pov, count in pov_counts.items() if count > 1}
    learner_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        relative = candidate["shard"]
        battle_tag = candidate["battle_tag"]
        learner = candidate["learner_pov"]
        if candidate["pov"] in duplicate_povs:
            exclude(relative, battle_tag, "ambiguous_learner_identity", learner)
        elif not rows_by_pov[candidate["pov"]]:
            exclude(relative, battle_tag, "missing_learner_decisions", learner)
        else:
            learner_candidates.append(candidate)

    # The staging root keeps target-invalid trajectories out of the final root.
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="replay-admission-", dir=sidecar.parent) as temporary_dir:
        staging_root = Path(temporary_dir) / "learner"
        candidate_index = Path(temporary_dir) / "trajectory_index.jsonl"
        candidate_stream = Path(temporary_dir) / "candidate_decisions.jsonl"
        candidate_sidecar = Path(temporary_dir) / "candidate_sidecar.jsonl"
        candidate_by_pov = {candidate["pov"]: candidate for candidate in learner_candidates}
        for candidate in learner_candidates:
            _link_or_copy(candidate["source"], staging_root / candidate["shard"] / candidate["source"].name)
        _write_decision_stream(candidate_stream, rows_by_pov, set(candidate_by_pov))
        admission["candidate_decision_records"] = sum(len(rows_by_pov[pov]) for pov in candidate_by_pov)
        try:
            indexed = build_trajectory_index(staging_root, candidate_index)
            if indexed["trajectories"] != len(candidate_by_pov):
                raise ValueError("ambiguous or stale staged learner trajectory identities")
            target_check = build_sidecar([candidate_stream], staging_root, candidate_sidecar, candidate_index)
        except Exception as exc:
            _record_error(errors, "replay_target_check", None, exc)
            admission["exclusions_by_reason"] = dict(sorted(exclusions.items()))
            report["stages"]["admission"] = admission
            return

        rejected = {
            (entry["battle_tag"], entry["learner_pov"]): entry["reason"]
            for entry in target_check["rejected_povs"]
        }
        if target_check["accepted"] + target_check["rejected"] != admission["candidate_decision_records"]:
            _record_error(errors, "replay_target_check", None, "target check did not account for every decision")
        target_candidates: list[dict[str, Any]] = []
        for candidate in learner_candidates:
            reason = rejected.get(candidate["pov"])
            if reason is not None:
                exclude(candidate["shard"], candidate["battle_tag"], f"target_{reason}", candidate["learner_pov"])
            else:
                target_candidates.append(candidate)
                admission["target_valid"] += 1
                admission["target_valid_replays"].append(
                    {
                        "shard": candidate["shard"].as_posix(),
                        "battle_tag": candidate["battle_tag"],
                        "learner_pov": candidate["learner_pov"],
                    }
                )
        if target_check["invalid_rows"] or target_check["rejected"] and not rejected:
            _record_error(errors, "replay_target_check", None, "target check returned an unassigned rejection")
        if not target_candidates:
            _record_error(errors, "replay_target_check", None, "no replay passed target validation")
        if not errors:
            final_stream = Path(temporary_dir) / "retained_decisions.jsonl"
            _write_decision_stream(final_stream, rows_by_pov, {candidate["pov"] for candidate in target_candidates})
            admission["filtered_decision_records"] = sum(
                len(rows_by_pov[candidate["pov"]]) for candidate in target_candidates
            )
            for candidate in target_candidates:
                _link_or_copy(candidate["source"], learner_only_root / candidate["shard"] / candidate["source"].name)
            temporary_index = _temporary_output(trajectory_index)
            temporary_sidecar = _temporary_output(sidecar)
            try:
                indexed = build_trajectory_index(learner_only_root, temporary_index)
                report["stages"]["trajectory_index"] = indexed
                if indexed["trajectories"] != len(target_candidates):
                    _record_error(errors, "trajectory_index", None, "ambiguous or stale learner trajectory identities")
                else:
                    sidecar_result = build_sidecar([final_stream], learner_only_root, temporary_sidecar, temporary_index)
                    report["stages"]["sidecar"] = sidecar_result
                    if sidecar_result["rejected"] or sidecar_result["invalid_rows"]:
                        _record_error(errors, "sidecar", None, "retained decision stream was not fully verified")
                    else:
                        os.replace(temporary_index, trajectory_index)
                        os.replace(temporary_sidecar, sidecar)
            except Exception as exc:
                _record_error(errors, "index_or_sidecar", None, exc)
            finally:
                for temporary in (temporary_index, temporary_sidecar):
                    if temporary.exists():
                        temporary.unlink()
    admission["exclusions_by_reason"] = dict(sorted(exclusions.items()))
    report["stages"]["admission"] = admission


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
    replay_admission: bool = False,
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
        "mode": "replay_admission" if replay_admission else "all_or_nothing",
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
        if replay_admission:
            for path in _fresh_replay_admission_outputs(parsed_root, learner_only_root, trajectory_index, sidecar):
                _record_error(errors, "preflight", None, f"replay admission requires a fresh output path: {path}")
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

            if not errors:
                if replay_admission:
                    _finalize_replay_admission(
                        raw_root,
                        valid_shards,
                        parsed_root,
                        learner_only_root,
                        trajectory_index,
                        sidecar,
                        pool_path,
                        report,
                        errors,
                    )
                else:
                    # Do not turn an invalid collection into partially finalized data.
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
    parser.add_argument(
        "--replay-admission",
        action="store_true",
        help="Salvage only independently parser-, learner-, and target-valid replays into fresh outputs.",
    )
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
        args.replay_admission,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit("PFSP finalization rejected; see the JSON report")


if __name__ == "__main__":
    main()
