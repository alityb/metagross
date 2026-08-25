#!/usr/bin/env python3
"""Run Cycle 12's frozen exact-format server-transport repair gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experimental.src.scripts import cycle12_replay_audit as v12
from experimental.src.scripts import run_cycle10_full_corpus_index as v10
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest


SPLIT_DOMAIN = "cycle12-split-20260815"
SCHEMA = "metagross-cycle12-compact-battle-index/v1"
_BASE_PROCESS_ONE = v10.process_one


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _install_contract() -> None:
    v10.v9 = v12
    v10.digest = digest
    v10.SPLIT_DOMAIN = SPLIT_DOMAIN
    v10.SCHEMA = SCHEMA


def process_one(task: tuple[Any, ...]) -> dict[str, Any]:
    _install_contract()
    return _BASE_PROCESS_ONE(task)


def deterministic_match(original: dict[str, Any] | None, repeated: dict[str, Any]) -> bool:
    if not original or original.get("status") != repeated.get("status"):
        return False
    if original["status"] == "pass":
        return (
            original["compact_sha256"] == repeated.get("compact_sha256")
            and original["canonical_public_sha256"] == repeated.get("canonical_public_sha256")
            and original["execution_sha256"] == repeated.get("execution_sha256")
        )
    return (
        original.get("failure_class") == repeated.get("failure_class")
        and original.get("failure_detail_sha256") == repeated.get("failure_detail_sha256")
        and original.get("relative_index") is None
        and repeated.get("relative_index") is None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--spot", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--showdown-repo", type=Path, required=True)
    parser.add_argument("--worktree-map", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers != 8:
        raise ValueError("frozen Cycle 12 worker count is exactly eight")
    _install_contract()
    v10.process_one = process_one
    manifest = verify_manifest(args.manifest)
    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines() if line]
    spot = [json.loads(line) for line in args.spot.read_text().splitlines() if line]
    exclusions = [json.loads(line) for line in args.exclusions.read_text().splitlines() if line]
    positives = [row for row in corpus if row["commit_present"]]
    negatives = [row for row in corpus if not row["commit_present"]]
    if len(corpus) != 20564 or len(positives) != 20560 or len(negatives) != 4:
        raise ValueError("frozen Cycle 12 corpus cardinality changed")
    if len(spot) != 256 or len(exclusions) != 69:
        raise ValueError("frozen Cycle 12 spot/exclusion cardinality changed")
    v10.verify_source_trees(corpus, manifest)
    worktrees = v10.load_json(args.worktree_map)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    diagnostic_root = args.output_dir / "failures"
    negative_passes = 0
    for row in negatives:
        present = subprocess.run(
            ["git", "-C", str(args.showdown_repo), "cat-file", "-e", f"{row['showdown_commit']}^{{commit}}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if not present and v10.v8.sha256_path(Path(row["raw_path"])) == row["raw_sha256"]:
            negative_passes += 1
    main_tasks = [
        (index, row, worktrees[row["showdown_commit"]], str(args.harness.resolve()),
         str(args.output_dir), str(diagnostic_root), True)
        for index, row in enumerate(corpus) if row["commit_present"]
    ]
    results = v10.run_tasks(main_tasks, args.workers, "full_corpus")
    by_battle = {row["battle_id"]: row for row in results}
    spot_root = args.output_dir / "determinism-repeat"
    spot_tasks = [
        (index, row, worktrees[row["showdown_commit"]], str(args.harness.resolve()),
         str(spot_root), str(args.output_dir / "determinism-failures"), False)
        for index, row in enumerate(spot)
    ]
    spot_results = v10.run_tasks(spot_tasks, args.workers, "determinism_repeat")
    deterministic_pass = sum(
        deterministic_match(by_battle.get(row["battle_id"]), repeated)
        for row, repeated in zip(spot, spot_results, strict=True)
    )
    deterministic_admitted = sum(
        by_battle.get(row["battle_id"], {}).get("status") == "pass"
        for row in spot
    )
    if spot_root.exists():
        shutil.rmtree(spot_root)
    totals = Counter(row["commit"] for row in results)
    passed_counts = Counter(row["commit"] for row in results if row["status"] == "pass")
    by_commit = {
        commit: {"total": total, "passed": passed_counts[commit],
                 "failed": total - passed_counts[commit],
                 "coverage": passed_counts[commit] / total, "major": total >= 100}
        for commit, total in sorted(totals.items())
    }
    failures = Counter(row["failure_class"] for row in results if row["status"] == "fail")
    details: dict[str, set[str]] = defaultdict(set)
    for row in results:
        if row["status"] == "fail":
            details[row["failure_class"]].add(row["failure_detail_sha256"])
    unknown = sum(count for name, count in failures.items()
                  if name.startswith(("internal_unclassified:", "unknown_semantic:")))
    passed = sum(row["status"] == "pass" for row in results)
    coverage = passed / len(positives)
    commit_gate = all(row["coverage"] >= 0.99 for row in by_commit.values() if row["major"])
    cluster_report = v10.assign_clusters(results, args.output_dir)
    post_run_integrity = "pass"
    integrity_detail = None
    try:
        post_manifest = verify_manifest(args.manifest)
        v10.verify_source_trees(corpus, post_manifest)
    except BaseException as exc:
        post_run_integrity = "fail"
        integrity_detail = digest(exc.__class__.__name__ + ":" + str(exc))
    status = "pass" if (
        negative_passes == 4 and coverage >= 0.99 and commit_gate and unknown == 0
        and deterministic_pass == 256
        and cluster_report["cross_split_cluster_leakage"] == 0
        and post_run_integrity == "pass"
    ) else "fail"
    report = {
        "schema": "metagross-cycle12-full-corpus-report/v1", "status": status,
        "corpus_rows": len(corpus), "positive_rows": len(positives),
        "custom_rule_exclusions": len(exclusions),
        "excluded_by_format_sha256": dict(sorted(Counter(
            row["formatid_sha256"] for row in exclusions
        ).items())),
        "positive_passed": passed, "positive_failed": len(positives) - passed,
        "overall_coverage": coverage, "negative_controls_passed": negative_passes,
        "by_commit": by_commit, "failure_classes": dict(sorted(failures.items())),
        "failure_detail_hashes_by_class": {
            key: sorted(values) for key, values in sorted(details.items())
        },
        "unknown_failure_count": unknown,
        "determinism_spot_total": 256, "determinism_spot_passed": deterministic_pass,
        "determinism_spot_admitted": deterministic_admitted,
        "determinism_spot_abstained": 256 - deterministic_admitted,
        "indexed_states": sum(row.get("state_count", 0) for row in results),
        "dependency_index": cluster_report,
        "post_run_frozen_integrity": post_run_integrity,
        "post_run_integrity_detail_sha256": integrity_detail,
        "workers": args.workers,
        "manifest_sha256": v10.v8.sha256_path(args.manifest),
        "corpus_sha256": v10.v8.sha256_path(args.corpus),
        "spot_sha256": v10.v8.sha256_path(args.spot),
        "exclusions_sha256": v10.v8.sha256_path(args.exclusions),
        "teacher_q_visit_fields_opened": 0, "training_rows_written": 0,
        "sealed_93_rows_read": 0, "cloud_gpu_paid_cost_usd": 0,
    }
    report_path = args.output_dir / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
