#!/usr/bin/env python3
"""Run the frozen Cycle 8 128-replay deterministic rematerialization audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from experimental.src.scripts.cycle8_replay_audit import (
    ReplayAuditError,
    canonical_json,
    canonical_public_lines,
    materialize_role,
    public_lines_from_capture,
    sha256_bytes,
    sha256_path,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n")


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ReplayAuditError("premeasurement manifest has no files")
    for row in files:
        source = Path(row["path"])
        if not source.is_file() or sha256_path(source) != row["sha256"]:
            raise ReplayAuditError(f"frozen file hash mismatch: {source}")
    runtime = manifest.get("showdown_runtime")
    if not isinstance(runtime, list) or not runtime:
        raise ReplayAuditError("premeasurement manifest has no Showdown runtimes")
    for row in runtime:
        worktree = Path(row["path"])
        actual_commit = subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual_commit != row["commit"]:
            raise ReplayAuditError(f"frozen Showdown commit mismatch: {worktree}")
        if tree_sha256(worktree / "dist") != row["dist_tree_sha256"]:
            raise ReplayAuditError(f"frozen Showdown dist hash mismatch: {worktree}")
    return manifest


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def stored_terminal(log: str) -> tuple[str, str]:
    terminal = [("win", line[5:]) for line in log.splitlines() if line.startswith("|win|")]
    terminal += [("tie", "") for line in log.splitlines() if line == "|tie"]
    if len(terminal) != 1:
        raise ReplayAuditError("stored public log lacks one terminal result")
    return terminal[0]


def capture_once(
    *, harness: Path, showdown: Path, raw_path: Path, output: Path,
) -> None:
    subprocess.run(
        [
            "node", str(harness), "--showdown", str(showdown),
            "--input", str(raw_path), "--out-dir", str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def audit_positive(
    *, row: Mapping[str, Any], harness: Path, showdown: Path, battle_dir: Path,
) -> dict[str, Any]:
    raw_path = Path(row["raw_path"])
    if sha256_path(raw_path) != row["raw_sha256"]:
        raise ReplayAuditError("selected raw file hash changed")
    raw = load_json(raw_path)
    inputlog = raw.get("inputlog")
    stored_log = raw.get("log")
    if not isinstance(inputlog, str) or not isinstance(stored_log, str):
        raise ReplayAuditError("selected raw record is incomplete")
    if sha256_bytes(inputlog.encode()) != row["inputlog_sha256"]:
        raise ReplayAuditError("selected inputlog hash changed")
    if sha256_bytes(stored_log.encode()) != row["public_log_sha256"]:
        raise ReplayAuditError("selected public-log hash changed")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(showdown), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual_commit != row["showdown_commit"]:
        raise ReplayAuditError("worktree commit mismatch")

    repeat_results = []
    for repeat in (1, 2):
        capture_dir = battle_dir / f"repeat-{repeat}" / "capture"
        capture_once(harness=harness, showdown=showdown, raw_path=raw_path, output=capture_dir)
        public = load_json(capture_dir / "public.json")
        if not public.get("replay_ended") or not isinstance(public.get("terminal"), Mapping):
            raise ReplayAuditError("replay did not reach terminal state")
        regenerated = public_lines_from_capture(public, inputlog=inputlog)
        stored = canonical_public_lines(stored_log.splitlines(), inputlog=inputlog)
        if regenerated != stored:
            raise ReplayAuditError("normalized public replay mismatch")
        kind, winner = stored_terminal(stored_log)
        captured_winner = public["terminal"].get("winner", "")
        if (kind == "win" and captured_winner != winner) or (kind == "tie" and captured_winner):
            raise ReplayAuditError("terminal result mismatch")

        role_outputs = {}
        for role in ("p1", "p2"):
            pov = load_json(capture_dir / f"{role}.json")
            derived = materialize_role(
                battle_id=str(raw["id"]), role=role, public_capture=public,
                pov_capture=pov, inputlog=inputlog,
            )
            derived_path = battle_dir / f"repeat-{repeat}" / role / "information-states.json"
            write_json(derived_path, derived)
            role_outputs[role] = {
                "derived": derived,
                "capture_sha256": sha256_path(capture_dir / f"{role}.json"),
                "states_sha256": sha256_path(derived_path),
            }
        repeat_results.append({
            "public": regenerated,
            "p1": role_outputs["p1"],
            "p2": role_outputs["p2"],
        })

    first, second = repeat_results
    if first["public"] != second["public"]:
        raise ReplayAuditError("normalized public output is nondeterministic")
    for role in ("p1", "p2"):
        if first[role] != second[role]:
            raise ReplayAuditError(f"{role} private/derived output is nondeterministic")
    states = first["p1"]["derived"]["states"] + first["p2"]["derived"]["states"]
    return {
        "requests": len(states),
        "mapped_commands": sum(state["chosen_action"] is not None for state in states),
        "forced_switch_requests": sum(state["pp_disable_sidecar"]["forced_switch"] for state in states),
        "trapped_requests": sum(state["pp_disable_sidecar"]["trapped"] for state in states),
        "tera_requests": sum(state["pp_disable_sidecar"]["can_tera"] for state in states),
        "revealed_facts": sum(len(state["typed_reveal_ledger"]["facts"]) for state in states),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--showdown-repo", type=Path, required=True)
    parser.add_argument("--worktree-map", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    verify_manifest(args.manifest)
    rows = [json.loads(line) for line in args.panel.read_text().splitlines() if line.strip()]
    if len(rows) != 128 or sum(bool(row["commit_present"]) for row in rows) != 124:
        raise ReplayAuditError("frozen panel cardinality changed")
    worktrees = load_json(args.worktree_map)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    failures: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    completed = 0
    negatives = 0
    for panel_index, row in enumerate(rows):
        if not row["commit_present"]:
            present = subprocess.run(
                ["git", "-C", str(args.showdown_repo), "cat-file", "-e", f"{row['showdown_commit']}^{{commit}}"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0
            if present:
                failures["negative_control_commit_became_available"] += 1
            else:
                negatives += 1
            continue
        try:
            showdown = Path(worktrees[row["showdown_commit"]])
            metrics = audit_positive(
                row=row, harness=args.harness, showdown=showdown,
                battle_dir=args.output_dir / "battles" / f"panel-{panel_index:03d}",
            )
            totals.update(metrics)
            completed += 1
        except (ReplayAuditError, subprocess.CalledProcessError, KeyError, OSError, ValueError) as exc:
            failures[type(exc).__name__ + ":" + str(exc)] += 1

    status = "pass" if completed == 124 and negatives == 4 and not failures else "fail"
    report = {
        "schema": "metagross-cycle8-replay-audit-report/v1",
        "status": status,
        "panel_rows": len(rows),
        "positive_replays_passed": completed,
        "negative_controls_passed": negatives,
        "failures": dict(sorted(failures.items())),
        "totals": dict(sorted(totals.items())),
        "sealed_93_rows_read": 0,
        "teacher_values_opened": 0,
        "training_rows_written": 0,
        "cloud_or_gpu_cost_usd": 0,
        "manifest_sha256": sha256_path(args.manifest),
        "panel_sha256": sha256_path(args.panel),
    }
    report_path = args.output_dir / "REPORT.json"
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
