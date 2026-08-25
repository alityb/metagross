#!/usr/bin/env python3
"""Summarize prospective terminal-MCTS H2H telemetry without opening states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def summarize_log(path: Path) -> dict[str, Any]:
    teacher_calls: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for line_number, row in enumerate(load_jsonl(path), 1):
        choice_override = row.get("choice_override")
        if not isinstance(choice_override, dict):
            continue
        teacher = choice_override.get("terminal_mcts_teacher")
        if isinstance(teacher, dict):
            teacher_calls.append((line_number, teacher, choice_override))

    overrides = []
    failures = []
    elapsed = []
    for line_number, teacher, choice_override in teacher_calls:
        seconds = teacher.get("elapsed_seconds")
        if isinstance(seconds, (int, float)):
            elapsed.append(float(seconds))
        if teacher.get("decision") == "override":
            # `overridden` is relative to the raw R1 action, not to the
            # accepted production search action.  The teacher is applied when
            # final_choice equals its selection and differs from the frozen
            # terminal_mcts_production_choice; it may coincidentally restore
            # the original raw R1 action.
            selected_action = teacher.get("selected_action")
            if choice_override.get("final_choice") != selected_action:
                raise ValueError(
                    f"teacher override was not applied at {path}:{line_number}"
                )
            if choice_override.get("terminal_mcts_production_choice") == selected_action:
                raise ValueError(
                    f"teacher override did not deviate from production at "
                    f"{path}:{line_number}"
                )
            if (
                choice_override.get("terminal_mcts_production_choice")
                != teacher.get("baseline_action")
            ):
                raise ValueError(
                    f"teacher baseline did not match production at {path}:{line_number}"
                )
            if choice_override.get("reason") != "certified_terminal_mcts_override":
                raise ValueError(
                    f"unexpected applied override reason at {path}:{line_number}"
                )
            if choice_override.get("selection_class") != "certified_terminal_teacher":
                raise ValueError(
                    f"unexpected applied selection class at {path}:{line_number}"
                )
            overrides.append(
                {
                    "line": line_number,
                    "baseline_action": teacher.get("baseline_action"),
                    "selected_action": selected_action,
                    "rollouts": teacher.get("rollouts"),
                    "schedule_advantages": teacher.get("schedule_advantages"),
                    "cluster_bootstrap_ci95": teacher.get("cluster_bootstrap_ci95"),
                }
            )
        reason = str(teacher.get("reason", ""))
        if reason.startswith("fail_closed:"):
            failures.append({"line": line_number, "reason": reason})

    return {
        "log": path.name,
        "teacher_calls": len(teacher_calls),
        "override_count": len(overrides),
        "overrides": overrides,
        "fail_closed_count": len(failures),
        "failures": failures,
        "elapsed_mean_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
        "elapsed_max_seconds": max(elapsed) if elapsed else None,
    }


def main() -> None:
    args = parse_args()
    logs = sorted((args.run_dir / "logs").glob("*.search.jsonl"))
    summaries = [summarize_log(path) for path in logs]
    report = {
        "schema": "metagross-terminal-mcts-h2h-telemetry-summary/v1",
        "run_dir": str(args.run_dir),
        "log_count": len(summaries),
        "teacher_calls": sum(row["teacher_calls"] for row in summaries),
        "override_count": sum(row["override_count"] for row in summaries),
        "fail_closed_count": sum(row["fail_closed_count"] for row in summaries),
        "logs": summaries,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
