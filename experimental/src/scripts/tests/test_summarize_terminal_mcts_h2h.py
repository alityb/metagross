from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "summarize_terminal_mcts_h2h.py"
SPEC = importlib.util.spec_from_file_location("summarize_terminal_mcts_h2h", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_counts_applied_decision_override(tmp_path: Path) -> None:
    path = tmp_path / "candidate.search.jsonl"
    write_rows(
        path,
        [
            {
                "choice_override": {
                    "overridden": True,
                    "reason": "certified_terminal_mcts_override",
                    "selection_class": "certified_terminal_teacher",
                    "final_choice": "switch forretress",
                    "terminal_mcts_production_choice": "switch moltres",
                    "terminal_mcts_teacher": {
                        "decision": "override",
                        "baseline_action": "switch moltres",
                        "selected_action": "switch forretress",
                    },
                }
            },
            {
                "choice_override": {
                    "overridden": False,
                    "terminal_mcts_teacher": {
                        "decision": "abstain",
                        "reason": "fail_closed:TimeoutExpired",
                    },
                }
            },
        ],
    )
    summary = MODULE.summarize_log(path)
    assert summary["teacher_calls"] == 2
    assert summary["override_count"] == 1
    assert summary["fail_closed_count"] == 1


def test_rejects_teacher_override_not_applied(tmp_path: Path) -> None:
    path = tmp_path / "candidate.search.jsonl"
    write_rows(
        path,
        [
            {
                "choice_override": {
                    "overridden": False,
                    "final_choice": "populationbomb",
                    "terminal_mcts_production_choice": "populationbomb",
                    "terminal_mcts_teacher": {
                        "decision": "override",
                        "baseline_action": "populationbomb",
                        "selected_action": "encore",
                    },
                }
            }
        ],
    )
    with pytest.raises(ValueError, match="was not applied"):
        MODULE.summarize_log(path)


def test_counts_override_that_restores_raw_r1_action(tmp_path: Path) -> None:
    path = tmp_path / "candidate.search.jsonl"
    write_rows(
        path,
        [
            {
                "choice_override": {
                    # This field compares final choice with raw R1, while the
                    # experiment compares the teacher with production search.
                    "overridden": False,
                    "raw_choice": "encore",
                    "reason": "certified_terminal_mcts_override",
                    "selection_class": "certified_terminal_teacher",
                    "final_choice": "encore",
                    "terminal_mcts_production_choice": "populationbomb",
                    "terminal_mcts_teacher": {
                        "decision": "override",
                        "baseline_action": "populationbomb",
                        "selected_action": "encore",
                    },
                }
            }
        ],
    )
    assert MODULE.summarize_log(path)["override_count"] == 1
