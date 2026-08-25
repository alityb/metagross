from argparse import Namespace
import json

import pytest

from scripts.collect_outcome_grounded_continuations import load_progress
from train.outcome_grounded import RESULT_SCHEMA


def args():
    return Namespace(
        root_iterations=20_000,
        continuation_iterations=2_048,
        rollouts=8,
        max_decisions=128,
        seed=7,
    )


def row():
    return {
        "schema": RESULT_SCHEMA,
        "battle_id": "battle",
        "root_id": "root",
        "schedule_id": 0,
        "baseline_action": "move",
        "candidate_actions": ["move", "switch x"],
        "configuration": {
            "root_iterations": 20_000,
            "continuation_iterations": 2_048,
            "rollouts": 8,
            "max_decisions": 128,
            "seed": 7,
        },
    }


def expected():
    return {
        ("root", 0): {
            "battle_id": "battle",
            "baseline_action": "move",
            "candidate_actions": ["move", "switch x"],
        }
    }


def test_progress_resume_accepts_exact_schedule_row(tmp_path):
    path = tmp_path / "progress.jsonl"
    path.write_text(json.dumps(row()) + "\n")
    assert load_progress(path, expected(), args()) == [row()]


def test_progress_resume_rejects_changed_configuration(tmp_path):
    changed = row()
    changed["configuration"]["max_decisions"] = 192
    path = tmp_path / "progress.jsonl"
    path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_progress(path, expected(), args())


def test_progress_resume_rejects_duplicate_schedule(tmp_path):
    path = tmp_path / "progress.jsonl"
    path.write_text(json.dumps(row()) + "\n" + json.dumps(row()) + "\n")
    with pytest.raises(ValueError, match="duplicate progress row"):
        load_progress(path, expected(), args())
