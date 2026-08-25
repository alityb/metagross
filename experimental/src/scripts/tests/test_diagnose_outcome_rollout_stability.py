from __future__ import annotations

import copy

import pytest

from scripts.diagnose_outcome_rollout_stability import verify_prefix
from train.outcome_grounded import RESULT_SCHEMA


def rows(rollouts: int):
    return [
        {
            "schema": RESULT_SCHEMA,
            "battle_id": "battle",
            "root_id": "root",
            "schedule_id": schedule,
            "baseline_action": "a",
            "candidate_actions": ["a"],
            "action_outcomes": {
                "a": [
                    {
                        "world_index": 0,
                        "rollout": rollout,
                        "outcome": float((schedule + rollout) % 2),
                        "decisions": 4 + rollout,
                    }
                    for rollout in range(rollouts)
                ]
            },
        }
        for schedule in (0, 1)
    ]


def test_verify_prefix_accepts_exact_first_eight_rollouts() -> None:
    assert verify_prefix(rows(8), rows(16)) == {
        "prefix_samples": 16,
        "missing": 0,
        "changed": 0,
    }


def test_verify_prefix_rejects_changed_frozen_outcome() -> None:
    extended = copy.deepcopy(rows(16))
    extended[0]["action_outcomes"]["a"][3]["outcome"] = 0.0
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_prefix(rows(8), extended)
