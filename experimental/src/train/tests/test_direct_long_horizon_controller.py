from __future__ import annotations

from copy import deepcopy

import pytest

from train.direct_long_horizon_controller import decide, shortlist_top_two
from train.outcome_grounded import RESULT_SCHEMA


def search_row(schedule: int, visits: tuple[float, float]) -> dict:
    return {
        "root_id": "root",
        "schedule_id": schedule,
        "root_statistics": {
            "aggregate_top_visit_mass": 0.60,
            "aggregate_top_two_margin": 0.10,
            "weighted_top_action_disagreement": 0.0,
            "weighted_js_divergence": 0.0,
        },
        "action_statistics": {
            "a": {"visit_mass": visits[0], "mean_value": 0.1},
            "b": {"visit_mass": visits[1], "mean_value": 0.2},
        },
    }


def result_rows(rollouts: int, alternative: float = 1.0) -> list[dict]:
    rows = []
    for schedule in (0, 1):
        outcomes = {}
        for action, value in (("a", 0.0), ("b", alternative)):
            outcomes[action] = [
                {"world_index": world, "rollout": rollout, "outcome": value}
                for world in range(8)
                for rollout in range(rollouts)
            ]
        rows.append({
            "schema": RESULT_SCHEMA,
            "battle_id": "battle",
            "root_id": "root",
            "schedule_id": schedule,
            "baseline_action": "a",
            "candidate_actions": ["a", "b"],
            "action_outcomes": outcomes,
        })
    return rows


def test_shortlist_is_label_blind_and_uses_mean_schedule_visits():
    assert shortlist_top_two([search_row(0, (0.7, 0.3)), search_row(1, (0.4, 0.6))]) == ("a", "b")


def test_four_rollout_high_confidence_can_override():
    report = decide(result_rows(4))
    assert report["decision"] == "override"
    assert report["selected_action"] == "b"
    assert all(report["checks"].values())


def test_four_rollout_inconclusive_extends_without_override():
    report = decide(result_rows(4, alternative=0.0))
    assert report["decision"] == "extend_to_16"
    assert report["selected_action"] == "a"


def test_sixteen_rollout_inconclusive_abstains():
    report = decide(result_rows(16, alternative=0.0))
    assert report["decision"] == "abstain"
    assert report["selected_action"] == "a"


def test_censoring_below_coverage_fails_closed():
    rows = result_rows(16)
    for row in rows:
        for action in ("a", "b"):
            for sample in row["action_outcomes"][action][:80]:
                sample["outcome"] = None
    report = decide(rows)
    assert report["decision"] == "abstain"
    assert not report["checks"]["terminal_coverage"]


def test_mismatched_tape_is_rejected():
    rows = result_rows(4)
    rows[0]["action_outcomes"]["b"].pop()
    with pytest.raises(ValueError, match="matched rollout tape"):
        decide(rows)


def test_third_candidate_is_rejected():
    rows = result_rows(4)
    rows[0]["candidate_actions"].append("c")
    with pytest.raises(ValueError, match="top-two"):
        decide(rows)
