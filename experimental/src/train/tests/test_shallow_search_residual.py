from __future__ import annotations

import pytest

from train.shallow_search_residual import battle_split, choose_action, is_ambiguous


def _row(top_mass: float = 0.55):
    return {
        "selected_action": "a",
        "root_statistics": {
            "aggregate_top_visit_mass": top_mass,
            "aggregate_top_two_margin": 0.10,
            "weighted_top_action_disagreement": 0.3,
            "weighted_js_divergence": 0.2,
        },
        "action_statistics": {
            "a": {"world_support": 1.0},
            "b": {"world_support": 1.0},
            "c": {"world_support": 0.5},
        },
    }


def test_battle_split_is_stable_and_covers_all_splits() -> None:
    splits = {battle_split(f"battle-{index}") for index in range(1000)}
    assert splits == {"train", "calibration", "test"}
    assert battle_split("same") == battle_split("same")


def test_ambiguity_is_disjunctive() -> None:
    assert is_ambiguous(_row()["root_statistics"])
    confident = _row(0.9)["root_statistics"]
    confident.update({
        "aggregate_top_two_margin": 0.5,
        "weighted_top_action_disagreement": 0.0,
        "weighted_js_divergence": 0.0,
    })
    assert not is_ambiguous(confident)


def test_controller_abstains_without_positive_lower_bound() -> None:
    action, reason, lower = choose_action(
        _row(), {"a": 0.0, "b": 0.04, "c": 1.0}, conformal_penalty=0.035
    )
    assert (action, reason) == ("a", "lcb_below_margin")
    assert lower == pytest.approx(0.005)


def test_controller_overrides_only_supported_alternative() -> None:
    action, reason, lower = choose_action(
        _row(), {"a": 0.0, "b": 0.08, "c": 1.0}, conformal_penalty=0.03
    )
    assert (action, reason) == ("b", "override")
    assert lower == pytest.approx(0.05)
