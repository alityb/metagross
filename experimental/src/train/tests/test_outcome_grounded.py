from __future__ import annotations

import pytest

from train.outcome_grounded import bootstrap_ci, stable_u64, stable_uniform, weighted_choice


def test_stable_random_contract() -> None:
    assert stable_u64("root", 1) == stable_u64("root", 1)
    assert stable_u64("root", 1) != stable_u64("root", 2)
    assert 0.0 <= stable_uniform("root", 1) < 1.0


def test_weighted_choice_uses_half_open_interval() -> None:
    actions = [("a", 1.0), ("b", 3.0)]
    assert weighted_choice(actions, 0.0) == "a"
    assert weighted_choice(actions, 0.249999) == "a"
    assert weighted_choice(actions, 0.25) == "b"
    assert weighted_choice(actions, 0.999999) == "b"


def test_weighted_choice_rejects_invalid_mass() -> None:
    with pytest.raises(ValueError):
        weighted_choice([("a", 0.0)], 0.5)


def test_bootstrap_interval_is_deterministic() -> None:
    first = bootstrap_ci([0.1, 0.2, 0.3], 7, repeats=500)
    second = bootstrap_ci([0.1, 0.2, 0.3], 7, repeats=500)
    assert first == second
    assert first[0] <= 0.2 <= first[1]
