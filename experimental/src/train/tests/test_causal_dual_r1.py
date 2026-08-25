from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from train.causal_dual_r1 import CausalDualR1Error, CausalR1PolicyState
from scripts.r1_public_events import _transformer_observation


def snapshot() -> dict:
    return {
        "schema": 6,
        "text_tokens": [1, 2],
        "numbers": [0.25, 0.5],
        "illegal_actions": [False] + [True] * 12,
        "name_table": {"tackle": 0},
        "trajectory": {
            "mode": "causal-history",
            "observation_rows": {
                "text_tokens": [[1, 2]],
                "numbers": [[0.25, 0.5]],
                "illegal_actions": [[False] + [True] * 12],
            },
            "rl2": [[0.0] * 14],
            "time_indices": [7],
        },
    }


def test_schema6_snapshot_and_advance_preserve_reward_first_rl2_contract():
    root_tracker = SimpleNamespace(state="before")
    next_tracker = SimpleNamespace(state="after")
    with patch(
        "train.causal_dual_r1.R1SwitchTracker.from_snapshot",
        return_value=root_tracker,
    ):
        state = CausalR1PolicyState.from_snapshot(snapshot(), object())
    state.advance(
        next_tracker,
        {
            "text_tokens": [3, 4],
            "numbers": [0.75, 1.0],
            "illegal_actions": [False] + [True] * 12,
            "name_table": {"tackle": 0},
        },
        "tackle",
        lambda before, after: 0.125 if (before, after) == ("before", "after") else 9,
    )
    assert state.time_indices == [7, 8]
    assert state.rl2[-1][0] == pytest.approx(0.125)
    assert state.rl2[-1][1] == 1.0
    assert sum(state.rl2[-1][2:]) == 0.0
    assert state.current_observation["text_tokens"] == [3, 4]


def test_noncausal_snapshot_fails_closed():
    value = snapshot()
    value["trajectory"]["mode"] = "legacy-stateless"
    with pytest.raises(CausalDualR1Error, match="not causal-history"):
        CausalR1PolicyState.from_snapshot(value, object())


def test_nomove_wait_defers_prior_action_until_next_learned_request():
    root_tracker = SimpleNamespace(state="decision")
    waiting_tracker = SimpleNamespace(state="waiting")
    resumed_tracker = SimpleNamespace(state="resumed")
    with patch(
        "train.causal_dual_r1.R1SwitchTracker.from_snapshot",
        return_value=root_tracker,
    ):
        state = CausalR1PolicyState.from_snapshot(snapshot(), object())
    waiting = {
        "text_tokens": [2, 3],
        "numbers": [0.5, 0.75],
        "illegal_actions": [True] * 13,
        "name_table": {"tackle": 0},
        "automatic_action": "nomove",
        "terminal": False,
    }
    state.advance(waiting_tracker, waiting, "tackle", lambda *_: 99.0)
    assert state.action_support(None) == [("nomove", 1.0)]
    assert len(state.rl2) == 1
    resumed = {
        "text_tokens": [3, 4],
        "numbers": [0.75, 1.0],
        "illegal_actions": [False] + [True] * 12,
        "name_table": {"tackle": 0},
        "automatic_action": None,
        "terminal": False,
    }
    state.advance(
        resumed_tracker,
        resumed,
        "nomove",
        lambda before, after: 0.5 if (before, after) == ("decision", "resumed") else 99.0,
    )
    assert len(state.rl2) == 2
    assert state.rl2[-1][0] == pytest.approx(0.5)
    assert state.rl2[-1][1] == 1.0
    assert state.pending_action_index is None


def test_transformer_abi_accepts_only_named_automatic_nomove_boundary():
    observation = _transformer_observation(
        {
            "text_tokens": [1],
            "numbers": [0.0],
            "illegal_actions": [True] * 13,
            "name_table": {"tackle": 0},
            "terminal": False,
            "automatic_action": "nomove",
        }
    )
    assert observation.automatic_action == "nomove"
