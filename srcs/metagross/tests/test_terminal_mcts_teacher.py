from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from srcs.metagross import run_foul_play


class FakeHarness:
    class Belief:
        @staticmethod
        def expand(_battle, _search, channel):
            return [(f"{channel}-world-{index}", 1.0) for index in range(8)], 8, 500

    belief = Belief()


class FakeSearch:
    @staticmethod
    def battle_to_poke_engine_state(world):
        return SimpleNamespace(to_string=lambda: f"state:{world}")


def environment() -> dict[str, str]:
    return {
        "METAGROSS_TERMINAL_MCTS_TEACHER_SLOT": "agent_a",
        "METAGROSS_PRIOR_NAMESPACE": "agent_a",
        "METAGROSS_TERMINAL_MCTS_PYTHON": "/python311",
        "METAGROSS_TERMINAL_MCTS_SCRIPT": "/teacher.py",
        "METAGROSS_TERMINAL_MCTS_PYTHONPATH": "/engine:/src",
        "METAGROSS_TERMINAL_MCTS_WORKERS": "4",
    }


def test_slot_gate_is_isolated():
    with patch.dict(os.environ, environment(), clear=True):
        assert run_foul_play._terminal_mcts_teacher_enabled()
    values = {**environment(), "METAGROSS_PRIOR_NAMESPACE": "agent_b"}
    with patch.dict(os.environ, values, clear=True):
        assert not run_foul_play._terminal_mcts_teacher_enabled()


def test_terminal_teacher_serializes_two_eight_world_schedules_without_logging_states():
    response = {
        "schema": "metagross-terminal-mcts-live-decision/v1",
        "decision": "override",
        "selected_action": "switch foo",
        "reason": "frozen_terminal_gate",
    }
    completed = SimpleNamespace(
        returncode=0, stdout=json.dumps(response) + "\n", stderr=""
    )
    battle = SimpleNamespace(battle_tag="battle-1")
    run_foul_play._PRIOR_STATE["context"] = {
        "tag": "battle-1", "decision_idx": 3
    }
    with (
        patch.dict(os.environ, environment(), clear=True),
        patch.object(run_foul_play, "_derived_seed", return_value=17),
        patch.object(
            run_foul_play, "request_player_actions", return_value={"move", "switch foo"}
        ),
        patch.object(run_foul_play.subprocess, "run", return_value=completed) as invoked,
    ):
        result = run_foul_play._terminal_mcts_teacher_decision(
            FakeHarness(), FakeSearch(), battle, "move"
        )
    assert result == response
    payload = json.loads(invoked.call_args.kwargs["input"])
    assert len(payload["schedules"]) == 2
    assert all(len(schedule["worlds"]) == 8 for schedule in payload["schedules"])
    assert all(
        abs(sum(world["weight"] for world in schedule["worlds"]) - 1.0) < 1e-12
        for schedule in payload["schedules"]
    )
    assert payload["request_actions"] == ["move", "switch foo"]
    assert "state:" not in completed.stdout


def test_terminal_teacher_rejects_action_outside_private_request():
    response = {
        "schema": "metagross-terminal-mcts-live-decision/v1",
        "decision": "override",
        "selected_action": "switch hidden",
    }
    completed = SimpleNamespace(
        returncode=0, stdout=json.dumps(response) + "\n", stderr=""
    )
    with (
        patch.dict(os.environ, environment(), clear=True),
        patch.object(run_foul_play, "_derived_seed", return_value=17),
        patch.object(run_foul_play, "request_player_actions", return_value={"move"}),
        patch.object(run_foul_play.subprocess, "run", return_value=completed),
    ):
        with pytest.raises(RuntimeError, match="invalid decision"):
            run_foul_play._terminal_mcts_teacher_decision(
                FakeHarness(), FakeSearch(), SimpleNamespace(battle_tag="battle-1"), "move"
            )
