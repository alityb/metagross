import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from srcs.metagross import run_foul_play
from srcs.metagross.causal_reveal_ledger import (
    CausalMoveEvent,
    CausalRevealFact,
    CausalRevealLedger,
    LEDGER_ATTRIBUTE,
    clear_public_protocol_lines,
    record_public_protocol_lines,
)


def candidate_teacher():
    return {
        "controller_schema": "metagross-cycle19-equal8192-production-selector/v1",
        "iterations_per_world": 8192, "schedule_count": 2, "world_count": 16,
        "reason": "frozen_equal8192_production_considered_visit_policy",
        "receipts": [
            {"schedule_index": schedule, "world_index": world, "total_visits": 8192}
            for schedule in range(2) for world in range(8)
        ],
    }


def make_battle(role: str):
    event = CausalMoveEvent(2, "beartic", "hypervoice", "intrinsic_public_execution", None)
    fact = CausalRevealFact(
        "beartic", "beartic", ("hypervoice",), None, False, (), None, None, (),
        move_events=(event,),
    )
    ledger = CausalRevealLedger(
        "battle-cycle32", role, "p1" if role == "p2" else "p2", "beartic",
        (fact,), "c" * 64,
    )
    battle = SimpleNamespace(
        battle_tag="battle-cycle32", rqid=4, turn=2,
        request_json={"rqid": 4, "forceSwitch": [False], "active": [{}]},
        user=SimpleNamespace(name=role, active=SimpleNamespace(name="hatterene")),
    )
    setattr(battle, LEDGER_ATTRIBUTE, ledger.to_payload())
    return battle


def configure(monkeypatch, role: str, public_username: str, configured_username: str):
    monkeypatch.setenv("METAGROSS_PRIOR_NAMESPACE", "agent_a")
    monkeypatch.setenv("METAGROSS_TERMINAL_MCTS_TEACHER_SLOT", "agent_a")
    monkeypatch.setitem(
        sys.modules, "config",
        SimpleNamespace(FoulPlayConfig=SimpleNamespace(username=configured_username)),
    )
    run_foul_play._PRIOR_STATE["context"] = {"tag": "battle-cycle32", "decision_idx": 1}
    clear_public_protocol_lines("battle-cycle32")
    record_public_protocol_lines(
        "battle-cycle32",
        [f"|player|{role}|{public_username}|1|", "|turn|2"],
    )


@pytest.mark.parametrize("role", ["p1", "p2"])
def test_public_role_maps_to_external_username_for_both_roles(monkeypatch, role):
    username = f"c32smk{role}123456"
    configure(monkeypatch, role, username, username)
    receipt = run_foul_play.cycle32_authenticated_candidate_boundary_receipt(
        make_battle(role), {"actions": ["drainingkiss"]}, 1,
        candidate_teacher(), "drainingkiss", max_decision_index=5, max_battle_turn=6,
    )
    identity = receipt["cycle31_candidate_attribution"]
    assert identity["internal_battle_role"] == role
    assert identity["external_authenticated_username"] == username
    assert "username" not in identity


@pytest.mark.parametrize(
    ("public_username", "configured_username"),
    [("c32smkx1234567", "different"), ("", "c32smkx1234567")],
)
def test_public_configured_username_mismatch_fails_closed(
    monkeypatch, public_username, configured_username
):
    configure(monkeypatch, "p1", public_username, configured_username)
    with pytest.raises(RuntimeError, match="public player mapping"):
        run_foul_play.cycle32_authenticated_candidate_boundary_receipt(
            make_battle("p1"), {"actions": ["drainingkiss"]}, 1,
            candidate_teacher(), "drainingkiss", max_decision_index=5, max_battle_turn=6,
        )


def test_identity_mapping_does_not_mutate_battle_or_teacher(monkeypatch):
    username = "c32smkx1234567"
    configure(monkeypatch, "p1", username, username)
    battle = make_battle("p1")
    teacher = candidate_teacher()
    battle_before = deepcopy(battle.__dict__)
    teacher_before = deepcopy(teacher)
    run_foul_play.cycle32_authenticated_candidate_boundary_receipt(
        battle, {"actions": ["drainingkiss"]}, 1, teacher, "drainingkiss",
        max_decision_index=5, max_battle_turn=6,
    )
    assert battle.__dict__ == battle_before
    assert teacher == teacher_before
