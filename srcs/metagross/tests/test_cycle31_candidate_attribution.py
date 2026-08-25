from copy import deepcopy
from types import SimpleNamespace

import pytest

from srcs.metagross import run_foul_play
from srcs.metagross.causal_reveal_ledger import (
    CausalMoveEvent,
    CausalRevealFact,
    CausalRevealLedger,
    LEDGER_ATTRIBUTE,
)


def fixture_battle():
    event = CausalMoveEvent(
        event_index=8,
        exact_public_species="brutebonnet",
        move="spore",
        authority="intrinsic_public_execution",
        derived_cause=None,
    )
    fact = CausalRevealFact(
        species="brutebonnet",
        exact_public_species="brutebonnet",
        moves=("spore",),
        current_item=None,
        item_status_revealed=False,
        consumed_items=(),
        current_ability=None,
        current_ability_authority=None,
        ability_history=(),
        move_events=(event,),
    )
    ledger = CausalRevealLedger(
        battle_tag="battle-cycle31",
        observer_role="p1",
        opponent_role="p2",
        opponent_active_species="brutebonnet",
        facts=(fact,),
        protocol_sha256="b" * 64,
    )
    battle = SimpleNamespace(
        battle_tag="battle-cycle31",
        rqid=7,
        turn=2,
        request_json={"rqid": 7, "forceSwitch": [False], "active": [{}]},
        user=SimpleNamespace(name="c31smkx1234567", active=SimpleNamespace(name="mew")),
    )
    setattr(battle, LEDGER_ATTRIBUTE, ledger.to_payload())
    return battle


def teacher():
    return {
        "controller_schema": "metagross-cycle19-equal8192-production-selector/v1",
        "iterations_per_world": 8192,
        "schedule_count": 2,
        "world_count": 16,
        "reason": "frozen_equal8192_production_considered_visit_policy",
        "receipts": [
            {"schedule_index": schedule, "world_index": world, "total_visits": 8192}
            for schedule in range(2)
            for world in range(8)
        ],
    }


def classify(monkeypatch, payload=None):
    monkeypatch.setenv("METAGROSS_PRIOR_NAMESPACE", "agent_a")
    monkeypatch.setenv("METAGROSS_TERMINAL_MCTS_TEACHER_SLOT", "agent_a")
    run_foul_play._PRIOR_STATE["context"] = {
        "tag": "battle-cycle31",
        "decision_idx": 2,
    }
    value = fixture_battle()
    return value, run_foul_play.cycle31_candidate_boundary_receipt(
        value,
        {"actions": ["psychic", "switch blissey"]},
        2,
        payload or teacher(),
        "psychic",
        max_decision_index=5,
        max_battle_turn=6,
    )


def test_complete_agent_a_candidate_binds_full_identity(monkeypatch):
    _battle, receipt = classify(monkeypatch)
    attribution = receipt["cycle31_candidate_attribution"]
    assert receipt["cycle30_dynamic_boundary"]["eligible"] is True
    assert attribution["namespace"] == "agent_a"
    assert attribution["username"] == "c31smkx1234567"
    assert attribution["observer_role"] == "p1"
    assert attribution["rqid"] == 7
    assert attribution["protocol_sha256"] == "b" * 64
    assert len(attribution["candidate_cells"]) == 16
    assert attribution["iterations_per_world"] == 8192


def test_agent_b_production_stream_cannot_arm(monkeypatch):
    monkeypatch.setenv("METAGROSS_PRIOR_NAMESPACE", "agent_b")
    monkeypatch.setenv("METAGROSS_TERMINAL_MCTS_TEACHER_SLOT", "agent_a")
    with pytest.raises(RuntimeError, match="agent-A candidate stream"):
        run_foul_play.cycle31_candidate_boundary_receipt(
            fixture_battle(), {"actions": ["psychic"]}, 2, teacher(), "psychic",
            max_decision_index=5, max_battle_turn=6,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.__setitem__("world_count", 15),
        lambda row: row["receipts"].pop(),
        lambda row: row["receipts"][0].__setitem__("total_visits", 8191),
        lambda row: row["receipts"][0].__setitem__("world_index", 1),
    ],
)
def test_incomplete_candidate_cohort_fails_closed(monkeypatch, mutation):
    payload = teacher()
    mutation(payload)
    with pytest.raises(RuntimeError):
        classify(monkeypatch, payload)


def test_attribution_marker_is_read_only(monkeypatch):
    value = fixture_battle()
    before = deepcopy(value.__dict__)
    monkeypatch.setenv("METAGROSS_PRIOR_NAMESPACE", "agent_a")
    monkeypatch.setenv("METAGROSS_TERMINAL_MCTS_TEACHER_SLOT", "agent_a")
    run_foul_play._PRIOR_STATE["context"] = {
        "tag": "battle-cycle31",
        "decision_idx": 2,
    }
    first = run_foul_play.cycle31_candidate_boundary_receipt(
        value, {"actions": ["psychic"]}, 2, teacher(), "psychic",
        max_decision_index=5, max_battle_turn=6,
    )
    second = run_foul_play.cycle31_candidate_boundary_receipt(
        value, {"actions": ["psychic"]}, 2, teacher(), "psychic",
        max_decision_index=5, max_battle_turn=6,
    )
    assert {k: v for k, v in first.items() if k != "decision_complete_time_ns"} == {
        k: v for k, v in second.items() if k != "decision_complete_time_ns"
    }
    assert value.__dict__ == before
