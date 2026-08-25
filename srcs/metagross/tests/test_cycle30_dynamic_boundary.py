from types import SimpleNamespace

import pytest

from srcs.metagross.causal_reveal_ledger import (
    CausalMoveEvent,
    CausalRevealFact,
    CausalRevealLedger,
    LEDGER_ATTRIBUTE,
)
from srcs.metagross.run_foul_play import cycle30_dynamic_boundary_evidence


def battle(*, intrinsic: bool = True, turn: int = 2, request=None):
    events = ()
    moves = ()
    if intrinsic:
        events = (
            CausalMoveEvent(
                event_index=4,
                exact_public_species="garchomp",
                move="earthquake",
                authority="intrinsic_public_execution",
                derived_cause=None,
            ),
        )
        moves = ("earthquake",)
    fact = CausalRevealFact(
        species="garchomp",
        exact_public_species="garchomp",
        moves=moves,
        current_item=None,
        item_status_revealed=False,
        consumed_items=(),
        current_ability=None,
        current_ability_authority=None,
        ability_history=(),
        move_events=events,
    )
    ledger = CausalRevealLedger(
        battle_tag="battle-cycle30",
        observer_role="p1",
        opponent_role="p2",
        opponent_active_species="garchomp",
        facts=(fact,),
        protocol_sha256="a" * 64,
    )
    value = SimpleNamespace(
        turn=turn,
        request_json=request
        or {"rqid": 3, "forceSwitch": [False], "active": [{}]},
    )
    setattr(value, LEDGER_ATTRIBUTE, ledger.to_payload())
    return value


def classify(value, legality=None, decision_index=2):
    return cycle30_dynamic_boundary_evidence(
        value,
        legality or {"actions": ["earthquake", "switch corviknight"]},
        decision_index,
        max_decision_index=5,
        max_battle_turn=6,
    )


def test_first_ordinary_intrinsic_move_root_is_eligible():
    evidence = classify(battle())
    assert evidence["eligible"] is True
    assert evidence["ordinary"] is True
    assert evidence["intrinsic_opponent_move_events"] == 1
    assert evidence["protocol_sha256"] == "a" * 64


@pytest.mark.parametrize(
    ("request_payload", "legality"),
    [
        ({"rqid": 3, "wait": True, "forceSwitch": [False]}, {"actions": ["earthquake"]}),
        ({"rqid": 3, "forceSwitch": [True]}, {"actions": ["switch corviknight"]}),
        ({"rqid": 3, "forceSwitch": [False]}, {"actions": ["recharge"]}),
        ({"rqid": 3, "forceSwitch": [False]}, {"actions": ["struggle"]}),
    ],
)
def test_nonordinary_requests_are_ineligible(request_payload, legality):
    evidence = classify(battle(request=request_payload), legality)
    assert evidence["eligible"] is False
    assert evidence["ordinary"] is False


def test_derived_or_absent_move_does_not_select_boundary():
    evidence = classify(battle(intrinsic=False))
    assert evidence["eligible"] is False
    assert evidence["intrinsic_opponent_move_events"] == 0


@pytest.mark.parametrize(
    ("decision_index", "turn"),
    [(6, 2), (2, 7)],
)
def test_frozen_decision_and_turn_bounds_are_fail_closed(decision_index, turn):
    evidence = classify(battle(turn=turn), decision_index=decision_index)
    assert evidence["eligible"] is False
    assert evidence["within_bounds"] is False


def test_classifier_is_read_only():
    value = battle()
    request_before = dict(value.request_json)
    ledger_before = getattr(value, LEDGER_ATTRIBUTE)
    first = classify(value)
    second = classify(value)
    assert first == second
    assert value.request_json == request_before
    assert getattr(value, LEDGER_ATTRIBUTE) == ledger_before
