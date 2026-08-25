from __future__ import annotations

from types import SimpleNamespace

import pytest

from search import causal_child_target_v1 as target
from srcs.metagross.causal_reveal_ledger import compile_reveal_bits, freeze_ledger


def test_child_information_state_replaces_hidden_exact_hp(monkeypatch) -> None:
    payload = {
        "schema": "old",
        "opponent": {"pokemon": [
            {"id": "mew", "hp": 151, "maxhp": 301, "moves": ["b", "a"]},
            {"id": "fainted", "hp": 0, "maxhp": 250, "moves": []},
            None,
        ]},
    }
    monkeypatch.setattr(target, "extract_public_search_state", lambda *_args: payload)
    result = target.child_information_state(object(), object())
    assert result["schema"] == target.SCHEMA
    assert result["opponent"]["pokemon"][0]["hp_percent"] == 51
    assert result["opponent"]["pokemon"][1]["hp_percent"] == 0
    assert "hp" not in result["opponent"]["pokemon"][0]
    assert "maxhp" not in result["opponent"]["pokemon"][0]
    assert result["opponent"]["pokemon"][0]["moves"] == ["a", "b"]


def _teacher(a_visits: int, b_visits: int, a_q: float, b_q: float) -> dict:
    total = a_visits + b_visits
    return {
        "total_visits": total,
        "side_one": [
            {"action": "a", "visits": a_visits, "total_score": a_q * a_visits, "q": a_q, "completed_q": None},
            {"action": "b", "visits": b_visits, "total_score": b_q * b_visits, "q": b_q, "completed_q": None},
        ],
        "side_two": [],
        "completed_q_available": False,
        "completed_q": None,
        "argmax_action": "a" if a_visits >= b_visits else "b",
    }


def test_group_aggregation_preserves_full_policy_q_disagreement_and_ess() -> None:
    members = [
        {
            "split": "train", "public_fingerprint": "fp", "public_state": {"x": 1},
            "legal_actions": ["a", "b"], "weight": 0.25, "target_id": "one",
            "teacher": _teacher(75, 25, 0.7, 0.4),
        },
        {
            "split": "train", "public_fingerprint": "fp", "public_state": {"x": 1},
            "legal_actions": ["a", "b"], "weight": 0.25, "target_id": "two",
            "teacher": _teacher(25, 75, 0.5, 0.8),
        },
    ]
    result = target.aggregate_target_members(members)
    assert result["visit_policy"] == {"a": 0.5, "b": 0.5}
    assert result["q"]["a"]["mean"] == pytest.approx(0.6)
    assert result["effective_sample_size"] == pytest.approx(2.0)
    assert result["hidden_world_argmax_disagreement"] is True
    assert result["same_teacher_control_is_independent"] is False


def test_grouping_fails_if_fingerprint_crosses_battle_split() -> None:
    rows = [
        {"split": "train", "public_fingerprint": "same"},
        {"split": "test", "public_fingerprint": "same"},
    ]
    with pytest.raises(target.CausalChildTargetError, match="crosses battle splits"):
        target.group_target_members(rows)


def test_teacher_snapshot_records_zero_visit_and_no_completed_q() -> None:
    result = SimpleNamespace(
        total_visits=4,
        side_one=[
            SimpleNamespace(move_choice="a", visits=4, total_score=3.0),
            SimpleNamespace(move_choice="b", visits=0, total_score=0.0),
        ],
        side_two=[SimpleNamespace(move_choice="x", visits=4, total_score=1.0)],
    )
    snapshot = target.snapshot_teacher_result(result)
    assert snapshot["side_one"][1]["q"] is None
    assert snapshot["completed_q_available"] is False
    assert snapshot["argmax_action"] == "a"


def test_showdown_p2_observer_still_maps_to_engine_side_one_perspective() -> None:
    ledger = freeze_ledger(
        "battle-p2-orientation",
        "p2",
        [
            "|switch|p2a: Own|Pikachu, L80|100/100",
            "|switch|p1a: Foe|Mew, L80|100/100",
            "|move|p1a: Foe|Psychic|p2a: Own",
        ],
    )
    assert ledger.observer_role == "p2"
    assert ledger.opponent_role == "p1"
    mew = SimpleNamespace(
        id="mew", item="none", ability="synchronize",
        moves=[SimpleNamespace(id="psychic"), *[SimpleNamespace(id="none") for _ in range(3)]],
    )
    none = SimpleNamespace(id="none", item="none", ability="none", moves=[])
    state = SimpleNamespace(side_two=SimpleNamespace(pokemon=[mew, *([none] * 5)]))
    # Foul Play always converts its local user to engine side one, regardless
    # of whether that user was Showdown p1 or p2.
    assert compile_reveal_bits(state, ledger, swap=False) == (1 | (1 << 6))
