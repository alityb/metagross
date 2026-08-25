from __future__ import annotations

import json
from pathlib import Path

import pytest

from experimental.src.scripts.score_cycle41_blind_role_repair import (
    pop_cohort,
    public_action_adjudication,
    public_role,
    receipt_key,
    validate_move_receipt_envelope,
)


def protocol_fixture(path: Path) -> None:
    rows = [{
        "direction": "received",
        "time_ns": 1,
        "message": "|player|p1|Alpha|\n|player|p2|Beta|\n",
    }]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def receipt(role: str = "p2", *, swap: bool = False, index: int = 0) -> dict:
    context = {
        "phase": "equal8192_candidate",
        "cohort": "fixed_two_by_eight",
        "battle_tag": "battle-test-1",
        "rqid": 7,
        "decision_index": 3,
        "root_id": "r" * 64,
        "declared_world_count": 16,
        "conversion_index": index,
        "schedule_index": index // 8,
        "world_index": index % 8,
    }
    return {
        "schema": "metagross-causal-move-conversion-receipt/v1",
        "battle_tag": "battle-test-1",
        "observer_role": role,
        "protocol_sha256": "p" * 64,
        "swap": swap,
        "execution_context": context,
        "move_receipt": {
            "schema": "metagross-causal-move-world-receipts/v1",
            "battle_tag": "battle-test-1",
            "protocol_sha256": "p" * 64,
            "moves": [],
            "derived_executions": [],
        },
        "receipt_time_ns": 10,
    }


def test_public_player_mapping_accepts_both_mirrored_roles(tmp_path: Path) -> None:
    path = tmp_path / "fixture.protocol.jsonl"
    protocol_fixture(path)
    assert public_role(path, "Alpha") == "p1"
    assert public_role(path, "Beta") == "p2"


def test_public_player_mapping_rejects_unregistered_name(tmp_path: Path) -> None:
    path = tmp_path / "fixture.protocol.jsonl"
    protocol_fixture(path)
    with pytest.raises(RuntimeError, match="missing public player-role"):
        public_role(path, "Gamma")


def test_p2_receipt_with_local_engine_side_one_orientation_is_valid() -> None:
    validate_move_receipt_envelope(receipt("p2", swap=False))


def test_swap_true_is_rejected_independently_of_public_role() -> None:
    with pytest.raises(RuntimeError, match="engine orientation"):
        validate_move_receipt_envelope(receipt("p2", swap=True))


def test_nested_protocol_mismatch_is_rejected() -> None:
    row = receipt("p1")
    row["move_receipt"]["protocol_sha256"] = "x" * 64
    with pytest.raises(RuntimeError, match="nested identity"):
        validate_move_receipt_envelope(row)


def test_p2_candidate_cohort_joins_only_to_authenticated_p2() -> None:
    rows = [receipt("p2", index=index) for index in range(16)]
    indexed = {receipt_key(rows[0]): rows}
    result = pop_cohort(
        indexed,
        "equal8192_candidate",
        "p2",
        ("battle-test-1", 7, 3, "r" * 64),
        11,
    )
    assert result == {"count": 16, "protocol_sha256": "p" * 64}
    assert indexed == {}


def test_role_mismatch_cannot_consume_other_observer_cohort() -> None:
    rows = [receipt("p2", index=index) for index in range(16)]
    indexed = {receipt_key(rows[0]): rows}
    with pytest.raises(RuntimeError, match="authenticated p1"):
        pop_cohort(
            indexed,
            "equal8192_candidate",
            "p1",
            ("battle-test-1", 7, 3, "r" * 64),
            11,
        )


@pytest.mark.parametrize(
    "line,expected",
    [
        ("|move|p2a: BetaMon|Moonblast|p1a: AlphaMon", "executed_move"),
        ("|cant|p2a: BetaMon|slp", "public_cant"),
        ("|faint|p2a: BetaMon", "fainted_before_action"),
        ("|-activate|p2a: BetaMon|confusion", "confusion_self_hit"),
    ],
)
def test_public_action_adjudication_accepts_same_actor_mechanics(
    line: str, expected: str
) -> None:
    protocol = [{"direction": "received", "time_ns": 2, "message": line}]
    assert public_action_adjudication(protocol, 1, "p2", "moonblast") == expected


def test_public_action_adjudication_rejects_other_actor_cant() -> None:
    protocol = [{"direction": "received", "time_ns": 2, "message": "|cant|p1a: AlphaMon|slp"}]
    assert public_action_adjudication(protocol, 1, "p2", "moonblast") is None


def test_wait_request_during_pivot_does_not_end_same_turn() -> None:
    protocol = [
        {"direction": "received", "time_ns": 2, "message": '|request|{"wait":true}'},
        {"direction": "received", "time_ns": 3, "message": "|move|p2a: BetaMon|Moonblast|p1a: AlphaMon"},
    ]
    assert public_action_adjudication(protocol, 1, "p2", "moonblast") == "executed_move"


def test_next_actionable_request_ends_adjudication() -> None:
    protocol = [
        {"direction": "received", "time_ns": 2, "message": '|request|{"active":[{}],"rqid":9}'},
        {"direction": "received", "time_ns": 3, "message": "|move|p2a: BetaMon|Moonblast|p1a: AlphaMon"},
    ]
    assert public_action_adjudication(protocol, 1, "p2", "moonblast") is None
