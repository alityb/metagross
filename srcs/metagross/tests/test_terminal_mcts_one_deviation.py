from __future__ import annotations

import pytest

from srcs.metagross.terminal_mcts_one_deviation import (
    OneDeviationController,
    OneDeviationProtocolError,
    assignment_for_username,
    assignment_manifest,
)


SEED = "2026081507"
PREFIX = "tm1b"


def _username(game_index: int) -> str:
    role = "x" if game_index % 2 else "y"
    return f"{PREFIX}{role}{game_index:03d}abcd"


def _teacher(
    *,
    decision: str,
    production: str = "move 1",
    selected: str = "switch 2",
    reason: str = "certified",
) -> dict[str, object]:
    return {
        "schema": "metagross-terminal-mcts-live-decision/v1",
        "decision": decision,
        "baseline_action": production,
        "selected_action": selected if decision == "override" else production,
        "reason": reason,
    }


def test_assignment_is_exactly_balanced_before_play() -> None:
    manifest = assignment_manifest(SEED)
    rows = manifest["assignments"]
    assert len(rows) == 20
    assert sum(row["arm"] == "teacher" for row in rows) == 10
    assert sum(row["arm"] == "production" for row in rows) == 10
    assert sum(
        row["arm"] == "teacher" and row["pair_leg"] == 1 for row in rows
    ) == 5
    assert sum(
        row["arm"] == "teacher" and row["pair_leg"] == 2 for row in rows
    ) == 5
    for pair in range(1, 11):
        pair_rows = [row for row in rows if row["pair_index"] == pair]
        assert {row["arm"] for row in pair_rows} == {"teacher", "production"}


def test_username_binds_game_pair_leg_and_arm() -> None:
    expected = assignment_manifest(SEED)["assignments"]
    for row in expected:
        actual = assignment_for_username(
            _username(row["game_index"]), username_prefix=PREFIX, seed=SEED
        )
        for field in ("game_index", "pair_index", "pair_leg", "arm"):
            assert actual[field] == row[field]

    with pytest.raises(OneDeviationProtocolError):
        assignment_for_username(
            "tm1by001abcd", username_prefix=PREFIX, seed=SEED
        )


def test_abstentions_do_not_consume_the_first_certified_opportunity() -> None:
    controller = OneDeviationController(seed=SEED, username_prefix=PREFIX)
    teacher_game = next(
        row["game_index"]
        for row in assignment_manifest(SEED)["assignments"]
        if row["arm"] == "teacher"
    )
    username = _username(teacher_game)
    choice, row = controller.observe(
        battle_tag="battle-fresh-1",
        username=username,
        decision_index=3,
        production_choice="move 1",
        teacher=_teacher(decision="abstain"),
    )
    assert choice == "move 1"
    assert row["eligible"] is False
    assert row["locked_after_decision"] is False
    assert controller.should_query("battle-fresh-1", username)

    choice, row = controller.observe(
        battle_tag="battle-fresh-1",
        username=username,
        decision_index=4,
        production_choice="move 1",
        teacher=_teacher(decision="override"),
    )
    assert choice == "switch 2"
    assert row["eligible"] is True
    assert row["intervention_applied"] is True
    assert row["teacher_query_index"] == 2
    assert not controller.should_query("battle-fresh-1", username)
    with pytest.raises(OneDeviationProtocolError):
        controller.observe(
            battle_tag="battle-fresh-1",
            username=username,
            decision_index=5,
            production_choice="move 2",
            teacher=_teacher(decision="override", production="move 2"),
        )


def test_production_arm_observes_but_does_not_apply_the_deviation() -> None:
    controller = OneDeviationController(seed=SEED, username_prefix=PREFIX)
    production_game = next(
        row["game_index"]
        for row in assignment_manifest(SEED)["assignments"]
        if row["arm"] == "production"
    )
    username = _username(production_game)
    choice, row = controller.observe(
        battle_tag="battle-fresh-2",
        username=username,
        decision_index=8,
        production_choice="move 1",
        teacher=_teacher(decision="override"),
    )
    assert choice == "move 1"
    assert row["eligible"] is True
    assert row["intervention_applied"] is False
    assert row["assignment"]["arm"] == "production"
    assert not controller.should_query("battle-fresh-2", username)


@pytest.mark.parametrize(
    "teacher,expected_failure",
    [
        (_teacher(decision="abstain", reason="fail_closed:TimeoutExpired"), "fail_closed:TimeoutExpired"),
        (_teacher(decision="override", production="move 9"), "invalid_certified_deviation"),
        ({"schema": "wrong", "decision": "override"}, "invalid_teacher_schema"),
    ],
)
def test_integrity_failures_lock_to_production(
    teacher: dict[str, object], expected_failure: str
) -> None:
    controller = OneDeviationController(seed=SEED, username_prefix=PREFIX)
    username = _username(1)
    choice, row = controller.observe(
        battle_tag="battle-fresh-failure",
        username=username,
        decision_index=1,
        production_choice="move 1",
        teacher=teacher,
    )
    assert choice == "move 1"
    assert row["eligible"] is False
    assert row["integrity_failure"] == expected_failure
    assert not controller.should_query("battle-fresh-failure", username)
