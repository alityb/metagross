import pytest

from experimental.src.scripts.monitor_cycle19_operational_smoke import (
    expected_command,
    validate_teacher,
)


def request():
    return {
        "rqid": 7,
        "side": {
            "pokemon": [
                {"ident": "p1: Lead", "details": "Iron Valiant, L78"},
                {"ident": "p1: Bench", "details": "Rotom-Wash, L82"},
            ]
        },
    }


def test_expected_move_tera_and_switch_commands():
    assert expected_command("moonblast", request()) == ("/choose move moonblast", "7")
    assert expected_command("moonblast-tera", request()) == (
        "/choose move moonblast terastallize",
        "7",
    )
    assert expected_command("switch rotomwash", request()) == ("/switch 2", "7")


def test_switch_mapping_fails_closed_on_ambiguity():
    duplicate = request()
    duplicate["side"]["pokemon"][0] = {
        "ident": "p1: Rotom-Wash",
        "details": "Rotom-Wash, L80",
    }
    with pytest.raises(RuntimeError, match="uniquely"):
        expected_command("switch rotomwash", duplicate)


def test_teacher_contract_rejects_low_mass_considered_action():
    receipts = []
    for schedule in range(2):
        for world in range(8):
            receipts.append(
                {
                    "schedule_index": schedule,
                    "world_index": world,
                    "total_visits": 8192,
                    "side_one": [{"action": "a"}, {"action": "b"}],
                }
            )
    teacher = {
        "controller_schema": "metagross-cycle19-equal8192-production-selector/v1",
        "schedule_count": 2,
        "world_count": 16,
        "iterations_per_world": 8192,
        "receipts": receipts,
        "prefilter_aggregate_policy": {"a": 0.9, "b": 0.1},
        "considered_choices": [{"action": "a", "mass": 0.9}],
        "selected_action": "a",
    }
    validate_teacher(teacher)
    teacher["considered_choices"].append({"action": "b", "mass": 0.1})
    with pytest.raises(RuntimeError, match="considered-choice"):
        validate_teacher(teacher)
