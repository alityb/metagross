from __future__ import annotations

import json

from experimental.src.scripts.monitor_cycle29_second_root_smoke import (
    validate_move_receipts,
)


def test_monitor_accepts_complete_empty_opening_cohorts(tmp_path) -> None:
    identity = {
        "battle_tag": "battle-cycle29",
        "rqid": 1,
        "decision_index": 0,
        "root_id": "a" * 64,
    }
    contexts = [
        {
            **identity,
            "phase": "production_control",
            "cohort": "adaptive_root_search",
            "declared_world_count": 16,
            "conversion_index": index,
            "schedule_index": None,
            "world_index": None,
        }
        for index in range(16)
    ] + [
        {
            **identity,
            "phase": "equal8192_candidate",
            "cohort": "fixed_two_by_eight",
            "declared_world_count": 16,
            "conversion_index": schedule * 8 + world,
            "schedule_index": schedule,
            "world_index": world,
        }
        for schedule in range(2)
        for world in range(8)
    ]
    rows = [{
        "schema": "metagross-causal-move-conversion-receipt/v1",
        "battle_tag": identity["battle_tag"],
        "observer_role": "p1",
        "protocol_sha256": "b" * 64,
        "swap": False,
        "execution_context": context,
        "move_receipt": {
            "schema": "metagross-causal-move-world-receipts/v1",
            "battle_tag": identity["battle_tag"],
            "protocol_sha256": "b" * 64,
            "moves": [],
            "derived_executions": [],
        },
        "receipt_time_ns": 10,
    } for context in contexts]
    directory = tmp_path / "move-receipts"
    directory.mkdir()
    (directory / "agenta-1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    result = validate_move_receipts(
        tmp_path,
        identity,
        {"public_execution_time_ns": 100},
        require_revealed_move=False,
    )
    assert result["production_receipts"] == 16
    assert result["candidate_receipts"] == 16
    assert result["opening_empty_receipt"] is True
