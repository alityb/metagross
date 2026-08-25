from unittest.mock import patch

from experimental.src.scripts import cycle18_equal8192_live_decision as live


class InlineExecutor:
    def __init__(self, max_workers): self.max_workers = max_workers
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def map(self, fn, rows):
        for row in rows:
            yield fake_task(row)


def fake_task(row):
    # Keep exact total visits and full legal support, but make action a dominant.
    return {"schedule_index": row["schedule_index"], "world_index": row["world_index"],
            "weight": row["weight"], "state_sha256": "0" * 64,
            "search_seed": row["search_seed"], "latency_ms": 1.0, "total_visits": 8192,
            "side_one": [{"action": "a", "N": 8192, "W": 0.0, "Q": 0.0},
                         {"action": "b", "N": 0, "W": 0.0, "Q": None}],
            "side_two": [{"action": "x", "N": 8192, "W": 0.0, "Q": 0.0}]}


def test_live_controller_uses_two_eight_world_schedules_and_full_receipts():
    payload = {"root_id": "r", "battle_id": "b", "production_choice": "b",
        "request_actions": ["a", "b"], "seed": 9,
        "schedules": [{"worlds": [{"state": "s", "weight": 1 / 8} for _ in range(8)]} for _ in range(2)]}
    with patch.object(live, "ProcessPoolExecutor", InlineExecutor):
        result = live.evaluate(payload, 8)
    assert result["selected_action"] == "a"
    assert result["decision"] == "override"
    assert result["world_count"] == 16
    assert len(result["receipts"]) == 16
    assert result["aggregate_policy"] == {"a": 1.0, "b": 0.0}
