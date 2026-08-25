import random
from types import SimpleNamespace
from unittest.mock import patch

from experimental.src.scripts import cycle19_equal8192_live_decision as live


class InlineExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, fn, rows):
        for row in rows:
            yield fake_task(row)


def fake_task(row):
    return {
        "schedule_index": row["schedule_index"],
        "world_index": row["world_index"],
        "weight": row["weight"],
        "state_sha256": "0" * 64,
        "search_seed": row["search_seed"],
        "latency_ms": 1.0,
        "total_visits": 8192,
        "side_one": [
            {"action": "a", "N": 4096, "W": 0.0, "Q": 0.0},
            {"action": "b", "N": 3686, "W": 0.0, "Q": 0.0},
            {"action": "c", "N": 410, "W": 0.0, "Q": 0.0},
        ],
        "side_two": [{"action": "x", "N": 8192, "W": 0.0, "Q": 0.0}],
    }


def test_low_mass_action_is_excluded_and_receipts_are_complete():
    payload = {
        "root_id": "r",
        "battle_id": "b",
        "production_choice": "c",
        "request_actions": ["a", "b", "c"],
        "seed": 9,
        "schedules": [
            {"worlds": [{"state": "s", "weight": 1 / 8} for _ in range(8)]}
            for _ in range(2)
        ],
    }
    with patch.object(live, "ProcessPoolExecutor", InlineExecutor):
        result = live.evaluate(payload, 8)
    assert [row["action"] for row in result["considered_choices"]] == ["a", "b"]
    assert result["prefilter_aggregate_policy"]["c"] > 0
    assert result["selected_action"] in {"a", "b"}
    assert result["world_count"] == 16
    assert all(row["total_visits"] == 8192 for row in result["receipts"])


def test_ties_and_seed_match_vendor_selector_byte_for_byte():
    from fp.search.main import select_move_from_mcts_results

    rows = [
        SimpleNamespace(move_choice="z", visits=40, total_score=0.0),
        SimpleNamespace(move_choice="a", visits=40, total_score=0.0),
        SimpleNamespace(move_choice="low", visits=20, total_score=0.0),
    ]
    result = SimpleNamespace(side_one=rows, side_two=[], total_visits=100)
    policy = {"z": 0.4, "a": 0.4, "low": 0.2}
    for seed in (0, 1, 7, 99, 2**63):
        random.seed(seed)
        expected = select_move_from_mcts_results([(result, 1.0, 0)])
        actual, considered = live.production_considered_sample(policy, seed)
        assert actual == expected
        assert [row["action"] for row in considered] == ["z", "a"]


def test_zero_mass_fallback_matches_vendor_selector():
    from fp.search.main import select_move_from_mcts_results

    rows = [
        SimpleNamespace(move_choice="b", visits=0, total_score=0.0),
        SimpleNamespace(move_choice="a", visits=0, total_score=0.0),
    ]
    result = SimpleNamespace(side_one=rows, side_two=[], total_visits=1)
    for seed in (3, 8, 13):
        random.seed(seed)
        expected = select_move_from_mcts_results([(result, 0.0, 0)])
        actual, considered = live.production_considered_sample({"b": 0.0, "a": 0.0}, seed)
        assert actual == expected
        assert [row["action"] for row in considered] == ["b", "a"]
