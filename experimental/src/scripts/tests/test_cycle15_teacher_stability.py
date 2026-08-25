from experimental.src.scripts.run_cycle15_teacher_stability import (
    _aggregate, _jsd, _tv, arm_metrics,
)


def test_distribution_metrics_are_symmetric_and_zero_on_identity():
    a = {"a": .75, "b": .25}
    b = {"a": .25, "b": .75}
    assert _tv(a, a) == 0
    assert _jsd(a, a) == 0
    assert _tv(a, b) == _tv(b, a) == .5
    assert _jsd(a, b) == _jsd(b, a) > 0


def test_aggregate_preserves_unvisited_q_as_a_world_level_concern():
    worlds = [{
        "weight": 1.0,
        "result": {"total_visits": 10, "side_one": [
            {"action": "a", "N": 10, "W": 4.0, "Q": .4},
            {"action": "b", "N": 0, "W": 0.0, "Q": None},
        ]},
    }]
    aggregate = _aggregate(worlds, ["a", "b"])
    assert aggregate["top1"] == "a"
    assert aggregate["policy"] == {"a": 1.0, "b": 0.0}


def test_arm_metrics_requires_stable_difference():
    cell = lambda top: {"top1": top, "policy": {"a": float(top == "a"), "b": float(top == "b")}, "gap": 1.0}
    root = {
        "arms": {"equal_20000": [
            {**cell("a"), "schedule_index": 0}, {**cell("a"), "schedule_index": 0},
            {**cell("a"), "schedule_index": 1}, {**cell("a"), "schedule_index": 1},
        ]},
        "production_exact": [
            {**cell("b"), "schedule_index": 0}, {**cell("b"), "schedule_index": 0},
            {**cell("b"), "schedule_index": 1}, {**cell("b"), "schedule_index": 1},
        ],
    }
    metrics = arm_metrics([root], "equal_20000")
    assert metrics["all_cell_top1_stability"] == 1
    assert metrics["stable_differences_from_production_exact"] == 1
