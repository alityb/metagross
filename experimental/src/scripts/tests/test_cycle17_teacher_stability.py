from experimental.src.scripts.run_cycle17_teacher_stability import invariant_snapshot_without_routing_rqid


def response(rqid):
    return {
        "r1_policy_snapshot": {"text_tokens": [1], "numbers": [2.0], "illegal_actions": [False],
            "name_table": {"a": 0}, "trajectory": {"observation_rows": {"x": [[1]]}, "rl2": [[0]], "time_indices": [0]}},
        "own_legality": {"authority": "private_showdown_request", "rqid": rqid,
            "force_switch": False, "trapped": False, "can_tera": False, "actions": ["a"]},
        "priors": {"a": 1.0}, "opp_priors": {}, "probs": [1.0],
    }


def test_only_routing_rqid_is_normalized():
    assert invariant_snapshot_without_routing_rqid(response(1)) == invariant_snapshot_without_routing_rqid(response(1000001))
    changed = response(1000001); changed["own_legality"]["trapped"] = True
    assert invariant_snapshot_without_routing_rqid(response(1)) != invariant_snapshot_without_routing_rqid(changed)
