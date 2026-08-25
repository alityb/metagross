from argparse import Namespace
import json

from scripts.audit_schema6_capture import audit


def snapshot(tag="battle-one", username="alice"):
    return {
        "schema": 6,
        "tag": tag,
        "username": username,
        "decision_idx": 0,
        "mask_fallback": False,
        "player_information_state": {
            "schema_version": 1,
            "opponent_public_team": [],
        },
        "player_observation_history": {},
        "text_tokens": [1],
        "numbers": [0.0],
        "illegal_actions": [False] + [True] * 12,
        "trajectory": {
            "mode": "causal-history",
            "observations": 1,
            "transitions": 0,
            "inference_length": 1,
            "action_receipts": [],
            "rl2": [[0.0] * 14],
            "time_indices": [0],
            "observation_rows": {
                "text_tokens": [[1]],
                "numbers": [[0.0]],
                "illegal_actions": [[False] + [True] * 12],
            },
        },
    }


def write(path, payloads):
    path.write_text("".join(json.dumps(row) + "\n" for row in payloads))


def test_capture_audit_admits_exact_complete_group(tmp_path):
    decisions, snapshots, output = tmp_path / "d.jsonl", tmp_path / "s.jsonl", tmp_path / "o.json"
    write(decisions, [
        {"record_type": "decision", "battle_tag": "one", "username": "alice", "prior_decision_idx": 0, "state": "state"},
        {"record_type": "battle_result", "battle_tag": "one", "username": "alice", "label": 1},
    ])
    write(snapshots, [snapshot()])
    result = audit(Namespace(
        decision_log=[decisions], prior_snapshot=[snapshots], output=output,
        minimum_battles=1, minimum_capture_rate=0.95,
    ))
    assert result["admitted"]
    assert result["capture_rate"] == 1.0


def test_capture_audit_rejects_missing_snapshot(tmp_path):
    decisions, snapshots, output = tmp_path / "d.jsonl", tmp_path / "s.jsonl", tmp_path / "o.json"
    write(decisions, [
        {"record_type": "decision", "battle_tag": "one", "username": "alice", "prior_decision_idx": 0, "state": "state"},
        {"record_type": "battle_result", "battle_tag": "one", "username": "alice", "label": 0},
    ])
    write(snapshots, [])
    result = audit(Namespace(
        decision_log=[decisions], prior_snapshot=[snapshots], output=output,
        minimum_battles=1, minimum_capture_rate=0.95,
    ))
    assert not result["admitted"]
    assert result["failures"] == {"missing_schema6_snapshots": 1}


def test_capture_audit_accepts_causal_root_with_h2h_result(tmp_path):
    decisions = tmp_path / "dual.jsonl"
    snapshots = tmp_path / "snapshots.jsonl"
    h2h = tmp_path / "result.json"
    output = tmp_path / "audit.json"
    write(decisions, [{
        "schema": "metagross-causal-dual-r1-root/v1",
        "identity": {
            "battle_tag": "battle-one",
            "username": "alice",
            "decision_idx": 0,
        },
        "state": "state",
    }])
    write(snapshots, [snapshot()])
    h2h.write_text(json.dumps({
        "games": [{"battle_tag": "battle-one", "winner_username": "alice"}],
    }))
    result = audit(Namespace(
        decision_log=[decisions], prior_snapshot=[snapshots], h2h_result=h2h,
        output=output, minimum_battles=1, minimum_capture_rate=0.95,
    ))
    assert result["admitted"]
    assert result["terminal_results"] == 1
