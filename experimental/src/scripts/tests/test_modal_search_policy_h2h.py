from __future__ import annotations

import importlib.util
import ast
import tempfile
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "modal_search_policy_h2h.py"
SPEC = importlib.util.spec_from_file_location("modal_search_policy_h2h", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_aggregates_counterbalanced_units():
    report = MODULE._summary(
        "action",
        "visits",
        [
            {"games": 4, "candidate_wins": 3},
            {"games": 4, "candidate_wins": 2},
        ],
    )
    assert report["games"] == 8
    assert report["candidate_wins"] == 5
    assert report["comparator_wins"] == 3
    assert report["candidate_winrate"] == 0.625
    assert report["cluster_normal95"][0] < 0.5
    assert report["candidate_advantage_established"] is False


def test_single_counterbalanced_unit_does_not_claim_an_advantage():
    report = MODULE._summary(
        "action",
        "visits",
        [{"games": 4, "candidate_wins": 4}],
    )
    assert report["cluster_normal95"] == [0.0, 1.0]
    assert report["candidate_advantage_established"] is False


def test_profile_checkpoint_paths_are_frozen():
    action = MODULE._checkpoint_path(MODULE.PROFILES["action"])
    r1 = MODULE._checkpoint_path(MODULE.PROFILES["r1"])
    assert str(action).endswith(
        "/search_policy_action_1k_seed20260812/ckpts/policy_weights/policy_epoch_1.pt"
    )
    assert str(r1).endswith(
        "/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"
    )


def test_worker_does_not_swallow_modal_preemption():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    worker = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_counterbalanced_unit"
    )
    caught = [
        handler.type.id
        for node in ast.walk(worker)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if isinstance(handler.type, ast.Name)
    ]
    assert "BaseException" not in caught


def test_persisted_unit_identity_is_fail_closed():
    payload = {
        "candidate": "action",
        "comparator": "r1",
        "unit_index": 22,
        "mirror_seed": 2026081923,
    }
    row = {
        **payload,
        "games": 4,
        "candidate_wins": 2,
        "comparator_wins": 2,
    }
    assert MODULE._validated_unit(row, payload) is row
    with __import__("pytest").raises(RuntimeError, match="identity"):
        MODULE._validated_unit({**row, "mirror_seed": 1}, payload)


def test_persistent_path_rejects_traversal(monkeypatch):
    monkeypatch.setattr(MODULE, "PERSIST_ROOT", Path(tempfile.gettempdir()))
    assert MODULE._persistent_unit_path("gate-1", 7).name == "0007.json"
    with __import__("pytest").raises(ValueError, match="run_id"):
        MODULE._persistent_unit_path("../escape", 7)
