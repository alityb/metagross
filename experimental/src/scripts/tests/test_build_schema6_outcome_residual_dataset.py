from argparse import Namespace
from copy import deepcopy
import hashlib
import json

import torch

from scripts.build_schema6_outcome_residual_dataset import SCHEMA, build
from scripts.collect_shallow_search_statistics import SCHEMA as SHALLOW_SCHEMA
from train.outcome_grounded import PANEL_SCHEMA


def write_json(path, payload):
    path.write_text(json.dumps(payload) + "\n")


def snapshot():
    return {
        "schema": 6,
        "decision_idx": 0,
        "mask_fallback": False,
        "player_information_state": {"schema_version": 1},
        "player_observation_history": {},
        "text_tokens": [1, 2],
        "numbers": [0.0],
        "illegal_actions": [False, False] + [True] * 11,
        "name_table": {"movea": 0, "moveb": 1},
        "trajectory": {
            "mode": "causal-history",
            "observations": 1,
            "transitions": 0,
            "inference_length": 1,
            "action_receipts": [],
            "rl2": [[0.0] * 14],
            "time_indices": [0],
            "observation_rows": {
                "text_tokens": [[1, 2]],
                "numbers": [[0.0]],
                "illegal_actions": [[False, False] + [True] * 11],
            },
        },
    }


def shallow_row(schedule_id):
    stats = {
        "movea": {
            "visit_mass": 0.55, "mean_value": 0.5, "value_std": 0.1,
            "visit_std": 0.1, "world_support": 1.0, "world_top_vote": 0.5,
        },
        "moveb": {
            "visit_mass": 0.45, "mean_value": 0.48, "value_std": 0.1,
            "visit_std": 0.1, "world_support": 1.0, "world_top_vote": 0.5,
        },
    }
    return {
        "schema": SHALLOW_SCHEMA,
        "root_id": "root",
        "pair_id": f"root:{schedule_id}",
        "selected_action": "movea",
        "action_statistics": stats,
        "root_statistics": {
            "visit_entropy": 0.9,
            "weighted_top_action_disagreement": 0.5,
            "weighted_js_divergence": 0.2,
            "aggregate_top_visit_mass": 0.55,
            "aggregate_top_two_margin": 0.1,
            "action_count": 2,
        },
    }


def test_build_materializes_only_certified_schema6_deviation(tmp_path):
    snapshot_path = tmp_path / "snapshots.jsonl"
    write_json(snapshot_path, snapshot())
    snapshot_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    panel = {
        "schema": PANEL_SCHEMA,
        "battle_id": "battle",
        "root_id": "root",
        "baseline_action": "movea",
        "candidate_actions": ["movea", "moveb"],
        "source_context": {
            "decision_idx": 0,
            "public_reveal_fractions": [0.5, 0.5, 0.0, 0.0],
            "r1_selection": {
                "policy_entropy": 0.8,
                "policy_top_gap": 0.1,
                "policy_top_probability": 0.55,
            },
            "causal_history": {
                "authority": "schema6_snapshot",
                "snapshot_schema": 6,
                "snapshot_source_path": str(snapshot_path),
                "snapshot_source_sha256": snapshot_hash,
                "snapshot_source_line": 1,
            },
        },
        "schedules": [{"schedule_id": 0, "worlds": []}, {"schedule_id": 1, "worlds": []}],
    }
    analysis = {
        "target_admitted_for_scale": True,
        "root_results": [{
            "root_id": "root",
            "baseline_action": "movea",
            "stable_action": "moveb",
            "alternatives": [{
                "action": "moveb",
                "stable_correction": True,
                "mean_advantage": 0.2,
                "cluster_bootstrap_ci95": [0.1, 0.3],
                "schedule_advantages": [0.15, 0.25],
            }],
        }],
    }
    panel_path, analysis_path, shallow_path = (
        tmp_path / "panel.jsonl", tmp_path / "analysis.json", tmp_path / "shallow.jsonl"
    )
    write_json(panel_path, panel)
    analysis_path.write_text(json.dumps(analysis))
    shallow_path.write_text("".join(json.dumps(shallow_row(i)) + "\n" for i in (0, 1)))
    output, report = tmp_path / "dataset.pt", tmp_path / "report.json"
    result = build(Namespace(
        panel=panel_path, analysis=analysis_path, shallow=shallow_path,
        output=output, report=report, minimum_records=1,
    ))
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert result["records"] == 1
    assert payload["schema"] == SCHEMA
    assert payload["records"][0]["baseline_index"] == 0
    assert payload["records"][0]["stable_index"] == 1
    assert payload["provenance"]["sampled_state_present"] is False
    assert "state" not in payload["records"][0]


def test_build_rejects_duplicate_teacher_root_evidence(tmp_path):
    snapshot_path = tmp_path / "snapshots.jsonl"
    write_json(snapshot_path, snapshot())
    snapshot_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    panel = {
        "schema": PANEL_SCHEMA, "battle_id": "battle", "root_id": "root",
        "baseline_action": "movea", "candidate_actions": ["movea", "moveb"],
        "source_context": {
            "decision_idx": 0, "public_reveal_fractions": [0.5] * 4,
            "r1_selection": {"policy_entropy": 0.8, "policy_top_gap": 0.1, "policy_top_probability": 0.55},
            "causal_history": {
                "authority": "schema6_snapshot", "snapshot_schema": 6,
                "snapshot_source_path": str(snapshot_path),
                "snapshot_source_sha256": snapshot_hash, "snapshot_source_line": 1,
            },
        },
        "schedules": [{"schedule_id": 0, "worlds": []}, {"schedule_id": 1, "worlds": []}],
    }
    evidence = {
        "root_id": "root", "baseline_action": "movea", "stable_action": None,
        "alternatives": [],
    }
    panel_path, analysis_path, shallow_path = tmp_path / "p.jsonl", tmp_path / "a.json", tmp_path / "s.jsonl"
    write_json(panel_path, panel)
    analysis_path.write_text(json.dumps({
        "target_admitted_for_scale": True,
        "root_results": [evidence, deepcopy(evidence)],
    }))
    shallow_path.write_text("".join(json.dumps(shallow_row(i)) + "\n" for i in (0, 1)))
    try:
        build(Namespace(
            panel=panel_path, analysis=analysis_path, shallow=shallow_path,
            output=tmp_path / "o.pt", report=tmp_path / "r.json", minimum_records=1,
        ))
    except ValueError as exc:
        assert "duplicate root evidence" in str(exc)
    else:
        raise AssertionError("duplicate teacher evidence was accepted")
