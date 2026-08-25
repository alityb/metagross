from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_switch_to_switch_outcome_panel import build
from train.outcome_grounded import PANEL_SCHEMA


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def source_root(index: int) -> dict:
    state = f"state-{index}"
    worlds = [{"world_index": world, "weight": 0.125, "state": state,
               "state_sha256": hashlib.sha256(state.encode()).hexdigest()} for world in range(8)]
    return {"schema": "metagross-public-mcts-root-panel/v1", "root_id": f"root-{index}",
            "battle_id": f"battle-{index}", "public_features": [0.0] * 18,
            "decision_idx": 3, "public_reveal_fractions": [0.0] * 4, "selection": {},
            "schedules": [{"schedule_id": schedule, "worlds": worlds} for schedule in (0, 1)]}


def search_row(index: int, schedule: int) -> dict:
    return {"root_id": f"root-{index}", "schedule_id": schedule,
            "selected_action": "switch a", "root_statistics": {
                "visit_entropy": 0.8, "weighted_top_action_disagreement": 0.3,
                "weighted_js_divergence": 0.2, "aggregate_top_visit_mass": 0.5,
                "aggregate_top_two_margin": 0.1,
            }, "action_statistics": {
                "switch a": {"visit_mass": 0.45, "mean_value": 0.5},
                "switch b": {"visit_mass": 0.35, "mean_value": 0.6},
                "attack": {"visit_mass": 0.20, "mean_value": 0.7},
            }}


def test_builder_uses_only_switches_and_rejects_opened_overlap(tmp_path: Path) -> None:
    panel = tmp_path / "panel.jsonl"; write_jsonl(panel, [source_root(i) for i in range(50)])
    shallow = tmp_path / "shallow.jsonl"
    write_jsonl(shallow, [search_row(i, schedule) for i in range(50) for schedule in (0, 1)])
    freeze = tmp_path / "freeze.json"; freeze.write_text(json.dumps({"root_ids": [], "battle_ids": []}))
    args = SimpleNamespace(panel=panel, shallow=shallow, development_freeze=[freeze],
        maximum_roots=10, seed=3, output=tmp_path / "out.jsonl", report=tmp_path / "report.json")
    report = build(args)
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert report["eligible_roots"] == 50 and len(rows) == 10
    assert all(row["schema"] == PANEL_SCHEMA for row in rows)
    assert all(row["candidate_actions"] == ["switch a", "switch b"] for row in rows)
    assert report["selection_used_oracle_50k"] is False
    freeze.write_text(json.dumps({"root_ids": ["root-0"], "battle_ids": []}))
    with pytest.raises(ValueError, match="overlaps opened outcome development"):
        build(args)
