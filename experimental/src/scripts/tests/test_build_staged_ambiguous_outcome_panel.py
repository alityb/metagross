import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_staged_ambiguous_outcome_panel import build
from train.outcome_grounded import PANEL_SCHEMA


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_builder_excludes_frozen_battles_and_requires_target_range(tmp_path: Path) -> None:
    root = {
        "battle_id": "fresh-battle", "root_id": "fresh-root", "decision_idx": 3,
        "public_reveal_fractions": [0.2, 0.1, 0.0, 0.0], "selection": {},
        "schedules": [{"schedule_id": 0, "worlds": []}, {"schedule_id": 1, "worlds": []}],
    }
    panel = tmp_path / "panel.jsonl"; write_jsonl(panel, [root])
    stats = {
        "visit_entropy": 0.8, "weighted_top_action_disagreement": 0.3,
        "weighted_js_divergence": 0.2, "aggregate_top_visit_mass": 0.5,
        "aggregate_top_two_margin": 0.1,
    }
    search = tmp_path / "search.jsonl"
    write_jsonl(search, [{
        "root_id": "fresh-root", "schedule_id": schedule, "selected_action": "a",
        "root_statistics": stats,
        "action_statistics": {
            "a": {"visit_mass": 0.5, "mean_value": 0.5},
            "b": {"visit_mass": 0.3, "mean_value": 0.6},
            "c": {"visit_mass": 0.2, "mean_value": 0.4},
        },
    } for schedule in (0, 1)])
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"root_ids": [], "battle_ids": []}))
    args = SimpleNamespace(
        panel=panel, shallow=search, development_freeze=freeze, consumed_panel=[],
        output=tmp_path / "out.jsonl", report=tmp_path / "report.json", seed=7,
        minimum_roots=1, maximum_roots=1,
    )
    report = build(args)
    assert report["roots"] == 1
    row = json.loads(args.output.read_text())
    assert row["schema"] == PANEL_SCHEMA
    assert row["candidate_actions"] == ["a", "b", "c"]
    freeze.write_text(json.dumps({"root_ids": [], "battle_ids": ["fresh-battle"]}))
    with pytest.raises(ValueError, match="outside frozen range"):
        build(args)
