from __future__ import annotations

import argparse
import json

from scripts.analyze_outcome_grounded_continuations import analyze
from train.outcome_grounded import RESULT_SCHEMA


def test_analysis_preserves_fully_censored_root(tmp_path) -> None:
    rows = []
    for schedule_id in (0, 1):
        samples = [
            {"world_index": 0, "rollout": rollout, "outcome": None, "decisions": 128}
            for rollout in range(8)
        ]
        rows.append({
            "schema": RESULT_SCHEMA,
            "battle_id": "battle",
            "root_id": "root",
            "schedule_id": schedule_id,
            "baseline_action": "a",
            "candidate_actions": ["a", "b"],
            "action_outcomes": {"a": samples, "b": samples},
        })
    source = tmp_path / "results.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "report.json"

    report = analyze(argparse.Namespace(results=source, output=output, seed=7))

    assert report["terminal_rate"] == 0.0
    assert report["half_split_best_agreement"] == 0.0
    assert report["raw_terminal_disagreements"] == 0
    assert report["stable_corrections"] == 0
    assert report["root_results"][0]["terminal_best_action"] is None
    assert report["root_results"][0]["terminal_q"] == {"a": None, "b": None}
