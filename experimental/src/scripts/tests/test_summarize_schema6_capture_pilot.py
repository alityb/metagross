import json
import subprocess
import sys
from pathlib import Path


def test_frozen_pilot_summary_admits_exact_strata(tmp_path):
    specifications = {
        "peer": (300, "production_r1_search_first", 600),
        "direct_r1": (100, "direct_r1", 100),
        "unguided": (100, "foul_play", 100),
    }
    for profile, (games, agent_b, groups) in specifications.items():
        directory = tmp_path / profile
        directory.mkdir()
        (directory / "result.json").write_text(json.dumps({"summary": {
            "completed_games": games,
            "agent_a": "production_r1_search_first",
            "agent_b": agent_b,
            "void_games": 0,
        }}))
        (directory / "schema6-capture-audit.json").write_text(json.dumps({
            "admitted": True, "groups": groups, "complete_groups": groups,
            "capture_rate": 1.0,
        }))
        (directory / "schema6-panel-bridge-audit.json").write_text(json.dumps({
            "admitted": True, "eligible_groups": groups, "candidate_rows": groups,
        }))
    output = tmp_path / "summary.json"
    script = Path(__file__).parents[1] / "summarize_schema6_capture_pilot.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "--output", str(output)],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert report["admitted"]
    assert report["completed_games"] == 500
    assert report["capture_rate"] == 1.0
    assert not report["scale_admitted"]
    assert len(report["pinned_runtime"]["capture_engine_binary_sha256"]) == 64
