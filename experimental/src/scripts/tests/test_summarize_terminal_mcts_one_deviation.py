from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from srcs.metagross.terminal_mcts_one_deviation import SCHEMA, assignment_manifest


SCRIPT = Path(__file__).parents[1] / "summarize_terminal_mcts_one_deviation.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_terminal_mcts_one_deviation", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SEED = "2026081507"


def _write_canary(tmp_path: Path) -> Path:
    run_dir = tmp_path / "canary"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    manifest = assignment_manifest(SEED)
    games = []
    teacher_seen = 0
    production_seen = 0
    for assignment in manifest["assignments"]:
        game_index = assignment["game_index"]
        arm = assignment["arm"]
        if arm == "teacher":
            teacher_seen += 1
            candidate_win = teacher_seen <= 8
        else:
            production_seen += 1
            candidate_win = production_seen <= 3
        pair_index = assignment["pair_index"]
        role = "x" if assignment["pair_leg"] == 1 else "y"
        username = f"tm1b{role}{game_index:03d}abcd"
        frozen_assignment = {
            **assignment,
            "randomization_seed": SEED,
            "schedule_sha256": manifest["schedule_sha256"],
            "username": username,
        }
        production_action = "move 1"
        teacher_action = "switch 2"
        final_choice = teacher_action if arm == "teacher" else production_action
        row = {
            "choice_override": {
                "terminal_mcts_teacher": {
                    "schema": "metagross-terminal-mcts-live-decision/v1",
                    "decision": "override",
                    "baseline_action": production_action,
                    "selected_action": teacher_action,
                },
                "terminal_mcts_production_choice": production_action,
                "terminal_mcts_one_deviation": {
                    "schema": SCHEMA,
                    "assignment": frozen_assignment,
                    "battle_tag": f"battle-fresh-{game_index}",
                    "decision_idx": 4,
                    "teacher_query_index": 1,
                    "production_action": production_action,
                    "teacher_action": teacher_action,
                    "teacher_decision_sha256": "a" * 64,
                    "eligible": True,
                    "intervention_applied": arm == "teacher",
                    "locked_after_decision": True,
                    "integrity_failure": None,
                    "continuation": "unchanged_500ms_r1_production_search",
                },
                "final_choice": final_choice,
            }
        }
        (logs / f"{username}.search.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
        games.append(
            {
                "game_index": game_index,
                "pair_index": pair_index,
                "pair_leg": assignment["pair_leg"],
                "pair_id": f"fresh-pair-{pair_index}",
                "battle_tag": f"battle-fresh-{game_index}",
                "winner": "agent_a" if candidate_win else "agent_b",
                "void": False,
            }
        )
    (run_dir / "result.json").write_text(
        json.dumps({"games": games}), encoding="utf-8"
    )
    return run_dir


def test_valid_large_effect_canary_passes_to_replication(tmp_path: Path) -> None:
    report = MODULE.summarize(_write_canary(tmp_path), seed=SEED)
    assert report["integrity"]["ok"] is True
    assert report["eligible_games"]["win_rate_effect"] == 0.5
    assert report["frozen_gate"]["decision"] == "PASS_TO_POWERED_REPLICATION"
    assert report["frozen_gate"]["deployment_authorized"] is False


def test_second_eligible_opportunity_is_rejected(tmp_path: Path) -> None:
    run_dir = _write_canary(tmp_path)
    path = next((run_dir / "logs").glob("*.search.jsonl"))
    payload = path.read_text(encoding="utf-8")
    path.write_text(payload + payload, encoding="utf-8")
    report = MODULE.summarize(run_dir, seed=SEED)
    assert report["integrity"]["ok"] is False
    assert report["frozen_gate"]["decision"] == "STOP_CYCLE1B"
    assert any("more than one eligible" in error for error in report["integrity"]["errors"])
