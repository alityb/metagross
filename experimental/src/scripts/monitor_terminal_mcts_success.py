#!/usr/bin/env python3
"""Detached integrity monitor for the frozen 50-to-200 H2H continuation."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--evaluator-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", default=30.0, type=float)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def wilson(wins: int, games: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games))
        / denominator
    )
    return center - margin, center + margin


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_progress(games: list[dict[str, Any]]) -> None:
    if len(games) % 2:
        raise ValueError("progress ended in an incomplete pair")
    indexes = [int(row["game_index"]) for row in games]
    if indexes != list(range(1, len(games) + 1)):
        raise ValueError("game indexes are not contiguous")
    if any(row.get("void") for row in games):
        raise ValueError("void game observed")
    if any(row.get("error") is not None for row in games):
        raise ValueError("game error observed")
    for offset in range(0, len(games), 2):
        left, right = games[offset : offset + 2]
        if left["pair_id"] != right["pair_id"]:
            raise ValueError("pair ID mismatch")
        if left["battle_seed"] != right["battle_seed"]:
            raise ValueError("mirrored battle seed mismatch")
        if left["agent_a_team_sha256"] != right["agent_b_team_sha256"]:
            raise ValueError("agent A/B team mirror mismatch")
        if left["agent_b_team_sha256"] != right["agent_a_team_sha256"]:
            raise ValueError("agent B/A team mirror mismatch")


def evaluator_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def finalize(root: Path, run_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(root / "experimental/src/scripts/summarize_terminal_mcts_h2h.py"),
            str(run_dir),
            "--output",
            str(run_dir / "teacher-telemetry-summary.json"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "experimental/src/scripts/aggregate_terminal_mcts_h2h.py"),
            "--canary-dir",
            str(root / "experimental/runs/terminal_mcts_direct_controller_20260815/prospective-canary-10"),
            "--extension-dir",
            str(root / "experimental/runs/terminal_mcts_direct_controller_20260815/prospective-extension-40-domain-v2"),
            "--success-dir",
            str(run_dir),
            "--frozen-futility-result",
            str(root / "experimental/runs/terminal_mcts_direct_controller_20260815/futility-look-50.json"),
            "--expected-futility-sha256",
            "4ba46a21d892f9b5ed0760f4bef582aac2dd8744abd3e77153e4a7e08601f26c",
            "--expected-canary-result-sha256",
            "9a225a4ccfd5640f989f6b1a0349c46de086ff97c6f1b1ec2ecabadc6948fd1e",
            "--expected-canary-telemetry-sha256",
            "dbbadda12ff7fc4f155edaea80d97603b07daf11023ba929d02954176fd0a8ec",
            "--output",
            str(run_dir.parent / "success-look-200.json"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    last_count = -1
    while True:
        try:
            final_path = args.run_dir / "result.json"
            progress_path = args.run_dir / "result.json.progress.json"
            source = final_path if final_path.exists() else progress_path
            games: list[dict[str, Any]] = []
            if source.exists():
                payload = load_json(source)
                raw_games = payload.get("games", [])
                if not isinstance(raw_games, list):
                    raise ValueError("games must be a list")
                games = raw_games
                validate_progress(games)
            extension_wins = sum(row.get("winner") == "agent_a" for row in games)
            aggregate_games = 50 + len(games)
            aggregate_wins = 23 + extension_wins
            lower, upper = wilson(aggregate_wins, aggregate_games)
            status = {
                "schema": "metagross-terminal-mcts-success-monitor/v1",
                "status": "complete" if final_path.exists() else "running",
                "extension_games": len(games),
                "extension_wins": extension_wins,
                "aggregate_games": aggregate_games,
                "aggregate_wins": aggregate_wins,
                "aggregate_winrate": aggregate_wins / aggregate_games,
                "wilson_ci95": [lower, upper],
                "success_boundary_met": lower > 0.5,
                "voids": 0,
                "semantic_breach": False,
                "evaluator_pid": args.evaluator_pid,
                "evaluator_alive": evaluator_alive(args.evaluator_pid),
                "sealed_confirmation_panel_rows_read": 0,
                "paid_compute_usd": 0,
            }
            atomic_json(args.output, status)
            if len(games) != last_count:
                print(json.dumps(status, sort_keys=True), flush=True)
                last_count = len(games)
            if final_path.exists():
                if len(games) != 150:
                    raise ValueError("final result did not contain 150 games")
                finalize(root, args.run_dir)
                return
            if not evaluator_alive(args.evaluator_pid):
                raise RuntimeError("evaluator exited before final result")
        except Exception as exc:
            atomic_json(
                args.output,
                {
                    "schema": "metagross-terminal-mcts-success-monitor/v1",
                    "status": "breach",
                    "semantic_breach": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "sealed_confirmation_panel_rows_read": 0,
                    "paid_compute_usd": 0,
                },
            )
            raise
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
