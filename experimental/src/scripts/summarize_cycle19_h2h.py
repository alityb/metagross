#!/usr/bin/env python3
"""Fail-closed Cycle 19 prospective H2H summarizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from experimental.src.scripts.monitor_cycle19_operational_smoke import (
    expected_command,
    received_request,
    rows,
    validate_teacher,
)
from experimental.src.scripts.verify_cycle19_h2h_freeze import verify


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engine_provenance(log_path: Path, engine_sha: str) -> dict:
    prefix = "POKE_ENGINE_PROVENANCE "
    lines = [line for line in log_path.read_text(errors="replace").splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(f"missing/duplicate engine provenance: {log_path}")
    value = json.loads(lines[0][len(prefix):])
    if (
        value.get("native_sha256") != engine_sha
        or value.get("native_reveal_masks") is not True
        or value.get("mode") != "exact_pinned_experimental_runtime"
    ):
        raise RuntimeError(f"wrong spawned engine provenance: {log_path}")
    return value


def validate_candidate_file(search_path: Path) -> tuple[int, list[float], int]:
    search_rows = rows(search_path)
    if not search_rows:
        raise RuntimeError(f"candidate game has no decision: {search_path}")
    protocol_path = search_path.with_name(
        search_path.name.removesuffix(".search.jsonl") + ".protocol.jsonl"
    )
    protocol = rows(protocol_path)
    sent = [
        row
        for row in protocol
        if row.get("direction") == "sent"
        and isinstance(row.get("messages"), list)
        and row["messages"]
        and str(row["messages"][0]).startswith(("/choose move ", "/switch "))
    ]
    sent_index = 0
    latencies = []
    overrides = 0
    for search_row in search_rows:
        override = search_row.get("choice_override") or {}
        teacher = override.get("terminal_mcts_teacher")
        if not isinstance(teacher, dict):
            raise RuntimeError(f"candidate decision lacks teacher receipt: {search_path}")
        validate_teacher(teacher)
        selected = str(teacher["selected_action"])
        if search_row.get("choice") != selected or override.get("final_choice") != selected:
            raise RuntimeError("candidate action was not installed")
        if str(teacher.get("reason", "")).startswith("fail_closed"):
            raise RuntimeError("candidate fell back")
        while sent_index < len(sent) and int(sent[sent_index].get("time_ns", 0)) < int(search_row["time_ns"]):
            sent_index += 1
        if sent_index >= len(sent):
            raise RuntimeError("candidate decision has no subsequent Showdown send")
        sent_row = sent[sent_index]
        sent_index += 1
        request = received_request(protocol, int(sent_row["time_ns"]))
        command, rqid = expected_command(selected, request)
        if sent_row["messages"] != [command, rqid]:
            raise RuntimeError("candidate sent command differs from selected action")
        latencies.append(float(teacher["elapsed_ms"]))
        overrides += int(teacher.get("decision") == "override")
    if len(sent) != len(search_rows):
        raise RuntimeError("candidate search/send decision counts differ")
    return len(search_rows), latencies, overrides


def wilson(wins: int, games: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return center - half, center + half


def summarize(run: Path, manifest_path: Path) -> dict:
    manifest = verify(manifest_path)
    result_path = run / "h2h-result.json"
    payload = json.loads(result_path.read_text())
    games = payload.get("games") or []
    summary = payload.get("summary") or {}
    if len(games) != 20 or summary.get("completed_games") != 20:
        raise RuntimeError("Cycle19 did not complete exactly 20 games")
    if summary.get("void_games") != 0 or summary.get("decisive_games") != 20:
        raise RuntimeError("Cycle19 contains a void or nondecisive game")
    if any(game.get("void") or game.get("error") or game.get("winner") not in {"agent_a", "agent_b"} for game in games):
        raise RuntimeError("Cycle19 contains a failed game row")
    if (
        sum(game.get("challenger") == "agent_a" for game in games) != 10
        or sum(game.get("acceptor") == "agent_a" for game in games) != 10
    ):
        raise RuntimeError("candidate challenger/acceptor orientation is not 10/10")
    by_pair: dict[int, list[dict]] = {}
    for game in games:
        by_pair.setdefault(int(game["pair_index"]), []).append(game)
    if set(by_pair) != set(range(1, 11)):
        raise RuntimeError("Cycle19 pair indices are incomplete")
    for pair_index, pair_games in by_pair.items():
        if {row["pair_leg"] for row in pair_games} != {1, 2}:
            raise RuntimeError(f"pair {pair_index} lacks both legs")
        if len({row["pair_id"] for row in pair_games}) != 1:
            raise RuntimeError(f"pair {pair_index} identity changed")
        if len({tuple(sorted((row["team_1_sha256"], row["team_2_sha256"]))) for row in pair_games}) != 1:
            raise RuntimeError(f"pair {pair_index} teams changed")
        assignments = {(row["agent_a_team_sha256"], row["agent_b_team_sha256"]) for row in pair_games}
        if len(assignments) != 2:
            raise RuntimeError(f"pair {pair_index} was not mirrored")

    engine_sha = manifest["engine"]["native_sha256"]
    logs = sorted((run / "h2h-logs").glob("*.log"))
    if len(logs) != 40:
        raise RuntimeError(f"expected 40 spawned Foul Play logs, found {len(logs)}")
    for log in logs:
        engine_provenance(log, engine_sha)
    search_paths = sorted((run / "h2h-logs").glob("*.search.jsonl"))
    if len(search_paths) != 40:
        raise RuntimeError("expected 40 search logs")
    for protocol_path in (run / "h2h-logs").glob("*.protocol.jsonl"):
        if any(
            row.get("direction") in {"send_failure", "send_rejected", "reconnect"}
            for row in rows(protocol_path)
        ):
            raise RuntimeError(f"operational protocol failure: {protocol_path}")
    candidate_paths = []
    comparator_paths = []
    for path in search_paths:
        has_candidate = any(
            isinstance((row.get("choice_override") or {}).get("terminal_mcts_teacher"), dict)
            for row in rows(path)
        )
        (candidate_paths if has_candidate else comparator_paths).append(path)
    if len(candidate_paths) != 20 or len(comparator_paths) != 20:
        raise RuntimeError("candidate/comparator game assignment is not 20/20")
    decision_count = 0
    override_count = 0
    latencies = []
    for path in candidate_paths:
        count, observed_latencies, overrides = validate_candidate_file(path)
        decision_count += count
        latencies.extend(observed_latencies)
        override_count += overrides
    for path in comparator_paths:
        if any((row.get("choice_override") or {}).get("terminal_mcts_teacher") for row in rows(path)):
            raise RuntimeError("candidate controller leaked into comparator")

    wins = int(summary.get("agent_a_wins"))
    losses = int(summary.get("agent_a_losses"))
    if wins + losses != 20:
        raise RuntimeError("candidate result accounting is incomplete")
    low, high = wilson(wins, 20)
    report = {
        "schema": "metagross-cycle19-h2h-result/v1",
        "status": "pass" if wins >= 13 else "fail",
        "games": 20,
        "mirrored_pairs": 10,
        "candidate_wins": wins,
        "candidate_losses": losses,
        "candidate_win_rate": wins / 20,
        "wilson95": [low, high],
        "candidate_game_files": 20,
        "comparator_game_files": 20,
        "candidate_decisions": decision_count,
        "candidate_overrides": override_count,
        "candidate_pass_through": decision_count - override_count,
        "candidate_mean_latency_ms": statistics.fmean(latencies),
        "candidate_p95_latency_ms": sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)],
        "spawned_engine_receipts": 40,
        "semantic_operational_integrity_failures": 0,
        "result_sha256": sha(result_path),
        "postrun_manifest_integrity": "pass",
        "gate": "continue_only_if_at_least_13_wins",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run.resolve(), args.manifest.resolve())
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
