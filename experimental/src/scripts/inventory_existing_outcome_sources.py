#!/usr/bin/env python3
"""Inventory local human/self-play/league/ladder artifacts for causal root mining."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_identity(path: Path, identity: Callable[[dict[str, Any]], str | None]) -> tuple[int, int]:
    rows = 0; identities: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line); rows += 1
            value = identity(row)
            if value is not None:
                identities.add(str(value))
    return rows, len(identities)


def csv_count(path: Path, header: bool) -> int:
    with path.open(newline="") as handle:
        rows = sum(1 for row in csv.reader(handle) if row)
    return max(0, rows - int(header))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources: list[dict[str, Any]] = []

    def add(source_id: str, kind: str, path: Path, records: int, battles: int | None, status: str, reason: str, **extra: Any) -> None:
        sources.append({
            "source_id": source_id, "kind": kind, "path": str(path), "exists": path.exists(),
            "records": records, "physical_battles": battles, "causal_root_status": status,
            "reason": reason, **extra,
        })

    human = Path("experimental/data/parsed_replays/index.csv")
    add("human_replays", "human", human, csv_count(human, True), 11_842, "requires_rematerialization",
        "terminal human trajectories exist, but no corrected-R1 snapshot, causal reveal mask, or frozen belief worlds are stored")
    gate = Path("experimental/data/gate1_pfsp_indexed/index.csv")
    add("gate1_pfsp", "pfsp_selfplay", gate, csv_count(gate, False), None, "requires_rematerialization",
        "trajectories exist; current causal snapshots and belief schedules must be rebuilt")
    league = Path("experimental/runs/online_rl_g5_league_5k_20260730/generation_005/collection/BATTLE_LEDGER.jsonl")
    league_rows, _ = jsonl_identity(league, lambda row: row.get("result", {}).get(" Battle ID"))
    league_traj = list(league.parent.glob("**/learner_trajectories/*.lz4"))
    add("league_g5", "league_selfplay", league, league_rows, league_rows, "ledger_only_local",
        "5,000-battle ledger is present but local causal R1 snapshots/belief worlds are absent", local_trajectory_files=len(league_traj))
    preserved_a = Path("experimental/runs/schema6_local_5000_20260814_r1/peer/agent-a-decisions.jsonl.dual-r1-roots.jsonl")
    preserved_b = Path("experimental/runs/schema6_local_5000_20260814_r1/peer/agent-b-decisions.jsonl.dual-r1-roots.jsonl")
    pa_rows, pa_battles = jsonl_identity(preserved_a, lambda row: row.get("identity", {}).get("battle_tag"))
    pb_rows, pb_battles = jsonl_identity(preserved_b, lambda row: row.get("identity", {}).get("battle_tag"))
    combined_tags = set()
    for path in (preserved_a, preserved_b):
        with path.open() as handle:
            for line in handle:
                if line.strip(): combined_tags.add(json.loads(line)["identity"]["battle_tag"])
    add("preserved_42", "corrected_r1_selfplay", preserved_a.parent, pa_rows + pb_rows, len(combined_tags), "root_materializable",
        "exact schema-6 corrected-R1 snapshots are present for the preserved 42-game branch plus two earlier smoke battles; requires panel materialization and search", pov_battles=[pa_battles, pb_battles], paid_branch_games=42, earlier_smoke_games=2)
    historical = Path("experimental/runs/outcome_residual_scale_20260814/panel-950.jsonl")
    h_rows, h_battles = jsonl_identity(historical, lambda row: row.get("battle_id"))
    add("historical_accepted_r1_950", "accepted_r1_selfplay", historical, h_rows, h_battles, "root_ready_search_ready",
        "exact causal roots and two belief schedules exist; 20k/50k search artifacts already computed")
    causal = Path("experimental/runs/causal_action_q_local_20260814/training-panel.jsonl")
    c_rows, c_battles = jsonl_identity(causal, lambda row: row.get("battle_id"))
    add("causal_action_q_1500", "accepted_r1_selfplay", causal, c_rows, c_battles, "root_ready_needs_search",
        "exact causal roots exist and are disjoint from the 950; previously consumed development data")
    for source_id, path in (
        ("mcts_v3_final", Path("experimental/data/mcts_v3_final/mcts_v3_targets.jsonl")),
        ("mcts_v3_round2", Path("experimental/data/mcts_v3_round2/mcts_v3_targets.jsonl")),
        ("mcts_v3_partial", Path("experimental/data/mcts_v3_partial_snapshot/mcts_v3_targets.jsonl")),
    ):
        rows, battles = jsonl_identity(path, lambda row: row.get("battle_tag"))
        add(source_id, "search_selfplay_targets", path, rows, battles, "represented_by_causal_panels",
            "policy/search targets overlap derived accepted-R1 archives; canonical root-ready panels take precedence")
    ladder_logs = list(Path("experimental/runs").glob("**/logs/*.log"))
    add("ladder_logs", "public_ladder", Path("experimental/runs"), len(ladder_logs), None, "evaluation_only_not_root_ready",
        "logs lack exact private request snapshots and frozen belief schedules; registry forbids development-label reuse")

    report = {
        "schema": "metagross-existing-outcome-source-inventory/v1",
        "purpose": "development_root_mining_only",
        "sources": sources,
        "summary": {
            "root_ready_battles": h_battles + c_battles,
            "root_materializable_battles": len(combined_tags),
            "immediately_search_ready_battles": h_battles,
            "human_trajectories": csv_count(human, True),
            "league_battles": league_rows,
            "rule": "do_not_mix_ledger_or_trajectory_counts_with_causally_valid_root_counts",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
