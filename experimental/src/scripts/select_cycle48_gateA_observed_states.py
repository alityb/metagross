#!/usr/bin/env python3
"""Freeze 64 label-blind Cycle48 Gate A TRAIN clusters with 8 chronological observed-state slots.

Selection is label-blind: it reads only split/cluster identity, per-role
actionable-state counts, chronological request indices, and frozen provenance
hashes. It never reads observed actions, teacher values, outcomes, or any
validation/test/sealed index. Cycle13's 200 already-opened clusters are
excluded, which also excludes every root opened by Cycles 13-17 and 45-47.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

CLUSTER_DOMAIN = "cycle48-gateA-cluster-v1"
ROLE_DOMAIN = "cycle48-gateA-role-v1"
SLOTS = 8
CLUSTER_COUNT = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank(domain: str, *parts: object) -> str:
    return hashlib.sha256("\0".join((domain, *(str(part) for part in parts))).encode("utf-8")).hexdigest()


def slot_base_positions(candidate_count: int) -> list[int]:
    """Evenly spread 8 chronological base positions across the candidate list."""
    if candidate_count < SLOTS:
        raise ValueError("cluster has fewer than eight actionable candidates")
    return [(index * (candidate_count - 1)) // (SLOTS - 1) for index in range(SLOTS)]


CANDIDATE_FIELDS = (
    "request_index", "command_input_index", "public_event_index",
    "model_information_fingerprint_sha256", "private_request_sha256",
    "causal_prefix_sha256", "legal_action_contract_sha256",
    "typed_reveal_ledger_sha256", "pp_disable_sidecar_sha256",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-index", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--exclude-selection", type=Path, required=True)
    parser.add_argument("--worktrees", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    corpus = {
        row["raw_sha256"]: row
        for row in (json.loads(line) for line in args.corpus.read_text().splitlines() if line)
    }
    excluded = {
        json.loads(line)["dependency_cluster_id"]
        for line in args.exclude_selection.read_text().splitlines() if line
    }
    if len(excluded) != 200:
        raise ValueError("Cycle13 exclusion list is not 200 clusters")
    worktrees = json.loads(args.worktrees.read_text())

    with gzip.open(args.master_index, "rt", encoding="utf-8") as handle:
        master = [json.loads(line) for line in handle if line.strip()]
    train = [row for row in master if row["split"] == "train"]
    if len(train) != 12179 or len({row["dependency_cluster_id"] for row in train}) != len(train):
        raise ValueError("Cycle12 TRAIN inventory or dependency uniqueness changed")

    ranked = sorted(train, key=lambda row: rank(CLUSTER_DOMAIN, row["dependency_cluster_id"]))
    selected: list[dict] = []
    scanned = ineligible_excluded = ineligible_short = ineligible_runtime = 0
    for battle_row in ranked:
        if len(selected) == CLUSTER_COUNT:
            break
        scanned += 1
        if battle_row["dependency_cluster_id"] in excluded:
            ineligible_excluded += 1
            continue
        path = args.index_root / battle_row["relative_index"]
        with gzip.open(path, "rt", encoding="ascii") as handle:
            battle = json.load(handle)
        if battle["split"] != "train" or battle["dependency_cluster_id"] != battle_row["dependency_cluster_id"]:
            raise ValueError("battle index split/cluster disagreement")
        if battle["showdown_commit"] not in worktrees:
            ineligible_runtime += 1
            continue
        source = corpus.get(battle["raw_sha256"])
        if source is None or not source["commit_present"]:
            ineligible_runtime += 1
            continue
        by_role: dict[str, list[dict]] = {}
        for state in battle["states"]:
            # Candidate predicate (label-blind): actionable with an observed
            # own command. Presence only is read, never the command content;
            # dangling final requests the human never answered are excluded
            # because they carry no behavior anchor.
            if state["actionable"] and state.get("observed_action") is not None:
                by_role.setdefault(state["role"], []).append(state)
        eligible_roles = sorted(role for role, states in by_role.items() if len(states) >= SLOTS)
        if not eligible_roles:
            ineligible_short += 1
            continue
        role = min(eligible_roles, key=lambda value: rank(
            ROLE_DOMAIN, battle["dependency_cluster_id"], value,
        ))
        candidates = sorted(by_role[role], key=lambda state: state["request_index"])
        if len({state["request_index"] for state in candidates}) != len(candidates):
            raise ValueError("duplicate request_index inside one role")
        selected.append({
            "schema": "metagross-cycle48-gateA-selection/v1",
            "dependency_cluster_id": battle["dependency_cluster_id"],
            "cluster_rank_sha256": rank(CLUSTER_DOMAIN, battle["dependency_cluster_id"]),
            "role_rank_sha256": rank(ROLE_DOMAIN, battle["dependency_cluster_id"], role),
            "split": "train",
            "battle_id": battle["battle_id"],
            "panel_index": battle_row["panel_index"],
            "relative_index": battle_row["relative_index"],
            "source": battle["source"],
            "raw_path": source["raw_path"],
            "raw_sha256": battle["raw_sha256"],
            "inputlog_sha256": battle["inputlog_sha256"],
            "public_log_sha256": battle["public_log_sha256"],
            "showdown_commit": battle["showdown_commit"],
            "role": role,
            "actionable_count": len(candidates),
            "slot_base_positions": slot_base_positions(len(candidates)),
            "candidates": [
                {key: state[key] for key in CANDIDATE_FIELDS} | {"role": state["role"]}
                for state in candidates
            ],
        })
    if len(selected) != CLUSTER_COUNT or len({row["dependency_cluster_id"] for row in selected}) != CLUSTER_COUNT:
        raise ValueError("could not select 64 unique fresh TRAIN clusters")
    if any(row["split"] != "train" for row in selected):
        raise ValueError("non-TRAIN row entered Cycle48 Gate A")
    if {row["dependency_cluster_id"] for row in selected} & excluded:
        raise ValueError("excluded Cycle13 cluster entered Cycle48 Gate A")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    report = {
        "schema": "metagross-cycle48-gateA-selection-report/v1",
        "master_index_sha256": sha256(args.master_index),
        "source_corpus_sha256": sha256(args.corpus),
        "excluded_selection_sha256": sha256(args.exclude_selection),
        "train_dependency_clusters": len(train),
        "clusters_scanned": scanned,
        "clusters_skipped_cycle13_excluded": ineligible_excluded,
        "clusters_skipped_fewer_than_8_commanded_actionable": ineligible_short,
        "clusters_skipped_runtime_provenance": ineligible_runtime,
        "selected_dependency_clusters": len(selected),
        "selected_state_slots": len(selected) * SLOTS,
        "observed_action_fields_in_selection": 0,
        "validation_or_test_index_files_opened": 0,
        "teacher_q_visit_outcome_fields_used": False,
        "selection_sha256": sha256(args.output),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
