#!/usr/bin/env python3
"""Freeze 200 label-blind actionable Cycle 12 TRAIN states by dependency cluster."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


CLUSTER_DOMAIN = "cycle13-train-cluster-v1"
STATE_DOMAIN = "cycle13-actionable-state-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank(domain: str, *parts: object) -> str:
    return hashlib.sha256("\0".join((domain, *(str(part) for part in parts))).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-index", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    corpus = {
        row["raw_sha256"]: row
        for row in (json.loads(line) for line in args.corpus.read_text().splitlines() if line)
    }
    with gzip.open(args.master_index, "rt", encoding="utf-8") as handle:
        master = [json.loads(line) for line in handle if line.strip()]
    train = [row for row in master if row["split"] == "train"]
    if len(train) != 12179 or len({row["dependency_cluster_id"] for row in train}) != len(train):
        raise ValueError("Cycle 12 TRAIN inventory or dependency uniqueness changed")
    candidates = []
    nonactionable_clusters = 0
    for battle_row in train:
        path = args.index_root / battle_row["relative_index"]
        with gzip.open(path, "rt", encoding="ascii") as handle:
            battle = json.load(handle)
        if battle["split"] != "train" or battle["dependency_cluster_id"] != battle_row["dependency_cluster_id"]:
            raise ValueError("battle index split/cluster disagreement")
        states = [state for state in battle["states"] if state["actionable"]]
        if not states:
            nonactionable_clusters += 1
            continue
        state = min(states, key=lambda row: rank(
            STATE_DOMAIN, battle["dependency_cluster_id"], row["role"],
            row["request_index"], row["model_information_fingerprint_sha256"],
        ))
        source = corpus.get(battle["raw_sha256"])
        if source is None or not source["commit_present"]:
            raise ValueError("selected battle lacks pinned raw provenance")
        candidates.append({
            "schema": "metagross-cycle13-train-state-selection/v1",
            "dependency_cluster_id": battle["dependency_cluster_id"],
            "cluster_rank_sha256": rank(CLUSTER_DOMAIN, battle["dependency_cluster_id"]),
            "state_rank_sha256": rank(
                STATE_DOMAIN, battle["dependency_cluster_id"], state["role"],
                state["request_index"], state["model_information_fingerprint_sha256"],
            ),
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
            "role": state["role"],
            "request_index": state["request_index"],
            "command_input_index": state["command_input_index"],
            "public_event_index": state["public_event_index"],
            "model_information_fingerprint_sha256": state["model_information_fingerprint_sha256"],
            "private_request_sha256": state["private_request_sha256"],
            "causal_prefix_sha256": state["causal_prefix_sha256"],
            "legal_action_contract_sha256": state["legal_action_contract_sha256"],
            "typed_reveal_ledger_sha256": state["typed_reveal_ledger_sha256"],
            "pp_disable_sidecar_sha256": state["pp_disable_sidecar_sha256"],
        })
    selected = sorted(candidates, key=lambda row: row["cluster_rank_sha256"])[:200]
    if len(selected) != 200 or len({row["dependency_cluster_id"] for row in selected}) != 200:
        raise ValueError("could not select 200 unique TRAIN clusters")
    if any(row["split"] != "train" for row in selected):
        raise ValueError("non-TRAIN row entered Cycle 13")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    report = {
        "schema": "metagross-cycle13-train-selection-report/v1",
        "master_index_sha256": sha256(args.master_index),
        "source_corpus_sha256": sha256(args.corpus),
        "train_dependency_clusters": len(train),
        "clusters_with_actionable_states": len(candidates),
        "clusters_without_actionable_states": nonactionable_clusters,
        "selected_states": len(selected),
        "selected_dependency_clusters": len({row["dependency_cluster_id"] for row in selected}),
        "validation_or_test_index_files_opened": 0,
        "teacher_q_visit_outcome_fields_used": False,
        "selection_sha256": sha256(args.output),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
