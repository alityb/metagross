#!/usr/bin/env python3
"""Materialize only certified outcome deviations over exact schema-6 R1 histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from scripts.build_causal_action_q_panel import schema6_history_valid
from scripts.build_mcts_v3_dataset import GroupRejected, map_move_string
from scripts.collect_outcome_grounded_continuations import read_panel
from scripts.collect_shallow_search_statistics import SCHEMA as SHALLOW_SCHEMA
from train.shallow_search_residual import FEATURE_NAMES, action_features


SCHEMA = "metagross-schema6-outcome-residual-dataset/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_snapshot(locator: dict[str, Any], cache: dict[Path, list[str]]) -> dict[str, Any]:
    if locator.get("authority") != "schema6_snapshot" or locator.get("snapshot_schema") != 6:
        raise ValueError("outcome root lacks schema-6 causal-history authority")
    path = Path(str(locator.get("snapshot_source_path", ""))).resolve()
    expected_hash = str(locator.get("snapshot_source_sha256", ""))
    if not path.is_file() or sha256(path) != expected_hash:
        raise ValueError(f"schema-6 snapshot source hash mismatch: {path}")
    if path not in cache:
        cache[path] = path.read_text().splitlines()
    line_number = locator.get("snapshot_source_line")
    if not isinstance(line_number, int) or not 1 <= line_number <= len(cache[path]):
        raise ValueError("schema-6 snapshot line is invalid")
    snapshot = json.loads(cache[path][line_number - 1])
    if not schema6_history_valid(snapshot):
        raise ValueError("located schema-6 snapshot has an invalid causal trajectory")
    return snapshot


def build(args: argparse.Namespace) -> dict[str, Any]:
    panel = read_panel(args.panel)
    analysis = json.loads(args.analysis.read_text())
    if not analysis.get("target_admitted_for_scale"):
        raise ValueError("outcome analysis did not pass the frozen teacher gate")
    root_result_rows = analysis.get("root_results", [])
    root_results = {row["root_id"]: row for row in root_result_rows}
    if len(root_results) != len(root_result_rows):
        raise ValueError("outcome analysis contains duplicate root evidence")
    if set(root_results) != {row["root_id"] for row in panel}:
        raise ValueError("outcome analysis root set differs from panel")
    shallow_rows = read_rows(args.shallow)
    if any(row.get("schema") != SHALLOW_SCHEMA for row in shallow_rows):
        raise ValueError("invalid shallow-search statistics")
    shallow = {row["pair_id"]: row for row in shallow_rows}
    if len(shallow) != len(shallow_rows):
        raise ValueError("shallow-search statistics contain duplicate pair evidence")

    records = []
    snapshot_cache: dict[Path, list[str]] = {}
    for root in panel:
        result = root_results[root["root_id"]]
        stable_action = result.get("stable_action")
        if stable_action is None:
            continue
        baseline = result["baseline_action"]
        alternatives = {
            row["action"]: row
            for row in result["alternatives"]
            if row.get("stable_correction")
        }
        evidence = alternatives.get(stable_action)
        if evidence is None:
            raise ValueError("stable action lacks certified evidence")
        context = root.get("source_context")
        if not isinstance(context, dict) or not isinstance(context.get("causal_history"), dict):
            raise ValueError("outcome panel lacks leak-free schema-6 source context")
        snapshot = load_snapshot(context["causal_history"], snapshot_cache)
        names = snapshot.get("name_table")
        if not isinstance(names, dict):
            raise ValueError("schema-6 snapshot lacks an action name table")
        try:
            baseline_index, _ = map_move_string(baseline, names)
            stable_index, _ = map_move_string(stable_action, names)
        except GroupRejected as exc:
            raise ValueError(f"certified action cannot map to R1 support: {exc}") from exc
        if baseline_index is None or stable_index is None or baseline_index == stable_index:
            raise ValueError("certified deviation does not map to two distinct R1 actions")

        selection = context.get("r1_selection")
        reveal = context.get("public_reveal_fractions")
        decision_idx = context.get("decision_idx")
        if not isinstance(selection, dict) or not isinstance(reveal, list) or not isinstance(decision_idx, int):
            raise ValueError("schema-6 source context is incomplete")
        features: dict[str, list[float]] = {}
        for action in (baseline, stable_action):
            vectors = []
            for schedule_id in (0, 1):
                row = shallow.get(f"{root['root_id']}:{schedule_id}")
                if row is None or action not in row["action_statistics"]:
                    raise ValueError("certified action is absent from live-search statistics")
                vectors.append(action_features(
                    row,
                    action,
                    reveal=[float(value) for value in reveal],
                    history=decision_idx + 1,
                    selection=selection,
                ))
            features[action] = np.mean(np.asarray(vectors, dtype=np.float64), axis=0).tolist()

        trajectory = snapshot["trajectory"]
        observation_rows = trajectory["observation_rows"]
        records.append({
            "battle_id": root["battle_id"],
            "root_id": root["root_id"],
            "baseline_action": baseline,
            "stable_action": stable_action,
            "baseline_index": int(baseline_index),
            "stable_index": int(stable_index),
            "text_tokens": observation_rows["text_tokens"],
            "numbers": observation_rows["numbers"],
            "illegal_actions": observation_rows["illegal_actions"],
            "rl2": trajectory["rl2"],
            "time_indices": trajectory["time_indices"],
            "baseline_search_features": features[baseline],
            "stable_search_features": features[stable_action],
            "mean_advantage": float(evidence["mean_advantage"]),
            "cluster_bootstrap_ci95": [float(value) for value in evidence["cluster_bootstrap_ci95"]],
            "schedule_advantages": [float(value) for value in evidence["schedule_advantages"]],
        })
    if len(records) < args.minimum_records:
        raise ValueError(f"only {len(records)} certified deviations; need {args.minimum_records}")
    if any(
        not math.isfinite(record["mean_advantage"])
        or record["cluster_bootstrap_ci95"][0] <= 0
        or any(value <= 0.01 for value in record["schedule_advantages"])
        for record in records
    ):
        raise ValueError("dataset contains a non-certified deviation")
    payload = {
        "schema": SCHEMA,
        "records": records,
        "feature_names": list(FEATURE_NAMES),
        "provenance": {
            "panel_sha256": sha256(args.panel),
            "analysis_sha256": sha256(args.analysis),
            "shallow_sha256": sha256(args.shallow),
            "sampled_state_present": False,
            "label_policy": "independently_stable_deviations_only",
        },
    }
    atomic_save(payload, args.output)
    report = {
        "schema": "metagross-schema6-outcome-residual-dataset-report/v1",
        "records": len(records),
        "mean_history": math.fsum(len(row["time_indices"]) for row in records) / len(records),
        "mean_advantage": math.fsum(row["mean_advantage"] for row in records) / len(records),
        "output_sha256": sha256(args.output),
        "provenance": payload["provenance"],
    }
    atomic_json(report, args.report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--shallow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-records", type=int, default=10)
    print(json.dumps(build(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
