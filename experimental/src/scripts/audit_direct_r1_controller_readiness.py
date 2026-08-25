#!/usr/bin/env python3
"""Fail-closed readiness audit for the frozen direct R1 controller."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--training-panel", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    certificate: dict[str, Any] = json.loads(args.certificate.read_text())
    if certificate.get("claim") != "causal_history_dual_r1_terminating_continuation_certificate":
        raise ValueError("wrong dual-R1 certificate claim")
    rows = [json.loads(line) for line in args.training_panel.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("training panel is empty")
    readiness = certificate.get("continuation_readiness", {})
    certificate_admitted = (
        readiness.get("status") == "admitted"
        and readiness.get("r1_continuation_value_allowed") is True
        and float(certificate.get("terminal_rate", 0.0)) >= 0.95
    )
    own_history = all(
        row.get("selection", {}).get("history_authority") == "schema6_snapshot"
        or row.get("causal_history", {}).get("authority") == "schema6_snapshot"
        for row in rows
    )
    # The frozen causal panel serializes one observer's policy authority. A
    # belief-world opponent tracker/snapshot is intentionally not invented.
    opponent_belief_history = all(
        row.get("opponent_causal_history", {}).get("authority") == "schema6_snapshot"
        for row in rows
    )
    blockers = []
    if not certificate_admitted:
        blockers.append("dual_r1_terminal_semantic_coverage_below_95pct")
    if not own_history:
        blockers.append("own_schema6_history_authority_missing")
    if not opponent_belief_history:
        blockers.append("belief_world_opponent_causal_history_not_reconstructible")
    allowed = not blockers
    report = {
        "schema": "metagross-direct-r1-controller-readiness/v1",
        "claim": "readiness_only_no_confirmation_opened",
        "inputs": {
            "certificate_sha256": sha256(args.certificate),
            "training_panel_sha256": sha256(args.training_panel),
            "protocol_sha256": sha256(args.protocol),
        },
        "certificate": {
            "terminal_rate": certificate.get("terminal_rate"),
            "terminal_rollouts": certificate.get("counts", {}).get("terminal_rollouts"),
            "rollouts": certificate.get("counts", {}).get("rollouts"),
            "failure_counts": certificate.get("failure_counts", {}),
            "admitted": certificate_admitted,
        },
        "panel": {
            "opened_training_roots_read": len(rows),
            "own_schema6_history_authority": own_history,
            "opponent_belief_history_authority": opponent_belief_history,
        },
        "blockers": blockers,
        "next_stage_allowed": allowed,
        "untouched_confirmation": {
            "materialized": False,
            "rows_read": 0,
            "outcomes_read": 0,
            "preserved": True,
        },
        "resources": {
            "local_cpu_only": True,
            "gpu_used": False,
            "cloud_used": False,
            "paid_cost_usd": 0.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if allowed else 2)


if __name__ == "__main__":
    main()
