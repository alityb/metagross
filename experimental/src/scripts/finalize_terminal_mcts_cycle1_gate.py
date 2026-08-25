#!/usr/bin/env python3
"""Bind the fresh latency canary to the frozen Cycle-1 opened-data gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", type=Path, required=True)
    parser.add_argument("--latency-report", type=Path, required=True)
    parser.add_argument("--latency-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    integration = json.loads(args.integration.read_text())
    latency = json.loads(args.latency_report.read_text())
    if (
        integration.get("schema") != "metagross-terminal-mcts-controller-opened-integration/v1"
        or latency.get("schema") != "metagross-outcome-grounded-collection-report/v1"
        or latency.get("engine_binary_sha256") != integration.get("engine_binary_sha256")
        or latency.get("configuration", {}).get("continuation_iterations") != 2048
        or latency.get("configuration", {}).get("rollouts") != 4
        or latency.get("samples") != 128
    ):
        raise ValueError("latency canary does not match Cycle-1 semantics")
    checks = dict(integration["checks"])
    checks["latency_canary"] = (
        args.latency_seconds <= 30.0 and latency.get("terminal_rate", 0.0) >= 0.95
    )
    admitted = all(checks.values())
    report = {
        "schema": "metagross-terminal-mcts-cycle1-gate/v1",
        "inputs": {
            "integration_sha256": sha256(args.integration),
            "latency_report_sha256": sha256(args.latency_report),
        },
        "checks": checks,
        "latency": {
            "wall_seconds": args.latency_seconds,
            "limit_seconds": 30.0,
            "samples": latency["samples"],
            "terminal_rate": latency["terminal_rate"],
            "continuation_searches": latency["continuation_searches"],
        },
        "aggregate": integration["aggregate"],
        "prospective_h2h_allowed": admitted,
        "confirmation": integration["confirmation"],
        "resources": {"local_cpu_only": True, "gpu_used": False, "cloud_used": False, "paid_cost_usd": 0.0},
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if admitted else 2)


if __name__ == "__main__":
    main()
