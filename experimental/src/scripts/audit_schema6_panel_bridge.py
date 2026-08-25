#!/usr/bin/env python3
"""Certify that fresh causal-root captures are directly panel-selectable."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from argparse import Namespace
from pathlib import Path

from scripts.build_causal_action_q_panel import choose_candidates, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-log", type=Path, action="append", required=True)
    parser.add_argument("--prior-snapshot", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-groups", type=int, default=1)
    parser.add_argument("--minimum-history", type=int, default=3)
    parser.add_argument("--minimum-legal-actions", type=int, default=4)
    parser.add_argument("--minimum-entropy", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    import poke_engine

    candidate_args = Namespace(
        decision_log=args.decision_log,
        prior_snapshot=args.prior_snapshot,
        history_authority="schema6_snapshot",
        terminal_trajectories=None,
        exclude_panel=[],
        minimum_history=args.minimum_history,
        minimum_legal_actions=args.minimum_legal_actions,
        minimum_entropy=args.minimum_entropy,
        seed=args.seed,
        battles=args.minimum_groups,
        purpose="training",
    )
    candidates, failures, split_audit = choose_candidates(candidate_args, poke_engine)
    groups = {(row["battle_tag"], row["username"]) for row in candidates}
    report = {
        "schema": "metagross-schema6-panel-bridge-audit/v1",
        "admitted": len(groups) >= args.minimum_groups,
        "eligible_groups": len(groups),
        "candidate_rows": len(candidates),
        "minimum_groups": args.minimum_groups,
        "seed": args.seed,
        "selection_thresholds": {
            "minimum_history": args.minimum_history,
            "minimum_legal_actions": args.minimum_legal_actions,
            "minimum_entropy": args.minimum_entropy,
        },
        "failures": failures,
        "split_contract": split_audit,
        "decision_log_sha256": [sha256(path) for path in args.decision_log],
        "prior_snapshot_sha256": [sha256(path) for path in args.prior_snapshot],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, args.output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["admitted"] else 2)


if __name__ == "__main__":
    main()
