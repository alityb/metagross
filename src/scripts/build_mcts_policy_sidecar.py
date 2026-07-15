#!/usr/bin/env python3
"""Build fail-closed MCTS policy targets aligned to parsed learner trajectories."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.mcts_policy_distillation import build_sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-root", required=True, type=Path)
    parser.add_argument("--decision-log", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--trajectory-index",
        type=Path,
        default=None,
        help="Parser-produced trajectory identity manifest; required for schema-v2 decisions.",
    )
    args = parser.parse_args()
    result = build_sidecar(args.decision_log, args.parsed_root, args.output, args.trajectory_index)
    print(result)
    if result["accepted"] == 0:
        raise SystemExit("no verified targets written; inspect rejected records")


if __name__ == "__main__":
    main()
