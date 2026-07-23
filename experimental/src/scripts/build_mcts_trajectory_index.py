#!/usr/bin/env python3
"""Write stable learner-POV identities for parsed PFSP trajectories."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.mcts_policy_distillation import build_trajectory_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build_trajectory_index(args.parsed_root, args.output))


if __name__ == "__main__":
    main()
