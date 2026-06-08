from __future__ import annotations

import argparse
import json


ABLATIONS = [
    "no_rlm_inference",
    "no_rlm_distillation",
    "no_belief_refinement",
    "no_mcts",
    "no_phase3",
    "pokenet_18m_vs_7m",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation suite manifest")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = {"ablations": ABLATIONS, "games_per_condition": {"simple_heuristics": 200, "foul_play": 200}}
    print(json.dumps(report, indent=2) if args.json else "\n".join(ABLATIONS))


if __name__ == "__main__":
    main()
