from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


def _posterior_from_truth(true_opponent_team: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    posterior: dict[str, list[dict[str, Any]]] = {}
    for species, info in (true_opponent_team or {}).items():
        if not isinstance(info, dict):
            continue
        posterior[species] = [
            {
                "set_index": 0,
                "probability": 1.0,
                "moves": info.get("moves", []),
                "item": info.get("item"),
                "ability": info.get("ability"),
            }
        ]
    return posterior


def annotate_pickle(path: Path, output_dir: Path) -> Path:
    samples = pickle.load(path.open("rb"))
    turns = []
    for sample in samples:
        turns.append(
            {
                "turn": int(sample.turn),
                "v_rlm": max(-1.0, min(1.0, float(sample.outcome) * 0.25)),
                "belief_posterior": _posterior_from_truth(sample.true_opponent_team),
                "human_action": int(sample.human_action),
                "outcome": float(sample.outcome),
            }
        )
    annotation = {"battle_id": path.stem, "turns": turns}
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}.json"
    output_path.write_text(json.dumps(annotation, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate parsed replay samples for Phase 1 smoke/full IL")
    parser.add_argument("--replay-dir", default="data/parsed")
    parser.add_argument("--output-dir", default="data/annotations")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--rlm-mode", choices=["heuristic", "anthropic", "local"], default="heuristic")
    args = parser.parse_args()
    paths = sorted(Path(args.replay_dir).glob("*.pkl"))
    if args.n is not None:
        paths = paths[: args.n]
    written = [str(annotate_pickle(path, Path(args.output_dir))) for path in paths]
    print(json.dumps({"annotated": len(written), "outputs": written[:10], "rlm_mode": args.rlm_mode}, indent=2))


if __name__ == "__main__":
    main()
