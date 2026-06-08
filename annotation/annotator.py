from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from belief.constraints import filter_by_seen_moves, normalize_name, uniform_posterior
from rlm.repl_env import RLMConfig, normalize_policy
from rlm.strategist import RLMStrategist

from .tools import iter_replay_paths


PHASE0_CONFIG = RLMConfig(max_iterations=20, max_sub_queries=30, truncate_len=8000, time_budget_ms=None)


@dataclass
class ReplayEventIndex:
    turns: dict[int, int]
    winner: str | None
    p2_species_by_slot: dict[str, str]
    p2_moves_by_slot: dict[str, set[str]]


def load_pool(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    pool_path = Path(path)
    if not pool_path.exists():
        return {}
    with pool_path.open() as handle:
        return json.load(handle)


def line_offset_by_turn(log: str) -> dict[int, int]:
    turns = {0: 0}
    for match in re.finditer(r"(?:^|\n)\|turn\|(\d+)", log):
        turns[int(match.group(1))] = match.start()
    return dict(sorted(turns.items()))


def parse_replay_index(log: str) -> ReplayEventIndex:
    winner: str | None = None
    p2_species_by_slot: dict[str, str] = {}
    p2_moves_by_slot: dict[str, set[str]] = {}
    for line in log.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        event = parts[1]
        if event == "win" and len(parts) >= 3:
            winner = parts[2].strip()
        if event in {"switch", "drag"} and len(parts) >= 4 and parts[2].startswith("p2"):
            slot = parts[2].split(":", 1)[0].strip()
            species = parts[3].split(",", 1)[0].strip()
            p2_species_by_slot[slot] = species
        if event == "move" and len(parts) >= 4 and parts[2].startswith("p2"):
            slot = parts[2].split(":", 1)[0].strip()
            p2_moves_by_slot.setdefault(slot, set()).add(parts[3].strip())
    return ReplayEventIndex(
        turns=line_offset_by_turn(log),
        winner=winner,
        p2_species_by_slot=p2_species_by_slot,
        p2_moves_by_slot=p2_moves_by_slot,
    )


def pool_lookup(pool: dict[str, list[dict[str, Any]]], species: str) -> list[dict[str, Any]]:
    if species in pool:
        return list(pool[species])
    normalized = normalize_name(species)
    for key, value in pool.items():
        if normalize_name(key) == normalized:
            return list(value)
    return []


def posterior_for_slot(pool: dict[str, list[dict[str, Any]]], species: str, moves: set[str]) -> list[dict[str, Any]]:
    candidates = pool_lookup(pool, species)
    filtered = filter_by_seen_moves(candidates, moves)
    return uniform_posterior(filtered or candidates)


def heuristic_value(turn: int, winner: str | None) -> float:
    if winner is None:
        return 0.0
    # Offline annotations are from the p1 perspective by convention.
    sign = 1.0 if winner else 0.0
    confidence = min(0.9, 0.2 + 0.02 * max(0, turn))
    return sign * confidence


def annotate_replay(
    *,
    replay_id: str,
    log: str,
    pool: dict[str, list[dict[str, Any]]],
    strategist: RLMStrategist,
    decision_stride: int = 1,
) -> dict[str, Any]:
    index = parse_replay_index(log)
    decisions: dict[str, Any] = {}
    turns = [turn for turn in index.turns if turn > 0 and turn % max(1, decision_stride) == 0]
    if not turns:
        turns = [0]
    for turn in turns:
        prefix_end = index.turns.get(turn + 1, len(log))
        prefix = log[:prefix_end]
        state = {
            "turn": turn,
            "winner": index.winner,
            "revealed_opponent": index.p2_species_by_slot,
            "seen_opponent_moves": {slot: sorted(moves) for slot, moves in index.p2_moves_by_slot.items()},
        }
        posterior = {
            slot: posterior_for_slot(pool, species, index.p2_moves_by_slot.get(slot, set()))
            for slot, species in index.p2_species_by_slot.items()
        }
        rlm_out = strategist.assess(log=prefix, state=state, pool=pool, base_policy=[1.0 / 14.0] * 14)
        refined = rlm_out.refined_belief or posterior
        decisions[str(turn)] = {
            "posterior": refined,
            "V_rlm": rlm_out.v_rlm if rlm_out.v_rlm != 0.0 else heuristic_value(turn, index.winner),
            "speed_constraints": {},
            "eliminated_by": ["move_seen"] if any(index.p2_moves_by_slot.values()) else [],
            "policy": normalize_policy(rlm_out.pi_rlm),
            "state": state,
            "rlm_trace": {
                "iterations": rlm_out.iterations,
                "sub_queries": rlm_out.sub_queries,
                "elapsed_ms": rlm_out.elapsed_ms,
                "observations": rlm_out.observations[-3:],
            },
        }
    return {"replay_id": replay_id, "winner": index.winner, "decisions": decisions}


def annotate_file(
    replay_path: Path,
    output_dir: Path,
    pool: dict[str, list[dict[str, Any]]],
    strategist: RLMStrategist,
    decision_stride: int,
) -> Path:
    if replay_path.suffix.lower() == ".json":
        raw = json.loads(replay_path.read_text())
        log = raw.get("log") or raw.get("battle_log") or "\n".join(raw.get("logs", []))
        replay_id = raw.get("replay_id") or replay_path.stem
    else:
        log = replay_path.read_text(errors="replace")
        replay_id = replay_path.stem
    annotation = annotate_replay(
        replay_id=replay_id,
        log=log,
        pool=pool,
        strategist=strategist,
        decision_stride=decision_stride,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{replay_id}.json"
    output_path.write_text(json.dumps(annotation, indent=2, sort_keys=True))
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 0 RLM-powered replay annotation")
    parser.add_argument("--replays", default="data/replays", help="Replay file or directory")
    parser.add_argument("--output", default="data/annotations", help="Annotation output directory")
    parser.add_argument("--pool", default="data/gen9_random_pool.json", help="Gen9 random pool JSON")
    parser.add_argument("--provider", choices=["heuristic", "anthropic", "local"], default="heuristic")
    parser.add_argument("--model", default=None, help="Provider model name")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decision-stride", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    pool = load_pool(args.pool)
    strategist = RLMStrategist.from_provider(args.provider, model=args.model, config=PHASE0_CONFIG)
    replay_paths = iter_replay_paths(args.replays)
    if args.limit is not None:
        replay_paths = replay_paths[: args.limit]
    output_dir = Path(args.output)
    written: list[str] = []
    for replay_path in replay_paths:
        output_path = annotate_file(replay_path, output_dir, pool, strategist, args.decision_stride)
        written.append(str(output_path))
    print(json.dumps({"annotated": len(written), "outputs": written[:10]}, indent=2))


if __name__ == "__main__":
    main()
