from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.state import Vocabulary, build_vocabulary, encode_state, generation_from_format, normalize_name
from pipeline.sample_types import TrainingSample


def _log_from_replay(raw: dict[str, Any]) -> str:
    log = raw.get("log") or raw.get("battle_log") or raw.get("logs") or ""
    if isinstance(log, list):
        return "\n".join(map(str, log))
    return str(log)


def _species_from_details(details: str) -> str:
    return details.split(",", 1)[0].strip()


def _slot(actor: str) -> str:
    return actor.split(":", 1)[0].strip()


def _player(actor: str) -> str:
    return actor[:2]


def _action_has_choice(parts: list[str]) -> bool:
    return not any(part.strip().startswith("[from]") for part in parts[4:])


def _team_entry(species: str, moves: set[str], item: str | None, ability: str | None) -> dict[str, Any]:
    return {"species": species, "moves": sorted(moves), "item": item, "ability": ability}


def parse_replay_file(path: str | Path, pool_path: str = "data/gen9_random_pool.json", format_hint: str | None = None) -> tuple[str, list[TrainingSample], str | None]:
    replay_path = Path(path)
    try:
        raw = json.loads(replay_path.read_text(encoding="utf-8"))
        battle_id = str(raw.get("id") or raw.get("battle_id") or replay_path.stem)
        log = _log_from_replay(raw)
        format_id = format_hint or raw.get("formatid") or raw.get("format") or "gen9randombattle"
        generation = generation_from_format(format_id, 9)
        vocab = build_vocabulary(pool_path)
        samples = parse_log(battle_id, log, vocab, generation=generation)
        return battle_id, samples, None
    except Exception as exc:  # noqa: BLE001 - parser logs and continues.
        return replay_path.stem, [], str(exc)


def parse_log(battle_id: str, log: str, vocab: Vocabulary, generation: int = 9) -> list[TrainingSample]:
    lines = [line for line in log.splitlines() if line]
    current_turn = 0
    max_turn = 0
    player_names: dict[str, str] = {}
    winner: str | None = None
    team_species: dict[str, list[str]] = {"p1": [], "p2": []}
    active: dict[str, str | None] = {"p1": None, "p2": None}
    hp: dict[str, float] = defaultdict(lambda: 1.0)
    seen_moves: dict[str, set[str]] = defaultdict(set)
    items: dict[str, str | None] = defaultdict(lambda: None)
    abilities: dict[str, str | None] = defaultdict(lambda: None)
    samples: list[TrainingSample] = []

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        event = parts[1]
        if event == "player" and len(parts) >= 4:
            player_names[parts[2]] = parts[3]
        elif event == "poke" and len(parts) >= 4 and parts[2] in team_species:
            species = _species_from_details(parts[3])
            if species and species not in team_species[parts[2]]:
                team_species[parts[2]].append(species)
        elif event in {"switch", "drag"} and len(parts) >= 4:
            side = _player(parts[2])
            species = _species_from_details(parts[3])
            active[side] = species
            if species not in team_species.get(side, []):
                team_species.setdefault(side, []).append(species)
        elif event == "turn" and len(parts) >= 3:
            try:
                current_turn = int(parts[2])
                max_turn = max(max_turn, current_turn)
            except ValueError:
                pass
        elif event == "win" and len(parts) >= 3:
            winner = parts[2].strip()
        elif event in {"-item", "-enditem"} and len(parts) >= 4:
            side = _player(parts[2])
            species = active.get(side)
            if species:
                items[f"{side}:{species}"] = parts[3]
        elif event == "-ability" and len(parts) >= 4:
            side = _player(parts[2])
            species = active.get(side)
            if species:
                abilities[f"{side}:{species}"] = parts[3]
        elif event == "move" and len(parts) >= 4:
            side = _player(parts[2])
            species = active.get(side)
            move = parts[3].strip()
            if species:
                seen_moves[f"{side}:{species}"].add(move)
            if side != "p1" or not species or not _action_has_choice(parts):
                continue
            available = sorted(seen_moves[f"p1:{species}"] | {move})[:4]
            if move not in available:
                available = [move] + available[:3]
            if len(available) < 2:
                continue
            action = available.index(move)
            own_team = _team_state("p1", team_species, active, seen_moves, items, abilities, hp)
            opp_team = _team_state("p2", team_species, active, seen_moves, items, abilities, hp)
            state = encode_state(
                {
                    "turn": current_turn,
                    "own_team": own_team,
                    "opponent_team": opp_team,
                    "available_moves": [{"move": mv, "disabled": False} for mv in available],
                    "available_switches": [],
                    "can_tera": False,
                },
                vocab=vocab,
                generation=generation,
            )
            true_opp = {
                species_name: _team_entry(
                    species_name,
                    seen_moves.get(f"p2:{species_name}", set()),
                    items.get(f"p2:{species_name}"),
                    abilities.get(f"p2:{species_name}"),
                )
                for species_name in team_species.get("p2", [])
            }
            samples.append(TrainingSample(battle_id, current_turn, state, action, 0.0, true_opp, generation=generation))
    if max_turn < 10:
        return []
    outcome = 1.0 if winner and winner == player_names.get("p1") else -1.0
    for sample in samples:
        sample.outcome = outcome
    return samples


def _team_state(
    side: str,
    team_species: dict[str, list[str]],
    active: dict[str, str | None],
    seen_moves: dict[str, set[str]],
    items: dict[str, str | None],
    abilities: dict[str, str | None],
    hp: dict[str, float],
) -> list[dict[str, Any]]:
    team = []
    for species in team_species.get(side, [])[:6]:
        team.append(
            {
                "species": species,
                "moves": sorted(seen_moves.get(f"{side}:{species}", set()))[:4],
                "item": items.get(f"{side}:{species}"),
                "ability": abilities.get(f"{side}:{species}"),
                "hp_fraction": hp.get(f"{side}:{species}", 1.0),
                "is_active": active.get(side) == species,
            }
        )
    return team


def process_one(args: tuple[Path, Path, str, str | None]) -> tuple[str, int, str | None]:
    raw_path, output_dir, pool_path, format_hint = args
    battle_id, samples, error = parse_replay_file(raw_path, pool_path, format_hint)
    if error is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"{battle_id}.pkl").open("wb") as handle:
            pickle.dump(samples, handle)
    return battle_id, len(samples), error


def parse_many(raw_dir: Path, output_dir: Path, pool_path: str, n: int | None, workers: int, format_hint: str | None = None) -> list[tuple[str, int, str | None]]:
    paths = sorted(raw_dir.glob("*.json"))
    if n is not None:
        paths = paths[:n]
    tasks = [(path, output_dir, pool_path, format_hint) for path in paths]
    if workers <= 1:
        return [process_one(task) for task in tasks]
    with mp.Pool(workers) as pool:
        return list(pool.imap_unordered(process_one, tasks))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PS replays into TrainingSample pickle files")
    parser.add_argument("--raw-dir", default="data/raw_replays")
    parser.add_argument("--output", default="data/parsed")
    parser.add_argument("--pool", default="data/gen9_random_pool.json")
    parser.add_argument("--format", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    n = args.n if args.n is not None else (100 if args.smoke else None)
    results = parse_many(Path(args.raw_dir), Path(args.output), args.pool, n, args.workers, args.format)
    counts = [count for _battle_id, count, error in results if error is None]
    errors = [(battle_id, error) for battle_id, _count, error in results if error is not None]
    parsed_files = sorted(Path(args.output).glob("*.pkl"))[: max(1, min(100, len(results)))]
    action_counts: Counter[int] = Counter()
    outcomes: Counter[float] = Counter()
    for path in parsed_files:
        try:
            samples = pickle.load(path.open("rb"))
        except Exception:
            continue
        for sample in samples:
            action_counts[sample.human_action] += 1
            outcomes[sample.outcome] += 1
    report = {
        "replays": len(results),
        "parsed": sum(1 for _id, _count, error in results if error is None),
        "errors": len(errors),
        "samples_total": sum(counts),
        "samples_per_replay_avg": (sum(counts) / len(counts)) if counts else 0.0,
        "samples_per_replay_min": min(counts) if counts else 0,
        "samples_per_replay_max": max(counts) if counts else 0,
        "action_distribution": dict(sorted(action_counts.items())),
        "outcome_balance": dict(outcomes),
        "first_errors": errors[:10],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
