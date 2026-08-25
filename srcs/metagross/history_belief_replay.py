"""Offline comparison of compiled public-history and current Foul Play beliefs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from srcs.metagross import shadow_replay
from srcs.metagross.history_belief import (
    candidate_fields,
    candidate_id,
    compile_public_belief,
)
from srcs.metagross.public_history import normalize_id


def _opponent_pokemon(battle: Any) -> dict[str, Any]:
    pokemon = list(battle.opponent.reserve)
    if battle.opponent.active is not None:
        pokemon.append(battle.opponent.active)
    result = {}
    for member in pokemon:
        species = normalize_id(member.name)
        if species in result:
            raise ValueError(f"ambiguous reconstructed opponent species {species}")
        result[species] = member
    return result


def _current_strict_weights(species: str, pokemon: Any, candidates: list[Any]) -> dict[str, float]:
    matches = []
    for candidate in candidates:
        if candidate.full_set_pkmn_can_have_set(
            pokemon,
            match_ability=True,
            match_item=True,
            speed_check=False,
            tera_check=True,
        ):
            fields = candidate_fields(candidate)
            matches.append((candidate_id(species, fields), fields["count"]))
    total = math.fsum(count for _identity, count in matches)
    return {
        identity: count / total
        for identity, count in matches
    } if total > 0 else {}


def _current_derived_constraints(pokemon: Any) -> dict[str, Any]:
    speed_range = getattr(pokemon, "speed_range", None)
    speed_min = getattr(speed_range, "min", None)
    speed_max = getattr(speed_range, "max", None)
    return {
        "removed_item": normalize_id(getattr(pokemon, "removed_item", None)) or None,
        "impossible_items": sorted(
            normalize_id(value) for value in getattr(pokemon, "impossible_items", ())
        ),
        "impossible_abilities": sorted(
            normalize_id(value) for value in getattr(pokemon, "impossible_abilities", ())
        ),
        "can_have_choice_item": bool(
            getattr(pokemon, "can_have_choice_item", True)
        ),
        "speed_range": {
            "min": speed_min if isinstance(speed_min, (int, float)) and math.isfinite(speed_min) else None,
            "max": speed_max if isinstance(speed_max, (int, float)) and math.isfinite(speed_max) else None,
        },
    }


def compare_reconstructed_beliefs(battles: dict[tuple[str, int], Any]) -> dict[str, Any]:
    rows = []
    for (tag, decision_idx), battle in sorted(battles.items()):
        events = getattr(battle, "_metagross_public_events", None)
        snapshot = getattr(battle, "_metagross_random_battle_sets", None)
        if not isinstance(events, tuple) or snapshot is None:
            raise ValueError("reconstructed battle lacks frozen history inputs")
        compiled = compile_public_belief(events, snapshot)
        opponent = _opponent_pokemon(battle)
        for belief in compiled.species:
            member = opponent.get(belief.species)
            current = _current_strict_weights(
                belief.species,
                member,
                list(snapshot.get(belief.species, ())),
            ) if member is not None else {}
            proposed = {candidate.candidate_id: candidate.weight for candidate in belief.candidates}
            union = set(current) | set(proposed)
            total_variation = 0.5 * math.fsum(
                abs(current.get(identity, 0.0) - proposed.get(identity, 0.0))
                for identity in union
            )
            rows.append(
                {
                    "tag": tag,
                    "decision_idx": decision_idx,
                    "species": belief.species,
                    "compiled_status": belief.status,
                    "compiled_reason": belief.reason,
                    "compiled_candidates": len(proposed),
                    "current_strict_candidates": len(current),
                    "support_equal": set(proposed) == set(current),
                    "total_variation": total_variation,
                    "current_derived_constraints": (
                        _current_derived_constraints(member) if member is not None else None
                    ),
                }
            )
    compared = [row for row in rows if row["compiled_status"] == "compiled"]
    return {
        "schema": "metagross-history-belief-replay/v1",
        "decisions": len(battles),
        "rows": len(rows),
        "compiled_rows": len(compared),
        "unsupported_rows": sum(row["compiled_status"] == "unsupported" for row in rows),
        "inconsistent_rows": sum(row["compiled_status"] == "inconsistent" for row in rows),
        "support_equal_rows": sum(row["support_equal"] for row in compared),
        "mean_total_variation": (
            math.fsum(row["total_variation"] for row in compared) / len(compared)
            if compared else None
        ),
        "max_total_variation": max(
            (row["total_variation"] for row in compared), default=None
        ),
        "comparisons": rows,
    }


def compare_capture(capture: Path) -> dict[str, Any]:
    protocol, searches, metadata = shadow_replay.load_capture(capture)
    battles = shadow_replay.reconstruct_battles(
        protocol, searches, metadata["manifest"]["ladder"]["username"]
    )
    report = compare_reconstructed_beliefs(battles)
    report["capture_digest"] = metadata["capture_digest"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path, nargs="?", default=shadow_replay.DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_capture(args.capture)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
