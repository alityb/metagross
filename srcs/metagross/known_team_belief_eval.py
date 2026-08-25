"""Truth-labeled offline calibration for Randbats history beliefs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from srcs.metagross import shadow_replay
from srcs.metagross.history_belief import (
    candidate_id,
    compile_public_belief,
)
from srcs.metagross.history_belief_replay import _current_strict_weights
from srcs.metagross.public_history import normalize_id


CORPUS_SCHEMA = "metagross-known-team-battle/v1"
REPORT_SCHEMA = "metagross-known-team-belief-eval/v1"


def load_corpus(path: Path) -> list[dict[str, Any]]:
    battles = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid corpus JSON at line {line_number}") from exc
            battle_id = row.get("battle_id") if isinstance(row, dict) else None
            if row.get("schema") != CORPUS_SCHEMA or not isinstance(battle_id, str):
                raise ValueError(f"invalid corpus row at line {line_number}")
            if battle_id in seen:
                raise ValueError(f"duplicate battle id {battle_id}")
            seen.add(battle_id)
            for side in ("p1", "p2"):
                team = ((row.get("teams") or {}).get(side) or {}).get("sets")
                view = (row.get("views") or {}).get(side)
                if not isinstance(team, list) or len(team) != 6 or not isinstance(view, dict):
                    raise ValueError(f"invalid {side} truth/view at line {line_number}")
                chunks, decisions = view.get("chunks"), view.get("decisions")
                if not isinstance(chunks, list) or not isinstance(decisions, list):
                    raise ValueError(f"invalid {side} protocol at line {line_number}")
                chunk_counts = [decision.get("chunk_count") for decision in decisions]
                if (
                    any(not isinstance(value, int) or not 1 <= value <= len(chunks) for value in chunk_counts)
                    or chunk_counts != sorted(chunk_counts)
                    or len(set(chunk_counts)) != len(chunk_counts)
                    or any(not isinstance(decision.get("action"), str) for decision in decisions)
                ):
                    raise ValueError(f"invalid {side} decisions at line {line_number}")
            battles.append(row)
    if not battles:
        raise ValueError("known-team corpus is empty")
    return battles


def battle_split(battle_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(battle_id.encode("utf-8")).digest()[:8], "big") % 10
    return "train" if bucket < 6 else "development" if bucket < 8 else "test"


def truth_candidate_id(team_set: Mapping[str, Any]) -> tuple[str, str]:
    species = normalize_id(team_set.get("species") or team_set.get("name"))
    fields = {
        "level": team_set.get("level", 100),
        "ability": normalize_id(team_set.get("ability")),
        "item": normalize_id(team_set.get("item")),
        "moves": tuple(sorted(normalize_id(move) for move in team_set.get("moves", ()))),
        "tera_type": normalize_id(team_set.get("teraType")) or None,
        "count": 1,
    }
    return species, candidate_id(species, fields)


def _reconstruct_view(row: Mapping[str, Any], observer: str):
    battle_id = row["battle_id"]
    tag = f"battle-{battle_id}-{observer}"
    view = row["views"][observer]
    actions_by_chunk = {
        decision["chunk_count"]: decision["action"] for decision in view["decisions"]
    }
    protocol = []
    rqid = 0
    for chunk in view["chunks"]:
        normalized_lines = []
        for line in chunk.splitlines():
            if line.startswith("|request|"):
                request_json = json.loads(line.removeprefix("|request|"))
                request_json["rqid"] = rqid
                rqid += 1
                line = "|request|" + json.dumps(
                    request_json, separators=(",", ":")
                )
            normalized_lines.append(line)
        protocol.append(
            {"direction": "received", "message": f">{tag}\n" + "\n".join(normalized_lines)}
        )
    searches = {
        (tag, index): {"choice": actions_by_chunk[chunk_count]}
        for index, chunk_count in enumerate(sorted(actions_by_chunk))
    }
    reconstructed = shadow_replay.reconstruct_battles(
        protocol, searches, f"Truth {observer.upper()}"
    )
    return [reconstructed[(tag, index)] for index in range(len(searches))]


def _pristine_candidates() -> dict[str, tuple[Any, ...]]:
    if str(shadow_replay.FOUL_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(shadow_replay.FOUL_PLAY_ROOT))
    from data.pkmn_sets import RandomBattleTeamDatasets

    RandomBattleTeamDatasets.initialize("gen9")
    candidates = {
        species: tuple(deepcopy(candidates))
        for species, candidates in RandomBattleTeamDatasets.pkmn_sets.items()
    }
    from data import pokedex

    for species, metadata in pokedex.items():
        normalized = normalize_id(species)
        base_species = normalize_id(metadata.get("baseSpecies"))
        if normalized not in candidates and base_species in candidates:
            candidates[normalized] = candidates[base_species]
    return candidates


def _candidate_pool_ids(species: str, candidates: Mapping[str, tuple[Any, ...]]) -> set[str]:
    from srcs.metagross.history_belief import candidate_fields

    return {
        candidate_id(species, candidate_fields(candidate))
        for candidate in candidates.get(species, ())
    }


def _score(probabilities: Mapping[str, float], truth: str) -> dict[str, Any]:
    probability = float(probabilities.get(truth, 0.0))
    ranking = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    rank = next(
        (index for index, (identity, _value) in enumerate(ranking, 1) if identity == truth),
        None,
    )
    return {
        "truth_probability": probability,
        "nll": -math.log(probability) if probability > 0 else None,
        "brier": math.fsum(
            (value - (identity == truth)) ** 2
            for identity, value in probabilities.items()
        ) + (0.0 if truth in probabilities else 1.0),
        "rank": rank,
        "support": len(probabilities),
    }


def _mixture(
    current: Mapping[str, float], proposed: Mapping[str, float], alpha: float
) -> dict[str, float]:
    if not current:
        return dict(proposed)
    if not proposed:
        return dict(current)
    return {
        identity: (1.0 - alpha) * current.get(identity, 0.0)
        + alpha * proposed.get(identity, 0.0)
        for identity in set(current) | set(proposed)
    }


def evaluation_rows(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pristine = _pristine_candidates()
    rows = []
    for battle in corpus:
        split = battle_split(battle["battle_id"])
        for observer, opponent in (("p1", "p2"), ("p2", "p1")):
            truth_sets = battle["teams"][opponent]["sets"]
            truth = dict(truth_candidate_id(team_set) for team_set in truth_sets)
            truth_moves = {
                normalize_id(team_set.get("species") or team_set.get("name")): {
                    normalize_id(move) for move in team_set.get("moves", ())
                }
                for team_set in truth_sets
            }
            has_illusion = any(
                normalize_id(team_set.get("ability")) == "illusion"
                for team_set in truth_sets
            )
            reconstructed_states = _reconstruct_view(battle, observer)
            identity_ambiguous = set()
            if has_illusion and reconstructed_states:
                for event in reconstructed_states[-1]._metagross_public_events:
                    species = getattr(event, "species", None)
                    move_id = getattr(event, "move_id", None)
                    if (
                        event.actor == "opponent"
                        and move_id is not None
                        and species in truth_moves
                        and move_id not in truth_moves[species]
                    ):
                        identity_ambiguous.add(species)
            for decision_idx, reconstructed in enumerate(reconstructed_states):
                compiled = compile_public_belief(
                    reconstructed._metagross_public_events, pristine
                )
                members = list(reconstructed.opponent.reserve)
                if reconstructed.opponent.active is not None:
                    members.append(reconstructed.opponent.active)
                member_by_species = {normalize_id(member.name): member for member in members}
                current_snapshot = reconstructed._metagross_random_battle_sets
                for belief in compiled.species:
                    truth_id = truth.get(belief.species)
                    if truth_id is None:
                        continue
                    truth_in_pool = truth_id in _candidate_pool_ids(
                        belief.species, pristine
                    )
                    proposed = {
                        candidate.candidate_id: candidate.weight
                        for candidate in belief.candidates
                    }
                    member = member_by_species.get(belief.species)
                    current = _current_strict_weights(
                        belief.species,
                        member,
                        list(
                            current_snapshot.get(
                                belief.species, pristine.get(belief.species, ())
                            )
                        ),
                    ) if member is not None else {}
                    mixtures = {
                        f"mixture_{int(alpha * 100):02d}": _score(
                            _mixture(current, proposed, alpha),
                            truth_id,
                        )
                        for alpha in (0.25, 0.5, 0.75)
                    }
                    ambiguous = belief.species in identity_ambiguous
                    rows.append(
                        {
                            "battle_id": battle["battle_id"],
                            "split": split,
                            "observer": observer,
                            "decision_idx": decision_idx,
                            "species": belief.species,
                            "compiled_status": belief.status,
                            "truth_in_pristine_pool": truth_in_pool,
                            "identity_ambiguous": ambiguous,
                            "admitted": (
                                belief.status != "unsupported"
                                and truth_in_pool
                                and not ambiguous
                            ),
                            "proposed": _score(proposed, truth_id),
                            "current": _score(current, truth_id),
                            **mixtures,
                        }
                    )
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = rows
    rows = [row for row in all_rows if row.get("admitted", True)]
    result: dict[str, Any] = {
        "rows": len(all_rows),
        "admitted_rows": len(rows),
        "battles": len({row["battle_id"] for row in all_rows}),
        "unsupported_identity_rows": sum(
            row.get("compiled_status") == "unsupported"
            or row.get("identity_ambiguous", False)
            for row in all_rows
        ),
        "out_of_pool_rows": sum(
            not row.get("truth_in_pristine_pool", True) for row in all_rows
        ),
        "inconsistent_admitted_rows": sum(
            row.get("compiled_status") == "inconsistent" and row.get("admitted", True)
            for row in all_rows
        ),
    }
    for method in ("proposed", "current", "mixture_25", "mixture_50", "mixture_75"):
        scored = [row[method] for row in rows]
        finite = [score for score in scored if score["nll"] is not None]
        result[method] = {
            "coverage": len(finite) / len(scored) if scored else 0.0,
            "zero_truth_probability": len(scored) - len(finite),
            "nll": (
                math.fsum(score["nll"] for score in finite) / len(finite)
                if len(finite) == len(scored) and finite else "infinity"
                if scored else None
            ),
            "brier": math.fsum(score["brier"] for score in scored) / len(scored) if scored else None,
            "mean_truth_probability": math.fsum(score["truth_probability"] for score in scored) / len(scored) if scored else None,
            "top1": sum(score["rank"] == 1 for score in scored) / len(scored) if scored else None,
            "top3": sum(score["rank"] is not None and score["rank"] <= 3 for score in scored) / len(scored) if scored else None,
        }
    result["paired_delta_proposed_minus_current"] = {
        "brier": result["proposed"]["brier"] - result["current"]["brier"] if rows else None,
        "mean_truth_probability": result["proposed"]["mean_truth_probability"] - result["current"]["mean_truth_probability"] if rows else None,
        "top1": result["proposed"]["top1"] - result["current"]["top1"] if rows else None,
        "top3": result["proposed"]["top3"] - result["current"]["top3"] if rows else None,
    }
    for method in ("mixture_25", "mixture_50", "mixture_75"):
        result[f"paired_delta_{method}_minus_current"] = {
            metric: result[method][metric] - result["current"][metric]
            for metric in ("brier", "mean_truth_probability", "top1", "top3")
        }
    result["battle_cluster_bootstrap_ci95"] = {
        method: _cluster_bootstrap(rows, method)
        for method in ("proposed", "mixture_25", "mixture_50", "mixture_75")
    }
    return result


def _paired_values(row: Mapping[str, Any], method: str = "proposed") -> dict[str, float]:
    proposed, current = row[method], row["current"]
    return {
        "brier": proposed["brier"] - current["brier"],
        "mean_truth_probability": proposed["truth_probability"] - current["truth_probability"],
        "top1": float(proposed["rank"] == 1) - float(current["rank"] == 1),
        "top3": float(proposed["rank"] is not None and proposed["rank"] <= 3)
        - float(current["rank"] is not None and current["rank"] <= 3),
    }


def _cluster_bootstrap(
    rows: list[dict[str, Any]], method: str = "proposed", samples: int = 10_000
) -> dict[str, list[float] | None]:
    by_battle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_battle[row["battle_id"]].append(row)
    battle_ids = sorted(by_battle)
    if len(battle_ids) < 2:
        return {metric: None for metric in ("brier", "mean_truth_probability", "top1", "top3")}
    rng = random.Random(0x4D45544147524F53)
    estimates: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sampled = [rng.choice(battle_ids) for _ in battle_ids]
        values: dict[str, list[float]] = defaultdict(list)
        for battle_id in sampled:
            for row in by_battle[battle_id]:
                for metric, value in _paired_values(row, method).items():
                    values[metric].append(value)
        for metric, metric_values in values.items():
            estimates[metric].append(math.fsum(metric_values) / len(metric_values))
    result = {}
    for metric in ("brier", "mean_truth_probability", "top1", "top3"):
        ordered = sorted(estimates[metric])
        result[metric] = [
            ordered[int(0.025 * (len(ordered) - 1))],
            ordered[int(0.975 * (len(ordered) - 1))],
        ]
    return result


def evaluate(path: Path) -> dict[str, Any]:
    corpus = load_corpus(path)
    rows = evaluation_rows(corpus)
    splits = {
        split: _summary([row for row in rows if row["split"] == split])
        for split in ("train", "development", "test")
    }
    return {
        "schema": REPORT_SCHEMA,
        "corpus": {"path": str(path), "battles": len(corpus)},
        "split_battles": {
            split: sorted(row["battle_id"] for row in corpus if battle_split(row["battle_id"]) == split)
            for split in ("train", "development", "test")
        },
        "all": _summary(rows),
        "splits": splits,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.corpus)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
