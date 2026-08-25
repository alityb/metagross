"""Fixed-world decision-utility gate for the frozen history-belief mixture."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from srcs.metagross import shadow_replay
from srcs.metagross import run_foul_play
from srcs.metagross.history_belief import candidate_fields, candidate_id, compile_public_belief
from srcs.metagross.history_belief_replay import _current_strict_weights
from srcs.metagross.known_team_belief_eval import (
    _candidate_pool_ids,
    _pristine_candidates,
    _reconstruct_view,
    load_corpus,
    truth_candidate_id,
)
from srcs.metagross.public_history import normalize_id


SCHEMA = "metagross-known-team-decision-gate/v1"
MIXTURE_ALPHA = 0.25
WORLD_COUNT = 16
WORLD_ITERATIONS = 20_000
TEACHER_ITERATIONS = 200_000


def _seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _normalized_counts(species: str, candidates: list[Any]) -> dict[str, float]:
    rows = [(candidate_id(species, candidate_fields(candidate)), candidate.pkmn_set.count) for candidate in candidates]
    total = math.fsum(count for _identity, count in rows)
    return {identity: count / total for identity, count in rows} if total > 0 else {}


def _sample_fixed_worlds(
    battle: Any,
    proposed: Mapping[str, Mapping[str, float]],
    pristine: Mapping[str, tuple[Any, ...]],
    count: int,
    seed: int,
):
    if str(shadow_replay.FOUL_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(shadow_replay.FOUL_PLAY_ROOT))
    from data.pkmn_sets import RandomBattleTeamDatasets
    from fp.search.helpers import populate_pkmn_from_set
    from fp.search.random_battles import (
        get_all_remaining_sets_for_revealed_pkmn,
        populate_randombattle_unrevealed_pkmn,
    )
    from fp.search.poke_engine_helpers import battle_to_poke_engine_state

    rng = random.Random(seed)
    source = deepcopy(battle)
    snapshot = source._metagross_random_battle_sets
    current_sets = RandomBattleTeamDatasets.pkmn_sets
    RandomBattleTeamDatasets.pkmn_sets = {name: list(rows) for name, rows in snapshot.items()}
    try:
        revealed = get_all_remaining_sets_for_revealed_pkmn(deepcopy(source), rng=rng)
        worlds = []
        for index in range(count):
            world = deepcopy(source)
            selected = {}
            members = list(world.opponent.reserve)
            if world.opponent.active is not None:
                members.append(world.opponent.active)
            for pokemon in members:
                current_candidates = revealed.get(pokemon.name, [])
                if not current_candidates or not pokemon.is_alive():
                    continue
                species = normalize_id(pokemon.name)
                current = _normalized_counts(species, list(current_candidates))
                alternative = proposed.get(species, {})
                candidate_by_id = {
                    candidate_id(species, candidate_fields(candidate)): candidate
                    for candidate in pristine.get(species, ())
                }
                support = sorted(set(current) | set(alternative))
                proposal = {
                    identity: 0.5 * current.get(identity, 0.0)
                    + 0.5 * alternative.get(identity, 0.0)
                    for identity in support
                }
                proposal_total = math.fsum(proposal.values())
                if proposal_total <= 0 or any(identity not in candidate_by_id for identity in support):
                    raise ValueError(f"coverage proposal for {species} is invalid")
                proposal = {
                    identity: probability / proposal_total
                    for identity, probability in proposal.items()
                }
                identity = rng.choices(
                    support, weights=[proposal[candidate_id_] for candidate_id_ in support]
                )[0]
                candidate = candidate_by_id[identity]
                mixture_probability = (
                    (1.0 - MIXTURE_ALPHA) * current.get(identity, 0.0)
                    + MIXTURE_ALPHA * alternative.get(identity, 0.0)
                )
                selected[species] = {
                    "candidate_id": identity,
                    "proposal_probability": proposal[identity],
                    "current_probability": current.get(identity, 0.0),
                    "mixture_probability": mixture_probability,
                }
                populate_pkmn_from_set(pokemon, candidate)
            populate_randombattle_unrevealed_pkmn(world, rng=rng)
            world.opponent.lock_moves()
            state = battle_to_poke_engine_state(world).to_string()
            worlds.append(
                {
                    "index": index,
                    "state": state,
                    "state_sha256": hashlib.sha256(state.encode("ascii")).hexdigest(),
                    "selected_candidates": selected,
                }
            )
        return worlds
    finally:
        RandomBattleTeamDatasets.pkmn_sets = current_sets


def _target_weights(worlds: list[dict], target: str) -> list[float]:
    if target not in {"current", "mixture"}:
        raise ValueError("unknown fixed-world target")
    ratios = []
    for world in worlds:
        ratio = 1.0
        for selected in world["selected_candidates"].values():
            proposal_probability = selected["proposal_probability"]
            target_probability = selected[f"{target}_probability"]
            ratio *= target_probability / proposal_probability
        ratios.append(ratio)
    total = math.fsum(ratios)
    if total <= 0:
        raise ValueError(f"{target} fixed support has zero sampled mass")
    return [ratio / total for ratio in ratios]


def _truth_battle(battle: Any, truth_sets: list[Mapping[str, Any]], pristine: Mapping[str, tuple[Any, ...]]):
    if str(shadow_replay.FOUL_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(shadow_replay.FOUL_PLAY_ROOT))
    from fp.battle import Pokemon
    from fp.search.helpers import populate_pkmn_from_set
    from data import pokedex

    def base_species(species: str) -> str:
        metadata = pokedex.get(species, {})
        return normalize_id(metadata.get("baseSpecies")) or species

    if any(normalize_id(row.get("ability")) == "illusion" for row in truth_sets):
        raise ValueError("truth injection does not support Illusion")
    result = deepcopy(battle)
    truth = {normalize_id(row.get("species") or row.get("name")): row for row in truth_sets}
    truth_by_base = {base_species(species): (species, row) for species, row in truth.items()}
    existing = list(result.opponent.reserve)
    if result.opponent.active is not None:
        existing.append(result.opponent.active)
    seen = set()
    for pokemon in existing:
        species = normalize_id(pokemon.name)
        row = truth.get(species)
        truth_species = species
        if row is None:
            matched = truth_by_base.get(base_species(species))
            if matched is not None:
                truth_species, row = matched
        if row is None:
            raise ValueError(f"revealed species {species} is absent from truth")
        truth_id = truth_candidate_id(row)[1]
        candidates = [
            candidate for candidate in pristine.get(truth_species, ())
            if candidate_id(truth_species, candidate_fields(candidate)) == truth_id
        ]
        if len(candidates) != 1:
            raise ValueError(f"truth candidate for {truth_species} is not unique")
        populate_pkmn_from_set(pokemon, candidates[0])
        seen.add(truth_species)
    for species, row in truth.items():
        if species in seen:
            continue
        pokemon = Pokemon(species, int(row.get("level", 100)))
        truth_id = truth_candidate_id(row)[1]
        candidates = [
            candidate for candidate in pristine.get(species, ())
            if candidate_id(species, candidate_fields(candidate)) == truth_id
        ]
        if len(candidates) != 1:
            raise ValueError(f"truth candidate for {species} is not unique")
        populate_pkmn_from_set(pokemon, candidates[0])
        result.opponent.reserve.append(pokemon)
    if len(result.opponent.reserve) + int(result.opponent.active is not None) != 6:
        raise ValueError("truth-injected opponent team does not contain six Pokemon")
    result.opponent.lock_moves()
    return result


def _visit_policy(result: Any, allowed: set[str]) -> dict[str, float]:
    total = float(result.total_visits)
    if total <= 0:
        raise ValueError("MCTS result has no visits")
    mass = {}
    for row in result.side_one:
        authorized = run_foul_play._authorized_action_name(row.move_choice, allowed)
        if authorized is None:
            continue
        mass[authorized] = mass.get(authorized, 0.0) + float(row.visits)
    mass_total = math.fsum(mass.values())
    if mass_total <= 0:
        raise ValueError("MCTS result has no request-authorized visits")
    return {action: visits / mass_total for action, visits in mass.items()}


def _aggregate(policies: list[Mapping[str, float]], weights: list[float]) -> dict[str, float]:
    mass = {}
    for policy, weight in zip(policies, weights, strict=True):
        for action, probability in policy.items():
            mass[action] = mass.get(action, 0.0) + weight * probability
    return dict(sorted(mass.items()))


def _argmax(policy: Mapping[str, float]) -> str:
    return sorted(policy, key=lambda action: (-policy[action], action))[0]


def _select_roots(corpus: list[dict], pristine: Mapping[str, tuple[Any, ...]]):
    roots = []
    exclusions = []
    for index, battle in enumerate(corpus):
        observer, opponent = (("p1", "p2") if index % 2 == 0 else ("p2", "p1"))
        truth_sets = battle["teams"][opponent]["sets"]
        if any(normalize_id(row.get("ability")) == "illusion" for row in truth_sets):
            exclusions.append(
                {"battle_id": battle["battle_id"], "observer": observer, "reason": "illusion_truth_unsupported"}
            )
            continue
        if any(
            truth_candidate_id(row)[1]
            not in _candidate_pool_ids(normalize_id(row.get("species") or row.get("name")), pristine)
            for row in truth_sets
        ):
            exclusions.append(
                {"battle_id": battle["battle_id"], "observer": observer, "reason": "truth_set_out_of_pool"}
            )
            continue
        states = _reconstruct_view(battle, observer)
        if not states:
            continue
        scored = []
        for decision_idx, state in enumerate(states):
            compiled = compile_public_belief(state._metagross_public_events, pristine)
            proposed = {
                belief.species: {row.candidate_id: row.weight for row in belief.candidates}
                for belief in compiled.species if belief.status == "compiled"
            }
            members = list(state.opponent.reserve)
            if state.opponent.active is not None:
                members.append(state.opponent.active)
            current_snapshot = state._metagross_random_battle_sets
            tv = 0.0
            for member in members:
                species = normalize_id(member.name)
                current = _current_strict_weights(
                    species,
                    member,
                    list(current_snapshot.get(species, pristine.get(species, ()))),
                )
                alternative = proposed.get(species, {})
                tv += 0.5 * math.fsum(
                    abs(current.get(identity, 0.0) - alternative.get(identity, 0.0))
                    for identity in set(current) | set(alternative)
                )
            scored.append((tv, decision_idx, state))
        _tv, decision_idx, state = max(scored, key=lambda row: (row[0], -row[1]))
        roots.append((battle, observer, opponent, decision_idx, state))
    return roots, exclusions


def run(corpus_path: Path, max_roots: int | None = None) -> dict[str, Any]:
    import poke_engine
    if str(shadow_replay.FOUL_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(shadow_replay.FOUL_PLAY_ROOT))
    from fp.search.poke_engine_helpers import battle_to_poke_engine_state

    if "seed" not in inspect.signature(poke_engine.monte_carlo_tree_search).parameters:
        raise RuntimeError("known-team decision gate requires seeded MCTS")
    corpus = load_corpus(corpus_path)
    pristine = _pristine_candidates()
    roots = []
    selected_roots, exclusions = _select_roots(corpus, pristine)
    if max_roots is not None:
        if max_roots < 1:
            raise ValueError("maximum root count must be positive")
        selected_roots = selected_roots[:max_roots]
    for battle, observer, opponent, decision_idx, reconstructed in selected_roots:
        compiled = compile_public_belief(reconstructed._metagross_public_events, pristine)
        proposed = {
            belief.species: {row.candidate_id: row.weight for row in belief.candidates}
            for belief in compiled.species if belief.status == "compiled"
        }
        world_seed = _seed(battle["battle_id"], observer, decision_idx, "worlds")
        worlds = _sample_fixed_worlds(
            reconstructed, proposed, pristine, WORLD_COUNT, world_seed
        )
        current_weights = _target_weights(worlds, "current")
        mixture_weights = _target_weights(worlds, "mixture")
        policies = []
        allowed = run_foul_play.request_player_actions(reconstructed)
        if not allowed:
            raise ValueError("root has no authoritative request support")
        for world in worlds:
            result = poke_engine.monte_carlo_tree_search(
                poke_engine.State.from_string(world["state"]),
                duration_ms=0,
                iterations=WORLD_ITERATIONS,
                threads=1,
                seed=_seed(battle["battle_id"], observer, decision_idx, "world", world["index"]),
            )
            policies.append(_visit_policy(result, allowed))
        current_policy = _aggregate(policies, current_weights)
        mixture_policy = _aggregate(policies, mixture_weights)
        try:
            truth = _truth_battle(
                reconstructed, battle["teams"][opponent]["sets"], pristine
            )
        except ValueError as exc:
            exclusions.append(
                {
                    "battle_id": battle["battle_id"],
                    "observer": observer,
                    "decision_idx": decision_idx,
                    "reason": str(exc),
                }
            )
            continue
        truth_state = battle_to_poke_engine_state(truth).to_string()
        teacher = _visit_policy(
            poke_engine.monte_carlo_tree_search(
                poke_engine.State.from_string(truth_state),
                duration_ms=0,
                iterations=TEACHER_ITERATIONS,
                threads=1,
                seed=_seed(battle["battle_id"], observer, decision_idx, "teacher"),
            ),
            allowed,
        )
        current_action, mixture_action = _argmax(current_policy), _argmax(mixture_policy)
        weight_tv = 0.5 * math.fsum(
            abs(current - mixture)
            for current, mixture in zip(current_weights, mixture_weights, strict=True)
        )
        policy_tv = 0.5 * math.fsum(
            abs(current_policy.get(action, 0.0) - mixture_policy.get(action, 0.0))
            for action in set(current_policy) | set(mixture_policy)
        )
        roots.append(
            {
                "battle_id": battle["battle_id"],
                "observer": observer,
                "decision_idx": decision_idx,
                "world_seed": world_seed,
                "request_action_support": sorted(allowed),
                "state_hashes": [world["state_sha256"] for world in worlds],
                "current_weights": current_weights,
                "mixture_weights": mixture_weights,
                "effective_sample_size": 1.0 / math.fsum(weight * weight for weight in mixture_weights),
                "max_mixture_weight": max(mixture_weights),
                "weight_total_variation": weight_tv,
                "policy_total_variation": policy_tv,
                "current_policy": current_policy,
                "mixture_policy": mixture_policy,
                "current_action": current_action,
                "mixture_action": mixture_action,
                "teacher_argmax": _argmax(teacher),
                "current_teacher_mass": teacher.get(current_action, 0.0),
                "mixture_teacher_mass": teacher.get(mixture_action, 0.0),
                "teacher_mass_delta": teacher.get(mixture_action, 0.0) - teacher.get(current_action, 0.0),
            }
        )
    if not roots:
        raise RuntimeError("no roots qualified for the known-team decision gate")
    deltas = [root["teacher_mass_delta"] for root in roots]
    native_path = Path(inspect.getfile(poke_engine.poke_engine)).resolve()
    return {
        "schema": SCHEMA,
        "configuration": {
            "mixture_alpha": MIXTURE_ALPHA,
            "world_count": WORLD_COUNT,
            "world_iterations": WORLD_ITERATIONS,
            "teacher_iterations": TEACHER_ITERATIONS,
            "fixed_support": True,
        },
        "input": {"path": str(corpus_path), "sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest()},
        "engine": {"native_path": str(native_path), "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest()},
        "summary": {
            "roots": len(roots),
            "excluded_roots": len(exclusions),
            "changed_actions": sum(root["current_action"] != root["mixture_action"] for root in roots),
            "mean_teacher_mass_delta": math.fsum(deltas) / len(deltas),
            "positive_roots": sum(delta > 0 for delta in deltas),
            "negative_roots": sum(delta < 0 for delta in deltas),
            "minimum_delta": min(deltas),
            "maximum_delta": max(deltas),
            "mean_effective_sample_size": math.fsum(root["effective_sample_size"] for root in roots) / len(roots),
            "mean_weight_total_variation": math.fsum(root["weight_total_variation"] for root in roots) / len(roots),
            "maximum_weight_total_variation": max(root["weight_total_variation"] for root in roots),
            "mean_policy_total_variation": math.fsum(root["policy_total_variation"] for root in roots) / len(roots),
            "maximum_policy_total_variation": max(root["policy_total_variation"] for root in roots),
            "gate_passed": math.fsum(deltas) > 0 and not any(delta <= -0.20 for delta in deltas),
        },
        "roots": roots,
        "exclusions": exclusions,
        "limitations": [
            "The teacher is seeded high-budget MCTS on the independently known true team, not game-outcome ground truth.",
            "Alternative weights are self-normalized on the current sampler's immutable finite support.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-roots", type=int)
    args = parser.parse_args()
    report = run(args.corpus, max_roots=args.max_roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
