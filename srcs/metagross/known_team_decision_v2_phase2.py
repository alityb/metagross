"""Immutable multi-alpha world bank and Phase-2 decision-utility census."""

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
from typing import Any, Mapping, Sequence

from srcs.metagross import run_foul_play, shadow_replay
from srcs.metagross.history_belief import candidate_fields, candidate_id
from srcs.metagross.known_team_belief_eval import _pristine_candidates, _reconstruct_view
from srcs.metagross.known_team_decision_v2 import (
    ENGINE_PYTHON,
    MEANINGFUL_HEADROOM,
    OBSERVED_PARTIAL_CANDIDATE,
    TEACHER_HALF_WIDTH_LIMIT,
    action_rows,
    argmax,
    canonical_json,
    load_corpus_v2,
    load_engine,
    root_beliefs,
    seed,
    sha256_bytes,
    sha256_path,
    validate_manifest,
    wilson_interval,
    write_json,
)
from srcs.metagross.public_history import normalize_id


SCHEMA = "metagross-known-team-decision-phase2/v2"
WORLD_BANK_SCHEMA = "metagross-known-team-world-bank/v2"
ALPHAS = tuple(index / 20 for index in range(21))
INITIAL_DRAWS_PER_ALPHA = 8
INITIAL_WORLD_DRAWS = len(ALPHAS) * INITIAL_DRAWS_PER_ALPHA
MAX_WORLD_DRAWS = 672
WORLD_ITERATIONS = 20_000
REPEAT_WORLD_ITERATIONS = 20_000
MINIMUM_ESS = 64.0
MAXIMUM_WEIGHT = 0.10
BENEFICIAL_DELTA = 0.02
HARMFUL_DELTA = -0.02
SEVERE_DELTA = -0.05
BOOTSTRAP_SAMPLES = 10_000
SEED_AGREEMENT_REQUIRED = 0.95


def alpha_key(alpha: float) -> str:
    return f"{round(alpha * 100):03d}"


def mixture_probability(current: float, history: float, alpha: float) -> float:
    return (1.0 - alpha) * current + alpha * history


def _candidate_distribution(
    species: str,
    current: Mapping[str, float],
    history: Mapping[str, float],
    alpha: float,
) -> tuple[list[str], dict[str, float]]:
    support = sorted(set(current) | set(history))
    proposal = {
        identity: mixture_probability(
            current.get(identity, 0.0), history.get(identity, 0.0), alpha
        )
        for identity in support
    }
    total = math.fsum(proposal.values())
    if total <= 0:
        raise ValueError(f"defensive proposal for {species} has zero mass")
    return support, {identity: probability / total for identity, probability in proposal.items()}


def _sample_world_draws(
    state: Any,
    current: Mapping[str, Mapping[str, float]],
    history: Mapping[str, Mapping[str, float]],
    pristine: Mapping[str, tuple[Any, ...]],
    count: int,
    world_seed: int,
) -> list[dict[str, Any]]:
    if str(shadow_replay.FOUL_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(shadow_replay.FOUL_PLAY_ROOT))
    from data.pkmn_sets import RandomBattleTeamDatasets
    from fp.search.helpers import populate_pkmn_from_set
    from fp.search.poke_engine_helpers import battle_to_poke_engine_state
    from fp.search.random_battles import populate_randombattle_unrevealed_pkmn

    if count % len(ALPHAS) != 0:
        raise ValueError("stratified world count must be divisible by 21 treatments")
    per_alpha = count // len(ALPHAS)
    snapshot = state._metagross_random_battle_sets
    prior_sets = RandomBattleTeamDatasets.pkmn_sets
    RandomBattleTeamDatasets.pkmn_sets = {name: list(rows) for name, rows in snapshot.items()}
    try:
        draws = []
        strata = [alpha for alpha in ALPHAS for _ in range(per_alpha)]
        for index, proposal_alpha in enumerate(strata):
            # An independent deterministic stream per stratum keeps every
            # expanded bank a prefix within each alpha treatment.
            draw_in_stratum = index % per_alpha
            rng = random.Random(seed(world_seed, alpha_key(proposal_alpha), draw_in_stratum))
            world = deepcopy(state)
            selected = {}
            members = list(world.opponent.reserve)
            if world.opponent.active is not None:
                members.append(world.opponent.active)
            for pokemon in members:
                if not pokemon.is_alive():
                    continue
                species = normalize_id(pokemon.name)
                current_species = current.get(species, {})
                history_species = history.get(species, {})
                if not current_species and not history_species:
                    continue
                support, proposal = _candidate_distribution(
                    species, current_species, history_species, proposal_alpha
                )
                candidates = {
                    candidate_id(species, candidate_fields(candidate)): candidate
                    for candidate in pristine.get(species, ())
                }
                if any(
                    identity != OBSERVED_PARTIAL_CANDIDATE and identity not in candidates
                    for identity in support
                ):
                    raise ValueError(f"world proposal for {species} is outside the frozen set pool")
                identity = rng.choices(support, weights=[proposal[row] for row in support])[0]
                if identity != OBSERVED_PARTIAL_CANDIDATE:
                    populate_pkmn_from_set(pokemon, candidates[identity])
                selected[species] = {
                    "candidate_id": identity,
                    "proposal_probability": proposal[identity],
                    "current_probability": current_species.get(identity, 0.0),
                    "history_probability": history_species.get(identity, 0.0),
                }
            populate_randombattle_unrevealed_pkmn(world, rng=rng)
            world.opponent.lock_moves()
            serialized = battle_to_poke_engine_state(world).to_string()
            draws.append(
                {
                    "draw_index": index,
                    "proposal_alpha": proposal_alpha,
                    "state": serialized,
                    "state_sha256": sha256_bytes(serialized.encode("ascii")),
                    "selected_candidates": selected,
                }
            )
        return draws
    finally:
        RandomBattleTeamDatasets.pkmn_sets = prior_sets


def unnormalized_ratio(draw: Mapping[str, Any], alpha: float) -> float:
    target_joint = 1.0
    for selected in draw["selected_candidates"].values():
        target = mixture_probability(
            selected["current_probability"], selected["history_probability"], alpha
        )
        target_joint *= target
    # The defensive proposal is a uniform mixture of the 21 *joint* alpha
    # treatments, not a product of per-species mean marginals.
    proposal_joint = math.fsum(
        math.prod(
            mixture_probability(
                selected["current_probability"],
                selected["history_probability"],
                proposal_alpha,
            )
            for selected in draw["selected_candidates"].values()
        )
        for proposal_alpha in ALPHAS
    ) / len(ALPHAS)
    if proposal_joint <= 0:
        raise ValueError("defensive joint proposal has zero mass at sampled world")
    return target_joint / proposal_joint


def normalized_weights(draws: Sequence[Mapping[str, Any]], alpha: float) -> list[float]:
    ratios = [unnormalized_ratio(draw, alpha) for draw in draws]
    total = math.fsum(ratios)
    if total <= 0:
        raise ValueError(f"alpha {alpha} has zero sampled support")
    return [ratio / total for ratio in ratios]


def weight_diagnostics(draws: Sequence[Mapping[str, Any]], alpha: float) -> dict[str, float]:
    weights = normalized_weights(draws, alpha)
    return {
        "ess": 1.0 / math.fsum(weight * weight for weight in weights),
        "maximum_weight": max(weights),
    }


def support_passes(draws: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        weight_diagnostics(draws, alpha)["ess"] >= MINIMUM_ESS
        and weight_diagnostics(draws, alpha)["maximum_weight"] <= MAXIMUM_WEIGHT
        for alpha in ALPHAS
    )


def collapse_draws(draws: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for draw in draws:
        grouped[draw["state_sha256"]].append(draw)
    return [
        {
            "state_sha256": state_hash,
            "state": group[0]["state"],
            "multiplicity": len(group),
            "draw_indices": [draw["draw_index"] for draw in group],
        }
        for state_hash, group in sorted(grouped.items())
    ]


def _search_unique_worlds(
    engine: Any,
    unique_worlds: Sequence[dict[str, Any]],
    allowed: set[str],
    identity: Sequence[object],
    iterations: int,
    repeat: int,
) -> dict[str, dict[str, Any]]:
    searches = {}
    for world in unique_worlds:
        result = engine.monte_carlo_tree_search(
            engine.State.from_string(world["state"]),
            duration_ms=0,
            iterations=iterations,
            threads=1,
            seed=seed(*identity, "phase2-world", world["state_sha256"], repeat),
        )
        searches[world["state_sha256"]] = {
            "repeat": repeat,
            "iterations": iterations,
            "actions": action_rows(result, allowed),
        }
    return searches


def build_world_bank(
    corpus_path: Path,
    manifest_path: Path,
    panel_path: Path,
    phase1_path: Path,
    *,
    max_roots: int | None = None,
    smoke: bool = False,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, corpus_path)
    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    if phase1.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise RuntimeError("Phase 1 was not produced by the frozen manifest")
    if not smoke and not phase1.get("summary", {}).get("gate_passed"):
        raise RuntimeError("Phase 1 did not authorize Phase 2")
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    corpus = {row["battle_id"]: row for row in load_corpus_v2(corpus_path)}
    selected = list(panel["roots"])
    if max_roots is not None:
        selected = selected[:max_roots]
    engine = load_engine()
    pristine = _pristine_candidates()
    roots = []
    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected = {
            "schema": WORLD_BANK_SCHEMA,
            "manifest_sha256": manifest["manifest_sha256"],
            "corpus_sha256": sha256_path(corpus_path),
            "panel_sha256": sha256_path(panel_path),
            "phase1_sha256": sha256_path(phase1_path),
            "smoke": smoke,
        }
        if all(checkpoint.get(key) == value for key, value in expected.items()):
            roots = list(checkpoint.get("roots") or [])
            expected_prefix = [
                (root["battle_id"], root["observer"], root["decision_idx"])
                for root in selected[: len(roots)]
            ]
            actual_prefix = [
                (root["battle_id"], root["observer"], root["decision_idx"])
                for root in roots
            ]
            if actual_prefix != expected_prefix:
                raise ValueError("Phase-2 checkpoint is not the frozen panel prefix")
    for root in selected[len(roots):]:
        battle = corpus[root["battle_id"]]
        state = _reconstruct_view(battle, root["observer"])[root["decision_idx"]]
        allowed = run_foul_play.request_player_actions(state)
        current, history, _tv, _affected = root_beliefs(state, pristine)
        requested = len(ALPHAS) if smoke else INITIAL_WORLD_DRAWS
        draws = _sample_world_draws(
            state,
            current,
            history,
            pristine,
            requested,
            seed(root["corpus_uid"], root["observer"], root["decision_idx"], "phase2-draws"),
        )
        while not smoke and not support_passes(draws) and len(draws) < MAX_WORLD_DRAWS:
            # Generate the next deterministic prefix by regenerating the bank at
            # its new size.  Sampling is treatment-blind and no utility has been
            # computed at this point.
            requested = min(MAX_WORLD_DRAWS, len(draws) + INITIAL_WORLD_DRAWS)
            draws = _sample_world_draws(
                state,
                current,
                history,
                pristine,
                requested,
                seed(root["corpus_uid"], root["observer"], root["decision_idx"], "phase2-draws"),
            )
        diagnostics = {alpha_key(alpha): weight_diagnostics(draws, alpha) for alpha in ALPHAS}
        supported = smoke or support_passes(draws)
        unique = collapse_draws(draws)
        searches = _search_unique_worlds(
            engine,
            unique,
            allowed,
            (root["corpus_uid"], root["observer"], root["decision_idx"]),
            256 if smoke else WORLD_ITERATIONS,
            0,
        ) if supported else {}
        repeat_searches = _search_unique_worlds(
            engine,
            unique,
            allowed,
            (root["corpus_uid"], root["observer"], root["decision_idx"]),
            256 if smoke else REPEAT_WORLD_ITERATIONS,
            1,
        ) if supported else {}
        roots.append(
            {
                **root,
                "supported": supported,
                "draw_count": len(draws),
                "unique_world_count": len(unique),
                "weight_diagnostics": diagnostics,
                "draws": draws,
                "unique_worlds": unique,
                "searches": searches,
                "repeat_searches": repeat_searches,
            }
        )
        if checkpoint_path is not None:
            write_json(
                checkpoint_path,
                {
                    "schema": WORLD_BANK_SCHEMA,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "corpus_sha256": sha256_path(corpus_path),
                    "panel_sha256": sha256_path(panel_path),
                    "phase1_sha256": sha256_path(phase1_path),
                    "smoke": smoke,
                    "complete": False,
                    "completed_roots": len(roots),
                    "total_roots": len(selected),
                    "roots": roots,
                },
            )
    return {
        "schema": WORLD_BANK_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "corpus_sha256": sha256_path(corpus_path),
        "panel_sha256": sha256_path(panel_path),
        "phase1_sha256": sha256_path(phase1_path),
        "configuration": {
            "alphas": list(ALPHAS),
            "proposal": "equal_mean_of_all_21_treatments",
            "initial_draws": INITIAL_WORLD_DRAWS,
            "maximum_draws": MAX_WORLD_DRAWS,
            "minimum_ess": MINIMUM_ESS,
            "maximum_normalized_weight": MAXIMUM_WEIGHT,
            "world_iterations": 256 if smoke else WORLD_ITERATIONS,
            "independent_repeat_iterations": 256 if smoke else REPEAT_WORLD_ITERATIONS,
            "treatment_blind_world_generation": True,
            "utility_inspected_during_adaptation": False,
        },
        "smoke": smoke,
        "complete": True,
        "roots": roots,
    }


def _draw_searches(root: Mapping[str, Any], repeat_key: str = "searches") -> list[Mapping[str, Any]]:
    searches = root[repeat_key]
    return [searches[draw["state_sha256"]]["actions"] for draw in root["draws"]]


def aggregate_visit_policy(
    draw_searches: Sequence[Mapping[str, Any]], weights: Sequence[float]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for search, weight in zip(draw_searches, weights, strict=True):
        total = math.fsum(float(row["visits"]) for row in search.values())
        for action, row in search.items():
            result[action] = result.get(action, 0.0) + weight * float(row["visits"]) / total
    return dict(sorted(result.items()))


def aggregate_advantages(
    draw_searches: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    baseline_action: str,
) -> dict[str, float]:
    support = set.intersection(*(set(search) for search in draw_searches))
    if baseline_action not in support:
        raise ValueError("B0 action is absent from a world action-value support")
    return {
        action: math.fsum(
            weight * (float(search[action]["q"]) - float(search[baseline_action]["q"]))
            for search, weight in zip(draw_searches, weights, strict=True)
        )
        for action in sorted(support)
    }


def policy_tv(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return 0.5 * math.fsum(
        abs(left.get(action, 0.0) - right.get(action, 0.0))
        for action in set(left) | set(right)
    )


def _bootstrap_mean(values: Sequence[tuple[str, float]], samples: int, channel: str) -> list[float]:
    by_battle: dict[str, list[float]] = defaultdict(list)
    for battle_id, value in values:
        by_battle[battle_id].append(value)
    battle_ids = sorted(by_battle)
    if len(battle_ids) < 2:
        return [math.nan, math.nan]
    rng = random.Random(seed("phase2-bootstrap", channel))
    estimates = []
    for _ in range(samples):
        sampled = [rng.choice(battle_ids) for _ in battle_ids]
        observations = [value for battle_id in sampled for value in by_battle[battle_id]]
        estimates.append(math.fsum(observations) / len(observations))
    estimates.sort()
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    ]


def _simultaneous_mean_bands(
    rows: Sequence[Mapping[str, Any]], selector: str, samples: int, *, overrides_only: bool = False
) -> dict[str, list[float]]:
    by_battle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_battle[row["battle_id"]].append(row)
    battle_ids = sorted(by_battle)
    observed = {}
    for key in (alpha_key(alpha) for alpha in ALPHAS[1:]):
        entries = [
            row["treatments"][key][selector]["teacher_delta"]
            for row in rows
            if not overrides_only or row["treatments"][key][selector]["changed"]
        ]
        observed[key] = math.fsum(entries) / len(entries) if entries else math.nan
    rng = random.Random(seed("phase2-simultaneous", selector, overrides_only))
    maximum_deviations = []
    for _ in range(samples):
        sampled = [rng.choice(battle_ids) for _ in battle_ids]
        bootstrap_rows = [row for battle_id in sampled for row in by_battle[battle_id]]
        deviations = []
        for key, center in observed.items():
            entries = [
                row["treatments"][key][selector]["teacher_delta"]
                for row in bootstrap_rows
                if not overrides_only or row["treatments"][key][selector]["changed"]
            ]
            if math.isnan(center) or not entries:
                continue
            estimate = math.fsum(entries) / len(entries)
            deviations.append(abs(estimate - center))
        if deviations:
            maximum_deviations.append(max(deviations))
    if not maximum_deviations:
        return {key: [math.nan, math.nan] for key in observed}
    maximum_deviations.sort()
    radius = maximum_deviations[int(0.95 * (len(maximum_deviations) - 1))]
    return {
        key: ([math.nan, math.nan] if math.isnan(center) else [center - radius, center + radius])
        for key, center in observed.items()
    }


def analyze_phase2(
    world_bank_path: Path,
    phase1_path: Path,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    smoke: bool = False,
) -> dict[str, Any]:
    bank = json.loads(world_bank_path.read_text(encoding="utf-8"))
    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    if bank.get("schema") != WORLD_BANK_SCHEMA:
        raise ValueError("invalid Phase-2 world bank")
    if not bank.get("complete"):
        raise ValueError("Phase-2 world bank is an incomplete checkpoint")
    if bank.get("phase1_sha256") != sha256_path(phase1_path):
        raise ValueError("Phase-2 world bank does not match the Phase-1 artifact")
    if not smoke and not phase1.get("summary", {}).get("gate_passed"):
        raise RuntimeError("Phase 1 did not authorize Phase 2")
    phase1_roots = {
        (root["battle_id"], root["observer"], root["decision_idx"]): root
        for root in phase1["roots"]
    }
    rows = []
    for root in bank["roots"]:
        identity = (root["battle_id"], root["observer"], root["decision_idx"])
        phase1_root = phase1_roots[identity]
        if not root["supported"]:
            rows.append({**{key: root[key] for key in ("battle_id", "observer", "decision_idx", "panel")}, "supported": False})
            continue
        searches = _draw_searches(root)
        repeat_searches = _draw_searches(root, "repeat_searches")
        current_weights = normalized_weights(root["draws"], 0.0)
        current_policy = aggregate_visit_policy(searches, current_weights)
        repeat_current_policy = aggregate_visit_policy(repeat_searches, current_weights)
        bank_current_action = argmax(current_policy)
        repeat_bank_current_action = argmax(repeat_current_policy)
        # S1 is paired against the exact frozen Phase-1 B0 action.  The alpha-0
        # bank selector is separately reported so sampling drift is visible.
        baseline_action = phase1_root["baseline_action"]
        repeat_baseline_action = baseline_action
        phase1_baseline_match = bank_current_action == baseline_action
        baseline_seed_agreement = (
            bank_current_action == repeat_bank_current_action == baseline_action
        )
        teacher = phase1_root["teacher_mean_q"]
        treatments = {}
        for alpha in ALPHAS:
            weights = normalized_weights(root["draws"], alpha)
            visit_policy = aggregate_visit_policy(searches, weights)
            advantages = aggregate_advantages(searches, weights, baseline_action)
            repeat_visit_policy = aggregate_visit_policy(repeat_searches, weights)
            repeat_advantages = aggregate_advantages(
                repeat_searches, weights, repeat_baseline_action
            )
            visit_action = argmax(visit_policy)
            advantage_action = argmax(advantages)
            repeat_visit_action = argmax(repeat_visit_policy)
            repeat_advantage_action = argmax(repeat_advantages)
            diagnostic = weight_diagnostics(root["draws"], alpha)
            treatments[alpha_key(alpha)] = {
                "alpha": alpha,
                "weight_diagnostics": diagnostic,
                "weight_total_variation_from_current": 0.5 * math.fsum(
                    abs(left - right) for left, right in zip(current_weights, weights, strict=True)
                ),
                "visit": {
                    "action": visit_action,
                    "policy": visit_policy,
                    "policy_total_variation_from_current": policy_tv(current_policy, visit_policy),
                    "changed": visit_action != baseline_action,
                    "repeat_action": repeat_visit_action,
                    "repeat_changed": repeat_visit_action != repeat_baseline_action,
                    "seed_agreement": visit_action == repeat_visit_action,
                    "crossing_verified": (
                        visit_action != baseline_action
                        and baseline_seed_agreement
                        and visit_action == repeat_visit_action
                    ),
                    "teacher_delta": teacher[visit_action] - teacher[baseline_action],
                },
                "advantage": {
                    "action": advantage_action,
                    "advantages": advantages,
                    "margin": advantages[advantage_action] - sorted(advantages.values(), reverse=True)[1] if len(advantages) > 1 else math.inf,
                    "changed": advantage_action != baseline_action,
                    "repeat_action": repeat_advantage_action,
                    "repeat_changed": repeat_advantage_action != repeat_baseline_action,
                    "seed_agreement": advantage_action == repeat_advantage_action,
                    "crossing_verified": (
                        advantage_action != baseline_action
                        and baseline_seed_agreement
                        and advantage_action == repeat_advantage_action
                    ),
                    "teacher_delta": teacher[advantage_action] - teacher[baseline_action],
                },
            }
        oracle_candidates = []
        for key, treatment in treatments.items():
            for selector in ("visit", "advantage"):
                oracle_candidates.append(
                    (
                        treatment[selector]["teacher_delta"],
                        -treatment["alpha"],
                        selector,
                        key,
                        treatment[selector]["action"],
                    )
                )
        oracle_delta, _negative_alpha, oracle_selector, oracle_key, oracle_action = max(oracle_candidates)
        rows.append(
            {
                **{key: root[key] for key in (
                    "battle_id", "corpus_uid", "observer", "decision_idx", "panel",
                    "turn", "belief_total_variation", "affected_living_species", "legal_actions", "tera_available",
                )},
                "supported": True,
                "baseline_action": baseline_action,
                "phase1_baseline_action": phase1_root["baseline_action"],
                "phase1_baseline_match": phase1_baseline_match,
                "baseline_policy": current_policy,
                "bank_current_action": bank_current_action,
                "repeat_bank_current_action": repeat_bank_current_action,
                "repeat_baseline_action": repeat_baseline_action,
                "baseline_seed_agreement": baseline_seed_agreement,
                "baseline_headroom": phase1_root["baseline_headroom"],
                "treatments": treatments,
                "oracle": {
                    "selector": oracle_selector,
                    "alpha_key": oracle_key,
                    "alpha": treatments[oracle_key]["alpha"],
                    "action": oracle_action,
                    "teacher_delta": oracle_delta,
                    "beneficial": oracle_delta >= BENEFICIAL_DELTA,
                },
            }
        )
    supported = [row for row in rows if row.get("supported")]
    representative = [row for row in supported if row["panel"] == "representative"]
    sensitivity = [row for row in supported if row["panel"] == "sensitivity"]
    bootstrap_count = min(bootstrap_samples, 250) if smoke else bootstrap_samples
    selector_summaries = {}
    simultaneous = {}
    for selector in ("visit", "advantage"):
        simultaneous[selector] = {
            "all_roots": _simultaneous_mean_bands(
                representative, selector, bootstrap_count
            ) if len(representative) >= 2 else {},
            "overrides": _simultaneous_mean_bands(
                representative, selector, bootstrap_count, overrides_only=True
            ) if len(representative) >= 2 else {},
        }
        per_alpha = {}
        for alpha in ALPHAS:
            key = alpha_key(alpha)
            entries = [row["treatments"][key][selector] for row in representative]
            changed = [entry for entry in entries if entry["changed"]]
            verified = [entry for entry in changed if entry["crossing_verified"]]
            deltas = [(row["battle_id"], row["treatments"][key][selector]["teacher_delta"]) for row in representative]
            per_alpha[key] = {
                "alpha": alpha,
                "crossings": len(changed),
                "beneficial_crossings": sum(entry["teacher_delta"] >= BENEFICIAL_DELTA for entry in verified),
                "harmful_crossings": sum(entry["teacher_delta"] <= HARMFUL_DELTA for entry in verified),
                "severe_regressions": sum(entry["teacher_delta"] <= SEVERE_DELTA for entry in changed),
                "unverified_crossings": len(changed) - len(verified),
                "seed_agreement": (
                    sum(entry["seed_agreement"] for entry in entries) / len(entries)
                    if entries else 0.0
                ),
                "verified_crossings": sum(entry["crossing_verified"] for entry in changed),
                "mean_baseline_headroom_captured": (
                    math.fsum(
                        min(1.0, max(0.0, entry["teacher_delta"] / row["baseline_headroom"]))
                        if row["baseline_headroom"] > 0 else 0.0
                        for row, entry in zip(representative, entries, strict=True)
                    ) / len(entries) if entries else 0.0
                ),
                "mean_all_root_delta": math.fsum(entry["teacher_delta"] for entry in entries) / len(entries) if entries else 0.0,
                "mean_all_root_delta_ci95": _bootstrap_mean(deltas, bootstrap_count, f"{selector}-{key}-all") if len(entries) >= 2 else [math.nan, math.nan],
                "mean_override_delta": math.fsum(entry["teacher_delta"] for entry in changed) / len(changed) if changed else 0.0,
                "mean_override_delta_ci95": _bootstrap_mean(
                    [(row["battle_id"], row["treatments"][key][selector]["teacher_delta"]) for row in representative if row["treatments"][key][selector]["changed"]],
                    bootstrap_count,
                    f"{selector}-{key}-override",
                ) if len(changed) >= 2 else [math.nan, math.nan],
            }
        selector_summaries[selector] = per_alpha
    oracle_beneficial = sum(row["oracle"]["beneficial"] for row in sensitivity)
    oracle_interval = wilson_interval(oracle_beneficial, len(sensitivity))
    oracle_mean_values = [(row["battle_id"], row["oracle"]["teacher_delta"]) for row in sensitivity]
    oracle_mean = math.fsum(value for _battle, value in oracle_mean_values) / len(oracle_mean_values) if oracle_mean_values else 0.0
    oracle_mean_ci = _bootstrap_mean(oracle_mean_values, bootstrap_count, "oracle-sensitivity") if len(oracle_mean_values) >= 2 else [math.nan, math.nan]
    oracle_passed = (
        oracle_interval[0] >= 0.05
        and not math.isnan(oracle_mean_ci[0])
        and oracle_mean_ci[0] > 0.002
    )

    def fixed_selector_passes(selector: str, key: str) -> bool:
        summary = selector_summaries[selector][key]
        all_lower = simultaneous[selector]["all_roots"].get(key, [math.nan])[0]
        override_lower = simultaneous[selector]["overrides"].get(key, [math.nan])[0]
        return (
            summary["beneficial_crossings"] >= 15
            and summary["severe_regressions"] == 0
            and summary["harmful_crossings"] <= summary["beneficial_crossings"]
            and not math.isnan(override_lower)
            and override_lower > 0.01
            and not math.isnan(all_lower)
            and all_lower > 0.002
            and summary["seed_agreement"] >= SEED_AGREEMENT_REQUIRED
            and summary["unverified_crossings"] == 0
        )

    visit_passes = [key for key in selector_summaries["visit"] if fixed_selector_passes("visit", key)]
    advantage_passes = [key for key in selector_summaries["advantage"] if fixed_selector_passes("advantage", key)]
    missed_stable_benefits = sum(
        any(
            row["treatments"][key]["advantage"]["teacher_delta"] >= BENEFICIAL_DELTA
            and row["treatments"][key]["advantage"]["crossing_verified"]
            and not row["treatments"][key]["visit"]["changed"]
            and not row["treatments"][key]["visit"]["repeat_changed"]
            for key in row["treatments"]
        )
        for row in representative
    )
    if not oracle_passed:
        decision = "end_belief_research"
        phase3_authorized = False
    elif visit_passes:
        decision = "freeze_s0_candidate"
        phase3_authorized = False
    elif (advantage_passes or oracle_passed) and missed_stable_benefits >= 10:
        decision = "authorize_phase3_maple_vertical_slice"
        phase3_authorized = True
    else:
        decision = "end_belief_research"
        phase3_authorized = False
    if smoke:
        decision = "smoke_only_no_authorization"
        phase3_authorized = False
    return {
        "schema": SCHEMA,
        "world_bank_sha256": sha256_path(world_bank_path),
        "phase1_sha256": sha256_path(phase1_path),
        "smoke": smoke,
        "summary": {
            "roots": len(rows),
            "supported_roots": len(supported),
            "representative_supported_roots": len(representative),
            "sensitivity_supported_roots": len(sensitivity),
            "root_coverage": len(supported) / len(rows) if rows else 0.0,
            "baseline_seed_agreement": (
                sum(row["baseline_seed_agreement"] for row in supported) / len(supported)
                if supported else 0.0
            ),
            "oracle_beneficial_sensitivity_roots": oracle_beneficial,
            "oracle_beneficial_frequency_wilson95": list(oracle_interval),
            "oracle_mean_gain": oracle_mean,
            "oracle_mean_gain_ci95": oracle_mean_ci,
            "oracle_passed": oracle_passed,
            "visit_passing_alphas": visit_passes,
            "advantage_passing_alphas": advantage_passes,
            "s1_benefits_missed_by_s0": missed_stable_benefits,
            "phase3_authorized": phase3_authorized,
            "decision": decision,
        },
        "selector_summaries": selector_summaries,
        "simultaneous_mean_bands": simultaneous,
        "roots": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bank = subparsers.add_parser("bank")
    bank.add_argument("--corpus", type=Path, required=True)
    bank.add_argument("--manifest", type=Path, required=True)
    bank.add_argument("--panel", type=Path, required=True)
    bank.add_argument("--phase1", type=Path, required=True)
    bank.add_argument("--output", type=Path, required=True)
    bank.add_argument("--max-roots", type=int)
    bank.add_argument("--smoke", action="store_true")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--world-bank", type=Path, required=True)
    analyze.add_argument("--phase1", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    analyze.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "bank":
        write_json(
            args.output,
            build_world_bank(
                args.corpus,
                args.manifest,
                args.panel,
                args.phase1,
                max_roots=args.max_roots,
                smoke=args.smoke,
                checkpoint_path=args.output,
            ),
        )
    else:
        write_json(
            args.output,
            analyze_phase2(
                args.world_bank,
                args.phase1,
                bootstrap_samples=args.bootstrap_samples,
                smoke=args.smoke,
            ),
        )


if __name__ == "__main__":
    main()
