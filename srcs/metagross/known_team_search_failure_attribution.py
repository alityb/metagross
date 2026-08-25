#!/usr/bin/env python3
"""Frozen budget/seed/selector attribution for Phase-1 headroom roots.

The experiment deliberately does *not* train or deploy a learned evaluator.  It
uses the immutable Phase-2 world bank to ask which of the known-team misses can
already be explained by finite search budget, Monte-Carlo seed noise, or the
root selector.  Anything left over remains a confounded information-state /
evaluation / opponent-modeling problem and is not automatically attributed to
the leaf evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from srcs.metagross.known_team_decision_v2 import (
    action_rows,
    argmax,
    canonical_json,
    load_engine,
    seed,
    sha256_bytes,
    sha256_path,
    write_json,
)
from srcs.metagross.known_team_decision_v2_phase2 import (
    WORLD_BANK_SCHEMA,
    aggregate_advantages,
    aggregate_visit_policy,
    normalized_weights,
)


SCHEMA = "metagross-known-team-search-failure-attribution/v1"
PROTOCOL_SCHEMA = "metagross-known-team-search-failure-attribution-protocol/v1"
PHASE1_SCHEMA = "metagross-known-team-decision-phase1/v2"
PARTICLES = 16
BUDGETS = (5_000, 20_000, 80_000)
REPEATS = 2
CURRENT_BUDGET = 20_000
HIGH_BUDGET = 80_000
BENEFICIAL_DELTA = 0.02
HARMFUL_DELTA = -0.02
MASTER_SEED = "metagross-known-team-search-failure-attribution-20260812"


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _protocol_payload(phase1_path: Path, bank_path: Path) -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "frozen_before_execution",
        "inputs": {
            "phase1": {"path": str(phase1_path), "sha256": sha256_path(phase1_path)},
            "world_bank": {"path": str(bank_path), "sha256": sha256_path(bank_path)},
        },
        "implementation": {
            "path": str(_implementation_path()),
            "sha256": sha256_path(_implementation_path()),
        },
        "configuration": {
            "master_seed": MASTER_SEED,
            "root_cohort": "representative_and_stable_meaningful_phase1_headroom",
            "particle_count": PARTICLES,
            "particle_selection": "seeded_systematic_resampling_from_alpha_0_weights",
            "budgets": list(BUDGETS),
            "repeats": REPEATS,
            "threads": 1,
            "current_budget": CURRENT_BUDGET,
            "high_budget": HIGH_BUDGET,
            "selectors": ["visit_argmax", "mean_q_advantage_argmax"],
            "beneficial_teacher_delta": BENEFICIAL_DELTA,
            "harmful_teacher_delta": HARMFUL_DELTA,
            "teacher_role": "diagnostic_only_known-team_counterfactual",
            "public_ladder_authorized": False,
        },
    }


def freeze_protocol(phase1_path: Path, bank_path: Path) -> dict[str, Any]:
    protocol = _protocol_payload(phase1_path, bank_path)
    protocol["protocol_sha256"] = sha256_bytes(canonical_json(protocol).encode("ascii"))
    return protocol


def validate_protocol(
    protocol: Mapping[str, Any], phase1_path: Path, bank_path: Path
) -> None:
    unhashed = dict(protocol)
    claimed = unhashed.pop("protocol_sha256", None)
    if claimed != sha256_bytes(canonical_json(unhashed).encode("ascii")):
        raise ValueError("attribution protocol content hash does not match")
    expected = _protocol_payload(phase1_path, bank_path)
    if unhashed != expected:
        raise ValueError("attribution inputs, implementation, or configuration differ from protocol")


def _identity(root: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(root["corpus_uid"]), str(root["observer"]), int(root["decision_idx"])


def _cohort(phase1: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        root
        for root in phase1["roots"]
        if root["panel"] == "representative"
        and root["headroom_stable"]
        and root["meaningful_headroom"]
    ]


def systematic_indices(
    weights: Sequence[float], count: int, offset_uniform: float
) -> list[int]:
    """Deterministic systematic resampling with one offset in [0, 1)."""
    if count <= 0 or not weights:
        raise ValueError("systematic resampling requires particles and weights")
    if not 0.0 <= offset_uniform < 1.0:
        raise ValueError("systematic resampling offset must be in [0, 1)")
    total = math.fsum(weights)
    if total <= 0 or any(weight < 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("systematic resampling weights are invalid")
    normalized = [weight / total for weight in weights]
    cumulative: list[float] = []
    running = 0.0
    for weight in normalized:
        running += weight
        cumulative.append(running)
    cumulative[-1] = 1.0
    result: list[int] = []
    cursor = 0
    start = offset_uniform / count
    for particle in range(count):
        target = start + particle / count
        while cursor + 1 < len(cumulative) and target > cumulative[cursor]:
            cursor += 1
        result.append(cursor)
    return result


def selected_particles(root: Mapping[str, Any]) -> list[dict[str, Any]]:
    weights = normalized_weights(root["draws"], 0.0)
    identity = _identity(root)
    offset = random.Random(seed(MASTER_SEED, *identity, "particles")).random()
    indices = systematic_indices(weights, PARTICLES, offset)
    return [root["draws"][index] for index in indices]


def _draw_searches(
    particles: Sequence[Mapping[str, Any]], searches: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    return [searches[particle["state_sha256"]]["actions"] for particle in particles]


def _selector_result(
    searches: Sequence[Mapping[str, Any]], baseline_action: str
) -> dict[str, Any]:
    weights = [1.0 / len(searches)] * len(searches)
    visit_policy = aggregate_visit_policy(searches, weights)
    advantages = aggregate_advantages(searches, weights, baseline_action)
    return {
        "visit_policy": visit_policy,
        "visit_action": argmax(visit_policy),
        "mean_q_advantages": advantages,
        "mean_q_advantage_action": argmax(advantages),
    }


def _search_particles(
    engine: Any,
    particles: Sequence[Mapping[str, Any]],
    allowed: set[str],
    identity: Sequence[object],
    budget: int,
    repeat: int,
) -> dict[str, dict[str, Any]]:
    searches: dict[str, dict[str, Any]] = {}
    for particle in particles:
        state_hash = particle["state_sha256"]
        if state_hash in searches:
            continue
        result = engine.monte_carlo_tree_search(
            engine.State.from_string(particle["state"]),
            duration_ms=0,
            iterations=budget,
            threads=1,
            seed=seed(MASTER_SEED, *identity, "budget", budget, "repeat", repeat, state_hash),
        )
        searches[state_hash] = {
            "iterations": budget,
            "repeat": repeat,
            "actions": action_rows(result, allowed),
        }
    return searches


def _teacher_delta(teacher: Mapping[str, float], action: str, baseline: str) -> float:
    return float(teacher[action]) - float(teacher[baseline])


def _summarize_budget(
    repeats: Sequence[Mapping[str, Any]], teacher: Mapping[str, float], baseline: str
) -> dict[str, Any]:
    visit_actions = [str(row["visit_action"]) for row in repeats]
    q_actions = [str(row["mean_q_advantage_action"]) for row in repeats]
    visit_deltas = [_teacher_delta(teacher, action, baseline) for action in visit_actions]
    q_deltas = [_teacher_delta(teacher, action, baseline) for action in q_actions]
    return {
        "repeats": list(repeats),
        "visit": {
            "actions": visit_actions,
            "seed_agreement": len(set(visit_actions)) == 1,
            "teacher_deltas": visit_deltas,
            "beneficial": len(set(visit_actions)) == 1 and min(visit_deltas) >= BENEFICIAL_DELTA,
            "harmful": len(set(visit_actions)) == 1 and max(visit_deltas) <= HARMFUL_DELTA,
        },
        "mean_q_advantage": {
            "actions": q_actions,
            "seed_agreement": len(set(q_actions)) == 1,
            "teacher_deltas": q_deltas,
            "beneficial": len(set(q_actions)) == 1 and min(q_deltas) >= BENEFICIAL_DELTA,
            "harmful": len(set(q_actions)) == 1 and max(q_deltas) <= HARMFUL_DELTA,
        },
    }


def classify_root(budgets: Mapping[str, Any]) -> str:
    current = budgets[str(CURRENT_BUDGET)]
    high = budgets[str(HIGH_BUDGET)]
    if current["visit"]["beneficial"]:
        return "particle_panel_or_current_search_resolved"
    if current["mean_q_advantage"]["beneficial"]:
        return "root_selector_or_visit_allocation"
    if high["visit"]["beneficial"] and not current["visit"]["beneficial"]:
        return "finite_search_budget"
    if high["mean_q_advantage"]["beneficial"] and not high["visit"]["beneficial"]:
        return "root_selector_or_visit_allocation"
    if not current["visit"]["seed_agreement"]:
        return "monte_carlo_seed_variance"
    return "unresolved_information_value_or_opponent_model"


def adaptive_budget_decision(
    checkpoints: Sequence[tuple[int, Mapping[str, Any]]], margin_threshold: float
) -> int:
    """Return the first stable checkpoint, otherwise the final budget.

    Stability requires two successive checkpoints to agree on the visit argmax
    and the newer checkpoint's top-two visit margin to clear the threshold.
    This function is the deployment-neutral controller contract; threshold
    selection and utility evaluation belong to a separate development study.
    """
    if not checkpoints or margin_threshold < 0 or not math.isfinite(margin_threshold):
        raise ValueError("adaptive budget checkpoints or threshold are invalid")
    previous_action: str | None = None
    previous_budget = -1
    for budget, policy in checkpoints:
        if budget <= previous_budget or not policy:
            raise ValueError("adaptive budget checkpoints must be ordered and nonempty")
        action = argmax(policy)
        masses = sorted((float(value) for value in policy.values()), reverse=True)
        margin = masses[0] - masses[1] if len(masses) > 1 else 1.0
        if previous_action == action and margin >= margin_threshold:
            return budget
        previous_action = action
        previous_budget = budget
    return checkpoints[-1][0]


def _partial_result(
    protocol: Mapping[str, Any], roots: Sequence[Mapping[str, Any]], complete: bool
) -> dict[str, Any]:
    counts = Counter(root["attribution"] for root in roots)
    return {
        "schema": SCHEMA,
        "protocol_sha256": protocol["protocol_sha256"],
        "complete": complete,
        "completed_roots": len(roots),
        "summary": {
            "attribution_counts": dict(sorted(counts.items())),
            "finite_search_budget_fraction": counts["finite_search_budget"] / len(roots)
            if roots
            else None,
            "particle_panel_or_current_search_resolved_fraction": counts[
                "particle_panel_or_current_search_resolved"
            ]
            / len(roots)
            if roots
            else None,
            "selector_fraction": counts["root_selector_or_visit_allocation"] / len(roots)
            if roots
            else None,
            "seed_variance_fraction": counts["monte_carlo_seed_variance"] / len(roots)
            if roots
            else None,
            "unresolved_fraction": counts["unresolved_information_value_or_opponent_model"]
            / len(roots)
            if roots
            else None,
            "learned_hidden_world_leaf_authorized": False,
            "public_ladder_authorized": False,
        },
        "roots": list(roots),
    }


def run(
    phase1_path: Path,
    bank_path: Path,
    protocol_path: Path,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    max_roots: int | None = None,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol, phase1_path, bank_path)
    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if phase1.get("schema") != PHASE1_SCHEMA or bank.get("schema") != WORLD_BANK_SCHEMA:
        raise ValueError("invalid Phase-1 or world-bank schema")
    if not bank.get("complete"):
        raise ValueError("world bank is incomplete")
    phase1_roots = _cohort(phase1)
    if max_roots is not None:
        phase1_roots = phase1_roots[:max_roots]
    bank_roots = {_identity(root): root for root in bank["roots"]}
    if any(_identity(root) not in bank_roots for root in phase1_roots):
        raise ValueError("world bank does not cover the frozen headroom cohort")

    roots: list[dict[str, Any]] = []
    if resume and checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("checkpoint does not match frozen protocol")
        roots = list(checkpoint.get("roots") or [])
        if [tuple(root["identity"]) for root in roots] != [
            _identity(root) for root in phase1_roots[: len(roots)]
        ]:
            raise ValueError("checkpoint is not the frozen cohort prefix")

    engine = load_engine()
    for phase1_root in phase1_roots[len(roots) :]:
        identity = _identity(phase1_root)
        bank_root = bank_roots[identity]
        particles = selected_particles(bank_root)
        allowed = set(phase1_root["legal_actions"])
        baseline = str(phase1_root["baseline_action"])
        teacher = phase1_root["teacher_mean_q"]
        budget_rows: dict[str, Any] = {}
        for budget in BUDGETS:
            repeat_rows = []
            for repeat in range(REPEATS):
                if budget == CURRENT_BUDGET:
                    stored = bank_root["searches" if repeat == 0 else "repeat_searches"]
                    searches = _draw_searches(particles, stored)
                else:
                    generated = _search_particles(
                        engine, particles, allowed, identity, budget, repeat
                    )
                    searches = _draw_searches(particles, generated)
                repeat_rows.append(_selector_result(searches, baseline))
            budget_rows[str(budget)] = _summarize_budget(repeat_rows, teacher, baseline)
        roots.append(
            {
                "identity": list(identity),
                "battle_id": phase1_root["battle_id"],
                "turn": phase1_root["turn"],
                "baseline_action": baseline,
                "teacher_best_action": phase1_root["teacher_best_action"],
                "baseline_headroom": phase1_root["baseline_headroom"],
                "particle_draw_indices": [int(row["draw_index"]) for row in particles],
                "particle_state_hashes": [row["state_sha256"] for row in particles],
                "unique_particle_states": len({row["state_sha256"] for row in particles}),
                "budgets": budget_rows,
                "attribution": classify_root(budget_rows),
            }
        )
        if checkpoint_path is not None:
            write_json(checkpoint_path, _partial_result(protocol, roots, False))
    return _partial_result(protocol, roots, True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--phase1", type=Path, required=True)
    freeze.add_argument("--world-bank", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--phase1", type=Path, required=True)
    execute.add_argument("--world-bank", type=Path, required=True)
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--checkpoint", type=Path)
    execute.add_argument("--resume", action="store_true")
    execute.add_argument("--max-roots", type=int)
    args = parser.parse_args()
    if args.command == "freeze":
        write_json(args.output, freeze_protocol(args.phase1, args.world_bank))
    else:
        write_json(
            args.output,
            run(
                args.phase1,
                args.world_bank,
                args.protocol,
                checkpoint_path=args.checkpoint,
                resume=args.resume,
                max_roots=args.max_roots,
            ),
        )


if __name__ == "__main__":
    main()
