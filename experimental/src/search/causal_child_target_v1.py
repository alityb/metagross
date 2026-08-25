"""Leak-free child information states and full-policy target aggregation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from search.public_search_state_v1 import canonical_bytes, extract_public_search_state


SCHEMA = "metagross-causal-child-information-state/v1"
GROUP_SCHEMA = "metagross-causal-child-search-target-group/v1"


class CausalChildTargetError(ValueError):
    pass


def child_information_state(state: Any, engine: Any) -> dict[str, Any]:
    """Project a state without exposing sampled opponent exact/max HP."""
    payload = extract_public_search_state(state, engine)
    payload["schema"] = SCHEMA
    for pokemon in payload["opponent"]["pokemon"]:
        if pokemon is None:
            continue
        hp = int(pokemon.pop("hp"))
        maxhp = int(pokemon.pop("maxhp"))
        if maxhp <= 0 or hp < 0 or hp > maxhp:
            raise CausalChildTargetError("invalid opponent HP authority")
        pokemon["hp_percent"] = 0 if hp == 0 else math.ceil(100 * hp / maxhp)
        # Engine move slots are a property of each hidden completion. Public
        # move knowledge is a set, so slot order cannot enter the fingerprint.
        pokemon["moves"] = sorted(pokemon["moves"])
    canonical_bytes(payload)
    return payload


def public_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def snapshot_teacher_result(result: Any) -> dict[str, Any]:
    total = int(result.total_visits)
    if total <= 0:
        raise CausalChildTargetError("teacher returned no visits")

    def side(rows: Sequence[Any]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            visits = int(row.visits)
            total_score = float(row.total_score)
            q = None if visits == 0 else total_score / visits
            if visits < 0 or not math.isfinite(total_score):
                raise CausalChildTargetError("invalid teacher action statistics")
            if q is not None and (not math.isfinite(q) or not 0.0 <= q <= 1.0):
                raise CausalChildTargetError("teacher Q is outside [0,1]")
            output.append({
                "action": str(row.move_choice),
                "visits": visits,
                "total_score": total_score,
                "q": q,
                "completed_q": None,
            })
        if not output or len({row["action"] for row in output}) != len(output):
            raise CausalChildTargetError("teacher actions are empty or duplicated")
        return output

    side_one = side(list(result.side_one))
    side_two = side(list(result.side_two))
    if sum(row["visits"] for row in side_one) != total:
        raise CausalChildTargetError("side-one visits do not sum to total")
    if sum(row["visits"] for row in side_two) != total:
        raise CausalChildTargetError("side-two visits do not sum to total")
    argmax = max(side_one, key=lambda row: (row["visits"], row["action"]))["action"]
    return {
        "total_visits": total,
        "side_one": side_one,
        "side_two": side_two,
        "completed_q_available": False,
        "completed_q": None,
        "argmax_action": argmax,
    }


def aggregate_target_members(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not members:
        raise CausalChildTargetError("target group has no members")
    fingerprints = {str(member["public_fingerprint"]) for member in members}
    splits = {str(member["split"]) for member in members}
    if len(fingerprints) != 1 or len(splits) != 1:
        raise CausalChildTargetError("target group crosses fingerprint or split")
    legal = tuple(members[0]["legal_actions"])
    if not legal or any(tuple(member["legal_actions"]) != legal for member in members):
        raise CausalChildTargetError("group legal actions disagree")
    weights = [float(member["weight"]) for member in members]
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise CausalChildTargetError("group has invalid member weight")
    total_weight = math.fsum(weights)
    normalized = [weight / total_weight for weight in weights]
    effective_sample_size = total_weight * total_weight / math.fsum(
        weight * weight for weight in weights
    )

    visit_mass = dict.fromkeys(legal, 0.0)
    q_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    argmax_mass = dict.fromkeys(legal, 0.0)
    for member, weight in zip(members, normalized, strict=True):
        teacher = member["teacher"]
        denominator = int(teacher["total_visits"])
        by_action = {row["action"]: row for row in teacher["side_one"]}
        if set(by_action) != set(legal) or denominator <= 0:
            raise CausalChildTargetError("teacher/legal action mismatch")
        for action in legal:
            row = by_action[action]
            visit_mass[action] += weight * int(row["visits"]) / denominator
            if row["q"] is not None:
                q_values[action].append((weight, float(row["q"])))
        argmax = str(teacher["argmax_action"])
        if argmax not in argmax_mass:
            raise CausalChildTargetError("teacher argmax is illegal")
        argmax_mass[argmax] += weight
    visit_total = math.fsum(visit_mass.values())
    if not math.isclose(visit_total, 1.0, abs_tol=1e-9):
        raise CausalChildTargetError("aggregate visit policy is not normalized")
    q_summary = {}
    for action in legal:
        rows = q_values[action]
        if not rows:
            q_summary[action] = {"mean": None, "min": None, "max": None, "weight": 0.0}
            continue
        mass = math.fsum(weight for weight, _value in rows)
        values = [value for _weight, value in rows]
        q_summary[action] = {
            "mean": math.fsum(weight * value for weight, value in rows) / mass,
            "min": min(values),
            "max": max(values),
            "weight": mass,
        }
    control = max(legal, key=lambda action: (visit_mass[action], action))
    return {
        "schema": GROUP_SCHEMA,
        "split": next(iter(splits)),
        "public_fingerprint": next(iter(fingerprints)),
        "public_state": members[0]["public_state"],
        "legal_actions": list(legal),
        "member_count": len(members),
        "total_weight": total_weight,
        "effective_sample_size": effective_sample_size,
        "visit_policy": visit_mass,
        "q": q_summary,
        "teacher_argmax_distribution": argmax_mass,
        "hidden_world_argmax_disagreement": sum(value > 0 for value in argmax_mass.values()) > 1,
        "same_teacher_one_hot_control_action": control,
        "same_teacher_control_is_independent": False,
        "member_ids": [str(member["target_id"]) for member in members],
    }


def group_target_members(members: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    fingerprint_splits: dict[str, set[str]] = defaultdict(set)
    for member in members:
        fingerprint = str(member["public_fingerprint"])
        split = str(member["split"])
        by_key[(split, fingerprint)].append(member)
        fingerprint_splits[fingerprint].add(split)
    crossing = sorted(
        fingerprint for fingerprint, splits in fingerprint_splits.items() if len(splits) > 1
    )
    if crossing:
        raise CausalChildTargetError("public fingerprint crosses battle splits")
    return [
        aggregate_target_members(by_key[key])
        for key in sorted(by_key)
    ]
