from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple


@dataclass(frozen=True)
class SelectiveSharedRootMetrics:
    weighted_top_action_disagreement: float
    weighted_js_divergence: float
    aggregate_top_visit_mass: float
    aggregate_top_two_margin: float
    world_count: int
    effective_world_count: float
    action_count: int
    has_multi_action_world: bool = True


@dataclass(frozen=True)
class SelectiveActionDecision:
    action: str
    overridden: bool
    reason: str


def _normalized_weights(weights: Sequence[float]) -> list[float]:
    clean = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in clean):
        raise ValueError("world weights must be finite and non-negative")
    total = sum(clean)
    if total <= 0.0:
        raise ValueError("world weights must have positive mass")
    return [weight / total for weight in clean]


def compute_selective_shared_root_metrics(
    policies: Sequence[Mapping[str, float]],
    world_weights: Sequence[float],
) -> SelectiveSharedRootMetrics:
    """Summarize disagreement among per-world canonical root policies."""
    if not policies or len(policies) != len(world_weights):
        raise ValueError("policies and world weights must have the same non-zero length")

    weights = _normalized_weights(world_weights)
    actions = sorted({str(action) for policy in policies for action in policy})
    if not actions:
        raise ValueError("root policies must contain at least one action")

    normalized_policies: list[dict[str, float]] = []
    for policy in policies:
        clean = {}
        for action, mass in policy.items():
            try:
                value = float(mass)
            except (TypeError, ValueError):
                value = 0.0
            if math.isfinite(value) and value > 0.0:
                clean[str(action)] = value
        total = sum(clean.values())
        if total > 0.0:
            normalized_policies.append({action: clean.get(action, 0.0) / total for action in actions})
        else:
            policy_actions = {str(action) for action in policy}
            if not policy_actions:
                raise ValueError("each root policy must contain at least one action")
            uniform = 1.0 / len(policy_actions)
            normalized_policies.append(
                {action: uniform if action in policy_actions else 0.0 for action in actions}
            )

    aggregate = {
        action: sum(weight * policy[action] for weight, policy in zip(weights, normalized_policies))
        for action in actions
    }
    votes: dict[str, float] = {}
    for weight, policy in zip(weights, normalized_policies):
        top_action = max(actions, key=lambda action: (policy[action], action))
        votes[top_action] = votes.get(top_action, 0.0) + weight

    js_divergence = 0.0
    for weight, policy in zip(weights, normalized_policies):
        for action in actions:
            probability = policy[action]
            mixture_probability = aggregate[action]
            if probability > 0.0 and mixture_probability > 0.0:
                js_divergence += weight * probability * math.log(probability / mixture_probability)

    ranked_mass = sorted(aggregate.values(), reverse=True)
    top_mass = ranked_mass[0]
    second_mass = ranked_mass[1] if len(ranked_mass) > 1 else 0.0
    effective_world_count = 1.0 / sum(weight * weight for weight in weights)
    return SelectiveSharedRootMetrics(
        weighted_top_action_disagreement=max(0.0, 1.0 - max(votes.values())),
        weighted_js_divergence=max(0.0, js_divergence),
        aggregate_top_visit_mass=top_mass,
        aggregate_top_two_margin=top_mass - second_mass,
        world_count=len(policies),
        effective_world_count=effective_world_count,
        action_count=len(actions),
        has_multi_action_world=any(len(policy) > 1 for policy in policies),
    )


@dataclass(frozen=True)
class ConfidenceMixture:
    alpha: float
    blended_distribution: dict[str, float]


def compute_confidence_mixture(
    *,
    paired_lcb: float,
    lcb_scale: float,
    baseline_action: str,
    shared_policy: Sequence[Tuple[str, float]],
) -> ConfidenceMixture:
    """Compute a confidence-weighted mixture between the baseline and shared policy.

    alpha = clamp(paired_lcb / lcb_scale, 0, 1).  When alpha is 0 the blended
    distribution is pure baseline.  When alpha is 1 the blended distribution is
    the shared policy (normalised).
    """
    if paired_lcb <= 0.0 or lcb_scale <= 0.0 or not math.isfinite(paired_lcb):
        return ConfidenceMixture(alpha=0.0, blended_distribution={baseline_action: 1.0})
    alpha = max(0.0, min(1.0, paired_lcb / lcb_scale))
    if alpha <= 0.0:
        return ConfidenceMixture(alpha=0.0, blended_distribution={baseline_action: 1.0})
    shared_prob_map = {action: prob for action, prob in shared_policy}
    legal_actions = sorted(set([baseline_action] + [a for a, _ in shared_policy]))
    blended: dict[str, float] = {}
    for action in legal_actions:
        baseline_prob = 1.0 if action == baseline_action else 0.0
        shared_prob = shared_prob_map.get(action, 0.0)
        blended[action] = (1.0 - alpha) * baseline_prob + alpha * shared_prob
    total = sum(blended.values())
    if total > 0.0:
        blended = {a: p / total for a, p in blended.items()}
    return ConfidenceMixture(alpha=alpha, blended_distribution=blended)


def should_trigger_selective_shared_root(
    metrics: SelectiveSharedRootMetrics,
    disagreement_threshold: float = 0.45,
    js_threshold: float = 0.25,
    top_mass_threshold: float = 0.65,
) -> bool:
    return (
        metrics.action_count > 1
        and metrics.has_multi_action_world
        and metrics.weighted_top_action_disagreement >= disagreement_threshold
        and metrics.weighted_js_divergence >= js_threshold
        and metrics.aggregate_top_visit_mass <= top_mass_threshold
    )


def decide_selective_action(
    *,
    mode: str,
    baseline_action: str,
    triggered: bool,
    shared_action: str | None = None,
    paired_available: bool = False,
    paired_lcb: float | None = None,
    lcb_margin: float = 0.0,
) -> SelectiveActionDecision:
    if mode not in {"audit", "override"}:
        raise ValueError("selective shared-root mode must be audit or override")
    if not triggered:
        return SelectiveActionDecision(baseline_action, False, "not_triggered")
    if shared_action is None:
        return SelectiveActionDecision(baseline_action, False, "shared_failed")
    if mode == "audit":
        return SelectiveActionDecision(baseline_action, False, "audit")
    if not paired_available or paired_lcb is None or not math.isfinite(paired_lcb):
        return SelectiveActionDecision(baseline_action, False, "paired_unavailable")
    if paired_lcb <= max(0.0, lcb_margin):
        return SelectiveActionDecision(baseline_action, False, "lcb_not_above_margin")
    return SelectiveActionDecision(shared_action, shared_action != baseline_action, "override")
