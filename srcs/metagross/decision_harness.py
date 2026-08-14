"""Typed composition contracts for the deterministic battle decision harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class PolicySnapshot:
    priors: tuple[tuple[str, float], ...]
    opponent_priors: tuple[tuple[str, float], ...] | None
    context: Mapping[str, object]


class PolicyInterface(Protocol):
    def observe(self, tag: str, lines: list[str]) -> None: ...

    def propose(self, battle: object) -> PolicySnapshot: ...

    def acknowledge(self, snapshot: PolicySnapshot, action: str) -> None: ...


class BeliefInterface(Protocol):
    def expand(self, battle: object, search: object, channel: str): ...


class SearchInterface(Protocol):
    def evaluate(self, states: list[str], duration_ms: int, threads: int): ...

    def solve_shared_root(
        self,
        states: list[str],
        particle_weights: list[float],
        iterations: int,
        continuation_iterations: int,
        seed: int,
    ): ...

    def holdout(
        self,
        states: list[str],
        baseline: str,
        candidate: str,
        continuation_steps: int,
        seeds: list[int],
        candidate_rank: int,
        *,
        request_channel: str = "certification-request",
        telemetry_key: str = "holdout",
    ): ...


class ControllerInterface(Protocol):
    def select(self, battle: object, mcts_results: object, priors: object, **kwargs): ...

    def select_shared(
        self, battle: object, shared_result: object, priors: object, seed: int, **kwargs
    ): ...


class VerifierInterface(Protocol):
    def certify(self, results: object, *args, **kwargs) -> dict[str, object]: ...

    def combine(self, certificates: dict[int, dict]) -> dict[str, object]: ...


@dataclass(frozen=True)
class DecisionHarness:
    policy: PolicyInterface
    belief: BeliefInterface
    search: SearchInterface
    controller: ControllerInterface
    verifier: VerifierInterface


@dataclass(frozen=True)
class CallablePolicy:
    observe_fn: Callable[[str, list[str]], None]
    propose_fn: Callable[[object], PolicySnapshot]
    acknowledge_fn: Callable[[PolicySnapshot, str], None]

    def observe(self, tag: str, lines: list[str]) -> None:
        self.observe_fn(tag, lines)

    def propose(self, battle: object) -> PolicySnapshot:
        return self.propose_fn(battle)

    def acknowledge(self, snapshot: PolicySnapshot, action: str) -> None:
        self.acknowledge_fn(snapshot, action)


@dataclass(frozen=True)
class CallableBelief:
    expand_fn: Callable[..., Any]

    def expand(self, battle: object, search: object, channel: str):
        return self.expand_fn(battle, search, channel)


@dataclass(frozen=True)
class CallableSearch:
    evaluate_fn: Callable[..., Any]
    holdout_fn: Callable[..., Any]
    solve_shared_root_fn: Callable[..., Any]

    def evaluate(self, states: list[str], duration_ms: int, threads: int):
        return self.evaluate_fn(states, duration_ms, threads)

    def holdout(
        self,
        states: list[str],
        baseline: str,
        candidate: str,
        continuation_steps: int,
        seeds: list[int],
        candidate_rank: int,
        *,
        request_channel: str = "certification-request",
        telemetry_key: str = "holdout",
    ):
        return self.holdout_fn(
            states,
            baseline,
            candidate,
            continuation_steps,
            seeds,
            candidate_rank,
            request_channel=request_channel,
            telemetry_key=telemetry_key,
        )

    def solve_shared_root(
        self,
        states: list[str],
        particle_weights: list[float],
        iterations: int,
        continuation_iterations: int,
        seed: int,
    ):
        return self.solve_shared_root_fn(
            states,
            particle_weights,
            iterations,
            continuation_iterations,
            seed,
        )


@dataclass(frozen=True)
class CallableController:
    select_fn: Callable[..., Any]
    select_shared_fn: Callable[..., Any] | None = None

    def select(self, battle: object, mcts_results: object, priors: object, **kwargs):
        return self.select_fn(battle, mcts_results, priors, **kwargs)

    def select_shared(
        self, battle: object, shared_result: object, priors: object, seed: int, **kwargs
    ):
        if self.select_shared_fn is None:
            raise RuntimeError("shared-root controller is not configured")
        return self.select_shared_fn(
            battle, shared_result, priors, seed=seed, **kwargs
        )


@dataclass(frozen=True)
class CallableVerifier:
    certify_fn: Callable[..., dict[str, object]]
    combine_fn: Callable[[dict[int, dict]], dict[str, object]]

    def certify(self, results: object, *args, **kwargs) -> dict[str, object]:
        return self.certify_fn(results, *args, **kwargs)

    def combine(self, certificates: dict[int, dict]) -> dict[str, object]:
        return self.combine_fn(certificates)


TAIL_CHECKS = (
    "candidate_catastrophe_rate",
    "symmetric_catastrophe_rate_gap",
    "symmetric_catastrophe_severity",
    "lower_tail_cvar",
)


@dataclass(frozen=True)
class RecursiveShadowPlan:
    triggered: bool
    operation: str | None
    reason: str
    candidate: str | None
    depth: int = 0


def _horizon_certificate(certificates: Mapping[object, object], horizon: int):
    return certificates.get(horizon) or certificates.get(str(horizon))


def _tail_unstable(first: Mapping[str, object], second: Mapping[str, object]) -> bool:
    first_checks = first.get("checks")
    second_checks = second.get("checks")
    if not isinstance(first_checks, Mapping) or not isinstance(second_checks, Mapping):
        return False
    return any(first_checks.get(name) != second_checks.get(name) for name in TAIL_CHECKS)


def plan_recursive_shadow(
    provisional: Mapping[str, object],
    certificates_by_action: Mapping[str, Mapping[object, object]],
) -> RecursiveShadowPlan:
    """Plan at most one diagnostic allocation after policy/search disagreement."""
    baseline = provisional.get("baseline")
    search_top = provisional.get("raw_choice")
    if not isinstance(baseline, str) or not isinstance(search_top, str):
        return RecursiveShadowPlan(False, None, "invalid_provisional", None)
    if baseline == search_top:
        return RecursiveShadowPlan(False, None, "policy_search_agreement", search_top)
    certificates = certificates_by_action.get(search_top)
    if not isinstance(certificates, Mapping):
        return RecursiveShadowPlan(False, None, "search_top_not_certified", search_top)
    first = _horizon_certificate(certificates, 1)
    second = _horizon_certificate(certificates, 2)
    if not isinstance(first, Mapping):
        return RecursiveShadowPlan(False, None, "missing_parent_evidence", search_top)
    if not isinstance(second, Mapping):
        return RecursiveShadowPlan(
            True, "horizon", "policy_search_disagreement", search_top, depth=1
        )
    if _tail_unstable(first, second):
        return RecursiveShadowPlan(
            True, "worlds", "catastrophic_tail_unstable", search_top, depth=1
        )
    return RecursiveShadowPlan(False, None, "catastrophic_tail_stable", search_top)


def execute_recursive_shadow(
    plan: RecursiveShadowPlan,
    played_action: str,
    allocate: Callable[[RecursiveShadowPlan], Mapping[str, object]],
) -> dict[str, object]:
    """Execute one bounded diagnostic node without exposing evidence to selection."""
    artifact: dict[str, object] = {
        "schema_version": 1,
        "evidence_kind": "bounded_recursive_shadow_v1",
        "shadow_only": True,
        "admission_eligible": False,
        "played_action": played_action,
        "played_action_unchanged": True,
        "triggered": plan.triggered,
        "operation": plan.operation,
        "trigger_reason": plan.reason,
        "candidate": plan.candidate,
        "limits": {"max_depth": 1, "max_candidates": 1, "max_remote_batches": 1},
        "nodes": [],
        "complete": True,
    }
    if not plan.triggered:
        artifact["stop_reason"] = plan.reason
        return artifact
    try:
        node = dict(allocate(plan))
        node["depth"] = plan.depth
        node["operation"] = plan.operation
        artifact["nodes"] = [node]
        artifact["stop_reason"] = "max_depth_reached"
    except Exception as exc:
        artifact["complete"] = False
        artifact["stop_reason"] = "shadow_allocation_failed"
        artifact["error_type"] = type(exc).__name__
    return artifact
