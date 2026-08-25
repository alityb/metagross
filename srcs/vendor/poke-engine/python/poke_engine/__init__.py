from dataclasses import dataclass
from enum import StrEnum
import math

from .poke_engine import *

_rust_paired_root_policy_evaluation = paired_root_policy_evaluation
_rust_shared_information_set_root_search = shared_information_set_root_search


class Weather(StrEnum):
    NONE = "none"
    SUN = "sun"
    RAIN = "rain"
    SAND = "sand"
    HAIL = "hail"
    SNOW = "snow"
    HARSH_SUN = "harshsun"
    HEAVY_RAIN = "heavyrain"


class Terrain(StrEnum):
    NONE = "none"
    GRASSY = "grassyterrain"
    ELECTRIC = "electricterrain"
    MISTY = "mistyterrain"
    PSYCHIC = "psychicterrain"


class PokemonIndex(StrEnum):
    P0 = "0"
    P1 = "1"
    P2 = "2"
    P3 = "3"
    P4 = "4"
    P5 = "5"


@dataclass
class IterativeDeepeningResult:
    """
    Result of an Iterative Deepening Expectiminimax Search

    :param side_one: The moves for side_one
    :type side_one: list[str]
    :param side_two: The moves for side_two
    :type side_two: list[str]
    :param matrix: A vector representing the payoff matrix of the search.
        Pruned branches are represented by None
    :type matrix: int
    :param depth_searched: The depth that was searched to
    :type depth_searched: int
    """

    side_one: list[str]
    side_two: list[str]
    matrix: list[float]
    depth_searched: int

    @classmethod
    def _from_rust(cls, rust_result):
        return cls(
            side_one=rust_result.s1,
            side_two=rust_result.s2,
            matrix=rust_result.matrix,
            depth_searched=rust_result.depth_searched,
        )

    def get_safest_move(self) -> str:
        """
        Get the safest move for side_one
        The safest move is the move that minimizes the loss for the turn

        :return: The safest move
        :rtype: str
        """
        safest_value = float("-inf")
        safest_s1_index = 0
        vec_index = 0
        for i in range(len(self.side_one)):
            worst_case_this_row = float("inf")
            for _ in range(len(self.side_two)):
                score = self.matrix[vec_index]
                if score < worst_case_this_row:
                    worst_case_this_row = score

            if worst_case_this_row > safest_value:
                safest_s1_index = i
                safest_value = worst_case_this_row

        return self.side_one[safest_s1_index]


@dataclass
class MctsSideResult:
    """
    Result of a Monte Carlo Tree Search for a single side

    :param move_choice: The move that was chosen
    :type move_choice: str
    :param total_score: The total score of the chosen move
    :type total_score: float
    :param visits: The number of times the move was chosen
    :type visits: int
    """

    move_choice: str
    total_score: float
    visits: int


@dataclass
class MctsResult:
    """
    Result of a Monte Carlo Tree Search

    :param side_one: Result for side one
    :type side_one: list[MctsSideResult]
    :param side_two: Result for side two
    :type side_two: list[MctsSideResult]
    :param total_visits: Total number of monte carlo iterations
    :type total_visits: int
    """

    side_one: list[MctsSideResult]
    side_two: list[MctsSideResult]
    total_visits: int
    learned_evaluations: int = 0
    hand_evaluations: int = 0
    terminal_evaluations: int = 0

    @classmethod
    def _from_rust(cls, rust_result):
        return cls(
            side_one=[
                MctsSideResult(
                    move_choice=i.move_choice,
                    total_score=i.total_score,
                    visits=i.visits,
                )
                for i in rust_result.s1
            ],
            side_two=[
                MctsSideResult(
                    move_choice=i.move_choice,
                    total_score=i.total_score,
                    visits=i.visits,
                )
                for i in rust_result.s2
            ],
            total_visits=rust_result.iteration_count,
            learned_evaluations=rust_result.learned_evaluations,
            hand_evaluations=rust_result.hand_evaluations,
            terminal_evaluations=rust_result.terminal_evaluations,
        )


@dataclass(frozen=True)
class PairedRootPolicyEvaluation:
    pairs: int
    baseline_sum: float
    candidate_sum: float
    delta_sum: float
    delta_squared_sum: float
    catastrophic_count: int
    candidate_better_count: int
    baseline_better_count: int
    equal_count: int
    baseline_terminal_count: int
    candidate_terminal_count: int
    continuation_iterations_executed: int
    candidate_catastrophic_count: int = 0
    baseline_catastrophic_count: int = 0
    candidate_catastrophic_severity_sum: float = 0.0
    baseline_catastrophic_severity_sum: float = 0.0
    baseline_nonterminal_evaluation_delta_sum: float = 0.0
    candidate_nonterminal_evaluation_delta_sum: float = 0.0
    baseline_nonterminal_count: int = 0
    candidate_nonterminal_count: int = 0

    @classmethod
    def _from_rust(cls, result):
        return cls(
            pairs=result.pairs,
            baseline_sum=result.baseline_sum,
            candidate_sum=result.candidate_sum,
            delta_sum=result.delta_sum,
            delta_squared_sum=result.delta_squared_sum,
            catastrophic_count=result.catastrophic_count,
            candidate_better_count=result.candidate_better_count,
            baseline_better_count=result.baseline_better_count,
            equal_count=result.equal_count,
            baseline_terminal_count=result.baseline_terminal_count,
            candidate_terminal_count=result.candidate_terminal_count,
            continuation_iterations_executed=result.continuation_iterations_executed,
            candidate_catastrophic_count=result.candidate_catastrophic_count,
            baseline_catastrophic_count=result.baseline_catastrophic_count,
            candidate_catastrophic_severity_sum=result.candidate_catastrophic_severity_sum,
            baseline_catastrophic_severity_sum=result.baseline_catastrophic_severity_sum,
            baseline_nonterminal_evaluation_delta_sum=(
                result.baseline_nonterminal_evaluation_delta_sum
            ),
            candidate_nonterminal_evaluation_delta_sum=(
                result.candidate_nonterminal_evaluation_delta_sum
            ),
            baseline_nonterminal_count=result.baseline_nonterminal_count,
            candidate_nonterminal_count=result.candidate_nonterminal_count,
        )


@dataclass(frozen=True)
class SharedRootPolicyEntry:
    action: str
    probability: float
    counterfactual_value: float


@dataclass(frozen=True)
class SharedRootSourceParticle:
    input_index: int
    input_weight: float


@dataclass(frozen=True)
class SharedRootContinuation:
    seed: int
    requested_iterations: int
    executed_iterations: int
    visits: int
    total_score: float
    total_score_bits: int
    payoff: float
    payoff_bits: int


@dataclass(frozen=True)
class SharedRootReplayConfiguration:
    iterations: int
    continuation_iterations: int
    seed: int
    prior_strength: float


@dataclass(frozen=True)
class SharedRootReplayParticle:
    canonical_index: int
    state: str
    normalized_weight: float
    source_particles: tuple[SharedRootSourceParticle, ...]
    opponent_action_support: tuple[str, ...]
    normalized_opponent_prior: tuple[float, ...] | None
    payoff_matrix: tuple[tuple[float, ...], ...]
    continuations: tuple[tuple[SharedRootContinuation, ...], ...]
    opponent_policy: tuple[float, ...]

    @classmethod
    def _from_rust(cls, particle):
        return cls(
            canonical_index=particle.canonical_index,
            state=particle.state,
            normalized_weight=particle.normalized_weight,
            source_particles=tuple(
                SharedRootSourceParticle(source.input_index, source.input_weight)
                for source in particle.source_particles
            ),
            opponent_action_support=tuple(particle.opponent_action_support),
            normalized_opponent_prior=(
                tuple(particle.normalized_opponent_prior)
                if particle.normalized_opponent_prior is not None
                else None
            ),
            payoff_matrix=tuple(tuple(row) for row in particle.payoff_matrix),
            continuations=tuple(
                tuple(
                    SharedRootContinuation(
                        **{
                            name: getattr(cell, name)
                            for name in SharedRootContinuation.__dataclass_fields__
                        }
                    )
                    for cell in row
                )
                for row in particle.continuations
            ),
            opponent_policy=tuple(particle.opponent_policy),
        )


@dataclass(frozen=True)
class SharedRootReplayCapture:
    schema_version: int
    solver_contract: str
    configuration: SharedRootReplayConfiguration
    own_action_support: tuple[str, ...]
    normalized_player_prior: tuple[float, ...] | None
    canonical_particles: tuple[SharedRootReplayParticle, ...]

    @classmethod
    def _from_rust(cls, capture):
        return cls(
            schema_version=capture.schema_version,
            solver_contract=capture.solver_contract,
            configuration=SharedRootReplayConfiguration(
                **{
                    name: getattr(capture.configuration, name)
                    for name in SharedRootReplayConfiguration.__dataclass_fields__
                }
            ),
            own_action_support=tuple(capture.own_action_support),
            normalized_player_prior=(
                tuple(capture.normalized_player_prior)
                if capture.normalized_player_prior is not None
                else None
            ),
            canonical_particles=tuple(
                SharedRootReplayParticle._from_rust(particle)
                for particle in capture.canonical_particles
            ),
        )


@dataclass(frozen=True)
class SharedRootDiagnostics:
    solver_contract: str
    iterations: int
    continuation_iterations: int
    seed: int
    prior_strength: float
    expected_value: float
    player_best_response_value: float
    opponent_best_response_value: float
    player_best_response_gain: float
    opponent_best_response_gain: float
    nash_conv: float
    exploitability: float
    player_regret_bound: float
    opponent_regret_bound: float
    total_regret_bound: float
    payoff_cells: int
    total_forced_continuation_iterations: int
    input_particle_count: int
    positive_particle_count: int
    canonical_particle_count: int
    normalized_weight_sum: float
    action_support_digest: str
    particle_digest: str
    payoff_digest: str
    player_prior_digest: str
    opponent_prior_digest: str

    @classmethod
    def _from_rust(cls, result):
        return cls(**{name: getattr(result, name) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SharedInformationSetRootResult:
    policy: tuple[SharedRootPolicyEntry, ...]
    opponent_policies: tuple[tuple[tuple[str, float], ...], ...]
    diagnostics: SharedRootDiagnostics
    replay_capture: SharedRootReplayCapture

    @classmethod
    def _from_rust(cls, result):
        return cls(
            policy=tuple(
                SharedRootPolicyEntry(
                    action=entry.action,
                    probability=entry.probability,
                    counterfactual_value=entry.counterfactual_value,
                )
                for entry in result.policy
            ),
            opponent_policies=tuple(
                tuple((action, probability) for action, probability in policy)
                for policy in result.opponent_policies
            ),
            diagnostics=SharedRootDiagnostics._from_rust(result.diagnostics),
            replay_capture=SharedRootReplayCapture._from_rust(result.replay_capture),
        )


def _validate_shared_prior(prior, label):
    if prior is None:
        return None
    if not isinstance(prior, (list, tuple)) or len(prior) > 64:
        raise TypeError(f"{label} must be a bounded sequence")
    result = []
    seen = set()
    for row in prior:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise TypeError(f"{label} entries must contain an action and probability")
        action, probability = row
        if (
            not isinstance(action, str)
            or not action
            or action.lower() in seen
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or probability < 0
        ):
            raise ValueError(f"{label} contains an invalid entry")
        seen.add(action.lower())
        result.append((action, float(probability)))
    if not result or math.fsum(probability for _, probability in result) <= 0:
        raise ValueError(f"{label} must contain positive probability mass")
    return result


def shared_information_set_root_search(
    states: list[State],
    particle_weights: list[float],
    iterations: int,
    continuation_iterations: int,
    seed: int,
    prior_strength: float = 0.0,
    s1_prior=None,
    s2_priors=None,
) -> SharedInformationSetRootResult:
    if not isinstance(states, (list, tuple)) or not 1 <= len(states) <= 64:
        raise ValueError("states must contain between 1 and 64 particles")
    if not isinstance(particle_weights, (list, tuple)) or len(particle_weights) != len(states):
        raise ValueError("particle_weights must match states")
    weights = []
    for weight in particle_weights:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("particle_weights must be finite and nonnegative")
        weights.append(float(weight))
    if abs(math.fsum(weights) - 1.0) > 1e-6:
        raise ValueError("particle_weights must sum to one")
    for name, value, upper in (
        ("iterations", iterations, 1_000_000),
        ("continuation_iterations", continuation_iterations, 1_000_000),
    ):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{name} must be between 1 and {upper}")
    if type(seed) is not int or not 0 <= seed <= (1 << 64) - 1:
        raise ValueError("seed must fit in an unsigned 64-bit integer")
    if isinstance(prior_strength, bool) or not isinstance(prior_strength, (int, float)) or not math.isfinite(prior_strength) or not 0 <= prior_strength <= 1_000:
        raise ValueError("prior_strength must be finite and in [0, 1000]")
    normalized_s1 = _validate_shared_prior(s1_prior, "s1_prior")
    if s2_priors is None:
        normalized_s2 = None
    else:
        if not isinstance(s2_priors, (list, tuple)) or len(s2_priors) != len(states):
            raise ValueError("s2_priors must contain one entry per particle")
        normalized_s2 = [
            _validate_shared_prior(prior, f"s2_priors[{index}]")
            for index, prior in enumerate(s2_priors)
        ]
    result = _rust_shared_information_set_root_search(
        list(states),
        weights,
        iterations,
        continuation_iterations,
        seed,
        float(prior_strength),
        normalized_s1,
        normalized_s2,
    )
    return SharedInformationSetRootResult._from_rust(result)


def paired_root_policy_evaluation(
    state: State,
    baseline_action: str,
    candidate_action: str,
    rollouts: int,
    continuation_iterations: int,
    continuation_steps: int,
    seed: int,
    opponent_priors=None,
) -> PairedRootPolicyEvaluation:
    if not isinstance(baseline_action, str) or not baseline_action:
        raise TypeError("baseline_action must be a nonempty string")
    if not isinstance(candidate_action, str) or not candidate_action:
        raise TypeError("candidate_action must be a nonempty string")
    for name, value in (
        ("rollouts", rollouts),
        ("continuation_iterations", continuation_iterations),
        ("continuation_steps", continuation_steps),
        ("seed", seed),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
    if not 1 <= rollouts <= 100_000:
        raise ValueError("rollouts must be between 1 and 100000")
    if not 1 <= continuation_iterations <= 10_000_000:
        raise ValueError("continuation_iterations must be between 1 and 10000000")
    if not 1 <= continuation_steps <= 100:
        raise ValueError("continuation_steps must be between 1 and 100")
    if 2 * rollouts * continuation_iterations * continuation_steps > 100_000_000:
        raise ValueError("total requested continuation iterations exceed 100000000")
    if not 0 <= seed <= (1 << 64) - 1:
        raise ValueError("seed must fit in an unsigned 64-bit integer")
    return PairedRootPolicyEvaluation._from_rust(
        _rust_paired_root_policy_evaluation(
            state,
            baseline_action,
            candidate_action,
            rollouts,
            continuation_iterations,
            continuation_steps,
            seed,
            opponent_priors,
        )
    )


def monte_carlo_tree_search(
    state: State,
    duration_ms: int = 1000,
    iterations: int = 0,
    threads: int = 1,
    s1_priors=None,
    s2_priors=None,
    c_puct: float = 2.0,
    seed: int | None = None,
) -> MctsResult:
    """
    Perform monte-carlo-tree-search on the given state and for the given duration

    :param state: the state to search through
    :type state: State
    :param duration_ms: time in milliseconds to run the search. ignored if iterations > 0
    :type duration_ms: int
    :param iterations: exact number of monte-carlo iterations to run
    :type iterations: int
    :param threads: number of threads to use for the search
    :type threads: int
    :return: the result of the search
    :rtype: MctsResult
    """
    return MctsResult._from_rust(
        mcts(
            state,
            duration_ms,
            iterations,
            threads,
            s1_priors=s1_priors,
            s2_priors=s2_priors,
            c_puct=c_puct,
            seed=seed,
        )
    )


def iterative_deepening_expectiminimax(
    state: State, duration_ms: int = 1000
) -> IterativeDeepeningResult:
    """
    Perform an iterative-deepening expectiminimax search on the given state and for the given duration

    :param state: the state to search through
    :type state: State
    :param duration_ms: time in milliseconds to run the search
    :type duration_ms: int
    :return: the result of the search
    :rtype: IterativeDeepeningResult
    """
    return IterativeDeepeningResult._from_rust(id(state, duration_ms))
