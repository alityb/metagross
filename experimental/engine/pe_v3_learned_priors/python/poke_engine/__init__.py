from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from . import poke_engine as _native
from .poke_engine import *


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
        )


@dataclass
class SharedRootPolicyEntry:
    action: str
    probability: float
    pulls: int
    estimated_value: float


@dataclass
class SharedRootDiagnostics:
    rounds: int
    continuation_iterations: int
    unique_payoff_cells_evaluated: int
    cache_hits: int
    total_forced_continuation_iterations: int
    world_pulls: list[int]
    elapsed_ms: int
    shared_policy_entropy: float
    shared_policy_max_probability: float
    human_prior_mix: float
    player_prior_mix: float
    player_prior_available: bool
    player_prior_coverage: float
    baseline_action: Optional[str]
    baseline_advantage_available: bool
    baseline_advantage_mean: Optional[float]
    baseline_advantage_standard_error: Optional[float]
    baseline_advantage_lcb: Optional[float]
    baseline_advantage_world_count: int
    baseline_advantage_effective_world_count: float
    lcb_z: float
    paired_evaluation_iterations: int


@dataclass
class SharedInformationSetRootResult:
    policy: list[SharedRootPolicyEntry]
    diagnostics: SharedRootDiagnostics


def shared_information_set_root_search(
    states: list[State],
    world_weights: list[float],
    duration_ms: int = 0,
    rounds: int = 0,
    continuation_iterations: int = 32,
    s1_prior: Optional[list[tuple[str, float]]] = None,
    player_prior_mix: float = 0.25,
    s2_priors: Optional[list[Optional[list[tuple[str, float]]]]] = None,
    human_prior_mix: float = 0.25,
    min_policy_probability: float = 0.02,
    seed: int = 0,
    baseline_action: Optional[str] = None,
    lcb_z: float = 1.645,
    paired_evaluation_iterations: int = 512,
) -> SharedInformationSetRootResult:
    """Find one side-one root policy shared by all positive-weight worlds."""
    result = _native.shared_information_set_root_search(
        states,
        world_weights,
        duration_ms=duration_ms,
        rounds=rounds,
        continuation_iterations=continuation_iterations,
        s1_prior=s1_prior,
        player_prior_mix=player_prior_mix,
        s2_priors=s2_priors,
        human_prior_mix=human_prior_mix,
        min_policy_probability=min_policy_probability,
        seed=seed,
        baseline_action=baseline_action,
        lcb_z=lcb_z,
        paired_evaluation_iterations=paired_evaluation_iterations,
    )
    return SharedInformationSetRootResult(
        policy=[
            SharedRootPolicyEntry(
                action=entry.action,
                probability=entry.probability,
                pulls=entry.pulls,
                estimated_value=entry.estimated_value,
            )
            for entry in result.policy
        ],
        diagnostics=SharedRootDiagnostics(
            rounds=result.diagnostics.rounds,
            continuation_iterations=result.diagnostics.continuation_iterations,
            unique_payoff_cells_evaluated=result.diagnostics.unique_payoff_cells_evaluated,
            cache_hits=result.diagnostics.cache_hits,
            total_forced_continuation_iterations=result.diagnostics.total_forced_continuation_iterations,
            world_pulls=result.diagnostics.world_pulls,
            elapsed_ms=result.diagnostics.elapsed_ms,
            shared_policy_entropy=result.diagnostics.shared_policy_entropy,
            shared_policy_max_probability=result.diagnostics.shared_policy_max_probability,
            human_prior_mix=result.diagnostics.human_prior_mix,
            player_prior_mix=result.diagnostics.player_prior_mix,
            player_prior_available=result.diagnostics.player_prior_available,
            player_prior_coverage=result.diagnostics.player_prior_coverage,
            baseline_action=result.diagnostics.baseline_action,
            baseline_advantage_available=result.diagnostics.baseline_advantage_available,
            baseline_advantage_mean=result.diagnostics.baseline_advantage_mean,
            baseline_advantage_standard_error=result.diagnostics.baseline_advantage_standard_error,
            baseline_advantage_lcb=result.diagnostics.baseline_advantage_lcb,
            baseline_advantage_world_count=result.diagnostics.baseline_advantage_world_count,
            baseline_advantage_effective_world_count=result.diagnostics.baseline_advantage_effective_world_count,
            lcb_z=result.diagnostics.lcb_z,
            paired_evaluation_iterations=result.diagnostics.paired_evaluation_iterations,
        ),
    )


def monte_carlo_tree_search(
    state: State,
    duration_ms: int = 1000,
    iterations: int = 0,
    threads: int = 1,
    s1_priors=None,
    s2_priors=None,
    c_puct: float = 2.0,
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
