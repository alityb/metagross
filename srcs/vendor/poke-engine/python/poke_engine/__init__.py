from dataclasses import dataclass
from enum import StrEnum

from .poke_engine import *

_rust_paired_root_policy_evaluation = paired_root_policy_evaluation


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
