import pytest

from poke_engine import (
    State,
    Side,
    Move,
    Pokemon,
    monte_carlo_tree_search,
    generate_instructions,
    calculate_damage,
    iterative_deepening_expectiminimax,
    paired_root_policy_evaluation,
    shared_information_set_root_search,
    Weather,
    Terrain,
)

state = State(
    side_one=Side(
        pokemon=[
            Pokemon(
                id="squirtle",
                level=100,
                types=("water", "typeless"),
                base_types=("water", "rock"),
                hp=100,
                maxhp=100,
                attack=100,
                defense=100,
                special_attack=100,
                special_defense=100,
                speed=100,
                status="none",
                moves=[
                    Move(id="watergun", pp=32),
                    Move(id="tackle", pp=32),
                    Move(id="quickattack", pp=32),
                    Move(id="leer", pp=32),
                ],
            ),
        ]
    ),
    side_two=Side(
        pokemon=[
            Pokemon(
                id="charmander",
                level=100,
                types=("fire", "typeless"),
                hp=100,
                maxhp=100,
                attack=100,
                defense=100,
                special_attack=100,
                special_defense=100,
                speed=100,
                status="none",
                moves=[
                    Move(id="ember", pp=32),
                    Move(id="tackle", pp=32),
                    Move(id="quickattack", pp=32),
                    Move(id="leer", pp=32),
                ],
            ),
        ]
    ),
    weather="none",
    weather_turns_remaining=-1,
    terrain="none",
    terrain_turns_remaining=-1,
    trick_room=False,
    trick_room_turns_remaining=-1,
    s1_can_tera=False,
    s2_can_tera=False,
)


def test_state_can_be_converted_to_and_from_a_string():
    serialized = state.to_string()
    State.from_string(serialized)
    serialized_again = state.to_string()
    assert serialized == serialized_again


def test_monte_carlo_search():
    monte_carlo_tree_search(state, 10)


def test_monte_carlo_search_uses_root_priors():
    result = monte_carlo_tree_search(
        state,
        duration_ms=0,
        iterations=128,
        s1_priors=[("leer", 1.0)],
        c_puct=100.0,
    )
    visits = {entry.move_choice: entry.visits for entry in result.side_one}
    assert visits["leer"] > max(
        visits[move] for move in visits if move != "leer"
    )


def test_root_priors_normalize_matches_without_suppressing_omitted_options():
    result = monte_carlo_tree_search(
        state,
        duration_ms=0,
        iterations=1000,
        s1_priors=[("leer", 0.01), ("watergun", 0.0), ("unknown", 0.99)],
        c_puct=100.0,
    )
    visits = {entry.move_choice: entry.visits for entry in result.side_one}
    assert visits["leer"] > 900
    assert all(visits[move] > 1 for move in visits if move != "leer")


@pytest.mark.parametrize(
    "priors",
    [
        [],
        [("leer", 0.0)],
        [("leer", 0.5), ("leer", 0.5)],
        [("leer", -0.1)],
        [("leer", 1.1)],
        [("leer", float("nan"))],
        [("leer", float("inf"))],
    ],
)
def test_monte_carlo_search_rejects_invalid_root_priors(priors):
    with pytest.raises(ValueError):
        monte_carlo_tree_search(state, duration_ms=0, iterations=1, s1_priors=priors)


def test_monte_carlo_search_rejects_priors_with_multiple_threads():
    with pytest.raises(ValueError, match="single-threaded"):
        monte_carlo_tree_search(
            state,
            duration_ms=0,
            iterations=1,
            threads=2,
            s1_priors=[("leer", 1.0)],
        )


def test_iterative_deepening_search():
    iterative_deepening_expectiminimax(state, 10)


def test_get_instructions():
    generate_instructions(state, "watergun", "ember")


def test_calculate_damage():
    calculate_damage(state, "watergun", "ember", True)


def test_generate_instructions_errors_when_move_does_not_exist():
    with pytest.raises(ValueError):
        generate_instructions(state, "not_a_move", "ember")


def test_paired_root_policy_evaluation_is_deterministic_and_aggregate_only():
    first = paired_root_policy_evaluation(state, "watergun", "tackle", 2, 3, 1, 7)
    second = paired_root_policy_evaluation(state, "watergun", "tackle", 2, 3, 1, 7)

    assert first == second
    assert first.pairs == 2
    assert first.continuation_iterations_executed <= 12
    assert first.catastrophic_count == first.candidate_catastrophic_count
    assert first.baseline_terminal_count + first.baseline_nonterminal_count == 2
    assert first.candidate_terminal_count + first.candidate_nonterminal_count == 2
    assert (
        0.5 * first.candidate_catastrophic_count
        <= first.candidate_catastrophic_severity_sum
        <= first.candidate_catastrophic_count
    )
    assert (
        0.5 * first.baseline_catastrophic_count
        <= first.baseline_catastrophic_severity_sum
        <= first.baseline_catastrophic_count
    )
    assert not hasattr(first, "outcomes")


def test_paired_root_policy_evaluation_arm_swap_is_symmetric():
    result = paired_root_policy_evaluation(state, "watergun", "tackle", 4, 3, 1, 19)
    swapped = paired_root_policy_evaluation(state, "tackle", "watergun", 4, 3, 1, 19)

    assert result.baseline_sum == swapped.candidate_sum
    assert result.candidate_sum == swapped.baseline_sum
    assert result.delta_sum == -swapped.delta_sum
    assert result.delta_squared_sum == swapped.delta_squared_sum
    assert result.candidate_catastrophic_count == swapped.baseline_catastrophic_count
    assert result.baseline_catastrophic_count == swapped.candidate_catastrophic_count
    assert (
        result.candidate_catastrophic_severity_sum
        == swapped.baseline_catastrophic_severity_sum
    )
    assert (
        result.baseline_catastrophic_severity_sum
        == swapped.candidate_catastrophic_severity_sum
    )
    assert (
        result.baseline_nonterminal_evaluation_delta_sum
        == swapped.candidate_nonterminal_evaluation_delta_sum
    )
    assert (
        result.candidate_nonterminal_evaluation_delta_sum
        == swapped.baseline_nonterminal_evaluation_delta_sum
    )


def test_paired_root_opponent_priors_are_deterministic_and_compatible():
    legacy = paired_root_policy_evaluation(state, "watergun", "tackle", 8, 3, 1, 31)
    explicit_none = paired_root_policy_evaluation(
        state, "watergun", "tackle", 8, 3, 1, 31, None
    )
    unmatched = paired_root_policy_evaluation(
        state, "watergun", "tackle", 8, 3, 1, 31, [("missing", 1.0)]
    )
    ember = paired_root_policy_evaluation(
        state, "watergun", "tackle", 8, 3, 1, 31, [("ember", 1.0)]
    )
    ember_again = paired_root_policy_evaluation(
        state, "watergun", "tackle", 8, 3, 1, 31, [("ember", 1.0)]
    )
    leer = paired_root_policy_evaluation(
        state, "watergun", "tackle", 8, 3, 1, 31, [("leer", 1.0)]
    )

    assert legacy == explicit_none
    assert legacy == unmatched
    assert ember == ember_again
    assert ember != leer


@pytest.mark.parametrize(
    "priors",
    [
        [("ember", 0.0)],
        [("ember", -0.1)],
        [("ember", 1.1)],
        [("ember", float("nan"))],
        [("ember", 0.5), ("ember", 0.5)],
    ],
)
def test_paired_root_opponent_priors_reject_invalid_entries(priors):
    with pytest.raises(ValueError, match="opponent priors"):
        paired_root_policy_evaluation(
            state, "watergun", "tackle", 1, 1, 1, 0, priors
        )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("rollouts", True, TypeError),
        ("rollouts", 0, ValueError),
        ("continuation_iterations", -1, ValueError),
        ("continuation_steps", 101, ValueError),
        ("seed", -1, ValueError),
    ],
)
def test_paired_root_policy_evaluation_rejects_malformed_bounds(field, value, error):
    arguments = {
        "rollouts": 1,
        "continuation_iterations": 1,
        "continuation_steps": 1,
        "seed": 0,
    }
    arguments[field] = value
    with pytest.raises(error):
        paired_root_policy_evaluation(state, "watergun", "tackle", **arguments)


def test_shared_root_search_is_reproducible_complete_and_nonmutating():
    before = state.to_string()
    first = shared_information_set_root_search(
        [state], [1.0], 2_000, 16, 41, s1_prior=[("watergun", 1.0)]
    )
    second = shared_information_set_root_search(
        [state], [1.0], 2_000, 16, 41, s1_prior=[("watergun", 1.0)]
    )

    assert first == second
    assert state.to_string() == before
    assert sum(entry.probability for entry in first.policy) == pytest.approx(1.0)
    assert {entry.action for entry in first.policy} == {
        "watergun",
        "tackle",
        "quickattack",
        "leer",
    }
    assert all(entry.probability >= 0 for entry in first.policy)
    assert first.diagnostics.solver_contract == "weighted-shared-rm-plus-v1"
    assert first.diagnostics.iterations == 2_000
    assert first.diagnostics.seed == 41
    assert first.diagnostics.payoff_cells == 16
    assert first.diagnostics.nash_conv == pytest.approx(
        first.diagnostics.player_best_response_gain
        + first.diagnostics.opponent_best_response_gain
    )
    assert first.diagnostics.exploitability == pytest.approx(
        first.diagnostics.nash_conv / 2
    )
    assert first.diagnostics.total_regret_bound == pytest.approx(
        first.diagnostics.player_regret_bound
        + first.diagnostics.opponent_regret_bound
    )
    assert len(first.opponent_policies) == 1
    assert sum(probability for _, probability in first.opponent_policies[0]) == pytest.approx(1.0)
    capture = first.replay_capture
    assert capture.schema_version == 1
    assert capture.configuration.iterations == 2_000
    assert capture.configuration.continuation_iterations == 16
    assert capture.configuration.seed == 41
    assert capture.own_action_support == tuple(sorted(entry.action for entry in first.policy))
    assert capture.normalized_player_prior == (0.0, 0.0, 0.0, 1.0)
    assert len(capture.canonical_particles) == 1
    particle = capture.canonical_particles[0]
    assert particle.state == before
    assert particle.normalized_weight == 1.0
    assert particle.source_particles[0].input_index == 0
    assert len(particle.payoff_matrix) == len(capture.own_action_support)
    assert len(particle.opponent_policy) == len(particle.opponent_action_support)
    for payoff_row, continuation_row in zip(
        particle.payoff_matrix, particle.continuations, strict=True
    ):
        assert len(payoff_row) == len(particle.opponent_action_support)
        for payoff, continuation in zip(payoff_row, continuation_row, strict=True):
            assert continuation.payoff == payoff
            assert continuation.requested_iterations == 16
            assert continuation.executed_iterations == 16
            assert continuation.visits == 16


def test_shared_root_search_merges_split_particles_and_ignores_zero_weight():
    single = shared_information_set_root_search([state], [1.0], 1_000, 8, 7)
    split = shared_information_set_root_search(
        [state, State.from_string(state.to_string())], [0.25, 0.75], 1_000, 8, 7
    )
    zero = shared_information_set_root_search(
        [state, State()], [1.0, 0.0], 1_000, 8, 7
    )

    assert single.policy == split.policy == zero.policy
    assert split.diagnostics.input_particle_count == 2
    assert split.diagnostics.positive_particle_count == 2
    assert split.diagnostics.canonical_particle_count == 1
    assert zero.diagnostics.positive_particle_count == 1
    assert [
        (source.input_index, source.input_weight)
        for source in split.replay_capture.canonical_particles[0].source_particles
    ] == [(0, 0.25), (1, 0.75)]
    assert [
        (source.input_index, source.input_weight)
        for source in zero.replay_capture.canonical_particles[0].source_particles
    ] == [(0, 1.0)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"particle_weights": [0.9]},
        {"particle_weights": [-1.0]},
        {"iterations": 0},
        {"continuation_iterations": 0},
        {"seed": -1},
        {"prior_strength": float("nan")},
        {"s1_prior": [("watergun", 0.5), ("watergun", 0.5)]},
        {"s2_priors": []},
    ],
)
def test_shared_root_search_rejects_invalid_contract(kwargs):
    arguments = {
        "states": [state],
        "particle_weights": [1.0],
        "iterations": 10,
        "continuation_iterations": 1,
        "seed": 0,
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        shared_information_set_root_search(**arguments)
