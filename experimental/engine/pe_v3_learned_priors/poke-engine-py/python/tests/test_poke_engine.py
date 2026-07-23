import pytest

from poke_engine import (
    State,
    Side,
    Move,
    Pokemon,
    monte_carlo_tree_search,
    shared_information_set_root_search,
    generate_instructions,
    calculate_damage,
    iterative_deepening_expectiminimax,
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
)


def test_state_can_be_converted_to_and_from_a_string():
    serialized = state.to_string()
    State.from_string(serialized)
    serialized_again = state.to_string()
    assert serialized == serialized_again


def test_monte_carlo_search():
    monte_carlo_tree_search(state, 10)


def test_monte_carlo_iteration_limit_is_exact_below_batch_size():
    assert monte_carlo_tree_search(state, duration_ms=0, iterations=7).total_visits == 7


def test_shared_information_set_root_search_smoke_preserves_state():
    serialized = state.to_string()
    result = shared_information_set_root_search(
        [state, State.from_string(serialized)],
        [1.0, 1.0],
        rounds=2,
        continuation_iterations=4,
        s1_prior=[("WATERGUN", 1.0), ("not_an_action", 99.0)],
        s2_priors=[[("ember", 1.0)], None],
        seed=3,
    )

    assert result.diagnostics.rounds == 2
    assert sum(result.diagnostics.world_pulls) == 2
    assert result.diagnostics.unique_payoff_cells_evaluated > 0
    assert result.diagnostics.cache_hits >= 0
    assert result.diagnostics.total_forced_continuation_iterations == (
        result.diagnostics.unique_payoff_cells_evaluated * 4
    )
    assert all(entry.pulls == 2 for entry in result.policy)
    assert sum(entry.probability for entry in result.policy) == pytest.approx(1.0)
    assert all(entry.probability == entry.probability for entry in result.policy)
    assert result.diagnostics.shared_policy_entropy >= 0.0
    assert result.diagnostics.shared_policy_max_probability == pytest.approx(
        max(entry.probability for entry in result.policy)
    )
    assert result.diagnostics.human_prior_mix == pytest.approx(0.25)
    assert result.diagnostics.player_prior_mix == pytest.approx(0.25)
    assert result.diagnostics.player_prior_available is True
    assert 0.0 < result.diagnostics.player_prior_coverage <= 1.0
    assert result.diagnostics.baseline_action is None
    assert result.diagnostics.baseline_advantage_available is False
    assert result.diagnostics.baseline_advantage_mean is None
    assert result.diagnostics.lcb_z == pytest.approx(1.645)
    assert result.diagnostics.paired_evaluation_iterations == 512
    assert state.to_string() == serialized


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("human_prior_mix", -0.01),
        ("human_prior_mix", 1.01),
        ("player_prior_mix", -0.01),
        ("player_prior_mix", 1.01),
        ("min_policy_probability", -0.01),
        ("min_policy_probability", 1.01),
    ],
)
def test_shared_information_set_root_search_rejects_invalid_probabilities(
    argument, value
):
    with pytest.raises(ValueError):
        shared_information_set_root_search(
            [state],
            [1.0],
            rounds=1,
            continuation_iterations=1,
            **{argument: value},
        )


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_shared_information_set_root_search_rejects_invalid_player_prior_values(value):
    with pytest.raises(ValueError):
        shared_information_set_root_search(
            [state],
            [1.0],
            rounds=1,
            continuation_iterations=1,
            s1_prior=[("watergun", value)],
        )


def test_shared_information_set_root_search_ignores_unknown_player_prior_actions():
    result = shared_information_set_root_search(
        [state],
        [1.0],
        rounds=2,
        continuation_iterations=1,
        s1_prior=[("not_an_action", 1.0)],
    )

    assert result.diagnostics.player_prior_available is False
    assert result.diagnostics.player_prior_coverage == 0.0


def test_shared_information_set_root_search_exposes_baseline_diagnostics():
    result = shared_information_set_root_search(
        [state],
        [5.0],
        rounds=1,
        continuation_iterations=1,
        baseline_action="WATERGUN",
        lcb_z=0.0,
        paired_evaluation_iterations=256,
    )

    diagnostics = result.diagnostics
    assert diagnostics.baseline_action == "watergun"
    assert diagnostics.baseline_advantage_available is True
    assert diagnostics.baseline_advantage_mean is not None
    assert diagnostics.baseline_advantage_standard_error == 0.0
    assert diagnostics.baseline_advantage_lcb == pytest.approx(
        diagnostics.baseline_advantage_mean
    )
    assert diagnostics.baseline_advantage_world_count == 1
    assert diagnostics.baseline_advantage_effective_world_count == 1.0
    assert diagnostics.paired_evaluation_iterations == 256


@pytest.mark.parametrize("lcb_z", [-0.1, float("nan"), float("inf")])
def test_shared_information_set_root_search_rejects_invalid_lcb_z(lcb_z):
    with pytest.raises(ValueError):
        shared_information_set_root_search(
            [state],
            [1.0],
            rounds=1,
            continuation_iterations=1,
            lcb_z=lcb_z,
        )


def test_shared_information_set_root_search_rejects_unknown_baseline():
    with pytest.raises(ValueError):
        shared_information_set_root_search(
            [state],
            [1.0],
            rounds=1,
            continuation_iterations=1,
            baseline_action="not_an_action",
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
