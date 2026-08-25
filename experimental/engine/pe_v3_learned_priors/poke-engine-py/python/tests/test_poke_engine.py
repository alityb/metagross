import pytest
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(WORKSPACE_ROOT / "experimental" / "src"))

from poke_engine import (
    State,
    Side,
    Move,
    Pokemon,
    monte_carlo_tree_search,
    monte_carlo_tree_search_with_s1_request,
    root_options,
    root_options_with_s1_request,
    r1_semantic_contract,
    step_with_uniform,
    step_with_uniform_debug,
    step_with_uniform_r1_semantic,
    step_with_uniform_r1_semantic_s1_request,
    shared_information_set_root_search,
    terminal_value,
    transition_debug_contract,
    generate_instructions,
    calculate_damage,
    compute_resource_features,
    iterative_deepening_expectiminimax,
    Weather,
    Terrain,
)
from scripts.r1_public_events import (
    BOOSTED_DOUBLE_SWITCH_CERTIFICATE,
    DECLARATIVE_BOOST_CERTIFICATE,
    LEFTOVERS_ACTIVATION_CERTIFICATE,
    MIXED_BOOST_SWITCH_CERTIFICATE,
    SILENT_MECHANICS_CERTIFICATE,
    private_basic_move_diagnostic,
    project_information_set_basic_move,
    project_information_set_switch,
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


def test_public_reveal_mask_reaches_interior_resource_features():
    def revealed_mon(species, move, speed):
        return Pokemon(
            id=species,
            hp=100,
            maxhp=100,
            attack=100,
            defense=100,
            special_attack=100,
            special_defense=100,
            speed=speed,
            moves=[Move(id=move), Move(id="protect"), Move(id="rest"), Move(id="leer")],
        )

    root = State(
        side_one=Side(pokemon=[revealed_mon("squirtle", "watergun", 50)]),
        side_two=Side(pokemon=[revealed_mon("charmander", "ember", 200)]),
        s1_public_reveals=1,
    )
    root_information = compute_resource_features(root)[16:20]

    child = step_with_uniform_r1_semantic(root, "watergun", "ember", 0.5).state
    child_information = compute_resource_features(child)[16:20]

    assert root.s1_public_reveals == 1
    assert child.s1_public_reveals == 65
    assert root_information == pytest.approx([1 / 6, 0.0, 0.0, 0.0])
    assert child_information == pytest.approx([1 / 6, 1 / 24, 0.0, 0.0])


def test_root_options_returns_canonical_actions_without_mutating_state():
    option_state = State(
        side_one=Side(
            pokemon=[
                state.side_one.pokemon[0],
                Pokemon(id="bulbasaur", moves=[Move(id="tackle")]),
            ]
        ),
        side_two=Side(
            pokemon=[
                state.side_two.pokemon[0],
                Pokemon(id="pikachu", moves=[Move(id="tackle")]),
            ]
        ),
    )
    serialized = option_state.to_string()

    side_one, side_two = root_options(state=option_state)

    assert {"watergun", "tackle", "quickattack", "leer", "switch bulbasaur"} <= set(
        side_one
    )
    assert {"ember", "tackle", "quickattack", "leer", "switch pikachu"} <= set(
        side_two
    )
    assert root_options(state=option_state) == (side_one, side_two)
    assert option_state.to_string() == serialized


def test_terminal_value_uses_side_one_battle_outcome():
    side_one_win = State(
        side_one=Side(pokemon=[state.side_one.pokemon[0]]),
        side_two=Side(pokemon=[]),
    )
    side_one_loss = State(
        side_one=Side(pokemon=[]),
        side_two=Side(pokemon=[state.side_two.pokemon[0]]),
    )

    assert terminal_value(state) == 0.0
    assert terminal_value(side_one_win) == 1.0
    assert terminal_value(side_one_loss) == -1.0


def test_transition_debug_contract_is_pinned():
    assert transition_debug_contract() == "poke-engine-0.0.47-r1-switch-v1"


def test_r1_semantic_contract_is_pinned():
    assert r1_semantic_contract() == "poke-engine-0.0.47-r1-item-activation-v1"


def _semantic_mon(name, move, speed=100, level=100):
    return Pokemon(
        id=name,
        level=level,
        hp=100,
        maxhp=100,
        speed=speed,
        moves=[
            Move(id=move),
            Move(id="protect"),
            Move(id="rest"),
            Move(id="sleeptalk"),
        ],
    )


def test_r1_semantic_trace_orders_executed_actions_and_damage():
    semantic_state = State(
        side_one=Side(pokemon=[_semantic_mon("pikachu", "tackle", 200)]),
        side_two=Side(pokemon=[_semantic_mon("charmander", "tackle", 50)]),
    )
    serialized = semantic_state.to_string()

    result = step_with_uniform_r1_semantic(
        semantic_state, "tackle", "tackle", 0.5
    )

    assert [event.kind for event in result.events] == [
        "action_executed", "damage", "action_executed", "damage"
    ]
    assert [event.side for event in result.events] == [
        "side_one", "side_two", "side_two", "side_one"
    ]
    assert [event.move_id for event in result.events if event.kind == "action_executed"] == [
        "TACKLE", "TACKLE"
    ]
    assert result.unaccounted_instruction_kinds == []
    assert semantic_state.to_string() == serialized
    assert all(
        not repr(instruction).startswith("RecordAction")
        for outcome in generate_instructions(semantic_state, "tackle", "tackle")
        for instruction in outcome.instruction_list
    )


def test_r1_semantic_trace_omits_unexecuted_action_after_ko():
    semantic_state = State(
        side_one=Side(
            pokemon=[_semantic_mon("pikachu", "seismictoss", 200, level=100)]
        ),
        side_two=Side(pokemon=[_semantic_mon("charmander", "tackle", 50)]),
    )

    result = step_with_uniform_r1_semantic(
        semantic_state, "seismictoss", "tackle", 0.5
    )

    assert [(event.kind, event.side, event.move_id) for event in result.events] == [
        ("action_executed", "side_one", "SEISMICTOSS"),
        ("damage", "side_two", None),
    ]
    assert result.state.side_two.pokemon[0].hp == 0


@pytest.mark.parametrize(
    ("move", "event_kind", "detail", "amount"),
    [
        ("thunderwave", "status_changed", "PARALYZE", None),
        ("swordsdance", "boost_changed", "attack", 2),
    ],
)
def test_r1_semantic_trace_exposes_status_and_boost_changes(
    move, event_kind, detail, amount
):
    semantic_state = State(
        side_one=Side(pokemon=[_semantic_mon("pikachu", move, 200)]),
        side_two=Side(pokemon=[_semantic_mon("charmander", "tackle", 50)]),
    )

    result = step_with_uniform_r1_semantic(
        semantic_state, move, "tackle", 0.99
    )
    event = next(event for event in result.events if event.kind == event_kind)

    assert event.detail == detail
    assert event.amount == amount
    assert result.unaccounted_instruction_kinds == []


def test_basic_move_projector_merges_hidden_equivalents_and_partitions_public_hp():
    def world(defense):
        return State(
            side_one=Side(
                pokemon=[_semantic_mon("pikachu", "tackle", 200)]
            ),
            side_two=Side(
                pokemon=[
                    Pokemon(
                        id="charmander",
                        hp=100,
                        maxhp=100,
                        defense=defense,
                        speed=50,
                        ability="none",
                        item="none",
                        moves=[
                            Move(id="swordsdance"),
                            Move(id="recover"),
                            Move(id="tackle"),
                            Move(id="thunderwave"),
                        ],
                    )
                ]
            ),
        )

    same_public = project_information_set_basic_move(
        sys.modules["poke_engine"],
        [world(100), world(100)],
        "tackle",
        "swordsdance",
        0.5,
    )
    different_public = project_information_set_basic_move(
        sys.modules["poke_engine"],
        [world(50), world(200)],
        "tackle",
        "swordsdance",
        0.5,
    )

    assert len(same_public.observation_classes) == 1
    assert len(same_public.observation_classes[0].next_states) == 2
    assert len(different_public.observation_classes) == 2
    serialized = repr(different_public.observation_classes)
    assert "defense" not in serialized
    assert "branch_probability" not in serialized


def _silent_mechanics_state(hp=100, hidden_defense=100):
    def pokemon(name, ability, move, speed):
        return Pokemon(
            id=name,
            hp=hp if name == "ragingbolt" else 100,
            maxhp=100,
            speed=speed,
            defense=hidden_defense if name == "goodra" else 100,
            item="leftovers",
            ability=ability,
            moves=[
                Move(id=move, pp=32),
                Move(id="recover"),
                Move(id="tackle"),
                Move(id="thunderwave"),
            ],
        )

    return State(
        side_one=Side(
            pokemon=[pokemon("ragingbolt", "protosynthesis", "calmmind", 200)],
            special_attack_boost=1,
            special_defense_boost=1,
        ),
        side_two=Side(
            pokemon=[pokemon("goodra", "sapsipper", "bulkup", 50)]
        ),
    )


def test_silent_item_ability_subset_preserves_boosts_without_revealing_identity():
    projection = project_information_set_basic_move(
        sys.modules["poke_engine"],
        [
            _silent_mechanics_state(hidden_defense=50),
            _silent_mechanics_state(hidden_defense=200),
        ],
        "calmmind",
        "bulkup",
        0.5,
    )

    assert projection.certificate == SILENT_MECHANICS_CERTIFICATE
    assert len(projection.observation_classes) == 1
    assert len(projection.observation_classes[0].next_states) == 2
    assert [event.kind for event in projection.observation_classes[0].events] == [
        "move",
        "boost",
        "boost",
        "move",
        "boost",
        "boost",
    ]
    serialized = repr(projection.observation_classes[0].events)
    assert "leftovers" not in serialized
    assert "protosynthesis" not in serialized
    assert "sapsipper" not in serialized


def test_leftovers_activation_has_typed_public_cause():
    projection = project_information_set_basic_move(
        sys.modules["poke_engine"],
        [_silent_mechanics_state(hp=50)],
        "calmmind",
        "bulkup",
        0.5,
    )

    assert projection.certificate == LEFTOVERS_ACTIVATION_CERTIFICATE
    events = projection.observation_classes[0].events
    assert [(event.kind, event.actor) for event in events[-2:]] == [
        ("item", "self"),
        ("hp", "self"),
    ]
    assert events[-2].item_id == "leftovers"
    assert all(
        not repr(instruction).startswith("RecordItemActivation")
        for outcome in generate_instructions(
            _silent_mechanics_state(hp=50), "calmmind", "bulkup"
        )
        for instruction in outcome.instruction_list
    )


def _declarative_boost_state(item, ability):
    def pokemon(name, selected_move, speed, held_item, active_ability):
        return Pokemon(
            id=name,
            hp=100,
            maxhp=100,
            speed=speed,
            item=held_item,
            ability=active_ability,
            moves=[
                Move(id=selected_move, pp=32),
                Move(id="recover"),
                Move(id="tackle"),
                Move(id="thunderwave"),
            ],
        )

    return State(
        side_one=Side(
            pokemon=[
                pokemon(
                    "ragingbolt", "calmmind", 200, "leftovers", "protosynthesis"
                )
            ]
        ),
        side_two=Side(
            pokemon=[pokemon("thundurus", "nastyplot", 50, item, ability)]
        ),
    )


def test_declarative_boost_rules_merge_silent_hidden_mechanics():
    projection = project_information_set_basic_move(
        sys.modules["poke_engine"],
        [
            _declarative_boost_state("heavydutyboots", "defiant"),
            _declarative_boost_state("widelens", "technician"),
        ],
        "calmmind",
        "nastyplot",
        0.5,
    )

    assert projection.certificate == DECLARATIVE_BOOST_CERTIFICATE
    assert len(projection.observation_classes) == 1
    assert len(projection.observation_classes[0].next_states) == 2
    serialized = repr(projection.observation_classes[0].events)
    for hidden_identity in ("heavydutyboots", "defiant", "widelens", "technician"):
        assert hidden_identity not in serialized


def test_declarative_boost_rules_reject_unregistered_ability_activation():
    assert private_basic_move_diagnostic(
        sys.modules["poke_engine"],
        _declarative_boost_state("heavydutyboots", "speedboost"),
        "calmmind",
        "nastyplot",
        0.5,
    ) == "UNSUPPORTED_MECHANIC_ACTIVATION"


def _mixed_boost_switch_state(
    hidden_move, target="munkidori", target_ability="none"
):
    return State(
        side_one=Side(
            pokemon=[
                Pokemon(
                    id="ragingbolt",
                    hp=100,
                    maxhp=100,
                    speed=200,
                    item="leftovers",
                    ability="protosynthesis",
                    moves=[Move(id="calmmind", pp=32)],
                ),
                Pokemon(
                    id="bulbasaur",
                    moves=[
                        Move(id="gigadrain"),
                        Move(id="protect"),
                        Move(id="sludgebomb"),
                        Move(id="synthesis"),
                    ],
                ),
            ]
        ),
        side_two=Side(
            pokemon=[
                Pokemon(
                    id="squirtle",
                    moves=[
                        Move(id="sludgewave"),
                        Move(id="psychic"),
                        Move(id="uturn"),
                        Move(id="toxic"),
                    ],
                ),
                Pokemon(
                    id=target,
                    level=100,
                    hp=100,
                    maxhp=100,
                    ability=target_ability,
                    moves=[
                        Move(id=hidden_move),
                        Move(id="protect"),
                        Move(id="rest"),
                        Move(id="sleeptalk"),
                    ],
                ),
            ]
        ),
    )


def test_mixed_boost_switch_projects_known_public_target_without_hidden_moves():
    projection = project_information_set_basic_move(
        sys.modules["poke_engine"],
        [
            _mixed_boost_switch_state("psychic"),
            _mixed_boost_switch_state("thunderbolt"),
        ],
        "calmmind",
        "switch munkidori",
        0.5,
        public_opponent={
            "munkidori": {
                "level": 100,
                "hp_fraction": 1.0,
                "status": "nostatus",
            }
        },
    )

    assert projection.certificate == MIXED_BOOST_SWITCH_CERTIFICATE
    assert len(projection.observation_classes) == 1
    assert len(projection.observation_classes[0].next_states) == 2
    events = projection.observation_classes[0].events
    assert [event.kind for event in events] == ["switch", "move", "boost", "boost"]
    assert events[0].species == "munkidori"
    assert "psychic" not in repr(events)
    assert "thunderbolt" not in repr(events)


def test_mixed_boost_switch_certifies_trapping_legality_without_revealing_ability():
    projection = project_information_set_basic_move(
        sys.modules["poke_engine"],
        [_mixed_boost_switch_state(
            "psychic", target="gothitelle", target_ability="shadowtag"
        )],
        "calmmind",
        "switch gothitelle",
        0.5,
        public_opponent={
            "gothitelle": {
                "level": 100,
                "hp_fraction": 1.0,
                "status": "nostatus",
            }
        },
    )
    observation_class = projection.observation_classes[0]
    assert all(not action.startswith("switch ") for action in observation_class.legal_actions)
    assert "shadowtag" not in repr(observation_class.events)


def test_mixed_boost_switch_partitions_worlds_by_public_next_legality():
    projection = project_information_set_basic_move(
        sys.modules["poke_engine"],
        [
            _mixed_boost_switch_state(
                "psychic", target="gothitelle", target_ability="shadowtag"
            ),
            _mixed_boost_switch_state(
                "psychic", target="gothitelle", target_ability="competitive"
            ),
        ],
        "calmmind",
        "switch gothitelle",
        0.5,
        public_opponent={
            "gothitelle": {
                "level": 100,
                "hp_fraction": 1.0,
                "status": "nostatus",
            }
        },
    )
    assert len(projection.observation_classes) == 2
    signatures = {
        any(action.startswith("switch ") for action in observation_class.legal_actions)
        for observation_class in projection.observation_classes
    }
    assert signatures == {False, True}
    assert all(
        "shadowtag" not in repr(observation_class.events)
        and "competitive" not in repr(observation_class.events)
        for observation_class in projection.observation_classes
    )


def test_mixed_boost_switch_rejects_public_hp_mismatch():
    state = _mixed_boost_switch_state("psychic", target="thundurus")
    assert private_basic_move_diagnostic(
        sys.modules["poke_engine"],
        state,
        "calmmind",
        "switch thundurus",
        0.5,
        public_opponent={
            "thundurus": {
                "level": 100,
                "hp_fraction": 0.5,
                "status": "nostatus",
            }
        },
    ) == "UNSUPPORTED_ACTION_PAIR"


def _four_moves(first):
    return [
        Move(id=first),
        Move(id="protect"),
        Move(id="rest"),
        Move(id="sleeptalk"),
    ]


def test_boosted_double_switch_suppresses_cleanup_and_certifies_legality():
    worlds = [
        State(
            side_one=Side(
                pokemon=[
                    Pokemon(id="ragingbolt", moves=_four_moves("calmmind")),
                    Pokemon(id="bulbasaur", moves=_four_moves("gigadrain")),
                ],
                special_attack_boost=2,
                special_defense_boost=2,
            ),
            side_two=Side(
                pokemon=[
                    Pokemon(id="squirtle", moves=_four_moves("tackle")),
                    Pokemon(
                        id="gothitelle",
                        ability="shadowtag",
                        moves=_four_moves(hidden_move),
                    ),
                ]
            ),
        )
        for hidden_move in ("psychic", "thunderbolt")
    ]
    projection = project_information_set_switch(
        sys.modules["poke_engine"],
        worlds,
        "switch bulbasaur",
        "switch gothitelle",
        0.5,
        public_opponent={
            "gothitelle": {
                "level": 50,
                "hp_fraction": 1.0,
                "status": "nostatus",
            }
        },
    )

    assert projection.certificate == BOOSTED_DOUBLE_SWITCH_CERTIFICATE
    assert [event.kind for event in projection.events] == ["switch", "switch"]
    assert all(not action.startswith("switch ") for action in projection.legal_actions)
    assert projection.cleared_self_boosts == (
        ("specialattack", 2),
        ("specialdefense", 2),
    )
    assert "psychic" not in repr(projection.events)
    assert "thunderbolt" not in repr(projection.events)


def test_step_with_uniform_selects_normalized_branch_without_mutating_state():
    serialized = state.to_string()
    outcomes = generate_instructions(state, "watergun", "ember")
    total = sum(outcome.percentage for outcome in outcomes)
    u = 0.314159
    target = u * total
    cumulative = 0.0
    expected_index = None
    for index, outcome in enumerate(outcomes):
        cumulative += outcome.percentage
        if target < cumulative:
            expected_index = index
            break
    assert expected_index is not None

    next_state, branch_index, branch_probability = step_with_uniform(
        state, "watergun", "ember", u
    )
    repeated_state, repeated_index, repeated_probability = step_with_uniform(
        state=state,
        side_one_action="watergun",
        side_two_action="ember",
        u=u,
    )

    expected_state = state.apply_instructions(outcomes[expected_index])
    assert branch_index == expected_index
    assert branch_probability == pytest.approx(
        outcomes[expected_index].percentage / total
    )
    assert next_state.to_string() == expected_state.to_string()
    assert repeated_state.to_string() == next_state.to_string()
    assert (repeated_index, repeated_probability) == (
        branch_index,
        branch_probability,
    )
    assert state.to_string() == serialized


def test_step_with_uniform_debug_exposes_exact_selected_branch_without_mutation():
    serialized = state.to_string()
    outcomes = generate_instructions(state, "watergun", "ember")

    result = step_with_uniform_debug(state, "watergun", "ember", 0.314159)
    ordinary = step_with_uniform(state, "watergun", "ember", 0.314159)
    expected = outcomes[result.branch_index]

    assert result.state.to_string() == ordinary[0].to_string()
    assert result.branch_index == ordinary[1]
    assert result.branch_probability == ordinary[2]
    assert result.selected_instructions.percentage == expected.percentage
    assert [repr(value) for value in result.selected_instructions.instruction_list] == [
        repr(value) for value in expected.instruction_list
    ]
    assert result.state.to_string() == state.apply_instructions(expected).to_string()
    assert state.to_string() == serialized


def test_switch_projector_is_noninterfering_across_hidden_engine_worlds():
    def complete_pokemon(name, type_name, fourth_move="leer"):
        return Pokemon(
            id=name,
            level=80,
            types=(type_name, "typeless"),
            base_types=(type_name, "typeless"),
            moves=[
                Move(id="tackle"),
                Move(id="protect"),
                Move(id="rest"),
                Move(id=fourth_move),
            ],
        )

    worlds = []
    for hidden_move in ("leer", "encore"):
        worlds.append(
            State(
                side_one=Side(
                    pokemon=[
                        complete_pokemon("charmander", "fire"),
                        complete_pokemon("bulbasaur", "grass"),
                    ]
                ),
                side_two=Side(
                    pokemon=[
                        complete_pokemon("squirtle", "water"),
                        complete_pokemon("pikachu", "electric"),
                        complete_pokemon("abra", "psychic", hidden_move),
                    ]
                ),
            )
        )

    projection = project_information_set_switch(
        sys.modules["poke_engine"],
        worlds,
        "switch bulbasaur",
        "switch pikachu",
        0.5,
    )

    assert [event.actor for event in projection.events] == ["self", "opponent"]
    assert [event.species for event in projection.events] == ["bulbasaur", "pikachu"]
    assert "abra" not in repr(projection.events)
    assert "encore" not in repr(projection.events)


def test_step_with_uniform_accepts_canonical_switch_action():
    switch_state = State(
        side_one=Side(
            pokemon=[
                state.side_one.pokemon[0],
                Pokemon(id="bulbasaur", moves=[Move(id="tackle")]),
            ]
        ),
        side_two=state.side_two,
    )
    serialized = switch_state.to_string()

    next_state, branch_index, branch_probability = step_with_uniform(
        switch_state, "switch bulbasaur", "ember", 0.0
    )

    assert next_state.side_one.active_index == "1"
    assert branch_index >= 0
    assert 0.0 < branch_probability <= 1.0
    assert switch_state.to_string() == serialized
    with pytest.raises(ValueError, match="Invalid canonical action for side one"):
        step_with_uniform(switch_state, "bulbasaur", "ember", 0.0)


@pytest.mark.parametrize(
    "u", [-0.1, 1.0, float("nan"), float("inf"), float("-inf")]
)
def test_step_with_uniform_rejects_invalid_uniform(u):
    with pytest.raises(ValueError, match=r"u must be finite and in \[0, 1\)"):
        step_with_uniform(state, "watergun", "ember", u)


def test_monte_carlo_search():
    monte_carlo_tree_search(state, 10)


def test_monte_carlo_iteration_limit_is_exact_below_batch_size():
    assert monte_carlo_tree_search(state, duration_ms=0, iterations=7).total_visits == 7


def _mcts_root_snapshot(result):
    return (
        [(entry.move_choice, entry.visits, entry.total_score) for entry in result.side_one],
        [(entry.move_choice, entry.visits, entry.total_score) for entry in result.side_two],
        result.total_visits,
    )


def test_seeded_monte_carlo_search_is_reproducible_and_exact():
    first = monte_carlo_tree_search(state, duration_ms=0, iterations=257, seed=42)
    second = monte_carlo_tree_search(state, duration_ms=0, iterations=257, seed=42)

    assert first.total_visits == 257
    assert _mcts_root_snapshot(first) == _mcts_root_snapshot(second)


def test_different_monte_carlo_seeds_can_change_visits():
    first = monte_carlo_tree_search(state, duration_ms=0, iterations=257, seed=7)
    second = monte_carlo_tree_search(state, duration_ms=0, iterations=257, seed=8)

    first_visits = [entry.visits for entry in first.side_one + first.side_two]
    second_visits = [entry.visits for entry in second.side_one + second.side_two]
    assert first_visits != second_visits


def test_seeded_monte_carlo_search_uses_root_priors():
    favor_watergun = monte_carlo_tree_search(
        state,
        duration_ms=0,
        iterations=64,
        seed=42,
        s1_priors=[("watergun", 1.0)],
        c_puct=100.0,
    )
    visits = {entry.move_choice: entry.visits for entry in favor_watergun.side_one}

    assert visits["watergun"] > max(
        visits[move] for move in visits if move != "watergun"
    )


def test_seeded_monte_carlo_search_rejects_multithreading():
    with pytest.raises(ValueError, match="seeded MCTS requires threads=1"):
        monte_carlo_tree_search(
            state, duration_ms=0, iterations=10, threads=2, seed=42
        )


def test_seeded_monte_carlo_search_requires_exact_iterations():
    with pytest.raises(
        ValueError, match="seeded MCTS requires an exact positive iteration count"
    ):
        monte_carlo_tree_search(state, duration_ms=10, seed=42)


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
    assert result.diagnostics.paired_evaluation_cells_evaluated == 0
    assert result.diagnostics.paired_evaluation_total_iterations == 0
    assert result.diagnostics.paired_evaluation_complete is False
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
    assert diagnostics.paired_evaluation_complete is True
    assert diagnostics.paired_evaluation_cells_evaluated > 0
    assert diagnostics.paired_evaluation_total_iterations == (
        diagnostics.paired_evaluation_cells_evaluated * 256
    )
    assert diagnostics.paired_evaluation_elapsed_ms >= 0


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


def test_exact_request_is_authoritative_for_root_switch_under_shadow_tag():
    trapped_state = State(
        side_one=Side(pokemon=[
            Pokemon(id="squirtle", hp=100, maxhp=100, moves=[Move(id="watergun", pp=32)]),
            Pokemon(id="pikachu", hp=100, maxhp=100, moves=[Move(id="thunderbolt", pp=32)]),
        ]),
        side_two=Side(pokemon=[
            Pokemon(id="gothitelle", hp=100, maxhp=100, ability="shadowtag", moves=[Move(id="psychic", pp=32)]),
        ]),
    )
    native, _ = root_options(trapped_state)
    assert "switch pikachu" not in native
    request_actions = ["watergun", "switch pikachu"]
    authoritative, opponent = root_options_with_s1_request(trapped_state, request_actions)
    assert authoritative == request_actions
    step = step_with_uniform_r1_semantic_s1_request(
        trapped_state, request_actions, "switch pikachu", opponent[0], 0.25,
    )
    assert step.state.side_one.active_index == "1"
    trapped_only, _ = root_options_with_s1_request(trapped_state, ["watergun"])
    assert trapped_only == ["watergun"]

    searched = monte_carlo_tree_search_with_s1_request(
        trapped_state,
        request_actions,
        duration_ms=0,
        iterations=32,
        threads=1,
        seed=7,
    )
    assert {row.move_choice for row in searched.side_one} == set(request_actions)
    assert searched.total_visits == 32


def test_request_authoritative_mcts_rejects_seeded_time_and_threads():
    actions, _ = root_options(state)
    with pytest.raises(ValueError, match="positive iteration"):
        monte_carlo_tree_search_with_s1_request(
            state, actions, duration_ms=1, iterations=0, threads=1, seed=1,
        )
    with pytest.raises(ValueError, match="threads=1"):
        monte_carlo_tree_search_with_s1_request(
            state, actions, duration_ms=0, iterations=1, threads=2, seed=1,
        )
