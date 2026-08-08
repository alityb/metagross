#![cfg(feature = "terastallization")]

use poke_engine::choices::Choices;
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::{PokemonMoveIndex, State};

#[test]
fn root_tera_permission_filters_only_the_denied_side() {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::TACKLE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::TACKLE);
    state.s1_can_tera = false;

    let (side_one, side_two) = state.root_get_all_options();
    assert!(!side_one
        .iter()
        .any(|choice| matches!(choice, MoveChoice::MoveTera(_))));
    assert!(side_two
        .iter()
        .any(|choice| matches!(choice, MoveChoice::MoveTera(_))));

    let (future_side_one, _) = state.get_all_options();
    assert!(future_side_one
        .iter()
        .any(|choice| matches!(choice, MoveChoice::MoveTera(_))));
}

#[test]
fn tera_permissions_round_trip_and_old_wire_defaults_to_permitted() {
    let mut state = State::default();
    state.s1_can_tera = false;
    state.s2_can_tera = false;

    let serialized = state.serialize();
    let round_trip = State::deserialize(&serialized);
    assert!(!round_trip.s1_can_tera);
    assert!(!round_trip.s2_can_tera);

    let mut old_wire_parts: Vec<&str> = serialized.split('/').collect();
    old_wire_parts.truncate(11);
    let old_wire = old_wire_parts.join("/");
    let from_old_wire = State::deserialize(&old_wire);
    assert!(from_old_wire.s1_can_tera);
    assert!(from_old_wire.s2_can_tera);
}
