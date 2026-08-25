from __future__ import annotations

from types import SimpleNamespace

import pytest

from train.resource_shadow import (
    FEATURE_COUNT,
    RESOURCE_FEATURE_COUNT,
    extract_resource_features,
    shadow_utility,
)


def _move(name: str, pp: int = 8):
    return SimpleNamespace(id=name, pp=pp, disabled=False)


def _pokemon(name: str, hp: int, *, tera: bool = False, revealed: bool = True):
    return SimpleNamespace(
        id=name,
        hp=hp,
        maxhp=100,
        status="NONE",
        terastallized=tera,
        item="LEFTOVERS" if revealed else "UNKNOWNITEM",
        ability="LEVITATE" if revealed else "NONE",
        moves=[_move("TACKLE") if revealed else _move("NONE", 0) for _ in range(4)],
    )


def _side(pokemon):
    return SimpleNamespace(
        pokemon=pokemon,
        active_index=0,
        attack_boost=0,
        defense_boost=0,
        special_attack_boost=0,
        special_defense_boost=0,
        speed_boost=0,
        substitute_health=0,
        side_conditions=SimpleNamespace(
            reflect=0,
            light_screen=0,
            aurora_veil=0,
            stealth_rock=0,
            spikes=0,
            toxic_spikes=0,
            sticky_web=0,
        ),
    )


def _state():
    own = [_pokemon(f"OWN{i}", 100) for i in range(6)]
    opponent = [_pokemon("OPP0", 50)] + [_pokemon("NONE", 0, revealed=False) for _ in range(5)]
    return SimpleNamespace(
        side_one=_side(own),
        side_two=_side(opponent),
        s1_can_tera=True,
        s2_can_tera=True,
        trick_room=SimpleNamespace(active=False),
        s1_public_reveals=0,
    )


def test_resource_features_are_bounded_and_player_information_only():
    state = _state()
    first = extract_resource_features(state, turn=12)
    hidden = state.side_two.pokemon[1]
    hidden.hp = 100
    hidden.maxhp = 100
    hidden.item = "LIFEORB"
    hidden.ability = "INTIMIDATE"
    hidden.moves = [_move("EARTHQUAKE", 16) for _ in range(4)]
    second = extract_resource_features(state, turn=12)
    assert first == second
    assert len(first) == FEATURE_COUNT
    assert all(0.0 <= value <= 1.0 for value in first)


def test_resource_features_track_conserved_resources():
    state = _state()
    rich = extract_resource_features(state)
    state.side_one.pokemon[1].hp = 0
    state.side_one.pokemon[0].moves[0].pp = 0
    state.s1_can_tera = False
    depleted = extract_resource_features(state)
    for index in (0, 1, 3, 4, 5, 6, 7):
        assert depleted[index] < rich[index]


def test_public_information_features_use_only_packed_mask():
    state = _state()
    state.s1_public_reveals = 1 | (1 << 6) | (1 << 30) | (1 << 36)
    first = extract_resource_features(state, include_public_information=True)
    assert first[16:20] == pytest.approx([1 / 6, 1 / 24, 1 / 6, 1 / 6])
    hidden = state.side_two.pokemon[1]
    hidden.id = "SECRET"
    hidden.item = "LIFEORB"
    hidden.ability = "INTIMIDATE"
    hidden.moves = [_move("EARTHQUAKE") for _ in range(4)]
    second = extract_resource_features(state, include_public_information=True)
    assert first[16:20] == second[16:20]


def test_public_information_features_fail_closed_without_mask():
    state = _state()
    del state.s1_public_reveals
    with pytest.raises(ValueError, match="causal reveal mask"):
        extract_resource_features(state, include_public_information=True)


def test_shadow_utility_rejects_negative_resource_prices():
    with pytest.raises(ValueError, match="non-negative"):
        shadow_utility([0.0] * FEATURE_COUNT, [-1.0] + [0.0] * (FEATURE_COUNT - 1))
    coefficients = [1.0] * RESOURCE_FEATURE_COUNT + [-1.0, 0.5]
    assert shadow_utility([0.5] * FEATURE_COUNT, coefficients) == pytest.approx(10.25)
