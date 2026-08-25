from types import SimpleNamespace

import numpy as np

from train.candidate_switch_residual import (
    CANDIDATE_FEATURE_NAMES, canonical_moves, residual_features, static_features,
    summarize_matchups, switch_target, type_vector,
)


def pokemon(name="pivot", *, types=("WATER", "FLYING"), moves=("UTURN", "ROOST")):
    return SimpleNamespace(
        id=name, hp=75, maxhp=100, attack=200, defense=150,
        special_attack=100, special_defense=175, speed=250, types=types,
        moves=[SimpleNamespace(id=move) for move in moves],
    )


def test_owned_switch_target_and_role_features_are_identity_free():
    fainted = pokemon("active")
    fainted.hp = 0
    target = pokemon()
    state = SimpleNamespace(side_one=SimpleNamespace(active_index=0, pokemon=[fainted, target]))
    assert switch_target(state, "switch pivot") is target
    assert sum(type_vector(target)) == 2
    assert static_features(target)[-2:] == [1.0, 1.0]
    assert not any("species" in name or "identity" in name for name in CANDIDATE_FEATURE_NAMES)


def test_canonical_moves_deduplicates_tera_and_excludes_switches():
    assert canonical_moves(["surf", "surf-tera", "switch pivot", "No Move"]) == ["surf"]


def test_residual_contract_is_finite_and_complete():
    matchup = summarize_matchups([[float(i) for i in range(13)]] * 4)
    values = residual_features(
        [1.0] + [0.0] * 17, [0.0] * 18,
        [1.0] * 10, [0.0] * 10,
        matchup, [0.0] * len(matchup),
    )
    assert len(values) == len(CANDIDATE_FEATURE_NAMES) == 54
    assert np.isfinite(values).all()
