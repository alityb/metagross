from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experimental" / "src"))
sys.path.insert(
    0,
    str(ROOT / "experimental/engine/pe_v3_learned_priors/poke-engine-py/python"),
)

import poke_engine  # noqa: E402
from scripts.run_public_search_state_gate_a import (  # noqa: E402
    _clone_pokemon,
    _clone_side,
    hidden_perturbation,
)
from search.public_search_policy_smoke import infer, make_policy, parameter_count  # noqa: E402
from search.public_search_state_v1 import (  # noqa: E402
    PublicSearchStateError,
    canonical_bytes,
    compile_side_one_reveal_mask,
    extract_public_search_state,
    install_side_one_reveal_mask,
)


SOURCE = (
    ROOT
    / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-a-decisions.jsonl.dual-r1-roots.jsonl"
)


def ordinary_row():
    with SOURCE.open() as handle:
        for line in handle:
            row = json.loads(line)
            state = poke_engine.State.from_string(row["state"])
            first, second = poke_engine.root_options(state)
            if len(first) >= 2 and second != ["No Move"]:
                return row, state, first, second
    raise AssertionError("fixture source has no ordinary joint root")


def masked_fixture():
    row, state, first, second = ordinary_row()
    bits = compile_side_one_reveal_mask(
        state, row["r1_policy_snapshot"]["player_information_state"]
    )
    return row, install_side_one_reveal_mask(state, bits), first, second


def test_root_mapping_matches_causal_snapshot_and_hidden_completion_is_inert():
    row, state, _, _ = masked_fixture()
    public = extract_public_search_state(state, poke_engine)
    snapshot = row["r1_policy_snapshot"]
    assert public["action_table"]["name_table"] == snapshot["name_table"]
    assert public["action_table"]["illegal_actions"] == snapshot["illegal_actions"]
    perturbed = hidden_perturbation(state, poke_engine)
    assert perturbed is not None
    assert canonical_bytes(extract_public_search_state(perturbed, poke_engine)) == canonical_bytes(public)


def test_empty_legacy_mask_fails_closed():
    _, state, _, _ = ordinary_row()
    with pytest.raises(PublicSearchStateError, match="fails closed"):
        extract_public_search_state(state, poke_engine)


def test_apply_reverse_restores_engine_and_public_bytes():
    _, state, first, second = masked_fixture()
    before_engine = state.to_string()
    before_public = canonical_bytes(extract_public_search_state(state, poke_engine))
    result = poke_engine.step_with_uniform_r1_semantic(state, first[0], second[0], 0.25)
    restored = result.state.reverse_instructions(result.selected_instructions)
    assert restored.to_string() == before_engine
    assert canonical_bytes(extract_public_search_state(restored, poke_engine)) == before_public


def test_perspective_swap_is_canonical():
    _, state, _, _ = masked_fixture()
    full = (1 << 42) - 1
    original = poke_engine.State(
        side_one=_clone_side(
            poke_engine, state.side_one,
            [_clone_pokemon(poke_engine, pokemon) for pokemon in state.side_one.pokemon],
        ),
        side_two=_clone_side(
            poke_engine, state.side_two,
            [_clone_pokemon(poke_engine, pokemon) for pokemon in state.side_two.pokemon],
        ),
        weather=state.weather,
        weather_turns_remaining=state.weather_turns_remaining,
        terrain=state.terrain,
        terrain_turns_remaining=state.terrain_turns_remaining,
        trick_room=state.trick_room,
        trick_room_turns_remaining=state.trick_room_turns_remaining,
        team_preview=state.team_preview,
        s1_threat=state.s1_threat,
        s2_threat=state.s2_threat,
        scout_value=state.scout_value,
        threat_matrix=state.threat_matrix,
        wincon_matrix=state.wincon_matrix,
        s1_public_reveals=full,
        s2_public_reveals=full,
    )
    swapped = poke_engine.State(
        side_one=_clone_side(
            poke_engine, original.side_two,
            [_clone_pokemon(poke_engine, pokemon) for pokemon in original.side_two.pokemon],
        ),
        side_two=_clone_side(
            poke_engine, original.side_one,
            [_clone_pokemon(poke_engine, pokemon) for pokemon in original.side_one.pokemon],
        ),
        weather=original.weather,
        weather_turns_remaining=original.weather_turns_remaining,
        terrain=original.terrain,
        terrain_turns_remaining=original.terrain_turns_remaining,
        trick_room=original.trick_room,
        trick_room_turns_remaining=original.trick_room_turns_remaining,
        team_preview=original.team_preview,
        s1_threat=original.s2_threat,
        s2_threat=original.s1_threat,
        scout_value=-original.scout_value,
        threat_matrix=original.threat_matrix,
        wincon_matrix=original.wincon_matrix,
        s1_public_reveals=full,
        s2_public_reveals=full,
    )
    right = extract_public_search_state(original, poke_engine, observer="side_two")
    left = extract_public_search_state(swapped, poke_engine, observer="side_one")
    assert canonical_bytes(right) == canonical_bytes(left)


def test_smoke_policy_is_deterministic_legal_and_in_parameter_range():
    _, state, _, _ = masked_fixture()
    public = extract_public_search_state(state, poke_engine)
    model = make_policy()
    assert 1_000_000 <= parameter_count(model) <= 5_000_000
    first = infer(model, [public])
    second = infer(model, [public])
    assert np.array_equal(first, second)
    illegal = np.asarray(public["action_table"]["illegal_actions"])
    assert np.all(first[0, illegal] == 0)
    assert first.sum() == pytest.approx(1.0)
