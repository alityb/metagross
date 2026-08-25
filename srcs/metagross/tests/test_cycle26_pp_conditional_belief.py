from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from srcs.metagross.causal_reveal_ledger import (
    CausalRevealLedgerError,
    LEDGER_ATTRIBUTE,
    bind_live_move_states,
    freeze_ledger,
    verify_sampled_move_states,
)


ROOT = Path(__file__).resolve().parents[3]


def move(name: str, pp: int, max_pp: int, disabled: bool = False):
    return SimpleNamespace(
        name=name, current_pp=pp, max_pp=max_pp, disabled=disabled
    )


def battle_with(pokemon):
    return SimpleNamespace(
        opponent=SimpleNamespace(active=pokemon, reserve=[])
    )


def test_magic_bounce_reflection_is_audited_but_not_intrinsic() -> None:
    ledger = freeze_ledger("battle-cycle22-regression", "p1", [
        "|switch|p1a: Skarmory|Skarmory, L80|100/100",
        "|switch|p2a: Hatterene|Hatterene, L80|100/100",
        "|move|p2a: Hatterene|Stealth Rock|p1a: Skarmory|[from] ability: Magic Bounce",
    ])
    fact = ledger.facts[0]
    assert fact.species == "hatterene"
    assert fact.moves == ()
    assert [(event.move, event.authority, event.derived_cause) for event in fact.move_events] == [
        ("stealthrock", "derived_public_execution", "ability: Magic Bounce")
    ]
    assert fact.current_ability == "magicbounce"


def test_intrinsic_move_binds_exact_live_pp_and_disabled_state() -> None:
    ledger = freeze_ledger("battle-intrinsic", "p1", [
        "|switch|p1a: Own|Pikachu, L80|100/100",
        "|switch|p2a: Hat|Hatterene, L80|100/100",
        "|move|p2a: Hat|Psychic|p1a: Own",
    ])
    tracked = SimpleNamespace(
        name="hatterene", moves=[move("psychic", 7, 16, True)]
    )
    bound = bind_live_move_states(battle_with(tracked), ledger)
    state = bound.facts[0].move_states[0]
    assert (state.move, state.current_pp, state.max_pp, state.disabled) == (
        "psychic", 7, 16, None
    )
    assert state.disable_authority == "world_mechanical_disable"
    assert state.authority == "causal_live_public_tracker"


def test_sampled_world_requires_exact_move_state_and_preserves_weight() -> None:
    ledger = freeze_ledger("battle-world", "p1", [
        "|switch|p1a: Own|Pikachu, L80|100/100",
        "|switch|p2a: Hat|Hatterene, L80|100/100",
        "|move|p2a: Hat|Psychic|p1a: Own",
    ])
    source = battle_with(SimpleNamespace(
        name="hatterene", moves=[move("psychic", 6, 16, True)]
    ))
    bound = bind_live_move_states(source, ledger)
    setattr(source, LEDGER_ATTRIBUTE, bound.to_payload())
    good = copy.deepcopy(source)
    worlds = [(good, 0.37)]
    verify_sampled_move_states(source, worlds)
    assert worlds[0][1] == 0.37

    # An ordinary opponent disabled bit is world-mechanical and may differ.
    bad = copy.deepcopy(good)
    bad.opponent.active.moves[0].disabled = False
    verify_sampled_move_states(source, [(bad, 0.37)])

    missing = copy.deepcopy(good)
    missing.opponent.active.moves = []
    with pytest.raises(CausalRevealLedgerError, match="missing causal move"):
        verify_sampled_move_states(source, [(missing, 0.37)])


def test_unrelated_hidden_completion_cannot_change_bound_move_state() -> None:
    ledger = freeze_ledger("battle-hidden", "p1", [
        "|switch|p1a: Own|Pikachu, L80|100/100",
        "|switch|p2a: Hat|Hatterene, L80|100/100",
        "|move|p2a: Hat|Psychic|p1a: Own",
    ])
    active = SimpleNamespace(name="hatterene", moves=[move("psychic", 5, 16)])
    left = SimpleNamespace(
        opponent=SimpleNamespace(
            active=copy.deepcopy(active),
            reserve=[SimpleNamespace(name="raichu", moves=[move("surf", 24, 24)])],
        )
    )
    right = copy.deepcopy(left)
    right.opponent.reserve[0].moves = [move("thunderbolt", 24, 24)]
    assert (
        bind_live_move_states(left, ledger).facts[0].move_states
        == bind_live_move_states(right, ledger).facts[0].move_states
    )


def test_sampler_population_copies_disabled_and_rejects_max_pp_drift(monkeypatch) -> None:
    vendor = ROOT / "srcs/vendor/foul-play"
    sys.path.insert(0, str(vendor))
    try:
        from fp.search import helpers

        monkeypatch.setattr(helpers, "log_pkmn_set", lambda *_args, **_kwargs: None)

        class FakePokemon:
            def __init__(self):
                self.moves = [move("psychic", 3, 16, True)]
                self.ability = None
                self.item = "unknownitem"
                self.terastallized = False
                self.tera_type = None

            def add_move(self, name):
                self.moves.append(move(name, 16, 16, False))

            def set_spread(self, *_args):
                pass

        sampled_set = SimpleNamespace(
            pkmn_moveset=SimpleNamespace(moves=["psychic"]),
            pkmn_set=SimpleNamespace(
                ability="magicbounce", item="leftovers", nature="serious",
                evs=(85,) * 6, tera_type="fairy",
            ),
        )
        pokemon = FakePokemon()
        helpers.populate_pkmn_from_set(pokemon, sampled_set)
        assert (pokemon.moves[0].current_pp, pokemon.moves[0].disabled) == (3, True)

        pokemon = FakePokemon()
        pokemon.moves[0].metagross_disable_authority = "world_mechanical_disable"
        helpers.populate_pkmn_from_set(pokemon, sampled_set)
        assert (pokemon.moves[0].current_pp, pokemon.moves[0].disabled) == (3, False)

        pokemon = FakePokemon()
        pokemon.moves[0].max_pp = 15
        with pytest.raises(ValueError, match="max PP disagrees"):
            helpers.populate_pkmn_from_set(pokemon, sampled_set)
    finally:
        sys.path.remove(str(vendor))
    LEDGER_ATTRIBUTE,
