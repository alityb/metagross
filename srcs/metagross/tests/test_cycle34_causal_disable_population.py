from __future__ import annotations

import sys
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from srcs.metagross.causal_reveal_ledger import (
    LEDGER_ATTRIBUTE,
    bind_live_move_states,
    freeze_ledger,
    verify_sampled_move_states,
)


ROOT = Path(__file__).resolve().parents[3]


def move(name: str, pp: int = 15, max_pp: int = 16, disabled: bool = False):
    return SimpleNamespace(
        name=name, current_pp=pp, max_pp=max_pp, disabled=disabled
    )


class FakePokemon:
    def __init__(self, known_move):
        self.moves = [known_move]
        self.ability = "dauntlessshield"
        self.item = "unknownitem"
        self.terastallized = False
        self.tera_type = None
        self.choice_item_received_after_last_move = False
        self.volatile_statuses = []

    def add_move(self, name):
        self.moves.append(move(name, pp=16, max_pp=16))

    def set_spread(self, *_args):
        pass


def sampled_set(*, item: str = "rustedshield", moves=None):
    return SimpleNamespace(
        pkmn_moveset=SimpleNamespace(moves=moves or ["heavyslam"]),
        pkmn_set=SimpleNamespace(
            ability="dauntlessshield", item=item, nature="serious",
            evs=(85,) * 6, tera_type="fighting",
        ),
    )


@pytest.fixture
def helpers(monkeypatch):
    vendor = ROOT / "srcs/vendor/foul-play"
    sys.path.insert(0, str(vendor))
    try:
        from fp.search import helpers as module

        monkeypatch.setattr(module, "log_pkmn_set", lambda *_args, **_kwargs: None)
        yield module
    finally:
        sys.path.remove(str(vendor))


def test_typed_causal_disable_overrides_false_public_tracker_bit(helpers) -> None:
    known = move("heavyslam", disabled=False)
    known.metagross_disable_authority = "causal_disable"
    known.metagross_causal_disabled = True
    pokemon = FakePokemon(known)

    helpers.populate_pkmn_from_set(pokemon, sampled_set())

    hydrated = pokemon.moves[0]
    assert (hydrated.current_pp, hydrated.max_pp, hydrated.disabled) == (15, 16, True)


def test_causal_disable_requires_typed_boolean_and_never_falls_back(helpers) -> None:
    known = move("heavyslam", disabled=True)
    known.metagross_disable_authority = "causal_disable"
    pokemon = FakePokemon(known)

    with pytest.raises(ValueError, match="lacks typed state"):
        helpers.populate_pkmn_from_set(pokemon, sampled_set())


def test_world_mechanical_disable_is_not_overwritten(helpers) -> None:
    known = move("heavyslam", disabled=True)
    known.metagross_disable_authority = "world_mechanical_disable"
    pokemon = FakePokemon(known)

    helpers.populate_pkmn_from_set(pokemon, sampled_set(item="choiceband"))

    hydrated = pokemon.moves[0]
    assert (hydrated.current_pp, hydrated.max_pp, hydrated.disabled) == (15, 16, False)


def test_typed_disable_and_choice_lock_are_additive_per_world(helpers) -> None:
    from fp.battle import Battler, LastUsedMove

    known = move("heavyslam", disabled=False)
    known.metagross_disable_authority = "causal_disable"
    known.metagross_causal_disabled = True
    pokemon = FakePokemon(known)
    helpers.populate_pkmn_from_set(
        pokemon, sampled_set(item="choiceband", moves=["heavyslam", "bodypress"])
    )
    side = Battler()
    side.active = pokemon
    side.last_used_move = LastUsedMove("zamazentacrowned", "heavyslam", 3)
    pokemon.name = "zamazentacrowned"
    side.lock_moves()

    states = {row.name: row.disabled for row in pokemon.moves}
    assert states == {"heavyslam": True, "bodypress": True}


def test_hidden_power_uses_the_same_typed_causal_contract(helpers) -> None:
    known = move("hiddenpowerice", disabled=False)
    known.metagross_disable_authority = "causal_disable"
    known.metagross_causal_disabled = True
    pokemon = FakePokemon(known)
    pokemon.ability = None
    hp_set = sampled_set()
    hp_set.pkmn_moveset.moves = ["hiddenpowerfire"]

    helpers.populate_pkmn_from_set(pokemon, hp_set)

    assert pokemon.moves[0].disabled is True


def test_preserved_cycle33_force_switch_transcript_is_repaired() -> None:
    vendor = ROOT / "srcs/vendor/foul-play"
    sys.path.insert(0, str(vendor))
    try:
        from constants import BattleType
        from data.pkmn_sets import RandomBattleTeamDatasets
        from fp.battle import Battle
        from fp.battle_modifier import update_battle
        from fp.search.random_battles import prepare_random_battles

        path = ROOT / (
            "experimental/runs/search_native_v2_cycle33_prospective_h2h_20260815/"
            "h2h-logs/c33h2hx013ffef.protocol.jsonl"
        )
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        requests = [
            json.loads(line.removeprefix("|request|"))
            for row in rows if row["direction"] == "received"
            for line in row["message"].splitlines()
            if line.startswith("|request|")
        ]
        battle = Battle("battle-gen9randombattle-4458")
        battle.user.name, battle.opponent.name = "p1", "p2"
        battle.generation = "gen9"
        battle.pokemon_format = "gen9randombattle"
        battle.battle_type = BattleType.RANDOM_BATTLE
        RandomBattleTeamDatasets.initialize("gen9")
        battle.start_non_team_preview_battle(
            requests[0],
            "|switch|p2a: Raging Bolt|Raging Bolt, L78|100/100",
        )
        battle.request_json = requests[0]
        for row in rows[13:23]:
            if row["direction"] == "received":
                update_battle(battle, row["message"])
        assert battle.rqid == 8 and battle.force_switch is True
        public = [
            line
            for row in rows[:23] if row["direction"] == "received"
            for line in row["message"].splitlines()
            if line.startswith("|") and not line.startswith("|request|")
        ]
        ledger = bind_live_move_states(
            battle, freeze_ledger(battle.battle_tag, battle.user.name, public)
        )
        setattr(battle, LEDGER_ATTRIBUTE, ledger.to_payload())
        state = next(
            state for fact in ledger.facts if fact.exact_public_species == "zamazentacrowned"
            for state in fact.move_states if state.move == "heavyslam"
        )
        assert (
            state.current_pp, state.max_pp, state.disable_authority,
            state.disabled, state.causal_disable_lifecycle_state,
        ) == (15, 16, "causal_disable", True, True)

        worlds = prepare_random_battles(battle, 8, rng=random.Random(2026081534))
        verify_sampled_move_states(battle, worlds)
        for sampled, weight in worlds:
            zamazenta = sampled.opponent.active
            heavy_slam = next(move for move in zamazenta.moves if move.name == "heavyslam")
            assert zamazenta.name == "zamazentacrowned"
            assert zamazenta.item == "rustedshield"
            assert (heavy_slam.current_pp, heavy_slam.max_pp, heavy_slam.disabled) == (
                15, 16, True
            )
            assert weight == pytest.approx(1 / 8)
    finally:
        sys.path.remove(str(vendor))


def test_force_switch_does_not_close_opponent_disable_lifecycle() -> None:
    prefix = [
        "|switch|p1a: Froslass|Froslass, L87|100/100",
        "|switch|p2a: Zamazenta|Zamazenta-Crowned, L68|100/100",
        "|move|p2a: Zamazenta|Heavy Slam|p1a: Froslass",
        "|-damage|p1a: Froslass|0 fnt",
        "|-start|p2a: Zamazenta|Disable|Heavy Slam|[from] ability: Cursed Body|[of] p1a: Froslass",
        "|faint|p1a: Froslass",
        "|upkeep",
    ]
    carried = freeze_ledger("battle-force-carry", "p1", prefix)
    fact = next(fact for fact in carried.facts if fact.species == "zamazenta")
    assert fact.disable_history[-1].disabled is True

    ended = freeze_ledger(
        "battle-force-end", "p1", [*prefix, "|-end|p2a: Zamazenta|Disable"]
    )
    fact = next(fact for fact in ended.facts if fact.species == "zamazenta")
    assert fact.disable_history[-1].disabled is False

    switched = freeze_ledger(
        "battle-force-switch", "p1",
        [*prefix, "|switch|p2a: Other|Mew, L80|100/100"],
    )
    fact = next(fact for fact in switched.facts if fact.species == "zamazenta")
    assert fact.disable_history[-1].disabled is False
