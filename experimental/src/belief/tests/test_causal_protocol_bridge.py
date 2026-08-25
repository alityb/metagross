from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from belief.causal_protocol_bridge import (
    CausalProtocolBridgeError,
    canonical_species,
    parse_causal_protocol,
    reconcile_causal_facts,
)


def request(role: str = "p1") -> dict:
    return {
        "rqid": 7,
        "active": [{"moves": [{"id": "tackle", "pp": 10, "disabled": False}]}],
        "side": {"id": role, "pokemon": []},
    }


def lines(events: list[str], own_request: dict | None = None) -> list[str]:
    own_request = own_request or request()
    return ["|init|battle", *events, "|request|" + json.dumps(own_request, separators=(",", ":"))]


def test_protocol_is_the_only_opponent_authority() -> None:
    own = request()
    facts = parse_causal_protocol(
        lines([
            "|switch|p2a: Mewtwo|Mewtwo, L72|100/100",
            "|-ability|p2a: Mewtwo|Pressure",
            "|move|p2a: Mewtwo|Psychic|p1a: Own",
            "|-item|p2a: Mewtwo|Leftovers",
        ], own),
        player_role="p1",
        private_request=own,
    )
    assert len(facts.opponent) == 1
    mewtwo = facts.opponent[0]
    assert mewtwo.species == "mewtwo"
    assert mewtwo.level == 72
    assert facts.opponent_active_species == "mewtwo"
    assert mewtwo.moves == ("psychic",)
    assert mewtwo.ability == "pressure"
    assert mewtwo.current_item == "leftovers"
    assert mewtwo.item_status_revealed


def test_showdown_empty_framing_line_is_ignored() -> None:
    own = request()
    facts = parse_causal_protocol(
        lines(["", "|switch|p2a: Foe|Mew, L80|100/100"], own),
        player_role="p1",
        private_request=own,
    )
    assert facts.opponent[0].species == "mew"


def test_private_request_cannot_add_opponent_defaults() -> None:
    own = request()
    polluted = copy.deepcopy(own)
    polluted["opponent"] = {"ability": "pressure", "item": "leftovers"}
    facts = parse_causal_protocol(
        lines(["|switch|p2a: Mewtwo|Mewtwo, L72|100/100"], polluted),
        player_role="p1",
        private_request=polluted,
    )
    assert facts.opponent[0].ability is None
    assert facts.opponent[0].current_item is None
    assert not facts.opponent[0].item_status_revealed


def test_consumed_item_records_history_but_current_item_is_none() -> None:
    own = request()
    facts = parse_causal_protocol(
        lines([
            "|switch|p2a: Foe|Snorlax, L80|100/100",
            "|-enditem|p2a: Foe|Sitrus Berry|[eat]",
        ], own), player_role="p1", private_request=own,
    )
    foe = facts.opponent[0]
    assert foe.current_item == "none"
    assert foe.historically_revealed_items == ("sitrusberry",)


def test_tagged_source_uses_of_actor() -> None:
    own = request()
    facts = parse_causal_protocol(
        lines([
            "|switch|p1a: Own|Pikachu, L80|100/100",
            "|switch|p2a: Foe|Ferrothorn, L80|100/100",
            "|-damage|p1a: Own|90/100|[from] ability: Iron Barbs|[of] p2a: Foe",
            "|-damage|p1a: Own|80/100|[from] item: Rocky Helmet|[of] p2a: Foe",
        ], own), player_role="p1", private_request=own,
    )
    foe = facts.opponent[0]
    assert foe.ability == "ironbarbs"
    assert foe.current_item == "rockyhelmet"


def test_replace_migrates_disguise_facts() -> None:
    own = request()
    facts = parse_causal_protocol(
        lines([
            "|switch|p2a: Gengar|Gengar, L80|100/100",
            "|move|p2a: Gengar|Knock Off|p1a: Own",
            "|replace|p2a: Zoroark|Zoroark, L80|90/100",
        ], own), player_role="p1", private_request=own,
    )
    assert [fact.species for fact in facts.opponent] == ["zoroark"]
    assert facts.opponent[0].moves == ("knockoff",)


def test_repeated_species_before_replace_fails_closed() -> None:
    own = request()
    with pytest.raises(CausalProtocolBridgeError, match="ambiguous repeated species"):
        parse_causal_protocol(
            lines([
                "|switch|p2a: Real|Gengar, L80|100/100",
                "|switch|p2a: Other|Mew, L80|100/100",
                "|switch|p2a: Disguise|Gengar, L80|100/100",
                "|replace|p2a: Zoroark|Zoroark, L80|90/100",
            ], own), player_role="p1", private_request=own,
        )


@pytest.mark.parametrize(
    ("public", "base"),
    [
        ("Alcremie-Ruby-Cream", "alcremie"),
        ("Sawsbuck-Winter", "sawsbuck"),
        ("Morpeko-Hangry", "morpeko"),
        ("Darmanitan-Galar-Zen", "darmanitangalar"),
        ("Minior-Violet", "minior"),
    ],
)
def test_frozen_form_aliases(public: str, base: str) -> None:
    assert canonical_species(public) == base


def test_role_and_request_mismatch_fail_closed() -> None:
    own = request()
    with pytest.raises(CausalProtocolBridgeError, match="role mismatch"):
        parse_causal_protocol(lines(["|switch|p2a: Foe|Mew, L80|100/100"], own), player_role="p2", private_request=own)
    changed = copy.deepcopy(own)
    changed["rqid"] = 8
    with pytest.raises(CausalProtocolBridgeError, match="protocol/private request mismatch"):
        parse_causal_protocol(lines(["|switch|p2a: Foe|Mew, L80|100/100"], own), player_role="p1", private_request=changed)


class Record(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FakeEngine:
    SideConditions = Record
    VolatileStatusDurations = Record
    Move = Record
    Pokemon = Record
    Side = Record
    State = Record


def fake_pokemon(species: str, *, item: str = "unknownitem", ability: str = "none") -> Record:
    return Record(
        id=species, level=80, types=("normal", "typeless"),
        base_types=("normal", "typeless"), hp=100, maxhp=100,
        ability=ability, base_ability=ability, item=item, nature="serious",
        evs=(85,) * 6, attack=100, defense=100, special_attack=100,
        special_defense=100, speed=100, status="none", rest_turns=0,
        sleep_turns=0, weight_kg=1.0,
        moves=[Record(id="none", pp=32, disabled=False) for _ in range(4)],
        terastallized=False, tera_type="normal",
    )


def fake_side(pokemon: list[Record]) -> Record:
    condition_names = (
        "spikes", "toxic_spikes", "stealth_rock", "sticky_web", "tailwind",
        "lucky_chant", "lunar_dance", "reflect", "light_screen", "aurora_veil",
        "crafty_shield", "safeguard", "mist", "protect", "healing_wish",
        "mat_block", "quick_guard", "toxic_count", "wide_guard",
    )
    duration_names = ("confusion", "encore", "lockedmove", "slowstart", "taunt", "yawn")
    return Record(
        pokemon=pokemon, side_conditions=Record(**{name: 0 for name in condition_names}),
        active_index="P0", baton_passing=False, shed_tailing=False,
        volatile_status_durations=Record(**{name: 0 for name in duration_names}),
        wish=(0, 0), future_sight=(0, 0, "none"), force_switch=False,
        force_trapped=False, slow_uturn_move=False, volatile_statuses=set(),
        substitute_health=0, attack_boost=0, defense_boost=0,
        special_attack_boost=0, special_defense_boost=0, speed_boost=0,
        accuracy_boost=0, evasion_boost=0, last_used_move="move:none",
        switch_out_move_second_saved_move="move:none",
    )


def test_reconciliation_repairs_only_event_certified_fields() -> None:
    own = request()
    facts = parse_causal_protocol(
        lines([
            "|switch|p2a: Foe|Mew, L80|100/100",
            "|move|p2a: Foe|Psychic|p1a: Own",
            "|-item|p2a: Foe|Leftovers",
            "|-ability|p2a: Foe|Synchronize",
        ], own), player_role="p1", private_request=own,
    )
    opponent = [fake_pokemon("mew")] + [fake_pokemon("none", item="none") for _ in range(5)]
    state = Record(
        side_one=fake_side([fake_pokemon("pikachu") for _ in range(6)]),
        side_two=fake_side(opponent), weather="none", weather_turns_remaining=0,
        terrain="none", terrain_turns_remaining=0, trick_room=False,
        trick_room_turns_remaining=0, team_preview=False, s1_threat=0.0,
        s2_threat=0.0, scout_value=0.0, threat_matrix=[], wincon_matrix=[],
        s1_public_reveals=0, s2_public_reveals=0,
    )
    reconciled = reconcile_causal_facts(state, FakeEngine, facts)
    mew = reconciled.state.side_two.pokemon[0]
    assert mew.moves[0].id == "psychic"
    assert mew.item == "leftovers"
    assert mew.ability == "synchronize"
    assert reconciled.reveal_bits & 1
    assert reconciled.reveal_bits & (1 << 6)
    assert reconciled.reveal_bits & (1 << 30)
    assert reconciled.reveal_bits & (1 << 36)
    assert reconciled.archival_repairs == (
        "move:mew:psychic", "item:mew:unknownitem->leftovers",
        "ability:mew:none->synchronize",
    )
