from __future__ import annotations

import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.prior_server import (  # noqa: E402
    BattleSession,
    append_private_jsonl,
    fresh_observation_space,
    player_information_state,
    session_key,
)


class PriorServerSessionKeyTests(unittest.TestCase):
    def test_empty_namespace_preserves_legacy_tag(self):
        self.assertEqual(session_key("", "battle-gen9randombattle-1"), "battle-gen9randombattle-1")

    def test_namespace_isolates_identical_battle_tags(self):
        tag = "battle-gen9randombattle-1"
        self.assertNotEqual(session_key("worker-1", tag), session_key("worker-2", tag))

    def test_raw_tag_is_unmodified_after_key_creation(self):
        tag = "battle-gen9randombattle-1"
        key = session_key("worker-1", tag)
        self.assertTrue(key.endswith(tag))
        self.assertIn("\0", key)

    def test_fresh_observation_space_is_reset_and_does_not_mutate_template(self):
        class FakeObservationSpace:
            def __init__(self):
                self.history = ["old-battle"]

            def reset(self):
                self.history = []

        template = FakeObservationSpace()
        isolated = fresh_observation_space(template)
        self.assertIsNot(isolated, template)
        self.assertEqual(isolated.history, [])
        self.assertEqual(template.history, ["old-battle"])

    def test_feed_line_retains_exact_protocol_prefix_including_request(self):
        session = BattleSession.__new__(BattleSession)
        session.protocol_lines = []
        session.pending_request = False
        session.tag = "battle-1"
        session.battle = SimpleNamespace(parse_request=lambda request: None)
        session.server = SimpleNamespace()
        raw = '|request|{"active": []}'
        session.feed_line(raw)
        self.assertEqual(session.protocol_lines, [raw])
        self.assertTrue(session.pending_request)

    def test_private_jsonl_append_forces_mode_0600(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decisions.jsonl"
            append_private_jsonl(str(path), b'{"decision":1}\n')
            path.chmod(0o644)
            append_private_jsonl(str(path), b'{"decision":2}\n')
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.read_bytes(), b'{"decision":1}\n{"decision":2}\n')

    def test_player_information_state_deep_copies_private_request(self):
        from metamon.interface import UniversalMove, UniversalPokemon

        pokemon = SimpleNamespace(species="pikachu", previous_move=None)
        request = {
            "active": [{"moves": [{"id": "thunderbolt", "pp": 24}]}],
            "side": {"pokemon": [{"ident": "p1: Pikachu"}]},
        }
        battle = SimpleNamespace(
            team={"p1: Pikachu": pokemon},
            active_pokemon=pokemon,
            opponent_team={},
            opponent_active_pokemon=None,
            _last_request=request,
        )
        state = SimpleNamespace(to_dict=lambda: {"turn": 1})
        universal_pokemon = UniversalPokemon(
            name="pikachu",
            hp_pct=1.0,
            types="electric typeless",
            item="noitem",
            ability="noability",
            lvl=100,
            status="nostatus",
            effect="noeffect",
            moves=[],
            atk_boost=0,
            spa_boost=0,
            def_boost=0,
            spd_boost=0,
            spe_boost=0,
            accuracy_boost=0,
            evasion_boost=0,
            base_atk=100,
            base_spa=100,
            base_def=100,
            base_spd=100,
            base_spe=100,
            base_hp=100,
            tera_type="normal",
            base_species="pikachu",
        )
        with (
            patch.object(
                UniversalPokemon,
                "from_Pokemon",
                return_value=universal_pokemon,
            ),
            patch.object(
                UniversalMove,
                "from_Move",
                return_value=UniversalMove.blank_move(),
            ),
        ):
            captured = player_information_state(battle, state)

        request["active"][0]["moves"][0]["pp"] = 0
        self.assertEqual(
            captured["private_request"]["active"][0]["moves"][0]["pp"], 24
        )

    def test_live_conversion_accepts_null_secondary_type_slot(self):
        from metamon.interface import UniversalPokemon

        pokemon = SimpleNamespace(
            gen=9,
            base_stats={},
            type=["steel", None],
            had_ability="moldbreaker",
            lvl=100,
            max_hp=100,
            moves={},
            nickname="Tinkaton",
            name="tinkaton",
            boosts=SimpleNamespace(to_dict=lambda: {}),
            current_hp=100,
            effects={},
            active_item="leftovers",
            status=None,
            active_ability="moldbreaker",
            last_used_move=None,
            tera_type=None,
        )

        converted = UniversalPokemon.metamon_to_poke_env(pokemon, is_active=True)

        self.assertEqual(converted.type_1.name, "STEEL")
        self.assertIsNone(converted.type_2)


if __name__ == "__main__":
    unittest.main()
