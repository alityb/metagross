from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.r1_public_events import (  # noqa: E402
    PublicEventProjectionError,
    SEMANTIC_TRACE_CERTIFICATE,
    _public_hp_fraction,
    project_information_set_observations,
    project_information_set_switch,
    project_information_set_transition,
)


class Instruction:
    def __init__(self, representation):
        self.representation = representation

    def __repr__(self):
        return self.representation


def pokemon(name, *, item="none", ability="none", hp=100, level=80):
    return SimpleNamespace(
        id=name,
        level=level,
        types=("normal", "typeless"),
        base_types=("normal", "typeless"),
        hp=hp,
        maxhp=100,
        ability=ability,
        base_ability=ability,
        item=item,
        nature="serious",
        evs=(85, 85, 85, 85, 85, 85),
        attack=100,
        defense=100,
        special_attack=100,
        special_defense=100,
        speed=100,
        status="none",
        rest_turns=0,
        sleep_turns=0,
        weight_kg=10.0,
        terastallized=False,
        tera_type="normal",
        moves=[SimpleNamespace(id="tackle", disabled=False, pp=32)],
    )


def side(active, reserve):
    fainted = [pokemon("pikachu", hp=0) for _ in range(4)]
    return SimpleNamespace(
        pokemon=[active, reserve, *fainted],
        active_index="0",
        baton_passing=False,
        shed_tailing=False,
        wish=(0, 0),
        future_sight=(0, "0"),
        force_switch=False,
        force_trapped=False,
        slow_uturn_move=False,
        volatile_statuses=set(),
        substitute_health=0,
        attack_boost=0,
        defense_boost=0,
        special_attack_boost=0,
        special_defense_boost=0,
        speed_boost=0,
        accuracy_boost=0,
        evasion_boost=0,
        last_used_move="move:none",
        switch_out_move_second_saved_move="none",
        side_conditions=SimpleNamespace(**{
            name: 0 for name in (
                "spikes", "toxic_spikes", "stealth_rock", "sticky_web", "tailwind",
                "lucky_chant", "lunar_dance", "reflect", "light_screen", "aurora_veil",
                "crafty_shield", "safeguard", "mist", "protect", "healing_wish",
                "mat_block", "quick_guard", "toxic_count", "wide_guard",
            )
        }),
        volatile_status_durations=SimpleNamespace(
            confusion=0, encore=0, lockedmove=0, slowstart=0, taunt=0, yawn=0
        ),
    )


def state(*, hidden_item="none"):
    return SimpleNamespace(
        side_one=side(pokemon("charmander"), pokemon("bulbasaur")),
        side_two=side(pokemon("squirtle"), pokemon("charizard", item=hidden_item)),
        weather="none",
        weather_turns_remaining=0,
        terrain="none",
        terrain_turns_remaining=0,
        trick_room=False,
        trick_room_turns_remaining=0,
        team_preview=False,
        s1_threat=0.0,
        s2_threat=0.0,
        scout_value=0.0,
        threat_matrix=[0.0] * 36,
        wincon_matrix=[0.0] * 36,
    )


class FakeEngine:
    def __init__(self, *, extra_instruction=None, mutate_hp=False):
        self.extra_instruction = extra_instruction
        self.mutate_hp = mutate_hp

    @staticmethod
    def transition_debug_contract():
        return "poke-engine-0.0.47-r1-switch-v1"

    @staticmethod
    def root_options(*, state):
        return ["tackle", "switch charmander"], ["tackle", "switch squirtle"]

    def step_with_uniform_debug(self, source, side_one_action, side_two_action, _u):
        target = copy.deepcopy(source)
        target.side_one.active_index = "1"
        target.side_two.active_index = "1"
        instructions = [Instruction("Switch SideOne: P0 -> P1")]
        if source.side_two.pokemon[1].item == "sentinelitem":
            instructions.append(
                Instruction("SetLastUsedMove SideOne: None -> Switch(P1)")
            )
            target.side_one.last_used_move = "switch:1"
        instructions.append(Instruction("Switch SideTwo: P0 -> P1"))
        if self.extra_instruction:
            instructions.append(Instruction(self.extra_instruction))
        if self.mutate_hp:
            target.side_two.pokemon[1].hp -= 1
        return SimpleNamespace(
            state=target,
            selected_instructions=SimpleNamespace(instruction_list=instructions),
        )


class FakeTracker:
    def __init__(self, *, terminal=False):
        self.applied = 0
        self.terminal = terminal

    def fork(self):
        return copy.deepcopy(self)

    def apply_switch_projection(self, projection):
        # This is the security boundary: observer code never receives the
        # simulator's mechanical states, even though the search result retains
        # them privately for continuation.
        if projection.next_states:
            raise AssertionError("mechanical states crossed observer boundary")
        self.applied += 1
        return {
            "text_tokens": [11, 22],
            "numbers": [0.25, 1.0],
            "illegal_actions": [True] * 13 if self.terminal else [False] + [True] * 12,
            "name_table": {"tackle": 0},
            "terminal": self.terminal,
            "ignored_private_key": "sentinelitem",
        }


class FakeSemanticEngine:
    def __init__(self, *, unaccounted=(), hard_switch=False):
        self.unaccounted = list(unaccounted)
        self.hard_switch = hard_switch

    @staticmethod
    def r1_semantic_contract():
        return "poke-engine-0.0.47-r1-item-activation-v1"

    @staticmethod
    def root_options(*, state):
        return ["hyperbeam", "switch bulbasaur"], ["tackle", "switch charizard"]

    def step_with_uniform_r1_semantic(
        self, source, side_one_action, side_two_action, _u
    ):
        target = copy.deepcopy(source)
        events = []
        if self.hard_switch:
            target.side_two.active_index = "1"
            events.append(
                SimpleNamespace(
                    kind="switch", side="side_two", pokemon_index="1",
                    move_id=None, amount=None, detail=None,
                )
            )
        target.side_two.pokemon[int(target.side_two.active_index)].hp = 75
        target.side_one.pokemon[0].hp = 80
        events.extend(
            [
                SimpleNamespace(
                    kind="action_executed", side="side_one", pokemon_index="0",
                    move_id="HYPERBEAM", amount=None, detail=None,
                ),
                SimpleNamespace(
                    kind="damage", side="side_two",
                    pokemon_index=target.side_two.active_index, move_id=None,
                    amount=25, detail=None,
                ),
                SimpleNamespace(
                    kind="action_executed", side="side_two",
                    pokemon_index=target.side_two.active_index, move_id="TACKLE",
                    amount=None, detail=None,
                ),
                SimpleNamespace(
                    kind="damage", side="side_one", pokemon_index="0",
                    move_id=None, amount=20, detail=None,
                ),
            ]
        )
        return SimpleNamespace(
            state=target,
            events=events,
            unaccounted_instruction_kinds=self.unaccounted,
        )


class R1PublicEventTests(unittest.TestCase):
    def test_opponent_nonfull_hp_that_rounds_to_100_displays_as_99(self):
        target = SimpleNamespace(hp=300, maxhp=301)
        self.assertEqual(_public_hp_fraction(target, "self"), 300 / 301)
        self.assertEqual(_public_hp_fraction(target, "opponent"), 0.99)

    def test_hidden_worlds_project_identical_masked_switch_events(self):
        worlds = [state(), state(hidden_item="sentinelitem")]
        result = project_information_set_switch(
            FakeEngine(),
            worlds,
            "switch bulbasaur",
            "switch charizard",
            0.5,
        )
        self.assertEqual([event.actor for event in result.events], ["self", "opponent"])
        self.assertEqual([event.species for event in result.events], ["bulbasaur", "charizard"])
        self.assertNotIn("sentinelitem", repr(result.events))
        self.assertTrue(all(event.hp_fraction == 1.0 for event in result.events))

    def test_any_unsupported_world_rejects_whole_information_set(self):
        with self.assertRaisesRegex(PublicEventProjectionError, "UNSUPPORTED_INFORMATION_SET"):
            project_information_set_switch(
                FakeEngine(extra_instruction="Damage SideOne: 1"),
                [state(), state(hidden_item="sentinelitem")],
                "switch bulbasaur",
                "switch charizard",
                0.5,
            )

    def test_unlisted_mechanical_delta_is_rejected_without_detail(self):
        with self.assertRaisesRegex(PublicEventProjectionError, "UNSUPPORTED_INFORMATION_SET") as caught:
            project_information_set_switch(
                FakeEngine(mutate_hp=True),
                [state()],
                "switch bulbasaur",
                "switch charizard",
                0.5,
            )
        self.assertEqual(str(caught.exception), "UNSUPPORTED_INFORMATION_SET")

    def test_move_turn_is_not_admitted(self):
        with self.assertRaisesRegex(PublicEventProjectionError, "UNSUPPORTED_INFORMATION_SET"):
            project_information_set_switch(
                FakeEngine(), [state()], "tackle", "switch charizard", 0.5
            )

    def test_transformer_bridge_strips_mechanical_states_and_partitions_worlds(self):
        worlds = [state(), state(hidden_item="sentinelitem")]
        root_tracker = FakeTracker()
        result = project_information_set_observations(
            FakeEngine(),
            worlds,
            root_tracker,
            "switch bulbasaur",
            "switch charizard",
            0.5,
        )
        self.assertEqual(len(result.observation_classes), 1)
        observation_class = result.observation_classes[0]
        self.assertEqual(observation_class.source_world_indices, (0, 1))
        self.assertEqual(len(observation_class.next_states), 2)
        self.assertEqual(root_tracker.applied, 0)
        self.assertEqual(observation_class.tracker.applied, 1)
        self.assertNotIn("sentinelitem", repr(result))
        self.assertNotIn("sentinelitem", repr(observation_class.observation.policy_payload()))
        self.assertNotIn("ignored_private_key", observation_class.observation.policy_payload())

    def test_transformer_bridge_requires_terminal_for_all_illegal_mask(self):
        class InvalidTracker(FakeTracker):
            def apply_switch_projection(self, projection):
                payload = super().apply_switch_projection(projection)
                payload["illegal_actions"] = [True] * 13
                payload["terminal"] = False
                return payload

        with self.assertRaisesRegex(
            PublicEventProjectionError, "INVALID_TRANSFORMER_OBSERVATION"
        ):
            project_information_set_observations(
                FakeEngine(),
                [state()],
                InvalidTracker(),
                "switch bulbasaur",
                "switch charizard",
                0.5,
            )

    def test_semantic_trace_admits_unlisted_moves_with_items_abilities_and_boosts(self):
        world = state()
        world.side_one.pokemon[0].moves[0].id = "hyperbeam"
        world.side_one.pokemon[0].item = "choicespecs"
        world.side_one.pokemon[0].ability = "blaze"
        world.side_one.attack_boost = 1
        result = project_information_set_transition(
            FakeSemanticEngine(), [world], "hyperbeam", "tackle", 0.5
        )
        self.assertEqual(result.certificate, SEMANTIC_TRACE_CERTIFICATE)
        self.assertEqual(len(result.observation_classes), 1)
        self.assertEqual(
            [event.kind for event in result.observation_classes[0].events],
            ["move", "hp", "move", "hp"],
        )

    def test_semantic_trace_rejects_any_unaccounted_instruction(self):
        with self.assertRaisesRegex(
            PublicEventProjectionError, "UNSUPPORTED_INFORMATION_SET"
        ):
            project_information_set_transition(
                FakeSemanticEngine(unaccounted=["ApplyVolatileStatus"]),
                [state()],
                "hyperbeam",
                "tackle",
                0.5,
            )

    def test_semantic_trace_can_reveal_opponent_hard_switch(self):
        result = project_information_set_transition(
            FakeSemanticEngine(hard_switch=True),
            [state()],
            "hyperbeam",
            "switch charizard",
            0.5,
        )
        self.assertEqual(result.certificate, SEMANTIC_TRACE_CERTIFICATE)
        switch = result.observation_classes[0].events[0]
        self.assertEqual((switch.kind, switch.actor, switch.species), (
            "switch", "opponent", "charizard"
        ))
        self.assertFalse(switch.previously_revealed)


if __name__ == "__main__":
    unittest.main()
