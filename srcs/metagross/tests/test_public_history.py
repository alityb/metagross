from __future__ import annotations

import unittest
from types import SimpleNamespace

from srcs.metagross.history_belief import compile_public_belief
from srcs.metagross.history_belief_replay import compare_reconstructed_beliefs
from srcs.metagross.public_history import (
    HazardAvoidanceEvent,
    ItemEvent,
    MoveEvent,
    PublicEventLedger,
    SwitchEvent,
    TeraEvent,
)


def candidate(*, count, moves, item="leftovers", ability="pressure", tera="water"):
    result = SimpleNamespace(
        pkmn_set=SimpleNamespace(
            level=80,
            ability=ability,
            item=item,
            tera_type=tera,
            count=count,
        ),
        pkmn_moveset=SimpleNamespace(moves=tuple(moves)),
    )
    result.full_set_pkmn_can_have_set = lambda pokemon, **_kwargs: (
        set(pokemon.revealed_moves).issubset(result.pkmn_moveset.moves)
        and pokemon.item == result.pkmn_set.item
    )
    return result


class PublicHistoryTest(unittest.TestCase):
    def test_ledger_projects_direct_public_facts_in_order(self):
        ledger = PublicEventLedger("p1")
        events = ledger.extend(
            [
                "|switch|p2a: Foe|Suicune, L80|87/100 par",
                "|move|p2a: Foe|Scald|p1a: Hero",
                "|-item|p2a: Foe|Leftovers",
                "|-terastallize|p2a: Foe|Water",
                "|-damage|p2a: Foe|75/100",
            ]
        )

        self.assertEqual([event.sequence for event in events], [0, 1, 2, 3])
        self.assertIsInstance(events[0], SwitchEvent)
        self.assertEqual(events[0].species, "suicune")
        self.assertEqual(events[0].level, 80)
        self.assertEqual(events[0].hp_fraction, 0.87)
        self.assertIsInstance(events[1], MoveEvent)
        self.assertEqual(events[1].species, "suicune")
        self.assertIsInstance(events[2], ItemEvent)
        self.assertIsInstance(events[3], TeraEvent)

    def test_ledger_snapshot_is_immutable_and_viewpoint_specific(self):
        ledger = PublicEventLedger("p2")
        first = ledger.extend(["|switch|p2a: Hero|Mew, L100|100/100"])
        ledger.append("|move|p2a: Hero|Psychic|p1a: Foe")

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].actor, "self")
        self.assertEqual(len(ledger.events), 2)

    def test_compiler_filters_and_count_weights_without_mutation(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|switch|p2a: Foe|Suicune, L80|100/100",
                "|move|p2a: Foe|Scald|p1a: Hero",
                "|-item|p2a: Foe|Leftovers",
            ]
        )
        candidates = [
            candidate(count=3, moves=("scald", "protect")),
            candidate(count=1, moves=("scald", "icebeam")),
            candidate(count=20, moves=("surf", "protect")),
        ]

        belief = compile_public_belief(ledger.events, {"suicune": candidates})

        result = belief.species[0]
        self.assertEqual(result.status, "compiled")
        self.assertEqual(sorted(row.weight for row in result.candidates), [0.25, 0.75])
        self.assertEqual(len(candidates), 3)

    def test_compiler_fails_closed_on_empty_support(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|switch|p2a: Foe|Suicune, L80|100/100",
                "|move|p2a: Foe|Scald|p1a: Hero",
            ]
        )
        belief = compile_public_belief(
            ledger.events,
            {"suicune": [candidate(count=1, moves=("surf",))]},
        )

        self.assertEqual(belief.species[0].status, "inconsistent")
        self.assertEqual(belief.species[0].candidates, ())

    def test_identity_replacement_is_unsupported_not_guessed(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|switch|p2a: Foe|Zoroark-Hisui, L80|100/100",
                "|replace|p2a: Foe|Pikachu, L80|100/100",
            ]
        )
        belief = compile_public_belief(
            ledger.events,
            {"pikachu": [candidate(count=1, moves=("thunderbolt",))]},
        )

        pikachu = next(row for row in belief.species if row.species == "pikachu")
        zoroark = next(row for row in belief.species if row.species == "zoroarkhisui")
        self.assertEqual(pikachu.status, "unsupported")
        self.assertEqual(zoroark.status, "unsupported")

    def test_transferred_item_is_recorded_but_not_used_as_set_identity(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|switch|p2a: Foe|Suicune, L80|100/100",
                "|-item|p2a: Foe|Choice Scarf|[from] move: Trick",
            ]
        )
        belief = compile_public_belief(
            ledger.events,
            {"suicune": [candidate(count=1, moves=("scald",), item="leftovers")]},
        )

        self.assertFalse(ledger.events[1].stable_identity_evidence)
        self.assertEqual(belief.species[0].status, "compiled")

    def test_knock_off_names_the_starting_item(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|switch|p2a: Foe|Suicune, L80|100/100",
                "|-enditem|p2a: Foe|Leftovers|[from] move: Knock Off",
            ]
        )
        belief = compile_public_belief(
            ledger.events,
            {
                "suicune": [
                    candidate(count=1, moves=("scald",), item="leftovers"),
                    candidate(count=1, moves=("scald",), item="lifeorb"),
                ]
            },
        )

        self.assertTrue(ledger.events[1].stable_identity_evidence)
        self.assertEqual(belief.species[0].candidates[0].item, "leftovers")

    def test_named_item_effect_is_direct_set_identity_evidence(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|switch|p2a: Foe|Suicune, L80|100/100",
                "|-damage|p2a: Foe|90/100|[from] item: Life Orb",
            ]
        )
        belief = compile_public_belief(
            ledger.events,
            {
                "suicune": [
                    candidate(count=1, moves=("scald",), item="lifeorb"),
                    candidate(count=1, moves=("scald",), item="leftovers"),
                ]
            },
        )

        self.assertIsInstance(ledger.events[1], ItemEvent)
        self.assertTrue(ledger.events[1].stable_identity_evidence)
        self.assertEqual(belief.species[0].candidates[0].item, "lifeorb")

    def test_named_effect_from_transferred_item_is_not_set_identity_evidence(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|switch|p2a: Foe|Suicune, L80|100/100",
                "|-item|p2a: Foe|Leftovers|[from] move: Trick",
                "|-heal|p2a: Foe|100/100|[from] item: Leftovers",
            ]
        )
        belief = compile_public_belief(
            ledger.events,
            {
                "suicune": [
                    candidate(count=1, moves=("scald",), item="choiceband"),
                    candidate(count=1, moves=("scald",), item="choicescarf"),
                ]
            },
        )

        self.assertFalse(ledger.events[1].stable_identity_evidence)
        self.assertFalse(ledger.events[2].stable_identity_evidence)
        self.assertEqual(len(belief.species[0].candidates), 2)

    def test_item_effect_owner_comes_from_of_annotation(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|switch|p1a: Hero|Corviknight, L80|100/100",
                "|switch|p2a: Foe|Mienshao, L80|100/100",
                "|-damage|p2a: Foe|84/100|[from] item: Rocky Helmet|[of] p1a: Hero",
            ]
        )

        item = next(event for event in ledger.events if isinstance(event, ItemEvent))
        self.assertEqual(item.actor, "self")
        self.assertEqual(item.species, "corviknight")
        self.assertEqual(item.item_id, "rockyhelmet")

    def test_compiler_rejects_noncontiguous_ledger(self):
        events = (
            SwitchEvent(2, "opponent", "mew", 100, 1.0, None, False, False, None),
        )
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            compile_public_belief(events, {})

    def test_first_switch_without_stealth_rock_damage_certifies_immunity(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|-sidestart|p2: Foe|move: Stealth Rock",
                "|switch|p2a: Foe|Suicune, L80|100/100",
                "|move|p1a: Hero|Protect|p1a: Hero",
            ]
        )
        candidates = [
            candidate(count=2, moves=("scald",), item="heavydutyboots"),
            candidate(count=3, moves=("scald",), ability="magicguard"),
            candidate(count=10, moves=("scald",), item="leftovers"),
        ]

        evidence = next(event for event in ledger.events if isinstance(event, HazardAvoidanceEvent))
        belief = compile_public_belief(ledger.events, {"suicune": candidates})

        self.assertTrue(evidence.avoided)
        self.assertEqual(len(belief.species[0].candidates), 2)
        self.assertEqual(sorted(row.weight for row in belief.species[0].candidates), [0.4, 0.6])

    def test_stealth_rock_damage_rejects_unsuppressed_avoidance_sets(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|-sidestart|p2: Foe|move: Stealth Rock",
                "|switch|p2a: Foe|Suicune, L80|100/100",
                "|-damage|p2a: Foe|88/100|[from] Stealth Rock",
            ]
        )
        candidates = [
            candidate(count=1, moves=("scald",), item="heavydutyboots"),
            candidate(count=1, moves=("scald",), item="leftovers"),
        ]

        evidence = next(event for event in ledger.events if isinstance(event, HazardAvoidanceEvent))
        belief = compile_public_belief(ledger.events, {"suicune": candidates})

        self.assertFalse(evidence.avoided)
        self.assertEqual(len(belief.species[0].candidates), 1)
        self.assertEqual(belief.species[0].candidates[0].item, "leftovers")

    def test_suppressed_effects_do_not_create_negative_hazard_evidence(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|-sidestart|p2: Foe|move: Stealth Rock",
                "|-fieldstart|move: Magic Room",
                "|switch|p1a: Hero|Weezing, L80|100/100",
                "|-ability|p1a: Hero|Neutralizing Gas",
                "|switch|p2a: Foe|Suicune, L80|100/100",
                "|-damage|p2a: Foe|88/100|[from] Stealth Rock",
            ]
        )
        candidates = [
            candidate(count=1, moves=("scald",), item="heavydutyboots"),
            candidate(count=1, moves=("scald",), ability="magicguard"),
        ]

        evidence = next(event for event in ledger.events if isinstance(event, HazardAvoidanceEvent))
        belief = compile_public_belief(ledger.events, {"suicune": candidates})

        self.assertTrue(evidence.item_effects_suppressed)
        self.assertTrue(evidence.ability_effects_suppressed)
        self.assertEqual(len(belief.species[0].candidates), 2)

    def test_later_switch_after_item_removal_is_not_used_as_starting_set_evidence(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|switch|p2a: Foe|Suicune, L80|100/100",
                "|-enditem|p2a: Foe|Heavy-Duty Boots|[from] move: Knock Off",
                "|-sidestart|p2: Foe|move: Stealth Rock",
                "|switch|p2a: Other|Mew, L80|100/100",
                "|-damage|p2a: Other|88/100|[from] Stealth Rock",
                "|switch|p2a: Foe|Suicune, L80|100/100",
                "|move|p1a: Hero|Protect|p1a: Hero",
            ]
        )

        self.assertFalse(
            any(
                isinstance(event, HazardAvoidanceEvent)
                and event.species == "suicune"
                for event in ledger.events
            )
        )

    def test_replay_comparison_reports_exact_strict_parity(self):
        ledger = PublicEventLedger("p1")
        ledger.extend(
            [
                "|switch|p2a: Foe|Suicune, L80|100/100",
                "|move|p2a: Foe|Scald|p1a: Hero",
                "|-item|p2a: Foe|Leftovers",
            ]
        )
        candidates = [
            candidate(count=3, moves=("scald", "protect")),
            candidate(count=1, moves=("scald", "icebeam")),
            candidate(count=20, moves=("surf", "protect")),
        ]
        opponent = SimpleNamespace(
            active=SimpleNamespace(
                name="suicune",
                revealed_moves=("scald",),
                item="leftovers",
                removed_item=None,
                impossible_items=set(),
                impossible_abilities=set(),
                can_have_choice_item=True,
                speed_range=SimpleNamespace(min=0, max=float("inf")),
            ),
            reserve=[],
        )
        battle = SimpleNamespace(
            opponent=opponent,
            _metagross_public_events=ledger.events,
            _metagross_random_battle_sets={"suicune": candidates},
        )

        report = compare_reconstructed_beliefs({("battle-1", 0): battle})

        self.assertEqual(report["support_equal_rows"], 1)
        self.assertEqual(report["mean_total_variation"], 0.0)
        self.assertTrue(
            report["comparisons"][0]["current_derived_constraints"][
                "can_have_choice_item"
            ]
        )


if __name__ == "__main__":
    unittest.main()
