from __future__ import annotations

import hashlib
import contextlib
import io
import inspect
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

from srcs.metagross import decision_harness
from srcs.metagross import launch
from srcs.metagross import prior_server
from srcs.metagross import run_foul_play


class LaunchTest(unittest.TestCase):
    @staticmethod
    def holdout_result(delta=0.2, pairs=64, catastrophic_count=0):
        baseline_sum = 0.4 * pairs
        candidate_sum = baseline_sum + delta * pairs
        return {
            "pairs": pairs,
            "baseline_sum": baseline_sum,
            "candidate_sum": candidate_sum,
            "delta_sum": delta * pairs,
            "delta_squared_sum": delta * delta * pairs,
            "catastrophic_count": catastrophic_count,
            "candidate_catastrophic_count": catastrophic_count,
            "baseline_catastrophic_count": 0,
            "candidate_catastrophic_severity_sum": 0.5 * catastrophic_count,
            "baseline_catastrophic_severity_sum": 0.0,
            "candidate_better_count": (
                pairs - catastrophic_count if delta > 0 else 0
            ),
            "baseline_better_count": (
                catastrophic_count if delta > 0 else pairs if delta < 0 else 0
            ),
            "equal_count": pairs if delta == 0 else 0,
            "baseline_terminal_count": 0,
            "candidate_terminal_count": 0,
            "baseline_nonterminal_evaluation_delta_sum": 0.0,
            "candidate_nonterminal_evaluation_delta_sum": delta * pairs,
            "baseline_nonterminal_count": pairs,
            "candidate_nonterminal_count": pairs,
            "continuation_iterations_executed": 2 * pairs * 64,
        }

    @staticmethod
    def mcts_result(options):
        return SimpleNamespace(
            side_one=[
                SimpleNamespace(move_choice=move, total_score=score, visits=visits)
                for move, score, visits in options
            ]
        )

    @staticmethod
    def choice_battle(**active_overrides):
        active = {
            "name": "testmon",
            "hp": 100,
            "max_hp": 100,
            "status": None,
            "item": "leftovers",
            "ability": "pressure",
            "boosts": {},
            "volatile_statuses": [],
        }
        active.update(active_overrides)
        return SimpleNamespace(
            battle_tag="battle-choice-test",
            user=SimpleNamespace(
                active=SimpleNamespace(**active), reserve=[], side_conditions={}
            ),
            opponent=SimpleNamespace(
                active=SimpleNamespace(
                    name="target",
                    hp=100,
                    max_hp=100,
                    status=None,
                    item="unknownitem",
                    ability=None,
                    boosts={},
                    volatile_statuses=[],
                ),
                reserve=[],
                side_conditions={},
            ),
        )

    def test_decision_harness_controller_preserves_golden_choice(self):
        battle = self.choice_battle(hp=100)
        results = [
            (
                self.mcts_result(
                    [("earthquake", 180.0, 200), ("recover", 20.0, 100)]
                ),
                1.0,
                0,
            )
        ]
        priors = [("recover", 0.9), ("earthquake", 0.1)]
        controller = decision_harness.CallableController(
            run_foul_play.select_final_choice
        )

        direct = run_foul_play.select_final_choice(
            battle, results, priors, {}, record_history=False
        )
        composed = controller.select(
            battle, results, priors, histories={}, record_history=False
        )

        self.assertEqual(direct, composed)
        self.assertEqual(composed[0], "recover")
        self.assertEqual(composed[1]["raw_choice"], "earthquake")
        self.assertEqual(
            composed[1]["reason"], "incomplete_or_nonpositive_evidence"
        )

    def test_production_harness_wires_existing_algorithms(self):
        with mock.patch.dict(
            os.environ, {"METAGROSS_PRIOR_SERVER": "http://127.0.0.1:8977"}
        ):
            harness = run_foul_play.build_decision_harness()

        self.assertIs(harness.belief.expand_fn, run_foul_play._prepare_search_battles)
        self.assertIs(harness.search.evaluate_fn, run_foul_play._remote_mcts_batch)
        self.assertIs(harness.search.holdout_fn, run_foul_play._remote_holdout_batch)
        self.assertIs(
            harness.controller.select_fn, run_foul_play.select_final_choice
        )
        self.assertIs(
            harness.verifier.certify_fn, run_foul_play.robust_holdout_certificate
        )
        self.assertIs(
            harness.verifier.combine_fn,
            run_foul_play.combined_robust_holdout_certificate,
        )

    def test_incomplete_favorable_world_evidence_falls_back_to_policy_baseline(self):
        results = [
            (
                self.mcts_result([("earthquake", 180.0, 200), ("recover", 20.0, 100)]),
                0.5,
                0,
            ),
            (self.mcts_result([("earthquake", 180.0, 200)]), 0.5, 1),
        ]

        choice, telemetry = run_foul_play.select_final_choice(
            self.choice_battle(hp=50),
            results,
            [("recover", 0.9), ("earthquake", 0.1)],
            {},
        )

        self.assertEqual(choice, "recover")
        self.assertEqual(telemetry["baseline"], "recover")
        self.assertEqual(telemetry["raw_choice"], "earthquake")
        self.assertEqual(telemetry["reason"], "incomplete_or_nonpositive_evidence")
        self.assertEqual(telemetry["coverage"], 0.5)
        self.assertFalse(telemetry["evidence"]["qualified"])

    def test_request_valid_policy_switch_wins_when_engine_omits_switches(self):
        battle = self.choice_battle(hp=50)
        battle.user.active.moves = []
        battle.user.reserve = [SimpleNamespace(name="blissey", hp=300)]
        results = [
            (
                self.mcts_result([("outrage", 90.0, 100)]),
                1.0,
                0,
            )
        ]

        choice, telemetry = run_foul_play.select_final_choice(
            battle,
            results,
            [("switch blissey", 0.8), ("outrage", 0.2)],
            {},
        )

        self.assertEqual(choice, "switch blissey")
        self.assertEqual(telemetry["reason"], "policy_baseline_missing_from_search")
        self.assertEqual(telemetry["missing_request_actions"], ["switch blissey"])

    def test_empty_authoritative_request_actions_return_no_move(self):
        battle = self.choice_battle(hp=50)
        battle.user.active.moves = []

        choice, telemetry = run_foul_play.select_final_choice(
            battle,
            [(self.mcts_result([("outrage", 90.0, 100)]), 1.0, 0)],
            [("outrage", 1.0)],
            {},
        )

        self.assertEqual(choice, "no move")
        self.assertEqual(telemetry["request_actions"], [])

    def test_policy_switch_form_alias_uses_authorized_engine_name(self):
        battle = self.choice_battle(hp=50)
        battle.user.active.moves = []
        battle.user.reserve = [SimpleNamespace(name="weezinggalar", hp=200)]

        choice, telemetry = run_foul_play.select_final_choice(
            battle,
            [(self.mcts_result([("switch weezinggalar", 50.0, 100)]), 1.0, 0)],
            [("switch weezing", 1.0)],
            {},
        )

        self.assertEqual(choice, "switch weezinggalar")
        self.assertEqual(telemetry["baseline"], "switch weezinggalar")

        battle.user.reserve = [SimpleNamespace(name="ogerponhearthflame", hp=200)]
        choice, _telemetry = run_foul_play.select_final_choice(
            battle,
            [
                (
                    self.mcts_result([("switch ogerponhearthflame", 50.0, 100)]),
                    1.0,
                    0,
                )
            ],
            [("switch ogerpon", 1.0)],
            {},
        )
        self.assertEqual(choice, "switch ogerponhearthflame")

    def test_adaptive_mcts_evidence_is_telemetry_only(self):
        results = [
            (
                self.mcts_result([("earthquake", 180.0, 200), ("recover", 20.0, 100)]),
                1.0,
                0,
            )
        ]

        choice, telemetry = run_foul_play.select_final_choice(
            self.choice_battle(hp=50),
            results,
            [("recover", 0.9), ("earthquake", 0.1)],
            {},
        )

        self.assertEqual(choice, "recover")
        self.assertEqual(telemetry["reason"], "incomplete_or_nonpositive_evidence")
        self.assertEqual(telemetry["coverage"], 1.0)
        self.assertTrue(telemetry["evidence"]["heuristic_qualified"])
        self.assertFalse(telemetry["evidence"]["qualified"])
        self.assertGreater(
            telemetry["evidence"]["paired_lower_confidence_bound"],
            run_foul_play.MIN_PAIRED_ADVANTAGE,
        )

    def test_strong_independent_holdout_certificate_admits_one_frozen_candidate(self):
        results = [self.holdout_result() for _ in range(16)]
        certificate = run_foul_play.independent_holdout_certificate(
            results, [1.0] * len(results), "earthquake", "recover", 0
        )

        self.assertTrue(certificate["qualified"])
        self.assertEqual(certificate["fresh_worlds"], 16)
        self.assertGreater(certificate["paired_lower_confidence_bound"], 0.02)
        self.assertTrue(all(certificate["checks"].values()))

        battle = self.choice_battle(hp=50)
        battle.user.active.moves = [
            SimpleNamespace(name="earthquake", disabled=False),
            SimpleNamespace(name="recover", disabled=False),
        ]
        histories = {}
        choice, telemetry = run_foul_play.select_final_choice(
            battle,
            [
                (
                    self.mcts_result(
                        [("earthquake", 180.0, 200), ("recover", 20.0, 100)]
                    ),
                    1.0,
                    0,
                )
            ],
            [("recover", 0.9), ("earthquake", 0.1)],
            histories,
            {"earthquake": certificate},
        )
        self.assertEqual(choice, "earthquake")
        self.assertEqual(
            telemetry["reason"], "independent_holdout_qualified_search_override"
        )
        self.assertTrue(telemetry["search_override_admitted"])
        self.assertEqual(len(histories[battle.battle_tag]), 1)

    def test_holdout_abstains_on_low_coverage_heterogeneity_and_downside(self):
        cases = {
            "too_few_worlds": [self.holdout_result() for _ in range(8)],
            "heterogeneous": [
                self.holdout_result(0.4 if index < 12 else -0.4) for index in range(16)
            ],
            "catastrophic": [
                self.holdout_result(catastrophic_count=2) for _ in range(16)
            ],
        }
        for name, results in cases.items():
            with self.subTest(name=name):
                certificate = run_foul_play.independent_holdout_certificate(
                    results, [1.0] * len(results), "earthquake", "recover", 5
                )
                self.assertFalse(certificate["qualified"])

    def test_robust_holdout_uses_symmetric_risk_and_evaluator_calibration(self):
        results = [self.holdout_result() for _ in range(16)]
        hashes = [hashlib.sha256(f"state-{index}".encode()).hexdigest() for index in range(16)]
        certificate = run_foul_play.robust_holdout_certificate(
            results,
            [1.0] * 16,
            hashes,
            hashes,
            "earthquake",
            "recover",
            0,
            1,
            0,
        )

        self.assertTrue(certificate["qualified"])
        self.assertTrue(all(certificate["checks"].values()))
        self.assertEqual(certificate["raw_results"], results)
        self.assertEqual(certificate["state_hashes"], hashes)

        evaluator_regression = [
            {
                **row,
                "candidate_nonterminal_evaluation_delta_sum": -row["pairs"],
            }
            for row in results
        ]
        rejected = run_foul_play.robust_holdout_certificate(
            evaluator_regression,
            [1.0] * 16,
            hashes,
            hashes,
            "earthquake",
            "recover",
            0,
            1,
            0,
        )
        self.assertFalse(rejected["qualified"])
        self.assertFalse(rejected["checks"]["evaluator_calibration"])

    def test_holdout_alpha_budget_is_global_across_decisions_and_all_looks(self):
        spent = 0.0
        for sequence_index in range(10_000):
            for rank in range(1, run_foul_play.HOLDOUT_CANDIDATE_COUNT + 1):
                for horizon_index in range(
                    len(run_foul_play.HOLDOUT_CONTINUATION_HORIZONS)
                ):
                    spent += (
                        run_foul_play.holdout_alpha(
                            sequence_index, rank, horizon_index
                        )
                        * run_foul_play.HOLDOUT_ALPHA_CHECKS_PER_LOOK
                    )
        self.assertLessEqual(spent, run_foul_play.HOLDOUT_ALPHA_BUDGET)
        self.assertLess(
            run_foul_play.holdout_alpha(1, 1, 0),
            run_foul_play.holdout_alpha(0, 1, 0),
        )

    def test_adaptive_robust_horizons_stop_rejected_candidates_early(self):
        results = [self.holdout_result(-0.2) for _ in range(16)]
        hashes = [hashlib.sha256(f"state-{index}".encode()).hexdigest() for index in range(16)]
        rejected = run_foul_play.robust_holdout_certificate(
            results,
            [1.0] * 16,
            hashes,
            hashes,
            "earthquake",
            "recover",
            0,
            1,
            0,
        )
        combined = run_foul_play.combined_robust_holdout_certificate({1: rejected})

        self.assertFalse(combined["qualified"])
        self.assertEqual(combined["executed_horizons"], [1])
        self.assertEqual(combined["stop_reason"], "rejected_at_horizon_1")

        passing = dict(rejected, qualified=True)
        with self.assertRaisesRegex(ValueError, "cannot stop early"):
            run_foul_play.combined_robust_holdout_certificate({1: passing})

    def test_top_k_selection_admits_highest_ranked_qualified_candidate(self):
        battle = self.choice_battle(hp=50)
        battle.user.active.moves = [
            SimpleNamespace(name="earthquake", disabled=False),
            SimpleNamespace(name="icebeam", disabled=False),
            SimpleNamespace(name="recover", disabled=False),
        ]
        results = [
            (
                self.mcts_result(
                    [
                        ("earthquake", 90.0, 100),
                        ("icebeam", 80.0, 90),
                        ("recover", 20.0, 50),
                    ]
                ),
                1.0,
                0,
            )
        ]
        evidence = {
            "earthquake": {"qualified": False, "coverage": 1.0},
            "icebeam": {"qualified": True, "coverage": 1.0},
        }

        choice, telemetry = run_foul_play.select_final_choice(
            battle,
            results,
            [("recover", 1.0)],
            {},
            evidence,
        )

        self.assertEqual(choice, "icebeam")
        self.assertTrue(telemetry["search_override_admitted"])
        self.assertEqual(
            telemetry["reason"], "independent_holdout_qualified_search_override"
        )

    def test_multi_horizon_holdout_requires_every_horizon_to_qualify(self):
        certificate = run_foul_play.independent_holdout_certificate(
            [self.holdout_result() for _ in range(16)],
            [1.0] * 16,
            "earthquake",
            "recover",
            0,
        )
        combined = run_foul_play.combined_holdout_certificate(
            {1: certificate, 2: dict(certificate)}
        )
        self.assertTrue(combined["qualified"])
        self.assertEqual(combined["horizons"], [1, 2])

        rejected = dict(certificate)
        rejected["qualified"] = False
        combined = run_foul_play.combined_holdout_certificate(
            {1: certificate, 2: rejected}
        )
        self.assertFalse(combined["qualified"])
        with self.assertRaisesRegex(ValueError, "every required horizon"):
            run_foul_play.combined_holdout_certificate({1: certificate})

    def test_provisional_selection_does_not_mutate_history(self):
        histories = {}
        battle = self.choice_battle(hp=50)
        run_foul_play.select_final_choice(
            battle,
            [(self.mcts_result([("recover", 1.0, 1)]), 1.0, 0)],
            [("recover", 1.0)],
            histories,
            record_history=False,
        )
        self.assertEqual(histories, {})

    def test_small_adaptive_mcts_advantage_does_not_override_policy(self):
        results = [
            (
                self.mcts_result([("earthquake", 72.0, 100), ("recover", 65.0, 100)]),
                1.0,
                0,
            )
        ]

        choice, telemetry = run_foul_play.select_final_choice(
            self.choice_battle(hp=50),
            results,
            [("recover", 0.9), ("earthquake", 0.1)],
            {},
        )

        self.assertEqual(choice, "recover")
        self.assertEqual(telemetry["reason"], "incomplete_or_nonpositive_evidence")

    def test_adaptive_search_cannot_authorize_magic_bounce_prediction(self):
        battle = self.choice_battle(hp=50)
        battle.opponent.active.ability = "magicbounce"
        results = [
            (
                self.mcts_result([("toxic", 190.0, 200), ("earthquake", 10.0, 100)]),
                1.0,
                0,
            )
        ]

        choice, _telemetry = run_foul_play.select_final_choice(
            battle,
            results,
            [("earthquake", 0.9), ("toxic", 0.1)],
            {},
        )

        self.assertEqual(choice, "earthquake")

    def test_repeated_magic_bounce_prediction_is_stopped_without_progress(self):
        battle = self.choice_battle(hp=50)
        battle.opponent.active.ability = "magicbounce"
        results = [
            (
                self.mcts_result([("toxic", 190.0, 200), ("earthquake", 10.0, 100)]),
                1.0,
                0,
            )
        ]
        histories = {}
        choice, telemetry = run_foul_play.select_final_choice(
            battle,
            results,
            [("toxic", 0.9), ("earthquake", 0.1)],
            histories,
            independent_evidence={"earthquake": {"qualified": True}},
        )

        self.assertEqual(choice, "earthquake")
        self.assertEqual(telemetry["reason"], "unqualified_revealed_magic_bounce")

    def test_canary_turn_5_allows_wish_without_a_pending_wish(self):
        battle = self.choice_battle(hp=50)
        battle.turn = 5
        battle.user.wish = (0, 50)

        choice, telemetry = run_foul_play.select_final_choice(
            battle,
            [(self.mcts_result([("wish", 90.0, 100), ("tackle", 1.0, 10)]), 1.0, 0)],
            [("wish", 0.9), ("tackle", 0.1)],
            {},
        )

        self.assertEqual(choice, "wish")
        self.assertEqual(telemetry["reason"], "policy_baseline")

    def test_canary_turn_6_rejects_wish_while_one_is_pending(self):
        battle = self.choice_battle(hp=50)
        battle.turn = 6
        battle.user.wish = (1, 50)

        choice, telemetry = run_foul_play.select_final_choice(
            battle,
            [(self.mcts_result([("wish", 90.0, 100), ("tackle", 1.0, 10)]), 1.0, 0)],
            [("wish", 0.9), ("tackle", 0.1)],
            {},
        )

        self.assertEqual(choice, "tackle")
        self.assertEqual(telemetry["reason"], "guaranteed_noop_wish_already_pending")

    def test_canary_turn_26_rejects_encore_after_the_target_switched(self):
        battle = self.choice_battle(hp=50)
        battle.turn = 26
        battle.opponent.last_used_move = SimpleNamespace(
            pokemon_name=None, move="switch target", turn=25
        )
        battle.opponent.active.moves = []
        moves = {"encore": {"flags": {"failencore": 1}}}

        with mock.patch.object(
            run_foul_play, "_showdown_move_data", side_effect=moves.get
        ):
            choice, telemetry = run_foul_play.select_final_choice(
                battle,
                [
                    (
                        self.mcts_result([("encore", 90.0, 100), ("tackle", 1.0, 10)]),
                        1.0,
                        0,
                    )
                ],
                [("encore", 0.9), ("tackle", 0.1)],
                {},
            )

        self.assertEqual(choice, "tackle")
        self.assertEqual(
            telemetry["reason"], "guaranteed_noop_encore_target_just_switched"
        )

    def test_canary_turn_46_rejects_a_move_blocked_by_revealed_bulletproof(self):
        battle = self.choice_battle(hp=50)
        battle.turn = 46
        battle.opponent.active.ability = "bulletproof"
        moves = {
            "aurasphere": {
                "category": "special",
                "flags": {"bullet": 1},
                "target": "normal",
                "type": "fighting",
            },
            "tackle": {
                "category": "physical",
                "flags": {"contact": 1},
                "target": "normal",
                "type": "normal",
            },
        }

        with mock.patch.object(
            run_foul_play, "_showdown_move_data", side_effect=moves.get
        ):
            choice, telemetry = run_foul_play.select_final_choice(
                battle,
                [
                    (
                        self.mcts_result(
                            [("aurasphere", 90.0, 100), ("tackle", 1.0, 10)]
                        ),
                        1.0,
                        0,
                    )
                ],
                [("aurasphere", 0.9), ("tackle", 0.1)],
                {},
                independent_evidence={"tackle": {"qualified": True}},
            )

        self.assertEqual(choice, "tackle")
        self.assertEqual(
            telemetry["reason"], "unqualified_revealed_bulletproof_immunity"
        )

    def test_encore_allows_an_eligible_observed_last_move(self):
        battle = self.choice_battle(hp=50)
        battle.opponent.last_used_move = SimpleNamespace(
            pokemon_name="target", move="tackle", turn=10
        )
        battle.opponent.active.moves = [SimpleNamespace(name="tackle", current_pp=10)]
        moves = {
            "encore": {"flags": {"failencore": 1}},
            "tackle": {"flags": {}},
        }

        with mock.patch.object(
            run_foul_play, "_showdown_move_data", side_effect=moves.get
        ):
            self.assertIsNone(run_foul_play._known_noop_reason(battle, "encore"))

    def test_encore_rejects_missing_incompatible_and_already_encored_targets(self):
        battle = self.choice_battle(hp=50)
        battle.opponent.active.moves = [
            SimpleNamespace(name="sleeptalk", current_pp=10)
        ]
        cases = [
            (
                SimpleNamespace(pokemon_name="", move="", turn=0),
                [],
                "encore_no_eligible_last_move",
            ),
            (
                SimpleNamespace(pokemon_name="target", move="sleeptalk", turn=10),
                [],
                "encore_last_move_incompatible",
            ),
            (
                SimpleNamespace(pokemon_name="target", move="sleeptalk", turn=10),
                ["encore"],
                "encore_target_already_encored",
            ),
        ]
        moves = {
            "encore": {"flags": {"failencore": 1}},
            "sleeptalk": {"flags": {"failencore": 1}},
        }

        with mock.patch.object(
            run_foul_play, "_showdown_move_data", side_effect=moves.get
        ):
            for last_used, volatiles, expected in cases:
                with self.subTest(expected=expected):
                    battle.opponent.last_used_move = last_used
                    battle.opponent.active.volatile_statuses = volatiles
                    self.assertEqual(
                        run_foul_play._known_noop_reason(battle, "encore"), expected
                    )

    def test_copied_trace_ability_uses_the_revealed_current_ability(self):
        battle = self.choice_battle(hp=50)
        battle.opponent.active.ability = "waterabsorb"
        battle.opponent.active.original_ability = "trace"
        move_data = {
            "category": "special",
            "flags": {"protect": 1},
            "target": "normal",
            "type": "water",
        }

        with mock.patch.object(
            run_foul_play, "_showdown_move_data", return_value=move_data
        ):
            self.assertEqual(
                run_foul_play._prediction_sensitive_reason(battle, "surf"),
                "revealed_waterabsorb_immunity",
            )

    def test_tera_effective_type_can_remove_or_add_an_immunity(self):
        battle = self.choice_battle(hp=50)
        battle.opponent.active.types = ["ground"]
        battle.opponent.active.terastallized = True
        battle.opponent.active.tera_type = "water"
        move_data = {
            "category": "special",
            "flags": {"protect": 1},
            "target": "normal",
            "type": "electric",
        }

        def effectiveness(_move_type, target_types):
            return 0 if "ground" in target_types else 1

        with (
            mock.patch.object(
                run_foul_play, "_showdown_move_data", return_value=move_data
            ),
            mock.patch.object(
                run_foul_play, "_type_effectiveness_modifier", side_effect=effectiveness
            ),
        ):
            self.assertIsNone(
                run_foul_play._prediction_sensitive_reason(battle, "thunderbolt")
            )
            battle.opponent.active.types = ["water"]
            battle.opponent.active.tera_type = "ground"
            self.assertEqual(
                run_foul_play._prediction_sensitive_reason(battle, "thunderbolt"),
                "known_type_immunity",
            )

    def test_mold_breaker_bypasses_revealed_ability_immunity(self):
        battle = self.choice_battle(hp=50, ability="moldbreaker")
        battle.opponent.active.ability = "waterabsorb"
        waterfall = {
            "category": "physical",
            "flags": {"contact": 1},
            "target": "normal",
            "type": "water",
        }
        toxic = {
            "category": "status",
            "flags": {"reflectable": 1},
            "target": "normal",
            "type": "poison",
        }

        with mock.patch.object(
            run_foul_play, "_showdown_move_data", return_value=waterfall
        ):
            self.assertIsNone(
                run_foul_play._prediction_sensitive_reason(battle, "waterfall")
            )
        battle.opponent.active.ability = "magicbounce"
        with mock.patch.object(
            run_foul_play, "_showdown_move_data", return_value=toxic
        ):
            self.assertIsNone(
                run_foul_play._prediction_sensitive_reason(battle, "toxic")
            )

    def test_current_target_immunity_uses_switch_as_prediction_safe_fallback(self):
        battle = self.choice_battle(hp=50)
        battle.opponent.active.ability = "waterabsorb"
        battle.user.active.moves = [SimpleNamespace(name="surf", disabled=False)]
        battle.user.reserve = [SimpleNamespace(name="blissey", hp=100)]
        move_data = {
            "category": "special",
            "flags": {"protect": 1},
            "target": "normal",
            "type": "water",
        }

        with mock.patch.object(
            run_foul_play, "_showdown_move_data", return_value=move_data
        ):
            choice, telemetry = run_foul_play.select_final_choice(
                battle,
                [
                    (
                        self.mcts_result(
                            [("surf", 90.0, 100), ("switch blissey", 1.0, 10)]
                        ),
                        1.0,
                        0,
                    )
                ],
                [("surf", 0.9), ("switch blissey", 0.1)],
                {},
                independent_evidence={
                    "switch blissey": {"qualified": True}
                },
            )

        self.assertEqual(choice, "switch blissey")
        self.assertEqual(
            telemetry["reason"], "unqualified_revealed_waterabsorb_immunity"
        )

    def test_prediction_guard_records_unqualified_fallback(self):
        battle = self.choice_battle(hp=50)
        battle.opponent.active.ability = "waterabsorb"
        battle.user.active.moves = [SimpleNamespace(name="surf", disabled=False)]
        battle.user.reserve = [SimpleNamespace(name="blissey", hp=100)]
        move_data = {
            "category": "special",
            "flags": {"protect": 1},
            "target": "normal",
            "type": "water",
        }

        with mock.patch.object(
            run_foul_play, "_showdown_move_data", return_value=move_data
        ):
            choice, telemetry = run_foul_play.select_final_choice(
                battle,
                [
                    (
                        self.mcts_result(
                            [("surf", 90.0, 100), ("switch blissey", 1.0, 10)]
                        ),
                        1.0,
                        0,
                    )
                ],
                [("surf", 0.9), ("switch blissey", 0.1)],
                {},
                independent_evidence={
                    "switch blissey": {"qualified": False}
                },
            )

        self.assertEqual(choice, "surf")
        self.assertEqual(
            telemetry["blocked_safeguard"],
            {
                "reason": "revealed_waterabsorb_immunity",
                "cause": "no_qualified_replacement",
                "candidates": ["switch blissey"],
            },
        )

    def test_guaranteed_noop_pathologies_choose_a_safe_action(self):
        cases = [
            ({"hp": 25}, "substitute", "substitute_insufficient_hp"),
            (
                {"hp": 50, "status": "slp", "item": "leftovers"},
                "rest",
                "rest_already_asleep_without_usable_chesto",
            ),
            (
                {"hp": 50, "boosts": {"special-attack": 6}},
                "nastyplot",
                "capped_boost",
            ),
        ]
        for active, blocked, reason in cases:
            with self.subTest(blocked=blocked):
                results = [
                    (
                        self.mcts_result([(blocked, 90.0, 100), ("tackle", 1.0, 10)]),
                        1.0,
                        0,
                    )
                ]
                choice, telemetry = run_foul_play.select_final_choice(
                    self.choice_battle(**active),
                    results,
                    [(blocked, 0.9), ("tackle", 0.1)],
                    {},
                )
                self.assertEqual(choice, "tackle")
                self.assertEqual(telemetry["reason"], f"guaranteed_noop_{reason}")

    def test_safety_replacement_cannot_select_request_forbidden_tera(self):
        battle = self.choice_battle(hp=25)
        battle.user.active.moves = [
            SimpleNamespace(name="substitute", disabled=False),
            SimpleNamespace(name="tackle", disabled=False),
        ]
        battle.user.active.can_terastallize = False
        results = [
            (
                self.mcts_result(
                    [
                        ("substitute", 90.0, 100),
                        ("tackle-tera", 70.0, 80),
                        ("tackle", 5.0, 10),
                    ]
                ),
                1.0,
                0,
            )
        ]

        choice, _telemetry = run_foul_play.select_final_choice(
            battle,
            results,
            [("substitute", 0.9), ("tackle", 0.1)],
            {},
        )

        self.assertEqual(choice, "tackle")

    def test_repeated_cycle_without_qualified_replacement_is_recorded(self):
        battle = self.choice_battle(hp=100)
        results = [
            (
                self.mcts_result([("recover", 50.0, 100), ("tackle", 90.0, 100)]),
                1.0,
                0,
            )
        ]
        histories = {}
        choices = [
            run_foul_play.select_final_choice(
                battle,
                results,
                [("recover", 0.9), ("tackle", 0.1)],
                histories,
            )
            for _ in range(3)
        ]

        self.assertEqual([row[0] for row in choices], ["recover"] * 3)
        self.assertEqual(
            choices[-1][1]["blocked_safeguard"],
            {
                "reason": "repeated_no_progress_period_1",
                "cause": "no_qualified_replacement",
                "candidates": ["tackle"],
            },
        )

    def test_one_known_sucker_punch_dodge_is_allowed_then_noop_is_blocked(self):
        battle = self.choice_battle(hp=50, boosts={"attack": 6})
        battle.opponent.active.moves = [SimpleNamespace(name="suckerpunch")]
        results = [
            (
                self.mcts_result(
                    [("swordsdance", 90.0, 100), ("earthquake", 10.0, 20)]
                ),
                1.0,
                0,
            )
        ]
        histories = {}

        first, _ = run_foul_play.select_final_choice(
            battle,
            results,
            [("swordsdance", 0.9), ("earthquake", 0.1)],
            histories,
        )
        second, telemetry = run_foul_play.select_final_choice(
            battle,
            results,
            [("swordsdance", 0.9), ("earthquake", 0.1)],
            histories,
        )

        self.assertEqual(first, "swordsdance")
        self.assertEqual(second, "earthquake")
        self.assertEqual(telemetry["reason"], "guaranteed_noop_capped_boost")

    def test_tera_side_effect_is_not_treated_as_a_guaranteed_noop(self):
        battle = self.choice_battle(hp=25, terastallized=False)

        self.assertIsNone(run_foul_play._known_noop_reason(battle, "substitute-tera"))

    def test_no_progress_cycle_allows_one_repeat_then_uses_safe_evidence(self):
        battle = self.choice_battle(hp=50)
        results = [
            (
                self.mcts_result(
                    [("willowisp", 40.0, 200), ("switch target", 90.0, 100)]
                ),
                1.0,
                0,
            )
        ]
        histories = {}
        choices = [
            run_foul_play.select_final_choice(
                battle,
                results,
                [("willowisp", 0.9), ("switch target", 0.1)],
                histories,
                independent_evidence={
                    "switch target": {"qualified": True}
                },
            )
            for _ in range(3)
        ]

        self.assertEqual(
            [row[0] for row in choices],
            ["willowisp", "willowisp", "switch target"],
        )
        self.assertEqual(choices[2][1]["reason"], "repeated_no_progress_period_1")

    def test_no_progress_cycle_detector_covers_periods_one_two_and_three(self):
        states = [object(), object(), object()]
        actions = ["a", "b", "c"]
        for period in (1, 2, 3):
            cycle = list(zip(states[:period], actions[:period]))
            self.assertIsNone(
                run_foul_play._repeated_no_progress_period(cycle, states[0], actions[0])
            )
            self.assertEqual(
                run_foul_play._repeated_no_progress_period(
                    cycle * 2, states[0], actions[0]
                ),
                period,
            )

    def test_semantic_no_progress_breaks_repeated_attack_despite_hp_changes(self):
        battle = self.choice_battle(hp=100)
        results = [
            (
                self.mcts_result(
                    [("ragingbull", 180.0, 200), ("switch backup", 20.0, 100)]
                ),
                1.0,
                0,
            )
        ]
        histories = {}
        choices = []
        for hp in (100, 80, 60):
            battle.user.active.hp = hp
            choices.append(
                run_foul_play.select_final_choice(
                    battle,
                    results,
                    [("ragingbull", 0.9), ("switch backup", 0.1)],
                    histories,
                    independent_evidence={
                        "switch backup": {"qualified": True}
                    },
                )
            )

        self.assertEqual(
            [row[0] for row in choices], ["ragingbull", "ragingbull", "switch backup"]
        )
        self.assertEqual(
            choices[-1][1]["reason"], "semantic_no_progress_repeated_action"
        )

    def test_semantic_no_progress_blocks_consecutive_protect(self):
        battle = self.choice_battle(hp=100)
        results = [
            (
                self.mcts_result([("protect", 180.0, 200), ("tackle", 20.0, 100)]),
                1.0,
                0,
            )
        ]
        histories = {}

        first, _ = run_foul_play.select_final_choice(
            battle,
            results,
            [("protect", 0.9), ("tackle", 0.1)],
            histories,
        )
        battle.user.active.hp = 90
        second, telemetry = run_foul_play.select_final_choice(
            battle,
            results,
            [("protect", 0.9), ("tackle", 0.1)],
            histories,
            independent_evidence={"tackle": {"qualified": True}},
        )

        self.assertEqual(first, "protect")
        self.assertEqual(second, "tackle")
        self.assertEqual(
            telemetry["reason"], "semantic_no_progress_consecutive_protect"
        )

    def test_semantic_no_progress_allows_protect_with_residual_progress(self):
        battle = self.choice_battle(hp=100)
        results = [
            (
                self.mcts_result([("protect", 180.0, 200), ("tackle", 20.0, 100)]),
                1.0,
                0,
            )
        ]
        histories = {}

        first, _ = run_foul_play.select_final_choice(
            battle,
            results,
            [("protect", 0.9), ("tackle", 0.1)],
            histories,
        )
        battle.opponent.active.hp = 90
        second, _ = run_foul_play.select_final_choice(
            battle,
            results,
            [("protect", 0.9), ("tackle", 0.1)],
            histories,
        )

        self.assertEqual((first, second), ("protect", "protect"))

    def test_semantic_no_progress_allows_short_recovery_sequence(self):
        battle = self.choice_battle(hp=100)
        results = [
            (
                self.mcts_result([("recover", 180.0, 200), ("tackle", 20.0, 100)]),
                1.0,
                0,
            )
        ]
        histories = {}
        choices = []
        for hp in (100, 90, 80, 70, 60, 50):
            battle.user.active.hp = hp
            choices.append(
                run_foul_play.select_final_choice(
                    battle,
                    results,
                    [("recover", 0.9), ("tackle", 0.1)],
                    histories,
                    independent_evidence={"tackle": {"qualified": True}},
                )
            )

        self.assertEqual([row[0] for row in choices[:5]], ["recover"] * 5)
        self.assertEqual(choices[-1][0], "tackle")
        self.assertEqual(
            choices[-1][1]["reason"], "semantic_no_progress_repeated_action"
        )

    def test_semantic_no_progress_allows_recovery_while_opponent_loses_hp(self):
        battle = self.choice_battle(hp=100)
        results = [
            (
                self.mcts_result([("recover", 180.0, 200), ("tackle", 20.0, 100)]),
                1.0,
                0,
            )
        ]
        histories = {}
        choices = []
        for hp in (100, 90, 80, 70, 60, 50):
            battle.opponent.active.hp = hp
            choices.append(
                run_foul_play.select_final_choice(
                    battle,
                    results,
                    [("recover", 0.9), ("tackle", 0.1)],
                    histories,
                )[0]
            )

        self.assertEqual(choices, ["recover"] * 6)

    def test_semantic_no_progress_allows_stat_progress_then_breaks(self):
        battle = self.choice_battle(hp=100)
        results = [
            (
                self.mcts_result(
                    [("strengthsap", 180.0, 200), ("shadowball", 20.0, 100)]
                ),
                1.0,
                0,
            )
        ]
        histories = {}
        choices = []
        for attack in (0, -1, -2, -2, -2):
            battle.opponent.active.boosts["attack"] = attack
            choices.append(
                run_foul_play.select_final_choice(
                    battle,
                    results,
                    [("strengthsap", 0.9), ("shadowball", 0.1)],
                    histories,
                    independent_evidence={"shadowball": {"qualified": True}},
                )[0]
            )

        self.assertEqual(choices[:4], ["strengthsap"] * 4)
        self.assertEqual(choices[-1], "shadowball")

    def test_verified_search_override_breaks_switch_carousel(self):
        battle = self.choice_battle(hp=100)
        histories = {}
        choices = []
        for active, target in (
            ("first", "second"),
            ("second", "third"),
            ("third", "fourth"),
            ("fourth", "fifth"),
        ):
            battle.user.active.name = active
            results = [
                (
                    self.mcts_result(
                        [("nastyplot", 180.0, 200), (f"switch {target}", 20.0, 100)]
                    ),
                    1.0,
                    0,
                )
            ]
            choices.append(
                run_foul_play.select_final_choice(
                    battle,
                    results,
                    [(f"switch {target}", 0.9), ("nastyplot", 0.1)],
                    histories,
                    independent_evidence=(
                        {"nastyplot": {"qualified": True}}
                        if target == "fifth"
                        else None
                    ),
                )
            )

        self.assertEqual(
            [row[0] for row in choices[:3]],
            ["switch second", "switch third", "switch fourth"],
        )
        self.assertEqual(choices[-1][0], "nastyplot")
        self.assertEqual(
            choices[-1][1]["reason"],
            "independent_holdout_qualified_search_override",
        )
        self.assertTrue(choices[-1][1]["search_override_admitted"])

    def test_repeated_sacrificial_pivot_is_blocked_after_sweeper_boost(self):
        battle = self.choice_battle(hp=100)
        battle.user.reserve = [
            SimpleNamespace(name="recipient", hp=100, max_hp=100),
            SimpleNamespace(name="backup", hp=100, max_hp=100),
        ]
        results = [
            (
                self.mcts_result([("uturn", 180.0, 200), ("thunderbolt", 20.0, 100)]),
                1.0,
                0,
            )
        ]
        histories = {}

        first, _ = run_foul_play.select_final_choice(
            battle, results, [("uturn", 0.9), ("thunderbolt", 0.1)], histories
        )
        battle.user.reserve[0].hp = 0
        battle.opponent.active.boosts["attack"] = 1
        second, telemetry = run_foul_play.select_final_choice(
            battle,
            results,
            [("uturn", 0.9), ("thunderbolt", 0.1)],
            histories,
            independent_evidence={"thunderbolt": {"qualified": True}},
        )

        self.assertEqual(first, "uturn")
        self.assertEqual(second, "thunderbolt")
        self.assertEqual(
            telemetry["reason"],
            "semantic_no_progress_repeated_sacrificial_pivot",
        )

    def test_sacrificial_pivot_guard_requires_loss_and_offensive_boost(self):
        for lose_recipient, boost_opponent in ((True, False), (False, True)):
            with self.subTest(
                lose_recipient=lose_recipient, boost_opponent=boost_opponent
            ):
                battle = self.choice_battle(hp=100)
                battle.user.reserve = [
                    SimpleNamespace(name="recipient", hp=100, max_hp=100)
                ]
                results = [
                    (
                        self.mcts_result(
                            [("uturn", 180.0, 200), ("thunderbolt", 20.0, 100)]
                        ),
                        1.0,
                        0,
                    )
                ]
                histories = {}
                run_foul_play.select_final_choice(
                    battle,
                    results,
                    [("uturn", 0.9), ("thunderbolt", 0.1)],
                    histories,
                )
                if lose_recipient:
                    battle.user.reserve[0].hp = 0
                if boost_opponent:
                    battle.opponent.active.boosts["attack"] = 1

                choice, _ = run_foul_play.select_final_choice(
                    battle,
                    results,
                    [("uturn", 0.9), ("thunderbolt", 0.1)],
                    histories,
                )

                self.assertEqual(choice, "uturn")

    def test_losing_stall_switches_after_hp_loss_and_opponent_boosts(self):
        battle = self.choice_battle(hp=100)
        battle.user.reserve = [
            SimpleNamespace(name="backup", hp=100, max_hp=100)
        ]
        histories = {battle.battle_tag: []}
        for user_hp, opponent_hp, attack, choice in (
            (100, 100, 0, "facade"),
            (85, 90, 1, "flareblitz"),
            (75, 100, 2, "facade"),
            (65, 95, 3, "flareblitz"),
        ):
            battle.user.active.hp = user_hp
            battle.opponent.active.hp = opponent_hp
            battle.opponent.active.boosts["attack"] = attack
            histories[battle.battle_tag].append(
                (run_foul_play.public_battle_state(battle), choice)
            )
        battle.user.active.hp = 50
        battle.opponent.active.hp = 100
        battle.opponent.active.boosts["attack"] = 4
        results = [
            (
                self.mcts_result(
                    [("facade", 180.0, 200), ("switch backup", 20.0, 100)]
                ),
                1.0,
                0,
            )
        ]

        choice, telemetry = run_foul_play.select_final_choice(
            battle,
            results,
            [("facade", 0.9), ("switch backup", 0.1)],
            histories,
            independent_evidence={"switch backup": {"qualified": True}},
        )

        self.assertEqual(choice, "switch backup")
        self.assertEqual(
            telemetry["reason"], "semantic_no_progress_losing_stall"
        )

    def test_losing_stall_keeps_agreed_action_when_switch_fails_holdout(self):
        battle = self.choice_battle(hp=100)
        battle.user.reserve = [
            SimpleNamespace(name="backup", hp=100, max_hp=100)
        ]
        histories = {battle.battle_tag: []}
        for user_hp, opponent_hp, attack in (
            (100, 100, 0),
            (85, 90, 1),
            (75, 100, 2),
            (65, 95, 3),
        ):
            battle.user.active.hp = user_hp
            battle.opponent.active.hp = opponent_hp
            battle.opponent.active.boosts["attack"] = attack
            histories[battle.battle_tag].append(
                (run_foul_play.public_battle_state(battle), "facade")
            )
        battle.user.active.hp = 50
        battle.opponent.active.hp = 100
        battle.opponent.active.boosts["attack"] = 4
        results = [
            (
                self.mcts_result(
                    [("facade", 180.0, 200), ("switch backup", 20.0, 100)]
                ),
                1.0,
                0,
            )
        ]

        choice, telemetry = run_foul_play.select_final_choice(
            battle,
            results,
            [("facade", 0.9), ("switch backup", 0.1)],
            histories,
            independent_evidence={"switch backup": {"qualified": False}},
        )

        self.assertEqual(choice, "facade")
        self.assertEqual(telemetry["reason"], "policy_baseline")
        self.assertEqual(
            telemetry["blocked_safeguard"],
            {
                "reason": "losing_stall",
                "cause": "no_qualified_replacement",
                "candidates": ["switch backup"],
            },
        )

    def test_losing_stall_requires_opponent_boost_and_no_hp_progress(self):
        battle = self.choice_battle(hp=100)
        battle.user.reserve = [
            SimpleNamespace(name="backup", hp=100, max_hp=100)
        ]
        histories = {battle.battle_tag: []}
        for user_hp, opponent_hp in ((100, 100), (85, 80), (70, 60), (55, 40)):
            battle.user.active.hp = user_hp
            battle.opponent.active.hp = opponent_hp
            histories[battle.battle_tag].append(
                (run_foul_play.public_battle_state(battle), "facade")
            )
        battle.user.active.hp = 40
        battle.opponent.active.hp = 20

        self.assertFalse(
            run_foul_play._losing_stall(
                histories[battle.battle_tag],
                run_foul_play.public_battle_state(battle),
            )
        )

    def test_terminal_pivot_without_reserve_uses_nonpivot_action(self):
        battle = self.choice_battle(hp=100)
        results = [
            (
                self.mcts_result([("uturn", 180.0, 200), ("woodhammer", 20.0, 100)]),
                1.0,
                0,
            )
        ]

        move_data = {
            "uturn": {"basePower": 70, "category": "Physical"},
            "woodhammer": {"basePower": 120, "category": "Physical"},
        }
        with mock.patch.object(
            run_foul_play,
            "_showdown_move_data",
            side_effect=lambda move: move_data.get(move),
        ):
            choice, telemetry = run_foul_play.select_final_choice(
                battle,
                results,
                [("uturn", 0.9), ("woodhammer", 0.1)],
                {},
            )

        self.assertEqual(choice, "woodhammer")
        self.assertEqual(telemetry["reason"], "terminal_pivot_without_reserve")

    def test_terminal_damaging_pivot_kept_without_credible_attack(self):
        battle = self.choice_battle(hp=100)
        results = [
            (
                self.mcts_result([("flipturn", 180.0, 200), ("yawn", 20.0, 100)]),
                1.0,
                0,
            )
        ]

        move_data = {
            "flipturn": {"basePower": 60, "category": "Physical"},
            "yawn": {"basePower": 0, "category": "Status"},
        }
        with mock.patch.object(
            run_foul_play,
            "_showdown_move_data",
            side_effect=lambda move: move_data.get(move),
        ):
            choice, telemetry = run_foul_play.select_final_choice(
                battle,
                results,
                [("flipturn", 0.9), ("yawn", 0.1)],
                {},
            )

        self.assertEqual(choice, "flipturn")
        self.assertEqual(telemetry["reason"], "policy_baseline")

    def test_progress_state_includes_timers_and_sleep_counters(self):
        battle = self.choice_battle(hp=50, sleep_turns=1, rest_turns=2)
        battle.weather_turns_remaining = 3
        battle.field_turns_remaining = 4
        battle.trick_room_turns_remaining = 2

        before = run_foul_play.public_battle_state(battle)
        battle.user.active.sleep_turns = 2
        after_sleep = run_foul_play.public_battle_state(battle)
        battle.weather_turns_remaining = 2
        after_weather = run_foul_play.public_battle_state(battle)

        self.assertNotEqual(before, after_sleep)
        self.assertNotEqual(after_sleep, after_weather)

        battle.user.wish = (1, 50)
        after_wish = run_foul_play.public_battle_state(battle)
        self.assertNotEqual(after_weather, after_wish)

    def test_repeated_capped_swords_dance_uses_best_mcts_score(self):
        battle = SimpleNamespace(
            battle_tag="battle-test",
            user=SimpleNamespace(
                active=SimpleNamespace(name="seviper", boosts={"attack": 6})
            ),
        )
        results = [
            (
                self.mcts_result(
                    [
                        ("swordsdance", 14.0, 100),
                        ("trailblaze", 12.0, 100),
                        ("earthquake", 13.5, 100),
                    ]
                ),
                1.0,
                0,
            )
        ]
        streaks = {}

        first, first_override = run_foul_play.guard_repeated_capped_swords_dance(
            battle, "swordsdance", results, streaks
        )
        second, second_override = run_foul_play.guard_repeated_capped_swords_dance(
            battle, "swordsdance", results, streaks
        )

        self.assertEqual(first, "swordsdance")
        self.assertIsNone(first_override)
        self.assertEqual(second, "earthquake")
        self.assertEqual(second_override["reason"], "repeated_capped_swords_dance")
        self.assertEqual(second_override["capped_streak"], 2)

    def test_capped_swords_dance_streak_resets_after_another_choice(self):
        battle = SimpleNamespace(
            battle_tag="battle-test",
            user=SimpleNamespace(
                active=SimpleNamespace(name="seviper", boosts={"attack": 6})
            ),
        )
        results = [(self.mcts_result([("swordsdance", 1.0, 1)]), 1.0, 0)]
        streaks = {}

        run_foul_play.guard_repeated_capped_swords_dance(
            battle, "swordsdance", results, streaks
        )
        run_foul_play.guard_repeated_capped_swords_dance(
            battle, "earthquake", results, streaks
        )
        choice, override = run_foul_play.guard_repeated_capped_swords_dance(
            battle, "swordsdance", results, streaks
        )

        self.assertEqual(choice, "swordsdance")
        self.assertIsNone(override)

    def test_empty_prior_legality_mask_recovers_only_plausible_actions(self):
        import numpy as np

        illegal = np.ones(13, dtype=bool)
        fallback, error = prior_server.recover_empty_legality_mask(illegal, [0, 5, 9])

        self.assertTrue(fallback)
        self.assertEqual(error, "no definitely valid actions")
        self.assertEqual(np.flatnonzero(~illegal).tolist(), [0, 5, 9])

    def test_empty_prior_legality_mask_without_candidates_fails_closed(self):
        import numpy as np

        with self.assertRaisesRegex(RuntimeError, "potentially valid"):
            prior_server.recover_empty_legality_mask(np.ones(13, dtype=bool), [])

    def test_nonempty_prior_legality_mask_is_unchanged(self):
        import numpy as np

        illegal = np.ones(13, dtype=bool)
        illegal[4] = False
        fallback, error = prior_server.recover_empty_legality_mask(illegal)

        self.assertFalse(fallback)
        self.assertIsNone(error)
        self.assertFalse(illegal[4])
        self.assertEqual(int(illegal.sum()), 12)

    def test_prior_request_requires_matching_unconsumed_rqid(self):
        self.assertEqual(
            prior_server.correlated_request_rqid(True, {"rqid": 17}, 17), 17
        )
        with self.assertRaisesRegex(RuntimeError, "unconsumed"):
            prior_server.correlated_request_rqid(False, {"rqid": 17}, 17)
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            prior_server.correlated_request_rqid(True, {"rqid": 18}, 17)

    def test_opponent_priors_require_complete_public_action_support(self):
        active = SimpleNamespace(
            unique_id="active",
            active=True,
            moves={f"move-{index}": object() for index in range(4)},
        )
        team = {
            "active": active,
            **{
                f"reserve-{index}": SimpleNamespace(
                    unique_id=f"reserve-{index}", active=False
                )
                for index in range(5)
            },
        }
        battle = SimpleNamespace(
            opponent_active_pokemon=active, opponent_team=team
        )

        self.assertTrue(prior_server.opponent_action_support_complete(battle))
        active.moves.pop("move-3")
        self.assertFalse(prior_server.opponent_action_support_complete(battle))
        active.moves["move-3"] = object()
        team.pop("reserve-4")
        self.assertFalse(prior_server.opponent_action_support_complete(battle))

    def test_opponent_prior_sanitizer_drops_active_and_unknown_actions(self):
        battle = SimpleNamespace(
            opponent=SimpleNamespace(
                active=SimpleNamespace(
                    name="hatterene",
                    moves=[
                        SimpleNamespace(name="drainingkiss"),
                        SimpleNamespace(name="psyshock"),
                    ],
                    terastallized=False,
                ),
                reserve=[
                    SimpleNamespace(name="arceusrock", hp=100),
                    SimpleNamespace(name="fainted", hp=0),
                ],
                tera_used=False,
            )
        )

        sanitized = dict(
            run_foul_play.sanitize_opponent_priors(
                battle,
                {
                    "switch hatterene": 0.3,
                    "switch arceusrock": 0.2,
                    "switch fainted": 0.1,
                    "drainingkiss": 0.2,
                    "drainingkiss-tera": 0.1,
                    "psyshock": 0.05,
                    "psyshock-tera": 0.05,
                    "energyball": 0.0,
                },
            )
            or []
        )

        self.assertEqual(
            set(sanitized),
            {
                "switch arceusrock",
                "drainingkiss",
                "drainingkiss-tera",
                "psyshock",
                "psyshock-tera",
            },
        )
        self.assertAlmostEqual(sum(sanitized.values()), 1.0)
        self.assertIsNone(
            run_foul_play.sanitize_opponent_priors(
                battle, {"switch hatterene": 1.0}
            )
        )

    def test_r1_remains_default(self):
        args = launch.parse_args(["--username", "bot"])
        self.assertEqual(args.profile, "r1")
        self.assertEqual(args.games, 200)
        self.assertEqual(args.search_parallelism, launch.DEFAULT_SEARCH_PARALLELISM)
        self.assertEqual(args.search_threads, launch.DEFAULT_SEARCH_THREADS)
        self.assertEqual(launch.SEARCH_TIME_MS, 500)
        self.assertEqual(launch.CPU_C_PUCT, 2.0)

    def test_search_parallelism_is_configurable_and_positive(self):
        args = launch.parse_args(
            ["--username", "bot", "--search-parallelism", "5", "--search-threads", "1"]
        )
        self.assertEqual(args.search_parallelism, 5)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launch.parse_args(["--username", "bot", "--search-parallelism", "0"])

    def test_remote_mcts_requires_pinned_engine_hash(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launch.parse_args(["--username", "bot", "--remote-mcts"])
        args = launch.parse_args(
            [
                "--username",
                "bot",
                "--remote-mcts",
                "--remote-engine-sha256",
                "a" * 64,
            ]
        )
        self.assertTrue(args.remote_mcts)
        self.assertEqual(args.remote_mcts_app, launch.DEFAULT_REMOTE_MCTS_APP)
        self.assertEqual(args.remote_mcts_function, launch.DEFAULT_REMOTE_MCTS_FUNCTION)

    def test_launcher_preflights_remote_search_and_holdout_before_matchmaking(self):
        args = launch.parse_args(
            [
                "--username",
                "bot",
                "--remote-mcts",
                "--remote-engine-sha256",
                "a" * 64,
            ]
        )
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "operations": ["search", "paired_holdout"]}),
            stderr="",
        )

        with mock.patch.object(launch.subprocess, "run", return_value=completed) as run:
            result = launch.preflight_remote_mcts(args)

        self.assertTrue(result["ok"])
        command = run.call_args.args[0]
        self.assertIn("srcs.metagross.remote_mcts_preflight", command)
        self.assertIn("paired_holdout", result["operations"])

    def test_http_remote_mcts_requires_loopback_url_and_environment_token(self):
        base = [
            "--username",
            "bot",
            "--remote-mcts",
            "--remote-mcts-transport",
            "http",
            "--remote-engine-sha256",
            "a" * 64,
            "--remote-mcts-instance-type",
            "c7a.8xlarge",
        ]
        with mock.patch.dict(os.environ, {}, clear=True):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                launch.parse_args(base)
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                launch.parse_args(
                    [*base, "--remote-mcts-url", "http://example.com/search"]
                )
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                launch.parse_args(
                    [*base, "--remote-mcts-url", "http://127.0.0.1:8765/search"]
                )
        with mock.patch.dict(
            os.environ, {"METAGROSS_REMOTE_MCTS_TOKEN": "s" * 32}, clear=True
        ):
            args = launch.parse_args(
                [*base, "--remote-mcts-url", "http://127.0.0.1:8765/search"]
            )
        self.assertEqual(args.remote_mcts_transport, "http")
        self.assertEqual(args.remote_mcts_url, "http://127.0.0.1:8765/search")
        self.assertEqual(args.remote_mcts_instance_type, "c7a.8xlarge")

    def test_http_client_posts_bearer_json_without_network(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, _limit):
                return json.dumps([{"ok": True}]).encode()

        environment = {
            "METAGROSS_REMOTE_MCTS_URL": "http://127.0.0.1:8765/search",
            "METAGROSS_REMOTE_MCTS_TOKEN": "secret",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("urllib.request.urlopen", return_value=Response()) as urlopen,
        ):
            self.assertEqual(
                run_foul_play._http_mcts_call([{"schema": 1}]), [{"ok": True}]
            )
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(json.loads(request.data), [{"schema": 1}])
        self.assertNotIn("secret", request.full_url)

    def test_g4_requires_explicit_three_game_canary(self):
        for extra in ([], ["--games", "3"], ["--confirm-g4-canary"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launch.parse_args(["--username", "bot", "--profile", "g4", *extra])
        args = launch.parse_args(
            [
                "--username",
                "bot",
                "--profile",
                "g4",
                "--games",
                "3",
                "--confirm-g4-canary",
            ]
        )
        self.assertEqual(args.games, 3)

    def test_g3_requires_explicit_three_game_canary(self):
        for extra in ([], ["--games", "3"], ["--confirm-g3-canary"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launch.parse_args(["--username", "bot", "--profile", "g3", *extra])
        args = launch.parse_args(
            [
                "--username",
                "bot",
                "--profile",
                "g3",
                "--games",
                "3",
                "--confirm-g3-canary",
            ]
        )
        self.assertEqual(args.games, 3)

    def test_candidate_continuation_is_explicit_and_bounded(self):
        for profile in ("g3", "g4"):
            args = launch.parse_args(
                [
                    "--username",
                    "bot",
                    "--profile",
                    profile,
                    "--games",
                    "25",
                    "--confirm-candidate-continuation",
                ]
            )
            self.assertEqual(args.games, 25)
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                launch.parse_args(
                    [
                        "--username",
                        "bot",
                        "--profile",
                        profile,
                        "--games",
                        "101",
                        "--confirm-candidate-continuation",
                    ]
                )

    def test_checkpoint_verification_accepts_only_pinned_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content = b"policy"
            profile = launch.PolicyProfile(
                "run", 7, hashlib.sha256(content).hexdigest()
            )
            path = profile.checkpoint_path(root)
            path.parent.mkdir(parents=True)
            path.write_bytes(content)
            verified_path, actual = launch.verify_checkpoint(profile, root)
            self.assertEqual(verified_path, path.resolve())
            self.assertEqual(actual, profile.sha256)
            path.write_bytes(b"other")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                launch.verify_checkpoint(profile, root)

    def test_prior_server_verifies_exact_checkpoint_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "r1" / "ckpts" / "policy_weights" / "policy_epoch_5.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"r1")
            expected = hashlib.sha256(b"r1").hexdigest()
            verified_path, actual = prior_server.verify_local_checkpoint(
                str(root), "r1", 5, expected
            )
            self.assertEqual(verified_path, path.resolve())
            self.assertEqual(actual, expected)
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                prior_server.verify_local_checkpoint(str(root), "r1", 5, "0" * 64)

    def test_profile_records_are_immutable(self):
        with self.assertRaises(Exception):
            launch.POLICY_PROFILES["g4"].checkpoint = 2
        with self.assertRaises(TypeError):
            launch.POLICY_PROFILES["other"] = launch.POLICY_PROFILES["r1"]

    def test_password_is_not_added_to_child_arguments(self):
        production_source = inspect.getsource(launch) + inspect.getsource(
            run_foul_play.main
        )
        self.assertNotIn("--ps-password", production_source)
        self.assertNotIn("sys.argv.extend", production_source)

    def test_experimental_engine_provenance_is_rejected(self):
        provenance = {
            "source_path": "/repo/experimental/engine/pe_v3_learned_priors",
            "editable": True,
            "mcts_parameters": [
                "state",
                "duration_ms",
                "threads",
                "s1_priors",
                "s2_priors",
                "c_puct",
                "seed",
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "source mismatch"):
            run_foul_play.validate_poke_engine_provenance(
                provenance, Path("/repo/srcs/vendor/poke-engine")
            )

    def test_production_environment_removes_experimental_value_model(self):
        source = {
            "KEEP": "yes",
            "METAGROSS_VALUE_MODEL": "model.pt",
            "METAGROSS_LEARNED_VALUE_WEIGHT": "0.5",
            "METAGROSS_REMOTE_MCTS_TOKEN": "secret",
        }
        self.assertEqual(launch.production_environment(source), {"KEEP": "yes"})

    def test_mcts_result_payload_preserves_training_targets(self):
        result = SimpleNamespace(
            side_one=[SimpleNamespace(move_choice="tackle", total_score=3.5, visits=7)],
            side_two=[SimpleNamespace(move_choice="protect", total_score=-1, visits=2)],
            total_visits=9,
        )
        payload = run_foul_play._mcts_result_payload(result)
        self.assertEqual(payload["total_visits"], 9)
        self.assertEqual(
            payload["side_one"],
            [{"move_choice": "tackle", "total_score": 3.5, "visits": 7}],
        )

    def test_mcts_result_payload_round_trip(self):
        engine = SimpleNamespace(
            MctsSideResult=SimpleNamespace, MctsResult=SimpleNamespace
        )
        payload = {
            "side_one": [{"move_choice": "tackle", "total_score": 3.5, "visits": 7}],
            "side_two": [{"move_choice": "protect", "total_score": 3.5, "visits": 7}],
            "total_visits": 9,
        }
        result = run_foul_play._mcts_result_from_payload(payload, engine)
        self.assertEqual(run_foul_play._mcts_result_payload(result), payload)

    def test_mcts_result_rejects_invalid_numeric_values(self):
        engine = SimpleNamespace(
            MctsSideResult=SimpleNamespace, MctsResult=SimpleNamespace
        )
        base = {
            "side_one": [{"move_choice": "tackle", "total_score": 1.0, "visits": 1}],
            "side_two": [],
            "total_visits": 1,
        }
        for field, value in (
            ("visits", -1),
            ("visits", True),
            ("total_score", float("nan")),
        ):
            payload = {**base, "side_one": [{**base["side_one"][0], field: value}]}
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(RuntimeError),
            ):
                run_foul_play._mcts_result_from_payload(payload, engine)

    def test_remote_response_requires_correlation_and_engine_identity(self):
        response = {
            "schema": run_foul_play.REMOTE_MCTS_SCHEMA,
            "request_id": "request",
            "index": 3,
            "ok": True,
            "engine": {
                "contract": run_foul_play.REMOTE_ENGINE_CONTRACT,
                "source_sha256": run_foul_play.ENGINE_SOURCE_SHA256,
                "native_sha256": "a" * 64,
            },
            "result": {},
        }
        with mock.patch.dict(
            "os.environ", {"METAGROSS_REMOTE_ENGINE_SHA256": "a" * 64}, clear=False
        ):
            self.assertIs(
                run_foul_play._validate_remote_response(response, "request", 3),
                response,
            )
            with self.assertRaisesRegex(RuntimeError, "correlation"):
                run_foul_play._validate_remote_response(response, "other", 3)
            response["engine"]["native_sha256"] = "b" * 64
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                run_foul_play._validate_remote_response(response, "request", 3)

    def test_http_response_requires_aws_identity_and_full_telemetry(self):
        response = {
            "schema": run_foul_play.REMOTE_MCTS_SCHEMA,
            "request_id": "request",
            "index": 0,
            "ok": True,
            "engine": {
                "contract": run_foul_play.REMOTE_ENGINE_CONTRACT,
                "source_sha256": run_foul_play.ENGINE_SOURCE_SHA256,
                "native_sha256": "a" * 64,
                "resources": {
                    "provider": "aws_ec2",
                    "instance_type": "c7a.8xlarge",
                    "logical_cpus": 32,
                    "memory_mib": 65536,
                    "worker_processes": 16,
                },
            },
            "timing": {
                "queue_ms": 1.0,
                "validation_ms": 0.1,
                "search_ms": 500.0,
                "worker_ms": 500.1,
                "batch_ms": 501.0,
                "batch_size": 32,
            },
        }
        environment = {
            "METAGROSS_REMOTE_MCTS_TRANSPORT": "http",
            "METAGROSS_REMOTE_ENGINE_SHA256": "a" * 64,
            "METAGROSS_REMOTE_MCTS_INSTANCE_TYPE": "c7a.8xlarge",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertIs(
                run_foul_play._validate_remote_response(response, "request", 0),
                response,
            )
            del response["timing"]["queue_ms"]
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                run_foul_play._validate_remote_response(response, "request", 0)

    def test_remote_holdout_uses_frozen_priors_and_deterministic_provenance(self):
        run_seed = "01" * 32
        context = {"tag": "battle-test", "decision_idx": 4}
        captured = []

        def remote(requests):
            captured.extend(requests)
            return [
                {
                    "schema": run_foul_play.REMOTE_MCTS_SCHEMA,
                    "request_id": request["request_id"],
                    "index": request["index"],
                    "ok": True,
                    "engine": {
                        "contract": run_foul_play.REMOTE_ENGINE_CONTRACT,
                        "source_sha256": run_foul_play.ENGINE_SOURCE_SHA256,
                        "native_sha256": "a" * 64,
                    },
                    "result": self.holdout_result(),
                    "timing": {"search_ms": 1.0, "batch_size": 1},
                }
                for request in requests
            ]

        previous = dict(run_foul_play._PRIOR_STATE)
        run_foul_play._PRIOR_STATE.update(
            {
                "context": context,
                "opp_priors": [("protect", 1.0)],
                "remote_search": {},
            }
        )
        environment = {
            "METAGROSS_RUN_SEED": run_seed,
            "METAGROSS_RNG_SCHEME": run_foul_play.RNG_SCHEME,
            "METAGROSS_REMOTE_ENGINE_SHA256": "a" * 64,
        }
        try:
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(run_foul_play, "_remote_mcts_call", side_effect=remote),
            ):
                results = run_foul_play._remote_holdout_batch(
                    ["state"], "recover", "earthquake", 1, [9], 2
                )
        finally:
            run_foul_play._PRIOR_STATE.clear()
            run_foul_play._PRIOR_STATE.update(previous)

        self.assertEqual(results, [self.holdout_result()])
        self.assertEqual(captured[0]["opponent_priors"], [["protect", 1.0]])
        self.assertEqual(
            captured[0]["request_id"],
            run_foul_play.deterministic_request_id(
                run_seed,
                "battle-test",
                4,
                "2:1:0",
                channel="certification-request",
            ),
        )

    def test_recursive_shadow_holdout_uses_separate_request_and_telemetry_channels(self):
        captured = []

        def remote(requests):
            captured.extend(requests)
            return [
                {
                    "schema": run_foul_play.REMOTE_MCTS_SCHEMA,
                    "request_id": request["request_id"],
                    "index": request["index"],
                    "ok": True,
                    "engine": {
                        "contract": run_foul_play.REMOTE_ENGINE_CONTRACT,
                        "source_sha256": run_foul_play.ENGINE_SOURCE_SHA256,
                        "native_sha256": "a" * 64,
                    },
                    "result": self.holdout_result(),
                    "timing": {"search_ms": 1.0, "batch_size": 1},
                }
                for request in requests
            ]

        previous = dict(run_foul_play._PRIOR_STATE)
        run_foul_play._PRIOR_STATE.update(
            {
                "context": {"tag": "battle-shadow", "decision_idx": 3},
                "opp_priors": None,
                "remote_search": {},
            }
        )
        environment = {
            "METAGROSS_RUN_SEED": "03" * 32,
            "METAGROSS_RNG_SCHEME": run_foul_play.RNG_SCHEME,
            "METAGROSS_REMOTE_ENGINE_SHA256": "a" * 64,
        }
        try:
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(run_foul_play, "_remote_mcts_call", side_effect=remote),
            ):
                run_foul_play._remote_holdout_batch(
                    ["state"],
                    "recover",
                    "earthquake",
                    2,
                    [11],
                    1,
                    request_channel="recursive-shadow-request",
                    telemetry_key="recursive_shadow",
                )
        finally:
            remote_search = run_foul_play._PRIOR_STATE["remote_search"]
            run_foul_play._PRIOR_STATE.clear()
            run_foul_play._PRIOR_STATE.update(previous)

        self.assertEqual(
            captured[0]["request_id"],
            run_foul_play.deterministic_request_id(
                "03" * 32,
                "battle-shadow",
                3,
                "1:2:0",
                channel="recursive-shadow-request",
            ),
        )
        self.assertIn("recursive_shadow", remote_search)
        self.assertNotIn("holdout", remote_search)

    def test_world_sampling_seed_is_channel_separated_and_restores_random(self):
        import random

        battle = SimpleNamespace(team_preview=False, battle_type="random")
        search_main = SimpleNamespace(
            deepcopy=lambda value: value,
            BattleType=SimpleNamespace(
                RANDOM_BATTLE="random",
                BATTLE_FACTORY="factory",
                STANDARD_BATTLE="standard",
            ),
            search_time_num_battles_randombattles=lambda _battle: (2, 250),
            prepare_random_battles=lambda _battle, count, rng=None: [
                ((rng or random).random(), 0.5) for _ in range(count)
            ],
        )
        previous = dict(run_foul_play._PRIOR_STATE)
        run_foul_play._PRIOR_STATE["context"] = {
            "tag": "battle-test",
            "decision_idx": 2,
        }
        environment = {
            "METAGROSS_RUN_SEED": "02" * 32,
            "METAGROSS_RNG_SCHEME": run_foul_play.RNG_SCHEME,
        }
        random.seed(123)
        before = random.getstate()
        try:
            with mock.patch.dict(os.environ, environment, clear=True):
                first = run_foul_play._prepare_search_battles(
                    battle, search_main, "selection-worlds"
                )[0]
                second = run_foul_play._prepare_search_battles(
                    battle, search_main, "selection-worlds"
                )[0]
                holdout = run_foul_play._prepare_search_battles(
                    battle, search_main, "certification-worlds"
                )[0]
        finally:
            run_foul_play._PRIOR_STATE.clear()
            run_foul_play._PRIOR_STATE.update(previous)

        self.assertEqual(first, second)
        self.assertNotEqual(first, holdout)
        self.assertEqual(random.getstate(), before)


if __name__ == "__main__":
    unittest.main()
