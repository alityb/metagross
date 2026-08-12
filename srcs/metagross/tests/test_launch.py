from __future__ import annotations

import asyncio
import hashlib
import contextlib
import io
import inspect
import json
import os
import stat
import struct
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

from srcs.metagross import decision_harness
from srcs.metagross import launch
from srcs.metagross import prior_server
from srcs.metagross import run_foul_play
from srcs.metagross.mcts_contract import _fnv1a64_digest


class LaunchTest(unittest.TestCase):
    def test_search_capture_is_written_private(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "search.jsonl"
            with mock.patch.dict(
                os.environ, {"METAGROSS_SEARCH_DUMP": str(path)}, clear=True
            ):
                run_foul_play._append_jsonl("METAGROSS_SEARCH_DUMP", {"value": 1})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

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

    @staticmethod
    def shared_root_result(particles=1, input_weights=None):
        policy = [
            {
                "action": "earthquake",
                "probability": 0.7,
                "counterfactual_value": 0.625,
            },
            {
                "action": "recover",
                "probability": 0.3,
                "counterfactual_value": 0.5,
            },
        ]
        input_weights = input_weights or [1.0 / particles] * particles
        payoffs = [row["counterfactual_value"] for row in policy]
        expected_value = sum(
            row["probability"] * row["counterfactual_value"] for row in policy
        )
        best_response = max(payoffs)
        none_digest = _fnv1a64_digest([b"none"])
        continuations = []
        for payoff in payoffs:
            total_score = payoff * 8
            continuations.append(
                [
                    {
                        "seed": 11,
                        "requested_iterations": 8,
                        "executed_iterations": 8,
                        "visits": 8,
                        "total_score": total_score,
                        "total_score_bits": struct.unpack(
                            "<I", struct.pack("<f", total_score)
                        )[0],
                        "payoff": payoff,
                        "payoff_bits": struct.unpack("<Q", struct.pack("<d", payoff))[0],
                    }
                ]
            )
        result = {
            "policy": policy,
            "opponent_policies": [
                [{"action": "protect", "probability": 1.0}]
            ],
            "diagnostics": {
                "solver_contract": "weighted-shared-rm-plus-v1",
                "iterations": 100,
                "continuation_iterations": 8,
                "seed": 7,
                "prior_strength": 1.0,
                "expected_value": expected_value,
                "player_best_response_value": best_response,
                "opponent_best_response_value": expected_value,
                "player_best_response_gain": best_response - expected_value,
                "opponent_best_response_gain": 0.0,
                "nash_conv": best_response - expected_value,
                "exploitability": (best_response - expected_value) / 2,
                "player_regret_bound": 0.0,
                "opponent_regret_bound": 0.0,
                "total_regret_bound": 0.0,
                "payoff_cells": len(policy),
                "total_forced_continuation_iterations": 8 * len(policy),
                "input_particle_count": particles,
                "positive_particle_count": particles,
                "canonical_particle_count": 1,
                "normalized_weight_sum": 1.0,
                "action_support_digest": _fnv1a64_digest(
                    [row["action"].encode() for row in sorted(policy, key=lambda row: row["action"])]
                ),
                "particle_digest": _fnv1a64_digest(
                    [b"state", struct.pack("<d", 1.0)]
                ),
                "payoff_digest": _fnv1a64_digest(
                    [struct.pack("<d", payoff) for payoff in payoffs]
                ),
                "player_prior_digest": none_digest,
                "opponent_prior_digest": none_digest,
            },
            "replay_capture": {
                "schema_version": 1,
                "solver_contract": "weighted-shared-rm-plus-v1",
                "configuration": {
                    "iterations": 100,
                    "continuation_iterations": 8,
                    "seed": 7,
                    "prior_strength": 1.0,
                },
                "own_action_support": sorted(row["action"] for row in policy),
                "normalized_player_prior": None,
                "canonical_particles": [
                    {
                        "canonical_index": 0,
                        "state": "state",
                        "normalized_weight": 1.0,
                        "source_particles": [
                            {"input_index": index, "input_weight": weight}
                            for index, weight in enumerate(input_weights)
                        ],
                        "opponent_action_support": ["protect"],
                        "normalized_opponent_prior": None,
                        "payoff_matrix": [[payoff] for payoff in payoffs],
                        "continuations": continuations,
                        "opponent_policy": [1.0],
                    }
                ],
            },
        }
        return result

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
            harness.search.solve_shared_root_fn,
            run_foul_play._remote_shared_root_batch,
        )
        self.assertIs(
            harness.controller.select_fn, run_foul_play.select_search_first_choice
        )
        self.assertIs(
            harness.controller.select_shared_fn,
            run_foul_play.select_shared_root_choice,
        )
        self.assertIs(
            harness.verifier.certify_fn, run_foul_play.robust_holdout_certificate
        )
        self.assertIs(
            harness.verifier.combine_fn,
            run_foul_play.combined_robust_holdout_certificate,
        )

    def test_certified_controller_remains_explicit_rollback_comparator(self):
        with mock.patch.dict(
            os.environ,
            {
                "METAGROSS_PRIOR_SERVER": "http://127.0.0.1:8977",
                "METAGROSS_CONTROLLER_MODE": "certified",
            },
        ):
            harness = run_foul_play.build_decision_harness()

        self.assertIs(harness.controller.select_fn, run_foul_play.select_final_choice)

    def test_unknown_controller_mode_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported METAGROSS_CONTROLLER_MODE"):
            run_foul_play.controller_select_fn("unknown")

    def test_shared_root_mode_is_explicitly_gated_and_has_rollback(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(run_foul_play.root_search_mode(), "independent_mcts")
        with (
            mock.patch.dict(
                os.environ,
                {"METAGROSS_ROOT_SEARCH_MODE": "shared_rm_plus"},
                clear=True,
            ),
            self.assertRaisesRegex(RuntimeError, "ALLOW_EXPERIMENTAL"),
        ):
            run_foul_play.root_search_mode()
        with mock.patch.dict(
            os.environ,
            {
                "METAGROSS_ROOT_SEARCH_MODE": "shared_rm_plus",
                "METAGROSS_ALLOW_EXPERIMENTAL_SHARED_ROOT": "1",
                "METAGROSS_CONTROLLER_MODE": "search_first",
            },
            clear=True,
        ):
            self.assertEqual(run_foul_play.root_search_mode(), "shared_rm_plus")

    def test_independent_ensemble_mode_is_explicitly_gated(self):
        with (
            mock.patch.dict(
                os.environ,
                {"METAGROSS_ROOT_SEARCH_MODE": "independent_ensemble"},
                clear=True,
            ),
            self.assertRaisesRegex(RuntimeError, "ALLOW_EXPERIMENTAL_ENSEMBLE"),
        ):
            run_foul_play.root_search_mode()
        with mock.patch.dict(
            os.environ,
            {
                "METAGROSS_ROOT_SEARCH_MODE": "independent_ensemble",
                "METAGROSS_ALLOW_EXPERIMENTAL_ENSEMBLE": "1",
                "METAGROSS_REQUIRE_REMOTE_MCTS": "1",
                "METAGROSS_ALLOW_INSECURE_LOOPBACK": "1",
                "METAGROSS_CONTROLLER_MODE": "search_first",
            },
            clear=True,
        ):
            self.assertEqual(run_foul_play.root_search_mode(), "independent_ensemble")

    def test_independent_ensemble_rejects_public_showdown_websocket(self):
        environment = {
            "METAGROSS_ROOT_SEARCH_MODE": "independent_ensemble",
            "METAGROSS_ALLOW_EXPERIMENTAL_ENSEMBLE": "1",
            "METAGROSS_REQUIRE_REMOTE_MCTS": "1",
            "METAGROSS_ALLOW_INSECURE_LOOPBACK": "1",
            "METAGROSS_CONTROLLER_MODE": "search_first",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            run_foul_play.require_local_ensemble_websocket(
                "ws://localhost:8000/showdown/websocket"
            )
            with self.assertRaisesRegex(RuntimeError, "loopback Showdown"):
                run_foul_play.require_local_ensemble_websocket(
                    "wss://sim3.psim.us/showdown/websocket"
                )

    def test_ensemble_results_split_each_world_weight_across_repeats(self):
        results = [object() for _index in range(6)]
        battles = [(object(), 0.75), (object(), 0.25)]
        weighted = run_foul_play._ensemble_weighted_results(results, battles, 3)
        self.assertEqual([row[0] for row in weighted], results)
        self.assertEqual([row[1] for row in weighted], [0.25, 1 / 12] * 3)
        self.assertEqual([row[2] for row in weighted], list(range(6)))
        with self.assertRaisesRegex(ValueError, "world weight"):
            run_foul_play._ensemble_weighted_results(
                results, [(object(), float("nan")), (object(), 0.25)], 3
            )

    def test_adaptive_ensemble_repeats_respect_wire_capacity(self):
        self.assertEqual(run_foul_play._adaptive_ensemble_repeat_count(8), 3)
        self.assertEqual(run_foul_play._adaptive_ensemble_repeat_count(16), 3)
        self.assertEqual(run_foul_play._adaptive_ensemble_repeat_count(21), 3)
        self.assertEqual(run_foul_play._adaptive_ensemble_repeat_count(22), 2)
        self.assertEqual(run_foul_play._adaptive_ensemble_repeat_count(32), 2)
        self.assertEqual(run_foul_play._adaptive_ensemble_repeat_count(64), 1)
        with self.assertRaisesRegex(RuntimeError, "world count"):
            run_foul_play._adaptive_ensemble_repeat_count(65)

    def test_shared_root_selector_samples_mixture_reproducibly(self):
        battle = self.choice_battle()
        result = self.shared_root_result()
        priors = [("recover", 0.9), ("earthquake", 0.1)]

        first = run_foul_play.select_shared_root_choice(
            battle, result, priors, seed=1, histories={}, record_history=False
        )
        second = run_foul_play.select_shared_root_choice(
            battle, result, priors, seed=1, histories={}, record_history=False
        )
        alternate = run_foul_play.select_shared_root_choice(
            battle, result, priors, seed=0, histories={}, record_history=False
        )

        self.assertEqual(first, second)
        self.assertEqual(first[0], "earthquake")
        self.assertEqual(alternate[0], "recover")
        self.assertEqual(first[1]["search_mass_kind"], "shared_policy_probability")
        self.assertEqual(first[1]["mixed_strategy_seed"], 1)
        self.assertEqual(first[1]["sampled_action"], "earthquake")
        self.assertEqual(first[1]["solver_diagnostics"], result["diagnostics"])

    def test_shared_root_selector_rejects_positive_illegal_mass(self):
        battle = self.choice_battle(
            moves=[SimpleNamespace(name="recover", disabled=False)],
            can_terastallize=False,
        )
        with self.assertRaisesRegex(RuntimeError, "request-illegal"):
            run_foul_play.select_shared_root_choice(
                battle,
                self.shared_root_result(),
                [("recover", 1.0)],
                seed=1,
                histories={},
                record_history=False,
            )

    def test_insecure_login_is_limited_to_actual_loopback_websockets(self):
        self.assertTrue(run_foul_play.is_loopback_websocket_uri("ws://localhost:8000/ws"))
        self.assertTrue(run_foul_play.is_loopback_websocket_uri("wss://127.0.0.1/ws"))
        self.assertTrue(run_foul_play.is_loopback_websocket_uri("ws://[::1]:8000/ws"))
        self.assertFalse(run_foul_play.is_loopback_websocket_uri("https://localhost/ws"))
        self.assertFalse(
            run_foul_play.is_loopback_websocket_uri("ws://localhost@example.com/ws")
        )

    def test_liveness_timeouts_are_positive_and_configurable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                run_foul_play.positive_environment_seconds("MISSING", 12.0), 12.0
            )
        for value in ("0", "-1", "nan", "invalid"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"TIMEOUT": value}, clear=True
            ), self.assertRaisesRegex(RuntimeError, "positive number"):
                run_foul_play.positive_environment_seconds("TIMEOUT", 12.0)

    def test_websocket_liveness_contract_enables_pings_and_receive_deadline(self):
        environment = {
            "METAGROSS_WEBSOCKET_KEEPALIVE": "1",
            "METAGROSS_WEBSOCKET_PING_INTERVAL_SECONDS": "20",
            "METAGROSS_WEBSOCKET_PING_TIMEOUT_SECONDS": "60",
            "METAGROSS_WEBSOCKET_RECEIVE_TIMEOUT_SECONDS": "0.01",
        }

        async def never_returns():
            await asyncio.Event().wait()

        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertEqual(
                run_foul_play.websocket_connect_kwargs({}),
                {"ping_interval": 20.0, "ping_timeout": 60.0, "close_timeout": 10},
            )
            with self.assertRaisesRegex(
                TimeoutError, "no Showdown websocket message within 0.01s"
            ):
                asyncio.run(run_foul_play.receive_websocket_message(never_returns))

    def test_reconnect_replay_returns_only_unseen_public_delta_and_request(self):
        room = "battle-gen9randombattle-1"
        messages = [
            "\n".join(
                [
                    f">{room}",
                    "|init|battle",
                    "|title|ReconnectSmoke vs. Opponent",
                    "|gametype|singles",
                    "|player|p1|ReconnectSmoke|1|",
                    "|player|p2|Opponent|1|",
                    "|start",
                    "|turn|3",
                    "|player|p1|ReconnectSmoke|1|",
                ]
            ),
            f'>{room}\n|request|{{"wait":true,"rqid":4}}',
            f">{room}\n|move|p2a: Target|U-turn|p1a: Bot\n|turn|4",
            f'>{room}\n|request|{{"active":[{{"moves":[]}}],"rqid":5}}',
        ]

        previously_seen = ["|gametype|singles", "|start", "|turn|3"]
        chunks, replayed = run_foul_play.reconnect_delta_chunks(
            messages, room, previously_seen
        )

        self.assertEqual(
            chunks,
            [
                f">{room}\n|move|p2a: Target|U-turn|p1a: Bot\n|turn|4",
                f'>{room}\n|request|{{"active":[{{"moves":[]}}],"rqid":5}}',
            ],
        )
        self.assertEqual(
            replayed,
            previously_seen + ["|move|p2a: Target|U-turn|p1a: Bot", "|turn|4"],
        )
        self.assertFalse(run_foul_play.actionable_showdown_request(messages[1]))
        self.assertTrue(run_foul_play.actionable_showdown_request(messages[-1]))

    def test_reconnect_replay_requires_current_request_and_history_prefix(self):
        room = "battle-test"
        with self.assertRaisesRegex(RuntimeError, "no current request"):
            run_foul_play.reconnect_delta_chunks(
                [f">{room}\n|init|battle"], room, []
            )
        with self.assertRaisesRegex(RuntimeError, "does not extend"):
            run_foul_play.reconnect_delta_chunks(
                [
                    f">{room}\n|init|battle\n|turn|2",
                    f">{room}\n|request|{{\"forceSwitch\":[true]}}",
                ],
                room,
                ["|turn|1"],
            )

    def test_reconnect_replay_preserves_wait_request_and_sent_choice(self):
        room = "battle-test"
        message = (
            f">{room}\n|request|{{\"active\":[{{\"moves\":[]}}],\"rqid\":8}}"
            "\n|sentchoice|move scald"
        )
        chunks, replayed = run_foul_play.reconnect_delta_chunks(
            [f">{room}\n|turn|4", message], room, ["|turn|4"]
        )

        self.assertEqual(chunks, [message.split("\n|sentchoice|")[0]])
        self.assertEqual(replayed, ["|turn|4"])
        self.assertEqual(run_foul_play.showdown_request_payload(message)["rqid"], 8)
        self.assertEqual(
            run_foul_play.showdown_sent_choice(message), "move scald"
        )
        self.assertEqual(
            run_foul_play.outbound_choice_identity(["/choose move scald", "8"]),
            ("move scald", 8),
        )
        self.assertEqual(
            run_foul_play.outbound_choice_identity(["/switch 3", "10"]),
            ("switch 3", 10),
        )

    def test_send_recovery_decision_table(self):
        request = {"active": [{"moves": []}], "rqid": 8}
        self.assertEqual(
            run_foul_play.send_recovery_action(request, None, "move scald", 8),
            "resend",
        )
        self.assertEqual(
            run_foul_play.send_recovery_action(
                request, "move scald", "move scald", 8
            ),
            "confirmed",
        )
        self.assertEqual(
            run_foul_play.send_recovery_action(
                {"active": [{"moves": []}], "rqid": 10},
                None,
                "move scald",
                8,
            ),
            "advanced",
        )
        self.assertEqual(
            run_foul_play.send_recovery_action(
                None, None, "move scald", 8, terminal=True
            ),
            "terminal",
        )
        with self.assertRaisesRegex(RuntimeError, "no request object"):
            run_foul_play.send_recovery_action(
                None, None, "move scald", 8
            )
        self.assertEqual(
            run_foul_play.send_recovery_action(
                {"wait": True, "rqid": 8}, None, "move scald", 8
            ),
            "wait",
        )
        with self.assertRaisesRegex(RuntimeError, "different pending choice"):
            run_foul_play.send_recovery_action(
                request, "move recover", "move scald", 8
            )
        with self.assertRaisesRegex(RuntimeError, "already has a pending choice"):
            run_foul_play.send_recovery_action(
                {"active": [{"moves": []}], "rqid": 10},
                "move recover",
                "move scald",
                8,
            )
        with self.assertRaisesRegex(RuntimeError, "older rqid"):
            run_foul_play.send_recovery_action(
                {"active": [{"moves": []}], "rqid": 6},
                None,
                "move scald",
                8,
            )
        with self.assertRaisesRegex(RuntimeError, "invalid rqid"):
            run_foul_play.send_recovery_action(
                {"active": [{"moves": []}], "rqid": True},
                None,
                "move scald",
                8,
            )
        with self.assertRaisesRegex(RuntimeError, "unexpectedly has a pending choice"):
            run_foul_play.send_recovery_action(
                {"wait": True, "rqid": 8},
                "move recover",
                "move scald",
                8,
            )
        with self.assertRaisesRegex(RuntimeError, "older rqid"):
            run_foul_play.send_recovery_action(
                {"wait": True, "rqid": 6}, None, "move scald", 8
            )
        with self.assertRaisesRegex(RuntimeError, "not an object"):
            run_foul_play.showdown_request_payload(
                ">battle-test\n|request|[]"
            )
        self.assertIsNone(
            run_foul_play.showdown_choice_error([">battle-test\n|turn|4"])
        )
        self.assertEqual(
            run_foul_play.showdown_choice_error(
                [">battle-test\n|error|[Invalid choice] Too late"]
            ),
            "|error|[Invalid choice] Too late",
        )

    def test_reconnect_replay_accepts_terminal_delta_without_request(self):
        room = "battle-test"
        chunks, replayed = run_foul_play.reconnect_delta_chunks(
            [
                f">{room}\n|init|battle\n|turn|4\n"
                '|request|{"active":[{"moves":[]}],"rqid":8}\n|win|Bot',
            ],
            room,
            ["|turn|4"],
        )

        self.assertEqual(chunks, [f">{room}\n|win|Bot"])
        self.assertEqual(replayed, ["|turn|4", "|win|Bot"])
        self.assertTrue(run_foul_play.terminal_showdown_message(chunks[0]))

    def test_battle_request_identity_is_stable_and_payload_bound(self):
        battle = SimpleNamespace(
            battle_tag="battle-test",
            rqid=7,
            request_json={"rqid": 7, "active": [{"moves": []}]},
        )
        key, fingerprint = run_foul_play.battle_request_identity(battle)
        reordered = SimpleNamespace(
            battle_tag="battle-test",
            rqid=7,
            request_json={"active": [{"moves": []}], "rqid": 7},
        )

        self.assertEqual(key, ("battle-test", 7))
        self.assertEqual(
            run_foul_play.battle_request_identity(reordered), (key, fingerprint)
        )
        battle.request_json["wait"] = True
        self.assertNotEqual(
            run_foul_play.battle_request_identity(battle)[1], fingerprint
        )

    def test_search_first_selector_table(self):
        cases = []

        agreement = self.choice_battle(hp=50)
        cases.append(
            (
                "agreement",
                agreement,
                [(self.mcts_result([("recover", 90.0, 100)]), 1.0, 0)],
                [("recover", 1.0)],
                None,
                "recover",
                "search_first_policy_agreement",
                "search_selection",
            )
        )

        disagreement = self.choice_battle(hp=50)
        cases.append(
            (
                "rejected_certificate_is_shadow_only",
                disagreement,
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
                {"earthquake": {"qualified": False, "coverage": 1.0}},
                "earthquake",
                "search_first_search_selection",
                "search_selection",
            )
        )

        illegal = self.choice_battle(hp=50)
        illegal.user.active.moves = [SimpleNamespace(name="recover", disabled=False)]
        cases.append(
            (
                "illegal_search_top",
                illegal,
                [
                    (
                        self.mcts_result(
                            [("earthquake", 180.0, 200), ("recover", 20.0, 100)]
                        ),
                        1.0,
                        0,
                    )
                ],
                [("recover", 1.0)],
                None,
                "recover",
                "search_first_policy_agreement",
                "search_selection",
            )
        )

        forced = self.choice_battle(hp=50)
        forced.force_switch = True
        forced.user.active.moves = [SimpleNamespace(name="earthquake", disabled=False)]
        forced.user.reserve = [SimpleNamespace(name="blissey", hp=100)]
        cases.append(
            (
                "forced_switch",
                forced,
                [
                    (
                        self.mcts_result(
                            [("earthquake", 180.0, 200), ("switch blissey", 20.0, 100)]
                        ),
                        1.0,
                        0,
                    )
                ],
                [("earthquake", 0.9), ("switch blissey", 0.1)],
                None,
                "switch blissey",
                "search_first_policy_agreement",
                "search_selection",
            )
        )

        noop = self.choice_battle(hp=25)
        noop.user.active.moves = [
            SimpleNamespace(name="substitute", disabled=False),
            SimpleNamespace(name="tackle", disabled=False),
        ]
        cases.append(
            (
                "guaranteed_noop",
                noop,
                [
                    (
                        self.mcts_result(
                            [("substitute", 180.0, 200), ("tackle", 20.0, 100)]
                        ),
                        1.0,
                        0,
                    )
                ],
                [("substitute", 1.0)],
                None,
                "tackle",
                "guaranteed_noop_substitute_insufficient_hp",
                "deterministic_correction",
            )
        )

        missing = self.choice_battle(hp=50)
        missing.user.active.moves = [SimpleNamespace(name="recover", disabled=False)]
        cases.append(
            (
                "missing_search_result",
                missing,
                [],
                [("recover", 1.0)],
                None,
                "recover",
                "search_infrastructure_policy_fallback",
                "infrastructure_fallback",
            )
        )

        malformed = self.choice_battle(hp=50)
        malformed.user.active.moves = [SimpleNamespace(name="recover", disabled=False)]
        cases.append(
            (
                "malformed_search_result",
                malformed,
                [object()],
                [("recover", 1.0)],
                None,
                "recover",
                "search_infrastructure_policy_fallback",
                "infrastructure_fallback",
            )
        )

        tera = self.choice_battle(hp=50, can_terastallize=True)
        tera.user.active.moves = [SimpleNamespace(name="tackle", disabled=False)]
        cases.append(
            (
                "tera_mapping",
                tera,
                [
                    (
                        self.mcts_result(
                            [("tackle-tera", 180.0, 200), ("tackle", 20.0, 100)]
                        ),
                        1.0,
                        0,
                    )
                ],
                [("tackle", 1.0)],
                None,
                "tackle-tera",
                "search_first_search_selection",
                "search_selection",
            )
        )

        for name, battle, results, priors, evidence, expected, reason, kind in cases:
            with self.subTest(name=name):
                choice, telemetry = run_foul_play.select_search_first_choice(
                    battle,
                    results,
                    priors,
                    histories={},
                    independent_evidence=evidence,
                    record_history=False,
                )
                self.assertEqual(choice, expected)
                self.assertEqual(telemetry["reason"], reason)
                self.assertEqual(telemetry["selection_class"], kind)
                self.assertEqual(telemetry["controller_mode"], "search_first")
                self.assertFalse(telemetry["verifier_shadow"]["selection_eligible"])
                request_actions = telemetry["request_actions"]
                if request_actions:
                    self.assertIn(choice, request_actions)

    def test_search_first_terminal_correction(self):
        battle = self.choice_battle(hp=100)
        battle.user.active.moves = [
            SimpleNamespace(name="uturn", disabled=False),
            SimpleNamespace(name="woodhammer", disabled=False),
        ]
        results = [
            (
                self.mcts_result(
                    [("uturn", 180.0, 200), ("woodhammer", 20.0, 100)]
                ),
                1.0,
                0,
            )
        ]
        move_data = {
            "uturn": {"basePower": 70, "category": "Physical"},
            "woodhammer": {"basePower": 120, "category": "Physical"},
        }

        with mock.patch.object(
            run_foul_play, "_showdown_move_data", side_effect=move_data.get
        ):
            choice, telemetry = run_foul_play.select_search_first_choice(
                battle,
                results,
                [("uturn", 1.0)],
                histories={},
                record_history=False,
            )

        self.assertEqual(choice, "woodhammer")
        self.assertEqual(telemetry["reason"], "terminal_pivot_without_reserve")
        self.assertEqual(telemetry["selection_class"], "deterministic_correction")

    def test_search_first_revealed_prediction_risk_is_shadow_only(self):
        battle = self.choice_battle(hp=50)
        battle.opponent.active.ability = "waterabsorb"
        battle.user.active.moves = [
            SimpleNamespace(name="surf", disabled=False),
            SimpleNamespace(name="tackle", disabled=False),
        ]
        move_data = {
            "category": "special",
            "flags": {"protect": 1},
            "target": "normal",
            "type": "water",
        }

        with mock.patch.object(
            run_foul_play, "_showdown_move_data", return_value=move_data
        ):
            choice, telemetry = run_foul_play.select_search_first_choice(
                battle,
                [
                    (
                        self.mcts_result(
                            [("surf", 180.0, 200), ("tackle", 20.0, 100)]
                        ),
                        1.0,
                        0,
                    )
                ],
                [("tackle", 1.0)],
                histories={},
                independent_evidence={"surf": {"qualified": False}},
                record_history=False,
            )

        self.assertEqual(choice, "surf")
        self.assertEqual(telemetry["shadow_risks"][0]["reason"], "revealed_waterabsorb_immunity")
        self.assertFalse(telemetry["shadow_risks"][0]["selection_eligible"])

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

    def test_scrappy_bypasses_ghost_type_immunity(self):
        battle = self.choice_battle(hp=50, ability="scrappy")
        battle.opponent.active.types = ["ghost", "fire"]
        move_data = {
            "category": "physical",
            "flags": {"protect": 1},
            "target": "normal",
            "type": "fighting",
        }

        with (
            mock.patch.object(
                run_foul_play, "_showdown_move_data", return_value=move_data
            ),
            mock.patch.object(
                run_foul_play, "_type_effectiveness_modifier", return_value=0
            ),
        ):
            self.assertIsNone(
                run_foul_play._prediction_sensitive_reason(battle, "triplearrows")
            )
            battle.user.active.ability = "overgrow"
            self.assertEqual(
                run_foul_play._prediction_sensitive_reason(battle, "triplearrows"),
                "known_type_immunity",
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
        del active.unique_id
        self.assertTrue(prior_server.opponent_action_support_complete(battle))
        active.unique_id = "active"
        distinct_active = SimpleNamespace(
            name="carbink",
            moves={f"move-{index}": object() for index in range(4)},
        )
        team["active"] = SimpleNamespace(name="carbink", active=True)
        battle.opponent_active_pokemon = distinct_active
        self.assertTrue(prior_server.opponent_action_support_complete(battle))
        team["active"] = active
        battle.opponent_active_pokemon = active
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

    def test_opponent_prior_sanitizer_does_not_restore_tera_after_switch(self):
        battle = SimpleNamespace(
            opponent=SimpleNamespace(
                active=SimpleNamespace(
                    name="carbink",
                    moves=[SimpleNamespace(name="bodypress")],
                    terastallized=False,
                ),
                reserve=[
                    SimpleNamespace(name="clefable", hp=1, terastallized=True)
                ],
            )
        )
        sanitized = dict(
            run_foul_play.sanitize_opponent_priors(
                battle,
                {"bodypress": 0.7, "bodypress-tera": 0.2, "switch clefable": 0.1},
            )
            or []
        )
        self.assertEqual(set(sanitized), {"bodypress", "switch clefable"})
        self.assertAlmostEqual(sanitized["bodypress"], 0.875)
        self.assertAlmostEqual(sanitized["switch clefable"], 0.125)

    def test_showdown_login_status_requires_exact_named_confirmation(self):
        self.assertEqual(
            run_foul_play.showdown_login_status(
                "|updateuser| Test User|1|0", "testuser"
            ),
            "confirmed",
        )
        self.assertEqual(
            run_foul_play.showdown_login_status(
                "|nametaken|testuser|Your authentication token was invalid.",
                "Test User",
            ),
            "rejected",
        )
        self.assertIsNone(
            run_foul_play.showdown_login_status("|updateuser| Guest 1|0|0", "testuser")
        )

    def test_r1_remains_default(self):
        args = launch.parse_args(["--username", "bot"])
        self.assertEqual(args.profile, "r1")
        self.assertEqual(args.games, 200)
        self.assertEqual(args.search_parallelism, launch.DEFAULT_SEARCH_PARALLELISM)
        self.assertEqual(args.search_threads, launch.DEFAULT_SEARCH_THREADS)
        self.assertEqual(args.controller_mode, "search-first")
        self.assertFalse(args.verifier_shadow)
        self.assertEqual(
            args.remote_mcts_timeout_seconds,
            launch.DEFAULT_REMOTE_MCTS_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            args.websocket_receive_timeout_seconds,
            launch.DEFAULT_WEBSOCKET_RECEIVE_TIMEOUT_SECONDS,
        )
        self.assertEqual(launch.SEARCH_TIME_MS, 500)
        self.assertEqual(launch.CPU_C_PUCT, 2.0)

    def test_certified_controller_is_explicit_launcher_rollback(self):
        args = launch.parse_args(
            [
                "--username",
                "bot",
                "--controller-mode",
                "certified",
                "--verifier-shadow",
            ]
        )

        self.assertEqual(args.controller_mode, "certified")
        self.assertTrue(args.verifier_shadow)

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

    def test_launcher_preflights_all_remote_operations_before_matchmaking(self):
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
            stdout=json.dumps(
                {
                    "ok": True,
                    "operations": ["search", "paired_holdout", "shared_root"],
                }
            ),
            stderr="",
        )

        with mock.patch.object(launch.subprocess, "run", return_value=completed) as run:
            result = launch.preflight_remote_mcts(args)

        self.assertTrue(result["ok"])
        command = run.call_args.args[0]
        self.assertIn("srcs.metagross.remote_mcts_preflight", command)
        self.assertIn("paired_holdout", result["operations"])
        self.assertIn("shared_root", result["operations"])

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

    def test_prior_health_requires_exact_spawned_instance_identity(self):
        expected = {
            "schema": 1,
            "nonce": "a" * 64,
            "pid": 123,
            "checkpoint_sha256": "b" * 64,
        }

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(self.payload).encode()

        healthy = {"ok": True, "identity": dict(expected)}
        with mock.patch.object(
            launch.urllib.request, "urlopen", return_value=Response(healthy)
        ):
            self.assertTrue(launch.prior_is_healthy("http://127.0.0.1:8977", expected))

        for field, value in (
            ("nonce", "c" * 64),
            ("pid", 456),
            ("checkpoint_sha256", "d" * 64),
        ):
            stale = {"ok": True, "identity": {**expected, field: value}}
            with self.subTest(field=field), mock.patch.object(
                launch.urllib.request, "urlopen", return_value=Response(stale)
            ):
                self.assertFalse(
                    launch.prior_is_healthy("http://127.0.0.1:8977", expected)
                )

        with mock.patch.object(
            launch.urllib.request,
            "urlopen",
            return_value=Response({"ok": True, "sessions": 0}),
        ):
            self.assertFalse(launch.prior_is_healthy("http://127.0.0.1:8977", expected))

    def test_prior_server_health_payload_reports_instance_contract(self):
        self.assertEqual(
            prior_server.prior_health_payload("a" * 64, "b" * 64, 3, pid=123),
            {
                "ok": True,
                "sessions": 3,
                "identity": {
                    "schema": 1,
                    "nonce": "a" * 64,
                    "pid": 123,
                    "checkpoint_sha256": "b" * 64,
                },
            },
        )

    def test_prior_request_cache_is_monotone_and_idempotent(self):
        self.assertEqual(prior_server.request_cache_status(4, None), "new")
        self.assertEqual(prior_server.request_cache_status(6, 4), "new")
        self.assertEqual(prior_server.request_cache_status(6, 6), "cached")
        with self.assertRaisesRegex(ValueError, "stale rqid"):
            prior_server.request_cache_status(4, 6)

        session = prior_server.BattleSession.__new__(prior_server.BattleSession)
        session.pending_request = True
        session.last_request_json = {"rqid": 6}
        session.last_request_sha256 = "a" * 64
        session.last_request_legality = {"actions": ["tackle"]}
        session.cached_rqid = 6
        session.cached_request_sha256 = "a" * 64
        session.cached_response = {"rqid": 6, "decision_idx": 2}
        session.decision_idx = 3
        self.assertEqual(
            session.compute_priors(
                expected_rqid=6, expected_request_sha256="a" * 64
            ),
            {"rqid": 6, "decision_idx": 2},
        )
        self.assertFalse(session.pending_request)
        self.assertEqual(session.decision_idx, 3)

    def test_private_request_support_is_authoritative(self):
        request = {
            "rqid": 9,
            "forceSwitch": [False],
            "active": [{
                "canTerastallize": "Fire",
                "trapped": False,
                "moves": [
                    {"id": "encore", "pp": 5, "disabled": False},
                    {"id": "protect", "pp": 10, "disabled": True},
                    {"id": "recover", "pp": 0, "disabled": False},
                ],
            }],
            "side": {"pokemon": [
                {"active": True, "condition": "100/100", "details": "Testmon, L80"},
                {"active": False, "condition": "50/100", "details": "Blissey, L80"},
                {"active": False, "condition": "0 fnt", "details": "Mew, L80"},
            ]},
        }
        expected = {
            "encore",
            "encore-tera",
            "switch blissey",
        }
        support = prior_server.request_action_support(request)
        self.assertEqual(set(support["actions"]), expected)
        battle = SimpleNamespace(rqid=9, request_json=request)
        self.assertEqual(run_foul_play.request_player_actions(battle), expected)
        self.assertEqual(
            prior_server.canonical_request_sha256(request),
            run_foul_play.battle_request_identity(
                SimpleNamespace(
                    battle_tag="battle-test", rqid=9, request_json=request
                )
            )[1],
        )

    def test_private_request_force_switch_ignores_trapping(self):
        request = {
            "rqid": 10,
            "forceSwitch": [True],
            "active": [{"trapped": True, "moves": []}],
            "side": {"pokemon": [
                {"active": True, "condition": "0 fnt", "details": "Lead, L80"},
                {"active": False, "condition": "1/100", "details": "Backup, L80"},
            ]},
        }
        self.assertEqual(
            set(prior_server.request_action_support(request)["actions"]),
            {"switch backup"},
        )
        self.assertEqual(
            run_foul_play.request_player_actions(
                SimpleNamespace(rqid=10, request_json=request)
            ),
            {"switch backup"},
        )

    def test_private_request_hash_mismatch_fails_closed(self):
        session = prior_server.BattleSession.__new__(prior_server.BattleSession)
        session.pending_request = True
        session.last_request_json = {"rqid": 6}
        session.last_request_sha256 = "a" * 64
        session.last_request_legality = {"actions": ["tackle"]}
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            session.compute_priors(
                expected_rqid=6, expected_request_sha256="b" * 64
            )

    def test_wait_for_health_rejects_stale_listener_then_reports_child_exit(self):
        process = SimpleNamespace(
            poll=mock.Mock(side_effect=[None, 48]), returncode=48
        )
        with (
            mock.patch.object(launch, "prior_is_healthy", return_value=False) as health,
            mock.patch.object(launch.time, "sleep"),
            self.assertRaisesRegex(RuntimeError, "prior server exited with code 48"),
        ):
            launch.wait_for_health(
                "http://127.0.0.1:8977",
                process,
                {
                    "schema": 1,
                    "nonce": "a" * 64,
                    "pid": 123,
                    "checkpoint_sha256": "b" * 64,
                },
                timeout=10,
            )

        self.assertEqual(health.call_count, 1)

    def test_launcher_never_starts_client_when_prior_identity_preflight_fails(self):
        prior = SimpleNamespace(
            pid=777,
            returncode=48,
            poll=mock.Mock(return_value=48),
        )
        account_lock = mock.Mock()
        nonce = "a" * 64
        checkpoint_sha = "b" * 64
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.dict(
                os.environ, {"METAGROSS_SHOWDOWN_PASSWORD": "secret"}, clear=True
            ),
            mock.patch.object(
                launch,
                "verify_checkpoint",
                return_value=(Path(temporary) / "policy.pt", checkpoint_sha),
            ),
            mock.patch.object(launch, "inspect_foul_play_engine", return_value={}),
            mock.patch.object(launch, "preflight_remote_mcts", return_value=None),
            mock.patch.object(launch, "build_world_manifest", return_value={}),
            mock.patch.object(launch, "acquire_account_lock", return_value=account_lock),
            mock.patch.object(launch.secrets, "token_hex", return_value=nonce),
            mock.patch.object(launch.signal, "signal", return_value=None),
            mock.patch.object(launch.subprocess, "Popen", return_value=prior) as popen,
            mock.patch.object(
                launch,
                "wait_for_health",
                side_effect=RuntimeError("prior server exited with code 48"),
            ) as wait,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "prior server exited with code 48"
            ):
                launch.main(
                    [
                        "--username",
                        "bot",
                        "--games",
                        "1",
                        "--output-root",
                        temporary,
                    ]
                )
            run_dir = next(Path(temporary).iterdir())
            manifest = json.loads((run_dir / "manifest.json").read_text())

        self.assertEqual(popen.call_count, 1)
        prior_environment = popen.call_args.kwargs["env"]
        self.assertEqual(prior_environment["METAGROSS_PRIOR_INSTANCE_NONCE"], nonce)
        expected_identity = {
            "schema": 1,
            "nonce": nonce,
            "pid": prior.pid,
            "checkpoint_sha256": checkpoint_sha,
        }
        self.assertEqual(wait.call_args.args[2], expected_identity)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(
            manifest["error"], "RuntimeError: prior server exited with code 48"
        )
        self.assertEqual(
            manifest["prior_instance"],
            {
                "health_identity_schema": 1,
                "pid": prior.pid,
                "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(),
                "checkpoint_sha256": checkpoint_sha,
            },
        )
        account_lock.close.assert_called_once_with()

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
            "METAGROSS_FAULT_DISCONNECT_AFTER_BATTLE_COMMANDS": "1",
            "METAGROSS_FAULT_STALL_AFTER_BATTLE_COMMANDS": "1",
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
            "side_two": [
                {"move_choice": "protect", "total_score": 0.0, "visits": 1}
            ],
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

    def test_remote_independent_ensemble_sends_one_bounded_correlated_batch(self):
        captured = []
        payload = {
            "side_one": [
                {"move_choice": "tackle", "total_score": 1.0, "visits": 1}
            ],
            "side_two": [
                {"move_choice": "protect", "total_score": 0.0, "visits": 1}
            ],
            "total_visits": 1,
        }

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
                    "result": payload,
                    "timing": {"search_ms": 1.0, "batch_size": 1},
                }
                for request in requests
            ]

        previous = dict(run_foul_play._PRIOR_STATE)
        run_foul_play._PRIOR_STATE.update(
            {
                "context": {"tag": "battle-ensemble", "decision_idx": 2},
                "priors": [("tackle", 1.0)],
                "opp_priors": None,
                "cpuct": 2.0,
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
                results = run_foul_play._remote_mcts_ensemble_batch(
                    ["state-a", "state-b"], 25, 1, 3
                )
                telemetry = dict(run_foul_play._PRIOR_STATE["remote_search"])
                three_repeat_requests = list(captured)
                captured.clear()
                adaptive_results = run_foul_play._remote_mcts_ensemble_batch(
                    [f"state-{index}" for index in range(32)], 25, 1, 2
                )
                adaptive_telemetry = dict(
                    run_foul_play._PRIOR_STATE["remote_search"]
                )
        finally:
            run_foul_play._PRIOR_STATE.clear()
            run_foul_play._PRIOR_STATE.update(previous)

        self.assertEqual(len(results), 6)
        self.assertEqual(len(three_repeat_requests), 6)
        self.assertEqual(
            [request["index"] for request in three_repeat_requests], list(range(6))
        )
        self.assertEqual(
            [request["state"] for request in three_repeat_requests],
            ["state-a", "state-b"] * 3,
        )
        self.assertEqual(
            len({request["request_id"] for request in three_repeat_requests}), 6
        )
        self.assertEqual(telemetry["operation"], "independent_ensemble")
        self.assertEqual(telemetry["repeat_count"], 3)
        self.assertEqual(telemetry["searches"], 6)
        self.assertEqual(len(adaptive_results), 64)
        self.assertEqual(len(captured), 64)
        self.assertEqual(adaptive_telemetry["repeat_count"], 2)
        self.assertEqual(adaptive_telemetry["searches"], 64)

    def test_remote_shared_root_sends_one_complete_weighted_cohort(self):
        captured = []
        result = self.shared_root_result(particles=2, input_weights=[0.25, 0.75])

        def remote(requests):
            captured.extend(requests)
            request = requests[0]
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
                    "result": result,
                    "timing": {"search_ms": 1.0, "batch_size": 1},
                }
            ]

        previous = dict(run_foul_play._PRIOR_STATE)
        run_foul_play._PRIOR_STATE.update(
            {
                "context": {"tag": "battle-shared", "decision_idx": 5},
                "priors": [("earthquake", 0.75), ("recover", 0.25)],
                "opp_priors": [("protect", 1.0)],
                "remote_search": None,
            }
        )
        environment = {
            "METAGROSS_RUN_SEED": "04" * 32,
            "METAGROSS_RNG_SCHEME": run_foul_play.RNG_SCHEME,
            "METAGROSS_REMOTE_ENGINE_SHA256": "a" * 64,
            "METAGROSS_REQUIRE_REMOTE_MCTS": "1",
            "METAGROSS_SHARED_ROOT_PRIOR_STRENGTH": "1.0",
        }
        try:
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(run_foul_play, "_remote_mcts_call", side_effect=remote),
            ):
                actual = run_foul_play._remote_shared_root_batch(
                    ["state-a", "state-b"], [0.25, 0.75], 100, 8, 7
                )
            telemetry = run_foul_play._PRIOR_STATE["remote_search"]
        finally:
            run_foul_play._PRIOR_STATE.clear()
            run_foul_play._PRIOR_STATE.update(previous)

        self.assertEqual(actual, result)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["operation"], "shared_root")
        self.assertEqual(captured[0]["states"], ["state-a", "state-b"])
        self.assertEqual(captured[0]["particle_weights"], [0.25, 0.75])
        self.assertEqual(
            captured[0]["s2_priors"],
            [[list(("protect", 1.0))], [list(("protect", 1.0))]],
        )
        self.assertEqual(telemetry["operation"], "shared_root")
        self.assertEqual(telemetry["solver_diagnostics"], result["diagnostics"])

    def test_local_shared_root_uses_the_same_typed_engine_contract(self):
        payload = self.shared_root_result()
        native = SimpleNamespace(
            policy=[SimpleNamespace(**row) for row in payload["policy"]],
            opponent_policies=[
                [(row["action"], row["probability"]) for row in policy]
                for policy in payload["opponent_policies"]
            ],
            diagnostics=SimpleNamespace(**payload["diagnostics"]),
            replay_capture=SimpleNamespace(
                **{
                    **payload["replay_capture"],
                    "configuration": SimpleNamespace(
                        **payload["replay_capture"]["configuration"]
                    ),
                    "canonical_particles": [
                        SimpleNamespace(
                            **{
                                **particle,
                                "source_particles": [
                                    SimpleNamespace(**source)
                                    for source in particle["source_particles"]
                                ],
                                "continuations": [
                                    [SimpleNamespace(**cell) for cell in row]
                                    for row in particle["continuations"]
                                ],
                            }
                        )
                        for particle in payload["replay_capture"][
                            "canonical_particles"
                        ]
                    ],
                }
            ),
        )
        engine = SimpleNamespace(
            State=SimpleNamespace(from_string=lambda value: f"parsed:{value}"),
            shared_information_set_root_search=mock.Mock(return_value=native),
        )
        previous = dict(run_foul_play._PRIOR_STATE)
        run_foul_play._PRIOR_STATE.update(
            {
                "priors": [("earthquake", 1.0)],
                "opp_priors": [("protect", 1.0)],
                "remote_search": None,
            }
        )
        try:
            with (
                mock.patch.dict(
                    os.environ,
                    {"METAGROSS_SHARED_ROOT_PRIOR_STRENGTH": "1.0"},
                    clear=True,
                ),
                mock.patch.dict("sys.modules", {"poke_engine": engine}),
            ):
                result = run_foul_play._remote_shared_root_batch(
                    ["state"], [1.0], 100, 8, 7
                )
            telemetry = run_foul_play._PRIOR_STATE["remote_search"]
        finally:
            run_foul_play._PRIOR_STATE.clear()
            run_foul_play._PRIOR_STATE.update(previous)

        self.assertEqual(result, payload)
        engine.shared_information_set_root_search.assert_called_once_with(
            ["parsed:state"],
            [1.0],
            100,
            8,
            7,
            1.0,
            [("earthquake", 1.0)],
            [[("protect", 1.0)]],
        )
        self.assertEqual(telemetry["transport"], "local")

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
