from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from srcs.metagross import remote_mcts_preflight
from srcs.metagross.mcts_contract import ENGINE_CONTRACT, ENGINE_SOURCE_SHA256


class RemoteMctsPreflightTest(unittest.TestCase):
    NATIVE = "a" * 64

    @classmethod
    def engine(cls):
        return {
            "contract": ENGINE_CONTRACT,
            "source_sha256": ENGINE_SOURCE_SHA256,
            "distribution_version": "0.0.47",
            "native_sha256": cls.NATIVE,
            "mcts_parameters": remote_mcts_preflight.MCTS_PARAMETERS,
            "holdout_parameters": remote_mcts_preflight.HOLDOUT_PARAMETERS,
            "shared_root_parameters": remote_mcts_preflight.SHARED_ROOT_PARAMETERS,
            "resources": {
                "physical_cores": 16.0,
                "vcpus_equivalent": 32,
                "memory_mib": 16384,
                "worker_processes": 16,
                "max_containers": 4,
                "max_world_concurrency": 64,
            },
        }

    @staticmethod
    def holdout_result():
        return {
            "pairs": 2,
            "baseline_sum": 0.5,
            "candidate_sum": 1.0,
            "delta_sum": 0.5,
            "delta_squared_sum": 0.25,
            "catastrophic_count": 0,
            "candidate_catastrophic_count": 0,
            "baseline_catastrophic_count": 1,
            "candidate_catastrophic_severity_sum": 0.0,
            "baseline_catastrophic_severity_sum": 0.5,
            "candidate_better_count": 1,
            "baseline_better_count": 0,
            "equal_count": 1,
            "baseline_terminal_count": 0,
            "candidate_terminal_count": 0,
            "baseline_nonterminal_evaluation_delta_sum": 1.0,
            "candidate_nonterminal_evaluation_delta_sum": -1.0,
            "baseline_nonterminal_count": 2,
            "candidate_nonterminal_count": 2,
            "continuation_iterations_executed": 8,
        }

    @staticmethod
    def shared_root_result():
        digest = "fnv1a64:" + "0" * 16
        return {
            "policy": [
                {
                    "action": "watergun",
                    "probability": 1.0,
                    "counterfactual_value": 0.5,
                }
            ],
            "opponent_policies": [
                [{"action": "ember", "probability": 1.0}]
            ],
            "diagnostics": {
                "solver_contract": "weighted-shared-rm-plus-v1",
                "iterations": 100,
                "continuation_iterations": 2,
                "seed": 11,
                "prior_strength": 1.0,
                "expected_value": 0.5,
                "player_best_response_value": 0.5,
                "opponent_best_response_value": 0.5,
                "player_best_response_gain": 0.0,
                "opponent_best_response_gain": 0.0,
                "nash_conv": 0.0,
                "exploitability": 0.0,
                "player_regret_bound": 0.0,
                "opponent_regret_bound": 0.0,
                "total_regret_bound": 0.0,
                "payoff_cells": 1,
                "total_forced_continuation_iterations": 2,
                "input_particle_count": 2,
                "positive_particle_count": 2,
                "canonical_particle_count": 1,
                "normalized_weight_sum": 1.0,
                "action_support_digest": digest,
                "particle_digest": digest,
                "payoff_digest": digest,
                "player_prior_digest": digest,
                "opponent_prior_digest": digest,
            },
        }

    def test_build_requests_exercises_search_holdout_and_shared_root(self):
        requests = remote_mcts_preflight.build_requests("state")

        self.assertEqual(
            [row["operation"] for row in requests],
            ["search", "search", "paired_holdout", "shared_root"],
        )
        self.assertEqual([requests[0]["duration_ms"], requests[1]["duration_ms"]], [250, 500])
        self.assertEqual(requests[2]["opponent_priors"], [["ember", 0.75], ["tackle", 0.25]])
        self.assertEqual(requests[3]["particle_weights"], [0.5, 0.5])

    def test_modal_preflight_validates_both_operations_and_full_identity(self):
        engine = self.engine()

        class Function:
            @staticmethod
            def from_name(_app, name):
                if name == "engine_info":
                    return SimpleNamespace(remote=lambda: engine)

                def remote(requests):
                    responses = []
                    for request in requests:
                        result = (
                            {
                                "side_one": [
                                    {
                                        "move_choice": "watergun",
                                        "total_score": 1.0,
                                        "visits": 1,
                                    }
                                ],
                                "side_two": [
                                    {
                                        "move_choice": "ember",
                                        "total_score": 1.0,
                                        "visits": 1,
                                    }
                                ],
                                "total_visits": 1,
                            }
                            if request["operation"] == "search"
                            else self.holdout_result()
                            if request["operation"] == "paired_holdout"
                            else self.shared_root_result()
                        )
                        responses.append(
                            {
                                "schema": request["schema"],
                                "request_id": request["request_id"],
                                "index": request["index"],
                                "engine": engine,
                                "ok": True,
                                "result": result,
                            }
                        )
                    return responses

                return SimpleNamespace(remote=remote)

        with (
            mock.patch.object(remote_mcts_preflight, "synthetic_state", return_value="state"),
            mock.patch.dict("sys.modules", {"modal": SimpleNamespace(Function=Function)}),
        ):
            result = remote_mcts_preflight.run_preflight(
                transport="modal", native_sha256=self.NATIVE
            )

        self.assertTrue(result["ok"])
        self.assertEqual(
            result["operations"], ["search", "paired_holdout", "shared_root"]
        )
        self.assertEqual(result["search_durations_ms"], [250, 500])

    def test_preflight_rejects_partial_engine_identity(self):
        engine = self.engine()
        del engine["holdout_parameters"]
        with self.assertRaisesRegex(RuntimeError, "holdout_parameters"):
            remote_mcts_preflight.validate_engine_identity(
                engine, self.NATIVE, "modal"
            )


if __name__ == "__main__":
    unittest.main()
