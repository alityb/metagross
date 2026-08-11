from __future__ import annotations

import unittest
import time
from types import SimpleNamespace
from unittest import mock

from srcs.metagross import modal_mcts, run_foul_play
from srcs.metagross.tests.shared_root_fixture import native_shared_root_result


def shared_root_result():
    return native_shared_root_result()


class ModalMctsTest(unittest.TestCase):
    def test_cloud_resource_contract(self):
        self.assertEqual(modal_mcts.APP_NAME, "metagross-mcts-r1-p16")
        self.assertEqual(modal_mcts.FUNCTION_NAME, "search_batch")
        self.assertEqual(modal_mcts.MAX_BATCH_SIZE, 16)
        self.assertEqual(modal_mcts.MAX_CONTAINERS, 4)
        self.assertEqual(modal_mcts.MAX_WORLD_CONCURRENCY, 64)
        self.assertEqual(modal_mcts.CLOUD_PHYSICAL_CORES, 16.0)
        self.assertEqual(modal_mcts.CLOUD_MEMORY_MIB, 16384)
        self.assertEqual(
            modal_mcts.CLOUD_RESOURCES,
            {
                "physical_cores": 16.0,
                "vcpus_equivalent": 32,
                "memory_mib": 16384,
                "worker_processes": 16,
                "max_containers": 4,
                "max_world_concurrency": 64,
            },
        )
        self.assertEqual(modal_mcts.REQUEST_SCHEMA, 4)
        self.assertEqual(len(modal_mcts.ENGINE_SOURCE_SHA256), 64)

    def test_prior_validation_preserves_values(self):
        self.assertEqual(
            modal_mcts._validate_priors(
                [["tackle", 0.75], ["protect", 0.25]], "priors"
            ),
            [("tackle", 0.75), ("protect", 0.25)],
        )

    def test_prior_validation_rejects_invalid_values(self):
        for value in ([["tackle"]], [["", 1.0]], [["tackle", float("nan")]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                modal_mcts._validate_priors(value, "priors")

    def test_worker_dispatches_holdout_with_validated_opponent_priors(self):
        aggregate = SimpleNamespace(
            pairs=2,
            baseline_sum=0.5,
            candidate_sum=1.0,
            delta_sum=0.5,
            delta_squared_sum=0.25,
            catastrophic_count=0,
            candidate_catastrophic_count=0,
            baseline_catastrophic_count=1,
            candidate_catastrophic_severity_sum=0.0,
            baseline_catastrophic_severity_sum=0.5,
            candidate_better_count=1,
            baseline_better_count=0,
            equal_count=1,
            baseline_terminal_count=0,
            candidate_terminal_count=0,
            baseline_nonterminal_evaluation_delta_sum=1.0,
            candidate_nonterminal_evaluation_delta_sum=-1.0,
            baseline_nonterminal_count=2,
            candidate_nonterminal_count=2,
            continuation_iterations_executed=8,
        )
        engine = SimpleNamespace(
            State=SimpleNamespace(from_string=lambda value: value),
            paired_root_policy_evaluation=mock.Mock(return_value=aggregate),
        )
        request = {
            "schema": modal_mcts.REQUEST_SCHEMA,
            "operation": "paired_holdout",
            "request_id": "request",
            "index": 0,
            "state": "state",
            "baseline_action": "tackle",
            "candidate_action": "protect",
            "rollouts": 2,
            "continuation_iterations": 2,
            "continuation_steps": 1,
            "seed": 7,
            "opponent_priors": [["protect", 1.0]],
        }
        with (
            mock.patch.object(modal_mcts, "_engine_identity", return_value={}),
            mock.patch.dict("sys.modules", {"poke_engine": engine}),
        ):
            response = modal_mcts._search_one(request, 1)

        self.assertTrue(response["ok"])
        self.assertEqual(len(response["result"]), 20)
        engine.paired_root_policy_evaluation.assert_called_once_with(
            "state", "tackle", "protect", 2, 2, 1, 7, [("protect", 1.0)]
        )

    def test_worker_dispatches_shared_root_as_one_particle_cohort(self):
        engine = SimpleNamespace(
            State=SimpleNamespace(from_string=lambda value: f"parsed:{value}"),
            shared_information_set_root_search=mock.Mock(
                return_value=shared_root_result()
            ),
        )
        request = {
            "schema": modal_mcts.REQUEST_SCHEMA,
            "operation": "shared_root",
            "request_id": "shared",
            "index": 0,
            "states": ["a", "b"],
            "particle_weights": [0.25, 0.75],
            "iterations": 100,
            "continuation_iterations": 8,
            "seed": 7,
            "prior_strength": 1.0,
            "s1_prior": [["tackle", 1.0]],
            "s2_priors": [None, [["protect", 1.0]]],
        }
        with (
            mock.patch.object(modal_mcts, "_engine_identity", return_value={}),
            mock.patch.dict("sys.modules", {"poke_engine": engine}),
        ):
            response = modal_mcts._search_one(request, 1)

        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["diagnostics"]["input_particle_count"], 2)
        engine.shared_information_set_root_search.assert_called_once_with(
            ["parsed:a", "parsed:b"],
            [0.25, 0.75],
            100,
            8,
            7,
            1.0,
            [("tackle", 1.0)],
            [None, [("protect", 1.0)]],
        )

    def test_modal_call_shards_large_batches_and_preserves_order(self):
        calls = []

        class Function:
            @staticmethod
            def remote(batch):
                calls.append(batch)
                return list(batch)

        payload = list(range(40))
        with mock.patch.object(
            run_foul_play, "_remote_mcts_function", return_value=Function()
        ):
            result = run_foul_play._modal_mcts_call(payload)

        self.assertEqual(result, payload)
        self.assertEqual(sorted(len(batch) for batch in calls), [8, 16, 16])
        self.assertTrue(all(len(batch) <= modal_mcts.MAX_BATCH_SIZE for batch in calls))

    def test_modal_call_keeps_small_batches_in_one_invocation(self):
        remote = mock.Mock(return_value=[{"ok": True}])
        function = mock.Mock(remote=remote)
        payload = [{"index": 0}]
        with mock.patch.object(
            run_foul_play, "_remote_mcts_function", return_value=function
        ):
            result = run_foul_play._modal_mcts_call(payload)

        self.assertEqual(result, [{"ok": True}])
        remote.assert_called_once_with(payload)

    def test_modal_call_has_a_hard_deadline(self):
        def slow_remote(_payload):
            time.sleep(0.1)
            return []

        function = mock.Mock(remote=slow_remote)
        with (
            mock.patch.object(
                run_foul_play, "_remote_mcts_function", return_value=function
            ),
            mock.patch.dict(
                "os.environ", {"METAGROSS_REMOTE_MCTS_TIMEOUT_SECONDS": "0.01"}
            ),
            self.assertRaisesRegex(TimeoutError, "exceeded 0.01s"),
        ):
            run_foul_play._modal_mcts_call([{"index": 0}])


if __name__ == "__main__":
    unittest.main()
