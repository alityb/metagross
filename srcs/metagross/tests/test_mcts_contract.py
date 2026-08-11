from __future__ import annotations

import unittest
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest import mock

from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    HOLDOUT_RESULT_FIELDS,
    REQUEST_SCHEMA,
    SHARED_ROOT_DIAGNOSTIC_FIELDS,
    compute_engine_source_sha256,
    engine_identity,
    holdout_result_payload,
    result_payload,
    shared_root_result_payload,
    validate_holdout_result_payload,
    validate_priors,
    validate_request,
    validate_result_payload,
    validate_shared_root_result_payload,
)
from srcs.metagross.mcts_contract import _fnv1a64_digest


def option(move: str, score: object = 0.0, visits: object = 0) -> SimpleNamespace:
    return SimpleNamespace(move_choice=move, total_score=score, visits=visits)


class MctsContractTest(unittest.TestCase):
    def test_v7_contract_and_schema_are_explicit(self):
        self.assertEqual(ENGINE_CONTRACT, "poke-engine-0.0.47-shared-root-v7")
        self.assertEqual(REQUEST_SCHEMA, 4)
        self.assertEqual(len(HOLDOUT_RESULT_FIELDS), 20)
        self.assertEqual(len(SHARED_ROOT_DIAGNOSTIC_FIELDS), 26)

    def test_engine_source_hash_matches_checkout(self):
        root = Path(__file__).resolve().parents[2] / "vendor" / "poke-engine"
        self.assertEqual(compute_engine_source_sha256(root), ENGINE_SOURCE_SHA256)

    def test_engine_identity_requires_exact_search_and_holdout_signatures(self):
        def search(
            state,
            duration_ms,
            iterations,
            threads,
            s1_priors,
            s2_priors,
            c_puct,
        ):
            return None

        def holdout(
            state,
            baseline_action,
            candidate_action,
            rollouts,
            continuation_iterations,
            continuation_steps,
            seed,
            opponent_priors=None,
        ):
            return None

        def shared_root(
            states,
            particle_weights,
            iterations,
            continuation_iterations,
            seed,
            prior_strength,
            s1_prior,
            s2_priors,
        ):
            return None

        engine = SimpleNamespace(
            monte_carlo_tree_search=search,
            paired_root_policy_evaluation=holdout,
            shared_information_set_root_search=shared_root,
        )
        native = SimpleNamespace(__file__=__file__)
        with (
            mock.patch.dict("sys.modules", {"poke_engine": engine}),
            mock.patch(
                "srcs.metagross.mcts_contract.importlib.metadata.version",
                return_value="0.0.47",
            ),
            mock.patch(
                "srcs.metagross.mcts_contract.importlib.import_module",
                return_value=native,
            ),
        ):
            identity = engine_identity({"provider": "test"})

        self.assertEqual(
            identity["holdout_parameters"],
            [
                "state",
                "baseline_action",
                "candidate_action",
                "rollouts",
                "continuation_iterations",
                "continuation_steps",
                "seed",
                "opponent_priors",
            ],
        )
        self.assertEqual(
            identity["shared_root_parameters"],
            [
                "states",
                "particle_weights",
                "iterations",
                "continuation_iterations",
                "seed",
                "prior_strength",
                "s1_prior",
                "s2_priors",
            ],
        )

    def test_priors_require_unique_bounded_probabilities_and_positive_mass(self):
        self.assertEqual(
            validate_priors([["tackle", 0.75], ["protect", 0.25]], "priors"),
            [("tackle", 0.75), ("protect", 0.25)],
        )
        invalid = (
            [],
            [["tackle", 0.0]],
            [["tackle", 0.5], ["tackle", 0.5]],
            [["tackle", -0.1]],
            [["tackle", 1.1]],
            [["tackle", float("nan")]],
            [["tackle", float("inf")]],
            [["tackle", True]],
            [["tackle", "1.0"]],
        )
        for priors in invalid:
            with self.subTest(priors=priors), self.assertRaises(ValueError):
                validate_priors(priors, "priors")

    def test_result_payload_accepts_terminal_no_op_visit_gap(self):
        result = SimpleNamespace(
            side_one=[option("tackle", 2.5, 3)],
            side_two=[option("protect", 0.5, 3)],
            total_visits=4,
        )
        self.assertEqual(result_payload(result)["total_visits"], 4)

    def test_result_payload_rejects_duplicate_and_invalid_options(self):
        invalid_sides = (
            [option("tackle"), option("tackle")],
            [option("tackle", float("nan"), 1)],
            [option("tackle", True, 1)],
            [option("tackle", -0.1, 1)],
            [option("tackle", 1.1, 1)],
            [option("tackle", 0.0, -1)],
            [option("tackle", 0.0, True)],
            [option("tackle", 1.0, 0)],
        )
        for side in invalid_sides:
            result = SimpleNamespace(side_one=side, side_two=side, total_visits=1)
            with self.subTest(side=side), self.assertRaises(ValueError):
                result_payload(result)

    def test_result_payload_rejects_inconsistent_visit_totals(self):
        for result in (
            SimpleNamespace(
                side_one=[option("tackle", 1.0, 2)],
                side_two=[option("protect", 1.0, 1)],
                total_visits=2,
            ),
            SimpleNamespace(
                side_one=[option("tackle", 1.0, 2)],
                side_two=[option("protect", 1.0, 2)],
                total_visits=1,
            ),
            SimpleNamespace(side_one=[], side_two=[], total_visits=True),
        ):
            with self.subTest(result=result), self.assertRaises(ValueError):
                result_payload(result)

    def test_validate_result_payload_reuses_producer_invariants(self):
        payload = {
            "side_one": [{"move_choice": "tackle", "total_score": 1.5, "visits": 2}],
            "side_two": [{"move_choice": "protect", "total_score": 0.5, "visits": 2}],
            "total_visits": 3,
        }
        self.assertEqual(validate_result_payload(payload), payload)

        payload["side_one"].append(dict(payload["side_one"][0]))
        with self.assertRaisesRegex(ValueError, "invalid option"):
            validate_result_payload(payload)

    def test_validate_result_payload_bounds_options_and_action_names(self):
        base = {
            "side_one": [],
            "side_two": [],
            "total_visits": 0,
        }
        oversized = {
            **base,
            "side_one": [
                {"move_choice": f"move{index}", "total_score": 0.0, "visits": 0}
                for index in range(65)
            ],
        }
        with self.assertRaisesRegex(ValueError, "side_one is invalid"):
            validate_result_payload(oversized)

        long_name = {
            **base,
            "side_one": [{"move_choice": "x" * 129, "total_score": 0.0, "visits": 0}],
        }
        with self.assertRaisesRegex(ValueError, "invalid option"):
            validate_result_payload(long_name)

    def test_search_and_holdout_operations_are_disjoint_and_bounded(self):
        common = {
            "schema": REQUEST_SCHEMA,
            "request_id": "request",
            "index": 0,
            "state": "state",
        }
        search = validate_request(
            {
                **common,
                "operation": "search",
                "duration_ms": 500,
                "threads": 1,
                "s1_priors": [["tackle", 1.0]],
                "s2_priors": None,
                "c_puct": 2.0,
            }
        )
        self.assertEqual(search["operation"], "search")
        holdout = validate_request(
            {
                **common,
                "operation": "paired_holdout",
                "baseline_action": "tackle",
                "candidate_action": "protect",
                "rollouts": 64,
                "continuation_iterations": 64,
                "continuation_steps": 1,
                "seed": 2**64 - 1,
                "opponent_priors": [["protect", 0.75], ["tackle", 0.25]],
            }
        )
        self.assertEqual(holdout["operation"], "paired_holdout")
        self.assertEqual(
            holdout["opponent_priors"], [("protect", 0.75), ("tackle", 0.25)]
        )
        shared_root = validate_request(
            {
                "schema": REQUEST_SCHEMA,
                "request_id": "shared",
                "index": 0,
                "operation": "shared_root",
                "states": ["state-a", "state-b"],
                "particle_weights": [0.25, 0.75],
                "iterations": 2_000,
                "continuation_iterations": 32,
                "seed": 7,
                "prior_strength": 1.0,
                "s1_prior": [["tackle", 1.0]],
                "s2_priors": [None, [["protect", 1.0]]],
            }
        )
        self.assertEqual(shared_root["operation"], "shared_root")
        self.assertEqual(shared_root["particle_weights"], [0.25, 0.75])
        self.assertEqual(shared_root["s2_priors"], [None, [("protect", 1.0)]])
        for field, value in (
            ("rollouts", True),
            ("rollouts", 0),
            ("continuation_iterations", 0),
            ("continuation_steps", 101),
            ("seed", -1),
        ):
            malformed = {
                **common,
                "operation": "paired_holdout",
                "baseline_action": "tackle",
                "candidate_action": "protect",
                "rollouts": 1,
                "continuation_iterations": 1,
                "continuation_steps": 1,
                "seed": 0,
                "opponent_priors": None,
                field: value,
            }
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_request(malformed)

    def test_shared_root_request_requires_one_complete_bounded_cohort(self):
        request = {
            "schema": REQUEST_SCHEMA,
            "request_id": "shared",
            "index": 0,
            "operation": "shared_root",
            "states": ["state"],
            "particle_weights": [1.0],
            "iterations": 100,
            "continuation_iterations": 8,
            "seed": 0,
            "prior_strength": 0.0,
            "s1_prior": None,
            "s2_priors": None,
        }
        malformed = (
            {**request, "state": "single"},
            {**request, "particle_weights": [0.9]},
            {**request, "particle_weights": [True]},
            {**request, "states": []},
            {**request, "iterations": 0},
            {**request, "continuation_iterations": 1_000_000},
            {**request, "seed": -1},
            {**request, "prior_strength": float("nan")},
            {**request, "s2_priors": []},
        )
        for candidate in malformed:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_request(candidate)

    def test_requests_fail_closed_on_old_schema_missing_priors_and_unknown_fields(self):
        holdout = {
            "schema": REQUEST_SCHEMA,
            "request_id": "request",
            "index": 0,
            "operation": "paired_holdout",
            "state": "state",
            "baseline_action": "tackle",
            "candidate_action": "protect",
            "rollouts": 1,
            "continuation_iterations": 1,
            "continuation_steps": 1,
            "seed": 0,
            "opponent_priors": None,
        }
        search = {
            "schema": REQUEST_SCHEMA,
            "request_id": "request",
            "index": 0,
            "operation": "search",
            "state": "state",
            "duration_ms": 250,
            "threads": 1,
            "s1_priors": None,
            "s2_priors": None,
            "c_puct": 2.0,
        }
        malformed = (
            {**holdout, "schema": REQUEST_SCHEMA - 1},
            {name: value for name, value in holdout.items() if name != "opponent_priors"},
            {**holdout, "duration_ms": 250},
            {**search, "rollouts": 1},
        )
        for request in malformed:
            with self.subTest(request=request), self.assertRaises(ValueError):
                validate_request(request)

    def test_holdout_result_is_aggregate_only_and_internally_consistent(self):
        result = SimpleNamespace(
            pairs=4,
            baseline_sum=1.0,
            candidate_sum=2.0,
            delta_sum=1.0,
            delta_squared_sum=0.5,
            catastrophic_count=0,
            candidate_catastrophic_count=0,
            baseline_catastrophic_count=1,
            candidate_catastrophic_severity_sum=0.0,
            baseline_catastrophic_severity_sum=0.5,
            candidate_better_count=2,
            baseline_better_count=1,
            equal_count=1,
            baseline_terminal_count=0,
            candidate_terminal_count=1,
            baseline_nonterminal_evaluation_delta_sum=2.0,
            candidate_nonterminal_evaluation_delta_sum=-1.0,
            baseline_nonterminal_count=4,
            candidate_nonterminal_count=3,
            continuation_iterations_executed=32,
        )
        payload = holdout_result_payload(result)
        self.assertEqual(validate_holdout_result_payload(payload), payload)
        self.assertEqual(
            validate_holdout_result_payload(
                payload, expected_pairs=4, maximum_executed=32
            ),
            payload,
        )
        self.assertEqual(set(payload), HOLDOUT_RESULT_FIELDS)
        self.assertNotIn("outcomes", payload)
        for mutation in (
            {"delta_sum": 2.0},
            {"delta_squared_sum": 0.1},
            {"candidate_better_count": 4},
            {"candidate_catastrophic_count": 1},
            {"baseline_nonterminal_count": 3},
            {"baseline_catastrophic_severity_sum": 0.4},
            {"candidate_nonterminal_evaluation_delta_sum": float("inf")},
            {"baseline_sum": float("nan")},
            {"extra": 1},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_holdout_result_payload({**payload, **mutation})
        with self.assertRaisesRegex(ValueError, "differs from the request"):
            validate_holdout_result_payload(payload, expected_pairs=5)
        with self.assertRaisesRegex(ValueError, "iteration bound"):
            validate_holdout_result_payload(payload, maximum_executed=31)

    def test_shared_root_result_is_typed_complete_and_internally_consistent(self):
        action_digest = _fnv1a64_digest([b"tackle"])
        particle_digest = _fnv1a64_digest([b"state", struct.pack("<d", 1.0)])
        payoff_digest = _fnv1a64_digest([struct.pack("<d", 0.5)])
        none_digest = _fnv1a64_digest([b"none"])
        diagnostics = SimpleNamespace(
            solver_contract="weighted-shared-rm-plus-v1",
            iterations=100,
            continuation_iterations=8,
            seed=7,
            prior_strength=1.0,
            expected_value=0.5,
            player_best_response_value=0.5,
            opponent_best_response_value=0.5,
            player_best_response_gain=0.0,
            opponent_best_response_gain=0.0,
            nash_conv=0.0,
            exploitability=0.0,
            player_regret_bound=0.01,
            opponent_regret_bound=0.02,
            total_regret_bound=0.03,
            payoff_cells=1,
            total_forced_continuation_iterations=8,
            input_particle_count=1,
            positive_particle_count=1,
            canonical_particle_count=1,
            normalized_weight_sum=1.0,
            action_support_digest=action_digest,
            particle_digest=particle_digest,
            payoff_digest=payoff_digest,
            player_prior_digest=none_digest,
            opponent_prior_digest=none_digest,
        )
        continuation = SimpleNamespace(
            seed=11,
            requested_iterations=8,
            executed_iterations=8,
            visits=8,
            total_score=4.0,
            total_score_bits=struct.unpack("<I", struct.pack("<f", 4.0))[0],
            payoff=0.5,
            payoff_bits=struct.unpack("<Q", struct.pack("<d", 0.5))[0],
        )
        capture = SimpleNamespace(
            schema_version=1,
            solver_contract="weighted-shared-rm-plus-v1",
            configuration=SimpleNamespace(
                iterations=100,
                continuation_iterations=8,
                seed=7,
                prior_strength=1.0,
            ),
            own_action_support=["tackle"],
            normalized_player_prior=None,
            canonical_particles=[
                SimpleNamespace(
                    canonical_index=0,
                    state="state",
                    normalized_weight=1.0,
                    source_particles=[SimpleNamespace(input_index=0, input_weight=1.0)],
                    opponent_action_support=["protect"],
                    normalized_opponent_prior=None,
                    payoff_matrix=[[0.5]],
                    continuations=[[continuation]],
                    opponent_policy=[1.0],
                )
            ],
        )
        result = SimpleNamespace(
            policy=[
                SimpleNamespace(
                    action="tackle", probability=1.0, counterfactual_value=0.5
                )
            ],
            opponent_policies=[[("protect", 1.0)]],
            diagnostics=diagnostics,
            replay_capture=capture,
        )
        payload = shared_root_result_payload(
            result,
            expected_particles=1,
            expected_iterations=100,
            expected_continuation_iterations=8,
            expected_seed=7,
        )
        self.assertEqual(validate_shared_root_result_payload(payload), payload)
        self.assertEqual(set(payload["diagnostics"]), SHARED_ROOT_DIAGNOSTIC_FIELDS)
        self.assertEqual(payload["replay_capture"]["canonical_particles"][0]["state"], "state")
        legacy = {key: value for key, value in payload.items() if key != "replay_capture"}
        self.assertEqual(validate_shared_root_result_payload(legacy), legacy)
        with self.assertRaisesRegex(ValueError, "missing its replay capture"):
            validate_shared_root_result_payload(legacy, require_replay_capture=True)
        with (
            mock.patch(
                "srcs.metagross.mcts_contract.MAX_SHARED_ROOT_REPLAY_BYTES", 1
            ),
            self.assertRaisesRegex(ValueError, "size bound"),
        ):
            validate_shared_root_result_payload(payload)
        for mutation in (
            {"nash_conv": 0.1},
            {"exploitability": 0.1},
            {"total_regret_bound": 0.04},
            {"input_particle_count": 2},
            {"payoff_digest": "bad"},
        ):
            malformed = {
                **payload,
                "diagnostics": {**payload["diagnostics"], **mutation},
            }
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_shared_root_result_payload(
                    malformed,
                    expected_particles=1,
                    expected_iterations=100,
                    expected_continuation_iterations=8,
                    expected_seed=7,
                )
        malformed_policy = {
            **payload,
            "policy": [{**payload["policy"][0], "probability": 0.5}],
        }
        with self.assertRaisesRegex(ValueError, "not normalized"):
            validate_shared_root_result_payload(malformed_policy)
        malformed_capture = {
            **payload,
            "replay_capture": {
                **payload["replay_capture"],
                "canonical_particles": [
                    {
                        **payload["replay_capture"]["canonical_particles"][0],
                        "payoff_matrix": [[0.6]],
                    }
                ],
            },
        }
        with self.assertRaises(ValueError):
            validate_shared_root_result_payload(malformed_capture)


if __name__ == "__main__":
    unittest.main()
