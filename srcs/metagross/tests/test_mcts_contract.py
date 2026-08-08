from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    HOLDOUT_RESULT_FIELDS,
    REQUEST_SCHEMA,
    compute_engine_source_sha256,
    engine_identity,
    holdout_result_payload,
    result_payload,
    validate_holdout_result_payload,
    validate_priors,
    validate_request,
    validate_result_payload,
)


def option(move: str, score: object = 0.0, visits: object = 0) -> SimpleNamespace:
    return SimpleNamespace(move_choice=move, total_score=score, visits=visits)


class MctsContractTest(unittest.TestCase):
    def test_v5_contract_and_schema_are_explicit(self):
        self.assertEqual(ENGINE_CONTRACT, "poke-engine-0.0.47-holdout-v5")
        self.assertEqual(REQUEST_SCHEMA, 3)
        self.assertEqual(len(HOLDOUT_RESULT_FIELDS), 20)

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

        engine = SimpleNamespace(
            monte_carlo_tree_search=search,
            paired_root_policy_evaluation=holdout,
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


if __name__ == "__main__":
    unittest.main()
