from __future__ import annotations

import unittest

from srcs.metagross.known_team_decision_gate import (
    _aggregate,
    _target_weights,
    _argmax,
    _visit_policy,
)


class KnownTeamDecisionGateTest(unittest.TestCase):
    def test_alternative_weights_normalize_candidate_ratios(self):
        worlds = [
            {
                "selected_candidates": {"mew": {
                    "candidate_id": "a",
                    "proposal_probability": 0.5,
                    "current_probability": 0.5,
                    "mixture_probability": 0.5625,
                }},
            },
            {
                "selected_candidates": {"mew": {
                    "candidate_id": "b",
                    "proposal_probability": 0.5,
                    "current_probability": 0.5,
                    "mixture_probability": 0.4375,
                }},
            },
        ]
        weights = _target_weights(worlds, "mixture")
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(weights[0], 0.5625)
        self.assertAlmostEqual(weights[1], 0.4375)

    def test_unmodeled_hidden_candidates_keep_ratio_one(self):
        worlds = [
            {"selected_candidates": {}},
            {"selected_candidates": {}},
        ]
        self.assertEqual(_target_weights(worlds, "mixture"), [0.5, 0.5])

    def test_current_target_can_zero_newly_restored_support(self):
        worlds = [
            {"selected_candidates": {"mew": {
                "candidate_id": "restored",
                "proposal_probability": 0.5,
                "current_probability": 0.0,
                "mixture_probability": 0.25,
            }}},
            {"selected_candidates": {"mew": {
                "candidate_id": "current",
                "proposal_probability": 0.5,
                "current_probability": 1.0,
                "mixture_probability": 0.75,
            }}},
        ]
        self.assertEqual(_target_weights(worlds, "current"), [0.0, 1.0])
        self.assertEqual(_target_weights(worlds, "mixture"), [0.25, 0.75])

    def test_aggregate_changes_only_weights(self):
        policies = [{"a": 0.8, "b": 0.2}, {"a": 0.1, "b": 0.9}]
        current = _aggregate(policies, [0.5, 0.5])
        alternative = _aggregate(policies, [0.75, 0.25])
        self.assertAlmostEqual(sum(current.values()), 1.0)
        self.assertAlmostEqual(sum(alternative.values()), 1.0)
        self.assertGreater(alternative["a"], current["a"])

    def test_argmax_breaks_ties_canonically(self):
        self.assertEqual(_argmax({"z": 0.5, "a": 0.5}), "a")

    def test_visit_policy_filters_request_illegal_actions(self):
        class Row:
            def __init__(self, move_choice, visits):
                self.move_choice = move_choice
                self.visits = visits

        result = type(
            "Result",
            (),
            {
                "total_visits": 100,
                "side_one": [Row("tackle", 30), Row("tackle-tera", 70)],
            },
        )()
        self.assertEqual(_visit_policy(result, {"tackle"}), {"tackle": 1.0})


if __name__ == "__main__":
    unittest.main()
