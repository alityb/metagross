from __future__ import annotations

import math
import unittest

from srcs.metagross.known_team_decision_v2_phase2 import (
    ALPHAS,
    aggregate_advantages,
    aggregate_visit_policy,
    alpha_key,
    collapse_draws,
    _candidate_distribution,
    normalized_weights,
    policy_tv,
    weight_diagnostics,
    _simultaneous_mean_bands,
)


class KnownTeamDecisionV2Phase2Test(unittest.TestCase):
    def test_alpha_grid_has_frozen_endpoints_and_steps(self):
        self.assertEqual(len(ALPHAS), 21)
        self.assertEqual(ALPHAS[0], 0.0)
        self.assertEqual(ALPHAS[-1], 1.0)
        self.assertEqual(alpha_key(0.05), "005")

    def test_stratum_distribution_is_the_requested_treatment(self):
        support, distribution = _candidate_distribution(
            "mew", {"a": 0.8, "b": 0.2}, {"a": 0.2, "b": 0.8}, 0.25
        )
        self.assertEqual(support, ["a", "b"])
        self.assertAlmostEqual(distribution["a"], 0.65)

    def test_endpoint_weights_normalize_and_can_zero_support(self):
        draws = [
            {"selected_candidates": {"mew": {
                "proposal_probability": 1.0,
                "current_probability": 1.0,
                "history_probability": 0.0,
            }}},
            {"selected_candidates": {"mew": {
                "proposal_probability": 1.0,
                "current_probability": 0.0,
                "history_probability": 1.0,
            }}},
        ]
        self.assertEqual(normalized_weights(draws, 0.0), [1.0, 0.0])
        self.assertEqual(normalized_weights(draws, 1.0), [0.0, 1.0])
        self.assertAlmostEqual(weight_diagnostics(draws, 0.5)["ess"], 2.0)

    def test_duplicate_states_collapse_without_losing_multiplicity(self):
        draws = [
            {"draw_index": 0, "state_sha256": "a", "state": "state-a"},
            {"draw_index": 1, "state_sha256": "a", "state": "state-a"},
            {"draw_index": 2, "state_sha256": "b", "state": "state-b"},
        ]
        collapsed = collapse_draws(draws)
        self.assertEqual(collapsed[0]["multiplicity"], 2)
        self.assertEqual(collapsed[0]["draw_indices"], [0, 1])

    def test_visit_and_paired_advantage_aggregation(self):
        searches = [
            {"a": {"visits": 8, "q": 0.6}, "b": {"visits": 2, "q": 0.7}},
            {"a": {"visits": 1, "q": 0.3}, "b": {"visits": 9, "q": 0.2}},
        ]
        weights = [0.75, 0.25]
        policy = aggregate_visit_policy(searches, weights)
        advantages = aggregate_advantages(searches, weights, "a")
        self.assertAlmostEqual(sum(policy.values()), 1.0)
        self.assertAlmostEqual(advantages["a"], 0.0)
        self.assertAlmostEqual(advantages["b"], 0.05)

    def test_policy_tv_is_symmetric(self):
        left = {"a": 0.8, "b": 0.2}
        right = {"a": 0.5, "b": 0.5}
        self.assertAlmostEqual(policy_tv(left, right), 0.3)
        self.assertAlmostEqual(policy_tv(right, left), 0.3)

    def test_simultaneous_override_bands_ignore_nonoverrides(self):
        rows = [
            {
                "battle_id": str(index),
                "treatments": {
                    alpha_key(alpha): {
                        "visit": {
                            "changed": index % 2 == 0,
                            "teacher_delta": 0.02 if index % 2 == 0 else 0.0,
                        }
                    }
                    for alpha in ALPHAS[1:]
                },
            }
            for index in range(10)
        ]
        bands = _simultaneous_mean_bands(
            rows, "visit", 100, overrides_only=True
        )
        self.assertTrue(all(interval[0] == interval[1] == 0.02 for interval in bands.values()))

    def test_simultaneous_override_bands_handle_no_overrides(self):
        rows = [
            {
                "battle_id": str(index),
                "treatments": {
                    alpha_key(alpha): {
                        "visit": {"changed": False, "teacher_delta": 0.0}
                    }
                    for alpha in ALPHAS[1:]
                },
            }
            for index in range(3)
        ]
        bands = _simultaneous_mean_bands(
            rows, "visit", 25, overrides_only=True
        )
        self.assertTrue(all(math.isnan(interval[0]) for interval in bands.values()))


if __name__ == "__main__":
    unittest.main()
