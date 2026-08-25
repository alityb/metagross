import unittest

from srcs.metagross.known_team_search_failure_attribution import (
    adaptive_budget_decision,
    classify_root,
    systematic_indices,
)


def budget(*, visit_beneficial=False, q_beneficial=False, visit_agreement=True):
    return {
        "visit": {
            "beneficial": visit_beneficial,
            "seed_agreement": visit_agreement,
        },
        "mean_q_advantage": {"beneficial": q_beneficial},
    }


class KnownTeamSearchFailureAttributionTest(unittest.TestCase):
    def test_systematic_resampling_is_deterministic_and_weighted(self):
        self.assertEqual(systematic_indices([0.1, 0.2, 0.7], 5, 0.0), [0, 1, 2, 2, 2])
        self.assertEqual(systematic_indices([0.1, 0.2, 0.7], 5, 0.5), [0, 1, 2, 2, 2])

    def test_budget_rescue_has_priority(self):
        rows = {
            "20000": budget(visit_agreement=False),
            "80000": budget(visit_beneficial=True, q_beneficial=True),
        }
        self.assertEqual(classify_root(rows), "finite_search_budget")

    def test_current_panel_rescue_is_separate_from_high_budget(self):
        rows = {
            "20000": budget(visit_beneficial=True),
            "80000": budget(visit_beneficial=True, q_beneficial=True),
        }
        self.assertEqual(
            classify_root(rows), "particle_panel_or_current_search_resolved"
        )

    def test_selector_rescue_requires_high_budget_q_benefit(self):
        rows = {
            "20000": budget(),
            "80000": budget(q_beneficial=True),
        }
        self.assertEqual(classify_root(rows), "root_selector_or_visit_allocation")

    def test_current_q_rescue_precedes_high_budget_visit_rescue(self):
        rows = {
            "20000": budget(q_beneficial=True),
            "80000": budget(visit_beneficial=True, q_beneficial=True),
        }
        self.assertEqual(classify_root(rows), "root_selector_or_visit_allocation")

    def test_seed_variance_and_unresolved_fallback(self):
        noisy = {
            "20000": budget(visit_agreement=False),
            "80000": budget(),
        }
        stable = {"20000": budget(), "80000": budget()}
        self.assertEqual(classify_root(noisy), "monte_carlo_seed_variance")
        self.assertEqual(
            classify_root(stable), "unresolved_information_value_or_opponent_model"
        )

    def test_adaptive_budget_requires_agreement_and_margin(self):
        checkpoints = [
            (5000, {"a": 0.55, "b": 0.45}),
            (20000, {"a": 0.65, "b": 0.35}),
            (80000, {"b": 0.8, "a": 0.2}),
        ]
        self.assertEqual(adaptive_budget_decision(checkpoints, 0.2), 20000)
        self.assertEqual(adaptive_budget_decision(checkpoints, 0.4), 80000)

    def test_adaptive_budget_rejects_unordered_checkpoints(self):
        with self.assertRaises(ValueError):
            adaptive_budget_decision([(20, {"a": 1.0}), (10, {"a": 1.0})], 0.1)


if __name__ == "__main__":
    unittest.main()
