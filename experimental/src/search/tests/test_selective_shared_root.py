from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from search.selective_shared_root import (  # noqa: E402
    SelectiveSharedRootMetrics,
    compute_confidence_mixture,
    compute_selective_shared_root_metrics,
    decide_selective_action,
    should_trigger_selective_shared_root,
)


class SelectiveSharedRootMetricTests(unittest.TestCase):
    def test_exact_metrics(self):
        metrics = compute_selective_shared_root_metrics(
            [{"a": 0.75, "b": 0.25}, {"a": 0.25, "b": 0.75}],
            [0.5, 0.5],
        )

        self.assertAlmostEqual(metrics.weighted_top_action_disagreement, 0.5)
        self.assertAlmostEqual(
            metrics.weighted_js_divergence,
            0.75 * math.log(1.5) + 0.25 * math.log(0.5),
        )
        self.assertAlmostEqual(metrics.aggregate_top_visit_mass, 0.5)
        self.assertAlmostEqual(metrics.aggregate_top_two_margin, 0.0)
        self.assertEqual(metrics.world_count, 2)
        self.assertAlmostEqual(metrics.effective_world_count, 2.0)

    def test_trigger_thresholds_are_inclusive_and_require_all(self):
        metrics = SelectiveSharedRootMetrics(0.45, 0.25, 0.65, 0.1, 4, 4.0, 2)
        self.assertTrue(should_trigger_selective_shared_root(metrics))
        self.assertFalse(
            should_trigger_selective_shared_root(
                SelectiveSharedRootMetrics(0.449, 0.25, 0.65, 0.1, 4, 4.0, 2)
            )
        )
        self.assertFalse(
            should_trigger_selective_shared_root(
                SelectiveSharedRootMetrics(0.45, 0.249, 0.65, 0.1, 4, 4.0, 2)
            )
        )
        self.assertFalse(
            should_trigger_selective_shared_root(
                SelectiveSharedRootMetrics(0.45, 0.25, 0.651, 0.1, 4, 4.0, 2)
            )
        )

    def test_one_action_never_triggers(self):
        metrics = compute_selective_shared_root_metrics([{"only": 1.0}] * 2, [1.0, 1.0])
        self.assertFalse(
            should_trigger_selective_shared_root(
                metrics, disagreement_threshold=0.0, js_threshold=0.0, top_mass_threshold=1.0
            )
        )

        forced_across_worlds = compute_selective_shared_root_metrics(
            [{"forced-a": 1.0}, {"forced-b": 1.0}], [1.0, 1.0]
        )
        self.assertFalse(
            should_trigger_selective_shared_root(
                forced_across_worlds,
                disagreement_threshold=0.0,
                js_threshold=0.0,
                top_mass_threshold=1.0,
            )
        )


class SelectiveActionDecisionTests(unittest.TestCase):
    def test_audit_never_overrides(self):
        decision = decide_selective_action(
            mode="audit",
            baseline_action="a",
            triggered=True,
            shared_action="b",
            paired_available=True,
            paired_lcb=1.0,
        )
        self.assertEqual(decision.action, "a")
        self.assertFalse(decision.overridden)
        self.assertEqual(decision.reason, "audit")

    def test_override_requires_lcb_above_margin(self):
        rejected = decide_selective_action(
            mode="override",
            baseline_action="a",
            triggered=True,
            shared_action="b",
            paired_available=True,
            paired_lcb=0.0,
            lcb_margin=-1.0,
        )
        accepted = decide_selective_action(
            mode="override",
            baseline_action="a",
            triggered=True,
            shared_action="b",
            paired_available=True,
            paired_lcb=0.01,
        )
        self.assertEqual(rejected.action, "a")
        self.assertEqual(accepted.action, "b")
        self.assertTrue(accepted.overridden)

    def test_fallbacks_return_baseline(self):
        unavailable = decide_selective_action(
            mode="override", baseline_action="a", triggered=True, shared_action="b"
        )
        failed = decide_selective_action(
            mode="override", baseline_action="a", triggered=True
        )
        not_triggered = decide_selective_action(
            mode="override", baseline_action="a", triggered=False, shared_action="b"
        )
        self.assertEqual(unavailable.action, "a")
        self.assertEqual(failed.action, "a")
        self.assertEqual(not_triggered.action, "a")


class ConfidenceMixtureTests(unittest.TestCase):
    def test_lcb_zero_gives_alpha_zero_always_baseline(self):
        mixture = compute_confidence_mixture(
            paired_lcb=0.0,
            lcb_scale=0.05,
            baseline_action="a",
            shared_policy=[("a", 0.3), ("b", 0.7)],
        )
        self.assertAlmostEqual(mixture.alpha, 0.0)
        self.assertAlmostEqual(mixture.blended_distribution["a"], 1.0)
        self.assertNotIn("b", mixture.blended_distribution)

    def test_lcb_below_scale_gives_partial_alpha(self):
        mixture = compute_confidence_mixture(
            paired_lcb=0.025,
            lcb_scale=0.05,
            baseline_action="a",
            shared_policy=[("a", 0.3), ("b", 0.7)],
        )
        self.assertAlmostEqual(mixture.alpha, 0.5)
        self.assertAlmostEqual(mixture.blended_distribution["a"], 0.65)
        self.assertAlmostEqual(mixture.blended_distribution["b"], 0.35)
        self.assertGreater(mixture.blended_distribution["a"], 0.0)
        self.assertGreater(mixture.blended_distribution["b"], 0.0)

    def test_lcb_above_scale_gives_alpha_one_always_shared(self):
        mixture = compute_confidence_mixture(
            paired_lcb=0.1,
            lcb_scale=0.05,
            baseline_action="a",
            shared_policy=[("a", 0.0), ("b", 1.0)],
        )
        self.assertAlmostEqual(mixture.alpha, 1.0)
        self.assertAlmostEqual(mixture.blended_distribution.get("a", 0.0), 0.0)
        self.assertAlmostEqual(mixture.blended_distribution["b"], 1.0)

    def test_negative_lcb_gives_alpha_zero(self):
        mixture = compute_confidence_mixture(
            paired_lcb=-0.5,
            lcb_scale=0.05,
            baseline_action="a",
            shared_policy=[("a", 0.3), ("b", 0.7)],
        )
        self.assertAlmostEqual(mixture.alpha, 0.0)
        self.assertAlmostEqual(mixture.blended_distribution["a"], 1.0)

    def test_baseline_not_in_shared_policy_gets_shared_mass(self):
        mixture = compute_confidence_mixture(
            paired_lcb=0.05,
            lcb_scale=0.05,
            baseline_action="c",
            shared_policy=[("a", 0.6), ("b", 0.4)],
        )
        self.assertAlmostEqual(mixture.alpha, 1.0)
        self.assertAlmostEqual(mixture.blended_distribution.get("c", 0.0), 0.0)
        self.assertAlmostEqual(mixture.blended_distribution["a"], 0.6)
        self.assertAlmostEqual(mixture.blended_distribution["b"], 0.4)


if __name__ == "__main__":
    unittest.main()
