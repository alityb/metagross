from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from belief.live_action_particles import (  # noqa: E402
    ActionEvidenceCache, bounded_candidates, cumulative_tempered_weights,
    request_if_enabled, validated_weights,
)


class LiveActionParticleTests(unittest.TestCase):
    def test_missing_or_invalid_evidence_keeps_uniform_fallback(self):
        ids = ["a", "b"]
        self.assertIsNone(validated_weights(ids, None))
        self.assertIsNone(validated_weights(ids, {"a": 1.0}))
        self.assertIsNone(validated_weights(ids, {"a": float("nan"), "b": 1.0}))

    def test_zero_likelihood_eliminates_only_that_candidate(self):
        self.assertEqual(validated_weights(["has-move", "missing-move"], {"has-move": 0.25, "missing-move": 0.0}), [0.25, 0.0])

    def test_bounded_selection_preserves_ranked_order(self):
        self.assertEqual(bounded_candidates(list(range(40)), 32), list(range(32)))

    def test_cache_reuses_validated_result(self):
        cache = ActionEvidenceCache()
        self.assertEqual(cache.put("state", ["a"], {"a": 1.0}), [1.0])
        self.assertEqual(cache.get("state"), [1.0])

    def test_disabled_mode_does_not_use_endpoint(self):
        calls = []
        self.assertIsNone(request_if_enabled(False, lambda: calls.append("called")))
        self.assertEqual(calls, [])

    def test_cumulative_history_compounds_and_tempers(self):
        weights = cumulative_tempered_weights([[0.8, 0.2], [0.75, 0.25]], 0.5)
        self.assertIsNotNone(weights)
        self.assertAlmostEqual(weights[0] / weights[1], (12.0) ** 0.5)

    def test_cumulative_zero_eliminates_and_all_zero_falls_back(self):
        self.assertEqual(cumulative_tempered_weights([[0.5, 0.5], [0.0, 1.0]]), [0.0, 1.0])
        self.assertIsNone(cumulative_tempered_weights([[0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
