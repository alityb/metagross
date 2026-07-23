from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from belief.action_conditioned_randbats import CandidateValidationError, load_active_candidates, update_from_action  # noqa: E402
from scripts.benchmark_action_conditioned_randbats import _metric_summary, benchmark_rows, validate_row  # noqa: E402


def valid_row() -> dict:
    return {
        "active_candidates": [
            {"candidate_id": "set-a", "prior_weight": 3, "moves": ["surf"]},
            {"candidate_id": "set-b", "prior_weight": 1, "moves": ["thunderbolt"]},
        ],
        "legal_actions": ["move surf", "switch pikachu"],
        "observed_action": "move surf",
        "action_likelihoods": {"set-a": 0.2, "set-b": 0.8},
        "label": "set-b",
    }


class ActionConditionedRandbatsTests(unittest.TestCase):
    def test_bayes_update_normalizes_and_changes_ranking(self):
        candidates = load_active_candidates(valid_row()["active_candidates"])
        posterior = update_from_action(candidates, valid_row()["action_likelihoods"])
        self.assertAlmostEqual(posterior.prior["set-a"], 0.75)
        self.assertAlmostEqual(posterior.posterior["set-a"], 3 / 7)
        self.assertAlmostEqual(sum(posterior.posterior.values()), 1.0)
        self.assertEqual(posterior.ranking()[0][0], "set-b")

    def test_allows_impossible_action_but_rejects_negative_likelihood(self):
        candidates = load_active_candidates(valid_row()["active_candidates"])
        posterior = update_from_action(candidates, {"set-a": 0.0, "set-b": 1.0})
        self.assertEqual(posterior.posterior["set-a"], 0.0)
        with self.assertRaisesRegex(CandidateValidationError, "non-negative and finite"):
            update_from_action(candidates, {"set-a": -1.0, "set-b": 1.0})

    def test_illegal_action_is_rejected(self):
        row = valid_row()
        row["observed_action"] = "move hydro-pump"
        with self.assertRaisesRegex(CandidateValidationError, "not legal"):
            validate_row(row)

    def test_label_leakage_and_membership_are_rejected(self):
        leaked = valid_row()
        leaked["active_candidates"][0]["later_label"] = "set-a"
        with self.assertRaisesRegex(CandidateValidationError, "forbidden label"):
            validate_row(leaked)
        unknown = valid_row()
        unknown["label"] = "not-active"
        with self.assertRaisesRegex(CandidateValidationError, "active candidate_id"):
            validate_row(unknown)

    def test_metrics_cover_generator_and_posterior(self):
        posterior, label, metadata = validate_row(valid_row())
        metrics = _metric_summary([(posterior, label, metadata)])
        self.assertEqual(metrics["coverage"], 1.0)
        self.assertEqual(metrics["generator_only"]["top1"], 0.0)
        self.assertEqual(metrics["posterior"]["top1"], 1.0)
        self.assertAlmostEqual(metrics["posterior"]["mean_label_probability"], 4 / 7)
        self.assertAlmostEqual(metrics["posterior"]["brier"], 18 / 49)

    def test_chronological_holdout_keeps_replays_together(self):
        earlier = valid_row()
        earlier.update({"replay_id": "early", "time": 10})
        later = valid_row()
        later.update({"replay_id": "late", "time": 20})
        report = benchmark_rows([validate_row(earlier), validate_row(later)])
        holdout = report["chronological_holdout"]
        self.assertTrue(holdout["available"])
        self.assertEqual(holdout["holdout_replay_ids"], ["late"])
        self.assertEqual(holdout["train"]["rows"], 1)
        self.assertEqual(holdout["holdout"]["rows"], 1)


if __name__ == "__main__":
    unittest.main()
