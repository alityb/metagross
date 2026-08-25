"""D1 attempt-1 void regression: non-finite priors must never reach the engine."""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_foul_play  # noqa: E402


class NonfinitePriorTests(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger("test-nonfinite-priors")

    def test_all_nan_revival_prompt_priors_are_discarded_whole(self):
        # Exact shape of the D1 attempt-1 failure: the Revival Blessing
        # revive prompt produced an all-NaN masked softmax.
        priors = {"bugbuzz": float("nan"), "bugbuzz-tera": float("nan"),
                  "earthpower": float("nan"), "earthpower-tera": float("nan")}
        self.assertEqual(run_foul_play.discard_nonfinite_priors(priors, "root", self.log), {})

    def test_single_bad_entry_discards_the_mapping(self):
        for bad in (float("nan"), float("inf"), -0.1, 1.5):
            priors = {"tackle": 0.6, "growl": bad}
            self.assertEqual(
                run_foul_play.discard_nonfinite_priors(priors, "root", self.log), {})
        self.assertEqual(
            run_foul_play.discard_nonfinite_priors({"": 0.5, "tackle": 0.5}, "opp", self.log), {})

    def test_finite_priors_pass_through_unchanged(self):
        priors = {"tackle": 0.75, "growl": 0.25, "switch pikachu": 0.0}
        self.assertIs(run_foul_play.discard_nonfinite_priors(priors, "root", self.log), priors)
        self.assertEqual(run_foul_play.discard_nonfinite_priors({}, "root", self.log), {})
        self.assertIsNone(run_foul_play.discard_nonfinite_priors(None, "opp", self.log))


if __name__ == "__main__":
    unittest.main()
