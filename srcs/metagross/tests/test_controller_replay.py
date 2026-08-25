from __future__ import annotations

import unittest

from srcs.metagross import controller_replay


class ControllerReplayTest(unittest.TestCase):
    def test_aggregate_requires_complete_legal_664_decision_corpus(self):
        report = {
            "capture": {"capture_digest": "digest"},
            "counts": {
                "decisions": 664,
                "illegal_certified_outputs": 0,
                "illegal_search_first_outputs": 0,
                "unexplained_changes": 0,
                "certified_recorded_mismatches": 0,
            },
            "gate": {"passed": True},
        }

        aggregate = controller_replay.aggregate([report])

        self.assertTrue(aggregate["gate"]["passed"])
        self.assertTrue(aggregate["gate"]["complete_664_decision_corpus"])

    def test_aggregate_fails_on_incomplete_or_illegal_corpus(self):
        report = {
            "capture": {"capture_digest": "digest"},
            "counts": {
                "decisions": 663,
                "illegal_certified_outputs": 0,
                "illegal_search_first_outputs": 1,
                "unexplained_changes": 0,
                "certified_recorded_mismatches": 0,
            },
            "gate": {"passed": True},
        }

        aggregate = controller_replay.aggregate([report])

        self.assertFalse(aggregate["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
