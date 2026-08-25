from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from srcs.metagross import shared_root_replay


class SharedRootReplayTest(unittest.TestCase):
    def test_records_reject_non_root_artifacts(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.jsonl"
            path.write_text(json.dumps({"record_type": "other"}) + "\n")
            with self.assertRaisesRegex(ValueError, "not a root capture"):
                shared_root_replay._records(path)

    def test_schedule_and_prior_selection_are_explicit(self):
        record = {
            "record_type": "teacher_root_capture",
            "behavior_schedule_id": 2,
            "recorded_player_priors": [["tackle", 1.0]],
            "recorded_opponent_priors": [["protect", 1.0]],
            "schedules": [
                {"schedule_id": 1, "worlds": []},
                {"schedule_id": 2, "worlds": [{"sampled_state": "s"}]},
            ],
        }
        schedule = shared_root_replay._schedule(record)
        self.assertEqual(schedule["schedule_id"], 2)
        self.assertEqual(
            shared_root_replay._priors(record, schedule),
            (
                [["tackle", 1.0]],
                [["protect", 1.0]],
            ),
        )

    def test_total_variation_uses_union_of_action_support(self):
        class Entry:
            def __init__(self, action, probability):
                self.action = action
                self.probability = probability

        class Result:
            def __init__(self, policy):
                self.policy = policy

        left = Result([Entry("a", 0.75), Entry("b", 0.25)])
        right = Result([Entry("a", 0.25), Entry("c", 0.75)])
        self.assertEqual(shared_root_replay._total_variation(left, right), 0.75)


if __name__ == "__main__":
    unittest.main()
