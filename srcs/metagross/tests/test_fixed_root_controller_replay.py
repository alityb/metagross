from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from srcs.metagross import fixed_root_controller_replay


class FixedRootControllerReplayTest(unittest.TestCase):
    def test_rejected_certificate_cannot_change_captured_search_choice(self):
        snapshot = {
            "total_visits": 10,
            "side_one": [
                {"action": "search", "visits": 8, "total_score": 4.0},
                {"action": "policy", "visits": 2, "total_score": 1.0},
            ],
            "side_two": [{"action": "opponent", "visits": 10, "total_score": 0.0}],
        }
        capture = {
            "record_type": "teacher_root_capture",
            "capture_sha256": "a" * 64,
            "identity": {
                "battle_tag": "battle-test",
                "battle_turn": 1,
                "decision_idx": 0,
                "namespace": "",
                "username": "test",
            },
            "recorded_player_priors": [["policy", 0.9], ["search", 0.1]],
            "behavior_schedule_id": 0,
            "schedules": [
                {
                    "schedule_id": 0,
                    "worlds": [
                        {"live_result": snapshot, "sample_weight": 1.0}
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "roots.jsonl"
            path.write_text(json.dumps(capture) + "\n", encoding="ascii")

            report = fixed_root_controller_replay.verify_files([path])

        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(report["roots"][0]["first_choice"], "search")
        self.assertTrue(report["roots"][0]["rejected_certificate_was_shadow_only"])


if __name__ == "__main__":
    unittest.main()
