from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from srcs.metagross.p16_handoff import audit_run


class P16HandoffTest(unittest.TestCase):
    def test_audit_accepts_joined_remote_trajectories(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            engine = "a" * 64
            manifest = {
                "status": "completed",
                "ladder": {"games": 1},
                "search": {
                    "execution": "modal",
                    "parallelism": 16,
                    "modal": {"engine_sha256": engine},
                },
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            decision = {"tag": "battle-test", "decision_idx": 0}
            search = {
                "context": decision,
                "remote_search": {
                    "worlds": 32,
                    "engine": {"native_sha256": engine},
                    "timings": [{}] * 32,
                },
            }
            rows = {
                "decisions.jsonl": [decision],
                "search.jsonl": [search],
                "protocol.jsonl": [{"message": "|win|bot"}],
                "ratings.jsonl": [{}],
            }
            for name, values in rows.items():
                (run_dir / name).write_text("".join(json.dumps(row) + "\n" for row in values))
            (run_dir / "client.log").write_text("clean\n")
            (run_dir / "prior.log").write_text("clean\n")
            self.assertEqual(audit_run(run_dir, 1, engine), {"decisions": 1, "outcomes": 1})


if __name__ == "__main__":
    unittest.main()
