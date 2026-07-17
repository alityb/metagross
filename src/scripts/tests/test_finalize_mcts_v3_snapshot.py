from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.finalize_mcts_v3_snapshot import finalize


def target(tag: str, username: str) -> dict:
    return {"schema": 3, "battle_tag": tag, "username": username, "decision_idx": 0,
            "text_tokens": [1], "numbers": [0.0], "illegal_actions": [False] + [True] * 12,
            "visit_target_13": [1.0] + [0.0] * 12}


class FinalizeMCTSV3SnapshotTests(unittest.TestCase):
    def test_merges_targets_and_retains_matching_pov(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = root / "targets" / "w1"
            targets.mkdir(parents=True)
            targets.joinpath("shard_00.jsonl").write_text(
                json.dumps(target("1", "learner")) + "\n" +
                json.dumps(target("2", "missing")) + "\n"
            )
            parsed = root / "parsed" / "w1" / "shard_00"
            parsed.mkdir(parents=True)
            parsed.joinpath("battle-1_learner_Unrated_learner_vs_peer_WIN.json.lz4").write_bytes(b"x")
            parsed.joinpath("battle-1_peer_Unrated_peer_vs_learner_LOSS.json.lz4").write_bytes(b"x")
            report = finalize(targets.parent, root / "parsed", root / "all.jsonl", root / "learner")
            self.assertEqual(report["targets"], 2)
            self.assertEqual(report["learner_trajectories"], 1)
            self.assertEqual(report["target_groups_without_parsed_trajectory"], 1)
            self.assertEqual(report["parse_only_trajectories"], 1)


if __name__ == "__main__":
    unittest.main()
