from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.probe_r1_observation_outcomes import ProbeError, binary_metrics, load_first_decisions, split_for_key  # noqa: E402


class ProbeR1ObservationOutcomesTests(unittest.TestCase):
    def test_loads_one_exact_first_decision_per_battle(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "w1" / "shard_00.jsonl"
            shard.parent.mkdir()
            rows = [
                {"schema": 3, "battle_tag": "battle-1", "username": "p1", "decision_idx": 0, "label": 1, "text_tokens": [1, 2], "numbers": [0.1], "illegal_actions": [False] * 13},
                {"schema": 3, "battle_tag": "battle-1", "username": "p1", "decision_idx": 1, "label": 1, "text_tokens": [2, 3], "numbers": [0.2], "illegal_actions": [False] * 13},
            ]
            shard.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="ascii")
            examples = load_first_decisions(root)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].label, 1)
        self.assertIn(split_for_key(examples[0].key), {"train", "development", "test"})

    def test_missing_first_decision_fails_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "w1" / "shard_00.jsonl"
            shard.parent.mkdir()
            shard.write_text(json.dumps({"schema": 3, "battle_tag": "b", "username": "u", "decision_idx": 2, "label": 0}) + "\n")
            with self.assertRaisesRegex(ProbeError, "lack decision_idx 0"):
                load_first_decisions(root)

    def test_binary_metrics_are_battle_level(self):
        metrics = binary_metrics(np.asarray([0.1, 0.8]), np.asarray([0, 1]))
        self.assertAlmostEqual(metrics["brier"], 0.025)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
