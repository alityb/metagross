from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.verify_r1_policy_snapshots import (  # noqa: E402
    PolicySnapshotParityError,
    compare_snapshots,
    load_snapshots,
)


def snapshot():
    return {
        "schema": 3,
        "text_tokens": [1, 2],
        "numbers": [0.5],
        "illegal_actions": [False, False] + [True] * 11,
        "probs": [0.75, 0.25] + [0.0] * 11,
    }


class VerifyR1PolicySnapshotsTests(unittest.TestCase):
    def test_loads_direct_and_nested_snapshots(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshots.jsonl"
            path.write_text(
                json.dumps(snapshot()) + "\n" + json.dumps({"r1_policy_snapshot": snapshot()}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_snapshots(path), [snapshot(), snapshot()])

    def test_exact_probability_parity_passes(self):
        report = compare_snapshots(
            np.asarray([snapshot()["probs"]]), [snapshot()], absolute_tolerance=0.0
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["max_absolute_error"], 0.0)

    def test_probability_drift_fails_closed(self):
        actual = np.asarray([snapshot()["probs"]], dtype=np.float64)
        actual[0, 0] -= 1e-4
        actual[0, 1] += 1e-4
        with self.assertRaisesRegex(PolicySnapshotParityError, "mismatched_snapshots"):
            compare_snapshots(actual, [snapshot()], absolute_tolerance=1e-7)


if __name__ == "__main__":
    unittest.main()
