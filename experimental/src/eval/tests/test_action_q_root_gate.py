from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.action_q_root_gate import evaluate_gate  # noqa: E402
from eval.neural_value_root_gate import RootResult  # noqa: E402


SHA = "a" * 64
MODEL = "b" * 64


def result(index: int, arm: str, chosen: float, action: str, battle: str | None = None) -> RootResult:
    return RootResult(
        pair_id=f"pair-{index}", arm=arm, budget_ms=500, elapsed_ms=500,
        selected_action=action, oracle_action="best", oracle_best_value=0.8,
        selected_oracle_value=chosen, oracle_artifact_sha256=SHA,
        value_head_sha256=MODEL if arm == "candidate" else None,
        certified_neural_leaves=5 if arm == "candidate" else 0,
        total_leaf_evaluations=5, root_id=f"root-{index}",
        battle_id=battle or f"battle-{index}",
    )


class ActionQRootGateTests(unittest.TestCase):
    def test_clear_advantage_passes(self):
        pairs = {
            f"pair-{i}": {
                "baseline": result(i, "baseline", 0.6, "other"),
                "candidate": result(i, "candidate", 0.8, "best"),
            }
            for i in range(100)
        }
        report = evaluate_gate(pairs)
        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["action_q_prior_coverage"], 1.0)

    def test_incomplete_guidance_fails(self):
        pairs = {
            f"pair-{i}": {
                "baseline": result(i, "baseline", 0.6, "other"),
                "candidate": result(i, "candidate", 0.8, "best"),
            }
            for i in range(100)
        }
        pairs["pair-0"]["candidate"] = RootResult(
            **{**pairs["pair-0"]["candidate"].__dict__, "certified_neural_leaves": 4}
        )
        self.assertFalse(evaluate_gate(pairs)["checks"]["action_q_prior_coverage"])

    def test_one_battle_is_rejected(self):
        pairs = {
            f"pair-{i}": {
                "baseline": result(i, "baseline", 0.6, "other", "same"),
                "candidate": result(i, "candidate", 0.8, "best", "same"),
            }
            for i in range(100)
        }
        with self.assertRaisesRegex(ValueError, "independent source battles"):
            evaluate_gate(pairs)


if __name__ == "__main__":
    unittest.main()
