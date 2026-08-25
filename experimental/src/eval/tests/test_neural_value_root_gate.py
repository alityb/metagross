from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.neural_value_root_gate import RootResult, evaluate_gate  # noqa: E402


SHA = "a" * 64
HEAD = "b" * 64


def result(
    pair: int,
    arm: str,
    chosen: float,
    action: str,
    *,
    battle_id: str | None = None,
) -> RootResult:
    return RootResult(
        pair_id=f"root-{pair}", arm=arm, budget_ms=500, elapsed_ms=499,
        selected_action=action, oracle_action="best", oracle_best_value=0.8,
        selected_oracle_value=chosen, oracle_artifact_sha256=SHA,
        value_head_sha256=HEAD if arm == "candidate" else None,
        certified_neural_leaves=20 if arm == "candidate" else 0,
        total_leaf_evaluations=100,
        root_id=f"root-{pair}",
        battle_id=battle_id or f"battle-{pair}",
    )


class NeuralValueRootGateTests(unittest.TestCase):
    def test_clear_paired_advantage_passes(self):
        pairs = {
            f"root-{i}": {
                "baseline": result(i, "baseline", 0.6, "other"),
                "candidate": result(i, "candidate", 0.8, "best"),
            }
            for i in range(100)
        }
        report = evaluate_gate(pairs)
        self.assertTrue(report["passed"])
        self.assertGreater(report["metrics"]["paired_bootstrap_95_lower"], 0)

    def test_no_independent_advantage_fails(self):
        pairs = {
            f"root-{i}": {
                "baseline": result(i, "baseline", 0.8, "best"),
                "candidate": result(i, "candidate", 0.8, "best"),
            }
            for i in range(100)
        }
        self.assertFalse(evaluate_gate(pairs)["checks"]["paired_regret_improvement_ci"])

    def test_many_roots_from_one_battle_are_not_independent(self):
        pairs = {
            f"root-{i}": {
                "baseline": result(
                    i, "baseline", 0.6, "other", battle_id="same-battle"
                ),
                "candidate": result(
                    i, "candidate", 0.8, "best", battle_id="same-battle"
                ),
            }
            for i in range(100)
        }
        with self.assertRaisesRegex(ValueError, "independent source battles"):
            evaluate_gate(pairs)


if __name__ == "__main__":
    unittest.main()
