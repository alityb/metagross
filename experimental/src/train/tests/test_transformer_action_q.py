from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.transformer_action_q import (  # noqa: E402
    DATASET_SCHEMA,
    action_q_loss,
    action_q_metrics,
    battle_split,
    build_stateless_batch,
    load_dataset,
)


class TransformerActionQTests(unittest.TestCase):
    def dataset(self, path: Path):
        records = []
        for index in range(1000):
            support = [False] * 13
            support[:2] = [True, True]
            records.append({
                "battle_id": f"battle-{index}",
                "root_id": f"root-{index}",
                "text_tokens": [index % 7, 2, 3],
                "numbers": [0.1, 0.2],
                "illegal_actions": [False, False] + [True] * 11,
                "teacher_support": support,
                "teacher_q": [0.8, 0.2] + [0.0] * 11,
                "historical_selected_index": 0,
                "source_identity_sha256": "a" * 64,
            })
        torch.save({"schema": DATASET_SCHEMA, "records": records, "provenance": {}}, path)

    def test_dataset_split_and_stateless_batch_are_deterministic(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.pt"
            self.dataset(path)
            dataset = load_dataset(path)
        self.assertEqual(battle_split("battle-1"), battle_split("battle-1"))
        indices = torch.tensor([0, 1])
        obs, rl2, time = build_stateless_batch(dataset, indices, torch.device("cpu"))
        self.assertEqual(obs["text_tokens"].shape, (2, 2, 3))
        self.assertTrue((obs["text_tokens"][:, 0] == 0).all())
        self.assertTrue((rl2 == 0).all())
        self.assertEqual(time.tolist(), [[[0], [1]], [[0], [1]]])

    def test_loss_and_metrics_reward_correct_action_order(self):
        support = torch.tensor([[True, True] + [False] * 11])
        q = torch.tensor([[0.8, 0.2] + [0.0] * 11])
        correct = torch.tensor([[0.3, -0.3] + [0.0] * 11])
        reversed_prediction = -correct
        self.assertLess(
            float(action_q_loss(correct, q, support)),
            float(action_q_loss(reversed_prediction, q, support)),
        )
        metrics = action_q_metrics(correct, q, support, torch.tensor([0]))
        self.assertEqual(metrics["top1_agreement"], 1.0)
        self.assertEqual(metrics["mean_oracle_regret"], 0.0)
        self.assertEqual(metrics["historical_top1_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
