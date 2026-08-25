from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.transformer_terminal_value import (  # noqa: E402
    TerminalValueDatasetError,
    TransformerValueHead,
    battle_balanced_bce,
    battle_split,
    build_value_batch,
    load_terminal_value_dataset,
    value_metrics,
)


def row(tag: str, pov: str, source: str, outcome: int, length: int = 2) -> dict:
    return {
        "schema": 1,
        "battle_tag": tag,
        "pov": pov,
        "source_kind": source,
        "outcome": outcome,
        "text_tokens": [[1, 2, 3] for _ in range(length)],
        "numbers": [[0.25, 0.5] for _ in range(length)],
        "illegal_actions": [[False] + [True] * 12 for _ in range(length)],
        "rl2": [[0.0] * 14] + [[0.0, 1.0] + [0.0] * 12 for _ in range(length - 1)],
    }


class TransformerTerminalValueTests(unittest.TestCase):
    def write_rows(self, rows):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "value.jsonl"
        path.write_text("".join(json.dumps(value) + "\n" for value in rows))
        self.addCleanup(directory.cleanup)
        return path

    def test_same_battle_povs_cannot_cross_splits_and_are_balanced(self):
        path = self.write_rows(
            [
                row("battle-a", "alice", "human", 1, length=2),
                row("battle-a", "bob", "human", 0, length=4),
                row("battle-b", "learner", "selfplay", 1, length=3),
                row("battle-c", "learner", "league", 0, length=2),
            ]
        )
        dataset = load_terminal_value_dataset([path])
        self.assertEqual(dataset.split_ids[0], dataset.split_ids[1])
        self.assertEqual(dataset.source_counts, {"human": 2, "selfplay": 1, "league": 1})
        obs, rl2, time_idxs, labels, weights = build_value_batch(
            dataset, torch.tensor([0, 1]), torch.device("cpu")
        )
        self.assertEqual(obs["text_tokens"].shape, (2, 4, 3))
        self.assertEqual(rl2.shape, (2, 4, 14))
        self.assertEqual(time_idxs.shape, (2, 4, 1))
        self.assertEqual(labels.shape, weights.shape)
        self.assertAlmostEqual(float(weights[0].sum()), 0.5)
        self.assertAlmostEqual(float(weights[1].sum()), 0.5)

    def test_duplicate_trajectory_is_rejected(self):
        duplicate = row("battle-a", "alice", "human", 1)
        path = self.write_rows([duplicate, duplicate])
        with self.assertRaisesRegex(TerminalValueDatasetError, "duplicate trajectory"):
            load_terminal_value_dataset([path])

    def test_rl2_must_start_at_causal_zero_boundary(self):
        invalid = row("battle-a", "alice", "human", 1)
        invalid["rl2"][0][1] = 1.0
        path = self.write_rows([invalid])
        with self.assertRaisesRegex(TerminalValueDatasetError, "causal zero"):
            load_terminal_value_dataset([path])

    def test_head_loss_and_metrics_ignore_zero_weight_padding(self):
        head = TransformerValueHead(embedding_dim=4, hidden_dim=3, dropout=0.0)
        logits = head(torch.zeros(2, 3, 4))
        self.assertEqual(logits.shape, (2, 3))
        labels = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        weights = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
        loss = battle_balanced_bce(torch.zeros_like(labels), labels, weights)
        self.assertAlmostEqual(float(loss), 0.693147, places=5)
        metrics = value_metrics(torch.zeros_like(labels), labels, weights)
        self.assertAlmostEqual(metrics["brier"], 0.25)

    def test_split_function_has_all_three_partitions(self):
        observed = {battle_split(f"battle-{index}") for index in range(1000)}
        self.assertEqual(observed, {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
