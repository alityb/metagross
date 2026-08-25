from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.transformer_value_oracle import (  # noqa: E402
    TransformerValueOracleError,
    append_branch_observations,
)


class Observation:
    def __init__(self, token: int):
        self.token = token

    def policy_payload(self):
        return {
            "text_tokens": [self.token, 2, 3],
            "numbers": [0.1, 0.2],
            "illegal_actions": [False] + [True] * 12,
            "name_table": {"move": 0},
            "terminal": False,
        }


class TransformerValueOracleTests(unittest.TestCase):
    def test_branch_batch_appends_only_observations_and_causal_rl2(self):
        prefix = {
            "text_tokens": torch.tensor([[1, 2, 3]], dtype=torch.int32),
            "numbers": torch.tensor([[0.0, 0.0]]),
            "illegal_actions": torch.tensor([[False] + [True] * 12]),
        }
        obs, rl2, times = append_branch_observations(
            prefix,
            torch.zeros(1, 14),
            [Observation(7), Observation(8)],
            [2, 4],
            [0.5, -0.25],
        )
        self.assertEqual(obs["text_tokens"].shape, (2, 2, 3))
        self.assertEqual(obs["text_tokens"][:, -1, 0].tolist(), [7, 8])
        self.assertEqual(rl2.shape, (2, 2, 14))
        self.assertEqual(rl2[0, -1, [0, 3]].tolist(), [0.5, 1.0])
        self.assertEqual(times.shape, (2, 2, 1))

    def test_mechanical_object_without_policy_payload_fails_closed(self):
        prefix = {
            "text_tokens": torch.tensor([[1]], dtype=torch.int32),
            "numbers": torch.tensor([[0.0]]),
            "illegal_actions": torch.tensor([[False] + [True] * 12]),
        }
        with self.assertRaisesRegex(TransformerValueOracleError, "not a transformer observation"):
            append_branch_observations(prefix, torch.zeros(1, 14), [object()], [0], [0.0])


if __name__ == "__main__":
    unittest.main()
