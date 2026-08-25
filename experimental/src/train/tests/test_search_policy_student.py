from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.search_policy_student import (  # noqa: E402
    HYBRID_ACTION_WEIGHT,
    NUM_ACTIONS,
    SearchPolicyDatasetError,
    battle_split,
    build_stateless_batch,
    deployment_policy_probs,
    load_search_policy_dataset,
    policy_cross_entropy,
)


def row(battle_tag: str, decision_idx: int = 0, **overrides):
    value = {
        "schema": 3,
        "battle_tag": battle_tag,
        "username": "learner",
        "decision_idx": decision_idx,
        "text_tokens": [1, 2, 3],
        "numbers": [0.5, 0.25],
        "illegal_actions": [False, False] + [True] * (NUM_ACTIONS - 2),
        "policy_probs": [0.6, 0.4] + [0.0] * (NUM_ACTIONS - 2),
        "selected_action_index": 1,
        "visit_target_13": [0.25, 0.75] + [0.0] * (NUM_ACTIONS - 2),
    }
    value.update(overrides)
    return value


def tags_for_all_splits():
    found = {}
    index = 0
    while len(found) < 3:
        tag = f"battle-{index}"
        found.setdefault(battle_split(tag), tag)
        index += 1
    return found


def write_dataset(root: Path, rows) -> Path:
    path = root / "targets.jsonl"
    path.write_text("".join(json.dumps(value) + "\n" for value in rows))
    return path


class FakeDistribution:
    def __init__(self, probs):
        self.probs = probs


class FakeAgent:
    pass_obs_keys_to_actor = ("illegal_actions",)

    def get_state_embedding(self, obs, rl2s, time_idxs, hidden_state=None):
        return torch.zeros((len(rl2s), 2, 4)), None

    def actor(self, embedding, straight_from_obs):
        illegal = straight_from_obs["illegal_actions"]
        legal = (~illegal).float()
        probs = legal / legal.sum(-1, keepdim=True).clamp_min(1.0)
        # Two gamma heads with deliberately different final-head probabilities.
        first = probs.unsqueeze(-2)
        second = torch.flip(probs, dims=(-1,)).unsqueeze(-2)
        return FakeDistribution(torch.cat((first, second), dim=-2))


class SearchPolicyStudentTests(unittest.TestCase):
    def test_loads_all_arms_and_splits_by_battle(self):
        tags = tags_for_all_splits()
        rows = [row(tag, decision) for tag in tags.values() for decision in range(2)]
        with TemporaryDirectory() as temporary:
            dataset = load_search_policy_dataset(
                write_dataset(Path(temporary), rows)
            )
        self.assertEqual(dataset.count, 6)
        self.assertEqual(dataset.split_counts, {name: 2 for name in tags})
        self.assertTrue(torch.equal(dataset.action_targets[:, 1], torch.ones(6)))
        self.assertTrue(torch.allclose(dataset.visit_targets[:, :2], torch.tensor([0.25, 0.75]).expand(6, 2)))
        hybrid = dataset.targets("hybrid")
        expected = (
            HYBRID_ACTION_WEIGHT * dataset.action_targets
            + (1.0 - HYBRID_ACTION_WEIGHT) * dataset.visit_targets
        )
        self.assertTrue(torch.allclose(hybrid, expected))
        self.assertTrue(torch.allclose(hybrid.sum(dim=-1), torch.ones(6)))

    def test_rejects_duplicate_decision_identity(self):
        tags = tags_for_all_splits()
        rows = [row(tag) for tag in tags.values()]
        rows.append(rows[0])
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(SearchPolicyDatasetError, "duplicate"):
                load_search_policy_dataset(write_dataset(Path(temporary), rows))

    def test_rejects_illegal_target_mass(self):
        tags = tags_for_all_splits()
        rows = [row(tag) for tag in tags.values()]
        rows[0]["visit_target_13"] = [0.25, 0.5, 0.25] + [0.0] * 10
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(SearchPolicyDatasetError, "illegal"):
                load_search_policy_dataset(write_dataset(Path(temporary), rows))

    def test_deployment_batch_and_final_gamma_head(self):
        tags = tags_for_all_splits()
        with TemporaryDirectory() as temporary:
            dataset = load_search_policy_dataset(
                write_dataset(Path(temporary), [row(tag) for tag in tags.values()])
            )
        obs, rl2s, time_idxs, targets = build_stateless_batch(
            dataset, torch.tensor([0]), "visits", torch.device("cpu")
        )
        self.assertTrue(obs["illegal_actions"][:, 0].all())
        self.assertEqual(tuple(rl2s.shape), (1, 2, 14))
        probs = deployment_policy_probs(FakeAgent(), obs, rl2s, time_idxs)
        # The fake actor's final gamma reverses the action dimension.
        self.assertEqual(float(probs[0, 0]), 0.0)
        self.assertEqual(float(probs[0, -2]), 0.5)
        self.assertEqual(float(probs[0, -1]), 0.5)
        self.assertEqual(tuple(targets.shape), (1, NUM_ACTIONS))

    def test_policy_loss_is_cross_entropy(self):
        probs = torch.tensor([[0.5, 0.5] + [0.0] * 11])
        target = torch.tensor([[0.75, 0.25] + [0.0] * 11])
        self.assertAlmostEqual(
            float(policy_cross_entropy(probs, target)),
            float(torch.log(torch.tensor(2.0))),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
