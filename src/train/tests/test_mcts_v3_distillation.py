from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from train.mcts_v3_distillation import (  # noqa: E402
    NUM_ACTIONS,
    V3BatchSampler,
    V3DatasetError,
    build_stateless_batch,
    load_v3_records,
    v3_distillation_terms,
)
from train.mcts_policy_distillation import add_distillation_loss  # noqa: E402


def make_record(**overrides):
    row = {
        "schema": 3,
        "battle_tag": "gen9randombattle-1",
        "username": "learner",
        "decision_idx": 0,
        "text_tokens": [1, 2, 3, 4],
        "numbers": [0.5, 0.25, 0.125],
        "illegal_actions": [False, False] + [True] * (NUM_ACTIONS - 2),
        "visit_target_13": [0.75, 0.25] + [0.0] * (NUM_ACTIONS - 2),
    }
    row.update(overrides)
    return row


def write_records(directory, rows, name="v3.jsonl"):
    path = Path(directory) / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


class TestLoadV3Records(unittest.TestCase):
    def test_happy_path(self):
        with TemporaryDirectory() as tmp:
            path = write_records(tmp, [make_record(), make_record(decision_idx=1)])
            records = load_v3_records(path)
        self.assertEqual(records["count"], 2)
        self.assertEqual(records["text_tokens"].shape, (2, 4))
        self.assertEqual(records["text_tokens"].dtype, torch.int32)
        self.assertEqual(records["numbers"].shape, (2, 3))
        self.assertEqual(records["illegal_actions"].shape, (2, NUM_ACTIONS))
        self.assertEqual(records["illegal_actions"].dtype, torch.bool)
        self.assertEqual(records["targets"].shape, (2, NUM_ACTIONS))
        self.assertAlmostEqual(records["targets"][0].sum().item(), 1.0, places=6)

    def assert_load_fails(self, row):
        with TemporaryDirectory() as tmp:
            path = write_records(tmp, [row])
            with self.assertRaises(V3DatasetError):
                load_v3_records(path)

    def test_rejects_wrong_schema(self):
        self.assert_load_fails(make_record(schema=2))

    def test_rejects_mass_on_illegal(self):
        illegal = [True, False] + [True] * (NUM_ACTIONS - 2)
        self.assert_load_fails(make_record(illegal_actions=illegal))

    def test_rejects_unnormalized_target(self):
        target = [0.5, 0.0] + [0.0] * (NUM_ACTIONS - 2)
        self.assert_load_fails(make_record(visit_target_13=target))

    def test_rejects_all_illegal(self):
        target = [0.0] * NUM_ACTIONS
        target[0] = 1.0
        self.assert_load_fails(
            make_record(illegal_actions=[True] * NUM_ACTIONS, visit_target_13=target)
        )

    def test_rejects_inconsistent_obs_shape(self):
        with TemporaryDirectory() as tmp:
            path = write_records(
                tmp, [make_record(), make_record(text_tokens=[1, 2, 3])]
            )
            with self.assertRaises(V3DatasetError):
                load_v3_records(path)

    def test_rejects_empty_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "v3.jsonl"
            path.write_text("")
            with self.assertRaises(V3DatasetError):
                load_v3_records(path)


class TestV3BatchSampler(unittest.TestCase):
    def test_epoch_covers_all_without_replacement(self):
        sampler = V3BatchSampler(count=10, batch_size=5, seed=1)
        seen = set(sampler.next_indices().tolist())
        seen |= set(sampler.next_indices().tolist())
        self.assertEqual(seen, set(range(10)))

    def test_batch_larger_than_count_clamps(self):
        sampler = V3BatchSampler(count=3, batch_size=8, seed=1)
        self.assertEqual(len(sampler.next_indices()), 3)

    def test_reshuffles_after_epoch(self):
        sampler = V3BatchSampler(count=6, batch_size=6, seed=1)
        first = sampler.next_indices().tolist()
        second = sampler.next_indices().tolist()
        self.assertEqual(sorted(first), sorted(second))


class TestBuildStatelessBatch(unittest.TestCase):
    def records(self):
        with TemporaryDirectory() as tmp:
            path = write_records(
                tmp, [make_record(), make_record(decision_idx=1), make_record(decision_idx=2)]
            )
            return load_v3_records(path)

    def test_layout_matches_prior_server(self):
        records = self.records()
        obs, rl2s, time_idxs, targets, illegal = build_stateless_batch(
            records, torch.tensor([0, 2]), torch.device("cpu")
        )
        self.assertEqual(obs["text_tokens"].shape, (2, 2, 4))
        self.assertTrue((obs["text_tokens"][:, 0] == 0).all())
        self.assertEqual(obs["numbers"].shape, (2, 2, 3))
        self.assertTrue((obs["numbers"][:, 0] == 0).all())
        self.assertEqual(obs["illegal_actions"].shape, (2, 2, NUM_ACTIONS))
        # blank first step must mask every action, exactly like the server
        self.assertTrue(obs["illegal_actions"][:, 0].all())
        self.assertTrue(torch.equal(obs["illegal_actions"][:, 1], illegal))
        self.assertEqual(rl2s.shape, (2, 2, NUM_ACTIONS + 1))
        self.assertTrue((rl2s == 0).all())
        self.assertEqual(time_idxs.shape, (2, 2, 1))
        self.assertTrue(torch.equal(time_idxs[0, :, 0], torch.tensor([0, 1])))
        self.assertEqual(targets.shape, (2, NUM_ACTIONS))


class FakeActorDist:
    def __init__(self, probs):
        self.probs = probs


class FakeAgent:
    """Mimics the AMAGO agent surface used by v3_distillation_terms."""

    pass_obs_keys_to_actor = ("numbers", "illegal_actions")

    def __init__(self, gammas=2):
        self.gammas = gammas
        self.actor_obs_seen = None

    def get_state_embedding(self, obs, rl2s, time_idxs, hidden_state=None):
        batch, steps = obs["text_tokens"].shape[:2]
        # the real MetamonTstepEncoder/TrajEncoder preserve sequence length,
        # so the last embedding corresponds to the real (second) obs step
        return torch.zeros((batch, steps, 8)), None

    def actor(self, emb, straight_from_obs):
        self.actor_obs_seen = straight_from_obs
        batch, seq = emb.shape[0], emb.shape[1]
        illegal = straight_from_obs["illegal_actions"][:, :seq]
        legal = (~illegal).float()
        probs = legal / legal.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        probs = probs.unsqueeze(-2).expand(batch, seq, self.gammas, NUM_ACTIONS)
        return FakeActorDist(probs)


class TestDistillationLoss(unittest.TestCase):
    def setUp(self):
        with TemporaryDirectory() as tmp:
            path = write_records(tmp, [make_record(), make_record(decision_idx=1)])
            self.records = load_v3_records(path)

    def terms(self):
        agent = FakeAgent()
        obs, rl2s, time_idxs, targets, illegal = build_stateless_batch(
            self.records, torch.tensor([0, 1]), torch.device("cpu")
        )
        return agent, v3_distillation_terms(agent, obs, rl2s, time_idxs, targets, illegal)

    def test_shapes_and_actor_slicing(self):
        agent, (probs, target, mask) = self.terms()
        self.assertEqual(probs.shape, (2, 2, NUM_ACTIONS))
        self.assertEqual(target.shape, (2, 1, NUM_ACTIONS))
        self.assertEqual(mask["valid"].shape, (2, 1, 1))
        self.assertEqual(mask["illegal_actions"].shape, (2, 1, NUM_ACTIONS))
        # actor side-channel obs are sliced to the embedding length (a no-op
        # for length-preserving encoders), matching the prior server's call
        self.assertEqual(agent.actor_obs_seen["illegal_actions"].shape[1], 2)

    def test_known_cross_entropy_value(self):
        _, (probs, target, mask) = self.terms()
        total = add_distillation_loss(
            torch.tensor(0.0), probs, target, mask, coefficient=1.0
        )
        # fake actor emits uniform-over-2-legal (p=0.5); target [.75,.25]
        # CE = -(0.75*log(.5) + 0.25*log(.5)) = log(2)
        self.assertAlmostEqual(total.item(), float(torch.log(torch.tensor(2.0))), places=5)

    def test_zero_coefficient_is_noop(self):
        _, (probs, target, mask) = self.terms()
        base = torch.tensor(1.234)
        total = add_distillation_loss(base, probs, target, mask, coefficient=0.0)
        self.assertEqual(total.item(), base.item())

    def test_gradient_flows_to_probs(self):
        _, (probs, target, mask) = self.terms()
        probs = probs.detach().clone().requires_grad_(True)
        total = add_distillation_loss(
            torch.tensor(0.0), probs, target, mask, coefficient=0.1
        )
        total.backward()
        self.assertIsNotNone(probs.grad)
        self.assertTrue(torch.isfinite(probs.grad).all())


class TestSmokeDatasetLoads(unittest.TestCase):
    """The real smoke output from build_mcts_v3_dataset.py must load cleanly."""

    SMOKE = ROOT.parent / "experiments" / "v3_smoke2" / "v3_targets.jsonl"

    def test_smoke_targets_load(self):
        if not self.SMOKE.is_file():
            self.skipTest("smoke targets not present")
        records = load_v3_records(self.SMOKE)
        self.assertGreater(records["count"], 100)


if __name__ == "__main__":
    unittest.main()
