from __future__ import annotations

import hashlib
import sys
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.online_rl_controller import (
    ControllerError,
    admitted_collection_source,
    arena_decision,
    collection_pool,
    validate_config,
    wilson_interval,
)


class OnlineRLControllerTests(unittest.TestCase):
    def test_wilson_interval_matches_generation_one_arena(self):
        low, high = wilson_interval(105, 200)
        self.assertAlmostEqual(low, 0.45599, places=4)
        self.assertAlmostEqual(high, 0.59306, places=4)

    def test_directional_result_advances_lineage_but_does_not_promote(self):
        decision = arena_decision(105, 200, lineage_floor=0.45, promotion_min_games=400)
        self.assertTrue(decision["advance_lineage"])
        self.assertFalse(decision["promote_accepted"])

    def test_collection_pool_keeps_current_and_accepted(self):
        current = {"id": "g1", "run_name": "g1", "checkpoint": 1, "sha256": "a" * 64}
        accepted = {"id": "r1", "run_name": "r1", "checkpoint": 5, "sha256": "b" * 64}
        pool = collection_pool(current, accepted)
        self.assertEqual(set(pool["profiles"]), {"g1", "r1"})
        self.assertEqual(sum(member["base_weight"] for member in pool["pfsp"]["pool"]), 1.0)

    def test_collection_pool_supports_weighted_current_and_history(self):
        current = {"id": "g4", "run_name": "g4", "checkpoint": 1, "sha256": "a" * 64}
        accepted = dict(current)
        history = {"id": "g3", "run_name": "g3", "checkpoint": 1, "sha256": "b" * 64}
        pool = collection_pool(current, accepted, [
            {"source": "current", "base_weight": 0.8},
            {"snapshot": history, "base_weight": 0.2},
        ])
        self.assertEqual(set(pool["profiles"]), {"g4", "g3"})
        self.assertEqual(pool["pfsp"]["pool"], [
            {"id": "g4", "base_weight": 0.8},
            {"id": "g3", "base_weight": 0.2},
        ])

    def test_config_rejects_underpowered_arena_and_accepts_safe_worker_defaults(self):
        config = {
            "generations": 1, "collection_games": 10, "arena_games": 20,
            "learner_steps": 10, "batch_size": 2, "promotion_min_games": 21,
        }
        with self.assertRaisesRegex(ControllerError, "arena_games"):
            validate_config(config)
        config["promotion_min_games"] = 20
        validate_config(config)

    def test_collection_admission_requires_fresh_exact_totals(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "collection_kind": "fresh", "failed_shards": 0,
                "requested_battles": 2, "completed_battles": 2,
                "learner_wins": 1, "learner_losses": 1,
                "learner_trajectory_count": 2,
                "chunks": [{"requested_battles": 2, "phases": [{
                    "completed_battles": 2, "learner_wins": 1, "learner_losses": 1,
                    "learner_trajectory_count": 2,
                }]}],
            }
            ledger_payload = '{"battle": 1}\n{"battle": 2}\n'
            (root / "BATTLE_LEDGER.jsonl").write_text(ledger_payload)
            manifest["battle_ledger"] = {
                "path": "BATTLE_LEDGER.jsonl",
                "records": 2,
                "sha256": hashlib.sha256(ledger_payload.encode()).hexdigest(),
            }
            (root / "MANIFEST.json").write_text(json.dumps(manifest))
            source = admitted_collection_source(2, root)
            self.assertEqual(source["learner_trajectory_count"], 2)
            manifest["collection_kind"] = "arena"
            (root / "MANIFEST.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ControllerError, "not an admitted collection"):
                admitted_collection_source(2, root)

    def test_scaled_config_has_requested_production_shape(self):
        config = json.loads(
            (ROOT.parent / "configs" / "online_rl_scaled_replay_5k.json").read_text()
        )
        validate_config(config)
        self.assertEqual(
            {key: config[key] for key in (
                "collection_games", "workers", "chunk_games", "collector_torch_threads",
                "arena_games", "learner_steps",
            )},
            {
                "collection_games": 5000, "workers": 4, "chunk_games": 25,
                "collector_torch_threads": 2,
                "arena_games": 500, "learner_steps": 3000,
            },
        )
        self.assertEqual(config["retain_fresh_generations"], "current")
        self.assertEqual(config["collection_backend"], "modal")
        self.assertFalse(config["automatic_promotion"])
        self.assertEqual(config["initial_current"]["id"], "online_g4_freshfix")
        self.assertEqual(config["initial_accepted"]["id"], "online_g4_freshfix")
        self.assertEqual(
            [entry["base_weight"] for entry in config["collection_opponents"]],
            [0.8, 0.1, 0.1],
        )

        config["collector_torch_threads"] = 0
        with self.assertRaisesRegex(ControllerError, "collector_torch_threads"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
