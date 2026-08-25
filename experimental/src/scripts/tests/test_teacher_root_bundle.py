from __future__ import annotations

import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from teacher_root_bundle import (  # noqa: E402
    RootCaptureConfig,
    RootBundleConfig,
    RootBundleError,
    append_root_bundle,
    append_root_capture,
    build_root_capture,
    build_root_bundle,
    config_from_environment,
    capture_config_from_environment,
    derive_seed,
    derive_schedule_seed,
    run_world_treatments,
)


@dataclass
class Option:
    move_choice: str
    visits: int
    total_score: float = 0.0


@dataclass
class Result:
    side_one: list[Option]
    side_two: list[Option]
    total_visits: int


class FakeState:
    @classmethod
    def from_string(cls, value):
        return (cls, value)


class FakeEngine:
    State = FakeState

    def __init__(self):
        self.calls = []

    def monte_carlo_tree_search(self, state, **kwargs):
        self.calls.append((state, kwargs))
        seed = kwargs["seed"]
        visits = kwargs["iterations"]
        return Result(
            [Option("move-a", visits - seed % 2), Option("move-b", seed % 2)],
            [Option("opp-a", visits)],
            visits,
        )


def config(path=Path("bundle.jsonl"), repeats=2, deep_multiplier=4):
    return RootBundleConfig(path, 10, repeats, deep_multiplier, 7, 2.0, True)


def identity():
    return {
        "namespace": "worker-1",
        "battle_tag": "battle-1",
        "username": "learner",
        "decision_idx": 3,
        "battle_turn": 4,
    }


def r1_snapshot(root_identity=None, priors=None):
    root_identity = root_identity or identity()
    priors = priors or [("move-a", 0.8), ("move-b", 0.2)]
    probabilities = [0.0] * 13
    name_table = {}
    for index, (action, mass) in enumerate(priors):
        name_table[action] = index
        probabilities[index] = mass
    return {
        "schema": 3,
        "tag": root_identity["battle_tag"],
        "namespace": root_identity["namespace"],
        "username": root_identity["username"],
        "decision_idx": root_identity["decision_idx"],
        "battle_turn": root_identity["battle_turn"],
        "text_tokens": [1, 2, 3],
        "numbers": [0.25, 0.5],
        "illegal_actions": [False] * len(priors) + [True] * (13 - len(priors)),
        "mask_fallback": False,
        "mask_fallback_error": None,
        "name_table": name_table,
        "probs": probabilities,
        "protocol_prefix": ["|request|{}"],
        "player_information_state": {
            "schema_version": 1, "universal_state": {}, "player_team": [],
            "opponent_public_team": [],
        },
        "player_observation_history": {
            "any_opponent_asleep": False, "any_opponent_frozen": False,
            "revealed_opponents": ["opponent"],
        },
        "continuation_observation_history": {
            "any_opponent_asleep": False, "any_opponent_frozen": False,
            "revealed_opponents": ["opponent", "player"],
        },
    }


class TeacherRootBundleTests(unittest.TestCase):
    def test_capture_configuration_requires_manifest_and_valid_schedule_count(self):
        with self.assertRaisesRegex(RootBundleError, "manifest SHA-256"):
            capture_config_from_environment(
                {"METAGROSS_TEACHER_ROOT_BUNDLE": "capture.jsonl"}
            )
        with self.assertRaisesRegex(RootBundleError, "SCHEDULES must be positive"):
            capture_config_from_environment(
                {
                    "METAGROSS_TEACHER_ROOT_BUNDLE": "capture.jsonl",
                    "METAGROSS_TEACHER_DETERMINIZATION_SCHEDULES": "0",
                    "METAGROSS_TEACHER_MANIFEST_SHA256": "a" * 64,
                }
            )

    def test_capture_only_schedules_are_seeded_private_and_self_hashed(self):
        capture_config = RootCaptureConfig(Path("capture.jsonl"), 2, 19, "a" * 64)
        schedules = [
            [("state-0-0", 0.5), ("state-0-1", 0.5)],
            [("state-1-0", 0.25), ("state-1-1", 0.75)],
        ]
        capture = build_root_capture(
            identity=identity(),
            player_priors=[("move-a", 0.8), ("move-b", 0.2)],
            opponent_priors=[("opp-a", 1.0)],
            r1_policy_snapshot=r1_snapshot(),
            schedules=schedules,
            config=capture_config,
        )
        self.assertEqual(capture["record_type"], "teacher_root_capture")
        self.assertEqual(capture["schema_version"], 3)
        self.assertEqual(capture["r1_policy_snapshot"], r1_snapshot())
        self.assertIsNone(capture["schedules"][0]["sampling_seed"])
        self.assertEqual(
            capture["schedules"][1]["sampling_seed"],
            derive_schedule_seed(19, identity(), 1),
        )
        self.assertNotIn("treatments", json.dumps(capture))
        self.assertNotIn("live_result", json.dumps(capture))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.jsonl"
            append_root_capture(path, capture)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text()), capture)

    def test_capture_rejects_policy_snapshot_drift(self):
        snapshot = r1_snapshot()
        snapshot["probs"][0] = 0.7
        snapshot["probs"][1] = 0.3
        with self.assertRaisesRegex(RootBundleError, "do not match"):
            build_root_capture(
                identity=identity(),
                player_priors=[("move-a", 0.8), ("move-b", 0.2)],
                opponent_priors=[("opp-a", 1.0)],
                r1_policy_snapshot=snapshot,
                schedules=[[("state", 1.0)]],
                config=RootCaptureConfig(Path("capture.jsonl"), 1, 19, "a" * 64),
            )

    def test_capture_prior_order_is_not_semantic(self):
        capture = build_root_capture(
            identity=identity(),
            player_priors=[("move-b", 0.2), ("move-a", 0.8)],
            opponent_priors=[("opp-a", 1.0)],
            r1_policy_snapshot=r1_snapshot(),
            schedules=[[("state", 1.0)]],
            config=RootCaptureConfig(Path("capture.jsonl"), 1, 19, "a" * 64),
        )
        self.assertEqual(capture["recorded_player_priors"][0][0], "move-b")

    def test_capture_rejects_fallback_mask_and_identity_mismatch(self):
        for field, value, message in (
            ("mask_fallback", True, "fallback legality"),
            ("username", "someone-else", "identity does not match"),
        ):
            snapshot = r1_snapshot()
            snapshot[field] = value
            if field == "mask_fallback":
                snapshot["mask_fallback_error"] = "mask failed"
            with self.subTest(field=field), self.assertRaisesRegex(RootBundleError, message):
                build_root_capture(
                    identity=identity(),
                    player_priors=[("move-a", 0.8), ("move-b", 0.2)],
                    opponent_priors=[("opp-a", 1.0)],
                    r1_policy_snapshot=snapshot,
                    schedules=[[("state", 1.0)]],
                    config=RootCaptureConfig(Path("capture.jsonl"), 1, 19, "a" * 64),
                )

    def test_configuration_requires_exact_iterations(self):
        with self.assertRaisesRegex(RootBundleError, "ITERATIONS must be positive"):
            config_from_environment(
                {
                    "METAGROSS_TEACHER_ROOT_BUNDLE": "out.jsonl",
                    "METAGROSS_TEACHER_ITERATIONS": "0",
                }
            )

    def test_configuration_validates_manifest_hash(self):
        with self.assertRaisesRegex(RootBundleError, "64 hexadecimal"):
            config_from_environment(
                {
                    "METAGROSS_TEACHER_ROOT_BUNDLE": "out.jsonl",
                    "METAGROSS_TEACHER_ITERATIONS": "10",
                    "METAGROSS_TEACHER_MANIFEST_SHA256": "not-a-hash",
                }
            )

    def test_seed_derivation_is_stable_and_treatment_specific(self):
        first = derive_seed(7, identity(), "a" * 64, 0, "S-B", 0)
        self.assertEqual(first, derive_seed(7, identity(), "a" * 64, 0, "S-B", 0))
        self.assertNotEqual(first, derive_seed(7, identity(), "a" * 64, 0, "U-B", 0))

    def test_world_treatments_hold_side_two_equal(self):
        engine = FakeEngine()
        live = Result(
            [Option("move-a", 5), Option("move-b", 5)],
            [Option("opp-a", 5), Option("opp-b", 5)],
            10,
        )
        unsharded_identity = identity()
        unsharded_identity["namespace"] = ""
        with patch.dict(sys.modules, {"poke_engine": engine}):
            captured = run_world_treatments(
                state_string="state",
                live_result=live,
                world_index=0,
                identity=unsharded_identity,
                player_priors=[("move-a", 0.8), ("move-b", 0.2)],
                opponent_priors=[("opp-a", 0.9), ("opp-b", 0.1)],
                config=config(),
            )
        self.assertEqual(set(captured["treatments"]), {"U-B", "S-B", "S-4B"})
        self.assertEqual(len(engine.calls), 6)
        u_kwargs = engine.calls[0][1]
        s_kwargs = engine.calls[2][1]
        self.assertEqual(u_kwargs["s2_priors"], s_kwargs["s2_priors"])
        self.assertNotEqual(u_kwargs["s1_priors"], s_kwargs["s1_priors"])
        self.assertTrue(all(call[1]["duration_ms"] == 0 for call in engine.calls))
        self.assertTrue(all(call[1]["threads"] == 1 for call in engine.calls))

    def test_deep_treatment_label_matches_multiplier(self):
        engine = FakeEngine()
        live = Result([Option("move-a", 10)], [Option("opp-a", 10)], 10)
        with patch.dict(sys.modules, {"poke_engine": engine}):
            captured = run_world_treatments(
                state_string="state",
                live_result=live,
                world_index=0,
                identity=identity(),
                player_priors=[("move-a", 1.0)],
                opponent_priors=None,
                config=config(repeats=1, deep_multiplier=2),
            )
        self.assertEqual(set(captured["treatments"]), {"U-B", "S-B", "S-2B"})
        self.assertEqual(captured["treatments"]["S-2B"][0]["iterations"], 20)

    def test_bundle_aggregates_worlds_and_is_self_hashed(self):
        results = []
        for index, weight in enumerate((0.25, 0.75)):
            live = Result([Option("move-a", 5), Option("move-b", 5)], [Option("opp-a", 10)], 10)
            capture = {
                "world_index": index,
                "state_sha256": str(index) * 64,
                "recorded_player_priors": [["move-a", 0.8], ["move-b", 0.2]],
                "effective_player_priors": [["move-a", 0.8], ["move-b", 0.2]],
                "recorded_opponent_priors": [],
                "equal_side_one_priors": [["move-a", 0.5], ["move-b", 0.5]],
                "equal_side_two_priors": [["opp-a", 1.0]],
                "treatments": {
                    "S-B": [
                        {
                            "repeat": 0,
                            "seed": index,
                            "iterations": 10,
                            "result": {
                                "total_visits": 10,
                                "side_one": [
                                    {"action": "move-a", "visits": 10 - index * 10, "total_score": 0.0},
                                    {"action": "move-b", "visits": index * 10, "total_score": 0.0},
                                ],
                                "side_two": [{"action": "opp-a", "visits": 10, "total_score": 0.0}],
                            },
                        }
                    ]
                },
                "sampled_state": f"state-{index}",
            }
            live._teacher_root_capture = {
                "identity": identity(),
                "configuration": {
                    "iterations": 10,
                    "repeats": 1,
                    "deep_multiplier": 1,
                    "base_seed": 7,
                    "c_puct": 2.0,
                },
                "world_index": index,
                "world": capture,
            }
            results.append((live, weight, index))
        bundle = build_root_bundle(results)
        policy = bundle["aggregate_treatments"]["S-B"][0]["side_one_policy"]
        self.assertEqual(policy, [
            {"action": "move-a", "probability": 0.25},
            {"action": "move-b", "probability": 0.75},
        ])
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.jsonl"
            append_root_bundle(path, bundle)
            stored = json.loads(path.read_text())
        self.assertEqual(stored["bundle_sha256"], bundle["bundle_sha256"])

    def test_bundle_rejects_missing_capture(self):
        live = Result([Option("move-a", 1)], [Option("opp-a", 1)], 1)
        with self.assertRaisesRegex(RootBundleError, "missing deterministic"):
            build_root_bundle([(live, 1.0, 0)])


if __name__ == "__main__":
    unittest.main()
