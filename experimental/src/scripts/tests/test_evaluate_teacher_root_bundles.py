from __future__ import annotations

import json
import stat
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_teacher_root_bundles import (  # noqa: E402
    evaluate_bundle,
    evaluate_capture,
    evaluate_file,
    validate_root_evaluation,
)
from scripts.teacher_root_bundle import (  # noqa: E402
    RootCaptureConfig,
    RootBundleError,
    build_root_capture,
    build_root_bundle,
    validate_root_bundle,
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

    @staticmethod
    def root_options(state):
        return ["move-a", "move-b"], ["opp-a"]

    @staticmethod
    def monte_carlo_tree_search(state, **kwargs):
        visits = kwargs["iterations"]
        return Result(
            [Option(action, visits if index == 0 else 0) for index, (action, _) in enumerate(kwargs["s1_priors"])],
            [Option(action, visits if index == 0 else 0) for index, (action, _) in enumerate(kwargs["s2_priors"])],
            visits,
        )


def source_bundle() -> dict:
    live = Result(
        [Option("move-a", 6), Option("move-b", 4)],
        [Option("opp-a", 10)],
        10,
    )
    live._teacher_root_capture = {
        "identity": {
            "namespace": "",
            "battle_tag": "battle-1",
            "username": "learner",
            "decision_idx": 0,
            "battle_turn": 1,
        },
        "configuration": {
            "iterations": 1,
            "repeats": 1,
            "deep_multiplier": 1,
            "base_seed": 1,
            "c_puct": 2.0,
            "input_manifest_sha256": "a" * 64,
        },
        "world_index": 0,
        "world": {
            "world_index": 0,
            "state_sha256": "4ba69735ca53765ed6a709edb56c6ea236b7193a3b29a6b390c346f0f4340e4e",
            "live_result": {
                "total_visits": 10,
                "side_one": [
                    {"action": "move-a", "visits": 6, "total_score": 0.0},
                    {"action": "move-b", "visits": 4, "total_score": 0.0},
                ],
                "side_two": [{"action": "opp-a", "visits": 10, "total_score": 0.0}],
            },
            "recorded_player_priors": [["move-a", 0.75], ["move-b", 0.25]],
            "effective_player_priors": [["move-a", 0.75], ["move-b", 0.25]],
            "recorded_opponent_priors": [["opp-a", 1.0]],
            "equal_side_one_priors": [["move-a", 0.5], ["move-b", 0.5]],
            "equal_side_two_priors": [["opp-a", 1.0]],
            "treatments": {
                "S-B": [
                    {
                        "repeat": 0,
                        "seed": 1,
                        "iterations": 1,
                        "result": {
                            "total_visits": 1,
                            "side_one": [
                                {"action": "move-a", "visits": 1, "total_score": 0.0},
                                {"action": "move-b", "visits": 0, "total_score": 0.0},
                            ],
                            "side_two": [{"action": "opp-a", "visits": 1, "total_score": 0.0}],
                        },
                    }
                ]
            },
            "sampled_state": "state",
        },
    }
    return build_root_bundle([(live, 1.0, 0)])


class OfflineTeacherEvaluationTests(unittest.TestCase):
    def test_capture_schedules_are_evaluated_and_aggregated_separately(self):
        capture = build_root_capture(
            identity={
                "namespace": "",
                "battle_tag": "battle-1",
                "username": "learner",
                "decision_idx": 0,
                "battle_turn": 1,
            },
            player_priors=[("move-a", 0.75), ("move-b", 0.25)],
            opponent_priors=[("opp-a", 1.0)],
            r1_policy_snapshot={
                "schema": 3, "tag": "battle-1", "namespace": "", "username": "learner",
                "decision_idx": 0, "battle_turn": 1, "text_tokens": [1], "numbers": [0.0],
                "illegal_actions": [False, False] + [True] * 11, "mask_fallback": False,
                "mask_fallback_error": None, "name_table": {"move-a": 0, "move-b": 1},
                "probs": [0.75, 0.25] + [0.0] * 11, "protocol_prefix": ["|request|{}"],
                "player_information_state": {"schema_version": 1, "universal_state": {}, "player_team": [], "opponent_public_team": []},
                "player_observation_history": {"any_opponent_asleep": False, "any_opponent_frozen": False, "revealed_opponents": ["opponent"]},
                "continuation_observation_history": {"any_opponent_asleep": False, "any_opponent_frozen": False, "revealed_opponents": ["opponent", "player"]},
            },
            schedules=[[("state-a", 1.0)], [("state-b", 1.0)]],
            config=RootCaptureConfig(Path("unused"), 2, 17, "a" * 64),
        )
        with patch.dict(sys.modules, {"poke_engine": FakeEngine()}):
            evaluated = evaluate_capture(
                capture,
                iterations=5,
                repeats=2,
                deep_multiplier=4,
                base_seed=23,
                c_puct=2.0,
                manifest_sha256="b" * 64,
            )
        validate_root_evaluation(evaluated)
        self.assertEqual([schedule["schedule_id"] for schedule in evaluated["schedules"]], [0, 1])
        self.assertTrue(all(schedule["world_count"] == 1 for schedule in evaluated["schedules"]))
        first_seed = evaluated["schedules"][0]["worlds"][0]["treatments"]["S-B"][0]["seed"]
        second_seed = evaluated["schedules"][1]["worlds"][0]["treatments"]["S-B"][0]["seed"]
        self.assertNotEqual(first_seed, second_seed)
        self.assertEqual(
            set(evaluated["schedules"][0]["aggregate_treatments"]),
            {"U-B", "S-B", "S-4B"},
        )

    def test_replays_source_with_actual_deep_multiplier_label(self):
        with patch.dict(sys.modules, {"poke_engine": FakeEngine()}):
            evaluated = evaluate_bundle(
                source_bundle(),
                iterations=10,
                repeats=2,
                deep_multiplier=2,
                base_seed=7,
                c_puct=2.0,
                manifest_sha256="b" * 64,
            )
        validate_root_bundle(evaluated)
        self.assertEqual(set(evaluated["aggregate_treatments"]), {"U-B", "S-B", "S-2B"})
        self.assertEqual(evaluated["configuration"]["source_input_manifest_sha256"], "a" * 64)
        self.assertEqual(evaluated["configuration"]["input_manifest_sha256"], "b" * 64)
        self.assertEqual(evaluated["configuration"]["execution"], "offline")
        repeats = evaluated["worlds"][0]["capture"]["treatments"]["S-2B"]
        self.assertEqual([repeat["iterations"] for repeat in repeats], [20, 20])

    def test_file_output_is_private_exclusive_and_deterministic(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text(json.dumps(source_bundle(), sort_keys=True) + "\n", encoding="ascii")
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            kwargs = {
                "iterations": 5,
                "repeats": 1,
                "deep_multiplier": 4,
                "base_seed": 11,
                "c_puct": 2.0,
                "manifest_sha256": "c" * 64,
            }
            with patch.dict(sys.modules, {"poke_engine": FakeEngine()}):
                evaluate_file(source, first, **kwargs)
                evaluate_file(source, second, **kwargs)
                with self.assertRaisesRegex(RootBundleError, "already exists"):
                    evaluate_file(source, first, **kwargs)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o600)

    def test_rejects_missing_sampled_state(self):
        source = source_bundle()
        del source["worlds"][0]["capture"]["sampled_state"]
        from scripts.teacher_root_bundle import _canonical_json
        import hashlib

        unhashed = dict(source)
        unhashed.pop("bundle_sha256")
        source["bundle_sha256"] = hashlib.sha256(_canonical_json(unhashed).encode("ascii")).hexdigest()
        with patch.dict(sys.modules, {"poke_engine": FakeEngine()}):
            with self.assertRaisesRegex(RootBundleError, "sampled private state"):
                evaluate_bundle(
                    source,
                    iterations=5,
                    repeats=1,
                    deep_multiplier=4,
                    base_seed=1,
                    c_puct=2.0,
                    manifest_sha256="d" * 64,
                )


if __name__ == "__main__":
    unittest.main()
