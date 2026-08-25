from __future__ import annotations

import hashlib
import json
import math
import stat
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.experiment_manifest import build_experiment_manifest  # noqa: E402
from scripts.calibrate_teacher_iterations import (  # noqa: E402
    CalibrationError,
    _canonical_json,
    _run_duration_world,
    calibrate,
    calibrate_file,
    parse_args,
    summarize_iterations,
    validate_report,
)
from scripts.teacher_root_bundle import build_root_bundle  # noqa: E402


GIT = {"commit": "a" * 40, "dirty": True, "dirty_diff_sha256": "b" * 64}
HOST = {
    "hostname": "test-host",
    "platform": "test-platform",
    "system": "TestOS",
    "release": "1",
    "machine": "test-machine",
    "python_implementation": "CPython",
    "python_version": "3.test",
    "python_executable": "/test/python",
}


def frozen_manifest() -> dict:
    return build_experiment_manifest(
        experiment_id="teacher-calibration-test",
        run_id="run-1",
        model_configuration={"policy": "r1"},
        engine_configuration={"name": "poke-engine"},
        search_configuration={"duration_ms": 500},
        belief_configuration={"worlds": 2},
        random_seeds={"capture": 7},
        resources={"workers": 8},
        metrics=["iterations"],
        gates=["descriptive"],
        sample_plan={"roots": 2},
        argv=["calibrate_teacher_iterations.py"],
        environment_keys=[],
        environ={},
        git_identity=GIT,
        host_identity=HOST,
        created_at_utc="2026-07-23T00:00:00Z",
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


def root_bundle(manifest_hash: str, root_number: int, *, worlds: int = 2) -> dict:
    results = []
    for world_index in range(worlds):
        state = f"private-state-{root_number}-{world_index}"
        live = Result([Option("move-a", 1), Option("move-b", 0)], [Option("opp-a", 1)], 1)
        live._teacher_root_capture = {
            "identity": {
                "namespace": "worker",
                "battle_tag": f"private-battle-{root_number}",
                "username": "private-user",
                "decision_idx": root_number,
                "battle_turn": 1,
            },
            "configuration": {
                "iterations": 1,
                "repeats": 1,
                "deep_multiplier": 1,
                "base_seed": 7,
                "c_puct": 2.0,
                "input_manifest_sha256": manifest_hash,
                "primary_side_two_treatment": "equal_legal_priors",
                "threads": 1,
            },
            "world_index": world_index,
            "world": {
                "world_index": world_index,
                "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
                "sampled_state": state,
                "live_result": {
                    "total_visits": 1,
                    "side_one": [
                        {"action": "move-a", "visits": 1, "total_score": 0.0},
                        {"action": "move-b", "visits": 0, "total_score": 0.0},
                    ],
                    "side_two": [{"action": "opp-a", "visits": 1, "total_score": 0.0}],
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
                            "seed": root_number * 10 + world_index,
                            "iterations": 1,
                            "result": {
                                "total_visits": 1,
                                "side_one": [
                                    {"action": "move-a", "visits": 1, "total_score": 0.0},
                                    {"action": "move-b", "visits": 0, "total_score": 0.0},
                                ],
                                "side_two": [
                                    {"action": "opp-a", "visits": 1, "total_score": 0.0}
                                ],
                            },
                        }
                    ]
                },
            },
        }
        results.append((live, 1.0 / worlds, world_index))
    return build_root_bundle(results)


class FakeState:
    @classmethod
    def from_string(cls, value):
        return value


class FakeEngine:
    State = FakeState

    def __init__(self):
        self.calls = []

    def monte_carlo_tree_search(self, state, **kwargs):
        self.calls.append((state, kwargs))
        if "seed" in kwargs:
            return Result([], [], kwargs["iterations"])
        world_index = int(state.rsplit("-", 1)[1])
        return Result([], [], 100 + 100 * world_index)


class FakeExecutor:
    instances = []

    def __init__(self, *, max_workers):
        self.max_workers = max_workers
        self.payloads = []
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def map(self, function, payloads):
        self.payloads = list(payloads)
        return [function(payload) for payload in self.payloads]


class StepClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.25
        return self.value


def write_inputs(directory: Path, *, root_count: int = 2, worlds: int = 2):
    manifest = frozen_manifest()
    manifest_path = directory / "manifest.json"
    input_path = directory / "roots.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    bundles = [root_bundle(manifest["manifest_sha256"], index, worlds=worlds) for index in range(root_count)]
    input_path.write_text("".join(json.dumps(bundle) + "\n" for bundle in bundles), encoding="ascii")
    return input_path, manifest_path, bundles


def reseal_bundle(bundle: dict) -> None:
    unhashed = dict(bundle)
    unhashed.pop("bundle_sha256", None)
    bundle["bundle_sha256"] = hashlib.sha256(_canonical_json(unhashed).encode("ascii")).hexdigest()


class CalibrateTeacherIterationsTests(unittest.TestCase):
    def setUp(self):
        FakeExecutor.instances = []

    def test_statistics_use_linear_percentiles_population_cv_and_half_up_rounding(self):
        summary = summarize_iterations([100, 200, 300, 400], round_to=100)
        self.assertEqual(summary["mean"], 250.0)
        self.assertEqual(summary["median"], 250.0)
        self.assertEqual(summary["p10"], 130.0)
        self.assertEqual(summary["p25"], 175.0)
        self.assertEqual(summary["p75"], 325.0)
        self.assertEqual(summary["p90"], 370.0)
        self.assertAlmostEqual(summary["cv"], math.sqrt(12500) / 250)
        self.assertEqual(summary["recommendation"], 300)
        self.assertEqual(summarize_iterations([150], round_to=100)["recommendation"], 200)
        self.assertEqual(
            summarize_iterations([100, 100, 100, 1000], round_to=100)["recommendation"],
            100,
        )

    def test_strictly_rejects_self_hash_state_hash_and_manifest_link_tampering(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path, manifest_path, bundles = write_inputs(directory, root_count=1)
            tampered = dict(bundles[0])
            tampered["world_count"] = 99
            input_path.write_text(json.dumps(tampered) + "\n", encoding="ascii")
            with self.assertRaisesRegex(CalibrationError, "hash does not match"):
                calibrate(input_path, manifest_path, duration_ms=10, executor_factory=FakeExecutor)

            wrong_state = root_bundle(frozen_manifest()["manifest_sha256"], 0)
            wrong_state["worlds"][0]["capture"]["state_sha256"] = "0" * 64
            reseal_bundle(wrong_state)
            input_path.write_text(json.dumps(wrong_state) + "\n", encoding="ascii")
            with self.assertRaisesRegex(CalibrationError, "state hash does not match"):
                calibrate(input_path, manifest_path, duration_ms=10, executor_factory=FakeExecutor)

            wrong_manifest = root_bundle("f" * 64, 0)
            input_path.write_text(json.dumps(wrong_manifest) + "\n", encoding="ascii")
            with self.assertRaisesRegex(CalibrationError, "does not link"):
                calibrate(input_path, manifest_path, duration_ms=10, executor_factory=FakeExecutor)

    def test_worker_uses_duration_mode_one_thread_treatment_priors_and_no_seed(self):
        engine = FakeEngine()
        base = {
            "state": "private-state-0-0",
            "state_sha256": "a" * 64,
            "duration_ms": 500,
            "c_puct": 2.0,
            "recorded_player_priors": [("move-a", 0.75), ("move-b", 0.25)],
            "recorded_opponent_priors": [("opp-a", 1.0)],
            "effective_player_priors": [("move-a", 0.75), ("move-b", 0.25)],
            "equal_side_one_priors": [("move-a", 0.5), ("move-b", 0.5)],
            "equal_side_two_priors": [("opp-a", 1.0)],
        }
        with patch.dict(sys.modules, {"poke_engine": engine}):
            for treatment in ("accepted-live", "S-B", "U-B"):
                _run_duration_world({**base, "treatment": treatment})
        for _, kwargs in engine.calls:
            self.assertEqual(kwargs["duration_ms"], 500)
            self.assertEqual(kwargs["iterations"], 0)
            self.assertEqual(kwargs["threads"], 1)
            self.assertNotIn("seed", kwargs)
        self.assertEqual(engine.calls[0][1]["s2_priors"], base["recorded_opponent_priors"])
        self.assertEqual(engine.calls[1][1]["s1_priors"], base["effective_player_priors"])
        self.assertEqual(engine.calls[2][1]["s1_priors"], base["equal_side_one_priors"])

    def test_accepted_live_allows_schema_v1_unnormalized_recorded_masses(self):
        engine = FakeEngine()
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path, manifest_path, bundles = write_inputs(directory, root_count=1, worlds=1)
            capture = bundles[0]["worlds"][0]["capture"]
            capture["recorded_player_priors"] = [["move-a", 3.0], ["move-b", 1.0]]
            capture["recorded_opponent_priors"] = [["opp-a", 2.0]]
            reseal_bundle(bundles[0])
            input_path.write_text(json.dumps(bundles[0]) + "\n", encoding="ascii")
            with patch.dict(sys.modules, {"poke_engine": engine}):
                calibrate(
                    input_path,
                    manifest_path,
                    duration_ms=500,
                    repeats=1,
                    executor_factory=FakeExecutor,
                    clock=StepClock(),
                )
        duration_kwargs = engine.calls[1][1]
        self.assertEqual(duration_kwargs["s1_priors"], [("move-a", 3.0), ("move-b", 1.0)])
        self.assertEqual(duration_kwargs["s2_priors"], [("opp-a", 2.0)])

    def test_uses_one_executor_per_root_repeat_and_root_balanced_world_means(self):
        engine = FakeEngine()
        with TemporaryDirectory() as temporary:
            input_path, manifest_path, _ = write_inputs(Path(temporary), root_count=2, worlds=2)
            with patch.dict(sys.modules, {"poke_engine": engine}):
                report = calibrate(
                    input_path,
                    manifest_path,
                    duration_ms=500,
                    repeats=2,
                    parallelism=8,
                    round_to=100,
                    executor_factory=FakeExecutor,
                    clock=StepClock(),
                )
        self.assertEqual(len(FakeExecutor.instances), 4)
        self.assertTrue(all(instance.max_workers == 8 for instance in FakeExecutor.instances))
        self.assertEqual(report["counts"]["executor_instances"], 4)
        self.assertEqual(report["counts"]["world_searches"], 8)
        self.assertEqual(report["counts"]["executor_waves"], 4)
        self.assertEqual(report["statistics"]["mean"], 150.0)
        self.assertEqual(report["statistics"]["recommendation"], 200)
        preflight_kwargs = engine.calls[0][1]
        self.assertEqual(preflight_kwargs["iterations"], 1)
        self.assertEqual(preflight_kwargs["seed"], 0)
        self.assertTrue(all("seed" not in kwargs for _, kwargs in engine.calls[1:]))

    def test_report_is_private_duration_mode_self_hashed_0600_and_exclusive(self):
        engine = FakeEngine()
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path, manifest_path, _ = write_inputs(directory, root_count=1, worlds=1)
            output_path = directory / "calibration.json"
            with patch.dict(sys.modules, {"poke_engine": engine}):
                report = calibrate_file(
                    input_path,
                    manifest_path,
                    output_path,
                    duration_ms=500,
                    repeats=1,
                    executor_factory=FakeExecutor,
                    clock=StepClock(),
                )
                with self.assertRaisesRegex(CalibrationError, "already exists"):
                    calibrate_file(
                        input_path,
                        manifest_path,
                        output_path,
                        duration_ms=500,
                        repeats=1,
                        executor_factory=FakeExecutor,
                    )
            stored = json.loads(output_path.read_text(encoding="ascii"))
            validate_report(stored)
            self.assertEqual(stored, report)
            self.assertEqual(stored["calibration_mode"], "duration_to_exact_iterations")
            self.assertIsNone(stored["configuration"]["seed"])
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            serialized = json.dumps(stored)
            self.assertNotIn("private-state", serialized)
            self.assertNotIn("private-battle", serialized)
            self.assertNotIn("private-user", serialized)
            self.assertIn(hashlib.sha256(b"private-state-0-0").hexdigest(), serialized)

    def test_cli_defaults_to_eight_workers_and_accepted_live(self):
        args = parse_args(
            [
                "--input",
                "roots.jsonl",
                "--input-manifest",
                "manifest.json",
                "--output",
                "calibration.json",
                "--duration-ms",
                "500",
            ]
        )
        self.assertEqual(args.parallelism, 8)
        self.assertEqual(args.treatment, "accepted-live")
        self.assertEqual(args.round_to, 1000)


if __name__ == "__main__":
    unittest.main()
