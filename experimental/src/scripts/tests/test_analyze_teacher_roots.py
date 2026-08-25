from __future__ import annotations

import hashlib
import json
import math
import stat
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.experiment_manifest import build_experiment_manifest  # noqa: E402
from scripts.analyze_teacher_roots import (  # noqa: E402
    AnalysisError,
    analyze,
    distribution_metrics,
    write_outputs,
)
from scripts.evaluate_teacher_root_bundles import evaluate_bundle, evaluate_capture  # noqa: E402
from scripts.teacher_root_bundle import RootCaptureConfig, build_root_capture  # noqa: E402
from scripts.tests.test_evaluate_teacher_root_bundles import (  # noqa: E402
    FakeEngine,
    source_bundle,
)


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


def manifest() -> dict:
    return build_experiment_manifest(
        experiment_id="teacher-analysis-test",
        run_id="run-1",
        model_configuration={"policy": "r1"},
        engine_configuration={"name": "poke-engine"},
        search_configuration={"iterations": 10},
        belief_configuration={"worlds": 1},
        random_seeds={"teacher": 7},
        resources={"workers": 1},
        metrics=["stability"],
        gates=["descriptive-only"],
        sample_plan={"roots": 1},
        argv=["analyze_teacher_roots.py"],
        environment_keys=[],
        environ={},
        git_identity=GIT,
        host_identity=HOST,
        created_at_utc="2026-07-24T00:00:00Z",
    )


def evaluated_bundle(manifest_sha256: str) -> dict:
    with patch.dict(sys.modules, {"poke_engine": FakeEngine()}):
        return evaluate_bundle(
            source_bundle(),
            iterations=10,
            repeats=2,
            deep_multiplier=4,
            base_seed=7,
            c_puct=2.0,
            manifest_sha256=manifest_sha256,
        )


class AnalyzeTeacherRootsTests(unittest.TestCase):
    def test_schedule_variance_is_descriptively_estimable(self):
        frozen = manifest()
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
            evaluation = evaluate_capture(
                capture,
                iterations=5,
                repeats=2,
                deep_multiplier=4,
                base_seed=23,
                c_puct=2.0,
                manifest_sha256=frozen["manifest_sha256"],
            )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "evaluation.jsonl"
            manifest_path = root / "manifest.json"
            input_path.write_text(json.dumps(evaluation) + "\n", encoding="ascii")
            manifest_path.write_text(json.dumps(frozen), encoding="ascii")
            report = analyze(input_path, manifest_path)
        self.assertEqual(report["counts"]["determinization_schedules"], 2)
        self.assertEqual(report["counts"]["schedule_pairs"], {"S-4B": 1, "S-B": 1, "U-B": 1})
        self.assertEqual(
            report["variance_components"]["determinization_schedule_variance"],
            "descriptively_estimable",
        )

    def test_distribution_metrics_are_symmetric_and_tie_aware(self):
        left = {"a": 0.5, "b": 0.5, "c": 0.0}
        right = {"a": 0.0, "b": 0.5, "c": 0.5}
        forward = distribution_metrics(left, right)
        reverse = distribution_metrics(right, left)
        self.assertAlmostEqual(forward["jensen_shannon_nats"], reverse["jensen_shannon_nats"])
        self.assertAlmostEqual(forward["total_variation"], 0.5)
        self.assertEqual(forward["top1_fractional_agreement"], 0.25)
        self.assertEqual(forward["top_set_overlap"], 1.0)
        self.assertGreaterEqual(forward["spearman_rank_correlation"], -1.0)
        self.assertLessEqual(forward["spearman_rank_correlation"], 1.0)

    def test_analysis_is_private_descriptive_and_manifest_linked(self):
        frozen = manifest()
        bundle = evaluated_bundle(frozen["manifest_sha256"])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "roots.jsonl"
            manifest_path = root / "input-manifest.json"
            input_path.write_text(json.dumps(bundle) + "\n", encoding="ascii")
            manifest_path.write_text(json.dumps(frozen) + "\n", encoding="ascii")
            report = analyze(input_path, manifest_path)
        self.assertEqual(report["claim_status"], "descriptive_only")
        self.assertEqual(report["counts"]["roots"], 1)
        self.assertEqual(report["counts"]["sampled_worlds"], 1)
        self.assertEqual(report["counts"]["repeat_pairs"], {"S-4B": 1, "S-B": 1, "U-B": 1})
        self.assertEqual(report["repeat_stability"]["S-B"]["metrics"]["jensen_shannon_nats"]["median"], 0.0)
        self.assertEqual(report["gates"]["teacher_qualification"]["status"], "not_evaluable")
        self.assertNotIn("battle-1", json.dumps(report))
        self.assertNotIn("learner", json.dumps(report))
        self.assertNotIn('"sampled_state"', json.dumps(report))

    def test_manifest_mismatch_duplicate_and_tamper_fail_closed(self):
        frozen = manifest()
        bundle = evaluated_bundle(frozen["manifest_sha256"])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(frozen), encoding="ascii")
            duplicate = root / "duplicate.jsonl"
            duplicate.write_text(json.dumps(bundle) + "\n" + json.dumps(bundle) + "\n", encoding="ascii")
            with self.assertRaisesRegex(AnalysisError, "duplicate root"):
                analyze(duplicate, manifest_path)
            tampered = dict(bundle)
            tampered["world_count"] = 9
            tamper_path = root / "tampered.jsonl"
            tamper_path.write_text(json.dumps(tampered) + "\n", encoding="ascii")
            with self.assertRaisesRegex(AnalysisError, "hash does not match"):
                analyze(tamper_path, manifest_path)
            other = manifest()
            other["run_id"] = "changed"
            from eval.experiment_manifest import _seal

            _seal(other)
            other_path = root / "other.json"
            other_path.write_text(json.dumps(other), encoding="ascii")
            valid_path = root / "valid.jsonl"
            valid_path.write_text(json.dumps(bundle) + "\n", encoding="ascii")
            with self.assertRaisesRegex(AnalysisError, "does not link"):
                analyze(valid_path, other_path)

    def test_recomputed_aggregate_must_match_world_policies(self):
        frozen = manifest()
        bundle = evaluated_bundle(frozen["manifest_sha256"])
        policy = bundle["aggregate_treatments"]["S-B"][0]["side_one_policy"]
        policy[0]["probability"] = 0.5
        policy[1]["probability"] = 0.5
        for entry in policy[2:]:
            entry["probability"] = 0.0
        from scripts.analyze_teacher_roots import _canonical_json

        unhashed = dict(bundle)
        unhashed.pop("bundle_sha256")
        bundle["bundle_sha256"] = hashlib.sha256(
            _canonical_json(unhashed).encode("ascii")
        ).hexdigest()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "roots.jsonl"
            manifest_path = root / "manifest.json"
            input_path.write_text(json.dumps(bundle) + "\n", encoding="ascii")
            manifest_path.write_text(json.dumps(frozen), encoding="ascii")
            with self.assertRaisesRegex(AnalysisError, "does not match its worlds"):
                analyze(input_path, manifest_path)

    def test_outputs_are_private_deterministic_and_exclusive(self):
        frozen = manifest()
        bundle = evaluated_bundle(frozen["manifest_sha256"])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "roots.jsonl"
            manifest_path = root / "manifest.json"
            input_path.write_text(json.dumps(bundle) + "\n", encoding="ascii")
            manifest_path.write_text(json.dumps(frozen), encoding="ascii")
            report = analyze(input_path, manifest_path)
            output_json = root / "analysis.json"
            output_markdown = root / "analysis.md"
            write_outputs(report, output_json, output_markdown, force=False)
            first = output_json.read_bytes()
            self.assertEqual(stat.S_IMODE(output_json.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(output_markdown.stat().st_mode), 0o600)
            with self.assertRaisesRegex(AnalysisError, "already exists"):
                write_outputs(report, output_json, output_markdown, force=False)
            write_outputs(report, output_json, output_markdown, force=True)
            self.assertEqual(output_json.read_bytes(), first)
            self.assertTrue(math.isfinite(report["matched_distribution_comparisons"]["P_vs_S-B"]["metrics"]["total_variation"]["mean"]))


if __name__ == "__main__":
    unittest.main()
