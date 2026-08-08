from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from srcs.metagross import offline_validation


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OfflineValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidate = self._write("candidate.bin", b"candidate")
        self.baseline = self._write("baseline.bin", b"baseline")
        self.capture = self.root / "capture"
        self.canary_capture = self.root / "canary-capture"
        self.capture.mkdir()
        self.canary_capture.mkdir()
        candidate_sha256 = digest(self.candidate)
        (self.capture / "manifest.json").write_text(
            json.dumps({"policy": {"sha256": candidate_sha256}})
        )
        (self.canary_capture / "manifest.json").write_text(
            json.dumps(
                {"checkpoint": {"sha256_verified": candidate_sha256}}
            )
        )
        self.capture_digest = digest(self.capture / "manifest.json")
        self.canary_digest = digest(self.canary_capture / "manifest.json")

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, name: str, content: bytes) -> Path:
        path = self.root / name
        path.write_bytes(content)
        return path

    def _games(self, winner: str, *, incomplete: bool = False) -> dict:
        games = []
        for pair_index in range(1, 7):
            for pair_leg in (1, 2):
                games.append(
                    {
                        "game_index": len(games) + 1,
                        "pair_index": pair_index,
                        "pair_id": f"pair-{pair_index}",
                        "pair_leg": pair_leg,
                        "battle_seed": f"battle-{pair_index}",
                        "team_1_sha256": f"team-1-{pair_index}",
                        "team_2_sha256": f"team-2-{pair_index}",
                        "winner": winner,
                        "void": False,
                        "error": None,
                    }
                )
        if incomplete:
            games.pop()
            games.append(
                {
                    "game_index": 12,
                    "pair_index": 7,
                    "pair_id": "pair-7",
                    "pair_leg": 1,
                    "battle_seed": "battle-7",
                    "team_1_sha256": "team-1-7",
                    "team_2_sha256": "team-2-7",
                    "winner": winner,
                    "void": False,
                    "error": None,
                }
            )
        return {"artifacts": self._registered_artifacts(), "games": games}

    def _registered_artifacts(self) -> dict:
        return {
            "candidate": {"id": "candidate", "sha256": digest(self.candidate)},
            "baseline": {"id": "baseline", "sha256": digest(self.baseline)},
        }

    def _fixture(self, *, incomplete: bool = False, normal_games: int = 3) -> Path:
        as_a = self.root / "as-a.json"
        as_b = self.root / "as-b.json"
        as_a.write_text(json.dumps(self._games("agent_a", incomplete=incomplete)))
        as_b.write_text(json.dumps(self._games("agent_b")))
        shadow = self.root / "shadow.json"
        shadow.write_text(
            json.dumps(
                {
                    "mode": "captured_v5_holdout",
                    "capture": {"capture_digest": self.capture_digest},
                    "engine": {
                        "contract": offline_validation.ENGINE_CONTRACT,
                        "source_sha256": offline_validation.ENGINE_SOURCE_SHA256,
                        "native_sha256": "a" * 64,
                    },
                    "counts": {
                        "decisions": 12,
                        "evaluated_disagreements": 12,
                        "holdout_failures": 0,
                    },
                }
            )
        )
        canary = self.root / "canary.json"
        canary.write_text(
            json.dumps(
                {
                    "capture": {"digest": self.canary_digest},
                    "identity": {"policy_sha256": digest(self.candidate)},
                    "integrity": {"passed": True, "failures": []},
                    "result": {"normal_wins": normal_games, "normal_losses": 0},
                }
            )
        )
        prereg = {
            "schema_version": 1,
            "candidate": {
                "id": "candidate",
                "path": self.candidate.name,
                "sha256": digest(self.candidate),
            },
            "baseline": {
                "id": "baseline",
                "path": self.baseline.name,
                "sha256": digest(self.baseline),
            },
            "captures": {
                "offline": {"path": "capture", "digest": self.capture_digest},
                "canary": {"path": "canary-capture", "digest": self.canary_digest},
            },
            "counterbalanced_ab": {
                "candidate_as_agent_a": {
                    "path": as_a.name,
                    "sha256": digest(as_a),
                    "games": 12,
                    "artifacts": self._registered_artifacts(),
                },
                "candidate_as_agent_b": {
                    "path": as_b.name,
                    "sha256": digest(as_b),
                    "games": 12,
                    "artifacts": self._registered_artifacts(),
                },
                "games_per_orientation": 12,
            },
            "pathology": {
                "command": ["python", "-m", "unittest", "pathology.fixture"],
                "cwd": ".",
                "expected_tests": ["pathology.fixture"],
            },
            "shadow_replay": {
                "report": {"path": shadow.name, "sha256": digest(shadow)},
                "capture_id": "offline",
                "expected_counts": {"decisions": 12, "holdout_failures": 0},
            },
            "canary_audit": {
                "report": {"path": canary.name, "sha256": digest(canary)},
                "capture_id": "canary",
            },
            "confidence": {
                "method": "exact_sign",
                "alpha": 0.05,
                "seed": 17,
                "bootstrap_repeats": 200,
            },
            "gates": {
                "minimum_total_games": 24,
                "minimum_candidate_win_rate": 0.6,
                "maximum_model_slot_gap": 0.0,
                "minimum_lower_confidence_bound": 0.5,
                "maximum_p_value": 0.05,
                "minimum_normal_combat_games": 3,
            },
        }
        path = self.root / "preregistration.json"
        path.write_text(json.dumps(prereg))
        return path

    def _capture_loader(self, path: Path):
        capture_digest = digest(path / "manifest.json")
        manifest = json.loads((path / "manifest.json").read_text())
        return [], {}, {"capture_digest": capture_digest, "manifest": manifest}

    def _run(self, path: Path) -> dict:
        completed = subprocess.CompletedProcess([], 0, "pathology.fixture ... ok", "")
        with patch.object(
            offline_validation.shadow_replay,
            "load_capture",
            side_effect=self._capture_loader,
        ):
            return offline_validation.run_validation(
                path, command_runner=lambda *args, **kwargs: completed
            )

    def test_pass_and_pathology_failure(self):
        preregistration = self._fixture()
        summary = self._run(preregistration)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["inference"]["clusters"], 6)

        failed = subprocess.CompletedProcess([], 1, "", "failure")
        with patch.object(
            offline_validation.shadow_replay,
            "load_capture",
            side_effect=self._capture_loader,
        ):
            summary = offline_validation.run_validation(
                preregistration, command_runner=lambda *args, **kwargs: failed
            )
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["pathology_fixtures"])

    def test_rejects_incomplete_pairs(self):
        preregistration = self._fixture(incomplete=True)
        summary = self._run(preregistration)
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["matched_pair_inference"])
        self.assertIn("incomplete", " ".join(summary["failures"]))

    def test_rejects_all_forfeit_evaluation(self):
        preregistration = self._fixture()
        payload = json.loads(preregistration.read_text())
        for orientation in ("candidate_as_agent_a", "candidate_as_agent_b"):
            artifact = payload["counterbalanced_ab"][orientation]
            path = self.root / artifact["path"]
            games = json.loads(path.read_text())
            for game in games["games"]:
                game["end_reason"] = "forfeit"
            path.write_text(json.dumps(games))
            artifact["sha256"] = digest(path)
        preregistration.write_text(json.dumps(payload))

        summary = self._run(preregistration)
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["matched_pair_inference"])

    def test_rejects_canary_below_normal_combat_minimum(self):
        summary = self._run(self._fixture(normal_games=0))
        self.assertFalse(summary["checks"]["canary_audit_report"])

    def test_rejects_dry_run_shadow_report(self):
        preregistration = self._fixture()
        payload = json.loads(preregistration.read_text())
        artifact = payload["shadow_replay"]["report"]
        path = self.root / artifact["path"]
        report = json.loads(path.read_text())
        report["mode"] = "dry_run"
        path.write_text(json.dumps(report))
        artifact["sha256"] = digest(path)
        preregistration.write_text(json.dumps(payload))

        summary = self._run(preregistration)

        self.assertFalse(summary["checks"]["shadow_replay_report"])

    def test_detects_capture_manifest_tamper(self):
        preregistration = self._fixture()
        (self.capture / "manifest.json").write_text("tampered")
        summary = self._run(preregistration)
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["capture_offline"])

    def test_rejects_capture_for_wrong_candidate(self):
        preregistration = self._fixture()
        manifest = self.capture / "manifest.json"
        manifest.write_text(
            json.dumps({"policy": {"sha256": digest(self.baseline)}})
        )
        payload = json.loads(preregistration.read_text())
        payload["captures"]["offline"]["digest"] = digest(manifest)
        preregistration.write_text(json.dumps(payload))

        summary = self._run(preregistration)
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["capture_offline"])
        self.assertIn("candidate SHA-256", " ".join(summary["failures"]))

    def test_rejects_wrong_ab_baseline_identity(self):
        for orientation in ("candidate_as_agent_a", "candidate_as_agent_b"):
            with self.subTest(orientation=orientation):
                preregistration = self._fixture()
                payload = json.loads(preregistration.read_text())
                artifact = payload["counterbalanced_ab"][orientation]
                path = self.root / artifact["path"]
                envelope = json.loads(path.read_text())
                envelope["artifacts"]["baseline"]["sha256"] = digest(
                    self.candidate
                )
                path.write_text(json.dumps(envelope))
                artifact["sha256"] = digest(path)
                preregistration.write_text(json.dumps(payload))

                summary = self._run(preregistration)
                self.assertFalse(summary["passed"])
                self.assertFalse(summary["checks"][orientation])
                self.assertIn(
                    "artifact envelope identity mismatch",
                    " ".join(summary["failures"]),
                )

    def test_rejects_wrong_canary_identity(self):
        preregistration = self._fixture()
        payload = json.loads(preregistration.read_text())
        report = self.root / payload["canary_audit"]["report"]["path"]
        canary = json.loads(report.read_text())
        canary["identity"]["policy_sha256"] = digest(self.baseline)
        report.write_text(json.dumps(canary))
        payload["canary_audit"]["report"]["sha256"] = digest(report)
        preregistration.write_text(json.dumps(payload))

        summary = self._run(preregistration)
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["canary_audit_report"])
        self.assertIn("canary policy SHA-256", " ".join(summary["failures"]))

    def test_rejects_missing_pathology_marker(self):
        preregistration = self._fixture()
        completed = subprocess.CompletedProcess([], 0, "ok", "")
        with patch.object(
            offline_validation.shadow_replay,
            "load_capture",
            side_effect=self._capture_loader,
        ):
            summary = offline_validation.run_validation(
                preregistration, command_runner=lambda *args, **kwargs: completed
            )

        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["pathology_fixtures"])
        self.assertEqual(summary["pathology"]["demonstrated_tests"], [])

    def test_bootstrap_inference_is_deterministic(self):
        confidence = {
            "method": "cluster_bootstrap",
            "alpha": 0.05,
            "seed": 42,
            "bootstrap_repeats": 200,
        }
        scores = [1.0, 0.75, 0.5, 1.0]
        self.assertEqual(
            offline_validation.matched_pair_inference(scores, confidence),
            offline_validation.matched_pair_inference(scores, confidence),
        )

    def test_cli_status_and_template(self):
        preregistration = self._fixture(normal_games=0)
        output = self.root / "summary.json"
        completed = subprocess.CompletedProcess([], 0, "pathology.fixture ... ok", "")
        with (
            patch.object(
                offline_validation.shadow_replay,
                "load_capture",
                side_effect=self._capture_loader,
            ),
            patch.object(offline_validation.subprocess, "run", return_value=completed),
        ):
            self.assertEqual(
                offline_validation.main(
                    [
                        "--preregistration",
                        str(preregistration),
                        "--output",
                        str(output),
                    ]
                ),
                1,
            )
        self.assertFalse(json.loads(output.read_text())["passed"])

        template = self.root / "template.json"
        self.assertEqual(
            offline_validation.main(["--generate-template", str(template)]), 0
        )
        offline_validation.validate_preregistration(json.loads(template.read_text()))


if __name__ == "__main__":
    unittest.main()
