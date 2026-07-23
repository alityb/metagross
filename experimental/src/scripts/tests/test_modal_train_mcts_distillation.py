from __future__ import annotations

import io
import json
import sys
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.modal_train_mcts_distillation import (
    R1_RUN_NAME,
    TRAIN_SOURCES,
    package_r1_checkpoint_archive,
    package_strict_learner,
    package_train_sources,
    validate_sidecar_coverage,
)


class ModalTrainMCTSDistillationTests(unittest.TestCase):
    def test_packages_only_required_training_sources(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in TRAIN_SOURCES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative)

            with tarfile.open(fileobj=io.BytesIO(package_train_sources(root)), mode="r:gz") as archive:
                self.assertEqual(sorted(archive.getnames()), sorted(TRAIN_SOURCES))

    def test_requires_complete_targets_for_every_trajectory(self):
        trajectories = {
            "gen9randombattle/one.json.lz4": 2,
            "gen9randombattle/two.json.lz4": 1,
        }
        rows = [
            {"schema_version": 1, "trajectory": "gen9randombattle/one.json.lz4", "timestep": 0, "target": [1.0] + [0.0] * 12},
            {"schema_version": 1, "trajectory": "gen9randombattle/one.json.lz4", "timestep": 1, "target": [1.0] + [0.0] * 12},
            {"schema_version": 1, "trajectory": "gen9randombattle/two.json.lz4", "timestep": 0, "target": [1.0] + [0.0] * 12},
        ]
        sidecar = "".join(json.dumps(row) + "\n" for row in rows)

        self.assertEqual(validate_sidecar_coverage(trajectories, sidecar), {"trajectories": 2, "targets": 3})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_sidecar_coverage(trajectories, "".join(json.dumps(row) + "\n" for row in rows[:-1]))

    def test_packages_nested_finalizer_output_with_matching_sidecar_paths(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "learner_only"
            trajectory = root / "round_00" / "shard_01" / "battle-gen9randombattle-1_a_Unrated_a_vs_b_WIN.json.lz4"
            trajectory.parent.mkdir(parents=True)
            trajectory.write_bytes(b"compressed trajectory")
            source_name = trajectory.relative_to(root).as_posix()
            sidecar = root.parent / "verified.jsonl"
            sidecar.write_text(
                "".join(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "trajectory": source_name,
                            "timestep": timestep,
                            "target": [1.0] + [0.0] * 12,
                        }
                    )
                    + "\n"
                    for timestep in range(2)
                )
            )

            archive, packaged_sidecar, coverage = package_strict_learner(root, sidecar)

            packaged_name = f"gen9randombattle/{source_name}"
            self.assertEqual(coverage, {"trajectories": 1, "targets": 2})
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                self.assertEqual(tar.getnames(), [packaged_name])
            self.assertEqual(json.loads(packaged_sidecar.splitlines()[0])["trajectory"], packaged_name)

    def test_packages_actual_randbats_exit_r1_run_layout(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "randbats_exit_r1"
            checkpoint = root / R1_RUN_NAME / "ckpts" / "policy_weights" / "policy_epoch_5.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"epoch five")
            (root / "ckpts" / "policy_weights").mkdir(parents=True)
            (root / "ckpts" / "policy_weights" / "policy_epoch_5.pt").write_bytes(b"do not package this duplicate")

            archive = package_r1_checkpoint_archive(root)

            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                self.assertEqual(tar.getnames(), [f"{R1_RUN_NAME}/ckpts/policy_weights/policy_epoch_5.pt"])


if __name__ == "__main__":
    unittest.main()
