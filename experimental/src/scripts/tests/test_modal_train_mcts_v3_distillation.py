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

from scripts.modal_train_mcts_v3_distillation import (  # noqa: E402
    V3_TRAIN_SOURCES,
    package_learner_trajectories,
    validate_v3_dataset,
)
from scripts.modal_train_mcts_distillation import package_train_sources  # noqa: E402

NUM_ACTIONS = 13


def make_record(**overrides):
    row = {
        "schema": 3,
        "text_tokens": [1, 2, 3],
        "numbers": [0.5, 0.25],
        "illegal_actions": [False, False] + [True] * (NUM_ACTIONS - 2),
        "visit_target_13": [0.75, 0.25] + [0.0] * (NUM_ACTIONS - 2),
    }
    row.update(overrides)
    return row


def to_jsonl(rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


class ValidateV3DatasetTests(unittest.TestCase):
    def test_accepts_valid_records(self):
        stats = validate_v3_dataset(to_jsonl([make_record(), make_record()]))
        self.assertEqual(stats, {"targets": 2, "text_len": 3, "numbers_len": 2})

    def test_rejects_wrong_schema(self):
        with self.assertRaisesRegex(ValueError, "unsupported schema"):
            validate_v3_dataset(to_jsonl([make_record(schema=2)]))

    def test_rejects_mass_on_illegal(self):
        illegal = [True, False] + [True] * (NUM_ACTIONS - 2)
        with self.assertRaisesRegex(ValueError, "illegal action"):
            validate_v3_dataset(to_jsonl([make_record(illegal_actions=illegal)]))

    def test_rejects_unnormalized_target(self):
        target = [0.5] + [0.0] * (NUM_ACTIONS - 1)
        with self.assertRaisesRegex(ValueError, "target mass"):
            validate_v3_dataset(to_jsonl([make_record(visit_target_13=target)]))

    def test_rejects_inconsistent_shapes(self):
        rows = [make_record(), make_record(text_tokens=[1, 2])]
        with self.assertRaisesRegex(ValueError, "inconsistent obs shape"):
            validate_v3_dataset(to_jsonl(rows))

    def test_rejects_empty(self):
        with self.assertRaisesRegex(ValueError, "no records"):
            validate_v3_dataset("")

    def test_smoke_dataset_passes(self):
        smoke = ROOT.parent / "experiments" / "v3_smoke2" / "v3_targets.jsonl"
        if not smoke.is_file():
            self.skipTest("smoke targets not present")
        stats = validate_v3_dataset(smoke.read_text())
        self.assertGreater(stats["targets"], 100)


class PackagingTests(unittest.TestCase):
    def test_train_sources_include_v3_module(self):
        self.assertIn("src/train/mcts_v3_distillation.py", V3_TRAIN_SOURCES)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in V3_TRAIN_SOURCES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative)
            payload = package_train_sources(root, sources=V3_TRAIN_SOURCES)
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                self.assertEqual(sorted(archive.getnames()), sorted(V3_TRAIN_SOURCES))

    def test_packages_learner_trajectories_under_format_root(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "learner_only"
            trajectory = root / "round_00" / "battle-gen9randombattle-1_a_vs_b_WIN.json.lz4"
            trajectory.parent.mkdir(parents=True)
            trajectory.write_bytes(b"compressed trajectory")
            payload, count = package_learner_trajectories(root)
            self.assertEqual(count, 1)
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                self.assertEqual(
                    archive.getnames(),
                    ["gen9randombattle/round_00/battle-gen9randombattle-1_a_vs_b_WIN.json.lz4"],
                )

    def test_rejects_empty_learner_root(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "learner_only"
            root.mkdir()
            with self.assertRaises(ValueError):
                package_learner_trajectories(root)


if __name__ == "__main__":
    unittest.main()
