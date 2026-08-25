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

from scripts.modal_train_online_rl import (
    MIX_WEIGHTS,
    archive_trajectory_count,
    dataset_yaml,
    package_fresh_sources,
    package_trajectories,
    prepare_training_datasets,
    safe_extract_archive,
)


class OnlineRLTrainingTests(unittest.TestCase):
    def test_training_mix_is_guarded_and_normalized(self):
        self.assertEqual(MIX_WEIGHTS, {"legacy": 0.70, "fresh": 0.20, "human": 0.10})
        self.assertAlmostEqual(sum(MIX_WEIGHTS.values()), 1.0)
        config = dataset_yaml()
        self.assertIn("/data/online_rl/legacy", config)
        self.assertIn("/data/online_rl/fresh", config)
        self.assertIn("/data/online_rl/human", config)

    def test_packages_nested_collector_outputs_under_format_root(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory = root / "learner_vs_frozen" / "learner_acceptor" / "learner_trajectories" / "battle-1.json.lz4"
            trajectory.parent.mkdir(parents=True)
            trajectory.write_bytes(b"trajectory")
            payload, count = package_trajectories(root, "fresh")
            self.assertEqual(count, 1)
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                self.assertEqual(
                    archive.getnames(),
                    ["online_rl/fresh/gen9randombattle/learner_vs_frozen/learner_acceptor/learner_trajectories/battle-1.json.lz4"],
                )

            archive_path = root / "fresh.tar.gz"
            archive_path.write_bytes(payload)
            self.assertEqual(archive_trajectory_count(archive_path, "fresh"), 1)

    def test_rejects_empty_dataset(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "has no .lz4"):
                package_trajectories(Path(temporary), "fresh")

    def test_cumulative_fresh_sources_are_admitted_and_generation_prefixed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = []
            for generation, outcome in ((2, "WIN"), (3, "LOSS")):
                collection = root / f"collection-{generation}"
                trajectory = collection / "chunk" / f"battle_{outcome}.json.lz4"
                trajectory.parent.mkdir(parents=True)
                trajectory.write_bytes(b"trajectory")
                manifest = {
                    "collection_kind": "fresh", "failed_shards": 0,
                    "requested_battles": 1, "completed_battles": 1,
                    "learner_wins": int(outcome == "WIN"),
                    "learner_losses": int(outcome == "LOSS"),
                    "learner_trajectory_count": 1,
                }
                manifest["chunks"] = [{"requested_battles": 1, "phases": [{
                    key: manifest[key] for key in (
                        "completed_battles", "learner_wins", "learner_losses",
                        "learner_trajectory_count",
                    )
                }]}]
                manifest_path = collection / "MANIFEST.json"
                manifest_path.write_text(json.dumps(manifest))
                sources.append({"generation": generation, "root": str(collection.resolve()),
                    "manifest": str(manifest_path.resolve()), **{
                        key: manifest[key] for key in (
                            "requested_battles", "completed_battles", "learner_wins",
                            "learner_losses", "learner_trajectory_count",
                        )
                    }})
            totals = {key: sum(source[key] for source in sources) for key in (
                "requested_battles", "completed_battles", "learner_wins",
                "learner_losses", "learner_trajectory_count",
            )}
            source_manifest = root / "FRESH_SOURCES.json"
            source_manifest.write_text(json.dumps({
                "record_type": "online_rl_fresh_sources", "sources": sources, "totals": totals,
            }))
            payload, count, _ = package_fresh_sources(source_manifest)
            self.assertEqual(count, 2)
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                self.assertEqual(
                    archive.getnames(),
                    [
                        "online_rl/fresh/gen9randombattle/generation_002/chunk/battle_WIN.json.lz4",
                        "online_rl/fresh/gen9randombattle/generation_003/chunk/battle_LOSS.json.lz4",
                    ],
                )
            sources[1]["learner_trajectory_count"] = 2
            source_manifest.write_text(json.dumps({
                "record_type": "online_rl_fresh_sources", "sources": sources, "totals": totals,
            }))
            with self.assertRaisesRegex(ValueError, "totals do not match collection"):
                package_fresh_sources(source_manifest)

    def test_preflight_clears_roots_and_matches_all_admission_counts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_root = root / "artifacts"
            anchor_root = root / "anchors"
            data_root = root / "data" / "online_rl"
            artifact_root.mkdir()
            anchor_root.mkdir()
            for name, archive_root in (
                ("fresh", artifact_root),
                ("legacy", anchor_root),
                ("human", anchor_root),
            ):
                source = root / f"{name}-source"
                trajectory = source / f"battle-{name}.json.lz4"
                source.mkdir()
                trajectory.write_bytes(name.encode())
                payload, _ = package_trajectories(source, name)
                (archive_root / f"{name}.tar.gz").write_bytes(payload)

            stale = data_root / "legacy"
            stale.mkdir(parents=True)
            (stale / "index.csv").write_text("filename\nstale.json.lz4\n")
            (stale / "stale.json.lz4").write_bytes(b"stale")

            counts = prepare_training_datasets(artifact_root, anchor_root, data_root)

            self.assertEqual(
                counts,
                {
                    stage: {"fresh": 1, "legacy": 1, "human": 1}
                    for stage in ("packaged", "extracted", "rebuilt_index", "cached_index")
                },
            )
            self.assertFalse((stale / "stale.json.lz4").exists())

            loader_counts = prepare_training_datasets(
                artifact_root,
                anchor_root,
                data_root,
                index_counter=lambda _root, cached: 1,
            )
            self.assertEqual(loader_counts["loader_rebuilt"], {
                "fresh": 1, "legacy": 1, "human": 1,
            })
            self.assertEqual(loader_counts["loader_cached"], {
                "fresh": 1, "legacy": 1, "human": 1,
            })

            with self.assertRaisesRegex(ValueError, "loader example count mismatch"):
                prepare_training_datasets(
                    artifact_root,
                    anchor_root,
                    data_root,
                    index_counter=lambda _root, cached: 1 if not cached else 2,
                )

            (artifact_root / "fresh-shard.tar.gz").write_bytes(
                (artifact_root / "fresh.tar.gz").read_bytes()
            )
            with self.assertRaisesRegex(ValueError, "fresh trajectory admission count mismatch"):
                prepare_training_datasets(artifact_root, anchor_root, data_root)

    def test_safe_extraction_rejects_absolute_traversal_and_links(self):
        for member_name, member_type in (
            ("/absolute.json.lz4", tarfile.REGTYPE),
            ("../traversal.json.lz4", tarfile.REGTYPE),
            ("online_rl/fresh/link.json.lz4", tarfile.SYMTYPE),
            ("online_rl/fresh/hard-link.json.lz4", tarfile.LNKTYPE),
        ):
            with self.subTest(member_name=member_name), TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive_path = root / "unsafe.tar.gz"
                with tarfile.open(archive_path, mode="w:gz") as archive:
                    member = tarfile.TarInfo(member_name)
                    member.type = member_type
                    if member_type == tarfile.REGTYPE:
                        member.size = 1
                        archive.addfile(member, io.BytesIO(b"x"))
                    else:
                        member.linkname = "target"
                        archive.addfile(member)
                with self.assertRaisesRegex(ValueError, "unsafe|links"):
                    safe_extract_archive(archive_path, root / "out")


if __name__ == "__main__":
    unittest.main()
