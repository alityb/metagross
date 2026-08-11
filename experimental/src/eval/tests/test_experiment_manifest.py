from __future__ import annotations

import hashlib
import json
import math
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.experiment_manifest import (  # noqa: E402
    REDACTED,
    ManifestError,
    build_experiment_manifest,
    collect_git_identity,
    content_addressed_filename,
    finalize_manifest,
    hash_tree,
    validate_manifest,
    write_manifest,
)


CREATED = "2026-07-23T12:00:00Z"
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


def build_manifest(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "experiment_id": "exp-001",
        "run_id": "run-001",
        "model_configuration": {"checkpoint": "model.bin"},
        "engine_configuration": {"name": "poke-engine", "threads": 2},
        "search_configuration": {"iterations": 100},
        "belief_configuration": {"worlds": 8},
        "random_seeds": {"python": 1, "engine": "seed-2"},
        "resources": {"workers": 2, "gpu_count": 0},
        "metrics": [{"name": "win_rate"}],
        "gates": [{"metric": "win_rate", "minimum": 0.55}],
        "sample_plan": {"games": 100, "paired": True},
        "dependency_revisions": {"poke-engine": "0.0.47"},
        "argv": ["python", "-m", "eval.run", "--n-games", "100"],
        "environment_keys": ["OMP_NUM_THREADS"],
        "environ": {"OMP_NUM_THREADS": "2"},
        "git_identity": GIT,
        "host_identity": HOST,
        "created_at_utc": CREATED,
    }
    values.update(overrides)
    return build_experiment_manifest(**values)  # type: ignore[arg-type]


def run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def initialize_git_repo(repo: Path) -> None:
    run_git(repo, "init", "-q")
    tracked = repo / "tracked.txt"
    tracked.write_text("tracked\n", encoding="ascii")
    run_git(repo, "add", "tracked.txt")
    run_git(
        repo,
        "-c",
        "user.name=Manifest Test",
        "-c",
        "user.email=manifest@example.invalid",
        "commit",
        "-q",
        "-m",
        "initial",
    )


class ExperimentManifestTests(unittest.TestCase):
    def test_tree_hash_is_deterministic_and_path_sensitive(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "nested").mkdir()
            (root / "a.txt").write_text("one", encoding="ascii")
            (root / "nested" / "b.txt").write_text("two", encoding="ascii")
            first = hash_tree(root)
            second = hash_tree(root)
            self.assertEqual(first, second)
            self.assertEqual(first["scheme"], "framed-tree-sha256-v1")
            self.assertEqual(first["files"], 2)
            self.assertEqual(first["bytes"], 6)
            (root / "nested" / "b.txt").rename(root / "nested" / "c.txt")
            self.assertNotEqual(first["sha256"], hash_tree(root)["sha256"])

    def test_tree_hash_records_symlink_without_following_it(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "target.txt").write_text("target", encoding="ascii")
            try:
                (root / "link").symlink_to("target.txt")
            except OSError:
                self.skipTest("symlinks are unavailable")
            identity = hash_tree(root)
            self.assertEqual(identity["files"], 1)
            self.assertEqual(identity["symlinks"], 1)

    def test_hash_is_deterministic_for_identical_content(self):
        first = build_manifest()
        second = build_manifest(engine_configuration={"threads": 2, "name": "poke-engine"})

        self.assertEqual(first, second)
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(
            content_addressed_filename(first),
            f"experiment_input-{first['manifest_sha256']}.json",
        )

    def test_environment_and_argv_secrets_are_redacted(self):
        manifest = build_manifest(
            argv=["runner", "--api-token", "plain-token", "--label", "safe"],
            environment_keys=["API_TOKEN", "SAFE_SETTING"],
            environ={"API_TOKEN": "plain-token", "SAFE_SETTING": "safe"},
        )

        self.assertEqual(manifest["argv"], ["runner", "--api-token", REDACTED, "--label", "safe"])
        self.assertEqual(manifest["environment"], {"API_TOKEN": REDACTED, "SAFE_SETTING": "safe"})

    def test_common_secret_keys_and_authorization_headers_are_redacted(self):
        secret_keys = [
            "PGPASSWORD",
            "MYSQL_PWD",
            "SECRET_KEY_BASE",
            "AWS_SECRET_ACCESS_KEY",
            "HTTP_AUTHORIZATION",
        ]
        manifest = build_manifest(
            argv=[
                "runner",
                "--header",
                "Authorization: Bearer opaque",
                "-H=Authorization: Basic opaque",
                "-H",
                "Proxy-Authorization: Bearer opaque",
            ],
            environment_keys=secret_keys,
            environ={key: "opaque" for key in secret_keys},
        )

        self.assertEqual(manifest["environment"], {key: REDACTED for key in secret_keys})
        self.assertEqual(
            manifest["argv"],
            ["runner", "--header", REDACTED, "-H=" + REDACTED, "-H", REDACTED],
        )

    def test_secret_keys_and_obvious_values_are_rejected(self):
        with self.assertRaisesRegex(ManifestError, "secret-bearing key"):
            build_manifest(model_configuration={"password": "not-safe"})
        with self.assertRaisesRegex(ManifestError, "obvious secret value"):
            build_manifest(model_configuration={"endpoint": "Bearer abcdefghijklmnop"})
        for key in ("PGPASSWORD", "MYSQL_PWD", "SECRET_KEY_BASE", "AWS_SECRET_ACCESS_KEY"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ManifestError, "secret-bearing key"):
                    build_manifest(model_configuration={"nested": {key: "not-safe"}})

    def test_write_refuses_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = build_manifest()
            replacement = build_manifest(run_id="run-002")
            write_manifest(path, manifest)
            original = path.read_bytes()

            with self.assertRaisesRegex(ManifestError, "already exists"):
                write_manifest(path, replacement)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_write_is_private_and_overwrite_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            first = build_manifest()
            second = build_manifest(run_id="run-002")
            write_manifest(path, first)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            write_manifest(path, second, overwrite=True)

            self.assertEqual(json.loads(path.read_text(encoding="ascii")), second)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_required_fields_are_validated(self):
        manifest = build_manifest()
        del manifest["resources"]

        with self.assertRaisesRegex(ManifestError, "missing required fields: resources"):
            validate_manifest(manifest)

        manifest = build_manifest()
        manifest["schema_version"] = True
        with self.assertRaisesRegex(ManifestError, "schema_version"):
            validate_manifest(manifest)

    def test_nonfinite_values_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ManifestError, "non-finite"):
                    build_manifest(search_configuration={"cpuct": value})

    def test_artifacts_are_hashed_with_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.bin"
            artifact.write_bytes(b"model bytes")
            manifest = build_manifest(artifacts={"model": artifact})

        record = manifest["artifacts"]["model"]  # type: ignore[index]
        self.assertEqual(record["sha256"], hashlib.sha256(b"model bytes").hexdigest())
        self.assertEqual(record["size_bytes"], len(b"model bytes"))

    def test_artifact_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.bin"
            link = root / "model.bin"
            target.write_bytes(b"model bytes")
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaisesRegex(ManifestError, "non-symlink"):
                build_manifest(artifacts={"model": link})

    def test_dirty_hash_covers_untracked_mode_and_symlink_target(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            initialize_git_repo(repo)
            executable = repo / "untracked.sh"
            executable.write_text("#!/bin/sh\n", encoding="ascii")
            executable.chmod(0o600)
            non_executable_hash = collect_git_identity(repo)["dirty_diff_sha256"]
            executable.chmod(0o700)
            executable_hash = collect_git_identity(repo)["dirty_diff_sha256"]

            self.assertNotEqual(non_executable_hash, executable_hash)

            link = repo / "untracked-link"
            try:
                link.symlink_to("target-one")
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            first_target_hash = collect_git_identity(repo)["dirty_diff_sha256"]
            link.unlink()
            link.symlink_to("target-two")
            second_target_hash = collect_git_identity(repo)["dirty_diff_sha256"]

            self.assertNotEqual(first_target_hash, second_target_hash)
            self.assertTrue(collect_git_identity(repo)["dirty"])

    def test_dirty_hash_covers_initialized_submodule_worktree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child"
            parent = root / "parent"
            child.mkdir()
            parent.mkdir()
            initialize_git_repo(child)
            initialize_git_repo(parent)
            run_git(
                parent,
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                "-q",
                str(child),
                "deps/child",
            )
            run_git(
                parent,
                "-c",
                "user.name=Manifest Test",
                "-c",
                "user.email=manifest@example.invalid",
                "commit",
                "-q",
                "-m",
                "add submodule",
            )

            nested_untracked = parent / "deps/child/local.txt"
            nested_untracked.write_text("one\n", encoding="ascii")
            first_hash = collect_git_identity(parent)["dirty_diff_sha256"]
            nested_untracked.write_text("two\n", encoding="ascii")
            second_hash = collect_git_identity(parent)["dirty_diff_sha256"]

            self.assertNotEqual(first_hash, second_hash)

    def test_completion_links_frozen_input_without_mutating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen_path = root / "input-manifest.json"
            completion_path = root / "completion-manifest.json"
            result_path = root / "results.json"
            artifact_path = root / "replays.tar"
            result_path.write_bytes(b"result")
            artifact_path.write_bytes(b"artifact")
            frozen = build_manifest()
            write_manifest(frozen_path, frozen)
            frozen_bytes = frozen_path.read_bytes()

            completion = finalize_manifest(
                frozen_path,
                completion_path,
                results={"summary": result_path},
                artifacts={"replays": artifact_path},
                counts={"games": 100, "wins": 57},
                created_at_utc="2026-07-23T13:00:00Z",
            )

            self.assertEqual(completion["frozen_input_manifest_sha256"], frozen["manifest_sha256"])
            self.assertEqual(completion["counts"], {"games": 100, "wins": 57})
            self.assertIn("summary", completion["result_hashes"])
            self.assertIn("replays", completion["artifact_hashes"])
            self.assertEqual(frozen_path.read_bytes(), frozen_bytes)
            self.assertTrue(completion_path.is_file())


if __name__ == "__main__":
    unittest.main()
