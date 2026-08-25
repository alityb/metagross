from __future__ import annotations

import hashlib
import json
import math
import stat
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.experiment_manifest import build_experiment_manifest  # noqa: E402
from scripts.select_teacher_root_panel import (  # noqa: E402
    PanelSelectionError,
    _canonical_json,
    battle_turn_bin,
    derive_stratum,
    parse_args,
    ranking_sha256,
    select_captures,
    select_file,
)
from scripts.teacher_root_bundle import (  # noqa: E402
    RootCaptureConfig,
    build_root_capture,
    validate_root_capture,
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


def frozen_manifest() -> dict:
    return build_experiment_manifest(
        experiment_id="teacher-panel-test",
        run_id="run-1",
        model_configuration={"policy": "r1"},
        engine_configuration={"name": "poke-engine"},
        search_configuration={"duration_ms": 500},
        belief_configuration={"worlds": 1},
        random_seeds={"capture": 7},
        resources={"workers": 1},
        metrics=["root-panel"],
        gates=["development-only"],
        sample_plan={"roots": 8},
        argv=["select_teacher_root_panel.py"],
        environment_keys=[],
        environ={},
        git_identity=GIT,
        host_identity=HOST,
        created_at_utc="2026-07-23T00:00:00Z",
    )


def capture(
    manifest_hash: str,
    number: int,
    *,
    turn: int = 1,
    priors: list[tuple[str, float]] | None = None,
) -> dict:
    root_priors = priors or [("move-a", 0.75), ("move-b", 0.25)]
    probabilities = [0.0] * 13
    name_table = {}
    for index, (action, mass) in enumerate(root_priors):
        name_table[action] = index
        probabilities[index] = mass
    return build_root_capture(
        identity={
            "namespace": "private-worker",
            "battle_tag": f"private-battle-{number}",
            "username": "private-player",
            "decision_idx": number,
            "battle_turn": turn,
        },
        player_priors=root_priors,
        opponent_priors=[("opp-a", 1.0)],
        r1_policy_snapshot={
            "schema": 3, "tag": f"private-battle-{number}", "namespace": "private-worker",
            "username": "private-player", "decision_idx": number, "battle_turn": turn,
            "text_tokens": [1], "numbers": [0.0],
            "illegal_actions": [False] * len(root_priors) + [True] * (13 - len(root_priors)),
            "mask_fallback": False, "mask_fallback_error": None, "name_table": name_table,
            "probs": probabilities, "protocol_prefix": ["|request|{}"],
            "player_information_state": {"schema_version": 1, "universal_state": {}, "player_team": [], "opponent_public_team": []},
            "player_observation_history": {"any_opponent_asleep": False, "any_opponent_frozen": False, "revealed_opponents": ["opponent"]},
            "continuation_observation_history": {"any_opponent_asleep": False, "any_opponent_frozen": False, "revealed_opponents": ["opponent", "player"]},
        },
        schedules=[[(f"private-state-{number}", 1.0)]],
        config=RootCaptureConfig(Path("unused"), 1, 7, manifest_hash),
    )


def reseal(value: dict) -> None:
    unhashed = dict(value)
    unhashed.pop("capture_sha256", None)
    value["capture_sha256"] = hashlib.sha256(
        _canonical_json(unhashed).encode("ascii")
    ).hexdigest()


def write_inputs(directory: Path, captures: list[dict] | None = None):
    manifest = frozen_manifest()
    records = captures or [
        capture(manifest["manifest_sha256"], index) for index in range(4)
    ]
    manifest_path = directory / "manifest.json"
    input_path = directory / "captures.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return manifest, input_path, manifest_path, records


class SelectTeacherRootPanelTests(unittest.TestCase):
    def test_frozen_strata_cover_turn_entropy_action_count_and_tera(self):
        manifest = frozen_manifest()
        low = capture(
            manifest["manifest_sha256"],
            1,
            turn=6,
            priors=[("move-a", 0.99), ("move-a-tera", 0.01)],
        )
        high = capture(
            manifest["manifest_sha256"],
            2,
            turn=31,
            priors=[("a", 1 / 3), ("b", 1 / 3), ("c", 1 / 3)],
        )
        self.assertEqual(
            [battle_turn_bin(turn) for turn in (0, 5, 6, 10, 11, 20, 21, 30, 31)],
            ["0-5", "0-5", "6-10", "6-10", "11-20", "11-20", "21-30", "21-30", "31+"],
        )
        low_stratum = derive_stratum(low)
        self.assertEqual(
            low_stratum["recorded_player_prior_entropy_bin"], "below_threshold"
        )
        self.assertEqual(low_stratum["legal_action_count"], 2)
        self.assertTrue(low_stratum["tera_available"])
        self.assertEqual(low_stratum["battle_turn_bin"], "6-10")
        self.assertEqual(
            derive_stratum(high)["recorded_player_prior_entropy_bin"],
            "at_or_above_threshold",
        )
        self.assertEqual(
            derive_stratum(high, entropy_threshold=math.log(3) + 1e-12)[
                "recorded_player_prior_entropy_bin"
            ],
            "below_threshold",
        )

    def test_selection_is_input_order_independent_ranked_and_resealed(self):
        manifest = frozen_manifest()
        records = [capture(manifest["manifest_sha256"], index) for index in range(6)]
        selected, strata = select_captures(records, per_stratum=2, seed=19)
        reversed_selected, _ = select_captures(
            list(reversed(records)), per_stratum=2, seed=19
        )
        expected_hashes = sorted(
            (record["capture_sha256"] for record in records),
            key=lambda digest: (ranking_sha256(19, digest), digest),
        )[:2]
        self.assertEqual(strata, 1)
        self.assertEqual(
            [record["sampling"]["source_capture_sha256"] for record in selected],
            expected_hashes,
        )
        self.assertEqual(selected, reversed_selected)
        for record in selected:
            validate_root_capture(record)
            self.assertNotEqual(
                record["capture_sha256"], record["sampling"]["source_capture_sha256"]
            )
            self.assertEqual(record["sampling"]["population_count"], 6)
            self.assertEqual(record["sampling"]["selected_count"], 2)
            self.assertEqual(record["sampling"]["inclusion_probability"], 1 / 3)
            self.assertEqual(record["sampling"]["poststratification_weight"], 3.0)

    def test_selection_caps_each_composite_stratum(self):
        manifest = frozen_manifest()
        manifest_hash = manifest["manifest_sha256"]
        records = [capture(manifest_hash, index) for index in range(3)]
        records.extend(
            capture(manifest_hash, 10 + index, turn=12) for index in range(2)
        )
        selected, strata = select_captures(records, per_stratum=1, seed=0)
        self.assertEqual(strata, 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            sorted(record["sampling"]["population_count"] for record in selected),
            [2, 3],
        )

    def test_file_output_and_summary_are_private_hashed_and_identity_free(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, input_path, manifest_path, _ = write_inputs(directory)
            output_path = directory / "panel.jsonl"
            summary_path = directory / "summary.json"
            summary = select_file(
                input_path,
                manifest_path,
                output_path,
                summary_path,
                per_stratum=2,
                seed=11,
            )
            output_payload = output_path.read_bytes()
            stored_summary = json.loads(summary_path.read_text(encoding="ascii"))
            self.assertEqual(stored_summary, summary)
            self.assertEqual(
                summary["output_sha256"],
                hashlib.sha256(output_payload).hexdigest(),
            )
            self.assertEqual(
                summary["input_sha256"],
                hashlib.sha256(input_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(summary["counts"]["population_count"], 4)
            self.assertEqual(summary["counts"]["selected_count"], 2)
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(summary_path.stat().st_mode), 0o600)
            serialized_summary = _canonical_json(summary)
            for private_value in (
                "private-state",
                "private-battle",
                "private-player",
                "identity",
                "sampled_state",
            ):
                self.assertNotIn(private_value, serialized_summary)

    def test_manifest_linkage_and_capture_hash_are_validated(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest, _, manifest_path, _ = write_inputs(directory)
            wrong = capture("f" * 64, 1)
            input_path = directory / "wrong.jsonl"
            input_path.write_text(json.dumps(wrong) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(PanelSelectionError, "does not link"):
                select_file(
                    input_path,
                    manifest_path,
                    directory / "panel.jsonl",
                    directory / "summary.json",
                    per_stratum=1,
                    seed=0,
                )
            tampered = capture(manifest["manifest_sha256"], 2)
            tampered["identity"]["battle_turn"] = 99
            input_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(PanelSelectionError, "hash does not match"):
                select_file(
                    input_path,
                    manifest_path,
                    directory / "panel.jsonl",
                    directory / "summary.json",
                    per_stratum=1,
                    seed=0,
                )

    def test_rejects_duplicate_missing_turn_and_already_sampled_captures(self):
        manifest = frozen_manifest()
        source = capture(manifest["manifest_sha256"], 1)
        with self.assertRaisesRegex(PanelSelectionError, "duplicate capture"):
            select_captures([source, source], per_stratum=1, seed=0)
        missing_turn = capture(manifest["manifest_sha256"], 2)
        missing_turn["identity"].pop("battle_turn")
        reseal(missing_turn)
        with self.assertRaisesRegex(PanelSelectionError, "battle_turn"):
            derive_stratum(missing_turn)
        sampled = dict(source)
        sampled["sampling"] = {}
        reseal(sampled)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, input_path, manifest_path, _ = write_inputs(directory, [sampled])
            with self.assertRaisesRegex(PanelSelectionError, "already sampled"):
                select_file(
                    input_path,
                    manifest_path,
                    directory / "panel.jsonl",
                    directory / "summary.json",
                    per_stratum=1,
                    seed=0,
                )

    def test_collision_refusal_force_and_distinct_paths(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, input_path, manifest_path, _ = write_inputs(directory)
            output_path = directory / "panel.jsonl"
            summary_path = directory / "summary.json"
            output_path.write_text("do-not-replace", encoding="ascii")
            with self.assertRaisesRegex(PanelSelectionError, "already exists"):
                select_file(
                    input_path,
                    manifest_path,
                    output_path,
                    summary_path,
                    per_stratum=1,
                    seed=0,
                )
            self.assertEqual(output_path.read_text(), "do-not-replace")
            select_file(
                input_path,
                manifest_path,
                output_path,
                summary_path,
                per_stratum=1,
                seed=0,
                force=True,
            )
            self.assertTrue(output_path.read_text().endswith("\n"))
            with self.assertRaisesRegex(PanelSelectionError, "paths must be distinct"):
                select_file(
                    input_path,
                    manifest_path,
                    input_path,
                    summary_path,
                    per_stratum=1,
                    seed=0,
                    force=True,
                )

    def test_invalid_parameters_and_cli_contract(self):
        manifest = frozen_manifest()
        source = capture(manifest["manifest_sha256"], 1)
        for bad_limit in (0, -1, True):
            with self.assertRaisesRegex(PanelSelectionError, "positive integer"):
                select_captures([source], per_stratum=bad_limit, seed=0)
        for bad_seed in (-1, 2**64, True):
            with self.assertRaisesRegex(PanelSelectionError, "unsigned 64-bit"):
                select_captures([source], per_stratum=1, seed=bad_seed)
        with self.assertRaisesRegex(PanelSelectionError, "finite and nonnegative"):
            derive_stratum(source, entropy_threshold=float("nan"))
        args = parse_args(
            [
                "--input",
                "in.jsonl",
                "--input-manifest",
                "manifest.json",
                "--output",
                "out.jsonl",
                "--summary",
                "summary.json",
                "--per-stratum",
                "3",
                "--seed",
                "17",
                "--force",
            ]
        )
        self.assertEqual(args.per_stratum, 3)
        self.assertEqual(args.seed, 17)
        self.assertEqual(args.entropy_threshold, 1.0)
        self.assertTrue(args.force)


if __name__ == "__main__":
    unittest.main()
