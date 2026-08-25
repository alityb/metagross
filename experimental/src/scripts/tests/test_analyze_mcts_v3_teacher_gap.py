from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import analyze_mcts_v3_teacher_gap as census  # noqa: E402
from scripts.analyze_mcts_v3_teacher_gap import (  # noqa: E402
    AnalysisError,
    InputSpec,
    analyze,
    calculate_row_metrics,
    cluster_bootstrap,
    load_dataset,
    validate_output_paths,
)

POLICY_SHA = "a" * 64


def record(
    *,
    tag: str = "battle-1",
    decision_idx: int = 0,
    parent: list[float] | None = None,
    target: list[float] | None = None,
    illegal: list[bool] | None = None,
    selected: int = 0,
    turn: int = 1,
    label: int | None = 1,
) -> dict:
    return {
        "schema": 3,
        "battle_tag": tag,
        "username": "learner",
        "decision_idx": decision_idx,
        "battle_turn": turn,
        "illegal_actions": illegal if illegal is not None else [False, False] + [True] * 11,
        "policy_probs": parent if parent is not None else [0.75, 0.25] + [0.0] * 11,
        "visit_target_13": target if target is not None else [0.5, 0.5] + [0.0] * 11,
        "selected_action_index": selected,
        "label": label,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def spec(path: Path, name: str = "round1", identity: str = "r1@epoch5", reports=()):
    return InputSpec(name, identity, POLICY_SHA, path, tuple(reports))


class AnalyzeMCTSV3TeacherGapTests(unittest.TestCase):
    def test_valid_metrics_counts_and_strata(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "round1.jsonl"
            write_jsonl(path, [record()])
            result = analyze([spec(path)], seed=4, repeats=20)
            dataset = result["datasets"][0]
            metrics = dataset["metrics"]
            self.assertEqual(metrics["decision_count"], 1)
            self.assertEqual(metrics["battle_count"], 1)
            self.assertEqual(metrics["top1_fractional_agreement"], 0.5)
            self.assertEqual(metrics["top1_set_overlap"], 1.0)
            self.assertAlmostEqual(metrics["total_variation"], 0.25)
            self.assertAlmostEqual(metrics["search_entropy_nats"], math.log(2.0))
            self.assertEqual(metrics["search_top_mass"], 0.5)
            self.assertEqual(metrics["search_top_margin"], 0.0)
            self.assertEqual(metrics["parent_probability_on_search_top_action"], 0.5)
            self.assertEqual(metrics["parent_probability_on_selected_action"], 0.75)
            self.assertIn("0-5", dataset["strata"]["battle_turn_bin"])
            self.assertIn("2", dataset["strata"]["legal_action_count"])
            self.assertIn("normal_move", dataset["strata"]["selected_action_kind"])
            self.assertIn("1", dataset["strata"]["label"])

    def test_streams_and_calculates_each_row_once_without_retaining_arrays(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            write_jsonl(path, [record(decision_idx=0), record(decision_idx=1)])
            original = census.calculate_row_metrics
            with patch.object(Path, "read_text", side_effect=AssertionError("not streaming")):
                with patch.object(census, "calculate_row_metrics", wraps=original) as calculate:
                    summary = load_dataset(spec(path))
            self.assertEqual(calculate.call_count, 2)
            self.assertEqual(summary.aggregate.decision_count, 2)
            self.assertFalse(hasattr(summary, "records"))
            self.assertFalse(hasattr(summary, "policy_probs"))

    def test_boundary_normalization_is_applied(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            parent = [0.50004, 0.50004] + [0.0] * 11
            target = [0.49996, 0.49996] + [0.0] * 11
            write_jsonl(path, [record(parent=parent, target=target)])
            metrics = analyze([spec(path)], repeats=5)["datasets"][0]["metrics"]
            self.assertAlmostEqual(metrics["parent_entropy_nats"], math.log(2.0))
            self.assertAlmostEqual(metrics["search_entropy_nats"], math.log(2.0))
            rejected = Path(temporary) / "rejected.jsonl"
            write_jsonl(rejected, [record(parent=[0.50006, 0.50006] + [0.0] * 11)])
            with self.assertRaisesRegex(AnalysisError, "expected 1"):
                load_dataset(spec(rejected))

    def test_kl_and_js_properties(self):
        illegal = [False, False] + [True] * 11
        left = [1.0, 0.0] + [0.0] * 11
        right = [0.0, 1.0] + [0.0] * 11
        same = dict(zip(census.METRIC_NAMES, calculate_row_metrics(left, left, illegal, 0)))
        forward = dict(zip(census.METRIC_NAMES, calculate_row_metrics(left, right, illegal, 1)))
        reverse = dict(zip(census.METRIC_NAMES, calculate_row_metrics(right, left, illegal, 0)))
        self.assertAlmostEqual(same["kl_target_parent_nats"], 0.0)
        self.assertAlmostEqual(same["jensen_shannon_nats"], 0.0)
        self.assertGreaterEqual(forward["kl_target_parent_nats"], 0.0)
        self.assertTrue(math.isfinite(forward["cross_entropy_nats"]))
        self.assertTrue(math.isfinite(forward["kl_target_parent_nats"]))
        self.assertAlmostEqual(
            forward["jensen_shannon_nats"], reverse["jensen_shannon_nats"]
        )
        self.assertLessEqual(forward["jensen_shannon_nats"], math.log(2.0) + 1e-15)

    def test_ties_have_fractional_semantics_and_rates(self):
        illegal = [False, False, False] + [True] * 10
        parent = [0.5, 0.5, 0.0] + [0.0] * 10
        target = [0.0, 0.5, 0.5] + [0.0] * 10
        metrics = dict(
            zip(census.METRIC_NAMES, calculate_row_metrics(parent, target, illegal, 1))
        )
        self.assertEqual(metrics["top1_fractional_agreement"], 0.25)
        self.assertEqual(metrics["top1_set_overlap"], 1.0)
        self.assertEqual(metrics["parent_top_tie_rate"], 1.0)
        self.assertEqual(metrics["search_top_tie_rate"], 1.0)
        self.assertEqual(metrics["top_action_disjoint_rate"], 0.0)
        self.assertEqual(metrics["search_top_margin"], 0.0)
        self.assertEqual(metrics["top3_agreement"], 1.0)

    def test_duplicate_malformed_illegal_mass_and_schema_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = [
                ([record(), record()], "duplicate key"),
                ([record(illegal=[False, 0] + [True] * 11)], "boolean"),
                ([record(parent=[0.5, 0.5] + [0.0] * 10)], "13-wide"),
                ([record(parent=[float("nan"), 1.0] + [0.0] * 11)], "non-finite"),
                ([record(parent=[0.75, 0.0, 0.25] + [0.0] * 10)], "mass on an illegal"),
            ]
            schema_row = record()
            schema_row["schema"] = 3.0
            cases.append(([schema_row], "schema must be integer 3"))
            for index, (rows, message) in enumerate(cases):
                path = root / f"bad-{index}.jsonl"
                write_jsonl(path, rows)
                with self.subTest(index=index), self.assertRaisesRegex(AnalysisError, message):
                    load_dataset(spec(path))

    def test_selected_action_requires_positive_target_mass(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            write_jsonl(path, [record(target=[1.0, 0.0] + [0.0] * 11, selected=1)])
            with self.assertRaisesRegex(AnalysisError, "zero visit-target mass"):
                load_dataset(spec(path))

    def test_selected_action_must_be_in_range_and_legal(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_of_range = root / "range.jsonl"
            write_jsonl(out_of_range, [record(selected=13)])
            with self.assertRaisesRegex(AnalysisError, "out of range"):
                load_dataset(spec(out_of_range))
            illegal = root / "illegal.jsonl"
            row = record(selected=2)
            row["visit_target_13"] = [0.5, 0.5, 0.0] + [0.0] * 10
            write_jsonl(illegal, [row])
            with self.assertRaisesRegex(AnalysisError, "is illegal"):
                load_dataset(spec(illegal))

    def test_null_and_known_labels_conflict_within_pov(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            write_jsonl(
                path,
                [record(decision_idx=0, label=None), record(decision_idx=1, label=1)],
            )
            with self.assertRaisesRegex(AnalysisError, "conflicting label state"):
                load_dataset(spec(path))

    def test_clustered_bootstrap_is_deterministic_from_battle_summaries(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            write_jsonl(
                path,
                [
                    record(tag="a", decision_idx=0),
                    record(tag="a", decision_idx=1),
                    record(tag="b", target=[0.0, 1.0] + [0.0] * 11, selected=1, label=0),
                ],
            )
            summary = load_dataset(spec(path))
            first = cluster_bootstrap(summary.battles, repeats=50, seed=17)
            second = cluster_bootstrap(summary.battles, repeats=50, seed=17)
            self.assertEqual(first, second)
            self.assertEqual(first["top1_fractional_agreement"]["cluster_unit"], "battle_tag")

    def test_parent_and_input_hashes_and_no_pooled_metrics(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "r1.jsonl"
            second = root / "r2.jsonl"
            write_jsonl(first, [record()])
            write_jsonl(second, [record(tag="battle-2")])
            result = analyze(
                [
                    InputSpec("round1", "r1", "A" * 64, first),
                    InputSpec("round2", "candidate-6k", "b" * 64, second),
                ],
                repeats=5,
            )
            self.assertEqual(result["datasets"][0]["parent_policy"]["sha256"], "a" * 64)
            self.assertEqual(
                result["datasets"][0]["input"]["sha256"],
                hashlib.sha256(first.read_bytes()).hexdigest(),
            )
            self.assertEqual(result["combined_counts"]["decision_count"], 2)
            self.assertNotIn("metrics", result["combined_counts"])
            self.assertNotIn("combined", result)

    def test_policy_hash_is_required_and_strict(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            write_jsonl(path, [record()])
            with self.assertRaisesRegex(AnalysisError, "64 hexadecimal"):
                load_dataset(InputSpec("round1", "r1", "abc", path))

    def test_report_provenance_is_allowlisted_and_hashed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.jsonl"
            report = root / "report.json"
            write_jsonl(data, [record()])
            report_content = {
                "groups_total": 9,
                "targets_written": 7,
                "rejection_reasons": {"missing_dump_row": 2},
                "match_stats": {
                    "skipped_forced_action_decisions": 3,
                    "visit_match_exact": 999,
                },
                "rejected_groups": [{"secret": "not embedded"}],
                "unknown_payload": {"secret": True},
            }
            report.write_text(json.dumps(report_content))
            result = analyze([spec(data, reports=(report,))], repeats=5)
            provenance = result["datasets"][0]["provenance_reports"][0]
            self.assertEqual(provenance["sha256"], hashlib.sha256(report.read_bytes()).hexdigest())
            self.assertEqual(provenance["counts"]["groups_total"], 9)
            self.assertEqual(
                provenance["count_maps"]["match_stats"],
                {"skipped_forced_action_decisions": 3},
            )
            self.assertNotIn("report", provenance)
            self.assertNotIn("unknown_payload", json.dumps(provenance))

    def test_output_path_collisions_and_existing_outputs_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.jsonl"
            report = root / "report.json"
            output = root / "out.json"
            write_jsonl(data, [record()])
            report.write_text("{}")
            input_spec = spec(data, reports=(report,))
            with self.assertRaisesRegex(AnalysisError, "collides"):
                validate_output_paths([input_spec], data, root / "out.md", False)
            with self.assertRaisesRegex(AnalysisError, "collides"):
                validate_output_paths([input_spec], output, report, False)
            with self.assertRaisesRegex(AnalysisError, "must differ"):
                validate_output_paths([input_spec], output, output, False)
            output.write_text("old")
            with self.assertRaisesRegex(AnalysisError, "already exists"):
                validate_output_paths([input_spec], output, root / "out.md", False)
            validate_output_paths([input_spec], output, root / "out.md", True)

    def test_cli_output_contract_overwrite_and_force(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.jsonl"
            output_json = root / "analysis.json"
            output_markdown = root / "analysis.md"
            write_jsonl(data, [record()])
            command = [
                sys.executable,
                str(ROOT / "scripts" / "analyze_mcts_v3_teacher_gap.py"),
                "--input",
                "round1",
                "r1@epoch5",
                POLICY_SHA,
                str(data),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_markdown),
                "--bootstrap-seed",
                "8",
                "--bootstrap-repeats",
                "10",
            ]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            output = json.loads(output_json.read_text())
            self.assertEqual(output["analysis_schema_version"], 2)
            self.assertEqual(output["bootstrap"]["seed"], 8)
            self.assertEqual(set(output), {
                "analysis_schema_version", "distribution_validation", "divergence_smoothing",
                "bootstrap", "metric_definitions", "stratification_definitions", "datasets",
                "combined_counts", "provenance_note",
            })
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)
            forced = subprocess.run([*command, "--force"], capture_output=True, text=True)
            self.assertEqual(forced.returncode, 0, forced.stderr)
            self.assertIn("`r1@epoch5`", output_markdown.read_text())


if __name__ == "__main__":
    unittest.main()
