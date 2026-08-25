import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from srcs.metagross.shared_root_matrix_audit import audit


class SharedRootMatrixAuditTest(unittest.TestCase):
    def test_screen_requires_all_frozen_conditions(self):
        configuration = {
            "iterations": 10,
            "continuation_iterations": 2,
            "seed": 3,
            "prior_strength": 1.0,
            "robust_contamination": [0.1, 0.25, 0.5],
            "severe_regression_delta": -0.2,
            "serialization_repeats": 3,
            "teacher": "teacher",
        }
        engine = {
            "contract": "contract",
            "source_sha256": "source",
            "native_extension_sha256": "native",
        }
        protocol = {
            "status": "frozen_before_execution",
            "input": {"sha256": "input"},
            "configuration": configuration,
            "engine": engine,
            "treatments": [
                "current_rm_mixed",
                "rm_policy_argmax",
                "better",
                "opponent_prior_expected",
                "regression",
            ],
            "screen": {
                "capture_size_limits": {
                    "native_capture_bytes": 100,
                    "envelope_bytes": 100,
                    "full_row_bytes": 200,
                }
            },
        }
        baseline = {
            "available_roots": 26,
            "poststratified_teacher_mass": 0.5,
            "poststratified_teacher_argmax_agreement_fraction": 0.6,
            "severe_regressions": 0,
        }
        report = {
            "schema_version": 1,
            "mode": "shared_root_matrix_diagnostics",
            "input": {"sha256": "input"},
            "prior_enrichment": None,
            "configuration": configuration,
            "engine": {
                "contract": "contract",
                "source_sha256": "source",
                "native_sha256": "native",
            },
            "counts": {
                "roots": 26,
                "schedules": 104,
                "teacher_comparisons": 312,
                "payoff_cells": 104,
                "schedules_with_complete_opponent_priors": 0,
            },
            "serialization_summary": {
                "native_capture": {"bytes": {"max": 90}},
                "envelope": {"bytes": {"max": 80}},
                "full_row": {"bytes": {"max": 150}},
            },
            "strategy_summaries": {
                "rm_policy_argmax": baseline,
                "better": {
                    **baseline,
                    "poststratified_teacher_mass": 0.51,
                },
                "opponent_prior_expected": {
                    **baseline,
                    "poststratified_teacher_mass": 0.7,
                    "poststratified_teacher_argmax_agreement_fraction": 0.7,
                },
                "regression": {
                    **baseline,
                    "poststratified_teacher_mass": 0.7,
                    "severe_regressions": 1,
                },
            },
            "roots": [
                {
                    "schedules": [
                        {
                            "schedule_id": schedule,
                            "capture_sha256": f"{root * 4 + schedule:064x}",
                            "payoff_cells": 1,
                            "teacher_repeats": [
                                {"repeat": repeat} for repeat in range(3)
                            ],
                            "strategy_aggregates": {
                                "strategies": {
                                    "opponent_prior_expected": {
                                        "selected_action": None
                                    }
                                }
                            },
                        }
                        for schedule in range(4)
                    ]
                }
                for root in range(26)
            ],
        }
        with TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "source.py"
            protocol_path = Path(temporary) / "protocol.json"
            report_path = Path(temporary) / "report.json"
            audit_protocol_path = Path(temporary) / "audit-protocol.json"
            source_path.write_text("source")
            source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            protocol["runner"] = {
                "path": str(source_path),
                "sha256": source_hash,
                "dependencies": [],
            }
            protocol_path.write_text(json.dumps(protocol))
            report_path.write_text(json.dumps(report))
            audit_protocol_path.write_text(
                json.dumps(
                    {
                        "status": "frozen_before_execution",
                        "matrix_protocol_sha256": hashlib.sha256(
                            protocol_path.read_bytes()
                        ).hexdigest(),
                        "matrix_report_sha256": hashlib.sha256(
                            report_path.read_bytes()
                        ).hexdigest(),
                        "runner": {"path": str(source_path), "sha256": source_hash},
                        "dependencies": [],
                    }
                )
            )
            result = audit(protocol_path, report_path, audit_protocol_path)
            report["prior_enrichment"] = {"sha256": "unexpected"}
            report_path.write_text(json.dumps(report))
            audit_protocol = json.loads(audit_protocol_path.read_text())
            audit_protocol["matrix_report_sha256"] = hashlib.sha256(
                report_path.read_bytes()
            ).hexdigest()
            audit_protocol_path.write_text(json.dumps(audit_protocol))
            with self.assertRaisesRegex(ValueError, "unregistered prior enrichment"):
                audit(protocol_path, report_path, audit_protocol_path)
        self.assertTrue(result["capture_screen_passed"])
        self.assertEqual(result["passing_selectors"], ["better"])
        self.assertTrue(result["opponent_prior_availability"]["prior_based_treatments_blocked"])
        self.assertFalse(
            result["selectors"]["opponent_prior_expected"]["checks"][
                "complete_opponent_prior_coverage"
            ]
        )

if __name__ == "__main__":
    unittest.main()
