#!/usr/bin/env python3
"""Apply the frozen selector and capture-size screen to matrix diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    protocol_path: Path, report_path: Path, audit_protocol_path: Path
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit_protocol = json.loads(audit_protocol_path.read_text(encoding="utf-8"))
    if (
        audit_protocol.get("status") != "frozen_before_execution"
        or audit_protocol["matrix_protocol_sha256"] != _sha256(protocol_path)
        or audit_protocol["matrix_report_sha256"] != _sha256(report_path)
    ):
        raise ValueError("matrix audit inputs differ from the frozen audit protocol")
    for source in [audit_protocol["runner"], *audit_protocol["dependencies"]]:
        if _sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("matrix audit source differs from the frozen audit protocol")
    if protocol.get("status") != "frozen_before_execution":
        raise ValueError("matrix diagnostic protocol is not frozen")
    for source in [protocol["runner"], *protocol["runner"]["dependencies"]]:
        if _sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("matrix diagnostic source differs from the protocol")
    if report.get("schema_version") != 1 or report.get("mode") != "shared_root_matrix_diagnostics":
        raise ValueError("matrix diagnostic report has an invalid contract")
    if report["input"]["sha256"] != protocol["input"]["sha256"]:
        raise ValueError("matrix diagnostic input differs from the protocol")
    prior_protocol = protocol.get("prior_enrichment")
    prior_report = report.get("prior_enrichment")
    if prior_protocol is None:
        if prior_report is not None:
            raise ValueError("matrix diagnostic has unregistered prior enrichment")
    elif (
        not isinstance(prior_report, dict)
        or prior_report.get("sha256") != prior_protocol.get("sha256")
        or prior_report.get("capture_panel", {}).get("sha256")
        != prior_protocol.get("capture_panel_sha256")
        or prior_report.get("counts") != prior_protocol.get("counts")
        or prior_report.get("provenance") != prior_protocol.get("provenance")
    ):
        raise ValueError("matrix diagnostic prior enrichment differs from the protocol")
    for name in (
        "iterations",
        "continuation_iterations",
        "seed",
        "prior_strength",
        "robust_contamination",
        "severe_regression_delta",
        "serialization_repeats",
        "teacher",
    ):
        if report["configuration"][name] != protocol["configuration"][name]:
            raise ValueError("matrix diagnostic configuration differs from the protocol")
    if (
        report["engine"]["contract"] != protocol["engine"]["contract"]
        or report["engine"]["source_sha256"]
        != protocol["engine"]["source_sha256"]
        or report["engine"]["native_sha256"]
        != protocol["engine"]["native_extension_sha256"]
    ):
        raise ValueError("matrix diagnostic engine differs from the protocol")
    expected_strategies = set(protocol["treatments"]) - {"current_rm_mixed"}
    if set(report["strategy_summaries"]) != expected_strategies:
        raise ValueError("matrix diagnostic treatments differ from the protocol")
    counts = report["counts"]
    roots = report["roots"]
    if not isinstance(roots, list) or len(roots) != 26:
        raise ValueError("matrix diagnostic root panel is incomplete")
    schedules = [schedule for root in roots for schedule in root["schedules"]]
    if (
        any(
            len(root["schedules"]) != 4
            or len({schedule["schedule_id"] for schedule in root["schedules"]}) != 4
            for root in roots
        )
        or len(schedules) != 104
        or any(
            len(schedule["teacher_repeats"]) != 3
            or len({repeat["repeat"] for repeat in schedule["teacher_repeats"]}) != 3
            or not isinstance(schedule.get("capture_sha256"), str)
            or len(schedule["capture_sha256"]) != 64
            for schedule in schedules
        )
    ):
        raise ValueError("matrix diagnostic schedule/repeat panel is incomplete")
    recomputed_counts = {
        "roots": len(roots),
        "schedules": len(schedules),
        "teacher_comparisons": sum(
            len(schedule["teacher_repeats"]) for schedule in schedules
        ),
        "payoff_cells": sum(schedule["payoff_cells"] for schedule in schedules),
        "schedules_with_complete_opponent_priors": sum(
            schedule["strategy_aggregates"]["strategies"][
                "opponent_prior_expected"
            ]["selected_action"]
            is not None
            for schedule in schedules
        ),
    }
    if counts != recomputed_counts or len({s["capture_sha256"] for s in schedules}) != 104:
        raise ValueError("matrix diagnostic counts or capture identities are inconsistent")
    capture_checks = {
        "all_104_schema4_captures_validated": counts["schedules"] == 104
        and len({schedule["capture_sha256"] for schedule in schedules}) == 104,
        "native_capture_size": report["serialization_summary"]["native_capture"]["bytes"][
            "max"
        ]
        <= protocol["screen"]["capture_size_limits"]["native_capture_bytes"],
        "envelope_size": report["serialization_summary"]["envelope"]["bytes"]["max"]
        <= protocol["screen"]["capture_size_limits"]["envelope_bytes"],
        "full_row_size": report["serialization_summary"]["full_row"]["bytes"]["max"]
        <= protocol["screen"]["capture_size_limits"]["full_row_bytes"],
    }
    baseline = report["strategy_summaries"]["rm_policy_argmax"]
    selectors = {}
    for name, summary in report["strategy_summaries"].items():
        if name == "rm_policy_argmax":
            continue
        checks = {
            "available_on_all_roots": summary["available_roots"] == counts["roots"],
            "complete_opponent_prior_coverage": (
                counts["schedules_with_complete_opponent_priors"] == counts["schedules"]
                if name == "opponent_prior_expected" or name.startswith("bounded_robust_")
                else True
            ),
            "poststratified_teacher_mass_noninferior": summary[
                "poststratified_teacher_mass"
            ]
            is not None
            and summary["poststratified_teacher_mass"]
            >= baseline["poststratified_teacher_mass"],
            "poststratified_teacher_argmax_agreement_noninferior": summary[
                "poststratified_teacher_argmax_agreement_fraction"
            ]
            is not None
            and summary["poststratified_teacher_argmax_agreement_fraction"]
            >= baseline["poststratified_teacher_argmax_agreement_fraction"],
            "no_severe_regressions": summary["severe_regressions"] == 0,
        }
        selectors[name] = {"checks": checks, "passed": all(checks.values())}
    capture_screen_passed = all(capture_checks.values())
    passing = sorted(
        name
        for name, result in selectors.items()
        if result["passed"] and capture_screen_passed
    )
    return {
        "schema_version": 1,
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "audit_protocol": {
            "path": str(audit_protocol_path),
            "sha256": _sha256(audit_protocol_path),
        },
        "capture_checks": capture_checks,
        "capture_screen_passed": capture_screen_passed,
        "opponent_prior_availability": {
            "available_schedules": counts["schedules_with_complete_opponent_priors"],
            "required_schedules": counts["schedules"],
            "prior_based_treatments_blocked": counts[
                "schedules_with_complete_opponent_priors"
            ]
            != counts["schedules"],
        },
        "reference": {
            "name": "rm_policy_argmax",
            "poststratified_teacher_mass": baseline["poststratified_teacher_mass"],
            "poststratified_teacher_argmax_agreement_fraction": baseline[
                "poststratified_teacher_argmax_agreement_fraction"
            ],
        },
        "selectors": selectors,
        "passing_selectors": passing,
        "selector_screen_passed": bool(passing) and capture_screen_passed,
        "decision": (
            "future_selector_preregistration_supported"
            if passing
            else "no_selector_supported_collect_opponent_priors"
        ),
        "authorization": {
            "new_games_authorized": False,
            "candidate_promotion_authorized": False,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audit-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit(
        args.protocol.expanduser().resolve(),
        args.report.expanduser().resolve(),
        args.audit_protocol.expanduser().resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
