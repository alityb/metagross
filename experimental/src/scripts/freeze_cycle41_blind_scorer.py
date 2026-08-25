#!/usr/bin/env python3
"""Freeze Cycle41's outcome-blind audit and every immutable Cycle40 input."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
C40 = ROOT / "experimental/runs/search_native_v2_cycle40_integrated_h2h_20260816"
RUN = ROOT / "experimental/runs/search_native_v2_cycle41_blind_scorer_repair_20260816"
RESULT_SHA = "67641e46cfa8dc0cfd9528d21fadb66c5ed08e414df7466d1967017acd97aca5"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = RUN / "PREOUTCOME_MANIFEST.json"
    if output.exists():
        raise FileExistsError(output)
    tests = json.loads((RUN / "PREFREEZE_TESTS.json").read_text())
    audit = json.loads((RUN / "PREOUTCOME_AUDIT.json").read_text())
    stop = json.loads((C40 / "STOP_REPORT.json").read_text())
    if tests.get("passed") != 29 or tests.get("failed") != 0:
        raise RuntimeError("Cycle41 tests failed")
    if sha(RUN / "prefreeze-junit.xml") != tests.get("junit_sha256"):
        raise RuntimeError("Cycle41 JUnit changed")
    if (
        audit.get("status") != "pass"
        or audit.get("outcome_or_win_fields_read") is not False
        or audit.get("candidate_streams") != 20
        or audit.get("comparator_streams") != 20
        or audit.get("candidate_public_roles") != {"p1": 10, "p2": 10}
        or audit.get("unjoined_receipt_cohorts") != 0
        or audit.get("semantic_operational_failures") != 0
        or sha(RUN / "PREOUTCOME_AUDIT.json") != tests.get("preoutcome_audit_sha256")
    ):
        raise RuntimeError("Cycle41 outcome-blind audit is not admitted")
    if (
        stop.get("status") != "STOP_FAIL_FROZEN_SCORER_INTEGRITY"
        or stop.get("outcome_fields_opened") is not False
        or stop.get("partial_or_final_score_opened") is not False
    ):
        raise RuntimeError("Cycle40 was not preserved unopened")
    result = C40 / "h2h-result.json"
    if sha(result) != RESULT_SHA or audit.get("result_bytes_sha256") != RESULT_SHA:
        raise RuntimeError("immutable Cycle40 result bytes changed")

    fixed = [
        RUN / "PROTOCOL.md",
        RUN / "PREFREEZE_TESTS.json",
        RUN / "prefreeze-junit.xml",
        RUN / "PREOUTCOME_AUDIT.json",
        C40 / "H2H_PREMEASUREMENT_MANIFEST.json",
        C40 / "STOP_REPORT.json",
        C40 / "PROTOCOL.md",
        C40 / "CANONICAL_H2H_ARGV.json",
        C40 / "PRIOR_IDENTITY_REGISTRY_V2.json",
        C40 / "h2h-result.json.pairs.json",
        result,
        C40 / "REGISTRATION_CONSUMPTION.json",
        C40 / "SHOWDOWN_LAUNCH.json",
        C40 / "h2h-runner.log",
        C40 / "h2h-showdown.log",
        ROOT / "experimental/src/scripts/score_cycle41_blind_role_repair.py",
        ROOT / "experimental/src/scripts/freeze_cycle41_blind_scorer.py",
        ROOT / "experimental/src/scripts/tests/test_cycle41_blind_role_scorer.py",
        ROOT / "experimental/src/scripts/verify_cycle33_h2h_freeze.py",
        ROOT / "experimental/src/scripts/summarize_cycle19_h2h.py",
        ROOT / "experimental/src/scripts/summarize_cycle33_h2h.py",
        ROOT / "experimental/src/scripts/monitor_cycle19_operational_smoke.py",
        ROOT / "experimental/src/scripts/monitor_cycle21_registered_form_smoke.py",
    ]
    dynamic = []
    for relative, pattern, expected in (
        ("h2h-logs", "*.log", 40),
        ("h2h-logs", "*.search.jsonl", 40),
        ("h2h-logs", "*.protocol.jsonl", 40),
        ("move-receipts", "*.jsonl", 40),
        ("engine-receipts", "*.json", 2),
    ):
        paths = sorted((C40 / relative).glob(pattern))
        if len(paths) != expected:
            raise RuntimeError(f"Cycle41 {relative}/{pattern} denominator changed")
        dynamic.extend(paths)
    ability = sorted((C40 / "ability-receipts").glob("*.jsonl"))
    if not ability:
        raise RuntimeError("Cycle41 ability receipts missing")
    dynamic.extend(ability)
    files = fixed + dynamic
    if any(not path.is_file() for path in files):
        raise RuntimeError("Cycle41 frozen input missing")
    manifest = {
        "schema": "metagross-cycle41-preoutcome-manifest/v1",
        "cycle": 41,
        "status": "frozen_after_outcome_blind_audit_before_score_opening",
        "protocol_sha256": sha(RUN / "PROTOCOL.md"),
        "preoutcome_audit_sha256": sha(RUN / "PREOUTCOME_AUDIT.json"),
        "immutable_cycle40_result_sha256": RESULT_SHA,
        "outcome_or_win_fields_read": False,
        "files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in files],
        "denominators": {
            "games": 20,
            "mirrored_pairs": 10,
            "candidate_streams": 20,
            "comparator_streams": 20,
            "candidate_public_p1": 10,
            "candidate_public_p2": 10,
            "move_receipt_files": 40,
            "ability_receipt_files": len(ability),
        },
        "gate": {
            "candidate_wins_to_continue": 13,
            "all_integrity_failures": 0,
            "threshold_changed": False,
        },
        "authorization": {
            "open_immutable_cycle40_score_once": True,
            "new_games": False,
            "training": False,
            "sealed93": False,
            "gpu_cloud_paid": False,
        },
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest_sha256": sha(output),
        "protocol_sha256": manifest["protocol_sha256"],
        "audit_sha256": manifest["preoutcome_audit_sha256"],
        "immutable_result_sha256": RESULT_SHA,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
