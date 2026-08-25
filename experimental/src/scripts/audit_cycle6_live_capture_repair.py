#!/usr/bin/env python3
"""Run the frozen Cycle 6 symmetric native-setter admission exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from scripts.audit_cycle5_live_capture import AdmissionError, ROOT, main as run_cycle5_gates
from scripts.run_public_search_state_gate_a import sha256
from srcs.metagross.causal_reveal_ledger import (
    CausalRevealLedgerError,
    STATE_SERIALIZATION_FIELDS,
    parse_state_serialization,
    serialize_state_fields,
    serialization_without_masks,
)


def validate_frozen_inputs(run_dir: Path) -> dict:
    frozen = json.loads((run_dir / "FROZEN_INPUTS.json").read_text())
    for relative, expected in frozen["files"].items():
        if sha256(ROOT / relative) != expected:
            raise AdmissionError(f"frozen input hash mismatch: {relative}")
    engine_source_paths = []
    for relative_root in frozen["engine_source_roots"]:
        source_root = ROOT / relative_root
        engine_source_paths.extend(
            path for path in source_root.rglob("*")
            if path.is_file() and "target" not in path.parts
        )
    digest = hashlib.sha256()
    for path in sorted(engine_source_paths):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    if digest.hexdigest() != frozen["engine_source_tree_sha256"]:
        raise AdmissionError("frozen engine source-tree hash mismatch")
    return frozen


def nonzero_matrix_state(engine):
    fields = parse_state_serialization(engine.State().to_string())
    fields["s1_threat"] = "1.25"
    fields["s2_threat"] = "-2.5"
    fields["scout_value"] = "3.75"
    fields["threat_matrix"] = ";".join(str(index + 1) for index in range(36))
    fields["wincon_matrix"] = ";".join(str(-(index + 1)) for index in range(36))
    return engine.State.from_string(serialize_state_fields(fields))


def setter_isolation_gates(engine) -> dict[str, bool | int]:
    state = nonzero_matrix_state(engine)
    original_string = state.to_string()
    original = parse_state_serialization(original_string)
    if tuple(original) != STATE_SERIALIZATION_FIELDS:
        raise AdmissionError("named state grammar order changed")
    if serialize_state_fields(original) != original_string:
        raise AdmissionError("13-field parse/serialize roundtrip failed")
    if engine.State.from_string(original_string).to_string() != original_string:
        raise AdmissionError("native 13-field roundtrip failed")
    try:
        parse_state_serialization(original_string + "/extra")
    except CausalRevealLedgerError:
        malformed_rejected = True
    else:
        raise AdmissionError("malformed serialization field count was accepted")

    rng = random.Random(2026081506)
    masks = [1, 3, (1 << 17) | 5, (1 << 42) - 1]
    masks.extend(rng.randrange(1, 1 << 42) for _ in range(128))
    for bits in masks:
        s1 = state.with_side_one_public_reveals(bits)
        s1_fields = parse_state_serialization(s1.to_string())
        if s1_fields != {**original, "s1_public_reveals": str(bits)}:
            raise AdmissionError("native side-one setter changed a non-mask field")
        s2 = state.with_side_two_public_reveals(bits)
        s2_fields = parse_state_serialization(s2.to_string())
        if s2_fields != {**original, "s2_public_reveals": str(bits)}:
            raise AdmissionError("native side-two setter changed a non-mask field")
        both = s1.with_side_two_public_reveals(bits)
        restored = both.with_side_one_public_reveals(0).with_side_two_public_reveals(0)
        if restored.to_string() != original_string:
            raise AdmissionError("two-mask set/clear did not restore exact bytes")
        if serialization_without_masks(both) != original_string:
            raise AdmissionError("named non-mask serialization parity failed")
    return {
        "field_count": len(STATE_SERIALIZATION_FIELDS),
        "parse_serialize_exact": True,
        "native_roundtrip_exact": True,
        "malformed_count_rejected": malformed_rejected,
        "native_side_one_isolated": True,
        "native_side_two_isolated": True,
        "two_mask_set_clear_exact": True,
        "property_masks_checked": len(masks),
        "nonzero_threat_wincon_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    final_path = run_dir / "cycle6-live-capture-admission.json"
    base_path = run_dir / "cycle5-live-capture-admission.json"
    archived_base_path = run_dir / "cycle6-base-cycle5-gates.json"
    if any(path.exists() for path in (final_path, base_path, archived_base_path)):
        raise AdmissionError("Cycle 6 report already exists")
    frozen = validate_frozen_inputs(run_dir)

    import poke_engine

    repair_gates = setter_isolation_gates(poke_engine)
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], "--run-dir", str(run_dir)]
        run_cycle5_gates()
    finally:
        sys.argv = original_argv
    base = json.loads(base_path.read_text())
    if base.get("status") != "pass":
        raise AdmissionError("unchanged Cycle 5 gates did not pass")
    base_path.rename(archived_base_path)
    report = {
        **base,
        "schema": "metagross-cycle6-live-capture-repair-admission/v1",
        "status": "pass",
        "cycle6_repair_gates": repair_gates,
        "authorization": {
            **base["authorization"],
            "freeze_target_collection_permission": True,
            "deployment": False,
        },
        "sidecar_boundary": {
            **base["sidecar_boundary"],
            "rust_interior_sidecar_updates_proven": False,
            "newly_simulated_consumed_item_updates_proven": False,
            "deployed_rust_interior_inference_authorized": False,
        },
        "hashes": {
            **base["hashes"],
            "protocol_sha256": frozen["protocol_sha256"],
            "frozen_inputs_sha256": sha256(run_dir / "FROZEN_INPUTS.json"),
            "base_gate_report_sha256": sha256(archived_base_path),
        },
    }
    report.pop("report_sha256", None)
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    final_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
