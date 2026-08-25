#!/usr/bin/env python3
"""Fail-closed aggregate auditor for joining dual-player schema-v3 snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_SCHEMA_VERSION = 1
FAILURE_CATEGORIES = (
    "unpaired_boundary",
    "duplicate_role",
    "nonreciprocal_identity",
    "public_prefix_mismatch",
    "invalid_snapshot",
    "duplicate_identity",
)
PRIVACY_DECLARATION = {
    "aggregate_only": True,
    "identifiers_emitted": False,
    "snapshots_emitted": False,
    "actions_emitted": False,
    "exception_strings_emitted": False,
    "per_boundary_rows_emitted": False,
}
CAPTURE_ONLY_CLAIM = "proves_capture_joinability_only"
FORBIDDEN_SOURCE_KEYS = {
    "namespace",
    "tag",
    "battle_tag",
    "username",
    "opponent_username",
    "decision_idx",
    "battle_turn",
    "probs",
    "illegal_actions",
    "name_table",
    "protocol_prefix",
    "player_information_state",
    "private_request",
    "action",
    "actions",
    "snapshot",
    "r1_policy_snapshot",
    "exception",
}
FORBIDDEN_STRING_FRAGMENTS = ("|request|", "battle-")
NON_BATTLE_MESSAGE_TYPES = {
    "c",
    "c:",
    "chat",
    "html",
    "inactive",
    "inactiveoff",
    "init",
    "j",
    "join",
    "l",
    "leave",
    "n",
    "name",
    "raw",
    "request",
    "t:",
    "title",
    "uhtml",
    "uhtmlchange",
}
HP_MESSAGE_FIELDS = {"switch": 4, "drag": 4, "-damage": 3, "-heal": 3}


class DualSnapshotAuditError(ValueError):
    pass


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_battle_tag(value: str) -> str:
    tag = value.strip().lower()
    if tag.startswith("battle-"):
        tag = tag[len("battle-") :]
    return tag


def normalize_username(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _snapshot_from_row(row: object) -> Mapping[str, Any] | None:
    if not isinstance(row, Mapping):
        return None
    wrapped = row.get("r1_policy_snapshot")
    if wrapped is not None:
        return wrapped if isinstance(wrapped, Mapping) else None
    return row


def _validated_snapshot(row: object) -> Mapping[str, Any] | None:
    snapshot = _snapshot_from_row(row)
    if snapshot is None:
        return None
    namespace = snapshot.get("namespace")
    tag = snapshot.get("tag")
    username = snapshot.get("username")
    opponent = snapshot.get("opponent_username")
    role = snapshot.get("player_role")
    probs = snapshot.get("probs")
    illegal = snapshot.get("illegal_actions")
    names = snapshot.get("name_table")
    information = snapshot.get("player_information_state")
    prefix = snapshot.get("protocol_prefix")
    if (
        snapshot.get("schema") != 3
        or not isinstance(namespace, str)
        or not namespace.strip()
        or not isinstance(tag, str)
        or not normalize_battle_tag(tag)
        or not isinstance(username, str)
        or not normalize_username(username)
        or role not in {"p1", "p2"}
        or not isinstance(opponent, str)
        or not normalize_username(opponent)
        or not _is_int(snapshot.get("decision_idx"))
        or snapshot["decision_idx"] < 0
        or not _is_int(snapshot.get("battle_turn"))
        or snapshot["battle_turn"] < 0
        or not isinstance(probs, list)
        or len(probs) != 13
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in probs
        )
        or not math.isclose(
            math.fsum(float(value) for value in probs),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not isinstance(illegal, list)
        or len(illegal) != 13
        or any(not isinstance(value, bool) for value in illegal)
        or any(flag and float(probability) != 0.0 for flag, probability in zip(illegal, probs))
        or snapshot.get("mask_fallback") is not False
        or not isinstance(names, Mapping)
        or not names
        or any(
            not isinstance(name, str)
            or not name
            or not _is_int(index)
            or not 0 <= index < 13
            for name, index in names.items()
        )
        or not isinstance(information, Mapping)
        or information.get("schema_version") != 1
        or not isinstance(information.get("private_request"), Mapping)
        or not isinstance(information["private_request"].get("side"), Mapping)
        or information["private_request"]["side"].get("id") != role
        or not isinstance(prefix, list)
        or any(not isinstance(line, str) for line in prefix)
    ):
        return None
    return snapshot


def _identity(snapshot: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        str(snapshot["namespace"]),
        normalize_battle_tag(str(snapshot["tag"])),
        normalize_username(str(snapshot["username"])),
        snapshot["decision_idx"],
    )


def _boundary(snapshot: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        str(snapshot["namespace"]),
        normalize_battle_tag(str(snapshot["tag"])),
        snapshot["battle_turn"],
    )


def _canonical_public_line(line: str) -> str | None:
    if not line.startswith("|"):
        return None
    fields = line.split("|")
    if len(fields) < 2 or not fields[1] or fields[1] in NON_BATTLE_MESSAGE_TYPES:
        return None
    hp_field = HP_MESSAGE_FIELDS.get(fields[1])
    if hp_field is not None and len(fields) > hp_field:
        match = re.fullmatch(r"(\d+)/(\d+)(.*)", fields[hp_field])
        if match and int(match.group(2)) > 0:
            hp, maximum = int(match.group(1)), int(match.group(2))
            displayed = math.ceil(100 * hp / maximum)
            if displayed == 100 and hp < maximum:
                displayed = 99
            fields[hp_field] = f"{displayed}/100{match.group(3)}"
    return "|".join(fields)


def _public_prefix(snapshot: Mapping[str, Any]) -> list[str]:
    canonical = (
        _canonical_public_line(line) for line in snapshot["protocol_prefix"]
    )
    return [line for line in canonical if line is not None]


def _is_forced_switch(snapshot: Mapping[str, Any]) -> bool:
    request = snapshot["player_information_state"]["private_request"]
    forced = request.get("forceSwitch", [])
    return isinstance(forced, list) and any(value is True for value in forced)


def audit_snapshots(
    rows: Sequence[object], *, input_file_sha256: Sequence[str]
) -> dict[str, Any]:
    """Audit loaded private rows and return only aggregate joinability evidence."""
    hashes = list(input_file_sha256)
    if len(hashes) < 2 or len(set(hashes)) != len(hashes) or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        raise DualSnapshotAuditError("at least two distinct valid input file hashes are required")

    failures = {category: 0 for category in FAILURE_CATEGORIES}
    valid: list[Mapping[str, Any]] = []
    for row in rows:
        snapshot = _validated_snapshot(row)
        if snapshot is None:
            failures["invalid_snapshot"] += 1
        else:
            valid.append(snapshot)

    by_identity: dict[tuple[object, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for snapshot in valid:
        by_identity[_identity(snapshot)].append(snapshot)
    candidates = []
    for identity_rows in by_identity.values():
        if len(identity_rows) != 1:
            failures["duplicate_identity"] += len(identity_rows)
        else:
            candidates.extend(identity_rows)

    boundaries: dict[tuple[object, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for snapshot in candidates:
        boundaries[_boundary(snapshot)].append(snapshot)

    valid_pairs = 0
    certified_one_sided = 0
    for boundary_rows in boundaries.values():
        by_prefix: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for snapshot in boundary_rows:
            by_prefix[tuple(_public_prefix(snapshot))].append(snapshot)
        ordinary_singletons = []
        for prefix_rows in by_prefix.values():
            if len(prefix_rows) == 1:
                if _is_forced_switch(prefix_rows[0]):
                    certified_one_sided += 1
                else:
                    ordinary_singletons.extend(prefix_rows)
                continue
            if len(prefix_rows) != 2:
                failures["unpaired_boundary"] += 1
                continue
            first, second = prefix_rows
            if first["player_role"] == second["player_role"]:
                failures["duplicate_role"] += 1
                continue
            first_user = normalize_username(str(first["username"]))
            second_user = normalize_username(str(second["username"]))
            if (
                normalize_username(str(first["opponent_username"])) != second_user
                or normalize_username(str(second["opponent_username"])) != first_user
            ):
                failures["nonreciprocal_identity"] += 1
                continue
            valid_pairs += 1
        if len(ordinary_singletons) == 2 and {
            snapshot["player_role"] for snapshot in ordinary_singletons
        } == {"p1", "p2"}:
            failures["public_prefix_mismatch"] += 1
        else:
            failures["unpaired_boundary"] += len(ordinary_singletons)

    failure_total = sum(failures.values())
    candidate_boundaries = valid_pairs + certified_one_sided + sum(
        failures[name]
        for name in (
            "unpaired_boundary",
            "duplicate_role",
            "nonreciprocal_identity",
            "public_prefix_mismatch",
        )
    )
    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "record_type": "dual_r1_policy_snapshot_join_audit",
        "status": "eligible" if valid_pairs > 0 and failure_total == 0 else "ineligible",
        "claim": CAPTURE_ONLY_CLAIM,
        "r1_continuation_value_allowed": False,
        "inputs": {
            "file_count": len(hashes),
            "file_sha256": hashes,
        },
        "counts": {
            "input_rows": len(rows),
            "valid_snapshot_rows": len(valid),
            "candidate_rows": len(candidates),
            "candidate_boundaries": candidate_boundaries,
            "valid_pairs": valid_pairs,
            "certified_one_sided_boundaries": certified_one_sided,
        },
        "failures": failures,
        "privacy": dict(PRIVACY_DECLARATION),
    }
    report["report_sha256"] = _sha256_bytes(_canonical_json(report).encode("ascii"))
    validate_report(report)
    return report


def _check_privacy(value: object) -> None:
    if isinstance(value, Mapping):
        if any(key in FORBIDDEN_SOURCE_KEYS for key in value):
            raise DualSnapshotAuditError("report contains a forbidden private key")
        for key, item in value.items():
            _check_privacy(key)
            _check_privacy(item)
    elif isinstance(value, list):
        for item in value:
            _check_privacy(item)
    elif isinstance(value, str) and any(fragment in value.lower() for fragment in FORBIDDEN_STRING_FRAGMENTS):
        raise DualSnapshotAuditError("report contains a forbidden private string")


def validate_report(report: Mapping[str, Any]) -> None:
    if set(report) != {
        "report_schema_version",
        "record_type",
        "status",
        "claim",
        "r1_continuation_value_allowed",
        "inputs",
        "counts",
        "failures",
        "privacy",
        "report_sha256",
    }:
        raise DualSnapshotAuditError("invalid report schema")
    if (
        report.get("report_schema_version") != REPORT_SCHEMA_VERSION
        or report.get("record_type") != "dual_r1_policy_snapshot_join_audit"
        or report.get("status") not in {"eligible", "ineligible"}
        or report.get("claim") != CAPTURE_ONLY_CLAIM
        or report.get("r1_continuation_value_allowed") is not False
        or report.get("privacy") != PRIVACY_DECLARATION
    ):
        raise DualSnapshotAuditError("invalid report declaration")
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {"file_count", "file_sha256"}:
        raise DualSnapshotAuditError("invalid report inputs")
    hashes = inputs.get("file_sha256")
    if (
        not _is_int(inputs.get("file_count"))
        or inputs["file_count"] < 2
        or not isinstance(hashes, list)
        or len(hashes) != inputs["file_count"]
        or len(set(hashes)) != len(hashes)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        )
    ):
        raise DualSnapshotAuditError("invalid report inputs")
    counts = report.get("counts")
    failures = report.get("failures")
    if (
        not isinstance(counts, Mapping)
        or set(counts)
        != {
            "input_rows",
            "valid_snapshot_rows",
            "candidate_rows",
            "candidate_boundaries",
            "valid_pairs",
            "certified_one_sided_boundaries",
        }
        or not isinstance(failures, Mapping)
        or set(failures) != set(FAILURE_CATEGORIES)
        or any(not _is_int(value) or value < 0 for value in [*counts.values(), *failures.values()])
    ):
        raise DualSnapshotAuditError("invalid report counts")
    if (
        counts["input_rows"]
        != failures["invalid_snapshot"] + counts["valid_snapshot_rows"]
        or counts["valid_snapshot_rows"]
        != failures["duplicate_identity"] + counts["candidate_rows"]
        or counts["candidate_boundaries"]
        != counts["valid_pairs"]
        + counts["certified_one_sided_boundaries"]
        + failures["unpaired_boundary"]
        + failures["duplicate_role"]
        + failures["nonreciprocal_identity"]
        + failures["public_prefix_mismatch"]
    ):
        raise DualSnapshotAuditError("report counts are not conserved")
    expected_status = (
        "eligible"
        if counts["valid_pairs"] > 0 and sum(failures.values()) == 0
        else "ineligible"
    )
    if report["status"] != expected_status:
        raise DualSnapshotAuditError("report status does not match counts")
    unhashed = dict(report)
    claimed = unhashed.pop("report_sha256", None)
    if claimed != _sha256_bytes(_canonical_json(unhashed).encode("ascii")):
        raise DualSnapshotAuditError("report hash does not match")
    _check_privacy(report)


def audit_files(input_paths: Sequence[Path]) -> dict[str, Any]:
    if len(input_paths) < 2:
        raise DualSnapshotAuditError("at least two --input files are required")
    rows: list[object] = []
    hashes: list[str] = []
    for path in input_paths:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise DualSnapshotAuditError("cannot read an input file") from exc
        hashes.append(_sha256_bytes(payload))
        for raw_line in payload.splitlines():
            if not raw_line.strip():
                continue
            try:
                rows.append(
                    json.loads(
                        raw_line,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
                    )
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                rows.append(None)
    return audit_snapshots(rows, input_file_sha256=hashes)


def write_report(report: Mapping[str, Any], path: Path, *, force: bool = False) -> None:
    validate_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and os.path.lexists(path):
        raise DualSnapshotAuditError("output already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path: Path | None = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write((_canonical_json(dict(report)) + "\n").encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise DualSnapshotAuditError("output already exists") from exc
            temporary_path.unlink()
        temporary_path = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if len(args.input) < 2:
        parser.error("--input must be supplied at least twice")
    return args


def main() -> None:
    args = parse_args()
    report = audit_files(args.input)
    write_report(report, args.output, force=args.force)
    print(_canonical_json(report))


if __name__ == "__main__":
    main()
