#!/usr/bin/env python3
"""Aggregate-only admission census for the private r1 basic-move certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import scripts.r1_public_events as r1_public_events_module
from eval.experiment_manifest import validate_manifest
from scripts.r1_public_events import (
    BASIC_MOVE_CERTIFICATE,
    BASIC_MOVE_PRIVATE_BLOCKER_CODES,
    BASIC_MOVE_PRIVATE_OUTCOME_CODES,
    BOOSTED_DOUBLE_SWITCH_CERTIFICATE,
    CERTIFICATE,
    DECLARATIVE_BOOST_CERTIFICATE,
    LEFTOVERS_ACTIVATION_CERTIFICATE,
    MIXED_BOOST_SWITCH_CERTIFICATE,
    R1_SEMANTIC_CONTRACT,
    SEMANTIC_TRACE_CERTIFICATE,
    SILENT_MECHANICS_CERTIFICATE,
    _norm,
    private_transition_blockers,
    private_transition_diagnostic,
    project_information_set_transition,
)
from scripts.teacher_root_bundle import validate_root_capture


REPORT_SCHEMA_VERSION = 1
DEFAULT_UNIFORMS = (0.01, 0.25, 0.5, 0.75, 0.99)
STRUCTURAL_REJECTION = "STRUCTURAL_REJECTION"
INFORMATION_SET_REJECTION = "INFORMATION_SET_REJECTION"
NO_COMMON_ACTION_PAIR = "NO_COMMON_ACTION_PAIR"
OUTCOME_CODES = (
    STRUCTURAL_REJECTION,
    *BASIC_MOVE_PRIVATE_OUTCOME_CODES,
    INFORMATION_SET_REJECTION,
    NO_COMMON_ACTION_PAIR,
)


class AdmissionCensusError(ValueError):
    pass


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise AdmissionCensusError("cannot hash a pinned input artifact") from exc


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        manifest = json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AdmissionCensusError(f"non-finite manifest value {value}")
            ),
        )
        validate_manifest(manifest)
    except Exception as exc:
        if isinstance(exc, AdmissionCensusError):
            raise
        raise AdmissionCensusError("invalid frozen input manifest") from exc
    if manifest.get("manifest_type") != "experiment_input":
        raise AdmissionCensusError("frozen manifest must be an experiment input")
    return manifest, _sha256_bytes(payload)


def _load_captures(path: Path, manifest_hash: str) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AdmissionCensusError("cannot read root captures") from exc
    captures = []
    seen_hashes: set[str] = set()
    seen_identities: set[tuple[Any, ...]] = set()
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            capture = json.loads(
                raw_line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    AdmissionCensusError(f"non-finite capture value {value}")
                ),
            )
            validate_root_capture(capture)
        except Exception as exc:
            raise AdmissionCensusError(
                f"invalid root capture at line {line_number}"
            ) from exc
        if capture.get("schema_version") != 3:
            raise AdmissionCensusError("admission census requires schema-v3 captures")
        configuration = capture.get("configuration", {})
        if configuration.get("input_manifest_sha256") != manifest_hash:
            raise AdmissionCensusError("root capture manifest linkage does not match")
        capture_hash = capture["capture_sha256"]
        identity = capture["identity"]
        identity_key = (
            identity.get("namespace"),
            identity.get("battle_tag"),
            identity.get("username"),
            identity.get("decision_idx"),
        )
        if capture_hash in seen_hashes or identity_key in seen_identities:
            raise AdmissionCensusError("duplicate root capture")
        seen_hashes.add(capture_hash)
        seen_identities.add(identity_key)
        captures.append(capture)
    if not captures:
        raise AdmissionCensusError("root-capture input contains no records")
    captures.sort(key=lambda capture: capture["capture_sha256"])
    return captures, _sha256_bytes(payload)


def _validate_uniforms(uniforms: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in uniforms)
    if not values or any(not math.isfinite(value) or not 0 <= value < 1 for value in values):
        raise AdmissionCensusError("uniforms must be finite values in [0, 1)")
    if len(set(values)) != len(values):
        raise AdmissionCensusError("uniforms must be unique")
    return tuple(sorted(values))


def _root_weight(capture: Mapping[str, Any]) -> float:
    sampling = capture.get("sampling")
    weight = (
        float(sampling["poststratification_weight"])
        if isinstance(sampling, Mapping)
        else 1.0
    )
    if not math.isfinite(weight) or weight <= 0:
        raise AdmissionCensusError("root weights must be finite and positive")
    return weight


def _add(
    counts: dict[str, int],
    masses: dict[str, list[float]],
    code: str,
    mass: float,
) -> None:
    counts[code] += 1
    masses[code].append(mass)


def _table(
    counts: Mapping[str, int],
    masses: Mapping[str, Sequence[float]],
    allowed_codes: Sequence[str],
    raw_denominator: int,
    mass_denominator: float,
) -> dict[str, dict[str, float | int]]:
    table = {}
    for code in allowed_codes:
        count = int(counts.get(code, 0))
        mass = math.fsum(masses.get(code, ()))
        table[code] = {
            "count": count,
            "raw_rate": count / raw_denominator if raw_denominator else 0.0,
            "weighted_mass": mass,
            "weighted_rate": mass / mass_denominator if mass_denominator else 0.0,
        }
    return table


def _deserialize_schedule(engine: Any, schedule: Mapping[str, Any]) -> list[tuple[Any, list[str], list[str], float]]:
    total_weight = float(schedule["world_weight_sum"])
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise AdmissionCensusError("schedule world weight must be positive")
    rows = []
    for world in schedule["worlds"]:
        try:
            state = engine.State.from_string(world["sampled_state"])
            side_one, side_two = engine.root_options(state=state)
        except Exception as exc:
            raise AdmissionCensusError("cannot deserialize captured world") from exc
        side_one = list(side_one)
        side_two = list(side_two)
        if not side_one or not side_two or len(set(side_one)) != len(side_one) or len(set(side_two)) != len(side_two):
            raise AdmissionCensusError("captured world has invalid root options")
        weight = float(world["sample_weight"]) / total_weight
        if not math.isfinite(weight) or weight < 0:
            raise AdmissionCensusError("captured world has invalid sample weight")
        rows.append((state, side_one, side_two, weight))
    return rows


def _public_opponent_registry(capture: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = capture["r1_policy_snapshot"]["player_information_state"][
        "opponent_public_team"
    ]
    return {
        _norm(row["pokemon"]["name"]): {
            "level": int(row["pokemon"]["lvl"]),
            "hp_fraction": float(row["pokemon"]["hp_pct"]),
            "status": str(row["pokemon"]["status"]),
        }
        for row in rows
    }


def analyze_captures(
    captures: Sequence[Mapping[str, Any]],
    *,
    engine: Any,
    uniforms: Sequence[float],
    capture_file_sha256: str,
    source_manifest_sha256: str,
    source_manifest_file_sha256: str,
    analysis_manifest_sha256: str,
    analysis_manifest_file_sha256: str,
    engine_binding_sha256: str,
    census_script_sha256: str,
    public_events_module_sha256: str,
) -> dict[str, Any]:
    """Compute an aggregate-only census from already loaded private captures."""
    uniform_values = _validate_uniforms(uniforms)
    try:
        contract = engine.r1_semantic_contract()
    except Exception as exc:
        raise AdmissionCensusError("engine semantic contract unavailable") from exc
    if contract != R1_SEMANTIC_CONTRACT:
        raise AdmissionCensusError("engine semantic contract does not match")
    if not captures:
        raise AdmissionCensusError("no captures to analyze")
    for capture in captures:
        validate_root_capture(capture)
        if capture.get("schema_version") != 3:
            raise AdmissionCensusError("admission census requires schema-v3 captures")

    root_weights = [_root_weight(capture) for capture in captures]
    root_weight_total = math.fsum(root_weights)
    world_blocker_counts: dict[str, int] = defaultdict(int)
    world_blocker_masses: dict[str, list[float]] = defaultdict(list)
    world_eligibility_counts: dict[str, int] = defaultdict(int)
    world_eligibility_masses: dict[str, list[float]] = defaultdict(list)
    world_outcome_counts: dict[str, int] = defaultdict(int)
    world_outcome_masses: dict[str, list[float]] = defaultdict(list)
    strict_blocker_counts: dict[str, int] = defaultdict(int)
    strict_blocker_masses: dict[str, list[float]] = defaultdict(list)
    strict_outcome_counts: dict[str, int] = defaultdict(int)
    strict_outcome_masses: dict[str, list[float]] = defaultdict(list)
    world_trials = world_certificate_trials = 0
    strict_trials = strict_certificate_trials = 0
    schedule_count = world_count = 0

    for capture, raw_root_weight in zip(captures, root_weights):
        root_mass = raw_root_weight / root_weight_total
        public_opponent = _public_opponent_registry(capture)
        schedules = capture["schedules"]
        schedule_mass = root_mass / len(schedules)
        schedule_count += len(schedules)
        for schedule in schedules:
            rows = _deserialize_schedule(engine, schedule)
            world_count += len(rows)
            for state, side_one_actions, side_two_actions, world_weight in rows:
                pair_count = len(side_one_actions) * len(side_two_actions)
                pair_mass = schedule_mass * world_weight / pair_count
                for side_one_action in side_one_actions:
                    for side_two_action in side_two_actions:
                        world_trials += 1
                        blockers = private_transition_blockers(
                            state,
                            side_one_action,
                            side_two_action,
                            public_opponent=public_opponent,
                        )
                        eligibility = "BLOCKED" if blockers else "ELIGIBLE"
                        _add(
                            world_eligibility_counts,
                            world_eligibility_masses,
                            eligibility,
                            pair_mass,
                        )
                        for blocker in blockers:
                            _add(
                                world_blocker_counts,
                                world_blocker_masses,
                                blocker,
                                pair_mass,
                            )
                        for uniform in uniform_values:
                            world_certificate_trials += 1
                            outcome_mass = pair_mass / len(uniform_values)
                            outcome = (
                                STRUCTURAL_REJECTION
                                if blockers
                                else private_transition_diagnostic(
                                    engine,
                                    state,
                                    side_one_action,
                                    side_two_action,
                                    uniform,
                                    public_opponent=public_opponent,
                                )
                            )
                            _add(
                                world_outcome_counts,
                                world_outcome_masses,
                                outcome,
                                outcome_mass,
                            )

            common_side_one = sorted(
                set.intersection(*(set(row[1]) for row in rows))
            )
            common_side_two = sorted(
                set.intersection(*(set(row[2]) for row in rows))
            )
            common_pair_count = len(common_side_one) * len(common_side_two)
            if common_pair_count == 0:
                for _uniform in uniform_values:
                    strict_certificate_trials += 1
                    _add(
                        strict_outcome_counts,
                        strict_outcome_masses,
                        NO_COMMON_ACTION_PAIR,
                        schedule_mass / len(uniform_values),
                    )
                continue
            strict_pair_mass = schedule_mass / common_pair_count
            states = [row[0] for row in rows]
            for side_one_action in common_side_one:
                for side_two_action in common_side_two:
                    strict_trials += 1
                    blockers = tuple(
                        code
                        for code in BASIC_MOVE_PRIVATE_BLOCKER_CODES
                        if any(
                            code
                            in private_transition_blockers(
                                state,
                                side_one_action,
                                side_two_action,
                                public_opponent=public_opponent,
                            )
                            for state in states
                        )
                    )
                    for blocker in blockers:
                        _add(
                            strict_blocker_counts,
                            strict_blocker_masses,
                            blocker,
                            strict_pair_mass,
                        )
                    for uniform in uniform_values:
                        strict_certificate_trials += 1
                        outcome_mass = strict_pair_mass / len(uniform_values)
                        if blockers:
                            outcome = STRUCTURAL_REJECTION
                        else:
                            diagnostics = [
                                private_transition_diagnostic(
                                    engine,
                                    state,
                                    side_one_action,
                                    side_two_action,
                                    uniform,
                                    public_opponent=public_opponent,
                                )
                                for state in states
                            ]
                            outcome = next(
                                (code for code in diagnostics if code != "ADMITTED"),
                                "ADMITTED",
                            )
                            if outcome == "ADMITTED":
                                try:
                                    project_information_set_transition(
                                        engine,
                                        states,
                                        side_one_action,
                                        side_two_action,
                                        uniform,
                                        public_opponent=public_opponent,
                                    )
                                except Exception:
                                    outcome = INFORMATION_SET_REJECTION
                        _add(
                            strict_outcome_counts,
                            strict_outcome_masses,
                            outcome,
                            outcome_mass,
                        )

    world_mass = math.fsum(world_eligibility_masses["ELIGIBLE"]) + math.fsum(
        world_eligibility_masses["BLOCKED"]
    )
    strict_mass = math.fsum(
        mass for values in strict_outcome_masses.values() for mass in values
    )
    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "record_type": "r1_basic_move_admission_census",
        "claim_status": "private_descriptive_only",
        "certificates": [
            CERTIFICATE,
            BOOSTED_DOUBLE_SWITCH_CERTIFICATE,
            BASIC_MOVE_CERTIFICATE,
            SILENT_MECHANICS_CERTIFICATE,
            DECLARATIVE_BOOST_CERTIFICATE,
            MIXED_BOOST_SWITCH_CERTIFICATE,
            LEFTOVERS_ACTIVATION_CERTIFICATE,
            SEMANTIC_TRACE_CERTIFICATE,
        ],
        "semantic_contract": R1_SEMANTIC_CONTRACT,
        "inputs": {
            "capture_file_sha256": capture_file_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "source_manifest_file_sha256": source_manifest_file_sha256,
            "analysis_manifest_sha256": analysis_manifest_sha256,
            "analysis_manifest_file_sha256": analysis_manifest_file_sha256,
            "engine_binding_sha256": engine_binding_sha256,
            "census_script_sha256": census_script_sha256,
            "public_events_module_sha256": public_events_module_sha256,
        },
        "configuration": {
            "uniforms": list(uniform_values),
            "root_weighting": "poststratification weight when present, otherwise equal roots",
            "schedule_weighting": "equal within root",
            "world_weighting": "normalized captured sample weights within schedule",
            "joint_action_weighting": "uniform over engine-reported legal joint actions",
            "blocker_semantics": "overlapping; one trial may contribute to multiple blockers",
        },
        "privacy": {
            "sampled_private_states_read": True,
            "sampled_private_states_emitted": False,
            "root_or_player_identifiers_emitted": False,
            "action_or_move_names_emitted": False,
            "exception_text_emitted": False,
            "per_root_rows_emitted": False,
        },
        "counts": {
            "roots": len(captures),
            "schedules": schedule_count,
            "sampled_worlds": world_count,
            "world_joint_action_trials": world_trials,
            "world_certificate_trials": world_certificate_trials,
            "strict_schedule_joint_action_trials": strict_trials,
            "strict_schedule_certificate_trials": strict_certificate_trials,
        },
        "world_trials": {
            "eligibility": _table(
                world_eligibility_counts,
                world_eligibility_masses,
                ("ELIGIBLE", "BLOCKED"),
                world_trials,
                world_mass,
            ),
            "overlapping_blockers": _table(
                world_blocker_counts,
                world_blocker_masses,
                BASIC_MOVE_PRIVATE_BLOCKER_CODES,
                world_trials,
                world_mass,
            ),
            "certificate_outcomes": _table(
                world_outcome_counts,
                world_outcome_masses,
                OUTCOME_CODES,
                world_certificate_trials,
                world_mass,
            ),
        },
        "strict_schedule_trials": {
            "overlapping_blockers": _table(
                strict_blocker_counts,
                strict_blocker_masses,
                BASIC_MOVE_PRIVATE_BLOCKER_CODES,
                strict_trials,
                strict_mass,
            ),
            "certificate_outcomes": _table(
                strict_outcome_counts,
                strict_outcome_masses,
                OUTCOME_CODES,
                strict_certificate_trials,
                strict_mass,
            ),
        },
        "continuation_readiness": {
            "status": "blocked",
            "r1_continuation_value_allowed": False,
            "coverage_basis": "one_step_uniform_joint_action_descriptive",
            "blockers": [
                "sequential_policy_weighted_coverage_not_measured",
                "opponent_pov_continuation_state_not_captured",
                "terminating_policy_rule_not_defined",
                "multi_turn_terminal_parity_not_certified",
            ],
        },
        "limitations": [
            "This is a private descriptive coverage census, not action-value or strength evidence.",
            "Overlapping blockers identify prevalence but do not estimate the gain from removing one blocker in isolation.",
            "Uniform legal joint-action weighting is diagnostic and is not a policy distribution.",
            "Only the declared certificates are tested; unsupported transitions remain rejected.",
        ],
    }
    report["report_sha256"] = hashlib.sha256(
        _canonical_json(report).encode("ascii")
    ).hexdigest()
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("report_schema_version") != REPORT_SCHEMA_VERSION or report.get(
        "record_type"
    ) != "r1_basic_move_admission_census":
        raise AdmissionCensusError("invalid admission-census report")
    if report.get("claim_status") != "private_descriptive_only":
        raise AdmissionCensusError("admission census has invalid claim status")
    readiness = report.get("continuation_readiness")
    if readiness is not None and (
        not isinstance(readiness, Mapping)
        or readiness.get("status") != "blocked"
        or readiness.get("r1_continuation_value_allowed") is not False
    ):
        raise AdmissionCensusError("admission census has invalid continuation readiness")
    unhashed = dict(report)
    claimed = unhashed.pop("report_sha256", None)
    actual = hashlib.sha256(_canonical_json(unhashed).encode("ascii")).hexdigest()
    if claimed != actual:
        raise AdmissionCensusError("admission-census report hash does not match")
    for section_name in ("world_trials", "strict_schedule_trials"):
        section = report.get(section_name)
        if not isinstance(section, Mapping):
            raise AdmissionCensusError("admission-census section is invalid")
        blockers = section.get("overlapping_blockers")
        outcomes = section.get("certificate_outcomes")
        if not isinstance(blockers, Mapping) or set(blockers) != set(
            BASIC_MOVE_PRIVATE_BLOCKER_CODES
        ):
            raise AdmissionCensusError("admission-census blocker table is invalid")
        if not isinstance(outcomes, Mapping) or set(outcomes) != set(OUTCOME_CODES):
            raise AdmissionCensusError("admission-census outcome table is invalid")
    serialized = _canonical_json(report)
    for forbidden in (
        "sampled_state",
        "state_sha256",
        "battle_tag",
        "username",
        "decision_idx",
        "name_table",
        "protocol_prefix",
    ):
        if forbidden in serialized:
            raise AdmissionCensusError("admission-census report contains private fields")


def census_file(
    input_path: Path,
    source_manifest_path: Path,
    analysis_manifest_path: Path,
    *,
    engine: Any,
    uniforms: Sequence[float] = DEFAULT_UNIFORMS,
) -> dict[str, Any]:
    source_manifest, source_manifest_file_hash = _load_manifest(
        source_manifest_path
    )
    analysis_manifest, analysis_manifest_file_hash = _load_manifest(
        analysis_manifest_path
    )
    engine_artifact = analysis_manifest.get("artifacts", {}).get("engine_binding")
    if not isinstance(engine_artifact, Mapping):
        raise AdmissionCensusError("frozen manifest does not pin an engine binding")
    engine_path = Path(str(engine_artifact.get("path", "")))
    engine_hash = _sha256_file(engine_path)
    if engine_hash != engine_artifact.get("sha256"):
        raise AdmissionCensusError("pinned engine binding hash does not match")
    runtime_extension = getattr(getattr(engine, "poke_engine", None), "__file__", None)
    if runtime_extension is None or Path(runtime_extension).resolve() != engine_path.resolve():
        raise AdmissionCensusError("runtime engine binding is not the pinned artifact")
    captures, capture_file_hash = _load_captures(
        input_path, str(source_manifest["manifest_sha256"])
    )
    return analyze_captures(
        captures,
        engine=engine,
        uniforms=uniforms,
        capture_file_sha256=capture_file_hash,
        source_manifest_sha256=str(source_manifest["manifest_sha256"]),
        source_manifest_file_sha256=source_manifest_file_hash,
        analysis_manifest_sha256=str(analysis_manifest["manifest_sha256"]),
        analysis_manifest_file_sha256=analysis_manifest_file_hash,
        engine_binding_sha256=engine_hash,
        census_script_sha256=_sha256_file(Path(__file__)),
        public_events_module_sha256=_sha256_file(
            Path(r1_public_events_module.__file__)
        ),
    )


def write_report(report: Mapping[str, Any], path: Path, *, force: bool = False) -> None:
    validate_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and os.path.lexists(path):
        raise AdmissionCensusError(f"output already exists: {path}")
    payload = (_canonical_json(dict(report)) + "\n").encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise AdmissionCensusError(f"output already exists: {path}") from exc
            temporary_path.unlink()
        temporary_path = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--analysis-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uniform", action="append", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    import poke_engine

    report = census_file(
        args.input,
        args.source_manifest,
        args.analysis_manifest,
        engine=poke_engine,
        uniforms=args.uniform or DEFAULT_UNIFORMS,
    )
    write_report(report, args.output, force=args.force)
    print(_canonical_json(report))


if __name__ == "__main__":
    main()
