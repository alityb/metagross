#!/usr/bin/env python3
"""Select a deterministic stratified panel of private teacher root captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from eval.experiment_manifest import validate_manifest  # noqa: E402
from scripts.teacher_root_bundle import (  # noqa: E402
    RootBundleError,
    validate_root_capture,
)


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_ENTROPY_THRESHOLD = 1.0


class PanelSelectionError(ValueError):
    """Raised when private panel selection cannot satisfy its contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PanelSelectionError(f"{description} must be a positive integer")
    return value


def _validate_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise PanelSelectionError("seed must fit an unsigned 64-bit integer")
    return seed


def _validate_entropy_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PanelSelectionError("entropy threshold must be numeric")
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise PanelSelectionError("entropy threshold must be finite and nonnegative")
    return threshold


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        manifest = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                PanelSelectionError(f"non-finite manifest constant {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, PanelSelectionError) as exc:
        raise PanelSelectionError(
            f"cannot read frozen input manifest {path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise PanelSelectionError("frozen input manifest must be an object")
    try:
        validate_manifest(manifest)
    except ValueError as exc:
        raise PanelSelectionError(f"invalid frozen input manifest: {exc}") from exc
    if manifest.get("manifest_type") != "experiment_input":
        raise PanelSelectionError(
            "panel selection requires an experiment_input manifest"
        )
    return manifest, hashlib.sha256(payload).hexdigest()


def battle_turn_bin(turn: Any) -> str:
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
        raise PanelSelectionError("capture battle_turn must be a nonnegative integer")
    if turn <= 5:
        return "0-5"
    if turn <= 10:
        return "6-10"
    if turn <= 20:
        return "11-20"
    if turn <= 30:
        return "21-30"
    return "31+"


def _validated_player_priors(capture: Mapping[str, Any]) -> list[tuple[str, float]]:
    entries = capture.get("recorded_player_priors")
    if not isinstance(entries, list) or not entries:
        raise PanelSelectionError("recorded player priors must be a nonempty list")
    priors: list[tuple[str, float]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2:
            raise PanelSelectionError(
                "recorded player priors must contain [action, mass] pairs"
            )
        action, raw_mass = entry
        if not isinstance(action, str) or not action or action in seen:
            raise PanelSelectionError(
                "recorded player priors contain an invalid or duplicate action"
            )
        if isinstance(raw_mass, bool) or not isinstance(raw_mass, (int, float)):
            raise PanelSelectionError("recorded player prior mass must be numeric")
        mass = float(raw_mass)
        if not math.isfinite(mass) or mass < 0.0:
            raise PanelSelectionError(
                "recorded player prior mass must be finite and nonnegative"
            )
        seen.add(action)
        priors.append((action, mass))
    total = math.fsum(mass for _, mass in priors)
    if total <= 0.0:
        raise PanelSelectionError("recorded player priors have no positive mass")
    return [(action, mass / total) for action, mass in priors]


def derive_stratum(
    capture: Mapping[str, Any], *, entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD
) -> dict[str, Any]:
    """Return the frozen composite stratum for one validated source capture."""
    threshold = _validate_entropy_threshold(entropy_threshold)
    identity = capture.get("identity")
    if not isinstance(identity, dict):
        raise PanelSelectionError("capture identity must be an object")
    priors = _validated_player_priors(capture)
    entropy = -math.fsum(mass * math.log(mass) for _, mass in priors if mass > 0.0)
    actions = [action for action, _ in priors]
    return {
        "battle_turn_bin": battle_turn_bin(identity.get("battle_turn")),
        "recorded_player_prior_entropy_bin": (
            "below_threshold" if entropy < threshold else "at_or_above_threshold"
        ),
        "recorded_player_prior_entropy_threshold": threshold,
        "legal_action_count": len(actions),
        "tera_available": any(action.endswith("-tera") for action in actions),
    }


def ranking_sha256(seed: int, capture_sha256: str) -> str:
    """Hash an unambiguous canonical tuple of the seed and source capture hash."""
    _validate_seed(seed)
    if (
        not isinstance(capture_sha256, str)
        or len(capture_sha256) != 64
        or any(character not in "0123456789abcdef" for character in capture_sha256)
    ):
        raise PanelSelectionError(
            "capture SHA-256 must be 64 lowercase hexadecimal characters"
        )
    material = [seed, capture_sha256]
    return hashlib.sha256(_canonical_json(material).encode("ascii")).hexdigest()


def _load_captures(
    path: Path, *, manifest_sha256: str
) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PanelSelectionError(
            f"cannot read teacher root captures {path}: {exc}"
        ) from exc
    captures: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            capture = json.loads(
                raw_line.decode("utf-8"),
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    PanelSelectionError(f"non-finite JSON constant {constant}")
                ),
            )
        except (UnicodeError, json.JSONDecodeError, PanelSelectionError) as exc:
            raise PanelSelectionError(
                f"{path}:{line_number}: invalid capture JSON: {exc}"
            ) from exc
        if not isinstance(capture, dict):
            raise PanelSelectionError(
                f"{path}:{line_number}: capture must be an object"
            )
        try:
            validate_root_capture(capture)
            if "sampling" in capture:
                raise PanelSelectionError("source capture is already sampled")
            configuration = capture.get("configuration")
            if not isinstance(configuration, dict) or (
                configuration.get("input_manifest_sha256") != manifest_sha256
            ):
                raise PanelSelectionError(
                    "capture does not link to the supplied frozen input manifest"
                )
            _validated_player_priors(capture)
        except (
            RootBundleError,
            PanelSelectionError,
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise PanelSelectionError(f"{path}:{line_number}: {exc}") from exc
        capture_sha256 = capture["capture_sha256"]
        if capture_sha256 in seen_hashes:
            raise PanelSelectionError(
                f"{path}:{line_number}: duplicate capture SHA-256"
            )
        seen_hashes.add(capture_sha256)
        captures.append(capture)
    if not captures:
        raise PanelSelectionError("teacher root-capture input contains no records")
    return captures, hashlib.sha256(payload).hexdigest()


def select_captures(
    captures: Sequence[Mapping[str, Any]],
    *,
    per_stratum: int,
    seed: int,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
) -> tuple[list[dict[str, Any]], int]:
    """Select and reseal captures, returning records and population stratum count."""
    limit = _positive_integer(per_stratum, "per-stratum")
    checked_seed = _validate_seed(seed)
    threshold = _validate_entropy_threshold(entropy_threshold)
    strata: dict[str, tuple[dict[str, Any], list[Mapping[str, Any]]]] = {}
    seen_hashes: set[str] = set()
    for capture in captures:
        try:
            validate_root_capture(capture)
        except (RootBundleError, TypeError, ValueError, OverflowError) as exc:
            raise PanelSelectionError(f"invalid source capture: {exc}") from exc
        if "sampling" in capture:
            raise PanelSelectionError("source capture is already sampled")
        source_hash = capture.get("capture_sha256")
        if not isinstance(source_hash, str):
            raise PanelSelectionError("capture is missing its SHA-256")
        if source_hash in seen_hashes:
            raise PanelSelectionError("duplicate capture SHA-256")
        seen_hashes.add(source_hash)
        stratum = derive_stratum(capture, entropy_threshold=threshold)
        key = _canonical_json(stratum)
        if key not in strata:
            strata[key] = (stratum, [])
        strata[key][1].append(capture)

    selected: list[dict[str, Any]] = []
    for key in sorted(strata):
        stratum, population = strata[key]
        ranked = sorted(
            population,
            key=lambda capture: (
                ranking_sha256(checked_seed, str(capture["capture_sha256"])),
                capture["capture_sha256"],
            ),
        )
        selected_count = min(limit, len(ranked))
        for source in ranked[:selected_count]:
            source_hash = str(source["capture_sha256"])
            sampled = dict(source)
            sampled.pop("capture_sha256")
            sampled["sampling"] = {
                "source_capture_sha256": source_hash,
                "stratum": dict(stratum),
                "population_count": len(ranked),
                "selected_count": selected_count,
                "inclusion_probability": selected_count / len(ranked),
                "poststratification_weight": len(ranked) / selected_count,
            }
            sampled["capture_sha256"] = hashlib.sha256(
                _canonical_json(sampled).encode("ascii")
            ).hexdigest()
            try:
                validate_root_capture(sampled)
            except RootBundleError as exc:
                raise PanelSelectionError(
                    f"selected capture failed reseal validation: {exc}"
                ) from exc
            selected.append(sampled)
    return selected, len(strata)


def _paths_collide(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _stage_private(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        return temporary_path
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


def _install_pair(
    output_path: Path,
    output_payload: bytes,
    summary_path: Path,
    summary_payload: bytes,
    *,
    force: bool,
) -> None:
    if not force:
        for path in (output_path, summary_path):
            if os.path.lexists(path):
                raise PanelSelectionError(f"output already exists: {path}")
    output_temporary = _stage_private(output_path, output_payload)
    summary_temporary: Path | None = None
    output_installed = False
    try:
        summary_temporary = _stage_private(summary_path, summary_payload)
        if force:
            os.replace(output_temporary, output_path)
            output_temporary = None
            os.replace(summary_temporary, summary_path)
            summary_temporary = None
        else:
            try:
                os.link(output_temporary, output_path, follow_symlinks=False)
                output_installed = True
                os.link(summary_temporary, summary_path, follow_symlinks=False)
            except FileExistsError as exc:
                raise PanelSelectionError(
                    f"output already exists: {exc.filename}"
                ) from exc
            output_temporary.unlink()
            output_temporary = None
            summary_temporary.unlink()
            summary_temporary = None
    except Exception:
        if output_installed and output_temporary is not None:
            try:
                if os.path.samefile(output_temporary, output_path):
                    output_path.unlink()
            except OSError:
                pass
        raise
    finally:
        if output_temporary is not None:
            output_temporary.unlink(missing_ok=True)
        if summary_temporary is not None:
            summary_temporary.unlink(missing_ok=True)


def select_file(
    input_path: Path,
    input_manifest_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    per_stratum: int,
    seed: int,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    force: bool = False,
) -> dict[str, Any]:
    paths = (input_path, input_manifest_path, output_path, summary_path)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if _paths_collide(left, right):
                raise PanelSelectionError(
                    "input, manifest, output, and summary paths must be distinct"
                )
    if not force:
        for path in (output_path, summary_path):
            if os.path.lexists(path):
                raise PanelSelectionError(f"output already exists: {path}")

    limit = _positive_integer(per_stratum, "per-stratum")
    checked_seed = _validate_seed(seed)
    threshold = _validate_entropy_threshold(entropy_threshold)
    manifest, manifest_file_sha256 = _load_manifest(input_manifest_path)
    manifest_sha256 = str(manifest["manifest_sha256"])
    captures, input_sha256 = _load_captures(input_path, manifest_sha256=manifest_sha256)
    selected, stratum_count = select_captures(
        captures,
        per_stratum=limit,
        seed=checked_seed,
        entropy_threshold=threshold,
    )
    output_payload = "".join(
        _canonical_json(capture) + "\n" for capture in selected
    ).encode("ascii")
    output_sha256 = hashlib.sha256(output_payload).hexdigest()
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "record_type": "teacher_root_panel_selection_summary",
        "input_sha256": input_sha256,
        "input_manifest_sha256": manifest_sha256,
        "input_manifest_file_sha256": manifest_file_sha256,
        "output_sha256": output_sha256,
        "configuration": {
            "per_stratum": limit,
            "seed": checked_seed,
            "recorded_player_prior_entropy_threshold": threshold,
        },
        "counts": {
            "population_count": len(captures),
            "selected_count": len(selected),
            "stratum_count": stratum_count,
        },
    }
    summary["summary_sha256"] = hashlib.sha256(
        _canonical_json(summary).encode("ascii")
    ).hexdigest()
    summary_payload = (_canonical_json(summary) + "\n").encode("ascii")
    _install_pair(
        output_path,
        output_payload,
        summary_path,
        summary_payload,
        force=force,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--per-stratum", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--entropy-threshold", type=float, default=DEFAULT_ENTROPY_THRESHOLD
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = select_file(
        args.input,
        args.input_manifest,
        args.output,
        args.summary,
        per_stratum=args.per_stratum,
        seed=args.seed,
        entropy_threshold=args.entropy_threshold,
        force=args.force,
    )
    print(_canonical_json(summary))


if __name__ == "__main__":
    main()
