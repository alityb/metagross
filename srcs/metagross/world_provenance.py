"""Deterministic world sampling and capture provenance primitives."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import math
import platform
import random
import re
import secrets
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path


RUN_SEED_BYTES = 32
RNG_SCHEME = "hmac-sha256-length-prefixed-v1"
LEDGER_SCHEME = "compact-json-tagged-float-hex-v2"
SEMANTIC_DIGEST_SCHEME = "sha256-canonical-semantic-capture-v1"

_IGNORED_SOURCE_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "external",
}
_NON_SEMANTIC_KEYS = {
    "capture_path",
    "created_at",
    "cwd",
    "directory",
    "elapsed_ms",
    "finished_at",
    "latencies",
    "latency",
    "output",
    "outputs",
    "path",
    "paths",
    "run_dir",
    "source_path",
    "started_at",
    "timestamp",
    "timestamps",
    "timing",
    "updated_at",
}
_CANONICAL_FLOAT_HEX = re.compile(
    r"-?0x(?:0\.0|[01]\.[0-9a-f]{13})p[+-](?:0|[1-9][0-9]*)\Z"
)


def generate_run_seed() -> str:
    """Return a public, cryptographically generated 256-bit run seed."""
    return secrets.token_hex(RUN_SEED_BYTES)


def _validated_run_seed(run_seed: str) -> bytes:
    if not isinstance(run_seed, str):
        raise TypeError("run seed must be a hexadecimal string")
    try:
        decoded = bytes.fromhex(run_seed)
    except ValueError as exc:
        raise ValueError("run seed must be hexadecimal") from exc
    if len(decoded) != RUN_SEED_BYTES or run_seed != run_seed.lower():
        raise ValueError("run seed must be 64 lowercase hexadecimal characters")
    return decoded


def _frame(parts: Iterable[bytes]) -> bytes:
    payload = bytearray()
    for part in parts:
        payload.extend(len(part).to_bytes(8, "big"))
        payload.extend(part)
    return bytes(payload)


def derive_seed(
    run_seed: str,
    channel: str,
    battle_tag: str,
    decision_index: int,
    cohort: str | int,
) -> int:
    """Derive a deterministic unsigned 64-bit seed for one isolated channel."""
    return int.from_bytes(
        _derivation_digest(run_seed, channel, battle_tag, decision_index, cohort)[:8],
        "big",
    )


def _derivation_digest(
    run_seed: str,
    channel: str,
    battle_tag: str,
    decision_index: int,
    cohort: str | int,
) -> bytes:
    if not isinstance(channel, str) or not channel:
        raise ValueError("channel must be a non-empty string")
    if not isinstance(battle_tag, str) or not battle_tag:
        raise ValueError("battle tag must be a non-empty string")
    if isinstance(decision_index, bool) or not isinstance(decision_index, int):
        raise TypeError("decision index must be an integer")
    if decision_index < 0:
        raise ValueError("decision index must be non-negative")
    if isinstance(cohort, bool) or not isinstance(cohort, (str, int)):
        raise TypeError("cohort must be a string or integer")

    message = _frame(
        (
            RNG_SCHEME.encode("ascii"),
            channel.encode("utf-8"),
            battle_tag.encode("utf-8"),
            str(decision_index).encode("ascii"),
            canonical_json(cohort).encode("utf-8"),
        )
    )
    return hmac.new(_validated_run_seed(run_seed), message, hashlib.sha256).digest()


@contextmanager
def seeded_global_random(seed: int) -> Iterator[None]:
    """Temporarily seed Python's process-global random generator."""
    previous_state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(previous_state)


def state_sha256(state: str) -> str:
    """Hash the exact UTF-8 bytes of an engine state string."""
    if not isinstance(state, str):
        raise TypeError("state must be a string")
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def verify_state_sha256(state: str, expected_sha256: str) -> bool:
    """Return whether an engine state has the expected exact UTF-8 hash."""
    return hmac.compare_digest(state_sha256(state), expected_sha256)


def deterministic_request_id(
    run_seed: str,
    battle_tag: str,
    decision_index: int,
    cohort: str | int,
    channel: str = "request-id",
) -> str:
    """Return a deterministic 128-bit hexadecimal request identifier."""
    return _derivation_digest(run_seed, channel, battle_tag, decision_index, cohort)[
        :16
    ].hex()


def canonical_json(value: object) -> str:
    """Serialize supported values as unique compact JSON with tagged hex floats."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON rejects non-finite floats")
        return '{"$float":' + json.dumps(value.hex(), ensure_ascii=False) + "}"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        if set(value) == {"$float"}:
            encoded = value["$float"]
            if (
                not isinstance(encoded, str)
                or _CANONICAL_FLOAT_HEX.fullmatch(encoded) is None
            ):
                raise TypeError("canonical JSON contains an invalid float tag")
            return '{"$float":' + json.dumps(encoded, ensure_ascii=False) + "}"
        if "$float" in value:
            raise TypeError("canonical JSON reserves the $float object key")
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=False)}:{canonical_json(value[key])}"
                for key in sorted(value)
            )
            + "}"
        )
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_jsonl_line(row: object) -> bytes:
    """Serialize one canonical compact JSONL ledger row."""
    return (canonical_json(row) + "\n").encode("utf-8")


def canonical_jsonl(rows: Iterable[object]) -> bytes:
    """Serialize ledger rows to canonical compact JSONL bytes."""
    return b"".join(canonical_jsonl_line(row) for row in rows)


def append_ledger_row(path: Path, row: object) -> None:
    """Append one canonical row to a binary JSONL ledger."""
    with path.open("ab") as handle:
        handle.write(canonical_jsonl_line(row))
        handle.flush()


def _restore_canonical_floats(value: object) -> object:
    if isinstance(value, dict) and set(value) == {"$float"}:
        encoded = value["$float"]
        if not isinstance(encoded, str) or _CANONICAL_FLOAT_HEX.fullmatch(encoded) is None:
            raise ValueError("invalid canonical float tag")
        try:
            restored = float.fromhex(encoded)
        except (OverflowError, ValueError) as exc:
            raise ValueError("invalid canonical float tag") from exc
        if math.isfinite(restored) and restored.hex() == encoded:
            return restored
        raise ValueError("invalid canonical float tag")
    if isinstance(value, list):
        return [_restore_canonical_floats(item) for item in value]
    if isinstance(value, dict):
        if "$float" in value:
            raise ValueError("invalid reserved canonical float key")
        return {key: _restore_canonical_floats(item) for key, item in value.items()}
    return value


def read_ledger(path: Path) -> list[dict]:
    """Read object rows from a byte-for-byte canonical JSONL ledger."""
    rows = []
    for line_number, line in enumerate(path.read_bytes().splitlines(keepends=True), 1):
        if line in {b"\n", b"\r", b"\r\n"}:
            raise ValueError(f"blank line at ledger row {line_number}")
        try:
            parsed = json.loads(line.decode("utf-8"))
            serialized = canonical_jsonl_line(parsed)
        except (
            json.JSONDecodeError,
            TypeError,
            UnicodeDecodeError,
            UnicodeEncodeError,
            ValueError,
        ) as exc:
            raise ValueError(f"invalid JSON at ledger row {line_number}") from exc
        if serialized != line:
            raise ValueError(f"noncanonical JSON at ledger row {line_number}")
        if not isinstance(parsed, dict):
            raise ValueError(f"ledger row {line_number} must be an object")
        try:
            rows.append(_restore_canonical_floats(parsed))
        except ValueError as exc:
            raise ValueError(f"invalid JSON at ledger row {line_number}") from exc
    return rows


def _is_non_semantic_key(key: str) -> bool:
    normalized = key.lower()
    return (
        normalized in _NON_SEMANTIC_KEYS
        or normalized == "pid"
        or normalized == "process_id"
        or normalized.endswith("_at")
        or normalized.endswith("_dir")
        or normalized.endswith("_path")
        or normalized.endswith("_pid")
        or normalized.endswith("_timestamp")
        or "latency" in normalized
    )


def _semantic_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _semantic_value(item)
            for key, item in value.items()
            if isinstance(key, str) and not _is_non_semantic_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    return value


def _load_capture_value(path: Path) -> object:
    if path.is_dir():
        capture = {}
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            if child.is_file() and child.suffix in {".json", ".jsonl"}:
                capture[child.name] = _load_capture_value(child)
        return capture
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON at ledger row {line_number}"
                    ) from exc
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def semantic_capture_digest(capture: object | Path) -> str:
    """Hash capture semantics while excluding location and runtime timing metadata."""
    value = _load_capture_value(capture) if isinstance(capture, Path) else capture
    envelope = {"scheme": SEMANTIC_DIGEST_SCHEME, "capture": _semantic_value(value)}
    return hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash the exact bytes of a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(
    root: Path,
    *,
    ignored_directories: Iterable[str] = (),
) -> str:
    """Hash relative names, types, and exact bytes of a relocatable source tree."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source tree not found: {root}")
    ignored = set(ignored_directories)
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if not any(part in ignored for part in path.relative_to(root).parts)
            and (path.is_file() or path.is_symlink())
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            kind = b"symlink"
            content = path.readlink().as_posix().encode("utf-8")
        else:
            kind = b"file"
            content = path.read_bytes()
        digest.update(_frame((kind, relative, content)))
    return digest.hexdigest()


def source_repository_provenance(root: Path) -> dict[str, object]:
    """Return the checked-out commit and actual source-tree digest."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = result.stdout.strip().lower()
    if result.returncode != 0 or len(commit) != 40:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"cannot identify Foul Play commit: {detail}")
    return {
        "commit": commit,
        "source_sha256": source_tree_sha256(
            root, ignored_directories=_IGNORED_SOURCE_DIRECTORIES
        ),
    }


def foul_play_provenance(root: Path) -> dict[str, object]:
    """Return the checked-out commit and actual Foul Play source-tree digest."""
    return source_repository_provenance(root)


def randbats_dataset_provenance(path: Path) -> dict[str, object]:
    """Return exact content provenance for the active random-battle dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"randbats dataset not found: {path}")
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def holdout_configuration(module_name: str = "srcs.metagross.run_foul_play") -> dict:
    """Read public holdout constants lazily, avoiding an import cycle."""
    source = importlib.import_module(module_name)
    thresholds = {
        "min_effective_worlds": source.HOLDOUT_MIN_EFFECTIVE_WORLDS,
        "min_pairs": source.HOLDOUT_MIN_PAIRS,
        "min_mean_advantage": source.HOLDOUT_MIN_MEAN_ADVANTAGE,
        "min_lower_bound": source.HOLDOUT_MIN_LOWER_BOUND,
        "min_positive_world_weight": source.HOLDOUT_MIN_POSITIVE_WORLD_WEIGHT,
        "max_catastrophic_rate": source.HOLDOUT_MAX_CATASTROPHIC_RATE,
        "min_sign_margin": source.HOLDOUT_MIN_SIGN_MARGIN,
        "alpha_budget": source.HOLDOUT_ALPHA_BUDGET,
        "cvar_tail_mass": source.HOLDOUT_CVAR_TAIL_MASS,
        "max_catastrophe_rate_gap": source.HOLDOUT_MAX_CATASTROPHE_RATE_GAP,
        "max_catastrophe_severity_gap": source.HOLDOUT_MAX_CATASTROPHE_SEVERITY_GAP,
        "min_evaluator_delta_difference": (
            source.HOLDOUT_MIN_EVALUATOR_DELTA_DIFFERENCE
        ),
        "alpha_checks_per_look": source.HOLDOUT_ALPHA_CHECKS_PER_LOOK,
    }
    return {
        "rollouts": source.HOLDOUT_ROLLOUTS,
        "continuation_iterations": source.HOLDOUT_CONTINUATION_ITERATIONS,
        "continuation_horizons": list(source.HOLDOUT_CONTINUATION_HORIZONS),
        "candidate_count": source.HOLDOUT_CANDIDATE_COUNT,
        "opponent_uniform_mix": source.HOLDOUT_OPPONENT_UNIFORM_MIX,
        "alpha_sequence": "global-run-decision-index-v1",
        "thresholds": thresholds,
    }


def manifest_provenance(
    run_seed: str,
    foul_play_root: Path,
    randbats_dataset: Path,
    metamon_root: Path,
    metagross_root: Path,
) -> dict[str, object]:
    """Build the launcher manifest's reproducibility fields."""
    _validated_run_seed(run_seed)
    return {
        "rng": {"scheme": RNG_SCHEME, "run_seed": run_seed},
        "ledger": {"scheme": LEDGER_SCHEME},
        "foul_play": foul_play_provenance(foul_play_root),
        "metamon": source_repository_provenance(metamon_root),
        "metagross": source_repository_provenance(metagross_root),
        "randbats_dataset": randbats_dataset_provenance(randbats_dataset),
        "python_version": platform.python_version(),
        "holdout": holdout_configuration(),
    }
