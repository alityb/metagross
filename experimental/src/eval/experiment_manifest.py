"""Frozen, reproducible manifests for experimental runs.

This module intentionally does not depend on the eval runner.  A caller can
write an input manifest before starting work and later write a separate
completion manifest without changing the frozen input record.

The self-hash and exclusive local write prevent accidental replacement; they
do not make preregistration tamper-proof.  Publish the manifest or hash to an
external append-only system, or sign it externally, when proof against later
local tampering is required.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
REDACTED = "[REDACTED]"
DEFAULT_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "SLURM_JOB_ID",
    "SLURM_PROCID",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_OBVIOUS_SECRET_VALUES = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)^bearer\s+\S+"),
    re.compile(r"(?i)^[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@"),
    re.compile(r"^AKIA[0-9A-Z]{16}$"),
    re.compile(r"^(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{16,}$"),
    re.compile(r"^xox[baprs]-[A-Za-z0-9-]{10,}$"),
    re.compile(r"^sk-[A-Za-z0-9_-]{20,}$"),
    re.compile(r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$"),
    re.compile(r"(?i)^(?:proxy-)?authorization\s*:\s*\S+"),
    re.compile(r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|secret)\s*[=:]\s*\S+"),
)

_INPUT_FIELDS = {
    "schema_version",
    "manifest_type",
    "experiment_id",
    "run_id",
    "created_at_utc",
    "argv",
    "environment",
    "git",
    "dependencies",
    "artifacts",
    "configurations",
    "random_seeds",
    "host",
    "resources",
    "preregistration",
    "manifest_sha256",
}
_COMPLETION_FIELDS = {
    "schema_version",
    "manifest_type",
    "experiment_id",
    "run_id",
    "created_at_utc",
    "frozen_input_manifest_sha256",
    "result_hashes",
    "artifact_hashes",
    "counts",
    "manifest_sha256",
}
_HOST_FIELDS = {
    "hostname",
    "platform",
    "system",
    "release",
    "machine",
    "python_implementation",
    "python_version",
    "python_executable",
}


class ManifestError(ValueError):
    """Raised when a manifest cannot be safely created or validated."""


def _is_secret_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    compact = normalized.replace("_", "")
    exact = {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "access_key",
        "private_key",
        "credential",
        "credentials",
        "cookie",
        "session_key",
        "authorization",
    }
    common_compact = {
        "pgpassword",
        "mysqlpwd",
        "secretkeybase",
        "awssecretaccesskey",
        "authorization",
        "proxyauthorization",
    }
    return (
        compact in common_compact
        or normalized in exact
        or any(normalized.endswith("_" + suffix) for suffix in exact)
    )


def _is_obvious_secret_value(value: str) -> bool:
    return any(pattern.search(value) for pattern in _OBVIOUS_SECRET_VALUES)


def _validate_json_value(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and value != REDACTED and _is_obvious_secret_value(value):
            raise ManifestError(f"obvious secret value at {path}")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestError(f"non-finite number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ManifestError(f"non-string object key at {path}")
            if _is_secret_key(key) and item != REDACTED:
                raise ManifestError(f"secret-bearing key must be redacted at {path}.{key}")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ManifestError(f"unsupported JSON value {type(value).__name__} at {path}")


def canonical_json(value: object) -> str:
    """Return the unique compact JSON encoding used for manifest hashes."""
    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _content_hash(manifest: Mapping[str, object]) -> str:
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json(unhashed).encode("ascii")).hexdigest()


def _hash_regular_file(path: Path) -> tuple[str, int, int]:
    """Return hash, size, and mode from one stable, no-follow descriptor."""
    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise ManifestError(f"cannot inspect artifact {path}: {exc}") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise ManifestError(f"artifact is not a regular non-symlink file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManifestError(f"cannot open artifact {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise ManifestError(f"artifact changed while opening: {path}")
        digest = hashlib.sha256()
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ManifestError(f"artifact changed while hashing: {path}")
    if bytes_read != before.st_size:
        raise ManifestError(f"artifact size changed while hashing: {path}")
    return digest.hexdigest(), bytes_read, before.st_mode


def sha256_file(path: Path | str) -> str:
    """Hash one regular, non-symlink file from a consistency-checked descriptor."""
    return _hash_regular_file(Path(path))[0]


def hash_tree(path: Path | str) -> dict[str, object]:
    """Hash a tree with path/type framing, without following symlinks."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise ManifestError(f"tree is not a directory: {root}")
    entries: list[tuple[bytes, str, Path]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(directory_names):
            item = parent / name
            if item.is_symlink():
                entries.append((item.relative_to(root).as_posix().encode("utf-8"), "symlink", item))
                directory_names.remove(name)
        for name in file_names:
            item = parent / name
            kind = "symlink" if item.is_symlink() else "file"
            entries.append((item.relative_to(root).as_posix().encode("utf-8"), kind, item))
    digest = hashlib.sha256()
    file_count = 0
    symlink_count = 0
    total_bytes = 0
    for relative, kind, item in sorted(entries, key=lambda entry: entry[0]):
        path_stat = item.lstat()
        if kind == "symlink" and stat.S_ISLNK(path_stat.st_mode):
            payload = os.readlink(item).encode("utf-8")
            record_type = b"L"
            symlink_count += 1
        elif kind == "file" and stat.S_ISREG(path_stat.st_mode):
            file_hash, size, _mode = _hash_regular_file(item)
            payload = size.to_bytes(8, "big") + bytes.fromhex(file_hash)
            record_type = b"F"
            file_count += 1
            total_bytes += size
        else:
            raise ManifestError(f"unsupported tree entry: {item}")
        digest.update(record_type)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {
        "scheme": "framed-tree-sha256-v1",
        "files": file_count,
        "symlinks": symlink_count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def hash_artifacts(artifacts: Mapping[str, Path | str]) -> dict[str, object]:
    """Return path, size, and SHA-256 metadata for named files."""
    if not isinstance(artifacts, Mapping):
        raise ManifestError("artifacts must be a mapping")
    hashed: dict[str, object] = {}
    for name, raw_path in artifacts.items():
        if not isinstance(name, str) or not name:
            raise ManifestError("artifact names must be non-empty strings")
        digest, size, _mode = _hash_regular_file(Path(raw_path))
        hashed[name] = {
            "path": str(raw_path),
            "sha256": digest,
            "size_bytes": size,
        }
    return hashed


def collect_environment(
    keys: Iterable[str] = DEFAULT_ENVIRONMENT_KEYS,
    environ: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Capture only selected variables, redacting anything secret-like."""
    source = os.environ if environ is None else environ
    selected: dict[str, str] = {}
    for key in keys:
        if not isinstance(key, str) or not key:
            raise ManifestError("environment keys must be non-empty strings")
        if key not in source:
            continue
        value = source[key]
        if not isinstance(value, str):
            raise ManifestError(f"environment value for {key} must be a string")
        selected[key] = REDACTED if _is_secret_key(key) or _is_obvious_secret_value(value) else value
    return selected


def _redact_argv(argv: Sequence[str]) -> list[str]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        raise ManifestError("argv must be a sequence of strings")
    sanitized: list[str] = []
    redact_next = False
    for argument in argv:
        if not isinstance(argument, str):
            raise ManifestError("argv entries must be strings")
        if redact_next:
            sanitized.append(REDACTED)
            redact_next = False
            continue
        if argument.startswith("-"):
            flag, separator, value = argument.partition("=")
            if _is_secret_key(flag.lstrip("-")):
                sanitized.append(flag + "=" + REDACTED if separator else flag)
                redact_next = not separator
                continue
            if separator and _is_obvious_secret_value(value):
                sanitized.append(flag + "=" + REDACTED)
                continue
        sanitized.append(REDACTED if _is_obvious_secret_value(argument) else argument)
    return sanitized


def _run_git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestError(f"cannot collect git identity in {repo}: {exc}") from exc


def _untracked_material(repo: Path) -> bytes:
    material = bytearray()
    untracked = _run_git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    for raw_name in sorted(name for name in untracked.split(b"\0") if name):
        path = repo / os.fsdecode(raw_name)
        try:
            path_stat = path.lstat()
        except OSError as exc:
            raise ManifestError(f"cannot inspect untracked path {path}: {exc}") from exc
        material.extend(b"untracked\0" + raw_name + b"\0")
        if stat.S_ISREG(path_stat.st_mode):
            digest, size, mode = _hash_regular_file(path)
            material.extend(f"regular\0{mode:o}\0{size}\0{digest}\0".encode("ascii"))
        elif stat.S_ISLNK(path_stat.st_mode):
            try:
                target = os.fsencode(os.readlink(path))
                after = path.lstat()
            except OSError as exc:
                raise ManifestError(f"cannot inspect untracked symlink {path}: {exc}") from exc
            fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(path_stat, field) != getattr(after, field) for field in fields):
                raise ManifestError(f"untracked symlink changed while hashing: {path}")
            material.extend(f"symlink\0{path_stat.st_mode:o}\0".encode("ascii") + target + b"\0")
        else:
            material.extend(
                f"special\0{path_stat.st_mode:o}\0{path_stat.st_rdev}\0".encode("ascii")
            )
    return bytes(material)


def _worktree_material(repo: Path) -> tuple[bytes, bool]:
    status = _run_git(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    material = bytearray(b"status\0" + status)
    material.extend(
        b"diff\0"
        + _run_git(repo, "diff", "--binary", "--no-ext-diff", "--submodule=log", "HEAD")
    )
    material.extend(_untracked_material(repo))
    return bytes(material), bool(status)


def collect_git_identity(repo: Path | str = ".") -> dict[str, object]:
    """Capture HEAD and tracked, untracked, symlink, mode, and submodule dirt."""
    repo_path = Path(repo)
    commit = _run_git(repo_path, "rev-parse", "HEAD").decode("ascii").strip()
    material, dirty = _worktree_material(repo_path)
    submodule_status = _run_git(repo_path, "submodule", "status", "--recursive")
    material += b"submodule-status\0" + submodule_status
    if any(line[:1] in {b"-", b"+", b"U"} for line in submodule_status.splitlines()):
        dirty = True

    initialized = _run_git(
        repo_path,
        "submodule",
        "foreach",
        "--recursive",
        "--quiet",
        'printf "%s\\0" "$displaypath"',
    )
    for raw_path in sorted(path for path in initialized.split(b"\0") if path):
        submodule = repo_path / os.fsdecode(raw_path)
        sub_head = _run_git(submodule, "rev-parse", "HEAD").strip()
        sub_material, sub_dirty = _worktree_material(submodule)
        material += b"submodule\0" + raw_path + b"\0head\0" + sub_head + b"\0" + sub_material
        dirty = dirty or sub_dirty

    dirty_hash: Optional[str] = None
    if dirty:
        dirty_hash = hashlib.sha256(material).hexdigest()
    return {"commit": commit, "dirty": dirty, "dirty_diff_sha256": dirty_hash}


def collect_host_identity() -> dict[str, str]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
    }


def _utc_timestamp(value: Optional[datetime | str]) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ManifestError("created_at_utc must be an ISO-8601 UTC timestamp") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ManifestError("created_at_utc must be a datetime or string")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ManifestError("created_at_utc must be timezone-aware UTC")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ManifestError(f"{name} must be an object")
    return value


def _validate_hash_records(value: object, name: str) -> None:
    records = _require_mapping(value, name)
    for record_name, record in records.items():
        if not isinstance(record_name, str) or not record_name:
            raise ManifestError(f"{name} names must be non-empty strings")
        item = _require_mapping(record, f"{name}.{record_name}")
        if set(item) != {"path", "sha256", "size_bytes"}:
            raise ManifestError(f"{name}.{record_name} has invalid fields")
        if not isinstance(item["path"], str):
            raise ManifestError(f"{name}.{record_name}.path must be a string")
        if not isinstance(item["sha256"], str) or not _SHA256_RE.fullmatch(item["sha256"]):
            raise ManifestError(f"{name}.{record_name}.sha256 is invalid")
        if isinstance(item["size_bytes"], bool) or not isinstance(item["size_bytes"], int) or item["size_bytes"] < 0:
            raise ManifestError(f"{name}.{record_name}.size_bytes must be a non-negative integer")


def validate_manifest(manifest: object) -> None:
    """Validate schema, JSON safety, secret safety, and the content hash."""
    value = _require_mapping(manifest, "manifest")
    _validate_json_value(value)
    manifest_type = value.get("manifest_type")
    expected_fields = _INPUT_FIELDS if manifest_type == "experiment_input" else _COMPLETION_FIELDS
    if manifest_type not in {"experiment_input", "experiment_completion"}:
        raise ManifestError("manifest_type must be experiment_input or experiment_completion")
    missing = expected_fields - set(value)
    extra = set(value) - expected_fields
    if missing:
        raise ManifestError(f"missing required fields: {', '.join(sorted(missing))}")
    if extra:
        raise ManifestError(f"unknown fields: {', '.join(sorted(extra))}")
    if isinstance(value["schema_version"], bool) or value["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION}")
    for field in ("experiment_id", "run_id"):
        if not isinstance(value[field], str) or not value[field]:
            raise ManifestError(f"{field} must be a non-empty string")
    if not isinstance(value["created_at_utc"], str):
        raise ManifestError("created_at_utc must be a string")
    normalized_time = _utc_timestamp(value["created_at_utc"])
    if normalized_time != value["created_at_utc"]:
        raise ManifestError("created_at_utc must use normalized UTC form")
    if not isinstance(value["manifest_sha256"], str) or not _SHA256_RE.fullmatch(value["manifest_sha256"]):
        raise ManifestError("manifest_sha256 is invalid")

    if manifest_type == "experiment_input":
        if not isinstance(value["argv"], list) or not all(isinstance(item, str) for item in value["argv"]):
            raise ManifestError("argv must be an array of strings")
        environment = _require_mapping(value["environment"], "environment")
        if not all(isinstance(key, str) and isinstance(item, str) for key, item in environment.items()):
            raise ManifestError("environment must map strings to strings")
        git = _require_mapping(value["git"], "git")
        if set(git) != {"commit", "dirty", "dirty_diff_sha256"}:
            raise ManifestError("git has invalid fields")
        if not isinstance(git["commit"], str) or not _COMMIT_RE.fullmatch(git["commit"]):
            raise ManifestError("git.commit is invalid")
        if not isinstance(git["dirty"], bool):
            raise ManifestError("git.dirty must be a boolean")
        dirty_hash = git["dirty_diff_sha256"]
        if git["dirty"]:
            if not isinstance(dirty_hash, str) or not _SHA256_RE.fullmatch(dirty_hash):
                raise ManifestError("dirty git state requires dirty_diff_sha256")
        elif dirty_hash is not None:
            raise ManifestError("clean git state must have a null dirty_diff_sha256")
        dependencies = _require_mapping(value["dependencies"], "dependencies")
        if not all(isinstance(key, str) and key and isinstance(item, str) and item for key, item in dependencies.items()):
            raise ManifestError("dependencies must map non-empty names to non-empty revisions")
        _validate_hash_records(value["artifacts"], "artifacts")
        configurations = _require_mapping(value["configurations"], "configurations")
        if set(configurations) != {"model", "engine", "search", "belief"}:
            raise ManifestError("configurations must contain model, engine, search, and belief")
        for name, configuration in configurations.items():
            _require_mapping(configuration, f"configurations.{name}")
        seeds = _require_mapping(value["random_seeds"], "random_seeds")
        for name, seed in seeds.items():
            if not isinstance(name, str) or not name or isinstance(seed, bool) or not isinstance(seed, (int, str)):
                raise ManifestError("random_seeds must map non-empty names to integer or string seeds")
        host = _require_mapping(value["host"], "host")
        if set(host) != _HOST_FIELDS or not all(isinstance(item, str) and item for item in host.values()):
            raise ManifestError("host must contain non-empty host, platform, and Python identity strings")
        _require_mapping(value["resources"], "resources")
        preregistration = _require_mapping(value["preregistration"], "preregistration")
        if set(preregistration) != {"metrics", "gates", "sample_plan"}:
            raise ManifestError("preregistration must contain metrics, gates, and sample_plan")
        if not isinstance(preregistration["metrics"], list):
            raise ManifestError("preregistration.metrics must be an array")
        if not isinstance(preregistration["gates"], list):
            raise ManifestError("preregistration.gates must be an array")
        _require_mapping(preregistration["sample_plan"], "preregistration.sample_plan")
    else:
        frozen_hash = value["frozen_input_manifest_sha256"]
        if not isinstance(frozen_hash, str) or not _SHA256_RE.fullmatch(frozen_hash):
            raise ManifestError("frozen_input_manifest_sha256 is invalid")
        _validate_hash_records(value["result_hashes"], "result_hashes")
        _validate_hash_records(value["artifact_hashes"], "artifact_hashes")
        counts = _require_mapping(value["counts"], "counts")
        for name, count in counts.items():
            if not isinstance(name, str) or not name or isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ManifestError("counts must map non-empty names to non-negative integers")

    if not _constant_time_equal(value["manifest_sha256"], _content_hash(value)):
        raise ManifestError("manifest_sha256 does not match manifest content")


def _constant_time_equal(left: object, right: str) -> bool:
    if not isinstance(left, str):
        return False
    import hmac

    return hmac.compare_digest(left, right)


def _seal(manifest: dict[str, object]) -> dict[str, object]:
    manifest["manifest_sha256"] = _content_hash(manifest)
    validate_manifest(manifest)
    return manifest


def content_addressed_filename(manifest: Mapping[str, object]) -> str:
    """Return a filename whose identity is the validated manifest self-hash."""
    validate_manifest(manifest)
    return f"{manifest['manifest_type']}-{manifest['manifest_sha256']}.json"


def build_experiment_manifest(
    *,
    experiment_id: str,
    run_id: str,
    model_configuration: Mapping[str, object],
    engine_configuration: Mapping[str, object],
    search_configuration: Mapping[str, object],
    belief_configuration: Mapping[str, object],
    random_seeds: Mapping[str, int | str],
    resources: Mapping[str, object],
    metrics: Sequence[object],
    gates: Sequence[object],
    sample_plan: Mapping[str, object],
    artifacts: Optional[Mapping[str, Path | str]] = None,
    dependency_revisions: Optional[Mapping[str, str]] = None,
    argv: Optional[Sequence[str]] = None,
    environment_keys: Iterable[str] = DEFAULT_ENVIRONMENT_KEYS,
    environ: Optional[Mapping[str, str]] = None,
    git_identity: Optional[Mapping[str, object]] = None,
    git_repo: Path | str = ".",
    host_identity: Optional[Mapping[str, str]] = None,
    created_at_utc: Optional[datetime | str] = None,
) -> dict[str, object]:
    """Build and seal the complete input manifest for a not-yet-started run."""
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "experiment_input",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "created_at_utc": _utc_timestamp(created_at_utc),
        "argv": _redact_argv(sys.argv if argv is None else argv),
        "environment": collect_environment(environment_keys, environ),
        "git": dict(collect_git_identity(git_repo) if git_identity is None else git_identity),
        "dependencies": dict(dependency_revisions or {}),
        "artifacts": hash_artifacts(artifacts or {}),
        "configurations": {
            "model": dict(model_configuration),
            "engine": dict(engine_configuration),
            "search": dict(search_configuration),
            "belief": dict(belief_configuration),
        },
        "random_seeds": dict(random_seeds),
        "host": dict(collect_host_identity() if host_identity is None else host_identity),
        "resources": dict(resources),
        "preregistration": {
            "metrics": list(metrics),
            "gates": list(gates),
            "sample_plan": dict(sample_plan),
        },
    }
    return _seal(manifest)


def write_manifest(path: Path | str, manifest: Mapping[str, object], *, overwrite: bool = False) -> None:
    """Durably install canonical JSON with mode 0600.

    The same-directory temporary file and exclusive default installation avoid
    partial writes and accidental replacement.  This is not tamper-proof
    preregistration: publish the manifest/hash to an external append-only
    system or sign it externally when that guarantee is required.
    """
    validate_manifest(manifest)
    output_path = Path(path)
    parent = output_path.parent
    payload = (canonical_json(manifest) + "\n").encode("ascii")
    descriptor = -1
    temporary_path: Optional[Path] = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=parent
        )
        temporary_path = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, output_path)
        else:
            try:
                os.link(temporary_path, output_path, follow_symlinks=False)
            except FileExistsError as exc:
                raise ManifestError(f"manifest already exists: {output_path}") from exc
            temporary_path.unlink()
        temporary_path = None

        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def create_experiment_manifest(path: Path | str, **kwargs: object) -> dict[str, object]:
    """Build and exclusively write an input manifest before a run."""
    manifest = build_experiment_manifest(**kwargs)  # type: ignore[arg-type]
    write_manifest(path, manifest)
    return manifest


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ManifestError(f"non-finite JSON constant: {constant}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError("manifest file must contain a JSON object")
    return value


def build_completion_manifest(
    frozen_manifest: Mapping[str, object],
    *,
    results: Optional[Mapping[str, Path | str]] = None,
    artifacts: Optional[Mapping[str, Path | str]] = None,
    counts: Optional[Mapping[str, int]] = None,
    created_at_utc: Optional[datetime | str] = None,
) -> dict[str, object]:
    """Build a completion record linked to a valid frozen input manifest."""
    validate_manifest(frozen_manifest)
    if frozen_manifest["manifest_type"] != "experiment_input":
        raise ManifestError("completion must reference an experiment_input manifest")
    completion: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "experiment_completion",
        "experiment_id": frozen_manifest["experiment_id"],
        "run_id": frozen_manifest["run_id"],
        "created_at_utc": _utc_timestamp(created_at_utc),
        "frozen_input_manifest_sha256": frozen_manifest["manifest_sha256"],
        "result_hashes": hash_artifacts(results or {}),
        "artifact_hashes": hash_artifacts(artifacts or {}),
        "counts": dict(counts or {}),
    }
    return _seal(completion)


def finalize_manifest(
    frozen_manifest_path: Path | str,
    completion_manifest_path: Path | str,
    *,
    results: Optional[Mapping[str, Path | str]] = None,
    artifacts: Optional[Mapping[str, Path | str]] = None,
    counts: Optional[Mapping[str, int]] = None,
    created_at_utc: Optional[datetime | str] = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Read a frozen input and write a distinct linked completion manifest."""
    frozen_path = Path(frozen_manifest_path)
    completion_path = Path(completion_manifest_path)
    if frozen_path.resolve() == completion_path.resolve():
        raise ManifestError("completion manifest must use a separate path")
    frozen = _load_manifest(frozen_path)
    completion = build_completion_manifest(
        frozen,
        results=results,
        artifacts=artifacts,
        counts=counts,
        created_at_utc=created_at_utc,
    )
    write_manifest(completion_path, completion, overwrite=overwrite)
    return completion
