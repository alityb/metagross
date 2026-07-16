#!/usr/bin/env python3
"""Launch the schema-v2 MCTS-distillation pilot on one Modal H100."""
from __future__ import annotations

import io
import json
import math
import os
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import modal


ROOT = Path(__file__).resolve().parents[2]
APP = modal.App("metagross-mcts-distillation-pilot")
app = APP  # Modal CLI discovers the conventional lowercase export.
VOLUME = modal.Volume.from_name("metagross-mcts-distillation-pilot", create_if_missing=True)

RUN_NAME = "mcts_schema_v2_distillation_pilot"
FORMAT = "gen9randombattle"
NUM_ACTIONS = 13
R1_RUN_NAME = "randbats_exit_r1"
TRAIN_SOURCES = (
    "src/scripts/run_finetune_variant.py",
    "src/train/finetune_toggles.py",
    "src/train/mcts_policy_distillation.py",
)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl")
    .pip_install(
        "torch",
        "numpy",
        "gymnasium<=0.29.1",
        "gin-config",
        "wandb",
        "einops",
        "tqdm",
        "lz4",
        "termcolor",
        "rich",
        "huggingface_hub",
        "datasets",
        "pandas",
        "scipy",
        "ratarmountcore",
        "poke-env @ git+https://github.com/UT-Austin-RPL/poke-env.git",
        "amago @ git+https://github.com/UT-Austin-RPL/amago@0974781a9096ff43df1b708312256f96fc2ab127",
    )
    .add_local_dir(
        ROOT / "external" / "metamon" / "metamon",
        "/usr/local/lib/python3.11/site-packages/metamon",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )
)


def package_train_sources(root: Path = ROOT, sources: tuple[str, ...] = TRAIN_SOURCES) -> bytes:
    """Package exactly the repository modules required by the pilot runner."""
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for relative in sources:
            source = root / relative
            if not source.is_file():
                raise ValueError(f"required training source is missing: {source}")
            archive.add(source, arcname=relative)
    return payload.getvalue()


def _learner_trajectory_paths(root: Path) -> set[str]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"learner-only root is missing: {root}")
    trajectories: set[str] = set()
    for path in sorted(root.rglob("*.json.lz4")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"learner trajectory must be a regular file: {path}")
        relative = path.relative_to(root).as_posix()
        trajectories.add(relative)
    if not trajectories:
        raise ValueError(f"learner-only root contains no trajectories: {root}")
    return trajectories


def validate_sidecar_paths(trajectory_paths: set[str], sidecar_text: str) -> dict[str, int]:
    """Require the sidecar to name every locally packaged trajectory exactly once or more."""
    targets: dict[str, set[int]] = defaultdict(set)
    for line_number, line in enumerate(sidecar_text.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"sidecar:{line_number}: invalid JSON") from exc
        trajectory = row.get("trajectory")
        timestep = row.get("timestep")
        path = PurePosixPath(trajectory) if isinstance(trajectory, str) else None
        if (
            row.get("schema_version") != 1
            or path is None
            or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)
            or str(path) not in trajectory_paths
            or not isinstance(timestep, int)
            or isinstance(timestep, bool)
            or timestep < 0
        ):
            raise ValueError(f"sidecar:{line_number}: invalid or unknown target path")
        if timestep in targets[str(path)]:
            raise ValueError(f"sidecar:{line_number}: duplicate target timestep")
        targets[str(path)].add(timestep)
    if set(targets) != trajectory_paths:
        raise ValueError("sidecar paths do not cover every learner trajectory")
    return {"trajectories": len(trajectory_paths), "targets": sum(map(len, targets.values()))}


def _package_dataset_root(root: Path) -> bytes:
    """Package a local custom dataset while preserving its format-root layout."""
    root = root.resolve()
    format_root = root / FORMAT
    if not format_root.is_dir():
        raise ValueError(f"dataset root must contain {FORMAT}/: {root}")
    trajectories = sorted(format_root.rglob("*.json.lz4"))
    if not trajectories:
        raise ValueError(f"dataset root contains no {FORMAT} trajectories: {root}")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for path in trajectories:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"dataset trajectory must be a regular file: {path}")
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return payload.getvalue()


def package_strict_learner(
    learner_only_root: Path,
    verified_sidecar: Path,
) -> tuple[bytes, bytes, dict[str, int]]:
    """Package nested finalizer output and remap sidecar paths to its archive."""
    source_paths = _learner_trajectory_paths(learner_only_root)
    sidecar_text = verified_sidecar.read_text(encoding="utf-8")
    validate_sidecar_paths(source_paths, sidecar_text)
    path_map = {
        source: (PurePosixPath(FORMAT) / PurePosixPath(source)).as_posix()
        for source in source_paths
    }
    rows = []
    for line in sidecar_text.splitlines():
        row = json.loads(line)
        row["trajectory"] = path_map[row["trajectory"]]
        rows.append(row)
    packaged_sidecar = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    coverage = validate_sidecar_paths(set(path_map.values()), packaged_sidecar.decode("utf-8"))

    root = learner_only_root.resolve()
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for source in sorted(source_paths):
            path = root / source
            archive.add(path, arcname=path_map[source], recursive=False)
    strict_tarball = payload.getvalue()
    return strict_tarball, packaged_sidecar, coverage


def package_r1_checkpoint_archive(r1_checkpoint_root: Path) -> bytes:
    """Package the actual randbats_exit_r1 run directory used by finetuning."""
    root = r1_checkpoint_root.resolve()
    run_dir = root / R1_RUN_NAME
    epoch_five = run_dir / "ckpts" / "policy_weights" / "policy_epoch_5.pt"
    if not epoch_five.is_file() or epoch_five.is_symlink():
        raise ValueError(f"R1 epoch-5 checkpoint is missing: {epoch_five}")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"R1 archive cannot contain symlinks: {path}")
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return payload.getvalue()


def _safe_member_name(member: tarfile.TarInfo) -> PurePosixPath:
    path = PurePosixPath(member.name)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe tar member: {member.name!r}")
    if not (member.isfile() or member.isdir()):
        raise ValueError(f"tar member must be a regular file or directory: {member.name!r}")
    return path


def _strict_trajectory_lengths(tarball: bytes) -> dict[str, int]:
    """Read strict learner trajectories below the archive's format root."""
    import lz4.frame

    trajectories: dict[str, int] = {}
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:*") as archive:
        for member in archive.getmembers():
            path = _safe_member_name(member)
            if member.isdir():
                if path.parts[0] != FORMAT:
                    raise ValueError(
                        "strict learner archive must contain only gen9randombattle/**/*.json.lz4 trajectories"
                    )
                continue
            if len(path.parts) < 2 or path.parts[0] != FORMAT or not path.name.endswith(".json.lz4"):
                raise ValueError(
                    "strict learner archive must contain only gen9randombattle/**/*.json.lz4 trajectories"
                )
            if member.name in trajectories:
                raise ValueError(f"duplicate strict learner trajectory: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read strict learner trajectory: {member.name}")
            try:
                raw = json.loads(lz4.frame.decompress(handle.read()).decode("utf-8"))
                actions = raw["actions"]
            except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as exc:
                raise ValueError(f"invalid strict learner trajectory: {member.name}") from exc
            if not isinstance(actions, list) or len(actions) < 2:
                raise ValueError(f"strict learner trajectory has no decision steps: {member.name}")
            trajectories[member.name] = len(actions) - 1
    if not trajectories:
        raise ValueError("strict learner archive contains no trajectories")
    return trajectories


def validate_sidecar_coverage(trajectory_lengths: dict[str, int], sidecar_text: str) -> dict[str, int]:
    """Require one valid MCTS target for every decision in every trajectory."""
    targets: dict[str, set[int]] = defaultdict(set)
    for line_number, line in enumerate(sidecar_text.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"sidecar:{line_number}: invalid JSON") from exc
        trajectory = row.get("trajectory")
        timestep = row.get("timestep")
        target = row.get("target")
        path = PurePosixPath(trajectory) if isinstance(trajectory, str) else None
        if (
            row.get("schema_version") != 1
            or path is None
            or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)
            or str(path) not in trajectory_lengths
            or not isinstance(timestep, int)
            or isinstance(timestep, bool)
            or timestep < 0
            or not isinstance(target, list)
            or len(target) != NUM_ACTIONS
        ):
            raise ValueError(f"sidecar:{line_number}: invalid or unknown target")
        try:
            masses = [float(value) for value in target]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"sidecar:{line_number}: non-numeric target") from exc
        if not all(math.isfinite(value) and value >= 0 for value in masses) or not math.isclose(
            sum(masses), 1.0, abs_tol=1e-6
        ):
            raise ValueError(f"sidecar:{line_number}: invalid target distribution")
        if timestep in targets[str(path)]:
            raise ValueError(f"sidecar:{line_number}: duplicate target timestep")
        targets[str(path)].add(timestep)

    for trajectory, length in sorted(trajectory_lengths.items()):
        expected = set(range(length))
        if targets.get(trajectory) != expected:
            raise ValueError(f"sidecar coverage is incomplete for {trajectory}")
    return {"trajectories": len(trajectory_lengths), "targets": sum(map(len, targets.values()))}


def verify_archive_sidecar_coverage(strict_tarball: bytes, sidecar_jsonl: bytes) -> dict[str, int]:
    """Validate the uploaded strict archive and its verified sidecar together."""
    try:
        sidecar_text = sidecar_jsonl.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("sidecar must be UTF-8 JSONL") from exc
    return validate_sidecar_coverage(_strict_trajectory_lengths(strict_tarball), sidecar_text)


def _extract_tarball(tarball: bytes, destination: str) -> None:
    """Extract regular archive members only after rejecting traversal and links."""
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:*") as archive:
        for member in archive.getmembers():
            _safe_member_name(member)
            archive.extract(member, destination)


@APP.function(image=IMAGE, gpu="H100", timeout=3600, volumes={"/data": VOLUME})
def train(
    strict_learner_tarball: bytes,
    sidecar_jsonl: bytes,
    human_anchor_tarball: bytes,
    r1_checkpoint_archive: bytes,
    train_sources_tarball: bytes,
) -> list[str]:
    """Run the fixed schema-v2 pilot after all artifact checks have passed."""
    import glob
    import subprocess
    import sys

    coverage = verify_archive_sidecar_coverage(strict_learner_tarball, sidecar_jsonl)
    print(f"Verified sidecar coverage: {coverage}", flush=True)

    os.environ.update(
        METAMON_CACHE_DIR="/data/metamon_cache",
        HF_HOME="/data/hf_home",
        WANDB_MODE="disabled",
        TORCHDYNAMO_DISABLE="1",
    )
    for directory in ("/data/metamon_cache", "/data/strict_learner", "/data/human_anchor", "/data/r1_checkpoint", "/data/repo"):
        os.makedirs(directory, exist_ok=True)
    _extract_tarball(strict_learner_tarball, "/data/strict_learner")
    _extract_tarball(human_anchor_tarball, "/data/human_anchor")
    _extract_tarball(r1_checkpoint_archive, "/data/r1_checkpoint")
    _extract_tarball(train_sources_tarball, "/data/repo")
    with open("/data/mcts_policy_targets.jsonl", "wb") as handle:
        handle.write(sidecar_jsonl)

    r1_epoch_five = f"/data/r1_checkpoint/{R1_RUN_NAME}/ckpts/policy_weights/policy_epoch_5.pt"
    if not os.path.isfile(r1_epoch_five):
        raise ValueError(f"R1 checkpoint archive is missing {R1_RUN_NAME} epoch 5")
    if not os.path.isdir(f"/data/strict_learner/{FORMAT}") or not os.path.isdir(f"/data/human_anchor/{FORMAT}"):
        raise ValueError("strict learner and human anchor archives must each expand to gen9randombattle/")

    transformer = "/usr/local/lib/python3.11/site-packages/amago/nets/transformer.py"
    with open(transformer) as handle:
        transformer_source = handle.read()
    if not transformer_source.startswith("import gin"):
        transformer_source = "import gin\n" + transformer_source
    if "@gin.configurable\nclass VanillaAttention" not in transformer_source:
        transformer_source = transformer_source.replace("class VanillaAttention", "@gin.configurable\nclass VanillaAttention", 1)
    with open(transformer, "w") as handle:
        handle.write(transformer_source)

    with open("/data/mcts_schema_v2_pilot.yaml", "w") as handle:
        handle.write(
            "replay_weight: 0.0\n"
            "custom_replays:\n"
            "  - dir: /data/strict_learner\n"
            "    weight: 0.90\n"
            "  - dir: /data/human_anchor\n"
            "    weight: 0.10\n"
            "formats:\n"
            "  - gen9randombattle\n"
        )

    command = [
        sys.executable,
        "/data/repo/src/scripts/run_finetune_variant.py",
        "--variant",
        "A_rating",
        "--run-name",
        RUN_NAME,
        "--dataset-config",
        "/data/mcts_schema_v2_pilot.yaml",
        "--save-dir",
        "/data/ckpts",
        "--epochs",
        "1",
        "--steps-per-epoch",
        "1000",
        "--batch-size",
        "24",
        "--dloader-workers",
        "8",
        "--prev-run-dir",
        "/data/r1_checkpoint",
        "--prev-run-name",
        R1_RUN_NAME,
        "--prev-checkpoint",
        "5",
        "--mcts-policy-sidecar",
        "/data/mcts_policy_targets.jsonl",
        "--mcts-policy-coeff",
        "0.1",
    ]
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, env=os.environ | {"PYTHONPATH": "/data/repo/src"}, capture_output=True, text=True)
    if result.returncode:
        print("TRAIN STDOUT:\n" + result.stdout[-12000:], flush=True)
        print("TRAIN STDERR:\n" + result.stderr[-30000:], flush=True)
        raise RuntimeError(f"finetune failed with exit code {result.returncode}")

    checkpoints = sorted(glob.glob(f"/data/ckpts/{RUN_NAME}/**/policy_epoch_1.pt", recursive=True))
    if not checkpoints:
        raise RuntimeError("pilot completed without an epoch-1 policy checkpoint")
    print("Pilot checkpoints:", checkpoints, flush=True)
    VOLUME.commit()
    return checkpoints


@APP.local_entrypoint()
def main(
    learner_only_root: str,
    sidecar: str,
    human_anchor_root: str,
    r1_checkpoint_root: str = str(ROOT / "src" / "nets" / "checkpoints" / "randbats_full" / "randbats_exit_r1"),
) -> None:
    """Package finalized local artifacts and invoke the Modal function."""
    paths = {
        "learner-only root": Path(learner_only_root),
        "sidecar": Path(sidecar),
        "human anchor root": Path(human_anchor_root),
        "R1 checkpoint root": Path(r1_checkpoint_root),
    }
    if not paths["learner-only root"].is_dir() or not paths["human anchor root"].is_dir() or not paths["R1 checkpoint root"].is_dir():
        raise ValueError("learner-only, human-anchor, and R1 inputs must be directories")
    if not paths["sidecar"].is_file():
        raise ValueError(f"sidecar is missing: {paths['sidecar']}")
    strict, sidecar_jsonl, coverage = package_strict_learner(paths["learner-only root"], paths["sidecar"])
    human_anchor = _package_dataset_root(paths["human anchor root"])
    r1_checkpoint = package_r1_checkpoint_archive(paths["R1 checkpoint root"])
    print(f"Local sidecar coverage: {coverage}", flush=True)
    result = train.remote(
        strict,
        sidecar_jsonl,
        human_anchor,
        r1_checkpoint,
        package_train_sources(),
    )
    print(f"Pilot checkpoints: {result}", flush=True)
