#!/usr/bin/env python3
"""Launch a schema-v3 MCTS-distillation run on one Modal H100.

Data contract (docs/mcts_policy_distillation.md, schema v3):
  - RL loss: strict learner parsed trajectories (90%) + human anchor (10%),
    exactly as r1 was trained. Replay parsing is only used for trajectories,
    never for MCTS targets.
  - Distillation aux loss: verified records from build_mcts_v3_dataset.py
    (obs dumped live by the prior server, joined fail-closed to MCTS visit
    distributions). Validated locally before upload AND remotely before
    training.

Ablation usage (1k/3k/6k update budgets):
  modal run src/scripts/modal_train_mcts_v3_distillation.py \
    --learner-only-root data/..._learner_only \
    --v3-dataset data/.../mcts_v3_targets.jsonl \
    --human-anchor-root data/parsed_replays \
    --steps-per-epoch 1000
"""
from __future__ import annotations

import io
import json
import math
import os
import tarfile
from pathlib import Path, PurePosixPath

import modal

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "gen9randombattle"
NUM_ACTIONS = 13
R1_RUN_NAME = "randbats_exit_r1"
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl")
    .pip_install("torch", "numpy", "gymnasium<=0.29.1", "gin-config", "wandb", "einops", "tqdm", "lz4", "termcolor", "rich", "huggingface_hub", "datasets", "pandas", "scipy", "ratarmountcore", "poke-env @ git+https://github.com/UT-Austin-RPL/poke-env.git", "amago @ git+https://github.com/UT-Austin-RPL/amago@0974781a9096ff43df1b708312256f96fc2ab127")
    .add_local_dir(ROOT / "external" / "metamon" / "metamon", "/usr/local/lib/python3.11/site-packages/metamon", copy=True, ignore=["__pycache__", "*.pyc"])
)


def _learner_trajectory_paths(root: Path) -> set[str]:
    root = root.resolve()
    paths = {path.relative_to(root).as_posix() for path in root.rglob("*.json.lz4") if path.is_file() and not path.is_symlink()}
    if not paths:
        raise ValueError("learner-only root contains no regular trajectories")
    return paths


def _package_dataset_root(root: Path) -> bytes:
    paths = sorted((root / FORMAT).rglob("*.json.lz4"))
    if not paths:
        raise ValueError(f"dataset root contains no {FORMAT} trajectories")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"invalid dataset trajectory: {path}")
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return payload.getvalue()


def package_train_sources(root: Path = ROOT, sources: tuple[str, ...] = ()) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for relative in sources:
            path = root / relative
            if not path.is_file():
                raise ValueError(f"missing training source: {path}")
            archive.add(path, arcname=relative, recursive=False)
    return payload.getvalue()


def package_r1_checkpoint_archive(root: Path) -> bytes:
    root = root.resolve()
    run_dir = root / R1_RUN_NAME
    checkpoint = run_dir / "ckpts" / "policy_weights" / "policy_epoch_5.pt"
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError("R1 epoch-5 checkpoint is missing")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"R1 archive contains symlink: {path}")
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return payload.getvalue()


def _safe_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts) or not (member.isfile() or member.isdir()):
        raise ValueError(f"unsafe tar member: {member.name!r}")


def _extract_tarball(payload: bytes, destination: str) -> None:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
        for member in archive.getmembers():
            _safe_member(member)
            archive.extract(member, destination)


def _strict_trajectory_lengths(payload: bytes) -> dict[str, int]:
    import lz4.frame
    lengths = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
        for member in archive.getmembers():
            _safe_member(member)
            path = PurePosixPath(member.name)
            if member.isfile() and len(path.parts) >= 2 and path.parts[0] == FORMAT and path.name.endswith(".json.lz4"):
                handle = archive.extractfile(member)
                raw = json.loads(lz4.frame.decompress(handle.read()).decode("utf-8"))
                if not isinstance(raw.get("actions"), list) or len(raw["actions"]) < 2:
                    raise ValueError(f"invalid learner trajectory: {member.name}")
                lengths[member.name] = len(raw["actions"]) - 1
    if not lengths:
        raise ValueError("learner archive contains no trajectories")
    return lengths
APP = modal.App("metagross-mcts-v3-distillation")
app = APP  # Modal CLI discovers the conventional lowercase export.
VOLUME = modal.Volume.from_name("metagross-mcts-v3-distillation", create_if_missing=True)

V3_SCHEMA = 3
V3_TRAIN_SOURCES = (
    "src/scripts/run_finetune_variant.py",
    "src/train/finetune_toggles.py",
    "src/train/mcts_policy_distillation.py",
    "src/train/mcts_v3_distillation.py",
)


def validate_v3_dataset(text: str) -> dict[str, int]:
    """Pure-python mirror of train.mcts_v3_distillation.load_v3_records checks.

    Runs both locally (before upload) and remotely (before training); the
    trainer's own loader validates a third time. Fail-closed everywhere.
    """
    count = 0
    text_len: int | None = None
    numbers_len: int | None = None
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"v3 dataset:{line_number}: invalid JSON") from exc
        if row.get("schema") != V3_SCHEMA:
            raise ValueError(f"v3 dataset:{line_number}: unsupported schema {row.get('schema')!r}")
        tokens = row.get("text_tokens")
        numbers = row.get("numbers")
        illegal = row.get("illegal_actions")
        target = row.get("visit_target_13")
        if not isinstance(tokens, list) or not tokens or not all(isinstance(t, int) for t in tokens):
            raise ValueError(f"v3 dataset:{line_number}: invalid text_tokens")
        if not isinstance(numbers, list) or not numbers:
            raise ValueError(f"v3 dataset:{line_number}: invalid numbers")
        if not isinstance(illegal, list) or len(illegal) != NUM_ACTIONS:
            raise ValueError(f"v3 dataset:{line_number}: invalid illegal_actions")
        if not isinstance(target, list) or len(target) != NUM_ACTIONS:
            raise ValueError(f"v3 dataset:{line_number}: invalid visit_target_13")
        if text_len is None:
            text_len, numbers_len = len(tokens), len(numbers)
        if len(tokens) != text_len or len(numbers) != numbers_len:
            raise ValueError(f"v3 dataset:{line_number}: inconsistent obs shape")
        try:
            masses = [float(value) for value in target]
            floats = [float(value) for value in numbers]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"v3 dataset:{line_number}: non-numeric fields") from exc
        if not all(math.isfinite(m) and m >= 0.0 for m in masses):
            raise ValueError(f"v3 dataset:{line_number}: invalid target mass")
        if not math.isclose(sum(masses), 1.0, abs_tol=1e-4):
            raise ValueError(f"v3 dataset:{line_number}: target mass {sum(masses)}")
        flags = [bool(flag) for flag in illegal]
        if any(mass > 0.0 and flag for mass, flag in zip(masses, flags)):
            raise ValueError(f"v3 dataset:{line_number}: target mass on illegal action")
        if all(flags):
            raise ValueError(f"v3 dataset:{line_number}: no legal actions")
        if not all(math.isfinite(value) for value in floats):
            raise ValueError(f"v3 dataset:{line_number}: non-finite numbers")
        count += 1
    if count == 0:
        raise ValueError("v3 dataset contains no records")
    return {"targets": count, "text_len": int(text_len), "numbers_len": int(numbers_len)}


def package_learner_trajectories(learner_only_root: Path) -> tuple[bytes, int]:
    """Package finalizer learner-only output under the format-root layout."""
    source_paths = _learner_trajectory_paths(learner_only_root)
    root = learner_only_root.resolve()
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for source in sorted(source_paths):
            path = root / source
            archive.add(path, arcname=f"{FORMAT}/{source}", recursive=False)
    return payload.getvalue(), len(source_paths)


@APP.function(image=IMAGE, gpu="H100", timeout=4 * 3600, volumes={"/data": VOLUME})
def train(
    learner_tarball: bytes,
    v3_dataset_jsonl: bytes,
    human_anchor_tarball: bytes,
    r1_checkpoint_archive: bytes,
    train_sources_tarball: bytes,
    run_name: str,
    steps_per_epoch: int,
    mcts_v3_coeff: float,
    mcts_v3_batch_size: int,
    batch_size: int,
) -> list[str]:
    """Run one schema-v3 distillation budget after all artifact checks pass."""
    import glob
    import subprocess
    import sys

    stats = validate_v3_dataset(v3_dataset_jsonl.decode("utf-8"))
    print(f"Verified v3 dataset: {stats}", flush=True)
    trajectories = _strict_trajectory_lengths(learner_tarball)
    print(f"Verified learner trajectories: {len(trajectories)}", flush=True)

    os.environ.update(
        METAMON_CACHE_DIR="/data/metamon_cache",
        HF_HOME="/data/hf_home",
        WANDB_MODE="disabled",
        TORCHDYNAMO_DISABLE="1",
    )
    for directory in ("/data/metamon_cache", "/data/strict_learner", "/data/human_anchor", "/data/r1_checkpoint", "/data/repo"):
        os.makedirs(directory, exist_ok=True)
    _extract_tarball(learner_tarball, "/data/strict_learner")
    _extract_tarball(human_anchor_tarball, "/data/human_anchor")
    _extract_tarball(r1_checkpoint_archive, "/data/r1_checkpoint")
    _extract_tarball(train_sources_tarball, "/data/repo")
    with open("/data/mcts_v3_targets.jsonl", "wb") as handle:
        handle.write(v3_dataset_jsonl)

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
        transformer_source = transformer_source.replace(
            "class VanillaAttention", "@gin.configurable\nclass VanillaAttention", 1
        )
    with open(transformer, "w") as handle:
        handle.write(transformer_source)

    with open("/data/mcts_v3_pilot.yaml", "w") as handle:
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
        "--variant", "A_rating",
        "--run-name", run_name,
        "--dataset-config", "/data/mcts_v3_pilot.yaml",
        "--save-dir", "/data/ckpts",
        "--epochs", "1",
        "--steps-per-epoch", str(steps_per_epoch),
        "--batch-size", str(batch_size),
        "--dloader-workers", "8",
        "--prev-run-dir", "/data/r1_checkpoint",
        "--prev-run-name", R1_RUN_NAME,
        "--prev-checkpoint", "5",
        "--mcts-v3-dataset", "/data/mcts_v3_targets.jsonl",
        "--mcts-v3-coeff", str(mcts_v3_coeff),
        "--mcts-v3-batch-size", str(mcts_v3_batch_size),
    ]
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(
        command, env=os.environ | {"PYTHONPATH": "/data/repo/src"}, capture_output=True, text=True
    )
    if result.returncode:
        print("TRAIN STDOUT:\n" + result.stdout[-12000:], flush=True)
        print("TRAIN STDERR:\n" + result.stderr[-30000:], flush=True)
        raise RuntimeError(f"finetune failed with exit code {result.returncode}")

    checkpoints = sorted(glob.glob(f"/data/ckpts/{run_name}/**/policy_epoch_1.pt", recursive=True))
    if not checkpoints:
        raise RuntimeError("run completed without an epoch-1 policy checkpoint")
    print("Checkpoints:", checkpoints, flush=True)
    VOLUME.commit()
    return checkpoints


@APP.local_entrypoint()
def main(
    learner_only_root: str,
    v3_dataset: str,
    human_anchor_root: str,
    r1_checkpoint_root: str = str(ROOT / "src" / "nets" / "checkpoints" / "randbats_full" / "randbats_exit_r1"),
    steps_per_epoch: int = 1000,
    mcts_v3_coeff: float = 0.1,
    mcts_v3_batch_size: int = 64,
    batch_size: int = 24,
    run_name: str = "",
) -> None:
    """Package finalized local artifacts and invoke the Modal function."""
    learner_root = Path(learner_only_root)
    dataset_path = Path(v3_dataset)
    anchor_root = Path(human_anchor_root)
    checkpoint_root = Path(r1_checkpoint_root)
    if not learner_root.is_dir() or not anchor_root.is_dir() or not checkpoint_root.is_dir():
        raise ValueError("learner-only, human-anchor, and R1 inputs must be directories")
    if not dataset_path.is_file():
        raise ValueError(f"v3 dataset is missing: {dataset_path}")

    dataset_jsonl = dataset_path.read_bytes()
    stats = validate_v3_dataset(dataset_jsonl.decode("utf-8"))
    print(f"Local v3 dataset stats: {stats}", flush=True)

    learner_tarball, trajectory_count = package_learner_trajectories(learner_root)
    print(f"Packaged learner trajectories: {trajectory_count}", flush=True)
    human_anchor = _package_dataset_root(anchor_root)
    r1_checkpoint = package_r1_checkpoint_archive(checkpoint_root)
    resolved_run_name = run_name or f"mcts_v3_distill_{steps_per_epoch}"

    result = train.remote(
        learner_tarball,
        dataset_jsonl,
        human_anchor,
        r1_checkpoint,
        package_train_sources(sources=V3_TRAIN_SOURCES),
        resolved_run_name,
        steps_per_epoch,
        mcts_v3_coeff,
        mcts_v3_batch_size,
        batch_size,
    )
    print(f"Checkpoints: {result}", flush=True)
