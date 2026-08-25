"""Continue frozen r1 on a guarded fresh/legacy/human trajectory mixture.

Intended for the first staged online-RL smoke. Inputs are packaged into immutable
archives, uploaded to a Modal Volume, and then consumed by the H100 learner.

Usage:
  modal run src/scripts/modal_train_online_rl.py \
    --fresh-root runs/online_rl_smoke \
    --legacy-root data/selfplay_parsed_indexed \
    --human-root data/parsed_replays \
    --run-name randbats_online_g1_smoke --steps 200
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import modal


SCRIPT_PATH = Path(__file__).resolve()
IS_LOCAL_CHECKOUT = len(SCRIPT_PATH.parents) > 2 and (SCRIPT_PATH.parents[1] / "train").is_dir()
SRC_ROOT = SCRIPT_PATH.parents[1] if IS_LOCAL_CHECKOUT else Path("/root")
WORKSPACE_ROOT = SRC_ROOT.parents[1] if IS_LOCAL_CHECKOUT else Path("/root")
R1_RUN_NAME = "randbats_exit_r1"
R1_CHECKPOINT = 5
R1_SHA256 = "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
FORMAT = "gen9randombattle"
MIX_WEIGHTS = {"legacy": 0.70, "fresh": 0.20, "human": 0.10}

APP = modal.App("metagross-online-rl-train")
app = APP
VOLUME = modal.Volume.from_name("metagross-online-rl", create_if_missing=True)
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
)
if IS_LOCAL_CHECKOUT:
    IMAGE = IMAGE.add_local_dir(
        WORKSPACE_ROOT / "srcs" / "vendor" / "metamon" / "metamon",
        "/usr/local/lib/python3.11/site-packages/metamon",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )


def trajectory_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("**/*.json.lz4") if path.is_file())


def package_trajectories(root: Path, dataset_name: str) -> tuple[bytes, int]:
    paths = trajectory_paths(root)
    if not paths:
        raise ValueError(f"{dataset_name} dataset has no .lz4 trajectories: {root}")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for path in paths:
            relative = path.relative_to(root)
            # ParsedReplayDataset expects the battle format directly below each
            # configured dataset root. Preserve source parents beneath it to
            # avoid collisions between collection shards.
            relative = Path(*(part for part in relative.parts if part != FORMAT))
            archive.add(path, arcname=str(Path("online_rl") / dataset_name / FORMAT / relative))
    return payload.getvalue(), len(paths)


def package_fresh_sources(manifest_path: Path) -> tuple[bytes, int, dict[str, object]]:
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("record_type") != "online_rl_fresh_sources":
        raise ValueError("invalid fresh source manifest record type")
    sources = source_manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("fresh source manifest must contain sources")
    generations: set[int] = set()
    admitted: list[tuple[int, Path, list[Path]]] = []
    total_fields = (
        "requested_battles", "completed_battles", "learner_wins",
        "learner_losses", "learner_trajectory_count",
    )
    calculated = {key: 0 for key in total_fields}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("generation"), int):
            raise ValueError("fresh source entry is invalid")
        generation = source["generation"]
        if generation in generations:
            raise ValueError(f"duplicate fresh generation: {generation}")
        generations.add(generation)
        root = Path(str(source.get("root", ""))).resolve()
        collection_manifest_path = Path(str(source.get("manifest", ""))).resolve()
        if collection_manifest_path != root / "MANIFEST.json":
            raise ValueError(f"fresh generation {generation} manifest is not rooted in its collection")
        collection = json.loads(collection_manifest_path.read_text(encoding="utf-8"))
        if collection.get("collection_kind") != "fresh" or collection.get("failed_shards"):
            raise ValueError(f"fresh generation {generation} is an arena or failed collection")
        values = {key: int(collection.get(key, -1)) for key in total_fields}
        units = collection.get("chunks", collection.get("shards"))
        if not isinstance(units, list) or not units or any(unit.get("error") for unit in units):
            raise ValueError(f"fresh generation {generation} has missing or failed chunks")
        derived = {key: 0 for key in total_fields}
        derived["requested_battles"] = sum(int(unit.get("requested_battles", 0)) for unit in units)
        for unit in units:
            for phase in unit.get("phases", []):
                for key in total_fields[1:]:
                    derived[key] += int(phase.get(key, 0))
        if values != derived:
            raise ValueError(f"fresh generation {generation} collection chunk totals do not match")
        if any(source.get(key) != value for key, value in values.items()):
            raise ValueError(f"fresh generation {generation} source manifest totals do not match collection")
        if not (
            values["completed_battles"] == values["requested_battles"]
            == values["learner_wins"] + values["learner_losses"]
            == values["learner_trajectory_count"]
        ):
            raise ValueError(f"fresh generation {generation} collection admission totals do not match")
        paths = trajectory_paths(root)
        if len(paths) != values["learner_trajectory_count"]:
            raise ValueError(f"fresh generation {generation} filesystem trajectory total does not match")
        admitted.append((generation, root, paths))
        for key in total_fields:
            calculated[key] += values[key]
    if source_manifest.get("totals") != calculated:
        raise ValueError("fresh source manifest cumulative totals do not match")

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for generation, root, paths in sorted(admitted):
            for path in paths:
                relative = path.relative_to(root)
                relative = Path(*(part for part in relative.parts if part != FORMAT))
                archive.add(
                    path,
                    arcname=str(Path("online_rl") / "fresh" / FORMAT / f"generation_{generation:03d}" / relative),
                )
    return payload.getvalue(), calculated["learner_trajectory_count"], source_manifest


def _validated_archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = Path(member.name)
        if not member.name or path == Path(".") or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member path: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"archive links are not allowed: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unsupported archive member type: {member.name}")
    return members


def safe_extract_archive(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    with tarfile.open(path, mode="r:gz") as archive:
        for member in _validated_archive_members(archive):
            target = destination.joinpath(*Path(member.name).parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"unsafe archive member path: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"could not read archive member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def archive_trajectory_count(path: Path, dataset_name: str) -> int:
    prefix = f"online_rl/{dataset_name}/{FORMAT}/"
    with tarfile.open(path, mode="r:gz") as archive:
        names = [member.name for member in _validated_archive_members(archive) if member.isfile()]
    if not names or any(not name.startswith(prefix) or not name.endswith(".json.lz4") for name in names):
        raise ValueError(f"invalid {dataset_name} trajectory archive: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate paths in {dataset_name} trajectory archive: {path}")
    return len(names)


def _reset_directory(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _rebuild_and_read_index(root: Path) -> tuple[int, int]:
    paths = trajectory_paths(root / FORMAT)
    index_path = root / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename"])
        writer.writerows([[str(path.relative_to(root))] for path in paths])
    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filename"]:
            raise ValueError(f"invalid rebuilt dataset index: {index_path}")
        cached = [row["filename"] for row in reader]
    if len(cached) != len(set(cached)):
        raise ValueError(f"duplicate paths in rebuilt dataset index: {index_path}")
    return len(paths), len(cached)


def _metamon_index_count(root: Path, use_cached_filenames: bool) -> int:
    from metamon.data import MetamonDataset

    dataset = MetamonDataset(
        dset_root=str(root),
        observation_space=None,
        action_space=None,
        reward_function=None,
        formats=[FORMAT],
        verbose=False,
        use_cached_filenames=use_cached_filenames,
    )
    return len(dataset)


def prepare_training_datasets(
    artifact_root: Path,
    anchor_root: Path,
    data_root: Path,
    index_counter=None,
) -> dict[str, dict[str, int]]:
    archives_by_name: dict[str, list[Path]] = {}
    packaged: dict[str, int] = {}
    for name in ("fresh", "legacy", "human"):
        source_root = artifact_root if name == "fresh" else anchor_root
        archives = sorted(source_root.glob(f"{name}*.tar.gz"))
        if not archives:
            raise ValueError(f"no {name} archives in {source_root}")
        archives_by_name[name] = archives
        packaged[name] = sum(archive_trajectory_count(path, name) for path in archives)

    for name in ("fresh", "legacy", "human"):
        _reset_directory(data_root / name)
    for archives in archives_by_name.values():
        for path in archives:
            safe_extract_archive(path, data_root.parent)

    extracted = {
        name: len(trajectory_paths(data_root / name / FORMAT))
        for name in ("fresh", "legacy", "human")
    }
    rebuilt_index: dict[str, int] = {}
    cached_index: dict[str, int] = {}
    loader_rebuilt: dict[str, int] = {}
    loader_cached: dict[str, int] = {}
    for name in ("fresh", "legacy", "human"):
        rebuilt_index[name], cached_index[name] = _rebuild_and_read_index(data_root / name)
        if index_counter is not None:
            loader_rebuilt[name] = index_counter(data_root / name, False)
            loader_cached[name] = index_counter(data_root / name, True)
    counts = {
        "packaged": packaged,
        "extracted": extracted,
        "rebuilt_index": rebuilt_index,
        "cached_index": cached_index,
    }
    for name in ("fresh", "legacy", "human"):
        values = {stage: stage_counts[name] for stage, stage_counts in counts.items()}
        if len(set(values.values())) != 1 or packaged[name] <= 0:
            raise ValueError(f"{name} trajectory admission count mismatch: {values}")
    if index_counter is not None:
        counts.update({"loader_rebuilt": loader_rebuilt, "loader_cached": loader_cached})
        for name in ("fresh", "legacy", "human"):
            if loader_rebuilt[name] <= 0 or loader_rebuilt[name] != loader_cached[name]:
                raise ValueError(
                    f"{name} loader example count mismatch: "
                    f"rebuilt={loader_rebuilt[name]}, cached={loader_cached[name]}"
                )
    return counts


def checkpoint_path(run_dir: Path, checkpoint_index: int, expected_sha256: str) -> Path:
    checkpoint = run_dir / "ckpts" / "policy_weights" / f"policy_epoch_{checkpoint_index}.pt"
    if not checkpoint.is_file():
        raise ValueError(f"base checkpoint is missing: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256.lower():
        raise ValueError("base checkpoint SHA-256 does not match")
    return checkpoint


def package_training_sources() -> bytes:
    files = [
        SRC_ROOT / "scripts" / "run_finetune_variant.py",
        SRC_ROOT / "train" / "finetune_toggles.py",
        SRC_ROOT / "train" / "gins" / "metagross_B_klanchor.gin",
    ]
    for path in files:
        if not path.is_file():
            raise ValueError(f"training source is missing: {path}")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(SRC_ROOT)))
    return payload.getvalue()


def dataset_yaml() -> str:
    return (
        "replay_weight: 0.0\n"
        "custom_replays:\n"
        f"  - dir: /data/online_rl/legacy\n    weight: {MIX_WEIGHTS['legacy']:.2f}\n"
        f"  - dir: /data/online_rl/fresh\n    weight: {MIX_WEIGHTS['fresh']:.2f}\n"
        f"  - dir: /data/online_rl/human\n    weight: {MIX_WEIGHTS['human']:.2f}\n"
        "formats:\n"
        f"  - {FORMAT}\n"
    )


def _train_from_artifacts(
    artifact_root: Path,
    anchor_root: Path,
    run_name: str,
    steps: int,
    batch_size: int,
    counts: dict[str, int],
) -> dict[str, object]:
    import glob
    import subprocess
    import sys

    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    os.environ.update(
        METAMON_CACHE_DIR="/data/metamon_cache",
        HF_HOME="/data/hf_home",
        WANDB_MODE="disabled",
        TORCHDYNAMO_DISABLE="1",
    )
    admission_counts = prepare_training_datasets(
        artifact_root,
        anchor_root,
        Path("/data/online_rl"),
        index_counter=_metamon_index_count,
    )
    for name, expected in counts.items():
        if admission_counts["packaged"][name] != expected:
            raise ValueError(
                f"{name} uploaded trajectory count mismatch: "
                f"expected {expected}, packaged {admission_counts['packaged'][name]}"
            )
    print(json.dumps({"trajectory_admission_counts": admission_counts}, sort_keys=True), flush=True)
    _reset_directory(Path("/data/repo"))
    safe_extract_archive(artifact_root / "sources.tar.gz", Path("/data/repo"))
    base = json.loads((artifact_root / "base.json").read_text(encoding="utf-8"))
    if not isinstance(base.get("run_name"), str) or not isinstance(base.get("checkpoint"), int):
        raise ValueError("invalid base checkpoint metadata")
    checkpoint_dir = Path("/data/base") / base["run_name"] / "ckpts" / "policy_weights"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        artifact_root / "base_policy.pt",
        checkpoint_dir / f"policy_epoch_{base['checkpoint']}.pt",
    )

    # Match the known working CPU-safe Metamon setup used by prior r1 trainers.
    transformer = "/usr/local/lib/python3.11/site-packages/amago/nets/transformer.py"
    with open(transformer, encoding="utf-8") as handle:
        source = handle.read()
    if not source.startswith("import gin"):
        source = "import gin\n" + source
    if "@gin.configurable\nclass VanillaAttention" not in source:
        source = source.replace("class VanillaAttention", "@gin.configurable\nclass VanillaAttention", 1)
    with open(transformer, "w", encoding="utf-8") as handle:
        handle.write(source)

    pretrained = "/usr/local/lib/python3.11/site-packages/metamon/rl/pretrained.py"
    with open(pretrained, encoding="utf-8") as handle:
        source = handle.read()
    if "gin_overrides=base_model.gin_overrides" not in source:
        source = source.replace(
            "            dataset_config=dataset_config,",
            "            dataset_config=dataset_config,\n            gin_overrides=base_model.gin_overrides,",
        )
        with open(pretrained, "w", encoding="utf-8") as handle:
            handle.write(source)

    config_path = "/data/online_rl/dataset.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(dataset_yaml())
    command = [
        sys.executable,
        "/data/repo/scripts/run_finetune_variant.py",
        "--variant", "B_klanchor",
        "--run-name", run_name,
        "--dataset-config", config_path,
        "--save-dir", "/data/online_rl/checkpoints",
        "--epochs", "1",
        "--steps-per-epoch", str(steps),
        "--batch-size", str(batch_size),
        "--dloader-workers", "8",
        "--prev-run-dir", "/data/base",
        "--prev-run-name", base["run_name"],
        "--prev-checkpoint", str(base["checkpoint"]),
        "--reward-function", "BinaryReward",
    ]
    result = subprocess.run(command, env=os.environ | {"PYTHONPATH": "/data/repo"}, text=True)
    if result.returncode:
        raise RuntimeError(f"online-RL continuation failed with exit code {result.returncode}")
    checkpoints = sorted(glob.glob(f"/data/online_rl/checkpoints/{run_name}/**/policy_epoch_*.pt", recursive=True))
    if not checkpoints:
        raise RuntimeError("training completed without publishing a checkpoint")
    report = {
        "schema_version": 1,
        "record_type": "staged_online_rl_training",
        "run_name": run_name,
        "base_run": base["run_name"],
        "base_checkpoint": base["checkpoint"],
        "base_checkpoint_sha256": base["sha256"],
        "variant": "B_klanchor",
        "reward_function": "BinaryReward",
        "steps": steps,
        "batch_size": batch_size,
        "mixture_weights": MIX_WEIGHTS,
        "trajectory_counts": admission_counts,
        "checkpoints": checkpoints,
    }
    report_path = f"/data/online_rl/checkpoints/{run_name}/ONLINE_RL_MANIFEST.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    VOLUME.commit()
    return report


@APP.function(image=IMAGE, gpu="H100", timeout=7200, volumes={"/data": VOLUME})
def train_from_volume(
    artifact_dir: str,
    run_name: str,
    steps: int,
    batch_size: int,
    counts_json: str,
    anchor_artifact_dir: str = "",
) -> dict[str, object]:
    artifact_root = Path("/data") / artifact_dir.strip("/")
    anchor_root = (
        Path("/data") / anchor_artifact_dir.strip("/")
        if anchor_artifact_dir
        else artifact_root
    )
    counts = json.loads(counts_json)
    if (
        not isinstance(counts, dict)
        or not set(counts).issubset({"fresh", "legacy", "human"})
        or not all(isinstance(value, int) and value > 0 for value in counts.values())
    ):
        raise ValueError("counts_json must contain positive fresh/legacy/human counts")
    return _train_from_artifacts(
        artifact_root, anchor_root, run_name, steps, batch_size, counts
    )


@APP.function(image=IMAGE, cpu=4, memory=8192, timeout=1800, volumes={"/data": VOLUME})
def finalize_deployable_checkpoint(run_name: str) -> dict[str, object]:
    """Strip KL-anchor-only modules and enforce the accepted policy schema."""
    import torch

    source = (
        Path("/data/online_rl/checkpoints")
        / run_name
        / "ckpts"
        / "policy_weights"
        / "policy_epoch_0.pt"
    )
    reference_path = Path("/data/online_rl_artifacts") / run_name / "base_policy.pt"
    if not source.is_file() or not reference_path.is_file():
        raise ValueError("training or reference checkpoint is missing")
    candidate = torch.load(source, map_location="cpu", weights_only=True)
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if not isinstance(candidate, dict) or not isinstance(reference, dict):
        raise ValueError("checkpoint payload is not a state dict")
    missing = sorted(set(reference) - set(candidate))
    shape_mismatches = sorted(
        key for key in reference.keys() & candidate.keys() if reference[key].shape != candidate[key].shape
    )
    if missing or shape_mismatches:
        raise ValueError(
            f"candidate policy schema mismatch: missing={missing[:5]} shapes={shape_mismatches[:5]}"
        )
    deployable = {key: candidate[key] for key in reference}
    changed_keys = 0
    squared_delta = 0.0
    squared_reference = 0.0
    for key, tensor in deployable.items():
        baseline = reference[key]
        if not torch.equal(tensor, baseline):
            changed_keys += 1
        if tensor.is_floating_point():
            squared_delta += float(torch.sum((tensor.float() - baseline.float()) ** 2))
            squared_reference += float(torch.sum(baseline.float() ** 2))
    # LocalFinetunedModel treats checkpoint 0 as an uninitialized sentinel, so
    # publish the first deployable continuation at index 1.
    destination = source.with_name("policy_epoch_1.pt")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(deployable, temporary)
    temporary.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    report = {
        "deployable_checkpoint": str(destination),
        "checkpoint_sha256": digest.hexdigest(),
        "reference_keys": len(reference),
        "removed_anchor_keys": len(candidate) - len(reference),
        "changed_keys": changed_keys,
        "relative_l2_delta": (squared_delta / max(squared_reference, 1e-30)) ** 0.5,
        "bytes": destination.stat().st_size,
    }
    manifest_path = Path("/data/online_rl/checkpoints") / run_name / "ONLINE_RL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deployable"] = report
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    VOLUME.commit()
    return report


@APP.function(image=IMAGE, gpu="H100", timeout=1800, volumes={"/data": VOLUME})
def validate_deployable_checkpoint(run_name: str, trajectories: int = 8) -> dict[str, object]:
    """Load through the production model path and compare against frozen r1."""
    os.environ.update(
        METAMON_CACHE_DIR="/data/metamon_cache",
        HF_HOME="/data/hf_home",
        WANDB_MODE="disabled",
        TORCHDYNAMO_DISABLE="1",
    )
    import math
    import torch
    import gin
    from amago.nets.transformer import VanillaAttention

    try:
        gin.external_configurable(VanillaAttention, module="transformer")
    except ValueError:
        pass
    import metamon.rl.pretrained as pretrained
    from metamon.data import ParsedReplayDataset
    from metamon.rl.metamon_to_amago import MetamonAMAGODataset

    if trajectories <= 0:
        raise ValueError("trajectories must be positive")
    model = pretrained.LocalFinetunedModel(
        base_model=pretrained.Kakuna,
        amago_ckpt_dir="/data/online_rl/checkpoints",
        model_name=run_name,
        default_checkpoint=1,
    )
    experiment = model.initialize_agent(checkpoint=1, log=False)
    agent = experiment.policy
    agent.eval()
    device = next(agent.parameters()).device
    dataset = ParsedReplayDataset(
        dset_root="/data/online_rl/human",
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        formats=[FORMAT],
        verbose=False,
    )
    wrapped = MetamonAMAGODataset(parsed_replay_dset=dataset)
    replay_data = [wrapped._process_data(dataset[index]) for index in range(min(trajectories, len(dataset)))]

    def infer(data):
        obs = {key: value.unsqueeze(0).to(device) for key, value in data.obs.items()}
        rl2s = data.rl2s.unsqueeze(0).to(device)
        time_idxs = data.time_idxs.unsqueeze(0).squeeze(-1).to(device)
        with torch.no_grad():
            embedding, _ = agent.get_state_embedding(
                obs=obs, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None
            )
            distribution = agent.actor(
                embedding,
                straight_from_obs={key: obs[key] for key in agent.pass_obs_keys_to_actor},
            )
        return distribution.probs[0, : data.actions.shape[0], -1, :].cpu()

    candidate_probabilities = [infer(data) for data in replay_data]
    reference_path = Path("/data/online_rl_artifacts") / run_name / "base_policy.pt"
    reference = torch.load(reference_path, map_location=device, weights_only=True)
    agent.load_state_dict(reference, strict=True)
    baseline_probabilities = [infer(data) for data in replay_data]

    eps = 1e-8
    entropy_sum = kl_sum = max_probability_sum = illegal_mass_sum = baseline_illegal_mass_sum = 0.0
    top1_changes = timesteps = 0
    for data, candidate, baseline in zip(replay_data, candidate_probabilities, baseline_probabilities):
        illegal = data.obs["illegal_actions"][: candidate.shape[0]].bool()
        illegal_mass_sum += float(candidate.masked_select(illegal).sum())
        baseline_illegal_mass_sum += float(baseline.masked_select(illegal).sum())
        candidate = candidate.masked_fill(illegal, 0.0)
        baseline = baseline.masked_fill(illegal, 0.0)
        candidate = candidate / candidate.sum(-1, keepdim=True).clamp_min(eps)
        baseline = baseline / baseline.sum(-1, keepdim=True).clamp_min(eps)
        entropy_sum += float(-(candidate * candidate.clamp_min(eps).log()).sum())
        kl_sum += float(
            (candidate * (candidate.clamp_min(eps).log() - baseline.clamp_min(eps).log())).sum()
        )
        max_probability_sum += float(candidate.max(-1).values.sum())
        top1_changes += int((candidate.argmax(-1) != baseline.argmax(-1)).sum())
        timesteps += candidate.shape[0]
    report = {
        "checkpoint": str(
            Path("/data/online_rl/checkpoints")
            / run_name
            / "ckpts"
            / "policy_weights"
            / "policy_epoch_1.pt"
        ),
        "trajectories": len(replay_data),
        "timesteps": timesteps,
        "mean_entropy": entropy_sum / timesteps,
        "mean_max_probability": max_probability_sum / timesteps,
        "mean_kl_from_base": kl_sum / timesteps,
        "top1_change_rate": top1_changes / timesteps,
        "mean_illegal_probability_mass": illegal_mass_sum / timesteps,
        "baseline_mean_illegal_probability_mass": baseline_illegal_mass_sum / timesteps,
    }
    numeric = [value for value in report.values() if isinstance(value, float)]
    report["passed"] = (
        timesteps > 0
        and all(math.isfinite(value) for value in numeric)
        and report["mean_illegal_probability_mass"]
        <= report["baseline_mean_illegal_probability_mass"] + 0.005
        and report["mean_entropy"] > 0.05
        and report["mean_max_probability"] < 0.995
    )
    if not report["passed"]:
        raise ValueError(f"deployable checkpoint validation failed: {report}")
    manifest_path = Path("/data/online_rl/checkpoints") / run_name / "ONLINE_RL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation"] = report
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    VOLUME.commit()
    return report


@APP.local_entrypoint()
def main(
    fresh_root: str = "",
    fresh_source_manifest: str = "",
    legacy_root: str = "",
    human_root: str = "",
    legacy_archive_dir: str = "",
    human_archive_dir: str = "",
    run_name: str = "randbats_online_g1_smoke",
    steps: int = 200,
    batch_size: int = 24,
    base_run_dir: str = str(WORKSPACE_ROOT / "srcs" / "models" / R1_RUN_NAME),
    base_run_name: str = R1_RUN_NAME,
    base_checkpoint: int = R1_CHECKPOINT,
    base_checkpoint_sha256: str = R1_SHA256,
    reuse_anchor_artifact: str = "",
) -> None:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    archives: dict[str, bytes] = {}
    archive_files: dict[str, list[Path]] = {}
    counts: dict[str, int] = {}
    if bool(fresh_root) == bool(fresh_source_manifest):
        raise ValueError("set exactly one of --fresh-root or --fresh-source-manifest")
    admitted_sources = None
    if fresh_source_manifest:
        archives["fresh"], counts["fresh"], admitted_sources = package_fresh_sources(
            Path(fresh_source_manifest).resolve()
        )
    else:
        # Retained only for the existing one-generation smoke invocation.
        archives["fresh"], counts["fresh"] = package_trajectories(Path(fresh_root).resolve(), "fresh")
    for name, root, archive_dir in (
        ("legacy", legacy_root, legacy_archive_dir),
        ("human", human_root, human_archive_dir),
    ):
        if reuse_anchor_artifact:
            continue
        if bool(root) == bool(archive_dir):
            raise ValueError(f"set exactly one of --{name}-root or --{name}-archive-dir")
        if root:
            archives[name], counts[name] = package_trajectories(Path(root).resolve(), name)
        else:
            paths = sorted(Path(archive_dir).resolve().glob(f"{name}-*.tar.gz"))
            if not paths:
                raise ValueError(f"no {name} shard archives in {archive_dir}")
            archive_files[name] = paths
            counts[name] = sum(archive_trajectory_count(path, name) for path in paths)
    print(json.dumps({"run_name": run_name, "steps": steps, "counts": counts, "weights": MIX_WEIGHTS}, sort_keys=True))
    artifact_dir = f"online_rl_artifacts/{run_name}"
    checkpoint = checkpoint_path(
        Path(base_run_dir).resolve(), base_checkpoint, base_checkpoint_sha256
    )
    base_metadata = {
        "run_name": base_run_name,
        "checkpoint": base_checkpoint,
        "sha256": base_checkpoint_sha256.lower(),
    }
    with VOLUME.batch_upload(force=True) as batch:
        for name, payload in archives.items():
            batch.put_file(io.BytesIO(payload), f"/{artifact_dir}/{name}.tar.gz")
        for name, paths in archive_files.items():
            for path in paths:
                batch.put_file(str(path), f"/{artifact_dir}/{path.name}")
        batch.put_file(io.BytesIO(package_training_sources()), f"/{artifact_dir}/sources.tar.gz")
        batch.put_file(str(checkpoint), f"/{artifact_dir}/base_policy.pt")
        batch.put_file(
            io.BytesIO((json.dumps(base_metadata, sort_keys=True) + "\n").encode()),
            f"/{artifact_dir}/base.json",
        )
        if admitted_sources is not None:
            batch.put_file(
                io.BytesIO((json.dumps(admitted_sources, indent=2, sort_keys=True) + "\n").encode()),
                f"/{artifact_dir}/FRESH_SOURCES.json",
            )
    report = train_from_volume.remote(
        artifact_dir,
        run_name,
        steps,
        batch_size,
        json.dumps(counts, sort_keys=True),
        reuse_anchor_artifact,
    )
    deployable = finalize_deployable_checkpoint.remote(run_name)
    validation = validate_deployable_checkpoint.remote(run_name, 8)
    print(json.dumps({"training": report, "deployable": deployable, "validation": validation}, indent=2, sort_keys=True))
