#!/usr/bin/env python3
"""Run the dedicated search-policy student on one Modal H100.

The 275 MiB schema-v3 dataset is uploaded once to the existing Metagross
volume.  The accepted r1 checkpoint is read from that volume by hash, and each
arm writes a production-compatible 642-key checkpoint back to the volume.
"""
from __future__ import annotations

import json
import os
import gzip
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal


SCRIPT_PATH = Path(__file__).resolve()
IS_LOCAL_CHECKOUT = (
    len(SCRIPT_PATH.parents) > 2
    and (SCRIPT_PATH.parents[1] / "train" / "search_policy_student.py").is_file()
)
SRC_ROOT = SCRIPT_PATH.parents[1] if IS_LOCAL_CHECKOUT else Path("/root")
WORKSPACE_ROOT = SRC_ROOT.parents[1] if IS_LOCAL_CHECKOUT else Path("/root")
R1_SHA256 = "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
DATASET_SHA256 = "85ba0de63725ef46bc19546f00faedaa4a95c9c428d733abe02ffc1404666e5b"
VOLUME_DATASET = "/data/search_policy_student/input/mcts_v3_targets.jsonl"
IMAGE_DATASET_GZIP = "/root/input/mcts_v3_targets.jsonl.gz"
DATASET_GZIP_SHA256 = "9b2a30777b205247c0ecf1054a24afcffa9bdd0c991bc35a66c1aca7e5ebf16f"


APP = modal.App("metagross-search-policy-student")
app = APP
VOLUME = modal.Volume.from_name("metagross-online-rl", create_if_missing=False)
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
    IMAGE = (
        IMAGE.add_local_dir(
        WORKSPACE_ROOT / "srcs" / "vendor" / "metamon" / "metamon",
        "/usr/local/lib/python3.11/site-packages/metamon",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
        )
        .add_local_file(
        SRC_ROOT / "train" / "search_policy_student.py",
        "/root/train/search_policy_student.py",
        copy=True,
        )
        .add_local_file(
        SRC_ROOT / "scripts" / "train_search_policy_student.py",
        "/root/scripts/train_search_policy_student.py",
        copy=True,
        )
        .add_local_file(
        WORKSPACE_ROOT
        / "experimental"
        / "runs"
        / "search_policy_student_mcts_v3_targets.jsonl.gz",
        IMAGE_DATASET_GZIP,
        copy=True,
        )
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@APP.function(image=IMAGE, gpu="H100", timeout=4 * 3600, volumes={"/data": VOLUME})
def train(
    arm: str,
    run_name: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_eval_records: int = 0,
    eval_interval: int = 500,
    early_stop_patience: int = 4,
    early_stop_min_steps: int = 2000,
) -> dict:
    if arm not in ("parent", "action", "visits", "hybrid"):
        raise ValueError(f"invalid arm: {arm}")
    archive = Path(IMAGE_DATASET_GZIP)
    if not archive.is_file() or _sha256(archive) != DATASET_GZIP_SHA256:
        raise ValueError("volume dataset archive is missing or has the wrong hash")
    dataset = Path("/tmp/mcts_v3_targets.jsonl")
    with gzip.open(archive, "rb") as source, dataset.open("wb") as output:
        shutil.copyfileobj(source, output)
    if _sha256(dataset) != DATASET_SHA256:
        raise ValueError("decompressed dataset has the wrong frozen hash")
    os.environ.update(
        METAMON_CACHE_DIR="/data/metamon_cache",
        HF_HOME="/data/hf_home",
        WANDB_MODE="disabled",
        TORCHDYNAMO_DISABLE="1",
    )
    command = [
        sys.executable,
        "/root/scripts/train_search_policy_student.py",
        "--dataset", str(dataset),
        "--base-root", "/data/accepted",
        "--base-run", "randbats_exit_r1",
        "--base-checkpoint", "5",
        "--base-sha256", R1_SHA256,
        "--arm", arm,
        "--output-root", "/data/search_policy_student/checkpoints",
        "--run-name", run_name,
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--weight-decay", "0.0",
        "--seed", str(seed),
        "--eval-interval", str(eval_interval),
        "--early-stop-patience", str(early_stop_patience),
        "--early-stop-min-steps", str(early_stop_min_steps),
    ]
    if max_eval_records:
        command.extend(["--max-eval-records", str(max_eval_records)])
    print("Running:", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        env=os.environ | {"PYTHONPATH": "/root"},
        text=True,
    )
    while process.poll() is None:
        time.sleep(15)
        VOLUME.commit()
    if process.returncode:
        raise RuntimeError(f"policy student failed with exit code {process.returncode}")
    report_path = (
        Path("/data/search_policy_student/checkpoints")
        / run_name
        / "SEARCH_POLICY_STUDENT.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    VOLUME.commit()
    return report


@APP.function(image=IMAGE, gpu="H100", timeout=1800, volumes={"/data": VOLUME})
def validate(run_name: str, checkpoint: int = 1) -> dict:
    """Reload a student through the same LocalFinetunedModel used in production."""
    os.environ.update(
        METAMON_CACHE_DIR="/data/metamon_cache",
        HF_HOME="/data/hf_home",
        WANDB_MODE="disabled",
        TORCHDYNAMO_DISABLE="1",
    )
    import gin
    import torch
    from amago.nets.transformer import VanillaAttention

    try:
        gin.external_configurable(VanillaAttention, module="transformer")
    except ValueError:
        pass
    import metamon.rl.pretrained as pretrained

    model = pretrained.LocalFinetunedModel(
        base_model=pretrained.Kakuna,
        amago_ckpt_dir="/data/search_policy_student/checkpoints",
        model_name=run_name,
        default_checkpoint=checkpoint,
    )
    experiment = model.initialize_agent(checkpoint=checkpoint, log=False)
    agent = experiment.policy
    agent.eval()
    device = next(agent.parameters()).device
    dataset_path = Path("/tmp/validation_mcts_v3_targets.jsonl")
    with gzip.open(Path(IMAGE_DATASET_GZIP), "rb") as source, dataset_path.open(
        "wb"
    ) as output:
        shutil.copyfileobj(source, output)
    from train.search_policy_student import (
        build_stateless_batch,
        deployment_policy_probs,
        load_search_policy_dataset,
    )

    dataset = load_search_policy_dataset(dataset_path)
    heldout = dataset.indices("test")[:1]
    obs, rl2s, time_idxs, _ = build_stateless_batch(
        dataset, heldout, "parent", device
    )

    with torch.no_grad():
        probs = deployment_policy_probs(agent, obs, rl2s, time_idxs)
    checkpoint_path = (
        Path("/data/search_policy_student/checkpoints")
        / run_name
        / "ckpts"
        / "policy_weights"
        / f"policy_epoch_{checkpoint}.pt"
    )
    result = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "keys": len(agent.state_dict()),
        "state_dict_parameters": sum(
            tensor.numel() for tensor in agent.state_dict().values()
        ),
        "probability_shape": list(probs.shape),
        "probability_sum": float(probs.sum()),
        "finite": bool(torch.isfinite(probs).all()),
        "illegal_mass": float(
            probs[dataset.illegal_actions[heldout].to(device)].sum()
        ),
    }
    if (
        result["keys"] != 642
        or result["state_dict_parameters"] != 142832563
        or result["probability_shape"] != [1, 13]
        or not result["finite"]
        or abs(result["probability_sum"] - 1.0) > 1e-5
        or result["illegal_mass"] > 1e-6
    ):
        raise RuntimeError(f"production-load validation failed: {result}")
    return result


@APP.function(image=IMAGE, gpu="H100", timeout=1800, volumes={"/data": VOLUME})
def evaluate_matrix(run_names_json: str, checkpoint: int = 1) -> dict:
    """Score every candidate against every target on common held-out roots."""
    run_names = json.loads(run_names_json)
    if not run_names or len(run_names) != len(set(run_names)):
        raise ValueError("run_names must be a non-empty unique list")
    os.environ.update(
        METAMON_CACHE_DIR="/data/metamon_cache",
        HF_HOME="/data/hf_home",
        WANDB_MODE="disabled",
        TORCHDYNAMO_DISABLE="1",
    )
    import gin
    import torch
    from amago.nets.transformer import VanillaAttention

    try:
        gin.external_configurable(VanillaAttention, module="transformer")
    except ValueError:
        pass
    import metamon.rl.pretrained as pretrained
    from train.search_policy_student import (
        batch_indices,
        build_stateless_batch,
        deployment_policy_probs,
        load_search_policy_dataset,
        policy_metrics,
    )

    dataset_path = Path("/tmp/matrix_mcts_v3_targets.jsonl")
    with gzip.open(Path(IMAGE_DATASET_GZIP), "rb") as source, dataset_path.open(
        "wb"
    ) as output:
        shutil.copyfileobj(source, output)
    dataset = load_search_policy_dataset(dataset_path)
    report: dict[str, object] = {
        "schema_version": 1,
        "record_type": "search_policy_student_common_heldout_matrix",
        "dataset_sha256": DATASET_SHA256,
        "runs": {},
    }
    for run_name in run_names:
        model = pretrained.LocalFinetunedModel(
            base_model=pretrained.Kakuna,
            amago_ckpt_dir="/data/search_policy_student/checkpoints",
            model_name=run_name,
            default_checkpoint=checkpoint,
        )
        experiment = model.initialize_agent(checkpoint=checkpoint, log=False)
        agent = experiment.policy
        agent.eval()
        device = next(agent.parameters()).device
        run_report: dict[str, object] = {}
        for split in ("validation", "test"):
            split_report: dict[str, object] = {}
            indices = dataset.indices(split)
            for arm in ("parent", "action", "visits", "hybrid"):
                totals = {
                    "cross_entropy_nats": 0.0,
                    "kl_target_student_nats": 0.0,
                    "top1_agreement": 0.0,
                }
                count = 0
                with torch.no_grad():
                    for batch in batch_indices(indices, 256):
                        obs, rl2s, time_idxs, targets = build_stateless_batch(
                            dataset, batch, arm, device
                        )
                        probs = deployment_policy_probs(
                            agent, obs, rl2s, time_idxs
                        )
                        metrics = policy_metrics(probs.float(), targets.float())
                        size = len(batch)
                        for key, value in metrics.items():
                            totals[key] += value * size
                        count += size
                split_report[arm] = {"records": count} | {
                    key: value / count for key, value in totals.items()
                }
            run_report[split] = split_report
        report["runs"][run_name] = run_report
        del agent, experiment, model
        torch.cuda.empty_cache()
    output_path = Path("/data/search_policy_student/common_heldout_matrix.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    VOLUME.commit()
    return report


@APP.local_entrypoint()
def main(
    dataset: str = str(
        WORKSPACE_ROOT
        / "experimental"
        / "data"
        / "mcts_v3_final"
        / "mcts_v3_targets.jsonl"
    ),
    dataset_gzip: str = str(
        WORKSPACE_ROOT
        / "experimental"
        / "runs"
        / "search_policy_student_mcts_v3_targets.jsonl.gz"
    ),
    arm: str = "parent",
    run_name: str = "search_policy_parent_smoke",
    steps: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-5,
    seed: int = 20260812,
    max_eval_records: int = 1024,
    eval_interval: int = 500,
    early_stop_patience: int = 4,
    early_stop_min_steps: int = 2000,
    validate_checkpoint: bool = True,
) -> None:
    dataset_path = Path(dataset)
    if not dataset_path.is_file() or _sha256(dataset_path) != DATASET_SHA256:
        raise ValueError("local dataset is missing or has the wrong frozen hash")
    gzip_path = Path(dataset_gzip)
    if not gzip_path.is_file() or _sha256(gzip_path) != DATASET_GZIP_SHA256:
        raise ValueError("local dataset gzip is missing or has the wrong frozen hash")
    report = train.remote(
        arm,
        run_name,
        steps,
        batch_size,
        learning_rate,
        seed,
        max_eval_records,
        eval_interval,
        early_stop_patience,
        early_stop_min_steps,
    )
    if validate_checkpoint:
        report["production_load_validation"] = validate.remote(run_name, 1)
    print(json.dumps(report, indent=2, sort_keys=True))
