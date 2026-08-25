#!/usr/bin/env python3
"""Train one direct search-policy student arm from exact schema-v3 roots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from train.search_policy_student import (
    SPLITS,
    batch_indices,
    build_stateless_batch,
    deployment_policy_probs,
    load_search_policy_dataset,
    policy_cross_entropy,
    policy_metrics,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_policy_only(agent) -> list[torch.nn.Parameter]:
    """Freeze every branch not used to produce deployment root priors."""
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    parameters: list[torch.nn.Parameter] = []
    for module in (agent.tstep_encoder, agent.traj_encoder, agent.actor):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            parameters.append(parameter)
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("policy modules unexpectedly share parameter objects")
    return parameters


def atomic_torch_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_agent(base_root: Path, run_name: str, checkpoint: int):
    import gin
    from amago.nets.transformer import VanillaAttention

    try:
        gin.external_configurable(VanillaAttention, module="transformer")
    except ValueError:
        pass
    import metamon.rl.pretrained as pretrained

    model = pretrained.LocalFinetunedModel(
        base_model=pretrained.Kakuna,
        amago_ckpt_dir=str(base_root),
        model_name=run_name,
        default_checkpoint=checkpoint,
    )
    experiment = model.initialize_agent(checkpoint=checkpoint, log=False)
    return experiment.policy


@torch.no_grad()
def evaluate(
    agent,
    dataset,
    split: str,
    arm: str,
    batch_size: int,
    max_records: int | None = None,
) -> dict:
    agent.eval()
    device = next(agent.parameters()).device
    totals = {
        "cross_entropy_nats": 0.0,
        "kl_target_student_nats": 0.0,
        "top1_agreement": 0.0,
    }
    count = 0
    split_indices = dataset.indices(split)
    if max_records is not None:
        split_indices = split_indices[:max_records]
    for indices in batch_indices(split_indices, batch_size):
        obs, rl2s, time_idxs, targets = build_stateless_batch(
            dataset, indices, arm, device
        )
        probs = deployment_policy_probs(agent, obs, rl2s, time_idxs)
        metrics = policy_metrics(probs.float(), targets.float())
        size = len(indices)
        for key, value in metrics.items():
            totals[key] += value * size
        count += size
    return {"records": count} | {key: value / count for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--base-run", default="randbats_exit_r1")
    parser.add_argument("--base-checkpoint", type=int, default=5)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument(
        "--arm", choices=("parent", "action", "visits", "hybrid"), required=True
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument(
        "--max-eval-records",
        type=int,
        default=None,
        help="Deterministic split-prefix cap for infrastructure smokes only.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-min-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--split-seed", type=int, default=20260812)
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.learning_rate <= 0
        or args.eval_interval < 1
        or args.early_stop_patience < 1
        or args.early_stop_min_delta < 0
        or args.early_stop_min_steps < 0
        or (args.max_eval_records is not None and args.max_eval_records < 1)
    ):
        parser.error("steps, batch size, and learning rate must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    base_checkpoint = (
        args.base_root
        / args.base_run
        / "ckpts"
        / "policy_weights"
        / f"policy_epoch_{args.base_checkpoint}.pt"
    )
    if not base_checkpoint.is_file():
        raise ValueError(f"missing base checkpoint: {base_checkpoint}")
    base_digest = sha256(base_checkpoint)
    if base_digest != args.base_sha256.lower():
        raise ValueError(
            f"base checkpoint hash mismatch: expected {args.base_sha256}, found {base_digest}"
        )

    dataset = load_search_policy_dataset(args.dataset, split_seed=args.split_seed)
    print(
        json.dumps(
            {
                "dataset_records": dataset.count,
                "split_records": dataset.split_counts,
                "split_battles": dataset.battle_counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    agent = initialize_agent(args.base_root, args.base_run, args.base_checkpoint)
    device = next(agent.parameters()).device
    parameters = configure_policy_only(agent)
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_indices = dataset.indices("train")
    # The test split is intentionally untouched until a checkpoint has been
    # selected using validation battles only.
    before = {
        "validation": evaluate(
            agent,
            dataset,
            "validation",
            args.arm,
            args.eval_batch_size,
            args.max_eval_records,
        )
    }

    # Keep the forward path numerically aligned with deployment. Gradients still
    # flow in eval mode; this only disables dropout/token augmentation.
    agent.eval()
    step = 0
    epoch = 0
    best_step = 0
    best_validation_kl = float("inf")
    best_state = None
    stale_evaluations = 0
    validation_history = []
    stopped_early = False
    output_dir = args.output_root / args.run_name / "ckpts" / "policy_weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "policy_epoch_1.pt"
    progress_path = args.output_root / args.run_name / "TRAINING_PROGRESS.json"
    training_started = time.monotonic()
    while step < args.steps:
        for indices in batch_indices(
            train_indices,
            args.batch_size,
            generator=generator,
            shuffle=True,
        ):
            optimizer.zero_grad(set_to_none=True)
            obs, rl2s, time_idxs, targets = build_stateless_batch(
                dataset, indices, args.arm, device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                probs = deployment_policy_probs(agent, obs, rl2s, time_idxs)
                loss = policy_cross_entropy(probs.float(), targets.float())
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            step += 1
            if step == 1 or step % 50 == 0 or step == args.steps:
                print(
                    json.dumps(
                        {
                            "arm": args.arm,
                            "epoch": epoch,
                            "step": step,
                            "loss": float(loss.detach()),
                            "grad_norm": float(grad_norm),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if step % args.eval_interval == 0 or step == args.steps:
                validation = evaluate(
                    agent,
                    dataset,
                    "validation",
                    args.arm,
                    args.eval_batch_size,
                    args.max_eval_records,
                )
                validation_history.append({"step": step} | validation)
                validation_kl = validation["kl_target_student_nats"]
                improved = validation_kl < (
                    best_validation_kl - args.early_stop_min_delta
                )
                if improved:
                    best_validation_kl = validation_kl
                    best_step = step
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in agent.state_dict().items()
                    }
                    stale_evaluations = 0
                    atomic_torch_save(best_state, output)
                else:
                    stale_evaluations += 1
                progress_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "arm": args.arm,
                            "run_name": args.run_name,
                            "step": step,
                            "best_step": best_step,
                            "best_validation_kl": best_validation_kl,
                            "stale_evaluations": stale_evaluations,
                            "validation_history": validation_history,
                            "checkpoint": str(output),
                            "checkpoint_sha256": sha256(output),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    json.dumps(
                        {
                            "arm": args.arm,
                            "step": step,
                            "validation": validation,
                            "best_step": best_step,
                            "best_validation_kl": best_validation_kl,
                            "stale_evaluations": stale_evaluations,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if (
                    step >= args.early_stop_min_steps
                    and stale_evaluations >= args.early_stop_patience
                ):
                    stopped_early = True
                    break
            if step >= args.steps:
                break
        epoch += 1
        if stopped_early:
            break
    training_seconds = time.monotonic() - training_started

    if best_state is None:
        raise RuntimeError("training ended without a validation checkpoint")
    agent.load_state_dict(best_state)

    after = {
        split: evaluate(
            agent,
            dataset,
            split,
            args.arm,
            args.eval_batch_size,
            args.max_eval_records,
        )
        for split in ("validation", "test")
    }
    state = {key: value.detach().cpu() for key, value in agent.state_dict().items()}
    atomic_torch_save(state, output)
    if set(state) != set(torch.load(base_checkpoint, map_location="cpu", weights_only=True)):
        raise RuntimeError("candidate state-dict schema differs from r1")
    report = {
        "schema_version": 2,
        "record_type": "search_policy_student",
        "arm": args.arm,
        "run_name": args.run_name,
        "architecture_change": "none",
        "trained_path": ["tstep_encoder", "traj_encoder", "actor"],
        "frozen_path": ["critics", "target_actor", "target_critics", "popart"],
        "deployment_output": "actor.probs[:, -1, -1, :]",
        "base_run": args.base_run,
        "base_checkpoint": args.base_checkpoint,
        "base_sha256": base_digest,
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "dataset_records": dataset.count,
        "split_records": dataset.split_counts,
        "split_battles": dataset.battle_counts,
        "max_eval_records": args.max_eval_records,
        "steps": args.steps,
        "completed_steps": step,
        "best_step": best_step,
        "stopped_early": stopped_early,
        "selection_split": "validation",
        "selection_metric": "kl_target_student_nats",
        "validation_history": validation_history,
        "early_stopping": {
            "eval_interval": args.eval_interval,
            "patience": args.early_stop_patience,
            "min_delta": args.early_stop_min_delta,
            "min_steps": args.early_stop_min_steps,
        },
        "training_seconds": training_seconds,
        "steps_per_second": step / training_seconds,
        "projected_1000_step_minutes": 1000.0 / (step / training_seconds) / 60.0,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "metrics_before": before,
        "metrics_after": after,
        "checkpoint": str(output),
        "checkpoint_sha256": sha256(output),
        "checkpoint_bytes": output.stat().st_size,
        "state_dict_keys": len(state),
        "parameters": sum(tensor.numel() for tensor in state.values()),
        "trainable_policy_parameters": sum(parameter.numel() for parameter in parameters),
    }
    report_path = args.output_root / args.run_name / "SEARCH_POLICY_STUDENT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
