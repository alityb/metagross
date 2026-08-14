#!/usr/bin/env python3
"""Compare repaired causal-history R1 with frozen legacy stateless inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from srcs.metagross.prior_server import legacy_stateless_trajectory_arrays


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution_metrics(causal: np.ndarray, stateless: np.ndarray) -> dict:
    midpoint = 0.5 * (causal + stateless)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        support = left > 0
        return float(np.sum(left[support] * np.log(left[support] / right[support])))

    return {
        "total_variation": float(0.5 * np.abs(causal - stateless).sum()),
        "jensen_shannon": 0.5 * kl(causal, midpoint) + 0.5 * kl(stateless, midpoint),
        "top1_changed": int(np.argmax(causal) != np.argmax(stateless)),
    }


def _summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": ordered[math.ceil(0.9 * len(ordered)) - 1],
    }


def _history_bin(length: int) -> str:
    if length == 1:
        return "1"
    if length <= 5:
        return "2-5"
    if length <= 10:
        return "6-10"
    if length <= 20:
        return "11-20"
    return "21+"


def compare(
    dump_path: Path,
    checkpoint_root: Path,
    run_name: str,
    checkpoint: int,
    expected_checkpoint_sha256: str,
) -> dict:
    checkpoint_path = (
        checkpoint_root / run_name / "ckpts" / "policy_weights"
        / f"policy_epoch_{checkpoint}.pt"
    ).resolve()
    if _sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    rows = [json.loads(line) for line in dump_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("decision dump is empty")

    receipts: dict[tuple[str, int], int] = {}
    for row in rows:
        for receipt in row["trajectory"].get("action_receipts", []):
            key = (row["tag"], int(receipt["decision_idx"]))
            action_idx = int(receipt["action_idx"])
            if key in receipts and receipts[key] != action_idx:
                raise ValueError("conflicting selected-action receipt")
            receipts[key] = action_idx

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("ACCELERATE_USE_CPU", "true")
    import metamon.rl.pretrained as pretrained

    model = pretrained.LocalFinetunedModel(
        base_model=pretrained.Kakuna,
        amago_ckpt_dir=str(checkpoint_root),
        model_name=run_name,
        default_checkpoint=checkpoint,
    )
    experiment = model.initialize_agent(checkpoint=checkpoint, log=False)
    agent = experiment.policy
    agent.eval()
    device = next(agent.parameters()).device
    metrics = []
    with torch.no_grad():
        for row in rows:
            current = {
                "text_tokens": np.asarray(row["text_tokens"]),
                "numbers": np.asarray(row["numbers"], dtype=np.float32),
                "illegal_actions": np.asarray(row["illegal_actions"], dtype=bool),
            }
            obs_np, rl2_np, time_np = legacy_stateless_trajectory_arrays(current)
            obs = {
                "text_tokens": torch.tensor(
                    obs_np["text_tokens"], dtype=torch.int32, device=device
                ).unsqueeze(0),
                "numbers": torch.nan_to_num(torch.tensor(
                    obs_np["numbers"], dtype=torch.float32, device=device
                ).unsqueeze(0)),
                "illegal_actions": torch.tensor(
                    obs_np["illegal_actions"], device=device
                ).unsqueeze(0),
            }
            embedding, _ = agent.get_state_embedding(
                obs=obs,
                rl2s=torch.tensor(rl2_np, dtype=torch.float32, device=device).unsqueeze(0),
                time_idxs=torch.tensor(time_np, device=device).long().unsqueeze(0),
                hidden_state=None,
            )
            distribution = agent.actor(
                embedding,
                straight_from_obs={
                    key: obs[key][:, : embedding.shape[1]]
                    for key in agent.pass_obs_keys_to_actor
                },
            )
            stateless = distribution.probs[0, -1, -1, :].cpu().numpy()
            illegal = current["illegal_actions"]
            stateless = stateless * (~illegal)
            stateless = stateless / stateless.sum()
            causal = np.asarray(row["probs"], dtype=np.float64)
            item = _distribution_metrics(causal, stateless)
            item["history_length"] = int(row["trajectory"]["observations"])
            receipt = receipts.get((row["tag"], int(row["decision_idx"])))
            if receipt is not None:
                item.update({
                    "selected_action_available": 1,
                    "causal_selected_probability": float(causal[receipt]),
                    "stateless_selected_probability": float(stateless[receipt]),
                    "causal_selected_top1": int(int(np.argmax(causal)) == receipt),
                    "stateless_selected_top1": int(int(np.argmax(stateless)) == receipt),
                })
            else:
                item["selected_action_available"] = 0
            metrics.append(item)

    def aggregate(items: list[dict]) -> dict:
        selected = [item for item in items if item["selected_action_available"]]
        result = {
            "rows": len(items),
            "total_variation": _summarize([item["total_variation"] for item in items]),
            "jensen_shannon": _summarize([item["jensen_shannon"] for item in items]),
            "top1_change_rate": statistics.fmean(item["top1_changed"] for item in items),
            "selected_action_rows": len(selected),
        }
        if selected:
            result["search_selected_action"] = {
                "causal_mean_probability": statistics.fmean(
                    item["causal_selected_probability"] for item in selected
                ),
                "stateless_mean_probability": statistics.fmean(
                    item["stateless_selected_probability"] for item in selected
                ),
                "causal_top1_agreement": statistics.fmean(
                    item["causal_selected_top1"] for item in selected
                ),
                "stateless_top1_agreement": statistics.fmean(
                    item["stateless_selected_top1"] for item in selected
                ),
            }
        return result

    bins = {}
    for label in ("1", "2-5", "6-10", "11-20", "21+"):
        selected = [item for item in metrics if _history_bin(item["history_length"]) == label]
        if selected:
            bins[label] = aggregate(selected)
    return {
        "schema_version": 1,
        "audit": "r1_causal_history_vs_legacy_stateless_v1",
        "decision_dump": str(dump_path.resolve()),
        "decision_dump_sha256": _sha256(dump_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "overall": aggregate(metrics),
        "by_history_length": bins,
        "caveat": (
            "Search-selected-action agreement is descriptive because the selected action "
            "was generated using the causal prior. Battle outcomes are the strength estimand."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision_dump", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        args.decision_dump.expanduser().resolve(),
        args.checkpoint_root.expanduser().resolve(),
        args.run_name,
        args.checkpoint,
        args.checkpoint_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
