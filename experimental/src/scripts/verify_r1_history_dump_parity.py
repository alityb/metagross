#!/usr/bin/env python3
"""Recompute frozen-R1 outputs from the durable causal-history decision dump."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


class HistoryParityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_session_rows(rows: list[dict]) -> list[tuple[tuple[str, str], list[dict]]]:
    """Keep independent battle histories separate inside a shared dump."""
    grouped: OrderedDict[tuple[str, str], list[dict]] = OrderedDict()
    for row in rows:
        namespace = row.get("namespace")
        tag = row.get("tag")
        if not isinstance(namespace, str) or not isinstance(tag, str) or not tag:
            raise HistoryParityError("dump row has no session identity")
        grouped.setdefault((namespace, tag), []).append(row)
    return list(grouped.items())


def verify(
    dump_path: Path,
    checkpoint_root: Path,
    run_name: str,
    checkpoint: int,
    expected_checkpoint_sha256: str,
    tolerance: float,
) -> dict:
    checkpoint_path = (
        checkpoint_root / run_name / "ckpts" / "policy_weights"
        / f"policy_epoch_{checkpoint}.pt"
    ).resolve()
    if _sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise HistoryParityError("checkpoint SHA-256 mismatch")
    rows = [json.loads(line) for line in dump_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise HistoryParityError("decision dump is empty")

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
    sessions = group_session_rows(rows)
    maximum_difference = 0.0
    mismatched_rows = 0
    with torch.no_grad():
        for _session_identity, session_rows in sessions:
            for row_index, row in enumerate(session_rows):
                trajectory = row["trajectory"]
                time_indices = [int(value) for value in trajectory["time_indices"]]
                if not time_indices or time_indices[-1] != row_index:
                    raise HistoryParityError(
                        "dump row and session-local absolute time index disagree"
                    )
                if time_indices != list(range(time_indices[0], row_index + 1)):
                    raise HistoryParityError("dump trajectory time indices are not contiguous")
                selected_rows = [session_rows[index] for index in time_indices]
                obs = {
                    "text_tokens": torch.tensor(
                        np.stack([item["text_tokens"] for item in selected_rows]),
                        dtype=torch.int32,
                        device=device,
                    ).unsqueeze(0),
                    "numbers": torch.tensor(
                        np.nan_to_num(np.stack([item["numbers"] for item in selected_rows])),
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0),
                    "illegal_actions": torch.tensor(
                        np.stack([item["illegal_actions"] for item in selected_rows]),
                        device=device,
                    ).unsqueeze(0),
                }
                rl2 = torch.tensor(
                    trajectory["rl2"], dtype=torch.float32, device=device
                ).unsqueeze(0)
                times = torch.tensor(
                    np.asarray(time_indices, dtype=np.int64).reshape(-1, 1),
                    device=device,
                ).long().unsqueeze(0)
                embedding, _ = agent.get_state_embedding(
                    obs=obs, rl2s=rl2, time_idxs=times, hidden_state=None
                )
                distribution = agent.actor(
                    embedding,
                    straight_from_obs={
                        key: obs[key][:, : embedding.shape[1]]
                        for key in agent.pass_obs_keys_to_actor
                    },
                )
                probabilities = distribution.probs[0, -1, -1, :].cpu().numpy()
                illegal = np.asarray(row["illegal_actions"], dtype=bool)
                probabilities = probabilities * (~illegal)
                probabilities = probabilities / probabilities.sum()
                expected = np.asarray(row["probs"], dtype=np.float64)
                difference = float(np.max(np.abs(probabilities - expected)))
                maximum_difference = max(maximum_difference, difference)
                mismatched_rows += difference > tolerance
    status = "pass" if mismatched_rows == 0 else "fail"
    return {
        "schema_version": 1,
        "audit": "r1_online_dump_offline_recompute_parity_v1",
        "status": status,
        "decision_dump": str(dump_path.resolve()),
        "decision_dump_sha256": _sha256(dump_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "sessions": len(sessions),
        "rows": len(rows),
        "tolerance": tolerance,
        "maximum_probability_absolute_difference": maximum_difference,
        "mismatched_rows": mismatched_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("decision_dump", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(
        args.decision_dump.expanduser().resolve(),
        args.checkpoint_root.expanduser().resolve(),
        args.run_name,
        args.checkpoint,
        args.checkpoint_sha256,
        args.tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
