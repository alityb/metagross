#!/usr/bin/env python3
"""Verify captured r1 policy snapshots against a frozen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SRC_ROOT.parents[1]
sys.path.insert(0, str(SRC_ROOT))


class PolicySnapshotParityError(ValueError):
    """Raised when a snapshot cannot be reproduced exactly enough."""


def load_snapshots(path: Path) -> list[dict[str, Any]]:
    snapshots = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PolicySnapshotParityError(f"{path}:{line_number}: invalid JSON") from exc
        snapshot = record.get("r1_policy_snapshot", record)
        if not isinstance(snapshot, dict) or snapshot.get("schema") != 3:
            raise PolicySnapshotParityError(
                f"{path}:{line_number}: missing r1 schema-v3 policy snapshot"
            )
        snapshots.append(snapshot)
    if not snapshots:
        raise PolicySnapshotParityError(f"{path}: no policy snapshots")
    return snapshots


def infer_snapshots(agent: Any, snapshots: Sequence[dict[str, Any]], device: Any):
    """Run the exact blank-plus-current stateless policy query used in serving."""
    import torch

    if not snapshots:
        raise PolicySnapshotParityError("no policy snapshots")
    text_rows = [snapshot.get("text_tokens") for snapshot in snapshots]
    number_rows = [snapshot.get("numbers") for snapshot in snapshots]
    illegal_rows = [snapshot.get("illegal_actions") for snapshot in snapshots]
    if any(not isinstance(row, list) or not row for row in text_rows):
        raise PolicySnapshotParityError("invalid text tokens")
    if any(not isinstance(row, list) or not row for row in number_rows):
        raise PolicySnapshotParityError("invalid numeric observations")
    if any(not isinstance(row, list) or len(row) != 13 for row in illegal_rows):
        raise PolicySnapshotParityError("invalid legality masks")
    if len({len(row) for row in text_rows}) != 1 or len({len(row) for row in number_rows}) != 1:
        raise PolicySnapshotParityError("inconsistent observation shapes")

    text_now = np.asarray(text_rows, dtype=np.int32)
    numbers_now = np.nan_to_num(np.asarray(number_rows, dtype=np.float32))
    illegal_now = np.asarray(illegal_rows, dtype=bool)
    text = torch.tensor(
        np.stack([np.zeros_like(text_now), text_now], axis=1),
        dtype=torch.int32,
        device=device,
    )
    numbers = torch.tensor(
        np.stack([np.zeros_like(numbers_now), numbers_now], axis=1),
        dtype=torch.float32,
        device=device,
    )
    illegal = torch.tensor(
        np.stack([np.ones_like(illegal_now), illegal_now], axis=1),
        dtype=torch.bool,
        device=device,
    )
    batch_size = len(snapshots)
    rl2s = torch.zeros((batch_size, 2, 14), device=device)
    time_idxs = (
        torch.arange(2, device=device)
        .long()
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch_size, 2, 1)
    )
    obs = {"text_tokens": text, "numbers": numbers, "illegal_actions": illegal}
    with torch.no_grad():
        embedding, _ = agent.get_state_embedding(
            obs=obs, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None
        )
        distributions = agent.actor(
            embedding,
            straight_from_obs={
                key: obs[key][:, : embedding.shape[1]]
                for key in agent.pass_obs_keys_to_actor
            },
        )
        probabilities = distributions.probs[:, -1, -1, :].cpu().numpy()
    probabilities *= ~illegal_now
    totals = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or np.any(totals <= 0):
        raise PolicySnapshotParityError("checkpoint produced invalid masked probabilities")
    return probabilities / totals


def compare_snapshots(
    actual: Any,
    snapshots: Sequence[dict[str, Any]],
    *,
    absolute_tolerance: float,
) -> dict[str, Any]:
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise PolicySnapshotParityError("absolute tolerance must be finite and nonnegative")
    expected = np.asarray([snapshot.get("probs") for snapshot in snapshots], dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    if expected.shape != (len(snapshots), 13) or actual_array.shape != expected.shape:
        raise PolicySnapshotParityError("captured and inferred probability shapes do not match")
    errors = np.abs(actual_array - expected)
    max_error = float(errors.max())
    mismatched = int(np.any(errors > absolute_tolerance, axis=1).sum())
    report = {
        "schema_version": 1,
        "record_type": "r1_policy_snapshot_parity",
        "snapshot_count": len(snapshots),
        "absolute_tolerance": absolute_tolerance,
        "max_absolute_error": max_error,
        "mismatched_snapshots": mismatched,
        "passed": mismatched == 0,
    }
    if mismatched:
        raise PolicySnapshotParityError(json.dumps(report, sort_keys=True))
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=WORKSPACE_ROOT / "srcs/models")
    parser.add_argument("--run-name", default="randbats_exit_r1")
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--checkpoint-sha256", default=None)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-7)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise PolicySnapshotParityError("batch size must be positive")

    checkpoint_path = (
        args.checkpoint_root
        / args.run_name
        / "ckpts"
        / "policy_weights"
        / f"policy_epoch_{args.checkpoint}.pt"
    )
    if args.checkpoint_sha256 and _sha256(checkpoint_path) != args.checkpoint_sha256.lower():
        raise PolicySnapshotParityError("checkpoint SHA-256 does not match")

    os.environ.setdefault(
        "METAMON_CACHE_DIR", str(WORKSPACE_ROOT / "srcs/runtime/metamon-cache")
    )
    os.environ.setdefault("WANDB_MODE", "disabled")
    import metamon.rl.pretrained as pretrained

    model = pretrained.LocalFinetunedModel(
        base_model=pretrained.Kakuna,
        amago_ckpt_dir=str(args.checkpoint_root),
        model_name=args.run_name,
        default_checkpoint=args.checkpoint,
    )
    experiment = model.initialize_agent(checkpoint=args.checkpoint, log=False)
    agent = experiment.policy
    agent.eval()
    device = next(agent.parameters()).device
    snapshots = load_snapshots(args.input)
    inferred = []
    for offset in range(0, len(snapshots), args.batch_size):
        inferred.append(infer_snapshots(agent, snapshots[offset : offset + args.batch_size], device))
    report = compare_snapshots(
        np.concatenate(inferred), snapshots, absolute_tolerance=args.absolute_tolerance
    )
    report["input"] = str(args.input.resolve())
    report["checkpoint"] = str(checkpoint_path.resolve())
    report["checkpoint_sha256"] = _sha256(checkpoint_path)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
