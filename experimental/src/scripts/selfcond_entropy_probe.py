#!/usr/bin/env python3
"""Self-conditioning ablation, part 1a (selfcond_ablation_20260828 prereg).

Offline replay of the frozen 699-decision causal dump under:
  full        — exact online trajectory (SANITY: must reproduce dumped probs)
  trunc-K     — last K steps, absolute time indices kept (the exact semantics
                of METAGROSS_HISTORY_TRUNCATE_STEPS / max_seq_len overflow:
                start = L-K slice, times stay absolute)
  masked      — full length, rl2 own-action one-hots (dims 1..13) zeroed,
                rewards kept (the METAGROSS_MASK_OWN_ACTIONS semantics)
The stateless reference profile comes from the paired stateless dump's
recorded probs directly (no recompute needed).

Output: per-condition mean legal-renormalized prior entropy by turn bucket
(0-9/10-19/20-29/30+), collapse magnitude C = mean(0-9) - mean(30+), the
sanity parity number, and the prereg's frozen threshold verdicts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

TRUNC_KS = (5, 10, 20, 40)
BUCKETS = ((0, 9), (10, 19), (20, 29), (30, 10 ** 9))


def bucket_of(turn: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= turn <= hi:
            return f"{lo}-{hi if hi < 10**9 else 'inf'}"
    raise AssertionError


def entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def group_sessions(rows: list[dict]) -> list[list[dict]]:
    grouped: OrderedDict[tuple[str, str], list[dict]] = OrderedDict()
    for row in rows:
        grouped.setdefault((row["namespace"], row["tag"]), []).append(row)
    return list(grouped.values())


def evaluate(agent, device, session_rows, row_index, row, condition):
    trajectory = row["trajectory"]
    time_indices = [int(v) for v in trajectory["time_indices"]]
    if not time_indices or time_indices[-1] != row_index:
        raise RuntimeError("dump row and session time index disagree")
    selected = [session_rows[i] for i in time_indices]
    rl2 = np.asarray(trajectory["rl2"], dtype=np.float32).copy()

    if condition == "masked":
        rl2[:, 1:] = 0.0
    elif condition.startswith("trunc"):
        k = int(condition.split("-")[1])
        if len(selected) > k:
            selected = selected[-k:]
            rl2 = rl2[-k:]
            time_indices = time_indices[-k:]

    obs = {
        "text_tokens": torch.tensor(
            np.stack([r["text_tokens"] for r in selected]),
            dtype=torch.int32, device=device).unsqueeze(0),
        "numbers": torch.tensor(
            np.nan_to_num(np.stack([r["numbers"] for r in selected])),
            dtype=torch.float32, device=device).unsqueeze(0),
        "illegal_actions": torch.tensor(
            np.stack([r["illegal_actions"] for r in selected]),
            device=device).unsqueeze(0),
    }
    rl2_t = torch.tensor(rl2, dtype=torch.float32, device=device).unsqueeze(0)
    times = torch.tensor(
        np.asarray(time_indices, dtype=np.int64).reshape(-1, 1),
        device=device).long().unsqueeze(0)
    embedding, _ = agent.get_state_embedding(
        obs=obs, rl2s=rl2_t, time_idxs=times, hidden_state=None)
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
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0:
        return None, None
    probabilities = probabilities / total
    return probabilities, entropy(probabilities)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal-dump", type=Path, required=True)
    ap.add_argument("--stateless-dump", type=Path, required=True)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--run-name", default="randbats_exit_r1")
    ap.add_argument("--checkpoint", type=int, default=5)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--per-decision-out", type=Path, default=None)
    args = ap.parse_args()

    ckpt = (args.checkpoint_root / args.run_name / "ckpts" / "policy_weights"
            / f"policy_epoch_{args.checkpoint}.pt")
    if hashlib.sha256(ckpt.read_bytes()).hexdigest() != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA-256 mismatch")

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("ACCELERATE_USE_CPU", "true")
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

    rows = [json.loads(l) for l in args.causal_dump.read_text().splitlines()
            if l.strip()]
    if args.limit_rows:
        rows = rows[: args.limit_rows]
    conditions = ["full"] + [f"trunc-{k}" for k in TRUNC_KS] + ["masked"]
    ent = {c: defaultdict(list) for c in conditions}
    degenerate = {c: 0 for c in conditions}
    sanity_max_diff = 0.0
    done = 0
    with torch.no_grad():
        for session_rows in group_sessions(rows):
            for row_index, row in enumerate(session_rows):
                turn = row.get("battle_turn")
                if turn is None:
                    continue
                for condition in conditions:
                    probs, h = evaluate(agent, device, session_rows,
                                        row_index, row, condition)
                    if h is None:
                        degenerate[condition] += 1
                        continue
                    ent[condition][bucket_of(int(turn))].append(h)
                    if args.per_decision_out:
                        with open(args.per_decision_out, "a") as pd:
                            pd.write(json.dumps({"turn": int(turn),
                                "condition": condition, "entropy": h}) + "\n")
                    if condition == "full":
                        expected = np.asarray(row["probs"], dtype=np.float64)
                        sanity_max_diff = max(
                            sanity_max_diff,
                            float(np.max(np.abs(probs - expected))))
                done += 1
                if done % 50 == 0:
                    print(f"progress {done} decisions "
                          f"(sanity max diff {sanity_max_diff:.2e})",
                          flush=True)

    # Stateless reference from the paired dump's recorded probs.
    sl_bucket = defaultdict(list)
    for line in args.stateless_dump.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        turn = r.get("battle_turn")
        if turn is None:
            continue
        p = np.asarray(r["probs"], dtype=np.float64)
        if p.sum() > 0:
            sl_bucket[bucket_of(int(turn))].append(entropy(p / p.sum()))

    def profile(bucket_map):
        keys = [f"{lo}-{hi if hi < 10**9 else 'inf'}" for lo, hi in BUCKETS]
        means = {k: (float(np.mean(bucket_map[k])) if bucket_map[k] else None)
                 for k in keys}
        counts = {k: len(bucket_map[k]) for k in keys}
        c = (means[keys[0]] - means[keys[-1]]
             if means[keys[0]] is not None and means[keys[-1]] is not None
             else None)
        return {"mean_entropy_by_bucket": means, "n_by_bucket": counts,
                "collapse": c}

    report = {
        "prereg": "selfcond_ablation_20260828 part 1a",
        "causal_dump_sha256": hashlib.sha256(
            args.causal_dump.read_bytes()).hexdigest(),
        "checkpoint_sha256": args.checkpoint_sha256,
        "decisions_evaluated": done,
        "sanity_max_abs_prob_diff": sanity_max_diff,
        "sanity_pass": sanity_max_diff <= 1e-5,
        "degenerate_counts": degenerate,
        "conditions": {c: profile(ent[c]) for c in conditions},
        "stateless_reference": profile(sl_bucket),
    }
    full_c = report["conditions"]["full"]["collapse"]
    masked_c = report["conditions"]["masked"]["collapse"]
    if full_c and masked_c is not None:
        ratio = masked_c / full_c
        report["masked_collapse_ratio"] = ratio
        report["verdict"] = ("H-self (self-conditioning)" if ratio <= 0.33
                             else "H-length" if ratio >= 0.66 else "mixed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("sanity_max_abs_prob_diff", "sanity_pass",
                       "masked_collapse_ratio", "verdict")
                      if k in report}, indent=1))


if __name__ == "__main__":
    main()
