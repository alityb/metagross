#!/usr/bin/env python3
"""Measure a policy's agreement with verified MCTS visit-policy targets."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def policy_probs(agent, observation_space, action_space, state, device):
    """Run the deployment policy on one parsed state with its replay legal mask."""
    import torch

    from metamon.interface import UniversalAction

    obs = dict(observation_space.state_to_obs(state))
    illegal = np.ones(13, dtype=bool)
    for action in UniversalAction.maybe_valid_actions(state):
        index = action_space.action_to_agent_output(state, action)
        illegal[index] = False
    obs["illegal_actions"] = illegal
    text = torch.tensor(
        np.stack([np.zeros_like(obs["text_tokens"]), obs["text_tokens"]]),
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    numbers = torch.tensor(
        np.stack([np.zeros_like(obs["numbers"]), obs["numbers"]]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    illegal_actions = torch.tensor(
        np.stack([np.ones(13, dtype=bool), illegal]), device=device
    ).unsqueeze(0)
    rl2s = torch.zeros((1, 2, 14), device=device)
    time_idxs = torch.arange(2, device=device).long().unsqueeze(0).unsqueeze(-1)
    batch = {"text_tokens": text, "numbers": torch.nan_to_num(numbers), "illegal_actions": illegal_actions}
    with torch.no_grad():
        embedding, _ = agent.get_state_embedding(
            obs=batch, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None
        )
        distribution = agent.actor(
            embedding,
            straight_from_obs={key: batch[key][:, : embedding.shape[1]] for key in agent.pass_obs_keys_to_actor},
        )
    return distribution.probs[0, -1, -1].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-root", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--local-run-dir", required=True)
    parser.add_argument("--local-run-name", required=True)
    parser.add_argument("--checkpoint", required=True, type=int)
    parser.add_argument("--max-targets", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("METAMON_CACHE_DIR", str(ROOT / "external" / "metamon_cache"))
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import metamon.rl.pretrained as pretrained
    from train.mcts_policy_distillation import _load_states, _load_trajectory, load_sidecar

    model = pretrained.LocalFinetunedModel(
        base_model=pretrained.Kakuna,
        amago_ckpt_dir=args.local_run_dir,
        model_name=args.local_run_name,
        default_checkpoint=args.checkpoint,
    )
    experiment = model.initialize_agent(checkpoint=args.checkpoint, log=False)
    agent = experiment.policy
    agent.eval()
    device = next(agent.parameters()).device
    targets_by_trajectory = load_sidecar(args.sidecar)

    count = matches = 0
    target_entropy = policy_entropy = cross_entropy = kl = 0.0
    for relative_path, timesteps in sorted(targets_by_trajectory.items()):
        trajectory = _load_trajectory(args.parsed_root / relative_path)
        states = _load_states(trajectory["states"][:-1])
        for timestep, target in sorted(timesteps.items()):
            if args.max_targets and count >= args.max_targets:
                break
            target_array = np.asarray(target, dtype=np.float64)
            probabilities = policy_probs(
                agent, model.observation_space, model.action_space, states[timestep], device
            ).astype(np.float64)
            support = target_array > 0
            target_entropy += float(-(target_array[support] * np.log(target_array[support])).sum())
            policy_entropy += float(-(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])).sum())
            ce = float(-(target_array[support] * np.log(probabilities[support].clip(1e-12))).sum())
            cross_entropy += ce
            kl += ce + float((target_array[support] * np.log(target_array[support])).sum())
            matches += int(np.argmax(target_array) == np.argmax(probabilities))
            count += 1
        if args.max_targets and count >= args.max_targets:
            break
    if not count:
        raise SystemExit("sidecar contains no targets")
    print(
        {
            "targets": count,
            "top1_agreement": matches / count,
            "target_entropy_nats": target_entropy / count,
            "policy_entropy_nats": policy_entropy / count,
            "cross_entropy_nats": cross_entropy / count,
            "mcts_to_policy_kl_nats": kl / count,
            "checkpoint": args.checkpoint,
        }
    )


if __name__ == "__main__":
    main()
