#!/usr/bin/env python3
"""Offline screen: action-match accuracy of a policy vs held-out human replays.

Primary Phase-1 screening metric: top-1 / top-3 agreement with HIGH-Elo human
actions on held-out games (split by game, never trained on). Legal-action
masking applied (the env masks illegal actions at play time, so ranking among
legal actions is the fair comparison). Steps where the human action could not
be reconstructed (missing_action_mask) are excluded.

Run in .venv-metamon (CPU ok, ~1-3 s/game for the 142M model).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="Kakuna")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--local-run-dir", default=None)
    parser.add_argument("--local-run-name", default=None)
    parser.add_argument("--local-base-model", default="Kakuna")
    parser.add_argument("--heldout-dir", default="data/parsed_replays_heldout")
    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--min-rating", type=int, default=2000,
                        help="High-Elo subset threshold (filename rating field)")
    parser.add_argument("--max-games", type=int, default=200)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--band-prompt", default=None,
                        help="Rating-band word (e.g. <gen1ubers>) to inject at text "
                             "position 0 — required when screening Toggle-A variants")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.environ.setdefault("METAMON_CACHE_DIR",
                          str(Path(__file__).resolve().parents[1] / "external" / "metamon_cache"))
    os.environ.setdefault("WANDB_MODE", "disabled")

    import numpy as np
    import torch

    from metamon.data import ParsedReplayDataset
    from metamon.rl.metamon_to_amago import MetamonAMAGODataset
    import metamon.rl.pretrained as _pt

    if args.local_run_dir:
        model = _pt.LocalFinetunedModel(
            base_model=getattr(_pt, args.local_base_model),
            amago_ckpt_dir=args.local_run_dir,
            model_name=args.local_run_name,
            default_checkpoint=args.checkpoint,
        )
        model_label = f"local:{args.local_run_name}@ckpt{args.checkpoint}"
    else:
        model = _pt.get_pretrained_model(args.agent)
        model_label = f"{args.agent}@ckpt{args.checkpoint or model.default_checkpoint}"
    print(f"SCREEN loading {model_label}", flush=True)

    experiment = model.initialize_agent(checkpoint=args.checkpoint, log=False)
    agent = experiment.policy
    agent.eval()

    dset = ParsedReplayDataset(
        dset_root=args.heldout_dir,
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        formats=[args.battle_format],
        min_rating=args.min_rating,
        verbose=False,
    )
    amago_dset = MetamonAMAGODataset(parsed_replay_dset=dset)
    n = len(dset)
    print(f"SCREEN heldout games (rating>={args.min_rating}): {n}", flush=True)
    if n == 0:
        print("FATAL: no held-out games matched", flush=True)
        sys.exit(2)

    band_token_id = None
    if args.band_prompt:
        band_token_id = int(model.observation_space.tokenizer.tokenize(args.band_prompt)[0])
        print(f"SCREEN band prompt {args.band_prompt} -> token {band_token_id}", flush=True)

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(n)[: args.max_games]

    top1_hits = top3_hits = steps = games = 0
    with torch.no_grad():
        for count, idx in enumerate(order, 1):
            try:
                rl_data = amago_dset._process_data(dset[int(idx)])
            except Exception as e:  # noqa: BLE001
                print(f"WARN skip idx={idx}: {type(e).__name__} {e}", flush=True)
                continue
            T = rl_data.actions.shape[0]
            if T < 1:
                continue
            if band_token_id is not None:
                rl_data.obs["text_tokens"][:, 0] = band_token_id
            obs = {k: v.unsqueeze(0) for k, v in rl_data.obs.items()}
            rl2s = rl_data.rl2s.unsqueeze(0)
            time_idxs = rl_data.time_idxs.unsqueeze(0).squeeze(-1)
            traj_emb, _ = agent.get_state_embedding(
                obs=obs, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None
            )
            dists = agent.actor(
                traj_emb,
                straight_from_obs={k: obs[k] for k in agent.pass_obs_keys_to_actor},
            )
            probs = dists.probs[0, :, -1, :]  # [T+1, A] at inference gamma
            probs = probs[:T]
            # mask illegal actions, renormalize
            illegal = rl_data.obs["illegal_actions"][:T].bool()
            probs = probs.masked_fill(illegal, 0.0)
            human = rl_data.actions.argmax(-1)  # [T]
            known = ~rl_data.obs["missing_action_mask"][:T, 0].bool()
            if known.sum() == 0:
                continue
            ranked = probs.argsort(dim=-1, descending=True)
            t1 = ranked[:, 0] == human
            t3 = (ranked[:, :3] == human.unsqueeze(-1)).any(-1)
            top1_hits += int(t1[known].sum())
            top3_hits += int(t3[known].sum())
            steps += int(known.sum())
            games += 1
            if count % 25 == 0:
                print(f"PROGRESS games={games} steps={steps} "
                      f"top1={top1_hits/max(steps,1):.4f} top3={top3_hits/max(steps,1):.4f}",
                      flush=True)

    result = {
        "model": model_label,
        "heldout_dir": args.heldout_dir,
        "min_rating": args.min_rating,
        "games": games,
        "steps": steps,
        "top1": round(top1_hits / max(steps, 1), 4),
        "top3": round(top3_hits / max(steps, 1), 4),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    print("SCREEN RESULT " + json.dumps(result, sort_keys=True), flush=True)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
