#!/usr/bin/env python3
"""True same-state control: evaluate the STATELESS serving input on the
causal dump's exact decision states (legacy 2-step input: zero-step +
current obs, rl2 zeros, times [0,1]) so causal-vs-stateless comparisons
are state-paired. Dumps per-decision stateless probs for confound checks."""
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np, torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE_ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--causal-dump", type=Path, required=True)
ap.add_argument("--checkpoint-root", type=Path, required=True)
ap.add_argument("--checkpoint-sha256", required=True)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()

ckpt = args.checkpoint_root / "randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"
assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == args.checkpoint_sha256
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("ACCELERATE_USE_CPU", "true")
import metamon.rl.pretrained as pretrained
model = pretrained.LocalFinetunedModel(base_model=pretrained.Kakuna,
    amago_ckpt_dir=str(args.checkpoint_root), model_name="randbats_exit_r1",
    default_checkpoint=5)
agent = model.initialize_agent(checkpoint=5, log=False).policy
agent.eval()
device = next(agent.parameters()).device

out = open(args.output, "w")
done = 0
with torch.no_grad():
    for line in args.causal_dump.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if row.get("battle_turn") is None: continue
        text = np.asarray(row["text_tokens"]); nums = np.nan_to_num(np.asarray(row["numbers"]))
        illegal = np.asarray(row["illegal_actions"], bool)
        obs = {"text_tokens": torch.tensor(np.stack([np.zeros_like(text), text]), dtype=torch.int32, device=device).unsqueeze(0),
               "numbers": torch.tensor(np.stack([np.zeros_like(nums), nums]), dtype=torch.float32, device=device).unsqueeze(0),
               "illegal_actions": torch.tensor(np.stack([np.ones(13, bool), illegal]), device=device).unsqueeze(0)}
        rl2 = torch.zeros((1, 2, 14), dtype=torch.float32, device=device)
        times = torch.tensor(np.arange(2, dtype=np.int64).reshape(-1, 1), device=device).long().unsqueeze(0)
        emb, _ = agent.get_state_embedding(obs=obs, rl2s=rl2, time_idxs=times, hidden_state=None)
        dist = agent.actor(emb, straight_from_obs={k: obs[k][:, :emb.shape[1]] for k in agent.pass_obs_keys_to_actor})
        p = dist.probs[0, -1, -1, :].cpu().numpy() * (~illegal)
        if p.sum() <= 0: continue
        p = p / p.sum()
        out.write(json.dumps({"request_sha256": row["request_sha256"],
            "battle_turn": int(row["battle_turn"]),
            "stateless_probs": [float(x) for x in p]}) + "\n")
        done += 1
        if done % 100 == 0: print(f"progress {done}", flush=True)
out.close(); print(f"done {done}")
