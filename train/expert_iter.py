#!/usr/bin/env python3
"""
Expert iteration trainer for gen1ou.

Trains a joint policy+value network from MCTS self-play data:
  - Value head: predict game outcome (0/1) from state features
  - Policy head: predict MCTS visit distribution per move

Architecture:
  State features (12-dim) → shared trunk → value head (scalar)
                                          → policy head (per-move score)

Policy head uses (state_features || move_features) → scalar Q(s,a):
  move_features: [is_status, is_physical, is_special, bp_norm, stab_bonus]
  K moves → K scores → softmax → target MCTS visit distribution

Loss = BCE(value) + KL(policy_softmax || mcts_visits)

Training targets from decision log:
  features  : 12-dim from compute_value_features (exact Rust extraction)
  mcts_visits: {move_str: fraction} — MCTS visit distribution
  label     : 1=won, 0=lost

Output model format:
  metagross_policy_value_v1
  # ... header comment ...
  value_dims 12 32 16 1
  value_w1 ...
  value_b1 ...
  ...
  policy_dims 17 32 16 1    # 12 state + 5 move features
  policy_w1 ...
  ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
STATE_FEAT = 12   # from compute_value_features
MOVE_FEAT  = 5    # [is_status, is_physical, is_special, bp_norm, stab_bonus]
POLICY_IN  = STATE_FEAT + MOVE_FEAT

# Move feature extraction from move name string.
# In gen1ou the viable moves are known; we use name-based heuristics.
# The full feature will be computed by Rust at inference; here we approximate.

PHYSICAL_MOVES = {
    "bodyslam", "earthquake", "hyperbeam", "seismictoss", "explosion",
    "selfdestruc", "selfdestruct", "rockslide", "doubleedge", "slash",
    "submission", "counter", "return", "swords", "cut", "strength",
    "megapunch", "megakick", "bide", "bind", "wrap", "clamp", "firespin",
    "pinmissile", "crabhammer", "kinesis", "stomp", "takedown", "headbutt",
    "tackle", "scratch", "vicegrip", "wingattack", "razorleaf", "crab",
    "doublekick", "drillpeck", "jumpkick", "highjumpkick",
}

STATUS_MOVES = {
    "thunderwave", "sleeppowder", "stunspore", "spore", "softboiled",
    "recover", "rest", "reflect", "lightscreen", "substitute", "agility",
    "swordsdance", "amnesia", "growl", "leer", "harden", "barrier",
    "smokescreen", "sandattack", "glare", "toxic", "poisonpowder",
    "lovelykiss", "hypnosis", "sing", "supersonic", "confuse", "disable",
    "teleport", "splash", "leechseed", "mimic", "metronome", "minimize",
    "defensecurl", "doubleteam", "acidarmor", "mist", "flash", "focus",
}

BASE_POWERS = {
    "hyperbeam": 150, "blizzard": 120, "fireblast": 120, "thunder": 110,
    "surf": 95, "thunderbolt": 95, "icebeam": 95, "psychic": 90,
    "earthquake": 100, "rockslide": 75, "bodyslam": 80, "slash": 70,
    "explosion": 250, "selfdestruct": 200, "seismictoss": 100,
    "doubleedge": 100, "megakick": 120, "megapunch": 80,
    "drillpeck": 80, "submission": 80, "crabhammer": 90,
    "pinmissile": 14, "bide": 0, "counter": 0,
}


def move_features(move_str: str) -> list[float]:
    """5 features for a move: is_status, is_phys, is_spec, bp_norm, stab_bonus."""
    m = move_str.lower().replace(" ", "").replace("-", "")
    is_status = 1.0 if m in STATUS_MOVES else 0.0
    is_phys   = 1.0 if m in PHYSICAL_MOVES and not is_status else 0.0
    is_spec   = 1.0 if not is_status and not is_phys else 0.0
    bp        = BASE_POWERS.get(m, 80.0 if not is_status else 0.0)
    bp_norm   = bp / 150.0
    stab      = 0.0  # unknown without species context; model will learn from state
    return [is_status, is_phys, is_spec, bp_norm, stab]


def switch_features() -> list[float]:
    """Move features for a switch action."""
    return [0.0, 0.0, 0.0, 0.0, 0.0]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(jsonl_paths: list[Path], max_rows: int = 0):
    """Load rows from decision-log JSONL files.

    Each row must have:
        features   : list[float] len 12
        mcts_visits: {move_str: float} (visit fractions, sum ≈ 1)
        label      : int 0/1
        battle_tag : str (for by-game split)

    Returns list of dicts.
    """
    import poke_engine as pe

    rows = []
    for path in jsonl_paths:
        for line in Path(path).read_text(errors="replace").splitlines():
            try:
                r = json.loads(line)
                if "state" not in r or "label" not in r or "battle_tag" not in r:
                    continue
                if "mcts_visits" not in r:
                    continue
                # Re-extract exact features via Rust
                state = pe.State.from_string(r["state"])
                feats = list(pe.compute_value_features(state))
                rows.append({
                    "features": feats,
                    "visits": r["mcts_visits"],      # {move: fraction}
                    "label": int(r["label"]),
                    "battle_tag": r["battle_tag"],
                })
                if max_rows and len(rows) >= max_rows:
                    return rows
            except Exception:
                pass
    return rows


def by_game_split(rows: list[dict], held_frac: float = 0.1, seed: int = 42):
    """Split by battle_tag to prevent leakage."""
    tags = list({r["battle_tag"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(tags)
    n_held = max(1, int(len(tags) * held_frac))
    held_set = set(tags[:n_held])
    train = [r for r in rows if r["battle_tag"] not in held_set]
    held  = [r for r in rows if r["battle_tag"] in held_set]
    return train, held


# ── Network ───────────────────────────────────────────────────────────────────
class PolicyValueNet:
    """Joint policy+value MLP, all numpy.

    Value: 12 → H1 → H2 → 1    (sigmoid output = win prob)
    Policy: 17 → H1 → H2 → 1   (Q-score per (state, move) pair)
    """

    def __init__(self, h1: int = 32, h2: int = 16,
                 lr: float = 0.003, l2: float = 1e-3, seed: int = 42):
        rng = np.random.default_rng(seed)
        s = 0.1
        # Value head
        self.vW1 = rng.normal(0, s, (STATE_FEAT, h1)).astype(np.float32)
        self.vb1 = np.zeros(h1, np.float32)
        self.vW2 = rng.normal(0, s, (h1, h2)).astype(np.float32)
        self.vb2 = np.zeros(h2, np.float32)
        self.vW3 = rng.normal(0, s, (h2, 1)).astype(np.float32)
        self.vb3 = np.float32(0.0)
        # Policy head
        self.pW1 = rng.normal(0, s, (POLICY_IN, h1)).astype(np.float32)
        self.pb1 = np.zeros(h1, np.float32)
        self.pW2 = rng.normal(0, s, (h1, h2)).astype(np.float32)
        self.pb2 = np.zeros(h2, np.float32)
        self.pW3 = rng.normal(0, s, (h2, 1)).astype(np.float32)
        self.pb3 = np.float32(0.0)
        self.h1, self.h2, self.lr, self.l2 = h1, h2, lr, l2

    def _sigmoid(self, x): return 1.0 / (1.0 + np.exp(-x.clip(-30, 30)))

    def value_forward(self, X: np.ndarray) -> tuple:
        a1 = np.tanh(X @ self.vW1 + self.vb1)
        a2 = np.tanh(a1 @ self.vW2 + self.vb2)
        out = self._sigmoid(a2 @ self.vW3 + self.vb3)[:, 0]
        return out, a1, a2

    def policy_q(self, X: np.ndarray) -> np.ndarray:
        """X: (K, POLICY_IN) → (K,) raw Q-scores."""
        a1 = np.tanh(X @ self.pW1 + self.pb1)
        a2 = np.tanh(a1 @ self.pW2 + self.pb2)
        return (a2 @ self.pW3 + self.pb3)[:, 0]

    def policy_probs(self, state_feats: np.ndarray,
                     move_list: list[str]) -> np.ndarray:
        """Compute softmax distribution over moves for one position."""
        K = len(move_list)
        if K == 0:
            return np.array([])
        sf = np.tile(state_feats, (K, 1))
        mf = np.array([move_features(m) if not m.startswith("switch")
                       else switch_features() for m in move_list],
                      dtype=np.float32)
        X = np.concatenate([sf, mf], axis=1)
        q = self.policy_q(X)
        q -= q.max()
        e = np.exp(q)
        return e / e.sum()

    def train_step(self, batch: list[dict]) -> tuple[float, float]:
        """One gradient step. Returns (value_loss, policy_loss)."""
        n = len(batch)
        # ── Value loss ────────────────────────────────────────────────────────
        X_val = np.array([r["features"] for r in batch], np.float32)
        y_val = np.array([r["label"] for r in batch], np.float32)
        vout, va1, va2 = self.value_forward(X_val)
        eps = 1e-7
        vloss = float(-np.mean(
            y_val * np.log(vout.clip(eps)) + (1-y_val) * np.log((1-vout).clip(eps))
        ))
        # Backward value
        d3 = (vout - y_val)[:, None] / n
        gvW3 = va2.T @ d3 + self.l2 * self.vW3
        gvb3 = float(d3.sum())
        d2   = (d3 @ self.vW3.T) * (1 - va2**2)
        gvW2 = va1.T @ d2 + self.l2 * self.vW2
        gvb2 = d2.sum(0)
        d1   = (d2 @ self.vW2.T) * (1 - va1**2)
        gvW1 = X_val.T @ d1 + self.l2 * self.vW1
        gvb1 = d1.sum(0)
        self.vW1 -= self.lr * gvW1; self.vb1 -= self.lr * gvb1
        self.vW2 -= self.lr * gvW2; self.vb2 -= self.lr * gvb2
        self.vW3 -= self.lr * gvW3; self.vb3 -= self.lr * gvb3

        # ── Policy loss (KL divergence) ────────────────────────────────────────
        ploss_total = 0.0
        gW1 = np.zeros_like(self.pW1); gb1 = np.zeros_like(self.pb1)
        gW2 = np.zeros_like(self.pW2); gb2 = np.zeros_like(self.pb2)
        gW3 = np.zeros_like(self.pW3); gb3_acc = 0.0

        for r in batch:
            sf = np.array(r["features"], np.float32)
            visits = r["visits"]  # {move: fraction}
            moves = list(visits.keys())
            if not moves:
                continue
            target = np.array([visits[m] for m in moves], np.float32)
            target = target / target.sum().clip(1e-9)  # ensure sum=1

            K = len(moves)
            sf_rep = np.tile(sf, (K, 1))
            mf = np.array([move_features(m) if not m.startswith("switch")
                           else switch_features() for m in moves], np.float32)
            Xp = np.concatenate([sf_rep, mf], axis=1)

            pa1 = np.tanh(Xp @ self.pW1 + self.pb1)
            pa2 = np.tanh(pa1 @ self.pW2 + self.pb2)
            q = (pa2 @ self.pW3 + self.pb3)[:, 0]
            q_s = q - q.max()
            eq = np.exp(q_s)
            prob = eq / eq.sum()

            # KL = sum(target * log(target/prob)) — gradient w.r.t q: (prob - target)
            kl = float(np.sum(target * np.log((target / prob.clip(eps)).clip(1e-9))))
            ploss_total += kl

            dpq = (prob - target)[:, None] / n   # (K,1)
            gpW3 = pa2.T @ dpq + self.l2 * self.pW3 / n
            gpb3 = float(dpq.sum())
            dp2  = (dpq @ self.pW3.T) * (1 - pa2**2)
            gpW2 = pa1.T @ dp2 + self.l2 * self.pW2 / n
            gpb2 = dp2.sum(0)
            dp1  = (dp2 @ self.pW2.T) * (1 - pa1**2)
            gpW1 = Xp.T @ dp1 + self.l2 * self.pW1 / n
            gpb1 = dp1.sum(0)

            gW1 += gpW1; gb1 += gpb1
            gW2 += gpW2; gb2 += gpb2
            gW3 += gpW3; gb3_acc += gpb3

        self.pW1 -= self.lr * gW1; self.pb1 -= self.lr * gb1
        self.pW2 -= self.lr * gW2; self.pb2 -= self.lr * gb2
        self.pW3 -= self.lr * gW3; self.pb3 -= self.lr * gb3_acc

        return vloss, ploss_total / n

    def to_txt(self) -> str:
        def fmt(a): return " ".join(f"{v:.6f}" for v in a.flatten())
        lines = [
            "metagross_policy_value_v1",
            f"# gen1ou expert iteration; state={STATE_FEAT} move={MOVE_FEAT} h={self.h1}x{self.h2}",
            f"value_dims {STATE_FEAT} {self.h1} {self.h2} 1",
            f"value_w1 {fmt(self.vW1)}", f"value_b1 {fmt(self.vb1)}",
            f"value_w2 {fmt(self.vW2)}", f"value_b2 {fmt(self.vb2)}",
            f"value_w3 {fmt(self.vW3)}", f"value_b3 {self.vb3:.6f}",
            f"policy_dims {POLICY_IN} {self.h1} {self.h2} 1",
            f"policy_w1 {fmt(self.pW1)}", f"policy_b1 {fmt(self.pb1)}",
            f"policy_w2 {fmt(self.pW2)}", f"policy_b2 {fmt(self.pb2)}",
            f"policy_w3 {fmt(self.pW3)}", f"policy_b3 {self.pb3:.6f}",
        ]
        return "\n".join(lines)


# ── Metrics ───────────────────────────────────────────────────────────────────
def eval_metrics(net: PolicyValueNet, held: list[dict]) -> dict:
    if not held:
        return {}
    X = np.array([r["features"] for r in held], np.float32)
    y = np.array([r["label"] for r in held], np.float32)
    vout, _, _ = net.value_forward(X)
    eps = 1e-7
    vbce = float(-np.mean(y*np.log(vout.clip(eps)) + (1-y)*np.log((1-vout).clip(eps))))
    vacc = float(np.mean((vout >= 0.5) == y))
    vbrier = float(np.mean((vout - y)**2))
    # Policy KL on held
    kl_sum = 0.0
    for r in held:
        visits = r["visits"]
        if not visits: continue
        moves = list(visits.keys())
        target = np.array([visits[m] for m in moves], np.float32)
        target /= target.sum().clip(1e-9)
        prob = net.policy_probs(np.array(r["features"], np.float32), moves)
        kl_sum += float(np.sum(target * np.log((target / prob.clip(eps)).clip(1e-9))))
    return {
        "value_bce": vbce, "value_acc": vacc, "value_brier": vbrier,
        "policy_kl": kl_sum / max(1, len(held)),
        "n_held": len(held), "base_rate": float(y.mean()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True, help="Decision log JSONL files")
    ap.add_argument("--out", required=True, help="Output model .txt path")
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--h1", type=int, default=32)
    ap.add_argument("--h2", type=int, default=16)
    ap.add_argument("--held-frac", type=float, default=0.1)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading from {args.jsonl}...")
    rows = load_data([Path(p) for p in args.jsonl], args.max_rows)
    print(f"  {len(rows)} rows loaded")
    if not rows:
        raise RuntimeError("No rows loaded")

    train, held = by_game_split(rows, args.held_frac, args.seed)
    print(f"  train={len(train)} held={len(held)}")

    net = PolicyValueNet(args.h1, args.h2, args.lr, seed=args.seed)
    rng = random.Random(args.seed)

    best_metrics = None
    best_txt = None
    best_score = float("inf")

    for epoch in range(args.epochs):
        rng.shuffle(train)
        v_losses, p_losses = [], []
        for i in range(0, len(train), args.batch):
            batch = train[i:i+args.batch]
            vl, pl = net.train_step(batch)
            v_losses.append(vl); p_losses.append(pl)
        m = eval_metrics(net, held)
        score = m.get("value_brier", 999) + 0.1 * m.get("policy_kl", 999)
        if score < best_score:
            best_score = score
            best_metrics = dict(m, epoch=epoch+1)
            best_txt = net.to_txt()
        if (epoch+1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}: "
                  f"val_bce={np.mean(v_losses):.4f} "
                  f"pol_kl={np.mean(p_losses):.4f} | "
                  f"held acc={m.get('value_acc',0):.3f} "
                  f"brier={m.get('value_brier',0):.4f} "
                  f"policy_kl={m.get('policy_kl',0):.4f}")

    Path(args.out).write_text(best_txt)
    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(best_metrics, indent=2))
    print(f"\nSaved: {args.out}")
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
