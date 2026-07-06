"""
Pokemon-NNUE for gen9randombattle.

Architecture:
  - Per-pokemon sparse accumulator (species+moves+item+ability+status+hp → 32d)
  - Active pokemon delta (boosts+tera+volatiles → 16d)
  - Side encoding: active(48d) + 5×reserve(160d) + hazards/screens(7d) = 215d
  - Battle: side_one(215) + side_two(215) = 430d
  - Value head: 430→128→32→1 with ReLU, sigmoid output

The first layer is sparse: at most ~15 features active per pokemon.
At inference in Rust, this is O(15×64) additions instead of 2256×64 matmul.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Vocabulary ─────────────────────────────────────────────────────────────────

SPECIES_VOCAB: dict[str, int] = {}   # built from data
MOVES_VOCAB:   dict[str, int] = {}
ITEMS_VOCAB:   dict[str, int] = {}
ABILITIES_VOCAB: dict[str, int] = {}

SPECIES_UNK = 0
MOVES_UNK   = 0
ITEMS_UNK   = 0
ABILITY_UNK = 0

STATUSES = ["NONE", "BURN", "SLEEP", "FREEZE", "PARALYZE", "POISON", "TOXIC"]
STATUS_IDX: dict[str, int] = {s: i for i, s in enumerate(STATUSES)}

# Feature offsets within the sparse input vector (built after vocab is loaded)
OFF_SPECIES  = 0             # 0..N_SPECIES-1
OFF_MOVE1    = None          # species_size
OFF_MOVE2    = None
OFF_MOVE3    = None
OFF_MOVE4    = None
OFF_ITEM     = None
OFF_ABILITY  = None
OFF_STATUS   = None          # 7 dims
OFF_HP       = None          # 1 float (treated as single bin for sparse lookup)
OFF_LEVEL    = None          # 1 float
POKEMON_IN_DIM = None


def build_vocab(jsonl_paths: list[Path]) -> None:
    """Scan all self-play data to build vocabularies."""
    global SPECIES_VOCAB, MOVES_VOCAB, ITEMS_VOCAB, ABILITIES_VOCAB
    global OFF_SPECIES, OFF_MOVE1, OFF_MOVE2, OFF_MOVE3, OFF_MOVE4
    global OFF_ITEM, OFF_ABILITY, OFF_STATUS, OFF_HP, OFF_LEVEL, POKEMON_IN_DIM

    import poke_engine
    species_set: set[str] = set()
    moves_set:   set[str] = set()
    items_set:   set[str] = set()
    abilities_set: set[str] = set()

    for path in jsonl_paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "state" not in row:
                continue
            state = poke_engine.State.from_string(row["state"])
            for side in [state.side_one, state.side_two]:
                for pkmn in side.pokemon:
                    species_set.add(pkmn.id)
                    items_set.add(pkmn.item)
                    abilities_set.add(pkmn.ability)
                    for mv in pkmn.moves:
                        moves_set.add(mv.id)

    # 0 = unknown/padding
    SPECIES_VOCAB   = {s: i+1 for i, s in enumerate(sorted(species_set))}
    MOVES_VOCAB     = {m: i+1 for i, m in enumerate(sorted(moves_set))}
    ITEMS_VOCAB     = {it: i+1 for i, it in enumerate(sorted(items_set))}
    ABILITIES_VOCAB = {a: i+1 for i, a in enumerate(sorted(abilities_set))}

    n_species   = len(SPECIES_VOCAB) + 1
    n_moves     = len(MOVES_VOCAB)   + 1
    n_items     = len(ITEMS_VOCAB)   + 1
    n_abilities = len(ABILITIES_VOCAB) + 1
    n_status    = len(STATUSES)   # 7
    n_hp_level  = 2               # two floats: hp_frac, level/100

    OFF_SPECIES  = 0
    OFF_MOVE1    = OFF_SPECIES + n_species
    OFF_MOVE2    = OFF_MOVE1   + n_moves
    OFF_MOVE3    = OFF_MOVE2   + n_moves
    OFF_MOVE4    = OFF_MOVE3   + n_moves
    OFF_ITEM     = OFF_MOVE4   + n_moves
    OFF_ABILITY  = OFF_ITEM    + n_items
    OFF_STATUS   = OFF_ABILITY + n_abilities
    OFF_HP       = OFF_STATUS  + n_status
    OFF_LEVEL    = OFF_HP      + 1
    POKEMON_IN_DIM = OFF_LEVEL + 1

    print(f"Vocab: species={n_species-1} moves={n_moves-1} items={n_items-1} "
          f"abilities={n_abilities-1} → pokemon_in_dim={POKEMON_IN_DIM}")


def save_vocab(path: Path) -> None:
    data = {
        "species":   SPECIES_VOCAB,
        "moves":     MOVES_VOCAB,
        "items":     ITEMS_VOCAB,
        "abilities": ABILITIES_VOCAB,
    }
    path.write_text(json.dumps(data, indent=2))


def load_vocab(path: Path) -> None:
    global SPECIES_VOCAB, MOVES_VOCAB, ITEMS_VOCAB, ABILITIES_VOCAB
    global OFF_SPECIES, OFF_MOVE1, OFF_MOVE2, OFF_MOVE3, OFF_MOVE4
    global OFF_ITEM, OFF_ABILITY, OFF_STATUS, OFF_HP, OFF_LEVEL, POKEMON_IN_DIM

    data = json.loads(path.read_text())
    SPECIES_VOCAB   = {k: int(v) for k, v in data["species"].items()}
    MOVES_VOCAB     = {k: int(v) for k, v in data["moves"].items()}
    ITEMS_VOCAB     = {k: int(v) for k, v in data["items"].items()}
    ABILITIES_VOCAB = {k: int(v) for k, v in data["abilities"].items()}

    n_species   = max(SPECIES_VOCAB.values())   + 1
    n_moves     = max(MOVES_VOCAB.values())     + 1
    n_items     = max(ITEMS_VOCAB.values())     + 1
    n_abilities = max(ABILITIES_VOCAB.values()) + 1
    n_status    = len(STATUSES)
    OFF_SPECIES  = 0
    OFF_MOVE1    = OFF_SPECIES + n_species
    OFF_MOVE2    = OFF_MOVE1   + n_moves
    OFF_MOVE3    = OFF_MOVE2   + n_moves
    OFF_MOVE4    = OFF_MOVE3   + n_moves
    OFF_ITEM     = OFF_MOVE4   + n_moves
    OFF_ABILITY  = OFF_ITEM    + n_items
    OFF_STATUS   = OFF_ABILITY + n_abilities
    OFF_HP       = OFF_STATUS  + n_status
    OFF_LEVEL    = OFF_HP      + 1
    POKEMON_IN_DIM = OFF_LEVEL + 1


# ── Feature encoding ────────────────────────────────────────────────────────────

def encode_pokemon_dense(pkmn, alive: bool) -> np.ndarray:
    """Dense 1D array for one pokemon. POKEMON_IN_DIM floats."""
    v = np.zeros(POKEMON_IN_DIM, dtype=np.float32)
    if not alive:
        return v
    sid = SPECIES_VOCAB.get(str(pkmn.id), SPECIES_UNK)
    v[OFF_SPECIES + sid] = 1.0
    move_offs = [OFF_MOVE1, OFF_MOVE2, OFF_MOVE3, OFF_MOVE4]
    for i, mv in enumerate(pkmn.moves):
        if i < 4:
            mid = MOVES_VOCAB.get(str(mv.id), MOVES_UNK)
            v[move_offs[i] + mid] = 1.0
    iid = ITEMS_VOCAB.get(str(pkmn.item), ITEMS_UNK)
    v[OFF_ITEM + iid] = 1.0
    aid = ABILITIES_VOCAB.get(str(pkmn.ability), ABILITY_UNK)
    v[OFF_ABILITY + aid] = 1.0
    sidx = STATUS_IDX.get(str(pkmn.status), 0)
    v[OFF_STATUS + sidx] = 1.0
    v[OFF_HP]    = max(0.0, min(1.0, pkmn.hp / max(1, pkmn.maxhp)))
    v[OFF_LEVEL] = pkmn.level / 100.0
    return v


ACTIVE_EXTRA_DIM = 90  # boosts(65) + tera(20) + volatile bits(4) + tailwind(1)
TYPES_LIST = ["NORMAL","FIRE","WATER","ELECTRIC","GRASS","ICE","FIGHTING","POISON",
              "GROUND","FLYING","PSYCHIC","BUG","ROCK","GHOST","DRAGON","DARK",
              "STEEL","FAIRY","STELLAR","TYPELESS"]
TYPE_IDX = {t: i for i, t in enumerate(TYPES_LIST)}
BOOST_RANGE = range(-6, 7)  # 13 values


def encode_active_extra(side) -> np.ndarray:
    """Active-pokemon-specific features not in the per-pokemon embedding."""
    v = np.zeros(ACTIVE_EXTRA_DIM, dtype=np.float32)
    offset = 0
    # 5 boosts × 13 values each = 65
    boosts = [side.attack_boost, side.defense_boost, side.special_attack_boost,
              side.special_defense_boost, side.speed_boost]
    for b in boosts:
        b = max(-6, min(6, int(b))) + 6  # shift to 0-12
        v[offset + b] = 1.0
        offset += 13
    # tera type (20 dims)
    active_pkmn = side.pokemon[int(side.active_index)]
    if active_pkmn.terastallized:
        tidx = TYPE_IDX.get(str(active_pkmn.tera_type), 0)
        v[offset + tidx] = 1.0
    offset += 20
    # volatile bits (4)
    vs = side.volatile_statuses if isinstance(side.volatile_statuses, set) else set(side.volatile_statuses)
    v[offset + 0] = 1.0 if side.substitute_health > 0 else 0.0
    v[offset + 1] = 1.0 if "CONFUSION" in vs else 0.0
    v[offset + 2] = 1.0 if "LEECHSEED" in vs else 0.0
    v[offset + 3] = 1.0 if "ENCORE" in vs else 0.0
    offset += 4
    # tailwind
    cond = side.side_conditions
    v[offset] = 1.0 if cond.tailwind > 0 else 0.0
    return v


SIDE_POKEMON_DIM = 32  # per-pokemon embedding output
SIDE_ACTIVE_DELTA_DIM = 16
ACTIVE_TOT_DIM = SIDE_POKEMON_DIM + SIDE_ACTIVE_DELTA_DIM  # 48
RESERVE_TOT_DIM = 5 * SIDE_POKEMON_DIM  # 160
SIDE_HAZARD_SCREEN_DIM = 7
SIDE_DIM = ACTIVE_TOT_DIM + RESERVE_TOT_DIM + SIDE_HAZARD_SCREEN_DIM  # 215
BATTLE_DIM = 2 * SIDE_DIM  # 430


def encode_side_pkmn_batch(side) -> np.ndarray:
    """Returns (6, POKEMON_IN_DIM) array, active first."""
    active_idx = int(side.active_index)
    out = np.zeros((6, POKEMON_IN_DIM), dtype=np.float32)
    order = [active_idx] + [i for i in range(6) if i != active_idx]
    for slot_out, slot_in in enumerate(order):
        pkmn = side.pokemon[slot_in]
        alive = pkmn.hp > 0
        out[slot_out] = encode_pokemon_dense(pkmn, alive)
    return out


def encode_hazards_screens(side) -> np.ndarray:
    cond = side.side_conditions
    v = np.zeros(SIDE_HAZARD_SCREEN_DIM, dtype=np.float32)
    v[0] = float(cond.stealth_rock > 0)
    v[1] = min(1.0, cond.spikes / 3.0)
    v[2] = float(cond.toxic_spikes > 0)
    v[3] = float(cond.sticky_web > 0)
    v[4] = float(cond.reflect > 0)
    v[5] = float(cond.light_screen > 0)
    v[6] = float(cond.aurora_veil > 0)
    return v


# ── Neural Network ─────────────────────────────────────────────────────────────

class PokemonEncoder(nn.Module):
    """Sparse first layer + one dense layer."""
    def __init__(self, in_dim: int, mid_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.w1 = nn.Linear(in_dim, mid_dim)
        self.w2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) sparse one-hot input
        h = F.relu(self.w1(x))
        return F.relu(self.w2(h))


class ActiveDeltaEncoder(nn.Module):
    def __init__(self, in_dim: int = ACTIVE_EXTRA_DIM, out_dim: int = SIDE_ACTIVE_DELTA_DIM):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.w(x))


class ValueHead(nn.Module):
    def __init__(self, in_dim: int = BATTLE_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class PokemonNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        assert POKEMON_IN_DIM is not None, "Call load_vocab() or build_vocab() first"
        self.pokemon_enc = PokemonEncoder(POKEMON_IN_DIM, 64, SIDE_POKEMON_DIM)
        self.active_delta = ActiveDeltaEncoder(ACTIVE_EXTRA_DIM, SIDE_ACTIVE_DELTA_DIM)
        self.value_head = ValueHead(BATTLE_DIM)

    def encode_side(
        self,
        pkmn_batch: torch.Tensor,     # (B, 6, POKEMON_IN_DIM)
        active_extra: torch.Tensor,   # (B, ACTIVE_EXTRA_DIM)
        hazards_screens: torch.Tensor,# (B, 7)
    ) -> torch.Tensor:
        B = pkmn_batch.shape[0]
        # Encode all 6 pokemon independently
        emb = self.pokemon_enc(pkmn_batch.view(B * 6, POKEMON_IN_DIM))
        emb = emb.view(B, 6, SIDE_POKEMON_DIM)  # (B, 6, 32)
        # Active embedding + delta
        active_emb   = emb[:, 0, :]                      # (B, 32)
        active_delta = self.active_delta(active_extra)    # (B, 16)
        active_full  = torch.cat([active_emb, active_delta], dim=-1)  # (B, 48)
        # Reserve embeddings (fainted mons give zero via their hp feature masking)
        reserve_flat = emb[:, 1:, :].reshape(B, RESERVE_TOT_DIM)    # (B, 160)
        return torch.cat([active_full, reserve_flat, hazards_screens], dim=-1)  # (B, 215)

    def forward(
        self,
        s1_pkmn: torch.Tensor,    s1_active: torch.Tensor,    s1_hs: torch.Tensor,
        s2_pkmn: torch.Tensor,    s2_active: torch.Tensor,    s2_hs: torch.Tensor,
    ) -> torch.Tensor:
        side1 = self.encode_side(s1_pkmn, s1_active, s1_hs)
        side2 = self.encode_side(s2_pkmn, s2_active, s2_hs)
        battle = torch.cat([side1, side2], dim=-1)
        return self.value_head(battle)


# ── Data loading ─────────────────────────────────────────────────────────────────

def load_dataset(jsonl_paths: list[Path]):
    """Load all self-play rows, re-featurize with NNUE encoding."""
    import poke_engine
    s1_pkmn_list, s1_active_list, s1_hs_list = [], [], []
    s2_pkmn_list, s2_active_list, s2_hs_list = [], [], []
    labels_list, tags_list = [], []

    for path in jsonl_paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "state" not in row or "label" not in row or "battle_tag" not in row:
                continue
            state = poke_engine.State.from_string(row["state"])
            s1, s2 = state.side_one, state.side_two
            s1_pkmn_list.append(encode_side_pkmn_batch(s1))
            s1_active_list.append(encode_active_extra(s1))
            s1_hs_list.append(encode_hazards_screens(s1))
            s2_pkmn_list.append(encode_side_pkmn_batch(s2))
            s2_active_list.append(encode_active_extra(s2))
            s2_hs_list.append(encode_hazards_screens(s2))
            labels_list.append(float(row["label"]))
            tags_list.append(row["battle_tag"])

    return (
        np.array(s1_pkmn_list),   np.array(s1_active_list),   np.array(s1_hs_list),
        np.array(s2_pkmn_list),   np.array(s2_active_list),   np.array(s2_hs_list),
        np.array(labels_list),    tags_list,
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    data_paths: list[Path],
    vocab_path: Path,
    model_out: Path,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 3e-4,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    import time
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Vocab
    if vocab_path.exists():
        print(f"Loading vocab from {vocab_path}")
        load_vocab(vocab_path)
    else:
        print("Building vocab...")
        build_vocab(data_paths)
        save_vocab(vocab_path)

    # Data
    print("Loading dataset...")
    t0 = time.time()
    (s1p, s1a, s1h, s2p, s2a, s2h, labels, tags) = load_dataset(data_paths)
    print(f"  Loaded {len(labels)} rows in {time.time()-t0:.1f}s")

    # By-game split
    games = list(set(tags))
    rng.shuffle(games)
    n_held = max(1, int(0.2 * len(games)))
    held_set = set(games[:n_held])
    train_mask = np.array([t not in held_set for t in tags])
    held_mask  = ~train_mask
    print(f"  Train: {train_mask.sum()} rows / {len(games)-n_held} games")
    print(f"  Held:  {held_mask.sum()} rows / {n_held} games")

    def to_t(arr): return torch.from_numpy(arr).to(device)
    tr = {k: to_t(v) for k, v in dict(
        s1p=s1p[train_mask], s1a=s1a[train_mask], s1h=s1h[train_mask],
        s2p=s2p[train_mask], s2a=s2a[train_mask], s2h=s2h[train_mask],
        y=labels[train_mask].astype(np.float32)).items()}
    hd = {k: to_t(v) for k, v in dict(
        s1p=s1p[held_mask],  s1a=s1a[held_mask],  s1h=s1h[held_mask],
        s2p=s2p[held_mask],  s2a=s2a[held_mask],  s2h=s2h[held_mask],
        y=labels[held_mask].astype(np.float32)).items()}

    model = PokemonNNUE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_tr = train_mask.sum()

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n_tr)
        total_loss = 0.0; steps = 0
        for start in range(0, n_tr, batch_size):
            ids = order[start:start+batch_size]
            pred = model(tr['s1p'][ids], tr['s1a'][ids], tr['s1h'][ids],
                         tr['s2p'][ids], tr['s2a'][ids], tr['s2h'][ids]).squeeze(-1)
            loss = F.binary_cross_entropy(pred, tr['y'][ids])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); steps += 1

        if (epoch+1) % 5 == 0 or epoch == epochs-1:
            model.eval()
            with torch.no_grad():
                ph = model(hd['s1p'], hd['s1a'], hd['s1h'],
                           hd['s2p'], hd['s2a'], hd['s2h']).squeeze(-1)
                brier = float(((ph - hd['y'])**2).mean())
                acc   = float(((ph >= 0.5) == (hd['y'] >= 0.5)).float().mean())
                # contested positions
                # contested positions: both sides have ≥2 alive mons
                # proxy: alive_diff feature is not extreme
                # Use held-out alive counts
                s1p_hd_np = s1p[held_mask]  # (N, 6, POKEMON_IN_DIM)
                s2p_hd_np = s2p[held_mask]
                alive1 = (s1p_hd_np[:, :, OFF_HP] > 0).sum(axis=1)
                alive2 = (s2p_hd_np[:, :, OFF_HP] > 0).sum(axis=1)
                cmask_np = (alive1 >= 2) & (alive2 >= 2)
                if cmask_np.sum() > 0:
                    cmask_t = torch.from_numpy(cmask_np).to(device)
                    ph_c = ph[cmask_t]; y_c = hd['y'][cmask_t]
                    cont_acc = float(((ph_c >= 0.5) == (y_c >= 0.5)).float().mean())
                else:
                    cont_acc = float('nan')
                print(f"Epoch {epoch+1:3d}: train_loss={total_loss/steps:.4f} "
                      f"heldout_brier={brier:.4f} heldout_acc={acc:.4f} "
                      f"contested_acc={cont_acc:.4f}")

    # Save model and vocab info in a single portable format
    export_model(model, model_out, vocab_path)
    return {"brier": brier, "acc": acc, "held_n": held_mask.sum()}


# ── Export for Rust ────────────────────────────────────────────────────────────

def export_model(model: PokemonNNUE, out_path: Path, vocab_path: Path) -> None:
    """
    Export model weights + vocab indices to a binary file readable by Rust.
    Format: metagross_nnue_v1 header + JSON vocab sizes + float32 weights.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def to_flat(param: nn.Parameter) -> np.ndarray:
        return param.detach().cpu().numpy().astype(np.float32).ravel()

    vocab_meta = {
        "n_species":   max(SPECIES_VOCAB.values())   + 1,
        "n_moves":     max(MOVES_VOCAB.values())     + 1,
        "n_items":     max(ITEMS_VOCAB.values())     + 1,
        "n_abilities": max(ABILITIES_VOCAB.values()) + 1,
        "n_status":    len(STATUSES),
        "pokemon_in_dim": POKEMON_IN_DIM,
        "species":    {k: v for k, v in SPECIES_VOCAB.items()},
        "moves":      {k: v for k, v in MOVES_VOCAB.items()},
        "items":      {k: v for k, v in ITEMS_VOCAB.items()},
        "abilities":  {k: v for k, v in ABILITIES_VOCAB.items()},
    }
    meta_bytes = json.dumps(vocab_meta).encode("utf-8")

    def write_layer(f, layer: nn.Linear):
        # weight: (out, in), bias: (out,)
        w = layer.weight.detach().cpu().numpy().astype(np.float32)
        b = layer.bias.detach().cpu().numpy().astype(np.float32)
        f.write(struct.pack("<II", w.shape[0], w.shape[1]))  # out_dim, in_dim
        f.write(w.tobytes())
        f.write(b.tobytes())

    with open(out_path, "wb") as f:
        f.write(b"metagross_nnue_v1\n")
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        # pokemon_enc
        write_layer(f, model.pokemon_enc.w1)
        write_layer(f, model.pokemon_enc.w2)
        # active_delta
        write_layer(f, model.active_delta.w)
        # value head
        write_layer(f, model.value_head.fc1)
        write_layer(f, model.value_head.fc2)
        write_layer(f, model.value_head.fc3)

    print(f"Saved NNUE model to {out_path} ({out_path.stat().st_size/1024:.1f}KB)")


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def compute_contested_accuracy(model: PokemonNNUE, hd: dict) -> float:
    """Accuracy on held-out contested positions (both sides have ≥2 alive)."""
    model.eval()
    with torch.no_grad():
        ph = model(hd['s1p'], hd['s1a'], hd['s1h'],
                   hd['s2p'], hd['s2a'], hd['s2h']).squeeze(-1)
    pred_labels = (ph >= 0.5)
    true_labels = (hd['y'] >= 0.5)
    return float((pred_labels == true_labels).float().mean())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default="external/selfplay_data_1k")
    parser.add_argument("--vocab-path",  default="nets/nnue/vocab.json")
    parser.add_argument("--model-out",   default="nets/nnue/pokemon_nnue.bin")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch-size",  type=int, default=1024)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    data_paths = list(Path(args.data_dir).glob("*.jsonl"))
    if not data_paths:
        raise FileNotFoundError(f"No .jsonl files found in {args.data_dir}")

    metrics = train(
        data_paths=data_paths,
        vocab_path=Path(args.vocab_path),
        model_out=Path(args.model_out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    print(f"\nFinal metrics: {metrics}")
