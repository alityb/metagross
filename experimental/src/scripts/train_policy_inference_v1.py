#!/usr/bin/env python3
"""Train the 1c action-in-context likelihood model (Policy Inference v1).

Pointer architecture: shared move embeddings scored against a context vector
(actor species/item/ability/tera + full set composition + opponent species +
turn features); softmax over the actor's four set moves. Battle-grouped
90/10 split. Baselines: uniform (ln 4) and per-(species,set) marginal move
frequency from the training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 64


def battle_split(battle_id: str) -> str:
    digest = hashlib.sha256(("pi-split-v1\0" + battle_id).encode()).digest()
    return "val" if digest[0] < 26 else "train"  # ~10% val


class Vocab:
    def __init__(self):
        self.index = {"<unk>": 0}

    def add(self, token: str) -> int:
        if token not in self.index:
            self.index[token] = len(self.index)
        return self.index[token]

    def get(self, token: str) -> int:
        return self.index.get(token, 0)


class PolicyInference(nn.Module):
    def __init__(self, n_moves, n_species, n_items, n_abilities, n_teras):
        super().__init__()
        self.move = nn.Embedding(n_moves, DIM)
        self.species = nn.Embedding(n_species, DIM)
        self.item = nn.Embedding(n_items, DIM // 2)
        self.ability = nn.Embedding(n_abilities, DIM // 2)
        self.tera = nn.Embedding(n_teras, DIM // 2)
        self.opp = nn.Embedding(n_species, DIM)
        self.ctx = nn.Sequential(
            nn.Linear(DIM * 3 + DIM // 2 * 3 + 2, 256), nn.ReLU(),
            nn.Linear(256, DIM))
        self.move_bias = nn.Embedding(n_moves, 1)

    def forward(self, batch):
        set_mean = self.move(batch["moves"]).mean(dim=1)
        ctx = self.ctx(torch.cat([
            self.species(batch["species"]), set_mean, self.opp(batch["opp"]),
            self.item(batch["item"]), self.ability(batch["ability"]),
            self.tera(batch["tera"]),
            batch["turn"].unsqueeze(1), batch["turns_in"].unsqueeze(1),
        ], dim=1))
        cand = self.move(batch["moves"])                      # B x 4 x D
        scores = torch.einsum("bd,bkd->bk", ctx, cand)
        scores = scores + self.move_bias(batch["moves"]).squeeze(-1)
        return scores                                          # B x 4


def encode(rows, vocabs, train: bool):
    moves_v, species_v, items_v, abilities_v, teras_v = vocabs
    out = []
    for r in rows:
        if len(r["moves"]) != 4:
            continue  # Ditto-style sets; likelihood fails open at eval too
        add = (lambda v, t: v.add(t)) if train else (lambda v, t: v.get(t))
        move_ids = [add(moves_v, m) for m in r["moves"]]
        target = r["moves"].index(r["chosen"])
        out.append((
            move_ids, add(species_v, r["species"]), add(species_v, r["opp_species"]),
            add(items_v, r["item"]), add(abilities_v, r["ability"]),
            add(teras_v, r["tera"]), r["turn"] / 40.0, r["turns_in"] / 10.0, target,
        ))
    return out


def batches(data, size, shuffle):
    order = list(range(len(data)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), size):
        chunk = [data[i] for i in order[start:start + size]]
        yield {
            "moves": torch.tensor([c[0] for c in chunk]),
            "species": torch.tensor([c[1] for c in chunk]),
            "opp": torch.tensor([c[2] for c in chunk]),
            "item": torch.tensor([c[3] for c in chunk]),
            "ability": torch.tensor([c[4] for c in chunk]),
            "tera": torch.tensor([c[5] for c in chunk]),
            "turn": torch.tensor([c[6] for c in chunk], dtype=torch.float32),
            "turns_in": torch.tensor([c[7] for c in chunk], dtype=torch.float32),
            "target": torch.tensor([c[8] for c in chunk]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026081701)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(2)

    train_rows, val_rows = [], []
    for line in args.rows.read_text().splitlines():
        r = json.loads(line)
        (val_rows if battle_split(r["battle_id"]) == "val" else train_rows).append(r)
    print(json.dumps({"train_rows": len(train_rows), "val_rows": len(val_rows)}))

    vocabs = (Vocab(), Vocab(), Vocab(), Vocab(), Vocab())
    train_data = encode(train_rows, vocabs, train=True)
    val_data = encode(val_rows, vocabs, train=False)

    # marginal baseline: per (species, set) chosen-move frequency from train
    marginal = defaultdict(Counter)
    for r in train_rows:
        marginal[(r["species"], tuple(r["moves"]))][r["chosen"]] += 1
    nll_marginal = nll_uniform = 0.0
    for r in val_rows:
        counts = marginal.get((r["species"], tuple(r["moves"])), Counter())
        total = sum(counts.values())
        p = (counts.get(r["chosen"], 0) + 1.0) / (total + 4.0)
        nll_marginal -= math.log(p)
        nll_uniform -= math.log(0.25)
    nll_marginal /= len(val_rows)
    nll_uniform /= len(val_rows)

    model = PolicyInference(*(len(v.index) for v in vocabs))
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = None
    for epoch in range(args.epochs):
        model.train()
        for batch in batches(train_data, args.batch, shuffle=True):
            optim.zero_grad()
            loss = F.cross_entropy(model(batch), batch["target"])
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            total = correct = 0
            nll = 0.0
            for batch in batches(val_data, 2048, shuffle=False):
                scores = model(batch)
                nll += F.cross_entropy(scores, batch["target"],
                                       reduction="sum").item()
                correct += (scores.argmax(1) == batch["target"]).sum().item()
                total += len(batch["target"])
            val_nll = nll / total
            row = {"epoch": epoch, "val_nll": round(val_nll, 5),
                   "val_top1": round(correct / total, 5),
                   "uniform_nll": round(nll_uniform, 5),
                   "marginal_nll": round(nll_marginal, 5)}
            print(json.dumps(row), flush=True)
            if best is None or val_nll < best:
                best = val_nll
                args.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.output_dir / "model.pt")
                (args.output_dir / "vocabs.json").write_text(json.dumps(
                    {name: v.index for name, v in
                     zip(("moves", "species", "items", "abilities", "teras"), vocabs)},
                    sort_keys=True))
                (args.output_dir / "metrics.json").write_text(
                    json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"best_val_nll": round(best, 5)}))


if __name__ == "__main__":
    main()
