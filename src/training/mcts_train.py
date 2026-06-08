from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.checkpoint import load_checkpoint, save_checkpoint
from model.network import PokeNet
from model.state import EncodedState, build_vocabulary, encode_state, stack_encoded


class MCTSVisitDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path, pool: str | Path = "data/gen9_random_pool.json"):
        self.records: list[dict[str, Any]] = []
        self.vocab = build_vocabulary(pool)
        root = Path(path)
        paths = [root] if root.is_file() else sorted(root.glob("*.jsonl"))
        for file in paths:
            for line in file.read_text().splitlines():
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        state = encode_state(record.get("state") or record.get("encoded_state"), vocab=self.vocab)
        visits = np.asarray(record.get("mcts_visit_distribution") or record.get("visits"), dtype=np.float32)
        if visits.shape[0] != 14:
            padded = np.zeros(14, dtype=np.float32)
            padded[: min(14, visits.shape[0])] = visits[:14]
            visits = padded
        total = float(visits.sum())
        visits = visits / total if total > 0 else np.full(14, 1.0 / 14.0, dtype=np.float32)
        return {"state": state, "visits": visits, "outcome": float(record.get("outcome", 0.0))}


def collate_mcts(records: list[dict[str, Any]]) -> dict[str, Any]:
    batch = stack_encoded([record["state"] for record in records])
    batch["visits"] = np.stack([record["visits"] for record in records])
    batch["outcomes"] = np.asarray([record["outcome"] for record in records], dtype=np.float32)
    return batch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 MCTS visit-distribution trainer")
    parser.add_argument("--data", required=True, help="JSONL file or directory of MCTS samples")
    parser.add_argument("--pool", default="data/gen9_random_pool.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="checkpoints/phase3.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset = MCTSVisitDataset(args.data, pool=args.pool)
    if len(dataset) == 0:
        raise SystemExit("No MCTS samples found")
    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint).to(device) if args.checkpoint else PokeNet(vocab=dataset.vocab).to(device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mcts)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            visits = torch.as_tensor(batch["visits"], dtype=torch.float32, device=device)
            outcomes = torch.as_tensor(batch["outcomes"], dtype=torch.float32, device=device)
            logits, values = model(batch)
            policy_loss = F.kl_div(F.log_softmax(logits, dim=-1), visits, reduction="batchmean")
            value_loss = F.mse_loss(values, outcomes)
            loss = policy_loss + value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        print(json.dumps({"epoch": epoch + 1, "loss": total_loss / max(1, len(loader))}))
    save_checkpoint(args.output, model, optimizer, phase="phase3", samples=len(dataset))
    print(json.dumps({"checkpoint": args.output, "samples": len(dataset)}, indent=2))


if __name__ == "__main__":
    main()
