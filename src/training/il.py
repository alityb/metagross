from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.checkpoint import save_checkpoint
from model.network import PokeNet
from model.state import build_vocabulary

from .dataset import ReplayDataset, StreamingReplayDataset, collate_replay_samples, torch_targets as replay_torch_targets
from .replay_reconstruction import Phase1AnnotationDataset, collate_decisions, torch_targets as annotation_torch_targets


def normalize_masked_targets(targets: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    mask = action_mask.float()
    masked = targets * mask
    target_sums = masked.sum(dim=-1, keepdim=True)
    mask_sums = mask.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(mask, 1.0 / mask.shape[-1])
    fallback = torch.where(mask_sums > 0, mask / mask_sums.clamp_min(1.0), uniform)
    return torch.where(target_sums > 0, masked / target_sums.clamp_min(1e-12), fallback)


def train_epoch(
    model: PokeNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_fn: Callable[[dict[str, Any], torch.device], dict[str, torch.Tensor]],
    smoke: bool = False,
    awr_beta: float = 1.0,
) -> dict[str, float]:
    """Train one epoch with AWR advantage-weighted policy loss (Metamon ExpRL).

    AWR: weight each (state, action) by exp(β × advantage) where
    advantage = Q(s,a) - mean_Q(s).  This means we imitate strongly
    above-average decisions and downweight below-average ones rather
    than treating all actions equally (pure BC).

    awr_beta=0.0 → pure behavioral cloning (original behaviour)
    awr_beta=1.0 → standard AWR (Metamon default, outperforms pure BC)
    """
    model.train()
    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "belief": 0.0, "distill": 0.0, "policy_distill": 0.0, "batches": 0.0}
    for step, batch in enumerate(loader, start=1):
        targets = target_fn(batch, device)
        logits, values = model(batch)
        action_mask = torch.as_tensor(batch["action_mask"], dtype=torch.bool, device=device)
        valid_actions = (targets["actions"] >= 0) & action_mask.gather(1, targets["actions"].clamp_min(0).unsqueeze(1)).squeeze(1)

        if valid_actions.any():
            # AWR advantage weighting
            if awr_beta > 0.0:
                with torch.no_grad():
                    # Q(s, a_taken) ≈ values (scalar baseline); advantage per-step
                    # = outcome - value_estimate. Clamp weights to [1e-3, 20] for stability.
                    adv = (targets["outcomes"][valid_actions] - values[valid_actions]).detach()
                    weights = torch.exp(awr_beta * adv).clamp(1e-3, 20.0)
                per_sample_ce = F.cross_entropy(
                    logits[valid_actions],
                    targets["actions"][valid_actions],
                    reduction="none",
                )
                policy_loss = (weights * per_sample_ce).mean()
            else:
                policy_loss = F.cross_entropy(logits[valid_actions], targets["actions"][valid_actions])
        else:
            policy_loss = logits.new_zeros(())

        value_loss = F.mse_loss(values, targets["outcomes"])
        belief_loss = targets["belief_entropy"].mean()
        has_annotation = targets.get("has_annotation")
        if has_annotation is not None and has_annotation.any():
            distill_loss = F.mse_loss(values[has_annotation], targets["v_rlm"][has_annotation])
        elif has_annotation is None:
            distill_loss = F.mse_loss(values, targets["v_rlm"])
        else:
            distill_loss = logits.new_zeros(())
        log_probs = F.log_softmax(logits, dim=-1)
        masked_targets = normalize_masked_targets(targets["policy_targets"], action_mask)
        policy_distill = -(masked_targets * log_probs).sum(dim=-1).mean()
        loss = policy_loss + 0.5 * value_loss + 0.1 * belief_loss + 0.2 * distill_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["policy"] += float(policy_loss.detach().cpu())
        totals["value"] += float(value_loss.detach().cpu())
        totals["belief"] += float(belief_loss.detach().cpu())
        totals["distill"] += float(distill_loss.detach().cpu())
        totals["policy_distill"] += float(policy_distill.detach().cpu())
        totals["batches"] += 1.0
        if step % 50 == 0 or smoke:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "policy": float(policy_loss.detach().cpu()), "value": float(value_loss.detach().cpu())}, sort_keys=True), flush=True)
        if smoke and step >= 10:
            break
    batches = max(1.0, totals.pop("batches"))
    return {key: value / batches for key, value in totals.items()}


@torch.no_grad()
def evaluate(model: PokeNet, loader: DataLoader, device: torch.device, target_fn: Callable[[dict[str, Any], torch.device], dict[str, torch.Tensor]]) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    value_loss = 0.0
    for batch in loader:
        targets = target_fn(batch, device)
        logits, values = model(batch)
        valid = targets["actions"] >= 0
        if valid.any():
            predictions = logits.argmax(dim=-1)
            correct += int((predictions[valid] == targets["actions"][valid]).sum().item())
            total += int(valid.sum().item())
        value_loss += float(F.mse_loss(values, targets["outcomes"]).cpu())
    return {"action_accuracy": (correct / total) if total else 0.0, "value_mse": value_loss / max(1, len(loader))}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1 imitation learning + RLM value distillation")
    parser.add_argument("--annotations", default="data/annotations")
    parser.add_argument("--data-dir", default="data/parsed")
    parser.add_argument("--annotation-dir", default="data/annotations")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--pool", default="data/gen9_random_pool.json")
    parser.add_argument("--output", default="checkpoints/phase1.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-every", type=int, default=5000)
    parser.add_argument("--n-val-games", type=int, default=100)
    parser.add_argument("--embedding-path", default=None)
    parser.add_argument("--nebraskinator-path", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    vocab = build_vocabulary(args.pool)
    target_fn: Callable[[dict[str, Any], torch.device], dict[str, torch.Tensor]]
    parsed_exists = args.data_dir and Path(args.data_dir).exists()
    if parsed_exists:
        # Use streaming dataset — loads one file at a time, constant RAM
        data_dirs = [args.data_dir]
        dataset: Any = StreamingReplayDataset(
            data_dirs,
            annotation_dir=args.annotation_dir or args.annotations,
            shuffle_files=True,
            shuffle_within=True,
        )
        collate_fn = collate_replay_samples
        target_fn = replay_torch_targets
        # num_workers > 0 splits files across workers; each worker opens its own files
        loader_kwargs = {"num_workers": 4, "pin_memory": args.device.startswith("cuda")}
    else:
        dataset = Phase1AnnotationDataset(args.annotations, vocab=vocab, max_decisions=args.max_decisions)
        collate_fn = collate_decisions
        target_fn = annotation_torch_targets
        loader_kwargs = {}
    if not isinstance(dataset, torch.utils.data.IterableDataset) and len(dataset) == 0:
        raise SystemExit("No Phase 1 examples found. Parse replays first or pass --annotations to existing JSON files.")
    is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(not is_iterable), collate_fn=collate_fn, **loader_kwargs)
    device = torch.device(args.device)
    model = PokeNet(vocab=vocab).to(device)
    if args.embedding_path:
        model.initialize_from_embeddings(args.embedding_path)
    if args.nebraskinator_path:
        model.initialize_from_nebraskinator(args.nebraskinator_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr_min)
    history: list[dict[str, float]] = []
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    best_winrate = -1.0
    global_step = 0
    for epoch in range(args.epochs):
        metrics = train_epoch(model, loader, optimizer, device, target_fn, smoke=args.smoke)
        global_step += metrics.pop("steps_this_epoch", len(loader))
        # For IterableDataset we can't evaluate on the same loader (it's exhausted).
        # Skip per-epoch evaluation; checkpoint every epoch instead.
        eval_metrics: dict[str, float] = {}
        metrics["epoch"] = float(epoch + 1)
        history.append(metrics)
        scheduler.step()
        print(json.dumps(metrics, sort_keys=True), flush=True)
        pseudo_winrate = max(0.0, min(1.0, metrics.get("eval_action_accuracy", metrics.get("policy", 0.0))))
        if checkpoint_dir:
            if pseudo_winrate >= best_winrate:
                best_winrate = pseudo_winrate
            save_checkpoint(checkpoint_dir / f"epoch{epoch+1}.pt", model, optimizer, phase="phase1", history=history, val_winrate=best_winrate, step=global_step)
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, phase="phase1", history=history, val_winrate=best_winrate, step=global_step)
        if args.smoke:
            break
    output = Path(args.output) if not checkpoint_dir else checkpoint_dir / "last.pt"
    save_checkpoint(output, model, optimizer, phase="phase1", history=history, val_winrate=max(best_winrate, 0.0), step=global_step, vocab_sizes={
        "species": vocab.species_size,
        "moves": vocab.move_size,
        "items": vocab.item_size,
        "abilities": vocab.ability_size,
    })
    if checkpoint_dir and not (checkpoint_dir / "best.pt").exists():
        save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, phase="phase1", history=history, val_winrate=max(best_winrate, 0.0), step=global_step)
    print(json.dumps({"checkpoint": str(output), "best": str(checkpoint_dir / "best.pt") if checkpoint_dir else None, "steps": global_step}, indent=2))


if __name__ == "__main__":
    main()
