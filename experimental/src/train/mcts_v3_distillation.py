"""Schema-v3 MCTS visit-policy distillation.

Trains the policy's action distribution toward high-budget MCTS visit
distributions using records built by scripts/build_mcts_v3_dataset.py:
observations dumped live by the prior server at decision time, joined
fail-closed to Foul Play's visit distributions on (tag, username,
decision_idx). No replay parsing is involved.

The auxiliary loss evaluates the policy EXACTLY the way the prior server
queries it at deployment: a stateless two-step batch with a blank padded
first step, zero rl2s, and the legality mask on the real step only. The
cross-entropy therefore optimizes the same conditional distribution that
MCTS consumes as root priors.

References: AlphaZero visit-policy distillation (arXiv:1712.01815);
behavior-constrained offline RL context in docs/mcts_policy_distillation.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

NUM_ACTIONS = 13
V3_SCHEMA = 3


class V3DatasetError(ValueError):
    pass


def load_v3_records(path: Path) -> dict:
    """Load and strictly validate v3 records; returns stacked CPU tensors.

    Fail-closed: any malformed record aborts the load. Enforced per record:
    schema 3, consistent obs shapes, 13-wide legality mask and target,
    target is a distribution, no target mass on illegal actions.
    """
    import torch

    text_rows: list[list[int]] = []
    number_rows: list[list[float]] = []
    illegal_rows: list[list[bool]] = []
    target_rows: list[list[float]] = []
    text_len: int | None = None
    numbers_len: int | None = None

    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise V3DatasetError(f"{path}:{line_number}: invalid JSON") from exc
        if row.get("schema") != V3_SCHEMA:
            raise V3DatasetError(f"{path}:{line_number}: unsupported schema {row.get('schema')!r}")

        text = row.get("text_tokens")
        numbers = row.get("numbers")
        illegal = row.get("illegal_actions")
        target = row.get("visit_target_13")
        if not isinstance(text, list) or not text or not all(isinstance(t, int) for t in text):
            raise V3DatasetError(f"{path}:{line_number}: invalid text_tokens")
        if not isinstance(numbers, list) or not numbers:
            raise V3DatasetError(f"{path}:{line_number}: invalid numbers")
        if not isinstance(illegal, list) or len(illegal) != NUM_ACTIONS:
            raise V3DatasetError(f"{path}:{line_number}: invalid illegal_actions")
        if not isinstance(target, list) or len(target) != NUM_ACTIONS:
            raise V3DatasetError(f"{path}:{line_number}: invalid visit_target_13")
        if text_len is None:
            text_len, numbers_len = len(text), len(numbers)
        if len(text) != text_len or len(numbers) != numbers_len:
            raise V3DatasetError(f"{path}:{line_number}: inconsistent obs shape")
        masses = [float(value) for value in target]
        if not all(math.isfinite(m) and m >= 0.0 for m in masses):
            raise V3DatasetError(f"{path}:{line_number}: non-finite or negative target mass")
        if not math.isclose(sum(masses), 1.0, abs_tol=1e-4):
            raise V3DatasetError(f"{path}:{line_number}: target mass {sum(masses)}")
        flags = [bool(flag) for flag in illegal]
        if any(mass > 0.0 and flag for mass, flag in zip(masses, flags)):
            raise V3DatasetError(f"{path}:{line_number}: target mass on illegal action")
        if all(flags):
            raise V3DatasetError(f"{path}:{line_number}: no legal actions")
        floats = [float(value) for value in numbers]
        if not all(math.isfinite(value) for value in floats):
            raise V3DatasetError(f"{path}:{line_number}: non-finite numbers")

        text_rows.append([int(t) for t in text])
        number_rows.append(floats)
        illegal_rows.append(flags)
        target_rows.append(masses)

    if not text_rows:
        raise V3DatasetError(f"{path}: no records")
    return {
        "text_tokens": torch.tensor(text_rows, dtype=torch.int32),
        "numbers": torch.tensor(number_rows, dtype=torch.float32),
        "illegal_actions": torch.tensor(illegal_rows, dtype=torch.bool),
        "targets": torch.tensor(target_rows, dtype=torch.float32),
        "count": len(text_rows),
    }


class V3BatchSampler:
    """Shuffled without-replacement index batches, reshuffled per epoch."""

    def __init__(self, count: int, batch_size: int, seed: int = 0):
        import torch

        if count < 1 or batch_size < 1:
            raise ValueError("count and batch_size must be positive")
        self.count = count
        self.batch_size = min(batch_size, count)
        self.generator = torch.Generator().manual_seed(seed)
        self._order = torch.randperm(count, generator=self.generator)
        self._cursor = 0

    def next_indices(self):
        import torch

        if self._cursor + self.batch_size > self.count:
            self._order = torch.randperm(self.count, generator=self.generator)
            self._cursor = 0
        indices = self._order[self._cursor: self._cursor + self.batch_size]
        self._cursor += self.batch_size
        return indices


def build_stateless_batch(records: dict, indices, device):
    """Mirror prior_server.compute_priors: blank first step, real second step."""
    import torch

    text = records["text_tokens"][indices].to(device)
    numbers = records["numbers"][indices].to(device)
    illegal = records["illegal_actions"][indices].to(device)
    batch = text.shape[0]

    text = torch.stack([torch.zeros_like(text), text], dim=1)          # [B, 2, L]
    numbers = torch.stack([torch.zeros_like(numbers), numbers], dim=1)  # [B, 2, N]
    numbers = torch.nan_to_num(numbers)
    illegal_steps = torch.stack(
        [torch.ones_like(illegal), illegal], dim=1
    )                                                                    # [B, 2, A]
    obs = {"text_tokens": text, "numbers": numbers, "illegal_actions": illegal_steps}
    rl2s = torch.zeros((batch, 2, NUM_ACTIONS + 1), device=device)
    time_idxs = (
        torch.arange(2, device=device).long().unsqueeze(0).unsqueeze(-1).expand(batch, 2, 1)
    )
    targets = records["targets"][indices].to(device)
    return obs, rl2s, time_idxs, targets, illegal


def v3_distillation_terms(agent, obs, rl2s, time_idxs, targets, illegal):
    """Return (probs, targets, mask) shaped for add_distillation_loss."""
    emb, _ = agent.get_state_embedding(
        obs=obs, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None
    )
    dists = agent.actor(
        emb,
        straight_from_obs={
            key: obs[key][:, : emb.shape[1]] for key in agent.pass_obs_keys_to_actor
        },
    )
    probs = dists.probs[:, -1]                    # [B, G, A]
    target = targets.unsqueeze(1)                 # [B, 1, A]
    mask = {
        "valid": targets.new_ones((targets.shape[0], 1, 1), dtype=bool),
        "illegal_actions": illegal.unsqueeze(1),  # [B, 1, A]
    }
    return probs, target, mask


def install_mcts_v3_distillation(
    dataset_path: str, coefficient: float, batch_size: int = 64, seed: int = 0
) -> None:
    """Install the v3-distillation agent used by the finetune runner."""
    if coefficient <= 0:
        raise ValueError("MCTS v3 distillation requires a positive coefficient")
    import gin
    import metamon.rl.custom_agent as ca

    from train.mcts_policy_distillation import add_distillation_loss

    records = load_v3_records(Path(dataset_path).resolve())
    sampler = V3BatchSampler(records["count"], batch_size, seed=seed)

    class MCTSV3DistillationAgent(ca.MetamonFinetuneAgent):
        def __init__(self, *args, mcts_policy_coeff: float = coefficient, **kwargs):
            super().__init__(*args, **kwargs)
            self.mcts_policy_coeff = mcts_policy_coeff

        def forward(self, batch, log_step: bool):
            total_loss = super().forward(batch, log_step)
            if self.mcts_policy_coeff <= 0:
                return total_loss
            device = batch.rl2s.device
            indices = sampler.next_indices()
            obs, rl2s, time_idxs, targets, illegal = build_stateless_batch(
                records, indices, device
            )
            probs, target, mask = v3_distillation_terms(
                self, obs, rl2s, time_idxs, targets, illegal
            )
            before = total_loss
            total_loss = add_distillation_loss(
                total_loss, probs, target, mask, self.mcts_policy_coeff
            )
            if log_step:
                self.update_info["MCTS V3 Policy Loss"] = (total_loss - before).detach()
            return total_loss

    gin.external_configurable(MCTSV3DistillationAgent, module="custom_agent")
    ca.MCTSV3DistillationAgent = MCTSV3DistillationAgent
