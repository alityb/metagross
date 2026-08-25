"""Leak-free batched value inference for certified interior observation classes."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from train.transformer_terminal_value import (
    NUM_ACTIONS,
    RL2_WIDTH,
    TransformerValueHead,
    frozen_r1_embeddings,
)


class TransformerValueOracleError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TransformerValueOracle:
    """Attach the terminal head to an exact, frozen r1 policy instance."""

    def __init__(
        self,
        agent: Any,
        head_path: Path,
        *,
        expected_base_sha256: str,
        expected_head_sha256: str | None = None,
    ):
        head_path = Path(head_path)
        if expected_head_sha256 is not None and _sha256(head_path) != expected_head_sha256:
            raise TransformerValueOracleError("value-head SHA-256 mismatch")
        payload = torch.load(head_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "metagross-transformer-terminal-value/v1":
            raise TransformerValueOracleError("unsupported value-head schema")
        provenance, architecture = payload.get("provenance", {}), payload.get("architecture", {})
        if provenance.get("base_checkpoint_sha256") != expected_base_sha256:
            raise TransformerValueOracleError("value head belongs to a different r1 checkpoint")
        try:
            self.head = TransformerValueHead(
                int(architecture["embedding_dim"]),
                int(architecture["hidden_dim"]),
                float(architecture["dropout"]),
            )
            self.head.load_state_dict(payload["state_dict"], strict=True)
        except (KeyError, TypeError, ValueError, RuntimeError):
            raise TransformerValueOracleError("invalid value-head architecture") from None
        self.agent = agent
        self.device = next(agent.parameters()).device
        self.head.to(self.device).eval()
        if sum(parameter.numel() for parameter in self.head.parameters()) != architecture.get("parameters"):
            raise TransformerValueOracleError("value-head parameter count mismatch")

    @torch.no_grad()
    def predict(
        self,
        obs: Mapping[str, torch.Tensor],
        rl2: torch.Tensor,
        time_idxs: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = frozen_r1_embeddings(self.agent, obs, rl2, time_idxs)
        if embeddings.shape[-1] != self.head.net[0].normalized_shape[0]:
            raise TransformerValueOracleError("r1 embedding width does not match value head")
        return self.head(embeddings)[:, -1].sigmoid()


def append_branch_observations(
    prefix_obs: Mapping[str, torch.Tensor],
    prefix_rl2: torch.Tensor,
    observations: Sequence[Any],
    previous_action_indices: Sequence[int],
    previous_rewards: Sequence[float],
    *,
    max_seq_len: int = 128,
    device: torch.device | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Append certified public observations to a shared transformer prefix.

    ``observations`` must expose ``policy_payload()``. Mechanical search states
    are neither accepted nor inspected by this function.
    """
    if (
        not observations
        or len(observations) != len(previous_action_indices)
        or len(observations) != len(previous_rewards)
    ):
        raise TransformerValueOracleError("unaligned branch inputs")
    if max_seq_len < 2:
        raise TransformerValueOracleError("transformer context must hold root and child")
    required = {"text_tokens", "numbers", "illegal_actions"}
    if set(prefix_obs) != required or prefix_rl2.ndim != 2 or prefix_rl2.shape[-1] != RL2_WIDTH:
        raise TransformerValueOracleError("invalid transformer prefix")
    prefix_length = prefix_rl2.shape[0]
    if any(value.ndim != 2 or value.shape[0] != prefix_length for value in prefix_obs.values()):
        raise TransformerValueOracleError("unaligned transformer prefix")
    target_device = device or prefix_rl2.device
    start = max(0, prefix_length + 1 - max_seq_len)
    kept = prefix_length - start
    batch = len(observations)
    obs_batch = {
        key: prefix_obs[key][start:].to(target_device).unsqueeze(0).expand(batch, -1, -1).clone()
        for key in required
    }
    rl2 = prefix_rl2[start:].to(target_device).unsqueeze(0).expand(batch, -1, -1).clone()
    for index, (observation, action, reward) in enumerate(
        zip(observations, previous_action_indices, previous_rewards, strict=True)
    ):
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < NUM_ACTIONS:
            raise TransformerValueOracleError("invalid previous action")
        if not math.isfinite(float(reward)):
            raise TransformerValueOracleError("invalid previous reward")
        try:
            payload = observation.policy_payload()
        except AttributeError:
            raise TransformerValueOracleError("branch is not a transformer observation") from None
        if not isinstance(payload, Mapping) or not required.issubset(payload):
            raise TransformerValueOracleError("invalid transformer observation payload")
    output_obs: dict[str, torch.Tensor] = {}
    for key, dtype in (("text_tokens", torch.int32), ("numbers", torch.float32), ("illegal_actions", torch.bool)):
        children = torch.stack(
            [torch.tensor(observation.policy_payload()[key], dtype=dtype, device=target_device) for observation in observations]
        )
        if children.shape[1:] != prefix_obs[key].shape[1:]:
            raise TransformerValueOracleError("branch observation shape changed")
        output_obs[key] = torch.cat((obs_batch[key], children.unsqueeze(1)), dim=1)
    child_rl2 = torch.zeros((batch, 1, RL2_WIDTH), dtype=torch.float32, device=target_device)
    for index, (action, reward) in enumerate(zip(previous_action_indices, previous_rewards, strict=True)):
        child_rl2[index, 0, 0] = float(reward)
        child_rl2[index, 0, 1 + action] = 1.0
    rl2 = torch.cat((rl2, child_rl2), dim=1)
    time_idxs = torch.arange(kept + 1, device=target_device).view(1, kept + 1, 1).expand(batch, -1, -1)
    return output_obs, rl2, time_idxs
