"""Leak-free causal-history R1 policy state for certified engine continuations."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.r1_public_events import R1SwitchTracker, _canonical_action


SCHEMA = "metagross-causal-dual-r1-root/v1"
CHECKPOINT_SHA256 = "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"


class CausalDualR1Error(ValueError):
    """Raised when a causal R1 continuation state fails closed."""


def _finite_rows(rows: Sequence[Sequence[Any]]) -> bool:
    return all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        for row in rows
        for value in row
    )


@dataclass
class CausalR1PolicyState:
    """One observer's public/private state and exact causal transformer history."""

    tracker: R1SwitchTracker
    observation_rows: dict[str, list[list[Any]]]
    rl2: list[list[float]]
    time_indices: list[int]
    current_observation: dict[str, Any]
    max_seq_len: int = 128
    pending_action_index: int | None = None
    pending_source_state: Any = None

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any], observation_space: Any):
        if snapshot.get("schema") != 6:
            raise CausalDualR1Error("causal continuation requires schema-6 snapshot")
        trajectory = snapshot.get("trajectory")
        if not isinstance(trajectory, Mapping) or trajectory.get("mode") != "causal-history":
            raise CausalDualR1Error("snapshot is not causal-history R1")
        raw_observations = trajectory.get("observation_rows")
        if not isinstance(raw_observations, Mapping) or set(raw_observations) != {
            "text_tokens", "numbers", "illegal_actions"
        }:
            raise CausalDualR1Error("snapshot has no exact trajectory observations")
        observations = {
            key: [list(row) for row in raw_observations[key]]
            for key in ("text_tokens", "numbers", "illegal_actions")
        }
        lengths = {len(rows) for rows in observations.values()}
        rl2 = [list(map(float, row)) for row in trajectory.get("rl2", [])]
        times = list(trajectory.get("time_indices", []))
        if (
            len(lengths) != 1
            or not lengths
            or next(iter(lengths)) == 0
            or len(rl2) != next(iter(lengths))
            or len(times) != next(iter(lengths))
            or any(len(row) != 14 for row in rl2)
            or not _finite_rows(rl2)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in times)
            or times != list(range(times[0], times[0] + len(times)))
        ):
            raise CausalDualR1Error("snapshot trajectory is malformed")
        if (
            observations["text_tokens"][-1] != list(snapshot.get("text_tokens", []))
            or not np.array_equal(
                np.nan_to_num(np.asarray(observations["numbers"][-1], dtype=np.float32)),
                np.asarray(snapshot.get("numbers", []), dtype=np.float32),
            )
            or observations["illegal_actions"][-1]
            != list(snapshot.get("illegal_actions", []))
        ):
            raise CausalDualR1Error("snapshot current observation is not trajectory tail")
        tracker = R1SwitchTracker.from_snapshot(snapshot, observation_space)
        return cls(
            tracker=tracker,
            observation_rows=observations,
            rl2=rl2,
            time_indices=times,
            current_observation={
                "text_tokens": list(snapshot["text_tokens"]),
                "numbers": list(snapshot["numbers"]),
                "illegal_actions": list(snapshot["illegal_actions"]),
                "name_table": dict(snapshot["name_table"]),
                "terminal": False,
                "automatic_action": None,
            },
        )

    def fork(self) -> "CausalR1PolicyState":
        return copy.deepcopy(self)

    def probabilities(self, agent: Any, device: Any) -> list[float]:
        import torch

        if self.current_observation.get("automatic_action") is not None:
            raise CausalDualR1Error("automatic boundary has no R1 policy")
        if self.current_observation.get("terminal"):
            raise CausalDualR1Error("terminal boundary has no R1 policy")
        obs = {
            "text_tokens": torch.tensor(
                np.asarray(self.observation_rows["text_tokens"]),
                dtype=torch.int32,
                device=device,
            ).unsqueeze(0),
            "numbers": torch.tensor(
                np.nan_to_num(np.asarray(self.observation_rows["numbers"])),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
            "illegal_actions": torch.tensor(
                np.asarray(self.observation_rows["illegal_actions"], dtype=bool),
                device=device,
            ).unsqueeze(0),
        }
        rl2 = torch.tensor(self.rl2, dtype=torch.float32, device=device).unsqueeze(0)
        times = torch.tensor(
            np.asarray(self.time_indices, dtype=np.int64).reshape(-1, 1),
            device=device,
        ).long().unsqueeze(0)
        with torch.no_grad():
            embedding, _ = agent.get_state_embedding(
                obs=obs, rl2s=rl2, time_idxs=times, hidden_state=None
            )
            distribution = agent.actor(
                embedding,
                straight_from_obs={
                    key: obs[key][:, : embedding.shape[1]]
                    for key in agent.pass_obs_keys_to_actor
                },
            )
        probabilities = distribution.probs[0, -1, -1, :].detach().cpu().numpy()
        illegal = np.asarray(self.current_observation["illegal_actions"], dtype=bool)
        probabilities = probabilities * (~illegal)
        total = float(probabilities.sum())
        if not np.isfinite(probabilities).all() or not math.isfinite(total) or total <= 0:
            raise CausalDualR1Error("causal R1 produced invalid probabilities")
        return [float(value / total) for value in probabilities]

    def action_support(
        self, probabilities: Sequence[float] | None
    ) -> list[tuple[str, float]]:
        automatic = self.current_observation.get("automatic_action")
        if automatic is not None:
            if automatic != "nomove" or probabilities is not None:
                raise CausalDualR1Error("invalid automatic continuation boundary")
            return [(automatic, 1.0)]
        if probabilities is None:
            raise CausalDualR1Error("learned continuation boundary has no policy")
        if len(probabilities) != 13:
            raise CausalDualR1Error("causal R1 probability vector has wrong size")
        illegal = self.current_observation["illegal_actions"]
        names = self.current_observation["name_table"]
        by_index: dict[int, str] = {}
        for name, index in names.items():
            canonical = _canonical_action(name)
            if index in by_index or canonical in by_index.values():
                raise CausalDualR1Error("causal R1 action table is not one-to-one")
            by_index[int(index)] = canonical
        support = []
        for index, probability in enumerate(probabilities):
            if probability <= 0:
                continue
            if illegal[index] or index not in by_index:
                raise CausalDualR1Error("causal R1 assigned mass outside legal support")
            support.append((by_index[index], float(probability)))
        if not support:
            raise CausalDualR1Error("causal R1 has empty legal support")
        return support

    def advance(
        self,
        next_tracker: R1SwitchTracker,
        observation: Mapping[str, Any],
        selected_action: str,
        reward_function: Any,
    ) -> None:
        if hasattr(observation, "policy_payload"):
            observation = observation.policy_payload()
        canonical = _canonical_action(selected_action)
        was_automatic = self.current_observation.get("automatic_action")
        next_automatic = observation.get("automatic_action")
        if was_automatic is None:
            action_indices = {
                _canonical_action(name): int(index)
                for name, index in self.current_observation["name_table"].items()
            }
            if canonical not in action_indices:
                raise CausalDualR1Error("selected action is absent from causal R1 support")
            if self.pending_action_index is not None or self.pending_source_state is not None:
                raise CausalDualR1Error("causal R1 has conflicting deferred transition")
            pending_action_index = action_indices[canonical]
            pending_source_state = self.tracker.state
        else:
            if was_automatic != "nomove" or canonical != "nomove":
                raise CausalDualR1Error("automatic continuation selected a learned action")
            if self.pending_action_index is None or self.pending_source_state is None:
                raise CausalDualR1Error("automatic continuation lost its deferred transition")
            pending_action_index = self.pending_action_index
            pending_source_state = self.pending_source_state

        self.tracker = next_tracker
        self.current_observation = {
            "text_tokens": list(observation["text_tokens"]),
            "numbers": list(observation["numbers"]),
            "illegal_actions": list(observation["illegal_actions"]),
            "name_table": dict(observation["name_table"]),
            "terminal": bool(observation.get("terminal", False)),
            "automatic_action": next_automatic,
        }
        if next_automatic is not None:
            self.pending_action_index = pending_action_index
            self.pending_source_state = pending_source_state
            return

        reward = float(reward_function(pending_source_state, next_tracker.state))
        if not math.isfinite(reward):
            raise CausalDualR1Error("causal R1 transition reward is non-finite")
        for key in ("text_tokens", "numbers", "illegal_actions"):
            self.observation_rows[key].append(list(observation[key]))
        row = [0.0] * 14
        row[0] = reward
        row[1 + pending_action_index] = 1.0
        self.rl2.append(row)
        self.time_indices.append(self.time_indices[-1] + 1)
        if len(self.time_indices) > self.max_seq_len:
            for key in self.observation_rows:
                self.observation_rows[key] = self.observation_rows[key][-self.max_seq_len :]
            self.rl2 = self.rl2[-self.max_seq_len :]
            self.time_indices = self.time_indices[-self.max_seq_len :]
        self.pending_action_index = None
        self.pending_source_state = None
