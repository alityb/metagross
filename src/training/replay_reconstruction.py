from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from model.state import EncodedState, Vocabulary, encode_state, stack_encoded


@dataclass
class DecisionExample:
    state: EncodedState
    action: int
    outcome: float
    v_rlm: float
    policy: np.ndarray
    belief_entropy: float
    replay_id: str
    turn: int


def iter_annotation_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(file for file in root.glob("*.json") if file.is_file())


def _opponent_team_from_posterior(posterior: dict[str, Any], state_meta: dict[str, Any]) -> list[dict[str, Any]]:
    revealed = state_meta.get("revealed_opponent") or {}
    team: list[dict[str, Any]] = []
    for slot in sorted(set(revealed) | set(posterior))[:6]:
        species = revealed.get(slot)
        entries = posterior.get(slot) or []
        top = entries[0] if entries and isinstance(entries[0], dict) else {}
        team.append(
            {
                "species": species or top.get("species"),
                "moves": top.get("moves", []),
                "item": top.get("item"),
                "ability": top.get("ability"),
                "is_active": slot.endswith("a"),
            }
        )
    return team


def _state_from_decision(decision: dict[str, Any], vocab: Vocabulary) -> EncodedState:
    if "encoded_state" in decision:
        raw = decision["encoded_state"]
        return EncodedState(
            species_ids=np.asarray(raw["species_ids"], dtype=np.int64),
            move_ids=np.asarray(raw["move_ids"], dtype=np.int64),
            item_ids=np.asarray(raw["item_ids"], dtype=np.int64),
            ability_ids=np.asarray(raw["ability_ids"], dtype=np.int64),
            pokemon_dense=np.asarray(raw["pokemon_dense"], dtype=np.float32),
            field=np.asarray(raw["field"], dtype=np.float32),
            active_indices=np.asarray(raw["active_indices"], dtype=np.int64),
            action_mask=np.asarray(raw["action_mask"], dtype=np.bool_),
        )
    state_meta = decision.get("state") or {}
    reconstructed = {
        "turn": state_meta.get("turn", 0),
        "own_team": state_meta.get("own_team", []),
        "opponent_team": _opponent_team_from_posterior(decision.get("posterior") or {}, state_meta),
        "available_moves": state_meta.get("available_moves", [{"move": i, "disabled": False} for i in range(4)]),
        "available_switches": state_meta.get("available_switches", []),
        "can_tera": state_meta.get("can_tera", False),
    }
    return encode_state(reconstructed, vocab=vocab)


def _action_from_decision(decision: dict[str, Any]) -> int:
    for key in ("action", "human_action", "action_index"):
        if key in decision:
            try:
                value = int(decision[key])
            except (TypeError, ValueError):
                return -1
            return value if 0 <= value < 14 else -1
    return -1


def _policy_from_decision(decision: dict[str, Any]) -> np.ndarray:
    raw = decision.get("policy") or decision.get("pi_rlm") or []
    policy = np.zeros(14, dtype=np.float32)
    for idx, value in enumerate(list(raw)[:14]):
        try:
            policy[idx] = max(0.0, float(value))
        except (TypeError, ValueError):
            policy[idx] = 0.0
    total = float(policy.sum())
    if total <= 0:
        policy[:] = 1.0 / 14.0
    else:
        policy /= total
    return policy


def _belief_entropy(decision: dict[str, Any]) -> float:
    entropies: list[float] = []
    posterior = decision.get("posterior") or {}
    if not isinstance(posterior, dict):
        return 0.0
    for entries in posterior.values():
        if not isinstance(entries, list) or not entries:
            continue
        probs = []
        for entry in entries:
            if isinstance(entry, dict):
                try:
                    probs.append(max(0.0, float(entry.get("probability", 0.0))))
                except (TypeError, ValueError):
                    continue
        total = sum(probs)
        if total <= 0:
            continue
        normalized = [prob / total for prob in probs]
        entropies.append(-sum(prob * math.log(max(prob, 1e-12)) for prob in normalized))
    return float(sum(entropies) / len(entropies)) if entropies else 0.0


class Phase1AnnotationDataset(Dataset[DecisionExample]):
    def __init__(self, annotations: str | Path, vocab: Vocabulary, max_decisions: int | None = None):
        self.examples: list[DecisionExample] = []
        for path in iter_annotation_paths(annotations):
            annotation = json.loads(path.read_text())
            replay_id = annotation.get("replay_id", path.stem)
            winner = annotation.get("winner")
            outcome = 1.0 if winner else 0.0
            for turn_text, decision in sorted(annotation.get("decisions", {}).items(), key=lambda item: int(item[0])):
                turn = int(turn_text)
                v_rlm = float(decision.get("V_rlm", 0.0))
                self.examples.append(
                    DecisionExample(
                        state=_state_from_decision(decision, vocab),
                        action=_action_from_decision(decision),
                        outcome=float(decision.get("outcome", outcome if winner is not None else v_rlm)),
                        v_rlm=v_rlm,
                        policy=_policy_from_decision(decision),
                        belief_entropy=_belief_entropy(decision),
                        replay_id=replay_id,
                        turn=turn,
                    )
                )
                if max_decisions is not None and len(self.examples) >= max_decisions:
                    return

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DecisionExample:
        return self.examples[index]


def collate_decisions(examples: Iterable[DecisionExample]) -> dict[str, Any]:
    batch = list(examples)
    model_batch = stack_encoded([example.state for example in batch])
    model_batch["actions"] = np.asarray([example.action for example in batch], dtype=np.int64)
    model_batch["outcomes"] = np.asarray([example.outcome for example in batch], dtype=np.float32)
    model_batch["v_rlm"] = np.asarray([example.v_rlm for example in batch], dtype=np.float32)
    model_batch["policy_targets"] = np.stack([example.policy for example in batch])
    model_batch["belief_entropy"] = np.asarray([example.belief_entropy for example in batch], dtype=np.float32)
    model_batch["metadata"] = [(example.replay_id, example.turn) for example in batch]
    return model_batch


def torch_targets(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "actions": torch.as_tensor(batch["actions"], dtype=torch.long, device=device),
        "outcomes": torch.as_tensor(batch["outcomes"], dtype=torch.float32, device=device),
        "v_rlm": torch.as_tensor(batch["v_rlm"], dtype=torch.float32, device=device),
        "policy_targets": torch.as_tensor(batch["policy_targets"], dtype=torch.float32, device=device),
        "belief_entropy": torch.as_tensor(batch["belief_entropy"], dtype=torch.float32, device=device),
    }
