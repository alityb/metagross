from __future__ import annotations

from dataclasses import dataclass

from model.state import EncodedState


@dataclass
class TrainingSample:
    battle_id: str
    turn: int
    encoded_state: EncodedState
    human_action: int
    outcome: float
    true_opponent_team: dict
    generation: int = 9
