"""Prospective one-deviation randomization for the terminal-MCTS teacher.

This module deliberately contains no battle simulator.  It assigns each fresh
game to one frozen arm before play, observes the first *certified* teacher
deviation, applies at most that one deviation, and then locks the controller so
the production continuation is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


SCHEMA = "metagross-terminal-mcts-one-deviation/v1"
EQUAL8192_SCHEMA = "metagross-terminal-mcts-one-deviation/v2"
ASSIGNMENT_SCHEMA = "metagross-terminal-mcts-one-deviation-assignment/v1"
EQUAL8192_CONTRACT = "cycle41_equal8192"
CANARY_GAMES = 20
CANARY_PAIRS = CANARY_GAMES // 2


class OneDeviationProtocolError(RuntimeError):
    """Raised when the frozen randomization or teacher contract is violated."""


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def frozen_assignment_schedule(seed: str, total_pairs: int = CANARY_PAIRS) -> dict[int, int]:
    """Return pair -> teacher leg, exactly balanced across the two mirror legs.

    Pair scores are fixed by SHA-256 before any battle is played.  The lowest
    half receive the teacher in leg 1; the rest receive it in leg 2.
    """
    if not seed:
        raise OneDeviationProtocolError("randomization seed must be non-empty")
    if total_pairs < 2 or total_pairs % 2:
        raise OneDeviationProtocolError("total_pairs must be a positive even integer")
    ranked = sorted(
        range(1, total_pairs + 1),
        key=lambda pair: (
            hashlib.sha256(f"{seed}\0pair\0{pair}".encode("utf-8")).digest(),
            pair,
        ),
    )
    leg_one_pairs = set(ranked[: total_pairs // 2])
    return {
        pair: 1 if pair in leg_one_pairs else 2
        for pair in range(1, total_pairs + 1)
    }


def assignment_manifest(seed: str, total_pairs: int = CANARY_PAIRS) -> dict[str, Any]:
    schedule = frozen_assignment_schedule(seed, total_pairs)
    rows = []
    for pair_index in range(1, total_pairs + 1):
        teacher_leg = schedule[pair_index]
        for pair_leg in (1, 2):
            game_index = pair_index * 2 - 2 + pair_leg
            rows.append(
                {
                    "game_index": game_index,
                    "pair_index": pair_index,
                    "pair_leg": pair_leg,
                    "arm": "teacher" if pair_leg == teacher_leg else "production",
                }
            )
    manifest = {
        "schema": ASSIGNMENT_SCHEMA,
        "seed": seed,
        "total_games": total_pairs * 2,
        "total_pairs": total_pairs,
        "assignments": rows,
    }
    manifest["schedule_sha256"] = _digest(manifest)
    return manifest


def assignment_for_username(
    username: str,
    *,
    username_prefix: str,
    seed: str,
    total_pairs: int = CANARY_PAIRS,
) -> dict[str, Any]:
    """Recover the pre-play assignment from the evaluator's frozen username."""
    match = re.fullmatch(
        rf"{re.escape(username_prefix)}([xy])(\d{{3}})[0-9a-f]{{4}}",
        username,
    )
    if match is None:
        raise OneDeviationProtocolError(
            f"username does not match the frozen evaluator identity: {username!r}"
        )
    role, index_text = match.groups()
    game_index = int(index_text)
    if not 1 <= game_index <= total_pairs * 2:
        raise OneDeviationProtocolError("game index is outside the frozen canary")
    pair_index = (game_index + 1) // 2
    pair_leg = 1 if game_index % 2 else 2
    expected_role = "x" if pair_leg == 1 else "y"
    if role != expected_role:
        raise OneDeviationProtocolError(
            "username role is inconsistent with the mirrored-pair schedule"
        )
    manifest = assignment_manifest(seed, total_pairs)
    row = next(
        item for item in manifest["assignments"] if item["game_index"] == game_index
    )
    return {
        **row,
        "randomization_seed": seed,
        "schedule_sha256": manifest["schedule_sha256"],
        "username": username,
    }


@dataclass
class _GameState:
    assignment: dict[str, Any]
    teacher_queries: int = 0
    locked: bool = False
    integrity_failure: str | None = None


class OneDeviationController:
    """State machine enforcing one prospective teacher opportunity per game."""

    def __init__(
        self,
        *,
        seed: str,
        username_prefix: str,
        total_pairs: int = CANARY_PAIRS,
        teacher_contract: str = "legacy_terminal_mcts",
    ):
        if not re.fullmatch(r"[a-z0-9]{1,8}", username_prefix):
            raise OneDeviationProtocolError("username_prefix is invalid")
        # Materialize once so malformed configuration fails before a game.
        assignment_manifest(seed, total_pairs)
        self.seed = seed
        self.username_prefix = username_prefix
        self.total_pairs = total_pairs
        if teacher_contract not in {"legacy_terminal_mcts", EQUAL8192_CONTRACT}:
            raise OneDeviationProtocolError("unknown teacher contract")
        self.teacher_contract = teacher_contract
        self._games: dict[tuple[str, str], _GameState] = {}

    def _state(self, battle_tag: str, username: str) -> _GameState:
        if not battle_tag:
            raise OneDeviationProtocolError("battle tag must be non-empty")
        key = (battle_tag, username)
        if key not in self._games:
            self._games[key] = _GameState(
                assignment_for_username(
                    username,
                    username_prefix=self.username_prefix,
                    seed=self.seed,
                    total_pairs=self.total_pairs,
                )
            )
        return self._games[key]

    def should_query(self, battle_tag: str, username: str) -> bool:
        return not self._state(battle_tag, username).locked

    def observe(
        self,
        *,
        battle_tag: str,
        username: str,
        decision_index: int,
        production_choice: str,
        teacher: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Consume one teacher result and return the real action plus audit row."""
        state = self._state(battle_tag, username)
        if state.locked:
            raise OneDeviationProtocolError("teacher queried after the game was locked")
        state.teacher_queries += 1
        assignment = dict(state.assignment)
        base = {
            "schema": EQUAL8192_SCHEMA if self.teacher_contract == EQUAL8192_CONTRACT else SCHEMA,
            "teacher_contract": self.teacher_contract,
            "assignment": assignment,
            "battle_tag": battle_tag,
            "decision_idx": int(decision_index),
            "teacher_query_index": state.teacher_queries,
            "production_action": production_choice,
            "continuation": "unchanged_500ms_r1_production_search",
        }
        if teacher.get("schema") != "metagross-terminal-mcts-live-decision/v1":
            state.locked = True
            state.integrity_failure = "invalid_teacher_schema"
            return production_choice, {
                **base,
                "eligible": False,
                "intervention_applied": False,
                "locked_after_decision": True,
                "integrity_failure": state.integrity_failure,
            }
        decision = teacher.get("decision")
        reason = str(teacher.get("reason", ""))
        if decision == "abstain" and not reason.startswith("fail_closed:"):
            return production_choice, {
                **base,
                "eligible": False,
                "intervention_applied": False,
                "locked_after_decision": False,
                "integrity_failure": None,
            }
        if decision == "abstain":
            # A failed teacher call makes first-opportunity ascertainment unknown.
            # Continue the real game with production, but make the canary invalid.
            state.locked = True
            state.integrity_failure = reason or "teacher_fail_closed"
            return production_choice, {
                **base,
                "eligible": False,
                "intervention_applied": False,
                "locked_after_decision": True,
                "integrity_failure": state.integrity_failure,
            }
        teacher_action = teacher.get("selected_action")
        if self.teacher_contract == EQUAL8192_CONTRACT:
            baseline_action = teacher.get("production_action")
            controller_ok = (
                teacher.get("controller_schema")
                == "metagross-cycle19-equal8192-production-selector/v1"
                and teacher.get("iterations_per_world") == 8192
                and teacher.get("schedule_count") == 2
                and teacher.get("world_count") == 16
                and isinstance(teacher.get("receipts"), list)
                and len(teacher["receipts"]) == 16
                and all(
                    isinstance(receipt, dict)
                    and receipt.get("total_visits") == 8192
                    for receipt in teacher["receipts"]
                )
            )
        else:
            baseline_action = teacher.get("baseline_action")
            controller_ok = True
        if (
            decision != "override"
            or not controller_ok
            or not isinstance(teacher_action, str)
            or not teacher_action
            or teacher_action == production_choice
            or baseline_action != production_choice
        ):
            state.locked = True
            state.integrity_failure = "invalid_certified_deviation"
            return production_choice, {
                **base,
                "eligible": False,
                "intervention_applied": False,
                "locked_after_decision": True,
                "integrity_failure": state.integrity_failure,
            }

        # The first certified deviation is the sole randomized opportunity.
        state.locked = True
        teacher_digest = _digest(teacher)
        apply_teacher = assignment["arm"] == "teacher"
        final_choice = teacher_action if apply_teacher else production_choice
        return final_choice, {
            **base,
            "eligible": True,
            "teacher_action": teacher_action,
            "teacher_decision_sha256": teacher_digest,
            "intervention_applied": apply_teacher,
            "locked_after_decision": True,
            "integrity_failure": None,
        }


__all__ = [
    "ASSIGNMENT_SCHEMA",
    "CANARY_GAMES",
    "CANARY_PAIRS",
    "EQUAL8192_CONTRACT",
    "EQUAL8192_SCHEMA",
    "OneDeviationController",
    "OneDeviationProtocolError",
    "SCHEMA",
    "assignment_for_username",
    "assignment_manifest",
    "frozen_assignment_schedule",
]
