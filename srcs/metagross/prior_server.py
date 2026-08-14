#!/usr/bin/env python3
"""Prior server: hosts the fine-tuned 142M policy + metamon live battle tracking,
and serves per-turn root priors to Foul Play over localhost HTTP.

Data flow (FP side patches in run_foul_play.py, METAGROSS_PRIOR_SERVER):
  POST /lines  {"tag": ..., "lines": [...]}   raw protocol lines (incl. |request|)
  GET  /priors?tag=...                        -> {"priors": {engine_move_str: prob}}
  POST /action {request identity + chosen action} exact causal action boundary
  POST /end    {"tag": ...}                   cleanup

The final chosen action is acknowledged before Foul Play returns it to the
Showdown sender. Public battle events are observations, never action labels.

This module intentionally contains only the accepted r1 inference path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import logging
import math
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[2]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def canonical_request_sha256(request: dict) -> str:
    return hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fresh_observation_space(template):
    """Return an episode-owned copy of the loaded model observation space."""
    observation_space = copy.deepcopy(template)
    observation_space.reset()
    return observation_space


def aligned_trajectory_arrays(
    observations: list[dict],
    actions: list[int],
    rewards: list[float],
    max_seq_len: int,
    time_offset: int = 0,
) -> tuple[dict[str, object], object, object]:
    """Build AMAGO's reward-first RL2 sequence from aligned request transitions."""
    import numpy as np

    if (
        not observations
        or len(actions) != len(rewards)
        or len(observations) != len(actions) + 1
    ):
        raise RuntimeError("policy trajectory is not request-aligned")
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len < 2:
        raise ValueError("policy maximum sequence length is invalid")
    if isinstance(time_offset, bool) or not isinstance(time_offset, int) or time_offset < 0:
        raise ValueError("policy trajectory time offset is invalid")
    for action in actions:
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < 13:
            raise RuntimeError("policy trajectory has an invalid action")
    for reward in rewards:
        if not math.isfinite(float(reward)):
            raise RuntimeError("policy trajectory has a non-finite reward")

    length = len(observations)
    rl2 = np.zeros((length, 14), dtype=np.float32)
    for index, (action, reward) in enumerate(zip(actions, rewards, strict=True), start=1):
        rl2[index, 0] = float(reward)
        rl2[index, 1 + action] = 1.0
    start = max(0, length - max_seq_len)
    batch = {
        key: np.stack([observation[key] for observation in observations[start:]])
        for key in ("text_tokens", "numbers", "illegal_actions")
    }
    time_indices = np.arange(
        time_offset + start, time_offset + length, dtype=np.int64
    ).reshape(-1, 1)
    return batch, rl2[start:], time_indices


def legacy_stateless_trajectory_arrays(current_observation: dict) -> tuple[
    dict[str, object], object, object
]:
    """Reproduce the accepted pre-repair two-step inference input exactly."""
    import numpy as np

    text = np.asarray(current_observation["text_tokens"])
    numbers = np.asarray(current_observation["numbers"])
    illegal = np.asarray(current_observation["illegal_actions"], dtype=bool)
    if illegal.shape != (13,):
        raise RuntimeError("legacy stateless legality mask has invalid shape")
    batch = {
        "text_tokens": np.stack([np.zeros_like(text), text]),
        "numbers": np.stack([np.zeros_like(numbers), numbers]),
        "illegal_actions": np.stack([np.ones(13, dtype=bool), illegal]),
    }
    rl2 = np.zeros((2, 14), dtype=np.float32)
    time_indices = np.arange(2, dtype=np.int64).reshape(-1, 1)
    return batch, rl2, time_indices


def request_action_support(request: object) -> dict[str, object]:
    """Derive exact own-side action support from a private Showdown request."""
    if not isinstance(request, dict):
        raise RuntimeError("battle request is not an object")
    rqid = request.get("rqid")
    if isinstance(rqid, bool) or not isinstance(rqid, int) or rqid < 0:
        raise RuntimeError("battle request has an invalid rqid")
    force_switch_rows = request.get("forceSwitch", [False])
    if not isinstance(force_switch_rows, list) or not force_switch_rows:
        raise RuntimeError("battle request has invalid forceSwitch metadata")
    force_switch = force_switch_rows[0]
    if not isinstance(force_switch, bool):
        raise RuntimeError("battle request has invalid forceSwitch metadata")
    active_rows = request.get("active", [])
    if active_rows is None:
        active_rows = []
    if not isinstance(active_rows, list):
        raise RuntimeError("battle request has invalid active metadata")
    active = active_rows[0] if active_rows else {}
    if not isinstance(active, dict):
        raise RuntimeError("battle request has invalid active metadata")
    trapped = active.get("trapped", False)
    if not isinstance(trapped, bool):
        raise RuntimeError("battle request has invalid trapped metadata")
    can_tera = active.get("canTerastallize", False)
    if can_tera is None:
        can_tera = False
    if not isinstance(can_tera, (bool, str)):
        raise RuntimeError("battle request has invalid Tera metadata")
    can_tera = bool(can_tera)

    actions: set[str] = set()
    if not force_switch:
        moves = active.get("moves", [])
        if not isinstance(moves, list):
            raise RuntimeError("battle request has invalid move metadata")
        for move in moves:
            if not isinstance(move, dict):
                raise RuntimeError("battle request has invalid move metadata")
            move_id = norm(move.get("id", ""))
            disabled = move.get("disabled", False)
            if not isinstance(disabled, bool):
                raise RuntimeError("battle request has invalid disabled metadata")
            pp = move.get("pp")
            if isinstance(pp, bool) or (pp is not None and not isinstance(pp, int)):
                raise RuntimeError("battle request has invalid PP metadata")
            if move_id and not disabled and pp != 0:
                actions.add(move_id)
                if can_tera:
                    actions.add(f"{move_id}-tera")

    side = request.get("side")
    if not isinstance(side, dict) or not isinstance(side.get("pokemon"), list):
        raise RuntimeError("battle request has invalid side metadata")
    if force_switch or not trapped:
        for pokemon in side["pokemon"]:
            if not isinstance(pokemon, dict):
                raise RuntimeError("battle request has invalid Pokemon metadata")
            if pokemon.get("active") is True:
                continue
            condition = pokemon.get("condition")
            if not isinstance(condition, str):
                raise RuntimeError("battle request has invalid Pokemon condition")
            hp_text = condition.split(" ", 1)[0].split("/", 1)[0]
            fainted = condition.endswith(" fnt") or hp_text == "0"
            if fainted:
                continue
            details = pokemon.get("details")
            if not isinstance(details, str) or not details:
                raise RuntimeError("battle request has invalid Pokemon details")
            species = norm(details.split(",", 1)[0])
            if species:
                actions.add(f"switch {species}")
    if not actions:
        raise RuntimeError("battle request contains no legal actions")
    return {
        "authority": "private_showdown_request",
        "rqid": rqid,
        "force_switch": force_switch,
        "trapped": trapped,
        "can_tera": can_tera,
        "actions": sorted(actions),
    }


FORCED_SHOWDOWN_ACTIONS = frozenset({"recharge", "struggle"})
UNLEARNED_REQUEST_ACTIONS = frozenset({"struggle"})


def forced_showdown_action(actions: object) -> str | None:
    """Return a sole automatic action that is outside the learned 13 actions."""
    if not isinstance(actions, (list, tuple)) or len(actions) != 1:
        return None
    action = actions[0]
    if not isinstance(action, str) or action not in FORCED_SHOWDOWN_ACTIONS:
        return None
    return action


def add_unlearned_action_priors(
    priors: dict[str, float], request_actions: set[str]
) -> dict[str, float]:
    """Give request-only actions an uninformative prior and renormalize."""
    missing = request_actions.intersection(UNLEARNED_REQUEST_ACTIONS).difference(priors)
    if not missing:
        return dict(priors)
    if not priors:
        raise RuntimeError("unlearned request action has no learned alternatives")
    neutral = sum(priors.values()) / len(priors)
    completed = dict(priors)
    completed.update({action: neutral for action in missing})
    total = sum(completed.values())
    return {action: probability / total for action, probability in completed.items()}


def private_request_move_name_table(request_actions: set[str]) -> dict[str, int]:
    """Map authoritative request moves to Metamon's alphabetical action slots."""
    move_ids = sorted({
        action.removesuffix("-tera")
        for action in request_actions
        if not action.startswith("switch ")
        and action not in UNLEARNED_REQUEST_ACTIONS
    })
    if len(move_ids) > 4:
        raise RuntimeError("private request contains more than four learned moves")
    table: dict[str, int] = {}
    for index, move_id in enumerate(move_ids):
        table[move_id] = index
        tera_action = f"{move_id}-tera"
        if tera_action in request_actions:
            table[tera_action] = index + 9
    return table


def private_request_switch_name_table(request_actions: set[str]) -> dict[str, int]:
    """Map authoritative request switches to Metamon's alphabetical slots."""
    switch_actions = sorted(
        action for action in request_actions if action.startswith("switch ")
    )
    if len(switch_actions) > 5:
        raise RuntimeError("private request contains more than five legal switches")
    return {action: index + 4 for index, action in enumerate(switch_actions)}


def request_action_policy_indices(
    action: str, name_table: dict[str, int]
) -> tuple[int, ...]:
    """Return every R1 policy slot that executes one Showdown request action."""
    if action == "struggle":
        # Metamon presents all four ordinary move slots and maps each to
        # Struggle, while replay supervision canonically records index 0.
        return (0, 1, 2, 3)
    index = name_table.get(action)
    if index is None:
        raise RuntimeError(f"request action is absent from policy table: {action}")
    return (index,)


def reconcile_private_active_pokemon(battle, request: object) -> bool:
    """Make the live tracker honor our private side identity under Illusion.

    Public switch protocol intentionally carries Zoroark's disguise.  The
    private request carries our real active party member, but Metamon's online
    request updater historically used that object only for available moves and
    left the public disguise in the active slot.  Replace that one slot from
    the authoritative request; no opponent information is involved.
    """
    if not isinstance(request, dict):
        return False
    side = request.get("side")
    if not isinstance(side, dict):
        return False
    active_rows = [
        row for row in side.get("pokemon", [])
        if isinstance(row, dict) and row.get("active") is True
    ]
    if len(active_rows) != 1:
        return False
    details = active_rows[0].get("details")
    role = getattr(battle, "player_role", None) or getattr(
        battle, "_player_role", None
    )
    if not isinstance(details, str) or role not in {"p1", "p2"}:
        return False
    turn = battle._current_turn
    player_one = role == "p1"
    active_slot = turn.get_active_pokemon(player_one)
    if not isinstance(active_slot, list) or len(active_slot) != 1:
        return False
    request_pokemon = battle._sim_protocol.get_or_create_pokemon_from_details(
        details=details,
        poke_list=turn.get_pokemon(player_one),
    )
    tracked = active_slot[0]
    if tracked is not None and tracked.unique_id == request_pokemon.unique_id:
        return False
    active_slot[0] = request_pokemon
    return True


def refresh_private_active_moves(battle, request: object) -> None:
    """Apply the authoritative move request after private identity repair.

    Metamon applies the active request before we reconcile an Illusion or other
    private/public identity mismatch.  Reapplying it to the now-current active
    Pokemon prevents stale accumulated moves from shifting the learned four
    move action slots.
    """
    if not isinstance(request, dict):
        return
    active_rows = request.get("active")
    role = getattr(battle, "player_role", None) or getattr(
        battle, "_player_role", None
    )
    active_slot = (
        battle._current_turn.get_active_pokemon(role == "p1")
        if role in {"p1", "p2"}
        else None
    )
    active_pokemon = (
        active_slot[0]
        if isinstance(active_slot, list) and len(active_slot) == 1
        else None
    )
    updater = getattr(battle, "_update_turn_from_active_request", None)
    if (
        not isinstance(active_rows, list)
        or len(active_rows) != 1
        or not isinstance(active_rows[0], dict)
        or active_pokemon is None
        or not callable(updater)
    ):
        return
    updater(active_rows[0], active_pokemon)


def session_key(namespace: str, tag: str) -> str:
    """Namespace the live session without changing the dumped replay tag."""
    return f"{namespace}\0{tag}" if namespace else tag


def recover_empty_legality_mask(illegal, fallback_actions=()):
    """Recover only request-plausible actions from an empty strict mask."""
    if illegal.all():
        recovered = {
            int(getattr(action, "action_idx", action)) for action in fallback_actions
        }
        recovered = {index for index in recovered if 0 <= index < len(illegal)}
        if not recovered:
            raise RuntimeError("no definitely or potentially valid actions")
        for index in recovered:
            illegal[index] = False
        return True, "no definitely valid actions"
    return False, None


def correlated_request_rqid(
    pending_request: bool, last_request: object, expected_rqid: int | None
) -> int:
    """Validate that one prior response belongs to the latest actionable request."""
    if not pending_request:
        raise RuntimeError("no unconsumed battle request")
    if not isinstance(last_request, dict):
        raise RuntimeError("battle request metadata is unavailable")
    rqid = last_request.get("rqid")
    if isinstance(rqid, bool) or not isinstance(rqid, int) or rqid < 0:
        raise RuntimeError("battle request has an invalid rqid")
    if expected_rqid is not None and rqid != expected_rqid:
        raise RuntimeError(
            f"battle request rqid mismatch: expected {expected_rqid}, found {rqid}"
        )
    return rqid


def opponent_action_catalog_complete(battle) -> bool:
    """Return whether all public opponent moves and roster members are known."""
    try:
        active = battle.opponent_active_pokemon
        team = list(battle.opponent_team.values())
        moves = list(active.moves.values())
    except (AttributeError, TypeError):
        return False
    if active is None or len(moves) < 4 or len(team) < 6:
        return False
    active_members = [pokemon for pokemon in team if pokemon.active]
    if len(active_members) != 1:
        return False
    member = active_members[0]
    if member is active:
        return True
    for attribute in ("unique_id", "identifier", "name"):
        member_value = getattr(member, attribute, None)
        active_value = getattr(active, attribute, None)
        if member_value is not None and member_value == active_value:
            return True
    return False


opponent_action_support_complete = opponent_action_catalog_complete


def verify_local_checkpoint(
    local_run_dir: str, local_run_name: str, checkpoint: int, expected_sha256: str
) -> tuple[Path, str]:
    path = (
        Path(local_run_dir)
        / local_run_name
        / "ckpts"
        / "policy_weights"
        / f"policy_epoch_{checkpoint}.pt"
    ).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"policy checkpoint not found: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not hmac.compare_digest(actual, expected_sha256):
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return path, actual


def prior_health_payload(
    instance_nonce: str | None,
    checkpoint_sha256: str | None,
    sessions: int,
    pid: int | None = None,
    runtime_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    identity = {
        "schema": 1,
        "nonce": instance_nonce,
        "pid": os.getpid() if pid is None else pid,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if runtime_identity:
        identity.update(runtime_identity)
    return {
        "ok": True,
        "sessions": sessions,
        "identity": identity,
    }


def request_cache_status(rqid: int, cached_rqid: int | None) -> str:
    if cached_rqid is None or rqid > cached_rqid:
        return "new"
    if rqid == cached_rqid:
        return "cached"
    raise ValueError(f"stale rqid {rqid} follows {cached_rqid}")


class BattleSession:
    """Tracks one battle: metamon backend battle + obs/action/reward history."""

    def __init__(
        self,
        session_id: str,
        tag: str,
        namespace: str,
        username: str,
        server,
    ):
        from metamon.env.metamon_battle import MetamonBackendBattle

        self.tag = tag
        self.session_id = session_id
        self.namespace = namespace
        self.username = username
        self.server = server
        self.obs_space = fresh_observation_space(server.obs_space)
        self.opponent_obs_space = fresh_observation_space(server.obs_space)
        logger = logging.getLogger(f"prior.{tag}")
        logger.setLevel(logging.ERROR)
        self.battle = MetamonBackendBattle(
            tag, username, logger, save_replays=False, gen=9
        )
        # attrs only initialized by parse_request; guard the pre-request window
        if not hasattr(self.battle, "_reviving"):
            self.battle._reviving = False
        self.obs_hist: list[dict] = []      # tokenized obs per decision point
        self.action_hist: list[int] = []    # action idx actually taken (len = len(obs)-1)
        self.reward_hist: list[float] = []
        self.action_receipts: list[dict[str, object]] = []
        self.trajectory_time_offset = 0
        # Schema-v3: monotone per-battle decision counter. Incremented once per
        # successful /priors response; echoed to FP and written to the dump so
        # MCTS visit targets can be joined to the exact observation served,
        # with no replay parsing.
        self.decision_idx = 0
        self.last_state = None
        self.last_name_table: dict[str, int] = {}  # engine_move_str -> action idx
        self.trajectory_reset_reason: str | None = None
        self.private_identity_corrections = 0
        self.pending_request = False
        self.last_request_json: dict | None = None
        self.last_request_sha256: str | None = None
        self.last_request_legality: dict[str, object] | None = None
        self.cached_rqid: int | None = None
        self.cached_request_sha256: str | None = None
        self.cached_response: dict | None = None
        # /lines and /priors arrive on different HTTP threads. Keep each battle
        # state consistent while allowing unrelated battles to proceed.
        self.lock = threading.RLock()

    def feed_line(self, line: str) -> None:
        if not line.startswith("|"):
            return
        parts = line.split("|")
        # parts[0] == "" for battle lines
        if len(parts) < 2 or parts[0] != "":
            return
        msg_type = parts[1]
        if msg_type in ("win", "tie"):
            # battle over: free the session (200-game gate runs would otherwise
            # accumulate obs history forever)
            with self.server.lock:
                self.server.sessions.pop(self.session_id, None)
            return
        if msg_type == "request":
            payload = "|".join(parts[2:]).strip()
            if payload:
                try:
                    request = json.loads(payload)
                    legality = request_action_support(request)
                    request_sha256 = canonical_request_sha256(request)
                    self.battle.parse_request(request)
                    if reconcile_private_active_pokemon(self.battle, request):
                        self.private_identity_corrections += 1
                    refresh_private_active_moves(self.battle, request)
                    self.last_request_json = copy.deepcopy(request)
                    self.last_request_sha256 = request_sha256
                    self.last_request_legality = legality
                    self.pending_request = True
                except Exception as e:  # noqa: BLE001
                    print(f"WARN request parse {self.tag}: {e!r}", flush=True)
            return
        try:
            self.battle.parse_message(parts)
        except Exception as e:  # noqa: BLE001
            # SimProtocol can raise on exotic messages; never kill the stream
            print(f"WARN msg parse {self.tag} {msg_type}: {e!r}", flush=True)

    def _reset_policy_trajectory(self, reason: str) -> None:
        self.obs_hist.clear()
        self.action_hist.clear()
        self.reward_hist.clear()
        self.action_receipts.clear()
        self.trajectory_time_offset = 0
        self.last_state = None
        self.trajectory_reset_reason = reason

    def acknowledge_action(
        self,
        action: str,
        expected_rqid: int,
        expected_request_sha256: str,
        expected_decision_idx: int,
    ) -> dict[str, object]:
        """Record the exact action selected for a served request, once."""
        if (
            isinstance(expected_rqid, bool)
            or not isinstance(expected_rqid, int)
            or expected_rqid < 0
            or isinstance(expected_decision_idx, bool)
            or not isinstance(expected_decision_idx, int)
            or expected_decision_idx < 0
            or not isinstance(action, str)
            or not action
            or not re.fullmatch(r"[0-9a-f]{64}", expected_request_sha256)
        ):
            raise ValueError("invalid action acknowledgement")
        if (
            self.cached_rqid != expected_rqid
            or self.cached_request_sha256 != expected_request_sha256
            or self.cached_response is None
            or self.cached_response.get("decision_idx") != expected_decision_idx
        ):
            raise RuntimeError("action acknowledgement does not match served priors")
        served_actions = self.cached_response.get("priors")
        if not isinstance(served_actions, dict) or action not in served_actions:
            raise RuntimeError("chosen action was absent from served policy support")

        forced_action = forced_showdown_action(list(served_actions))
        if forced_action is not None:
            if action != forced_action:
                raise RuntimeError("automatic request acknowledged a different action")
            return {"ok": True, "automatic": True, "idempotent": True}

        if self.action_receipts and self.action_receipts[-1]["rqid"] == expected_rqid:
            receipt = self.action_receipts[-1]
            if (
                receipt["request_sha256"] != expected_request_sha256
                or receipt["decision_idx"] != expected_decision_idx
                or receipt["action"] != action
            ):
                raise RuntimeError("conflicting action acknowledgement")
            return {"ok": True, "automatic": False, "idempotent": True}
        if len(self.action_hist) != len(self.obs_hist) - 1:
            raise RuntimeError("policy trajectory already has an unresolved action boundary")
        action_idx = self.last_name_table.get(action)
        if action == "struggle":
            action_idx = 0
        if action_idx is None:
            raise RuntimeError("chosen action has no RL2 action index")
        self.action_hist.append(action_idx)
        self.action_receipts.append(
            {
                "rqid": expected_rqid,
                "request_sha256": expected_request_sha256,
                "decision_idx": expected_decision_idx,
                "action": action,
                "action_idx": action_idx,
                "source": "selected_action_ack",
            }
        )
        return {"ok": True, "automatic": False, "idempotent": False}

    def compute_priors(
        self,
        requester_username: str | None = None,
        expected_rqid: int | None = None,
        expected_request_sha256: str | None = None,
    ) -> dict:
        rqid = correlated_request_rqid(
            self.pending_request, self.last_request_json, expected_rqid
        )
        if (
            not isinstance(expected_request_sha256, str)
            or len(expected_request_sha256) != 64
            or self.last_request_sha256 != expected_request_sha256
        ):
            raise RuntimeError("battle request SHA-256 mismatch")
        if self.last_request_legality is None:
            raise RuntimeError("battle request legality is unavailable")
        cache_status = request_cache_status(rqid, self.cached_rqid)
        if cache_status == "cached":
            if (
                self.cached_response is None
                or self.cached_request_sha256 != expected_request_sha256
            ):
                raise RuntimeError("cached rqid has no prior response")
            self.pending_request = False
            return dict(self.cached_response)

        # Recharge and Struggle requests have exactly one legal command and do
        # not correspond to any of Metamon's learned 13 actions.  Showdown still
        # asks the client to submit that command, so return an authoritative
        # point mass without running or mutating the policy trajectory.  The
        # next discretionary request will close the preceding learned-action
        # transition across this automatic boundary, matching training data
        # where forced rows are excluded.
        forced_action = forced_showdown_action(
            self.last_request_legality.get("actions")
        )
        if forced_action is not None:
            decision_idx = self.decision_idx
            response = {
                "priors": {forced_action: 1.0},
                "opp_priors": {},
                "probs": [0.0] * 13,
                "turn": len(self.obs_hist),
                "decision_idx": decision_idx,
                "battle_turn": getattr(self.battle, "turn", None),
                "rqid": rqid,
                "request_sha256": expected_request_sha256,
                "own_legality": self.last_request_legality,
                "opponent_support": {
                    "status": "ineligible",
                    "reason": "forced_player_action",
                },
                "trajectory": {
                    "observations": len(self.obs_hist),
                    "transitions": len(self.action_hist),
                    "inference_length": 0,
                    "reset_reason": None,
                    "automatic_action": forced_action,
                },
            }
            self.decision_idx += 1
            self.pending_request = False
            self.cached_rqid = rqid
            self.cached_request_sha256 = expected_request_sha256
            self.cached_response = response
            return dict(response)
        import numpy as np
        import torch

        from metamon.interface import UniversalState, UniversalAction

        state = UniversalState.from_Battle(self.battle)
        # Complete the previous request transition before appending this observation.
        if self.last_state is not None:
            if len(self.action_hist) != len(self.obs_hist):
                raise RuntimeError("missing selected-action acknowledgement")
            else:
                try:
                    reward = self.server.reward_fn(self.last_state, state)
                except Exception as exc:
                    raise RuntimeError("reward boundary failed") from exc
                else:
                    reward = float(reward)
                    if math.isfinite(reward):
                        self.reward_hist.append(reward)
                    else:
                        raise RuntimeError("non-finite reward boundary")
        if len(self.reward_hist) != len(self.action_hist):
            raise RuntimeError("misaligned reward boundary")
        if len(self.action_receipts) != len(self.action_hist):
            raise RuntimeError("action receipt boundary is misaligned")
        self.last_state = state

        obs = self.obs_space.state_to_obs(state)
        request_actions = set(self.last_request_legality["actions"])
        # The private Showdown request is authoritative for the current move
        # set, including after Illusion and forced drag. Its learned move slots
        # use the same normalized alphabetical ordering as Metamon.
        name_table = private_request_move_name_table(request_actions)
        name_table.update(private_request_switch_name_table(request_actions))

        mapped_actions: dict[str, int] = {}
        for action in request_actions:
            if action == "struggle":
                mapped_actions[action] = 0
                continue
            if action in UNLEARNED_REQUEST_ACTIONS:
                continue
            if action in name_table:
                mapped_actions[action] = name_table[action]
                continue
            if action.startswith("switch "):
                target = norm(action.removeprefix("switch "))
                candidates = [
                    (name, index)
                    for name, index in name_table.items()
                    if name.startswith("switch ")
                    and (
                        target.startswith(norm(name.removeprefix("switch ")))
                        or norm(name.removeprefix("switch ")).startswith(target)
                    )
                ]
                if len(candidates) == 1:
                    mapped_actions[candidates[0][0]] = candidates[0][1]
                    continue
            raise RuntimeError(f"request action is absent from policy table: {action}")
        illegal = np.ones(13, dtype=bool)
        for action in mapped_actions:
            for index in request_action_policy_indices(action, mapped_actions):
                illegal[index] = False
        mask_fallback = False
        mask_fallback_error = None
        obs = dict(obs)
        obs["illegal_actions"] = illegal
        current_obs = obs
        self.obs_hist.append(current_obs)
        max_seq_len = int(getattr(self.server.agent, "max_seq_len", 128))
        if len(self.obs_hist) > max_seq_len:
            overflow = len(self.obs_hist) - max_seq_len
            self.obs_hist[:] = self.obs_hist[-max_seq_len:]
            self.action_hist[:] = self.action_hist[-(max_seq_len - 1):]
            self.reward_hist[:] = self.reward_hist[-(max_seq_len - 1):]
            self.action_receipts[:] = self.action_receipts[-(max_seq_len - 1):]
            self.trajectory_time_offset += overflow
        if self.server.trajectory_mode == "legacy-stateless":
            trajectory_obs, trajectory_rl2, trajectory_time = (
                legacy_stateless_trajectory_arrays(current_obs)
            )
        else:
            trajectory_obs, trajectory_rl2, trajectory_time = aligned_trajectory_arrays(
                self.obs_hist,
                self.action_hist,
                self.reward_hist,
                max_seq_len,
                self.trajectory_time_offset,
            )
        T = len(trajectory_time)
        trajectory_reset_reason = self.trajectory_reset_reason
        self.trajectory_reset_reason = None
        device = self.server.device
        text = torch.tensor(
            trajectory_obs["text_tokens"],
            dtype=torch.int32,
            device=device,
        ).unsqueeze(0)
        numbers = torch.tensor(
            trajectory_obs["numbers"],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        numbers = torch.nan_to_num(numbers)
        ill = torch.tensor(
            trajectory_obs["illegal_actions"], device=device
        ).unsqueeze(0)
        rl2s = torch.tensor(trajectory_rl2, dtype=torch.float32, device=device).unsqueeze(0)
        time_idxs = torch.tensor(trajectory_time, device=device).long().unsqueeze(0)
        obs_batch = {"text_tokens": text, "numbers": numbers, "illegal_actions": ill}

        agent = self.server.agent
        with torch.no_grad():
            emb, _ = agent.get_state_embedding(
                obs=obs_batch, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None
            )
            dists = agent.actor(
                emb,
                straight_from_obs={
                    # Match actor side-channel tensors (especially numbers)
                    # to the state-embedding sequence exactly.
                    k: obs_batch[k][:, : emb.shape[1]]
                    for k in agent.pass_obs_keys_to_actor
                },
            )
            probs = dists.probs[0, -1, -1, :].cpu().numpy()  # last step, inference gamma

        if not np.isfinite(probs).all():
            raise RuntimeError("non-finite root policy probabilities")

        probs = probs * (~illegal)
        if probs.sum() <= 0:
            probs = (~illegal).astype(float)
        probs = probs / probs.sum()

        self.last_name_table = mapped_actions

        priors = {}
        for name, idx in mapped_actions.items():
            indices = request_action_policy_indices(name, mapped_actions)
            priors[name] = float(sum(probs[index] for index in indices))
        priors = add_unlearned_action_priors(priors, request_actions)

        decision_idx = self.decision_idx
        battle_turn = getattr(self.battle, "turn", None)
        opponent_priors = self.compute_opponent_priors()
        if self.server.dump_path:
            # Fail closed: if the dump write fails, the whole /priors request
            # fails (500), and METAGROSS_REQUIRE_PRIORS=1 discards the game.
            # A decision must never be played whose observation was not
            # durably recorded.
            dump_row = {
                "schema": 5,
                "tag": self.tag,
                "namespace": self.namespace,
                "decision_idx": decision_idx,
                "battle_turn": battle_turn,
                "rqid": rqid,
                "request_sha256": expected_request_sha256,
                "own_legality": self.last_request_legality,
                # The requester's actual PS username (per-game generated by
                # eval.run) is the join key against decision logs; the launch
                # --username is only a fallback label.
                "username": requester_username or self.username,
                # exactly what the policy consumed (numbers nan_to_num'ed to
                # match the inference path above)
                "text_tokens": current_obs["text_tokens"].tolist(),
                "numbers": np.nan_to_num(
                    np.asarray(current_obs["numbers"], dtype=np.float32)
                ).tolist(),
                "illegal_actions": [bool(x) for x in illegal],
                "mask_fallback": mask_fallback,
                "mask_fallback_error": mask_fallback_error,
                "name_table": mapped_actions,
                "probs": [float(p) for p in probs],
                "opponent_prior": self.last_opponent_prior_evidence,
                "trajectory": {
                    "mode": self.server.trajectory_mode,
                    "observations": len(self.obs_hist),
                    "transitions": len(self.action_hist),
                    "inference_length": T,
                    "reset_reason": trajectory_reset_reason,
                    "action_receipts": list(self.action_receipts),
                    "private_identity_corrections": self.private_identity_corrections,
                    "rl2": trajectory_rl2.tolist(),
                    "time_indices": trajectory_time[:, 0].tolist(),
                },
            }
            line = json.dumps(dump_row, separators=(",", ":")) + "\n"
            with self.server.dump_lock:
                with open(self.server.dump_path, "a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
        self.decision_idx += 1
        self.pending_request = False

        response = {
            "priors": priors,
            "opp_priors": opponent_priors,
            "probs": [float(p) for p in probs],
            "turn": T,
            "decision_idx": decision_idx,
            "battle_turn": battle_turn,
            "rqid": rqid,
            "request_sha256": expected_request_sha256,
            "own_legality": self.last_request_legality,
            "opponent_support": self.last_opponent_prior_evidence,
            "trajectory": {
                "mode": self.server.trajectory_mode,
                "observations": len(self.obs_hist),
                "transitions": len(self.action_hist),
                "inference_length": T,
                "reset_reason": trajectory_reset_reason,
                "action_receipts": list(self.action_receipts),
                "private_identity_corrections": self.private_identity_corrections,
            },
        }
        self.cached_rqid = rqid
        self.cached_request_sha256 = expected_request_sha256
        self.cached_response = response
        return dict(response)

    def compute_opponent_priors(self) -> dict:
        """Compute priors for the OPPONENT's moves from the opponent's POV.

        FP's modeled opponent currently sees our full team (mirror assumption).
        We bias the opponent's action distribution toward what a human would do,
        using the same 142M policy evaluated from the opponent's perspective.

        The opponent's POV = same game with sides swapped. We build a flipped
        UniversalState and run the policy on it. The opponent's legal moves
        are their active mon's moves + switches; we map those to engine move
        strings in the opponent's option order.
        """
        import numpy as np
        import torch
        from metamon.interface import UniversalState, UniversalAction, consistent_move_order, consistent_pokemon_order

        self.last_opponent_prior_evidence = {
            "support_complete": False,
            "catalog_complete": False,
            "authority": "public_inference",
            "request_observed": False,
            "support_kind": "publicly_possible_superset",
            "status": "ineligible",
            "reason": "incomplete_public_action_support",
            "name_table": {},
            "illegal_actions": [],
            "raw_priors": {},
            "mass_sum": 0.0,
        }
        try:
            catalog_complete = opponent_action_catalog_complete(self.battle)
            self.last_opponent_prior_evidence["catalog_complete"] = catalog_complete
            if not catalog_complete:
                return {}
            opp_battle = self._make_opp_battle()
            if opp_battle is None:
                self.last_opponent_prior_evidence.update(
                    status="error", reason="opponent_battle_construction_failed"
                )
                return {}
            # Build a real opponent battle view instead of mutating obsolete
            # UniversalState field names. This gives state_to_obs the expected
            # player_active_pokemon / available_switches layout.
            flipped = UniversalState.from_Battle(opp_battle)
            obs = self.opponent_obs_space.state_to_obs(flipped)
            # opponent's legal actions from the flipped battle
            illegal = np.ones(13, dtype=bool)
            try:
                for a in UniversalAction.definitely_valid_actions(flipped, opp_battle):
                    illegal[a.action_idx] = False
            except Exception as exc:
                self.last_opponent_prior_evidence.update(
                    status="error",
                    reason=f"definite_action_support_failed:{type(exc).__name__}",
                )
                return {}
            recover_empty_legality_mask(
                illegal, UniversalAction.maybe_valid_actions(flipped)
            )
            obs = dict(obs)
            obs["illegal_actions"] = illegal

            # single-step inference (no history for opponent — we don't track
            # their action/reward sequence; this is a stateless prior).
            # state_to_obs returns 1D arrays for a single state.
            # The transformer requires T>=2, so pad with a blank first step.
            tt = obs["text_tokens"]  # (L,)
            nn = obs["numbers"]      # (N,)
            tt = np.stack([np.zeros_like(tt), tt])  # (2, L)
            nn = np.stack([np.zeros_like(nn), nn])  # (2, N)
            T = 2
            device = self.server.device
            ill_opp = np.ones((T, 13), dtype=bool)
            ill_opp[-1] = illegal  # only the real step has the mask
            text = torch.tensor(tt, dtype=torch.int32, device=device).unsqueeze(0)  # [1, 2, L]
            numbers = torch.tensor(nn, dtype=torch.float32, device=device).unsqueeze(0)
            numbers = torch.nan_to_num(numbers)
            ill_t = torch.tensor(ill_opp, device=device).unsqueeze(0)  # [1, 2, A]
            rl2s = torch.zeros((1, T, 14), device=device)
            time_idxs = torch.arange(T, device=device).long().unsqueeze(0).unsqueeze(-1)
            obs_batch = {"text_tokens": text, "numbers": numbers, "illegal_actions": ill_t}

            agent = self.server.agent
            with torch.no_grad():
                try:
                    emb, _ = agent.get_state_embedding(
                        obs=obs_batch, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None
                    )
                    dists = agent.actor(
                        emb,
                        straight_from_obs={
                            k: obs_batch[k][:, : emb.shape[1]]
                            for k in agent.pass_obs_keys_to_actor
                        },
                    )
                    probs = dists.probs[0, -1, -1, :].cpu().numpy()
                    if not np.isfinite(probs).all():
                        raise RuntimeError("non-finite opponent policy probabilities")
                except (ValueError, RuntimeError) as e:
                    if "not enough values" in str(e) or "shape" in str(e).lower():
                        self.last_opponent_prior_evidence.update(
                            status="error",
                            reason=f"opponent_policy_shape_failed:{type(e).__name__}",
                        )
                        return {}
                    raise

            probs = probs * (~illegal)
            if probs.sum() <= 0:
                probs = (~illegal).astype(float)
            probs = probs / probs.sum()

            # map to opponent's engine move strings
            opp_name_table: dict[str, int] = {}
            try:
                opp_active = opp_battle.active_pokemon
                opp_moves = consistent_move_order(
                    list(opp_active.moves.values())
                ) if opp_active else []
            except Exception:
                opp_moves = []
            try:
                opp_bench = consistent_pokemon_order(
                    [p for p in opp_battle.team.values() if not p.fainted and not p.active]
                )
            except Exception:
                opp_bench = []
            for i, mv in enumerate(opp_moves[:4]):
                opp_name_table[mv.id] = i
                opp_name_table[f"{mv.id}-tera"] = i + 9
            for i, p in enumerate(opp_bench[:5]):
                opp_name_table[f"switch {norm(p.name)}"] = i + 4

            opp_priors = {}
            for name, idx in opp_name_table.items():
                if not illegal[idx]:
                    opp_priors[name] = float(probs[idx])
            self.last_opponent_prior_evidence = {
                "support_complete": False,
                "catalog_complete": True,
                "authority": "public_inference",
                "request_observed": False,
                "support_kind": "publicly_possible_superset",
                "status": "used",
                "reason": None,
                "name_table": opp_name_table,
                "illegal_actions": [bool(value) for value in illegal],
                "raw_priors": opp_priors,
                "mass_sum": math.fsum(opp_priors.values()),
            }
            return opp_priors
        except Exception as e:
            self.last_opponent_prior_evidence.update(
                status="error", reason=f"{type(e).__name__}: {e}"
            )
            import traceback
            traceback.print_exc()
            print(f"WARN opponent priors failed: {e!r}", flush=True)
            return {}

    def _flip_state(self, state):
        """Swap player/opponent in a UniversalState for opponent-POV inference."""
        from metamon.interface import UniversalState
        # UniversalState fields: player_team, opponent_team, active_pokemon,
        # opponent_active_pokemon, etc. — swap them
        flipped = UniversalState.__new__(UniversalState)
        for attr in dir(state):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(state, attr)
                setattr(flipped, attr, val)
            except Exception:
                pass
        # swap player/opponent fields
        if hasattr(state, "player_team"):
            flipped.player_team = state.opponent_team
            flipped.opponent_team = state.player_team
        if hasattr(state, "active_pokemon"):
            flipped.active_pokemon = state.opponent_active_pokemon
            flipped.opponent_active_pokemon = state.active_pokemon
        if hasattr(state, "player_side_conditions"):
            flipped.player_side_conditions = state.opponent_side_conditions
            flipped.opponent_side_conditions = state.player_side_conditions
        return flipped

    def _make_opp_battle(self):
        """Create a minimal battle-like object for opponent legal-action check."""
        # The opponent's legal actions = their active mon's moves + switches
        # We can use the original battle but swap team/opponent_team refs
        class OppBattleView:
            pass
        view = OppBattleView()
        try:
            view.active_pokemon = self.battle.opponent_active_pokemon
            view.team = self.battle.opponent_team
            view.opponent_active_pokemon = self.battle.active_pokemon
            view.opponent_team = self.battle.team
            view.weather = self.battle.weather
            view.fields = self.battle.fields
            view.side_conditions = self.battle.opponent_side_conditions
            view.opponent_side_conditions = self.battle.side_conditions
            view.force_switch = False
            view.reviving = False
            view.won = False
            view.lost = False
            view.can_tera = not any(
                bool(getattr(pokemon, "terastallized", False))
                for pokemon in view.team.values()
            )
            view.available_moves = list(view.active_pokemon.moves.values())
            view.available_switches = [
                pokemon
                for pokemon in view.team.values()
                if not pokemon.fainted and not pokemon.active
            ]
            view.battle_tag = self.battle._battle_tag
            # Metamon only needs a species list for this field. The opponent's
            # original preview is not available from the flipped view, so use
            # the revealed team as a conservative proxy.
            view.teampreview_opponent_team = list(self.battle.team.values())
        except Exception:
            return None
        return view


class PriorServer:
    def __init__(self, args):
        os.environ.setdefault(
            "METAMON_CACHE_DIR", str(ROOT / "srcs" / "runtime" / "metamon-cache")
        )
        os.environ.setdefault("WANDB_MODE", "disabled")
        import metamon.rl.pretrained as _pt

        if args.local_run_dir:
            model = _pt.LocalFinetunedModel(
                base_model=getattr(_pt, args.local_base_model),
                amago_ckpt_dir=args.local_run_dir,
                model_name=args.local_run_name,
                default_checkpoint=args.checkpoint,
            )
            label = f"local:{args.local_run_name}@ckpt{args.checkpoint}"
        else:
            model = _pt.get_pretrained_model(args.agent)
            label = args.agent
        print(f"PRIOR_SERVER loading {label}", flush=True)
        experiment = model.initialize_agent(checkpoint=args.checkpoint, log=False)
        self.agent = experiment.policy
        self.agent.eval()
        self.device = next(self.agent.parameters()).device
        self.obs_space = model.observation_space
        self.reward_fn = model.reward_function
        self.username = args.username
        self.trajectory_mode = args.trajectory_mode
        self.sessions: dict[str, BattleSession] = {}
        self.lock = threading.Lock()
        # Schema-v3 observation dump (one JSONL row per /priors decision)
        dump = getattr(args, "decision_dump", None) or os.environ.get(
            "METAGROSS_PRIOR_DUMP"
        )
        self.dump_path = str(dump) if dump else None
        self.dump_lock = threading.Lock()
        if self.dump_path:
            Path(self.dump_path).parent.mkdir(parents=True, exist_ok=True)
            print(f"PRIOR_SERVER decision dump -> {self.dump_path}", flush=True)
        print("PRIOR_SERVER ready", flush=True)

    def session(self, session_id: str, tag: str, namespace: str) -> BattleSession:
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = BattleSession(
                    session_id,
                    tag,
                    namespace,
                    self.username,
                    self,
                )
            return self.sessions[session_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="Kakuna")
    parser.add_argument("--local-run-dir", default=None)
    parser.add_argument("--local-run-name", default=None)
    parser.add_argument("--local-base-model", default="Kakuna")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--checkpoint-sha256", default=None)
    parser.add_argument("--username", required=True,
                        help="FP's showdown username (to identify our side)")
    parser.add_argument(
        "--trajectory-mode",
        choices=("causal-history", "legacy-stateless"),
        default="causal-history",
        help="Frozen player-policy inference contract; default is repaired causal history.",
    )
    parser.add_argument("--port", type=int, default=8977)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--instance-nonce",
        default=os.environ.get("METAGROSS_PRIOR_INSTANCE_NONCE"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--decision-dump",
        default=None,
        help="JSONL path: append one schema-v3 row (obs, legal mask, name table) "
        "per served /priors decision. Env fallback: METAGROSS_PRIOR_DUMP.",
    )
    args = parser.parse_args()

    if args.instance_nonce is not None and not re.fullmatch(
        r"[0-9a-f]{64}", args.instance_nonce
    ):
        parser.error("prior instance nonce must contain 64 lowercase hexadecimal characters")

    if args.checkpoint_sha256:
        if not args.local_run_dir or not args.local_run_name or args.checkpoint is None:
            parser.error(
                "--checkpoint-sha256 requires --local-run-dir, --local-run-name, and --checkpoint"
            )
        checkpoint_path, checkpoint_sha256 = verify_local_checkpoint(
            args.local_run_dir,
            args.local_run_name,
            args.checkpoint,
            args.checkpoint_sha256,
        )
        print(
            f"PRIOR_CHECKPOINT path={checkpoint_path} sha256={checkpoint_sha256}",
            flush=True,
        )

    server = PriorServer(args)
    python_executable = Path(sys.executable).resolve()
    runtime_identity = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_executable": str(python_executable),
        "python_executable_sha256": hashlib.sha256(python_executable.read_bytes()).hexdigest(),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_base_prefix": str(Path(sys.base_prefix).resolve()),
        "argv": list(sys.argv),
        "host": args.host,
        "port": args.port,
        "decision_dump": str(Path(args.decision_dump).resolve()) if args.decision_dump else None,
        "environment": dict(sorted(os.environ.items())),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            tag = data.get("tag", "")
            namespace = data.get("namespace", "")
            if not isinstance(tag, str) or not isinstance(namespace, str):
                self._json(400, {"error": "tag and namespace must be strings"})
                return
            live_session = session_key(namespace, tag)
            if path == "/lines":
                lines = data.get("lines")
                if not isinstance(lines, list) or any(
                    not isinstance(line, str) for line in lines
                ):
                    self._json(400, {"error": "lines must be a list of strings"})
                    return
                sess = server.session(live_session, tag, namespace)
                with sess.lock:
                    for line in lines:
                        sess.feed_line(line)
                self._json(200, {"ok": True})
            elif path == "/action":
                try:
                    sess = server.session(live_session, tag, namespace)
                    with sess.lock:
                        result = sess.acknowledge_action(
                            data.get("action"),
                            data.get("rqid"),
                            data.get("request_sha256"),
                            data.get("decision_idx"),
                        )
                    self._json(200, result)
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001
                    self._json(409, {"error": f"{type(exc).__name__}: {exc}"})
            elif path == "/end":
                with server.lock:
                    server.sessions.pop(live_session, None)
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "unknown"})

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/priors":
                query = parse_qs(parsed.query)
                tag = query.get("tag", [""])[0]
                namespace = query.get("namespace", [""])[0]
                requester_username = query.get("username", [""])[0] or None
                rqid_text = query.get("rqid", [""])[0]
                request_sha256 = query.get("request_sha256", [""])[0]
                if not isinstance(tag, str) or not isinstance(namespace, str):
                    self._json(400, {"error": "tag and namespace must be strings"})
                    return
                try:
                    expected_rqid = int(rqid_text)
                    if expected_rqid < 0 or str(expected_rqid) != rqid_text:
                        raise ValueError
                except ValueError:
                    self._json(400, {"error": "rqid must be a non-negative integer"})
                    return
                if not re.fullmatch(r"[0-9a-f]{64}", request_sha256):
                    self._json(400, {"error": "request_sha256 must be lowercase SHA-256"})
                    return
                try:
                    sess = server.session(session_key(namespace, tag), tag, namespace)
                    with sess.lock:
                        result = sess.compute_priors(
                            requester_username, expected_rqid, request_sha256
                        )
                    self._json(200, result)
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    self._json(500, {"error": f"{type(e).__name__}: {e}"})
            elif parsed.path == "/health":
                self._json(
                    200,
                    prior_health_payload(
                        args.instance_nonce,
                        args.checkpoint_sha256,
                        len(server.sessions),
                        runtime_identity=runtime_identity,
                    ),
                )
            else:
                self._json(404, {"error": "unknown"})

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"PRIOR_SERVER listening on {args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
