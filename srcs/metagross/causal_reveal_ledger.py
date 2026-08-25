"""Production causal public-reveal capture and engine-mask propagation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
FORM_HELPER = ROOT / "srcs/metagross/export_showdown_public_form_contract.cjs"
FORM_ABILITY_HELPER = ROOT / "srcs/metagross/export_showdown_form_ability_contract.cjs"
LEDGER_ATTRIBUTE = "metagross_causal_reveal_ledger_v4"
LEDGER_SCHEMA = "metagross-causal-public-reveal-ledger/v4"
MOVE_RECEIPT_ATTRIBUTE = "metagross_causal_move_receipts_v1"
ENGINE_PAYLOAD_SCHEMA = "metagross-causal-engine-state/v1"
VALID_MASK = (1 << 42) - 1
MAX_TRACKED_BATTLES = 128
STATE_SERIALIZATION_FIELDS = (
    "side_one",
    "side_two",
    "weather",
    "terrain",
    "trick_room",
    "team_preview",
    "s1_threat",
    "s2_threat",
    "scout_value",
    "threat_matrix",
    "wincon_matrix",
    "s1_public_reveals",
    "s2_public_reveals",
)


class CausalRevealLedgerError(RuntimeError):
    pass


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


_FORM_CONTRACT: dict[str, str] | None = None
_FORM_ABILITY_CONTRACT: dict[str, str] | None = None
_FORM_LOCK = threading.Lock()
_PROTOCOL_LINES: "OrderedDict[str, list[str]]" = OrderedDict()
_PROTOCOL_LOCK = threading.Lock()
_ABILITY_RECEIPT_LOCK = threading.Lock()
_MOVE_RECEIPT_LOCK = threading.Lock()


def public_form_contract() -> Mapping[str, str]:
    global _FORM_CONTRACT
    with _FORM_LOCK:
        if _FORM_CONTRACT is None:
            completed = subprocess.run(
                ["node", str(FORM_HELPER)], cwd=ROOT, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            payload = json.loads(completed.stdout)
            rows = payload.get("rows")
            if (
                payload.get("schema") != "metagross-showdown-public-form-contract/v1"
                or not isinstance(rows, list)
                or payload.get("mapping_count") != len(rows)
                or len(rows) < 100
            ):
                raise CausalRevealLedgerError("invalid Showdown public-form contract")
            mapping: dict[str, str] = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise CausalRevealLedgerError("invalid Showdown public-form row")
                source, target = row.get("source"), row.get("target")
                if (
                    not isinstance(source, str) or norm(source) != source or not source
                    or not isinstance(target, str) or norm(target) != target or not target
                    or source in mapping
                ):
                    raise CausalRevealLedgerError("invalid Showdown public-form mapping")
                mapping[source] = target
            _FORM_CONTRACT = mapping
        return _FORM_CONTRACT


def canonical_species(value: Any) -> str:
    species = norm(value)
    return public_form_contract().get(species, species)


def form_ability_contract() -> Mapping[str, str]:
    """Pinned exact-form abilities only where Showdown has one possible value."""
    global _FORM_ABILITY_CONTRACT
    with _FORM_LOCK:
        if _FORM_ABILITY_CONTRACT is None:
            completed = subprocess.run(
                ["node", str(FORM_ABILITY_HELPER)], cwd=ROOT, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            payload = json.loads(completed.stdout)
            rows = payload.get("rows")
            if (
                payload.get("schema")
                != "metagross-showdown-form-ability-contract/v1"
                or not isinstance(rows, list)
                or payload.get("row_count") != len(rows)
                or len(rows) < 100
            ):
                raise CausalRevealLedgerError("invalid Showdown form-ability contract")
            mapping: dict[str, str] = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise CausalRevealLedgerError("invalid Showdown form-ability row")
                exact = row.get("exact_species")
                ability = row.get("current_ability")
                if (
                    not isinstance(exact, str) or norm(exact) != exact or not exact
                    or not isinstance(ability, str) or norm(ability) != ability or not ability
                    or row.get("authority")
                    != "showdown-exact-species-unique-ability"
                    or exact in mapping
                ):
                    raise CausalRevealLedgerError("invalid Showdown form-ability fields")
                mapping[exact] = ability
            _FORM_ABILITY_CONTRACT = mapping
        return _FORM_ABILITY_CONTRACT


def record_public_protocol_lines(tag: str, lines: Sequence[str]) -> None:
    """Record public lines only; private requests can never enter the ledger."""
    if not isinstance(tag, str) or not tag:
        raise CausalRevealLedgerError("battle tag is required")
    public = [
        line for line in lines
        if isinstance(line, str) and line.startswith("|") and not line.startswith("|request|")
    ]
    with _PROTOCOL_LOCK:
        stored = _PROTOCOL_LINES.setdefault(tag, [])
        if len(public) >= len(stored) and public[: len(stored)] == stored:
            # Reconnect can replay the complete public prefix.
            stored[:] = public
        elif len(stored) >= len(public) and stored[: len(public)] == public:
            # Idempotent replay of an already-recorded leading chunk.
            pass
        else:
            stored.extend(public)
        _PROTOCOL_LINES.move_to_end(tag)
        while len(_PROTOCOL_LINES) > MAX_TRACKED_BATTLES:
            _PROTOCOL_LINES.popitem(last=False)


def clear_public_protocol_lines(tag: str) -> None:
    with _PROTOCOL_LOCK:
        _PROTOCOL_LINES.pop(tag, None)


def protocol_lines_for_battle(tag: str) -> tuple[str, ...]:
    with _PROTOCOL_LOCK:
        rows = tuple(_PROTOCOL_LINES.get(tag, ()))
    if not rows:
        raise CausalRevealLedgerError("battle has no captured causal public protocol")
    return rows


@dataclass
class _MutableFact:
    species: str
    exact_public_species: str
    moves: set[str] = field(default_factory=set)
    current_item: str | None = None
    item_status_revealed: bool = False
    consumed_items: set[str] = field(default_factory=set)
    current_ability: str | None = None
    current_ability_authority: str | None = None
    ability_history: list["CausalAbilityEvent"] = field(default_factory=list)
    move_events: list["CausalMoveEvent"] = field(default_factory=list)
    disable_history: list["CausalDisableEvent"] = field(default_factory=list)
    causal_disable_states: dict[str, bool] = field(default_factory=dict)
    form_history: list["CausalFormEvent"] = field(default_factory=list)


@dataclass(frozen=True)
class CausalAbilityEvent:
    event_index: int
    exact_public_species: str
    ability: str
    authority: str
    protocol_tag: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "exact_public_species": self.exact_public_species,
            "ability": self.ability,
            "authority": self.authority,
            "protocol_tag": self.protocol_tag,
        }


@dataclass(frozen=True)
class CausalMoveEvent:
    event_index: int
    exact_public_species: str
    move: str
    authority: str
    derived_cause: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "exact_public_species": self.exact_public_species,
            "move": self.move,
            "authority": self.authority,
            "derived_cause": self.derived_cause,
        }


@dataclass(frozen=True)
class CausalDisableEvent:
    event_index: int
    exact_public_species: str
    move: str
    disabled: bool
    authority: str
    protocol_tag: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "exact_public_species": self.exact_public_species,
            "move": self.move,
            "disabled": self.disabled,
            "authority": self.authority,
            "protocol_tag": self.protocol_tag,
        }


@dataclass(frozen=True)
class CausalFormEvent:
    event_index: int
    exact_public_species: str
    canonical_species: str
    source_ability: str
    protocol_tag: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "exact_public_species": self.exact_public_species,
            "canonical_species": self.canonical_species,
            "source_ability": self.source_ability,
            "protocol_tag": self.protocol_tag,
        }


@dataclass(frozen=True)
class CausalMoveState:
    move: str
    current_pp: int
    max_pp: int
    disabled: bool | None
    disable_authority: str = "causal_disable"
    causal_disable_lifecycle_state: bool | None = None
    authority: str = "causal_live_public_tracker"

    def to_payload(self) -> dict[str, Any]:
        return {
            "move": self.move,
            "current_pp": self.current_pp,
            "max_pp": self.max_pp,
            "disabled": self.disabled,
            "disable_authority": self.disable_authority,
            "causal_disable_lifecycle_state": self.causal_disable_lifecycle_state,
            "authority": self.authority,
        }


@dataclass(frozen=True)
class CausalRevealFact:
    species: str
    exact_public_species: str
    moves: tuple[str, ...]
    current_item: str | None
    item_status_revealed: bool
    consumed_items: tuple[str, ...]
    current_ability: str | None
    current_ability_authority: str | None
    ability_history: tuple[CausalAbilityEvent, ...]
    move_events: tuple[CausalMoveEvent, ...] = ()
    move_states: tuple[CausalMoveState, ...] = ()
    disable_history: tuple[CausalDisableEvent, ...] = ()
    form_history: tuple[CausalFormEvent, ...] = ()

    @property
    def ability(self) -> str | None:
        """Compatibility alias; new consumers should use current_ability."""
        return self.current_ability

    def to_payload(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "exact_public_species": self.exact_public_species,
            "moves": list(self.moves),
            "current_item": self.current_item,
            "item_status_revealed": self.item_status_revealed,
            "consumed_items": list(self.consumed_items),
            "current_ability": self.current_ability,
            "current_ability_authority": self.current_ability_authority,
            "ability_history": [event.to_payload() for event in self.ability_history],
            "move_events": [event.to_payload() for event in self.move_events],
            "move_states": [state.to_payload() for state in self.move_states],
            "disable_history": [event.to_payload() for event in self.disable_history],
            "form_history": [event.to_payload() for event in self.form_history],
        }


@dataclass(frozen=True)
class CausalRevealLedger:
    battle_tag: str
    observer_role: str
    opponent_role: str
    opponent_active_species: str
    facts: tuple[CausalRevealFact, ...]
    protocol_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": LEDGER_SCHEMA,
            "battle_tag": self.battle_tag,
            "observer_role": self.observer_role,
            "opponent_role": self.opponent_role,
            "opponent_active_species": self.opponent_active_species,
            "facts": [fact.to_payload() for fact in self.facts],
            "protocol_sha256": self.protocol_sha256,
            "pp_disable_contract": (
                "exact-causal-pp;typed-causal-vs-world-mechanical-disable;"
                "existing-sampled-move-only;missing-fails-closed"
            ),
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CausalRevealLedger":
        if payload.get("schema") != LEDGER_SCHEMA:
            raise CausalRevealLedgerError("invalid causal ledger schema")
        facts = payload.get("facts")
        if not isinstance(facts, list):
            raise CausalRevealLedgerError("invalid causal ledger facts")
        parsed = []
        for row in facts:
            if not isinstance(row, Mapping):
                raise CausalRevealLedgerError("invalid causal ledger fact")
            species = row.get("species")
            exact_species = row.get("exact_public_species")
            moves = row.get("moves")
            consumed = row.get("consumed_items")
            history = row.get("ability_history")
            move_history = row.get("move_events")
            move_states = row.get("move_states")
            disable_history = row.get("disable_history")
            form_history = row.get("form_history")
            if (
                not isinstance(species, str) or canonical_species(species) != species
                or not isinstance(exact_species, str) or norm(exact_species) != exact_species
                or canonical_species(exact_species) != species
                or not isinstance(moves, list) or any(norm(move) != move for move in moves)
                or not isinstance(consumed, list) or any(norm(item) != item for item in consumed)
                or not isinstance(row.get("item_status_revealed"), bool)
                or not isinstance(history, list)
                or not isinstance(move_history, list)
                or not isinstance(move_states, list)
                or not isinstance(disable_history, list)
                or not isinstance(form_history, list)
            ):
                raise CausalRevealLedgerError("invalid causal ledger fact fields")
            parsed_move_history = []
            previous_move_index = -1
            for event in move_history:
                if not isinstance(event, Mapping):
                    raise CausalRevealLedgerError("invalid causal move event")
                event_index = event.get("event_index")
                event_species = event.get("exact_public_species")
                move = event.get("move")
                authority = event.get("authority")
                cause = event.get("derived_cause")
                if (
                    not isinstance(event_index, int) or event_index <= previous_move_index
                    or not isinstance(event_species, str) or norm(event_species) != event_species
                    or not isinstance(move, str) or norm(move) != move or not move
                    or authority not in {"intrinsic_public_execution", "derived_public_execution"}
                    or (authority == "intrinsic_public_execution" and cause is not None)
                    or (authority == "derived_public_execution" and (
                        not isinstance(cause, str) or not cause
                    ))
                ):
                    raise CausalRevealLedgerError("invalid causal move history fields")
                parsed_move_history.append(CausalMoveEvent(
                    event_index=event_index,
                    exact_public_species=event_species,
                    move=move,
                    authority=authority,
                    derived_cause=cause,
                ))
                previous_move_index = event_index
            parsed_move_states = []
            seen_move_states: set[str] = set()
            for state in move_states:
                if not isinstance(state, Mapping):
                    raise CausalRevealLedgerError("invalid causal move state")
                move = state.get("move")
                current_pp = state.get("current_pp")
                max_pp = state.get("max_pp")
                disabled = state.get("disabled")
                disable_authority = state.get("disable_authority")
                lifecycle_state = state.get("causal_disable_lifecycle_state")
                if (
                    not isinstance(move, str) or norm(move) != move or not move
                    or move in seen_move_states
                    or not isinstance(current_pp, int) or isinstance(current_pp, bool)
                    or not isinstance(max_pp, int) or isinstance(max_pp, bool)
                    or max_pp <= 0 or current_pp < 0 or current_pp > max_pp
                    or disable_authority not in {
                        "causal_disable", "world_mechanical_disable"
                    }
                    or (
                        disable_authority == "causal_disable"
                        and (disabled is not True or lifecycle_state is not True)
                    )
                    or (
                        disable_authority == "world_mechanical_disable"
                        and disabled is not None
                    )
                    or (
                        lifecycle_state is not None
                        and not isinstance(lifecycle_state, bool)
                    )
                    or (
                        disable_authority == "world_mechanical_disable"
                        and lifecycle_state is True
                    )
                    or state.get("authority") not in {
                        "causal_live_public_tracker",
                        "causal_event_counted_public_tracker",
                    }
                ):
                    raise CausalRevealLedgerError("invalid causal move state fields")
                parsed_move_states.append(CausalMoveState(
                    move=move, current_pp=current_pp, max_pp=max_pp,
                    disabled=disabled, disable_authority=disable_authority,
                    causal_disable_lifecycle_state=lifecycle_state,
                    authority=state.get("authority"),
                ))
                seen_move_states.add(move)
            if seen_move_states and seen_move_states != set(moves):
                raise CausalRevealLedgerError(
                    "causal move states disagree with intrinsic revealed moves"
                )
            parsed_disable_history = []
            previous_disable_index = -1
            for event in disable_history:
                if not isinstance(event, Mapping):
                    raise CausalRevealLedgerError("invalid causal disable event")
                event_index = event.get("event_index")
                event_species = event.get("exact_public_species")
                move = event.get("move")
                if (
                    not isinstance(event_index, int) or event_index <= previous_disable_index
                    or not isinstance(event_species, str) or norm(event_species) != event_species
                    or not isinstance(move, str) or norm(move) != move or not move
                    or not isinstance(event.get("disabled"), bool)
                    or event.get("authority") != "explicit_public_disable_lifecycle"
                    or event.get("protocol_tag") not in {"-start", "-end", "switch", "drag"}
                ):
                    raise CausalRevealLedgerError("invalid causal disable history fields")
                parsed_disable_history.append(CausalDisableEvent(
                    event_index=event_index,
                    exact_public_species=event_species,
                    move=move,
                    disabled=event["disabled"],
                    authority=event["authority"],
                    protocol_tag=event["protocol_tag"],
                ))
                previous_disable_index = event_index
            parsed_form_history = []
            previous_form_index = -1
            for event in form_history:
                if not isinstance(event, Mapping):
                    raise CausalRevealLedgerError("invalid causal form event")
                event_index = event.get("event_index")
                exact = event.get("exact_public_species")
                canonical = event.get("canonical_species")
                source_ability = event.get("source_ability")
                if (
                    not isinstance(event_index, int) or event_index <= previous_form_index
                    or not isinstance(exact, str) or norm(exact) != exact or not exact
                    or not isinstance(canonical, str) or canonical_species(exact) != canonical
                    or not isinstance(source_ability, str) or norm(source_ability) != source_ability
                    or event.get("protocol_tag") != "-formechange"
                ):
                    raise CausalRevealLedgerError("invalid causal form history fields")
                parsed_form_history.append(CausalFormEvent(
                    event_index=event_index,
                    exact_public_species=exact,
                    canonical_species=canonical,
                    source_ability=source_ability,
                    protocol_tag="-formechange",
                ))
                previous_form_index = event_index
            parsed_history = []
            previous_index = -1
            for event in history:
                if not isinstance(event, Mapping):
                    raise CausalRevealLedgerError("invalid causal ability event")
                event_index = event.get("event_index")
                event_species = event.get("exact_public_species")
                ability = event.get("ability")
                authority = event.get("authority")
                protocol_tag = event.get("protocol_tag")
                if (
                    not isinstance(event_index, int) or event_index <= previous_index
                    or not isinstance(event_species, str) or norm(event_species) != event_species
                    or not isinstance(ability, str) or norm(ability) != ability or not ability
                    or authority not in {
                        "explicit_public_event",
                        "rule_implied_form_transition",
                        "rule_implied_switch_reactivation",
                    }
                    or not isinstance(protocol_tag, str) or not protocol_tag
                ):
                    raise CausalRevealLedgerError("invalid causal ability history fields")
                parsed_history.append(CausalAbilityEvent(
                    event_index=event_index, exact_public_species=event_species,
                    ability=ability, authority=authority, protocol_tag=protocol_tag,
                ))
                previous_index = event_index
            current_ability = row.get("current_ability")
            current_authority = row.get("current_ability_authority")
            if (
                current_ability is not None
                and (not isinstance(current_ability, str) or norm(current_ability) != current_ability)
            ) or current_authority not in {
                None,
                "explicit_public_event",
                "rule_implied_form_transition",
                "rule_implied_switch_reactivation",
            }:
                raise CausalRevealLedgerError("invalid current ability fields")
            if bool(current_ability) != bool(current_authority):
                raise CausalRevealLedgerError("incomplete current ability authority")
            if current_ability is not None and not parsed_history:
                raise CausalRevealLedgerError(
                    "current ability/history presence disagrees"
                )
            if current_ability is not None and (
                parsed_history[-1].ability != current_ability
                or parsed_history[-1].authority != current_authority
            ):
                raise CausalRevealLedgerError("current ability disagrees with ordered history")
            parsed.append(CausalRevealFact(
                species=species, exact_public_species=exact_species, moves=tuple(moves),
                current_item=row.get("current_item"),
                item_status_revealed=row["item_status_revealed"],
                consumed_items=tuple(consumed), current_ability=current_ability,
                current_ability_authority=current_authority,
                ability_history=tuple(parsed_history),
                move_events=tuple(parsed_move_history),
                move_states=tuple(parsed_move_states),
                disable_history=tuple(parsed_disable_history),
                form_history=tuple(parsed_form_history),
            ))
        ledger = cls(
            battle_tag=str(payload.get("battle_tag") or ""),
            observer_role=str(payload.get("observer_role") or ""),
            opponent_role=str(payload.get("opponent_role") or ""),
            opponent_active_species=str(payload.get("opponent_active_species") or ""),
            facts=tuple(parsed), protocol_sha256=str(payload.get("protocol_sha256") or ""),
        )
        if ledger.observer_role not in {"p1", "p2"} or ledger.opponent_role not in {"p1", "p2"}:
            raise CausalRevealLedgerError("invalid causal ledger roles")
        if ledger.opponent_role == ledger.observer_role or not ledger.battle_tag or not ledger.protocol_sha256:
            raise CausalRevealLedgerError("incomplete causal ledger identity")
        if (
            ledger.opponent_active_species
            and ledger.opponent_active_species not in {fact.species for fact in ledger.facts}
        ):
            raise CausalRevealLedgerError("causal ledger active species is absent")
        return ledger


def _side(ident: str) -> str:
    match = re.match(r"^(p[12])", str(ident or ""))
    return match.group(1) if match else ""


def freeze_ledger(tag: str, observer_role: str, lines: Sequence[str]) -> CausalRevealLedger:
    if observer_role not in {"p1", "p2"}:
        raise CausalRevealLedgerError("invalid live observer role")
    opponent_role = "p2" if observer_role == "p1" else "p1"
    active = {"p1": "", "p2": ""}
    exact_active = {"p1": "", "p2": ""}
    activations: dict[tuple[str, str], int] = {}
    facts: dict[str, _MutableFact] = {}

    def ensure(species: str, exact_species: str | None = None) -> _MutableFact:
        if not species:
            raise CausalRevealLedgerError("public event has no species")
        exact = norm(exact_species or species)
        fact = facts.setdefault(species, _MutableFact(species, exact))
        if exact_species:
            fact.exact_public_species = exact
        return fact

    def actor(ident: str) -> tuple[str, str]:
        side = _side(ident)
        return side, active.get(side, "")

    def record_ability(
        fact: _MutableFact,
        ability: str,
        event_index: int,
        authority: str,
        protocol_tag: str,
    ) -> None:
        normalized = norm(ability)
        if not normalized:
            raise CausalRevealLedgerError("public ability event has no ability")
        event = CausalAbilityEvent(
            event_index=event_index,
            exact_public_species=fact.exact_public_species,
            ability=normalized,
            authority=authority,
            protocol_tag=protocol_tag,
        )
        fact.ability_history.append(event)
        fact.current_ability = normalized
        fact.current_ability_authority = authority

    def merge_identity(old: str, new: str) -> None:
        previous = facts.pop(old, None)
        if previous is None:
            return
        target = facts.get(new) or ensure(new, previous.exact_public_species)
        target.moves.update(previous.moves)
        if previous.item_status_revealed:
            target.item_status_revealed = True
            target.current_item = previous.current_item
        target.consumed_items.update(previous.consumed_items)
        target.ability_history.extend(previous.ability_history)
        target.move_events.extend(previous.move_events)
        target.disable_history.extend(previous.disable_history)
        target.causal_disable_states.update(previous.causal_disable_states)
        target.form_history.extend(previous.form_history)
        target.ability_history.sort(key=lambda event: event.event_index)
        if target.ability_history:
            latest = target.ability_history[-1]
            target.current_ability = latest.ability
            target.current_ability_authority = latest.authority

    for event_index, line in enumerate(lines):
        if line.startswith("|request|"):
            raise CausalRevealLedgerError("private request entered public ledger input")
        if not line.startswith("|"):
            raise CausalRevealLedgerError("malformed public protocol line")
        parts = line.split("|")
        tag_name = parts[1] if len(parts) > 1 else ""
        if tag_name in {"switch", "drag", "replace", "detailschange"} and len(parts) >= 4:
            side = _side(parts[2])
            exact_species = norm(parts[3].split(",", 1)[0])
            species = canonical_species(exact_species)
            if side:
                old = active.get(side, "")
                if side == opponent_role and tag_name in {"switch", "drag"} and old:
                    previous_fact = facts.get(old)
                    if previous_fact is not None:
                        for disabled_move, disabled in list(
                            previous_fact.causal_disable_states.items()
                        ):
                            if disabled:
                                previous_fact.causal_disable_states[disabled_move] = False
                                previous_fact.disable_history.append(CausalDisableEvent(
                                    event_index=event_index,
                                    exact_public_species=previous_fact.exact_public_species,
                                    move=disabled_move,
                                    disabled=False,
                                    authority="explicit_public_disable_lifecycle",
                                    protocol_tag=tag_name,
                                ))
                active[side] = species
                exact_active[side] = exact_species
                activation_count = 0
                if tag_name in {"switch", "drag"}:
                    key = (side, species)
                    activations[key] = activations.get(key, 0) + 1
                    activation_count = activations[key]
                if side == opponent_role:
                    fact = ensure(species, exact_species)
                    if tag_name == "replace" and old and old != species:
                        if activations.get((side, old), 0) > 1:
                            raise CausalRevealLedgerError(
                                "ambiguous repeated species before Illusion replace"
                            )
                        merge_identity(old, species)
                    elif tag_name == "detailschange" and old and old != species:
                        merge_identity(old, species)
                    if tag_name in {"switch", "drag"} and activation_count > 1:
                        # A switch/drag ends transient ability replacement and
                        # reactivates the exact public form.  Never carry the
                        # previous ability across that boundary.  The only
                        # admissible replacement is Showdown's pinned unique
                        # ability for the newly public exact form.
                        implied = form_ability_contract().get(exact_species)
                        if implied is None:
                            fact.current_ability = None
                            fact.current_ability_authority = None
                        else:
                            record_ability(
                                fact,
                                implied,
                                event_index,
                                "rule_implied_switch_reactivation",
                                tag_name,
                            )
                    if tag_name == "detailschange":
                        fact = ensure(species, exact_species)
                        implied = form_ability_contract().get(exact_species)
                        if implied is None:
                            raise CausalRevealLedgerError(
                                f"unsupported public form ability transition: {exact_species}"
                            )
                        record_ability(
                            fact, implied, event_index,
                            "rule_implied_form_transition", "detailschange",
                        )
        elif tag_name == "-formechange" and len(parts) >= 4:
            side, species = actor(parts[2])
            if side == opponent_role:
                exact_species = norm(parts[3].split(",", 1)[0])
                canonical = canonical_species(exact_species)
                if canonical not in {"morpeko", "minior"}:
                    # Cycle 27 changes only its preregistered source-pinned
                    # families. Other battle-form contracts retain the prior
                    # behavior and are neither generalized nor inferred here.
                    continue
                cause_rows = [
                    value.split(":", 1)[1].strip()
                    for value in parts[4:]
                    if value.strip().lower().startswith("[from] ability:")
                ]
                if len(cause_rows) != 1:
                    raise CausalRevealLedgerError(
                        "unsupported public battle-form cause"
                    )
                source_ability = norm(cause_rows[0])
                allowed = (
                    canonical == "morpeko"
                    and exact_species in {"morpeko", "morpekohangry"}
                    and source_ability == "hungerswitch"
                ) or (
                    canonical == "minior"
                    and (
                        exact_species == "miniormeteor"
                        or exact_species == "minior"
                        or exact_species.startswith("minior")
                    )
                    and source_ability == "shieldsdown"
                )
                if not allowed or species != canonical:
                    raise CausalRevealLedgerError(
                        f"unsupported public battle-form transition: {exact_species}"
                    )
                fact = ensure(species, exact_species)
                fact.exact_public_species = exact_species
                exact_active[side] = exact_species
                fact.form_history.append(CausalFormEvent(
                    event_index=event_index,
                    exact_public_species=exact_species,
                    canonical_species=canonical,
                    source_ability=source_ability,
                    protocol_tag="-formechange",
                ))
        elif tag_name == "poke" and len(parts) >= 4 and parts[2] == opponent_role:
            exact_species = norm(parts[3].split(",", 1)[0])
            ensure(canonical_species(exact_species), exact_species)
        elif tag_name == "move" and len(parts) >= 4:
            side, species = actor(parts[2])
            if side == opponent_role:
                fact = ensure(species, exact_active[side])
                move = norm(parts[3])
                if not move:
                    raise CausalRevealLedgerError("public move event has no move")
                derived_causes = [
                    value.strip()[6:].strip()
                    for value in parts[4:]
                    if value.strip().lower().startswith("[from]")
                ]
                if len(derived_causes) > 1 or any(not cause for cause in derived_causes):
                    raise CausalRevealLedgerError("ambiguous public move provenance")
                # Showdown synthesizes Struggle only when no ordinary move is
                # usable (pokemon.ts:getMoves) and explicitly skips normal PP
                # deduction (battle-actions.ts:useMove).  It is causal public
                # history, but never evidence that Struggle occupies a set
                # slot or has a normal PP/disable state.
                if move == "struggle" and not derived_causes:
                    fact.move_events.append(CausalMoveEvent(
                        event_index=event_index,
                        exact_public_species=fact.exact_public_species,
                        move=move,
                        authority="derived_public_execution",
                        derived_cause="mechanic: Struggle",
                    ))
                elif derived_causes:
                    fact.move_events.append(CausalMoveEvent(
                        event_index=event_index,
                        exact_public_species=fact.exact_public_species,
                        move=move,
                        authority="derived_public_execution",
                        derived_cause=derived_causes[0],
                    ))
                else:
                    fact.moves.add(move)
                    fact.move_events.append(CausalMoveEvent(
                        event_index=event_index,
                        exact_public_species=fact.exact_public_species,
                        move=move,
                        authority="intrinsic_public_execution",
                        derived_cause=None,
                    ))
        elif tag_name in {"-item", "-enditem"} and len(parts) >= 4:
            side, species = actor(parts[2])
            if side == opponent_role:
                fact = ensure(species)
                item = norm(parts[3])
                fact.item_status_revealed = True
                fact.current_item = item if tag_name == "-item" else "none"
                if tag_name == "-enditem":
                    fact.consumed_items.add(item)
        elif tag_name == "-ability" and len(parts) >= 4:
            side, species = actor(parts[2])
            if side == opponent_role:
                record_ability(
                    ensure(species, exact_active[side]), parts[3], event_index,
                    "explicit_public_event", "-ability",
                )
        elif tag_name == "-transform" and len(parts) >= 4:
            side, _species = actor(parts[2])
            if side == opponent_role:
                raise CausalRevealLedgerError(
                    "unsupported public ability-changing transform event"
                )
        elif tag_name == "-start" and len(parts) >= 5 and norm(parts[3]) == "disable":
            side, species = actor(parts[2])
            if side == opponent_role:
                move = norm(parts[4])
                cause_rows = [
                    value for value in parts[5:]
                    if value.strip().lower() == "[from] ability: cursed body"
                ]
                if not move or len(cause_rows) != 1:
                    raise CausalRevealLedgerError("unsupported public Disable start")
                fact = ensure(species, exact_active[side])
                fact.causal_disable_states[move] = True
                fact.disable_history.append(CausalDisableEvent(
                    event_index=event_index,
                    exact_public_species=fact.exact_public_species,
                    move=move,
                    disabled=True,
                    authority="explicit_public_disable_lifecycle",
                    protocol_tag="-start",
                ))
        elif tag_name == "-end" and len(parts) >= 4 and norm(parts[3]) == "disable":
            side, species = actor(parts[2])
            if side == opponent_role:
                fact = ensure(species, exact_active[side])
                active_disabled = [
                    move for move, disabled in fact.causal_disable_states.items()
                    if disabled
                ]
                if len(active_disabled) != 1:
                    raise CausalRevealLedgerError("ambiguous public Disable end")
                move = active_disabled[0]
                fact.causal_disable_states[move] = False
                fact.disable_history.append(CausalDisableEvent(
                    event_index=event_index,
                    exact_public_species=fact.exact_public_species,
                    move=move,
                    disabled=False,
                    authority="explicit_public_disable_lifecycle",
                    protocol_tag="-end",
                ))

        source_ident = ""
        for value in parts[3:]:
            if value.strip().lower().startswith("[of] "):
                source_ident = value.strip()[5:].strip()
        source_side, source_species = actor(source_ident or (parts[2] if len(parts) > 2 else ""))
        if source_side == opponent_role and source_species:
            for value in parts[3:]:
                lowered = value.strip().lower()
                if lowered.startswith("[from] item:") or lowered.startswith("item:"):
                    item = norm(value.split(":", 1)[1])
                    fact = ensure(source_species)
                    fact.item_status_revealed = True
                    # A subsequent causal effect line (for example Sitrus
                    # Berry healing) may name an item after ``|-enditem|`` has
                    # already certified its consumption.  Preserve that cause
                    # without resurrecting the consumed held item.
                    if item not in fact.consumed_items:
                        fact.current_item = item
                elif lowered.startswith("[from] ability:") or lowered.startswith("ability:"):
                    record_ability(
                        ensure(source_species, exact_active[source_side]),
                        value.split(":", 1)[1], event_index,
                        "explicit_public_event", tag_name,
                    )

    frozen = tuple(
        CausalRevealFact(
            species=name, exact_public_species=fact.exact_public_species,
            moves=tuple(sorted(fact.moves)),
            current_item=fact.current_item,
            item_status_revealed=fact.item_status_revealed,
            consumed_items=tuple(sorted(fact.consumed_items)),
            current_ability=fact.current_ability,
            current_ability_authority=fact.current_ability_authority,
            ability_history=tuple(fact.ability_history),
            move_events=tuple(fact.move_events),
            move_states=(),
            disable_history=tuple(fact.disable_history),
            form_history=tuple(fact.form_history),
        )
        for name, fact in sorted(facts.items())
    )
    if not frozen:
        raise CausalRevealLedgerError("causal ledger has no opponent species")
    protocol_bytes = json.dumps(list(lines), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return CausalRevealLedger(
        battle_tag=tag, observer_role=observer_role, opponent_role=opponent_role,
        opponent_active_species=active[opponent_role], facts=frozen,
        protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
    )


def _live_opponent_pokemon(battle: Any) -> list[Any]:
    opponent = getattr(battle, "opponent", None)
    active = getattr(opponent, "active", None)
    reserve = list(getattr(opponent, "reserve", ()) or ())
    return ([active] if active is not None else []) + reserve


def _unique_live_fact_pokemon(battle: Any, fact: CausalRevealFact) -> Any:
    rows = _live_opponent_pokemon(battle)
    exact = [row for row in rows if norm(getattr(row, "name", "")) == fact.exact_public_species]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise CausalRevealLedgerError(
            f"causal live move-state exact-form mapping is ambiguous: {fact.exact_public_species}"
        )
    if fact.exact_public_species == fact.species:
        canonical = [
            row for row in rows
            if canonical_species(getattr(row, "name", "")) == fact.species
        ]
        if len(canonical) == 1:
            return canonical[0]
    raise CausalRevealLedgerError(
        f"causal live move-state exact-form mapping failed: {fact.exact_public_species}"
    )


def bind_live_move_states(battle: Any, ledger: CausalRevealLedger) -> CausalRevealLedger:
    """Bind public intrinsic moves to exact unsampled live tracker state."""
    bound_facts = []
    for fact in ledger.facts:
        states = []
        if fact.moves:
            pokemon = _unique_live_fact_pokemon(battle, fact)
            for move in fact.moves:
                matches = [
                    row for row in getattr(pokemon, "moves", ())
                    if norm(getattr(row, "name", "")) == move
                ]
                if len(matches) != 1:
                    raise CausalRevealLedgerError(
                        f"causal live move/PP-disable authority missing: {fact.exact_public_species}/{move}"
                    )
                row = matches[0]
                current_pp = getattr(row, "current_pp", None)
                max_pp = getattr(row, "max_pp", None)
                tracker_disabled = getattr(row, "disabled", None)
                pp_events = getattr(row, "metagross_causal_pp_events", None)
                causal_disable_states: dict[str, bool] = {}
                for event in fact.disable_history:
                    causal_disable_states[event.move] = event.disabled
                causal_disabled = causal_disable_states.get(move)
                disable_authority = (
                    "causal_disable" if causal_disabled is True
                    else "world_mechanical_disable"
                )
                disabled = True if causal_disabled is True else None
                if (
                    not isinstance(current_pp, int) or isinstance(current_pp, bool)
                    or not isinstance(max_pp, int) or isinstance(max_pp, bool)
                    or max_pp <= 0 or current_pp < 0 or current_pp > max_pp
                    or not isinstance(tracker_disabled, bool)
                    or not isinstance(pp_events, list)
                    or any(
                        not isinstance(event, Mapping)
                        or event.get("sequence") != index
                        or not isinstance(event.get("total_cost"), int)
                        or isinstance(event.get("total_cost"), bool)
                        or event.get("total_cost") <= 0
                        or event.get("pressure_extra") not in {0, 1}
                        or (
                            event.get("pressure_extra") == 1
                            and event.get("pressure_authority") not in {
                                "observer_private_request",
                                "explicit_public_ability_event",
                            }
                        )
                        for index, event in enumerate(pp_events)
                    )
                    or sum(event["total_cost"] for event in pp_events)
                    != max_pp - current_pp
                ):
                    raise CausalRevealLedgerError(
                        f"invalid causal live PP-disable state: {fact.exact_public_species}/{move}"
                    )
                states.append(CausalMoveState(
                    move=move, current_pp=current_pp, max_pp=max_pp,
                    disabled=disabled, disable_authority=disable_authority,
                    causal_disable_lifecycle_state=causal_disabled,
                    authority="causal_event_counted_public_tracker",
                ))
                setattr(row, "metagross_disable_authority", disable_authority)
                setattr(row, "metagross_causal_disabled", disabled)
        bound_facts.append(CausalRevealFact(
            species=fact.species,
            exact_public_species=fact.exact_public_species,
            moves=fact.moves,
            current_item=fact.current_item,
            item_status_revealed=fact.item_status_revealed,
            consumed_items=fact.consumed_items,
            current_ability=fact.current_ability,
            current_ability_authority=fact.current_ability_authority,
            ability_history=fact.ability_history,
            move_events=fact.move_events,
            move_states=tuple(states),
            disable_history=fact.disable_history,
            form_history=fact.form_history,
        ))
    return CausalRevealLedger(
        battle_tag=ledger.battle_tag,
        observer_role=ledger.observer_role,
        opponent_role=ledger.opponent_role,
        opponent_active_species=ledger.opponent_active_species,
        facts=tuple(bound_facts),
        protocol_sha256=ledger.protocol_sha256,
    )


def freeze_and_attach_battle_ledger(battle: Any) -> CausalRevealLedger:
    tag = str(getattr(battle, "battle_tag", "") or "")
    observer = str(getattr(getattr(battle, "user", None), "name", "") or "")
    ledger = bind_live_move_states(
        battle, freeze_ledger(tag, observer, protocol_lines_for_battle(tag))
    )
    setattr(battle, LEDGER_ATTRIBUTE, ledger.to_payload())
    return ledger


def attached_ledger(battle: Any) -> CausalRevealLedger:
    payload = getattr(battle, LEDGER_ATTRIBUTE, None)
    if not isinstance(payload, Mapping):
        raise CausalRevealLedgerError("sampled battle lacks causal ledger")
    return CausalRevealLedger.from_payload(payload)


def verify_sampled_ledgers(source: Any, worlds: Sequence[tuple[Any, Any]]) -> None:
    expected = attached_ledger(source).canonical_bytes()
    for sampled, _weight in worlds:
        if attached_ledger(sampled).canonical_bytes() != expected:
            raise CausalRevealLedgerError("belief sampler changed causal ledger")


def verify_sampled_move_states(source: Any, worlds: Sequence[tuple[Any, Any]]) -> None:
    """Require every sampled completion to preserve exact causal move state."""
    ledger = attached_ledger(source)
    original_weights = [weight for _sampled, weight in worlds]
    derived_executions = [
        event.to_payload()
        for fact in ledger.facts
        for event in fact.move_events
        if event.authority == "derived_public_execution"
    ]
    for sampled, _weight in worlds:
        receipts = []
        for fact in ledger.facts:
            if not fact.moves:
                continue
            pokemon = _unique_live_fact_pokemon(sampled, fact)
            expected = {state.move: state for state in fact.move_states}
            if set(expected) != set(fact.moves):
                raise CausalRevealLedgerError(
                    f"causal move-state contract incomplete: {fact.exact_public_species}"
                )
            for move, state in expected.items():
                matches = [
                    row for row in getattr(pokemon, "moves", ())
                    if norm(getattr(row, "name", "")) == move
                ]
                if len(matches) != 1:
                    raise CausalRevealLedgerError(
                        f"sampled world missing causal move: {fact.exact_public_species}/{move}"
                    )
                row = matches[0]
                if (
                    getattr(row, "current_pp", None) != state.current_pp
                    or getattr(row, "max_pp", None) != state.max_pp
                    or (
                        state.disable_authority == "causal_disable"
                        and getattr(row, "disabled", None) is not state.disabled
                    )
                ):
                    raise CausalRevealLedgerError(
                        f"sampled world changed causal PP-disable state: {fact.exact_public_species}/{move}"
                    )
                receipts.append({
                    "exact_public_species": fact.exact_public_species,
                    "move": move,
                    "current_pp": state.current_pp,
                    "max_pp": state.max_pp,
                    "disable_authority": state.disable_authority,
                    "causal_disabled": state.disabled,
                    "causal_disable_lifecycle_state": (
                        state.causal_disable_lifecycle_state
                    ),
                    "world_disabled": bool(getattr(row, "disabled", False)),
                })
        setattr(sampled, MOVE_RECEIPT_ATTRIBUTE, {
            "schema": "metagross-causal-move-world-receipts/v1",
            "battle_tag": ledger.battle_tag,
            "protocol_sha256": ledger.protocol_sha256,
            "moves": receipts,
            "derived_executions": copy.deepcopy(derived_executions),
        })
    if [weight for _sampled, weight in worlds] != original_weights:
        raise CausalRevealLedgerError("causal verification changed sampler weights")


def parse_state_serialization(state_string: str) -> dict[str, str]:
    values = state_string.split("/")
    if len(values) != len(STATE_SERIALIZATION_FIELDS):
        raise CausalRevealLedgerError(
            f"engine serialization has {len(values)} fields; expected "
            f"{len(STATE_SERIALIZATION_FIELDS)}"
        )
    return dict(zip(STATE_SERIALIZATION_FIELDS, values, strict=True))


def serialize_state_fields(fields: Mapping[str, str]) -> str:
    if set(fields) != set(STATE_SERIALIZATION_FIELDS):
        raise CausalRevealLedgerError("engine serialization field names disagree")
    return "/".join(str(fields[name]) for name in STATE_SERIALIZATION_FIELDS)


def serialization_without_masks(state: Any) -> str:
    fields = parse_state_serialization(state.to_string())
    fields["s1_public_reveals"] = "0"
    fields["s2_public_reveals"] = "0"
    return serialize_state_fields(fields)


def compile_reveal_bits(state: Any, ledger: CausalRevealLedger, *, swap: bool) -> int:
    opponent = state.side_one if swap else state.side_two
    exact_slots: dict[str, int] = {}
    exact_ambiguous: set[str] = set()
    canonical_slots: dict[str, int] = {}
    canonical_ambiguous: set[str] = set()
    for index, pokemon in enumerate(opponent.pokemon):
        exact_species = norm(pokemon.id)
        species = canonical_species(exact_species)
        if exact_species in {"", "none"} or species in {"", "none"}:
            continue
        if exact_species in exact_slots:
            exact_ambiguous.add(exact_species)
        exact_slots[exact_species] = index
        if species in canonical_slots:
            canonical_ambiguous.add(species)
        canonical_slots[species] = index
    for species in exact_ambiguous:
        exact_slots.pop(species, None)
    for species in canonical_ambiguous:
        canonical_slots.pop(species, None)
    bits = 0
    claimed: set[int] = set()
    for fact in ledger.facts:
        slot = exact_slots.get(fact.exact_public_species)
        if slot is None and fact.exact_public_species == fact.species:
            slot = canonical_slots.get(fact.species)
        if slot is None or slot in claimed:
            raise CausalRevealLedgerError(
                f"public exact species mapping failed: {fact.exact_public_species}"
            )
        claimed.add(slot)
        bits |= 1 << slot
        pokemon = opponent.pokemon[slot]
        move_states = {state.move: state for state in fact.move_states}
        if fact.moves and set(move_states) != set(fact.moves):
            raise CausalRevealLedgerError(
                f"causal live PP-disable authority missing: {fact.exact_public_species}"
            )
        for move in fact.moves:
            matches = [index for index, row in enumerate(pokemon.moves) if norm(row.id) == move]
            if len(matches) != 1:
                raise CausalRevealLedgerError(
                    f"public move/PP-disable authority missing: {fact.species}/{move}"
                )
            engine_move = pokemon.moves[matches[0]]
            expected_state = move_states[move]
            if (
                int(getattr(engine_move, "pp", -1)) != expected_state.current_pp
                or (
                    expected_state.disable_authority == "causal_disable"
                    and bool(getattr(engine_move, "disabled", None))
                    != expected_state.disabled
                )
            ):
                raise CausalRevealLedgerError(
                    f"engine move PP-disable state mismatch: {fact.exact_public_species}/{move}"
                )
            bits |= 1 << (6 + slot * 4 + matches[0])
        if fact.item_status_revealed:
            expected = fact.current_item or "none"
            if norm(pokemon.item) != norm(expected):
                raise CausalRevealLedgerError(f"public item mismatch: {fact.species}")
            bits |= 1 << (30 + slot)
        if fact.current_ability is not None:
            if norm(pokemon.ability) != fact.current_ability:
                raise CausalRevealLedgerError(
                    f"public current ability mismatch: {fact.exact_public_species}"
                )
            bits |= 1 << (36 + slot)
    if bits <= 0 or bits & ~VALID_MASK:
        raise CausalRevealLedgerError("invalid compiled causal mask")
    return bits


def hydrate_certified_abilities(
    state: Any,
    ledger: CausalRevealLedger,
    *,
    swap: bool,
    receipt_context: Mapping[str, Any] | None = None,
) -> Any:
    """Install only ordered, causally certified current abilities.

    Exact public form identity is mandatory here: unlike non-ability reveal
    bits, base-form fallback would allow a sampled hidden same-base completion
    to receive a public form's mechanics.
    """
    opponent = state.side_one if swap else state.side_two
    exact_slots: dict[str, int] = {}
    ambiguous: set[str] = set()
    for index, pokemon in enumerate(opponent.pokemon):
        exact = norm(pokemon.id)
        if exact in {"", "none"}:
            continue
        if exact in exact_slots:
            ambiguous.add(exact)
        exact_slots[exact] = index
    installed = state
    receipts = []
    for fact in ledger.facts:
        if fact.current_ability is None:
            continue
        if not fact.ability_history:
            raise CausalRevealLedgerError("certified current ability has no history")
        latest = fact.ability_history[-1]
        if (
            latest.ability != fact.current_ability
            or latest.authority != fact.current_ability_authority
            or latest.exact_public_species != fact.exact_public_species
        ):
            raise CausalRevealLedgerError(
                "certified current ability disagrees with latest exact-form event"
            )
        exact = fact.exact_public_species
        if exact in ambiguous or exact not in exact_slots:
            raise CausalRevealLedgerError(
                f"certified ability exact-form mapping is not unique: {exact}"
            )
        update_base = latest.authority in {
            "rule_implied_form_transition",
            "rule_implied_switch_reactivation",
        }
        if update_base:
            implied = form_ability_contract().get(exact)
            if implied != fact.current_ability:
                raise CausalRevealLedgerError(
                    f"certified form ability no longer matches pinned contract: {exact}"
                )
        elif latest.authority != "explicit_public_event":
            raise CausalRevealLedgerError("unsupported certified ability authority")
        setter_name = (
            "with_side_one_pokemon_ability" if swap
            else "with_side_two_pokemon_ability"
        )
        setter = getattr(installed, setter_name, None)
        if not callable(setter):
            raise CausalRevealLedgerError("engine lacks certified ability setter")
        try:
            installed = setter(
                exact_slots[exact], fact.current_ability, update_base
            )
        except Exception as exc:
            raise CausalRevealLedgerError(
                f"certified ability installation failed: {exact}"
            ) from exc
        installed_opponent = installed.side_one if swap else installed.side_two
        installed_pokemon = installed_opponent.pokemon[exact_slots[exact]]
        if norm(installed_pokemon.ability) != fact.current_ability:
            raise CausalRevealLedgerError(
                f"certified current ability installation did not persist: {exact}"
            )
        if update_base and norm(installed_pokemon.base_ability) != fact.current_ability:
            raise CausalRevealLedgerError(
                f"certified base ability installation did not persist: {exact}"
            )
        receipts.append({
            "exact_public_species": exact,
            "slot": exact_slots[exact],
            "authority": latest.authority,
            "installed_current_ability": fact.current_ability,
            "installed_base_ability": (
                fact.current_ability if update_base else "preserved_not_recorded"
            ),
            "update_base": update_base,
        })
    receipt_dir = os.environ.get("METAGROSS_CAUSAL_ABILITY_RECEIPT_DIR")
    if receipts and receipt_dir:
        directory = Path(receipt_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        namespace = norm(os.environ.get("METAGROSS_PRIOR_NAMESPACE") or "unknown")
        path = directory / f"{namespace}-{os.getpid()}.jsonl"
        payload = {
            "schema": (
                "metagross-certified-ability-installation/v2"
                if receipt_context is not None
                else "metagross-certified-ability-installation/v1"
            ),
            "battle_tag": ledger.battle_tag,
            "observer_role": ledger.observer_role,
            "protocol_sha256": ledger.protocol_sha256,
            "swap": swap,
            "installations": receipts,
        }
        if receipt_context is not None:
            context = copy.deepcopy(dict(receipt_context))
            required = {
                "phase", "cohort", "battle_tag", "rqid", "decision_index",
                "root_id", "declared_world_count", "conversion_index",
                "schedule_index", "world_index",
            }
            if set(context) != required or context["battle_tag"] != ledger.battle_tag:
                raise CausalRevealLedgerError("invalid execution-only receipt context")
            payload["execution_context"] = context
            payload["receipt_time_ns"] = time.time_ns()
        line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        with _ABILITY_RECEIPT_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
    return installed


def install_observer_mask(state: Any, bits: int, *, swap: bool, engine: Any) -> Any:
    if bits <= 0 or bits & ~VALID_MASK:
        raise CausalRevealLedgerError("invalid observer mask")
    before = serialization_without_masks(state)
    if int(state.s1_public_reveals) != 0 or int(state.s2_public_reveals) != 0:
        raise CausalRevealLedgerError("converter supplied non-causal reveal bits")
    if swap:
        installed = state.with_side_two_public_reveals(bits)
        if int(installed.s1_public_reveals) != 0 or int(installed.s2_public_reveals) != bits:
            raise CausalRevealLedgerError("side-two reveal-mask installation failed")
    else:
        installed = state.with_side_one_public_reveals(bits)
        if int(installed.s1_public_reveals) != bits or int(installed.s2_public_reveals) != 0:
            raise CausalRevealLedgerError("side-one reveal-mask installation failed")
    if serialization_without_masks(installed) != before:
        raise CausalRevealLedgerError("mask installation changed non-mask engine bytes")
    return installed


def convert_battle_with_causal_ledger(
    battle: Any,
    converter: Callable[..., Any],
    engine: Any,
    *,
    swap: bool = False,
    receipt_context: Mapping[str, Any] | None = None,
) -> Any:
    ledger = attached_ledger(battle)
    state = converter(battle, swap=swap)
    state = hydrate_certified_abilities(
        state, ledger, swap=swap, receipt_context=receipt_context
    )
    bits = compile_reveal_bits(state, ledger, swap=swap)
    installed = install_observer_mask(state, bits, swap=swap, engine=engine)
    _emit_move_world_receipt(
        battle, ledger, installed, swap=swap, receipt_context=receipt_context
    )
    return installed


def _emit_move_world_receipt(
    battle: Any,
    ledger: CausalRevealLedger,
    state: Any,
    *,
    swap: bool,
    receipt_context: Mapping[str, Any] | None,
) -> None:
    directory_value = os.environ.get("METAGROSS_CAUSAL_MOVE_RECEIPT_DIR")
    if not directory_value:
        return
    if receipt_context is None:
        raise CausalRevealLedgerError(
            "causal move receipt requires execution-only context"
        )
    receipt = getattr(battle, MOVE_RECEIPT_ATTRIBUTE, None)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != "metagross-causal-move-world-receipts/v1"
        or receipt.get("battle_tag") != ledger.battle_tag
        or receipt.get("protocol_sha256") != ledger.protocol_sha256
        or not isinstance(receipt.get("moves"), list)
        or not isinstance(receipt.get("derived_executions"), list)
    ):
        raise CausalRevealLedgerError("causal move world receipt is absent or invalid")
    opponent = state.side_one if swap else state.side_two
    slots: dict[str, list[Any]] = {}
    for pokemon in opponent.pokemon:
        exact = norm(pokemon.id)
        if exact not in {"", "none"}:
            slots.setdefault(exact, []).append(pokemon)
    for row in receipt["moves"]:
        if not isinstance(row, Mapping):
            raise CausalRevealLedgerError("invalid causal move world receipt row")
        candidates = slots.get(row.get("exact_public_species"), [])
        if len(candidates) != 1:
            raise CausalRevealLedgerError("causal move receipt engine slot disagrees")
        moves = [
            move for move in candidates[0].moves
            if norm(move.id) == row.get("move")
        ]
        if (
            len(moves) != 1
            or int(moves[0].pp) != row.get("current_pp")
            or bool(moves[0].disabled) is not row.get("world_disabled")
        ):
            raise CausalRevealLedgerError(
                "causal move receipt differs from search engine world"
            )
    context = copy.deepcopy(dict(receipt_context))
    required = {
        "phase", "cohort", "battle_tag", "rqid", "decision_index",
        "root_id", "declared_world_count", "conversion_index",
        "schedule_index", "world_index",
    }
    if set(context) != required or context["battle_tag"] != ledger.battle_tag:
        raise CausalRevealLedgerError("invalid causal move receipt context")
    payload = {
        "schema": "metagross-causal-move-conversion-receipt/v1",
        "battle_tag": ledger.battle_tag,
        "observer_role": ledger.observer_role,
        "protocol_sha256": ledger.protocol_sha256,
        "swap": swap,
        "execution_context": context,
        "move_receipt": copy.deepcopy(dict(receipt)),
        "receipt_time_ns": time.time_ns(),
    }
    directory = Path(directory_value).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    namespace = norm(os.environ.get("METAGROSS_PRIOR_NAMESPACE") or "unknown")
    path = directory / f"{namespace}-{os.getpid()}.jsonl"
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with _MOVE_RECEIPT_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def causal_engine_payload(state: Any, ledger: CausalRevealLedger, *, swap: bool) -> dict[str, Any]:
    payload = {
        "schema": ENGINE_PAYLOAD_SCHEMA,
        "observer": "side_two" if swap else "side_one",
        "state": state.to_string(),
        "ledger": ledger.to_payload(),
    }
    json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return payload


def clone_ledger_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(payload))
