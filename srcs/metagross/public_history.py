"""Typed append-only projection of directly observed Showdown battle facts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeAlias


EVENT_SCHEMA = "metagross-public-event/v2"


def normalize_id(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _actor(ident: str, observer_role: str) -> str | None:
    role = ident.split(":", 1)[0].strip()[:2]
    if role == observer_role:
        return "self"
    if role in {"p1", "p2"}:
        return "opponent"
    return None


def _details(details: str) -> tuple[str, int | None]:
    fields = [field.strip() for field in details.split(",")]
    species = normalize_id(fields[0]) if fields else ""
    level = next(
        (
            int(match.group(1))
            for field in fields[1:]
            if (match := re.fullmatch(r"L(\d+)", field)) is not None
        ),
        None,
    )
    return species, level


def _condition(condition: str) -> tuple[float | None, str | None, bool]:
    fields = condition.strip().split()
    hp_fraction = None
    if fields:
        hp = fields[0]
        if hp.endswith("%"):
            try:
                hp_fraction = float(hp[:-1]) / 100.0
            except ValueError:
                pass
        elif "/" in hp:
            current, maximum = hp.split("/", 1)
            try:
                maximum_value = float(maximum)
                if maximum_value > 0:
                    hp_fraction = float(current) / maximum_value
            except ValueError:
                pass
    fainted = "fnt" in fields or hp_fraction == 0.0
    status = next(
        (normalize_id(field) for field in fields[1:] if field != "fnt"), None
    )
    return hp_fraction, status, fainted


@dataclass(frozen=True)
class SwitchEvent:
    sequence: int
    actor: str
    species: str
    level: int | None
    hp_fraction: float | None
    status: str | None
    forced: bool
    replaced_identity: bool
    previous_species: str | None
    kind: str = "switch"
    schema: str = EVENT_SCHEMA


@dataclass(frozen=True)
class MoveEvent:
    sequence: int
    actor: str
    species: str | None
    move_id: str
    kind: str = "move"
    schema: str = EVENT_SCHEMA


@dataclass(frozen=True)
class AbilityEvent:
    sequence: int
    actor: str
    species: str | None
    ability_id: str
    stable_identity_evidence: bool
    kind: str = "ability"
    schema: str = EVENT_SCHEMA


@dataclass(frozen=True)
class ItemEvent:
    sequence: int
    actor: str
    species: str | None
    item_id: str
    removed: bool
    stable_identity_evidence: bool
    kind: str = "item"
    schema: str = EVENT_SCHEMA


@dataclass(frozen=True)
class TeraEvent:
    sequence: int
    actor: str
    species: str | None
    tera_type: str
    kind: str = "tera"
    schema: str = EVENT_SCHEMA


@dataclass(frozen=True)
class StatusEvent:
    sequence: int
    actor: str
    species: str | None
    status: str
    kind: str = "status"
    schema: str = EVENT_SCHEMA


@dataclass(frozen=True)
class FaintEvent:
    sequence: int
    actor: str
    species: str | None
    kind: str = "faint"
    schema: str = EVENT_SCHEMA


@dataclass(frozen=True)
class HazardAvoidanceEvent:
    sequence: int
    actor: str
    species: str
    hazard_id: str
    avoided: bool
    item_effects_suppressed: bool
    ability_effects_suppressed: bool
    kind: str = "hazard_avoidance"
    schema: str = EVENT_SCHEMA


PublicEvent: TypeAlias = (
    SwitchEvent
    | MoveEvent
    | AbilityEvent
    | ItemEvent
    | TeraEvent
    | StatusEvent
    | FaintEvent
    | HazardAvoidanceEvent
)


class PublicEventLedger:
    """Append protocol lines while exposing only immutable public event snapshots."""

    def __init__(self, observer_role: str):
        if observer_role not in {"p1", "p2"}:
            raise ValueError("observer role must be p1 or p2")
        self.observer_role = observer_role
        self._events: list[PublicEvent] = []
        self._active_species: dict[str, str] = {}
        self._seen_species: dict[str, set[str]] = {"self": set(), "opponent": set()}
        self._stealth_rock: dict[str, bool] = {"self": False, "opponent": False}
        self._pending_rock_switch: tuple[str, str] | None = None
        self._magic_room = False
        self._active_ability: dict[str, str | None] = {"self": None, "opponent": None}
        self._item_identity_stable: dict[tuple[str, str], bool] = {}

    @property
    def events(self) -> tuple[PublicEvent, ...]:
        return tuple(self._events)

    def _record_hazard_avoidance(self, avoided: bool) -> HazardAvoidanceEvent | None:
        if self._pending_rock_switch is None:
            return None
        actor, species = self._pending_rock_switch
        event = HazardAvoidanceEvent(
            len(self._events),
            actor,
            species,
            "stealthrock",
            avoided,
            item_effects_suppressed=self._magic_room,
            ability_effects_suppressed="neutralizinggas" in self._active_ability.values(),
        )
        self._events.append(event)
        self._pending_rock_switch = None
        return event

    def append(self, line: str) -> PublicEvent | None:
        """Append one directly observed fact; unrelated protocol lines are ignored."""
        if not isinstance(line, str) or not line.startswith("|"):
            raise ValueError("protocol line must start with '|'")
        parts = line.split("|")
        if len(parts) < 3:
            return None
        message = parts[1]
        actor = _actor(parts[2], self.observer_role)
        if message == "-fieldstart" and any(
            normalize_id(field.removeprefix("move:")) == "magicroom"
            for field in parts[2:]
        ):
            self._magic_room = True
            return None
        if message == "-fieldend" and any(
            normalize_id(field.removeprefix("move:")) == "magicroom"
            for field in parts[2:]
        ):
            self._magic_room = False
            return None
        if message == "-sidestart" and actor is not None and any(
            normalize_id(field.removeprefix("move:")) == "stealthrock"
            for field in parts[3:]
        ):
            self._stealth_rock[actor] = True
            return None
        if message == "-sideend" and actor is not None and any(
            normalize_id(field.removeprefix("move:")) == "stealthrock"
            for field in parts[3:]
        ):
            self._stealth_rock[actor] = False
            return None
        if (
            message == "-damage"
            and actor is not None
            and self._pending_rock_switch == (actor, self._active_species.get(actor))
            and any(normalize_id(field.removeprefix("[from]")) == "stealthrock" for field in parts[4:])
        ):
            return self._record_hazard_avoidance(False)
        if self._pending_rock_switch is not None and message in {
            "move", "turn", "upkeep", "switch", "drag", "replace", "faint"
        }:
            self._record_hazard_avoidance(True)
        if actor is None:
            return None
        sequence = len(self._events)
        event: PublicEvent | None = None
        if message in {"switch", "drag", "replace"} and len(parts) >= 5:
            species, level = _details(parts[3])
            if not species:
                return None
            hp_fraction, status, _fainted = _condition(parts[4])
            previous_species = self._active_species.get(actor)
            event = SwitchEvent(
                sequence,
                actor,
                species,
                level,
                hp_fraction,
                status,
                forced=message == "drag",
                replaced_identity=message == "replace",
                previous_species=previous_species,
            )
            self._active_species[actor] = species
            self._active_ability[actor] = None
            self._item_identity_stable.setdefault((actor, species), True)
            first_appearance = species not in self._seen_species[actor]
            self._seen_species[actor].add(species)
            if first_appearance and self._stealth_rock[actor] and message != "replace":
                self._pending_rock_switch = (actor, species)
        elif message == "move" and len(parts) >= 4:
            event = MoveEvent(
                sequence, actor, self._active_species.get(actor), normalize_id(parts[3])
            )
        elif message == "-ability" and len(parts) >= 4:
            ability_id = normalize_id(parts[3])
            event = AbilityEvent(
                sequence,
                actor,
                self._active_species.get(actor),
                ability_id,
                stable_identity_evidence=not any(
                    field.startswith("[from] move:") or field == "[from] ability: Trace"
                    for field in parts[4:]
                ),
            )
            if event.stable_identity_evidence:
                self._active_ability[actor] = ability_id
        elif message in {"-item", "-enditem"} and len(parts) >= 4:
            transferred = message == "-item" and any(
                field.startswith("[from] move:") for field in parts[4:]
            )
            identity_key = (actor, self._active_species.get(actor) or "")
            identity_stable = self._item_identity_stable.get(identity_key, True)
            event = ItemEvent(
                sequence,
                actor,
                self._active_species.get(actor),
                normalize_id(parts[3]),
                removed=message == "-enditem",
                stable_identity_evidence=identity_stable and not transferred,
            )
            if transferred:
                self._item_identity_stable[identity_key] = False
        elif message in {"-damage", "-heal", "-status"}:
            item_id = next(
                (
                    normalize_id(field.removeprefix("[from] item:"))
                    for field in parts[3:]
                    if field.startswith("[from] item:")
                ),
                "",
            )
            if item_id:
                owner = next(
                    (
                        _actor(field.removeprefix("[of] "), self.observer_role)
                        for field in parts[3:]
                        if field.startswith("[of] ")
                    ),
                    actor,
                )
                if owner is None:
                    return None
                event = ItemEvent(
                    sequence,
                    owner,
                    self._active_species.get(owner),
                    item_id,
                    removed=False,
                    stable_identity_evidence=self._item_identity_stable.get(
                        (owner, self._active_species.get(owner) or ""), True
                    ),
                )
        elif message == "-terastallize" and len(parts) >= 4:
            event = TeraEvent(
                sequence, actor, self._active_species.get(actor), normalize_id(parts[3])
            )
        elif message == "-status" and len(parts) >= 4:
            event = StatusEvent(
                sequence, actor, self._active_species.get(actor), normalize_id(parts[3])
            )
        elif message == "faint":
            event = FaintEvent(sequence, actor, self._active_species.get(actor))
            self._active_ability[actor] = None
        if event is not None:
            self._events.append(event)
        return event

    def extend(self, lines: list[str] | tuple[str, ...]) -> tuple[PublicEvent, ...]:
        for line in lines:
            self.append(line)
        return self.events
