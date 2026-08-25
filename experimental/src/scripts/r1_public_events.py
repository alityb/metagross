"""Information-safe public-event projection for certified r1 continuations."""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


EVENT_SCHEMA = "r1-public-event/v1"
CERTIFICATE = "r1-switch-v1"
BOOSTED_DOUBLE_SWITCH_CERTIFICATE = "r1-boosted-double-switch-v1"
BASIC_MOVE_CERTIFICATE = "r1-basic-move-v1"
SILENT_MECHANICS_CERTIFICATE = "r1-silent-mechanics-v1"
DECLARATIVE_BOOST_CERTIFICATE = "r1-declarative-boosts-v1"
MIXED_BOOST_SWITCH_CERTIFICATE = "r1-mixed-boost-switch-v1"
MIXED_BOOST_SWITCH_TARGETS = frozenset({"gothitelle", "munkidori", "thundurus"})
LEFTOVERS_ACTIVATION_CERTIFICATE = "r1-leftovers-activation-v1"
SEMANTIC_TRACE_CERTIFICATE = "r1-semantic-trace-v1"
ENGINE_DEBUG_CONTRACT = "poke-engine-0.0.47-r1-switch-v1"
R1_SEMANTIC_CONTRACT = "poke-engine-0.0.47-r1-item-activation-v1"
SWITCH_ACTION = re.compile(r"^switch ([a-z0-9]+)$")
SWITCH_INSTRUCTION = re.compile(
    r"^Switch (SideOne|SideTwo): P([0-5]) -> P([0-5])$"
)
LAST_MOVE_INSTRUCTION = re.compile(
    r"^SetLastUsedMove (SideOne|SideTwo): "
    r"(?:None|Move\(M[0-3]\)|Switch\(P[0-5]\)) -> Switch\(P([0-5])\)$"
)
BOOST_RESET_INSTRUCTION = re.compile(
    r"^Boost SideOne (SpecialAttack|SpecialDefense): (-[12])$"
)
SIDE_CONDITION_FIELDS = (
    "spikes", "toxic_spikes", "stealth_rock", "sticky_web", "tailwind",
    "lucky_chant", "lunar_dance", "reflect", "light_screen", "aurora_veil",
    "crafty_shield", "safeguard", "mist", "protect", "healing_wish",
    "mat_block", "quick_guard", "toxic_count", "wide_guard",
)
VOLATILE_DURATION_FIELDS = ("confusion", "encore", "lockedmove", "slowstart", "taunt", "yawn")
SIDE_FIELDS = (
    "baton_passing", "shed_tailing", "wish", "future_sight", "force_switch",
    "force_trapped", "slow_uturn_move", "volatile_statuses", "substitute_health",
    "attack_boost", "defense_boost", "special_attack_boost",
    "special_defense_boost", "speed_boost", "accuracy_boost", "evasion_boost",
    "switch_out_move_second_saved_move",
)
POKEMON_FIELDS = (
    "id", "level", "types", "base_types", "hp", "maxhp", "ability",
    "base_ability", "item", "nature", "evs", "attack", "defense",
    "special_attack", "special_defense", "speed", "status", "rest_turns",
    "sleep_turns", "weight_kg", "terastallized", "tera_type",
)
STATE_FIELDS = (
    "weather", "weather_turns_remaining", "terrain", "terrain_turns_remaining",
    "trick_room", "trick_room_turns_remaining", "team_preview", "s1_threat",
    "s2_threat", "scout_value", "threat_matrix", "wincon_matrix",
)
TRAPPING_ABILITIES = {"arenatrap", "magnetpull", "shadowtag"}
BASIC_MOVE_IDS = {
    "agility", "bulkup", "calmmind", "nastyplot", "nightshade", "recover",
    "roost", "seismictoss", "slackoff", "softboiled", "spore", "swordsdance",
    "tackle", "thunderwave", "toxic", "willowisp",
}
BASIC_MOVE_PRIVATE_BLOCKER_CODES = (
    "GLOBAL_FIELD",
    "SIDE_FORCED_OR_PIVOT",
    "SIDE_BOOST",
    "SIDE_CONDITION",
    "SIDE_VOLATILE",
    "SIDE_DELAYED_EFFECT",
    "SIDE_OTHER",
    "TERA_OR_TYPE_CHANGE",
    "ACTIVE_FAINTED",
    "ACTIVE_STATUS",
    "ACTIVE_ITEM",
    "ACTIVE_ABILITY",
    "ACTION_SWITCH",
    "ACTION_TERA",
    "ACTION_UNLISTED",
)
BASIC_MOVE_PRIVATE_OUTCOME_CODES = (
    "ADMITTED",
    "UNSUPPORTED_PUBLIC_PRESTATE",
    "UNSUPPORTED_ACTION_PAIR",
    "UNSUPPORTED_ENGINE_DELTA",
    "UNACCOUNTED_INSTRUCTION",
    "UNSUPPORTED_EXECUTED_ACTION",
    "UNSUPPORTED_HP_SEQUENCE",
    "UNSUPPORTED_STATUS",
    "UNSUPPORTED_SEMANTIC_EVENT",
    "UNSUPPORTED_MECHANIC_ACTIVATION",
    "ENGINE_OR_BINDING_ERROR",
)
ENGINE_STATUS = {
    "burn": "brn",
    "freeze": "frz",
    "paralyze": "par",
    "poison": "psn",
    "sleep": "slp",
    "toxic": "tox",
    "none": "nostatus",
}
SIDE_FORCED_OR_PIVOT_FIELDS = {
    "baton_passing",
    "shed_tailing",
    "force_switch",
    "force_trapped",
    "slow_uturn_move",
}
SIDE_BOOST_FIELDS = {
    "attack_boost",
    "defense_boost",
    "special_attack_boost",
    "special_defense_boost",
    "speed_boost",
    "accuracy_boost",
    "evasion_boost",
}
SIDE_VOLATILE_FIELDS = {"volatile_statuses", "substitute_health"}
SIDE_DELAYED_EFFECT_FIELDS = {"wish", "future_sight"}
SELF_BOOST_RULES = {
    "agility": (("speed", 2),),
    "bulkup": (("attack", 1), ("defense", 1)),
    "calmmind": (("specialattack", 1), ("specialdefense", 1)),
    "nastyplot": (("specialattack", 2),),
    "swordsdance": (("attack", 2),),
}


class PublicEventProjectionError(ValueError):
    """Constant-code failure that never includes hidden mechanical values."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PublicSwitchEvent:
    actor: str
    species: str
    level: int
    hp_fraction: float
    status: str
    previously_revealed: bool
    schema: str = EVENT_SCHEMA
    kind: str = "switch"

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InformationSetSwitchProjection:
    events: tuple[PublicSwitchEvent, ...]
    next_states: tuple[Any, ...]
    legal_actions: tuple[str, ...] | None = None
    cleared_self_boosts: tuple[tuple[str, int], ...] = ()
    certificate: str = CERTIFICATE


@dataclass(frozen=True)
class PublicMoveEvent:
    actor: str
    move_id: str
    schema: str = EVENT_SCHEMA
    kind: str = "move"


@dataclass(frozen=True)
class PublicHPEvent:
    actor: str
    hp_fraction: float
    fainted: bool
    schema: str = EVENT_SCHEMA
    kind: str = "hp"


@dataclass(frozen=True)
class PublicStatusEvent:
    actor: str
    status: str
    schema: str = EVENT_SCHEMA
    kind: str = "status"


@dataclass(frozen=True)
class PublicBoostEvent:
    actor: str
    stat: str
    amount: int
    schema: str = EVENT_SCHEMA
    kind: str = "boost"


@dataclass(frozen=True)
class PublicItemEvent:
    actor: str
    item_id: str
    active_after: str
    schema: str = EVENT_SCHEMA
    kind: str = "item"


@dataclass(frozen=True)
class OwnPrivateDelta:
    active_species: str
    hp: int
    max_hp: int
    status: str
    moves: tuple[tuple[str, int, bool], ...]


@dataclass(frozen=True)
class BasicMoveObservationClass:
    events: tuple[Any, ...]
    own_delta: OwnPrivateDelta
    legal_actions: tuple[str, ...]
    next_states: tuple[Any, ...]
    source_world_indices: tuple[int, ...]
    certificate: str = BASIC_MOVE_CERTIFICATE


@dataclass(frozen=True)
class BasicMoveProjection:
    observation_classes: tuple[BasicMoveObservationClass, ...]
    certificate: str = BASIC_MOVE_CERTIFICATE


@dataclass(frozen=True)
class TransformerObservation:
    """The only continuation payload that may cross into the r1 model."""

    text_tokens: tuple[int, ...]
    numbers: tuple[float, ...]
    illegal_actions: tuple[bool, ...]
    name_table: tuple[tuple[str, int], ...]
    terminal: bool = False
    automatic_action: str | None = None

    def policy_payload(self) -> dict[str, Any]:
        """Return fresh primitive containers suitable for model inference."""
        return {
            "text_tokens": list(self.text_tokens),
            "numbers": list(self.numbers),
            "illegal_actions": list(self.illegal_actions),
            "name_table": dict(self.name_table),
            "terminal": self.terminal,
            "automatic_action": self.automatic_action,
        }


@dataclass(frozen=True)
class TransformerObservationClass:
    """One public observation class plus private search-only continuations."""

    observation: TransformerObservation
    source_world_indices: tuple[int, ...]
    next_states: tuple[Any, ...] = field(repr=False, compare=False)
    tracker: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class InformationSetTransformerProjection:
    """Leak-free transformer views partitioning every source search world."""

    observation_classes: tuple[TransformerObservationClass, ...]
    certificate: str


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _canonical_action(value: str) -> str:
    action = str(value).lower()
    switch = SWITCH_ACTION.fullmatch(action)
    if switch is not None:
        return f"switch {_norm(switch.group(1))}"
    if action.endswith("-tera"):
        return f"{_norm(action[:-5])}-tera"
    return _norm(action)


def _canonical_value(value: Any) -> Any:
    """Normalize binding enum casing without discarding mechanical content."""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, tuple):
        return tuple(_canonical_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_canonical_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_canonical_value(item) for item in value))
    return value


def _side(state: Any, name: str) -> Any:
    return state.side_one if name == "SideOne" else state.side_two


def _active(side: Any) -> Any:
    return side.pokemon[int(side.active_index)]


def _quiet_value(value: Any) -> bool:
    return _canonical_value(value) in (
        False,
        0,
        (0, 0),
        (0, "0"),
        (),
        "none",
        set(),
    )


def _quiet_side(side: Any) -> bool:
    if any(not _quiet_value(getattr(side, field)) for field in SIDE_FIELDS):
        return False
    if any(getattr(side.side_conditions, field) != 0 for field in SIDE_CONDITION_FIELDS):
        return False
    if any(getattr(side.volatile_status_durations, field) != 0 for field in VOLATILE_DURATION_FIELDS):
        return False
    active = _active(side)
    return not active.terastallized and tuple(active.types) == tuple(active.base_types)


def _require_quiet_state(state: Any) -> None:
    if (
        state.team_preview
        or _norm(state.weather) != "none"
        or _norm(state.terrain) != "none"
        or state.trick_room
        or not _quiet_side(state.side_one)
        or not _quiet_side(state.side_two)
    ):
        raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_PRESTATE")


def _require_boosted_double_switch_state(state: Any) -> int:
    if (
        state.team_preview
        or _norm(state.weather) != "none"
        or _norm(state.terrain) != "none"
        or state.trick_room
    ):
        raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_PRESTATE")
    boost_fields = {
        "attack_boost",
        "defense_boost",
        "special_attack_boost",
        "special_defense_boost",
        "speed_boost",
        "accuracy_boost",
        "evasion_boost",
    }
    for side in (state.side_one, state.side_two):
        if any(
            not _quiet_value(getattr(side, field))
            for field in SIDE_FIELDS
            if field not in boost_fields
        ) or any(
            getattr(side.side_conditions, field) != 0
            for field in SIDE_CONDITION_FIELDS
        ) or any(
            getattr(side.volatile_status_durations, field) != 0
            for field in VOLATILE_DURATION_FIELDS
        ):
            raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_PRESTATE")
        active = _active(side)
        if (
            _norm(active.status) != "none"
            or active.terastallized
            or tuple(active.types) != tuple(active.base_types)
        ):
            raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_PRESTATE")
    stage = int(state.side_one.special_attack_boost)
    if (
        stage not in {1, 2}
        or int(state.side_one.special_defense_boost) != stage
        or any(
            int(getattr(state.side_one, field)) != 0
            for field in boost_fields
            if field not in {"special_attack_boost", "special_defense_boost"}
        )
        or any(int(getattr(state.side_two, field)) != 0 for field in boost_fields)
    ):
        raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_PRESTATE")
    return stage


def _classify_actions(state: Any, side_one_action: str, side_two_action: str) -> dict[str, str]:
    actions = {"SideOne": side_one_action, "SideTwo": side_two_action}
    switches = {
        side: match.group(1)
        for side, action in actions.items()
        if (match := SWITCH_ACTION.fullmatch(action)) is not None
    }
    alive = {side: _active(_side(state, side)).hp > 0 for side in actions}
    if all(alive.values()):
        if len(switches) != 2:
            raise PublicEventProjectionError("UNSUPPORTED_ACTION_PAIR")
    else:
        for side, is_alive in alive.items():
            action = actions[side]
            if is_alive and action != "No Move":
                raise PublicEventProjectionError("UNSUPPORTED_ACTION_PAIR")
            if not is_alive and side not in switches:
                raise PublicEventProjectionError("UNSUPPORTED_ACTION_PAIR")
    return switches


def _target_indices(state: Any, switches: Mapping[str, str]) -> dict[str, int]:
    result = {}
    for side_name, species in switches.items():
        side = _side(state, side_name)
        matches = [
            index
            for index, pokemon in enumerate(side.pokemon)
            if pokemon.hp > 0 and _norm(pokemon.id) == species
        ]
        if len(matches) != 1 or matches[0] == int(side.active_index):
            raise PublicEventProjectionError("SWITCH_TARGET_MISMATCH")
        if _norm(side.pokemon[matches[0]].ability) == "illusion":
            raise PublicEventProjectionError("ILLUSION_UNSUPPORTED")
        result[side_name] = matches[0]
    return result


def _parse_switch_instructions(
    instructions: Sequence[Any],
    state: Any,
    target_indices: Mapping[str, int],
) -> None:
    units: list[tuple[str, int, int]] = []
    cursor = 0
    while cursor < len(instructions):
        switch = SWITCH_INSTRUCTION.fullmatch(repr(instructions[cursor]))
        if switch is None:
            raise PublicEventProjectionError("UNSUPPORTED_INSTRUCTION")
        side_name, previous, target = switch.group(1), int(switch.group(2)), int(switch.group(3))
        units.append((side_name, previous, target))
        cursor += 1
        if cursor < len(instructions):
            bookkeeping = LAST_MOVE_INSTRUCTION.fullmatch(repr(instructions[cursor]))
            if bookkeeping is not None:
                if bookkeeping.group(1) != side_name or int(bookkeeping.group(2)) != target:
                    raise PublicEventProjectionError("INSTRUCTION_GRAMMAR_MISMATCH")
                cursor += 1
    if len(units) != len(target_indices) or {unit[0] for unit in units} != set(target_indices):
        raise PublicEventProjectionError("INSTRUCTION_GRAMMAR_MISMATCH")
    for side_name, previous, target in units:
        if previous != int(_side(state, side_name).active_index) or target != target_indices[side_name]:
            raise PublicEventProjectionError("INSTRUCTION_GRAMMAR_MISMATCH")


def _parse_boosted_switch_instructions(
    instructions: Sequence[Any],
    state: Any,
    target_indices: Mapping[str, int],
    stage: int,
) -> None:
    cursor = 0

    def consume_switch(side_name: str) -> None:
        nonlocal cursor
        if cursor >= len(instructions):
            raise PublicEventProjectionError("INSTRUCTION_GRAMMAR_MISMATCH")
        switch = SWITCH_INSTRUCTION.fullmatch(repr(instructions[cursor]))
        if (
            switch is None
            or switch.group(1) != side_name
            or int(switch.group(2)) != int(_side(state, side_name).active_index)
            or int(switch.group(3)) != target_indices[side_name]
        ):
            raise PublicEventProjectionError("INSTRUCTION_GRAMMAR_MISMATCH")
        cursor += 1
        if cursor < len(instructions):
            bookkeeping = LAST_MOVE_INSTRUCTION.fullmatch(repr(instructions[cursor]))
            if bookkeeping is not None:
                if (
                    bookkeeping.group(1) != side_name
                    or int(bookkeeping.group(2)) != target_indices[side_name]
                ):
                    raise PublicEventProjectionError("INSTRUCTION_GRAMMAR_MISMATCH")
                cursor += 1

    consume_switch("SideTwo")
    for stat in ("SpecialAttack", "SpecialDefense"):
        if cursor >= len(instructions):
            raise PublicEventProjectionError("INSTRUCTION_GRAMMAR_MISMATCH")
        boost = BOOST_RESET_INSTRUCTION.fullmatch(repr(instructions[cursor]))
        if boost is None or boost.group(1) != stat or int(boost.group(2)) != -stage:
            raise PublicEventProjectionError("INSTRUCTION_GRAMMAR_MISMATCH")
        cursor += 1
    consume_switch("SideOne")
    if cursor != len(instructions):
        raise PublicEventProjectionError("INSTRUCTION_GRAMMAR_MISMATCH")


def _displayed_hp_fraction(pokemon: Any) -> float:
    if pokemon.maxhp <= 0:
        raise PublicEventProjectionError("OPPONENT_REVEAL_MISMATCH")
    displayed = math.ceil(100.0 * pokemon.hp / pokemon.maxhp)
    if displayed == 100 and pokemon.hp < pokemon.maxhp:
        displayed = 99
    return displayed / 100.0


def _require_boosted_switch_targets(
    state: Any,
    switches: Mapping[str, str],
    targets: Mapping[str, int],
    public_opponent: Mapping[str, Mapping[str, Any]],
    observer_side: str = "SideOne",
) -> None:
    for side_name, target_index in targets.items():
        pokemon = _side(state, side_name).pokemon[target_index]
        if (
            pokemon.terastallized
            or tuple(pokemon.types) != tuple(pokemon.base_types)
            or _norm(pokemon.status) != "none"
        ):
            raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_PRESTATE")
        if side_name != observer_side:
            known = public_opponent.get(switches[side_name])
            if (
                known is None
                and pokemon.hp != pokemon.maxhp
                or known is not None
                and (
                    int(known.get("level", -1)) != int(pokemon.level)
                    or _norm(str(known.get("status", ""))) != "nostatus"
                    or not math.isclose(
                        float(known.get("hp_fraction", 0.0)),
                        _displayed_hp_fraction(pokemon),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
            ):
                raise PublicEventProjectionError("OPPONENT_REVEAL_MISMATCH")


def _pokemon_snapshot(pokemon: Any) -> tuple[Any, ...]:
    return tuple(_canonical_value(getattr(pokemon, field)) for field in POKEMON_FIELDS) + (
        tuple((_canonical_value(move.id), move.disabled, move.pp) for move in pokemon.moves),
    )


def _side_snapshot(
    side: Any, overrides: Mapping[str, Any] | None = None
) -> tuple[Any, ...]:
    overrides = overrides or {}
    return (
        tuple(
            _canonical_value(overrides.get(field, getattr(side, field)))
            for field in SIDE_FIELDS
        ),
        tuple(getattr(side.side_conditions, field) for field in SIDE_CONDITION_FIELDS),
        tuple(getattr(side.volatile_status_durations, field) for field in VOLATILE_DURATION_FIELDS),
        tuple(_pokemon_snapshot(pokemon) for pokemon in side.pokemon),
    )


def _state_snapshot(
    state: Any, side_one_overrides: Mapping[str, Any] | None = None
) -> tuple[Any, ...]:
    globals_ = tuple(_canonical_value(getattr(state, field, None)) for field in STATE_FIELDS)
    return (
        globals_,
        _side_snapshot(state.side_one, side_one_overrides),
        _side_snapshot(state.side_two),
    )


def _without_public_reveal_metadata(instructions: Sequence[Any]) -> tuple[Any, ...]:
    """Keep the pinned mechanics grammar independent of search-only reveal deltas."""
    return tuple(
        instruction
        for instruction in instructions
        if not repr(instruction).startswith("PublicReveal")
    )


def _events_for_world(
    state: Any,
    debug_result: Any,
    switches: Mapping[str, str],
    target_indices: Mapping[str, int],
    public_opponent: Mapping[str, Mapping[str, Any]],
    observer_side: str,
    boost_stage: int | None = None,
) -> tuple[PublicSwitchEvent, ...]:
    mechanics_instructions = _without_public_reveal_metadata(
        debug_result.selected_instructions.instruction_list
    )
    if boost_stage is None:
        _parse_switch_instructions(mechanics_instructions, state, target_indices)
        expected_snapshot = _state_snapshot(state)
    else:
        _parse_boosted_switch_instructions(
            mechanics_instructions,
            state,
            target_indices,
            boost_stage,
        )
        expected_snapshot = _state_snapshot(
            state,
            {
                "special_attack_boost": 0,
                "special_defense_boost": 0,
            },
        )
    if expected_snapshot != _state_snapshot(debug_result.state):
        raise PublicEventProjectionError("POSTSTATE_DELTA_MISMATCH")
    events = []
    for side_name in (observer_side, "SideTwo" if observer_side == "SideOne" else "SideOne"):
        if side_name not in switches:
            continue
        pokemon = _side(debug_result.state, side_name).pokemon[target_indices[side_name]]
        if _norm(pokemon.id) != switches[side_name]:
            raise PublicEventProjectionError("SWITCH_TARGET_MISMATCH")
        actor = "self" if side_name == observer_side else "opponent"
        if actor == "self":
            hp_fraction = float(pokemon.hp) / float(pokemon.maxhp)
            status = "nostatus" if _norm(pokemon.status) == "none" else _norm(pokemon.status)
            previously_revealed = True
        else:
            known = public_opponent.get(switches[side_name])
            previously_revealed = known is not None
            if known is None:
                if pokemon.hp != pokemon.maxhp or _norm(pokemon.status) != "none":
                    raise PublicEventProjectionError("OPPONENT_REVEAL_MISMATCH")
                hp_fraction, status = 1.0, "nostatus"
            else:
                hp_fraction = float(known["hp_fraction"])
                status = str(known["status"])
                if int(known["level"]) != int(pokemon.level):
                    raise PublicEventProjectionError("OPPONENT_REVEAL_MISMATCH")
        level = int(pokemon.level) if actor == "self" or known is None else int(known["level"])
        events.append(
            PublicSwitchEvent(
                actor=actor,
                species=switches[side_name],
                level=level,
                hp_fraction=hp_fraction,
                status=status,
                previously_revealed=previously_revealed,
            )
        )
    return tuple(events)


def project_information_set_switch(
    engine: Any,
    states: Sequence[Any],
    side_one_action: str,
    side_two_action: str,
    u: float,
    *,
    public_opponent: Mapping[str, Mapping[str, Any]] | None = None,
    observer_side: str = "SideOne",
) -> InformationSetSwitchProjection:
    """Project one certified switch event atomically across every hidden world."""
    if not states or observer_side not in {"SideOne", "SideTwo"}:
        raise PublicEventProjectionError("UNSUPPORTED_INFORMATION_SET")
    try:
        contract = engine.transition_debug_contract()
    except Exception:
        raise PublicEventProjectionError("UNPINNED_ENGINE_BUILD") from None
    if contract != ENGINE_DEBUG_CONTRACT:
        raise PublicEventProjectionError("UNPINNED_ENGINE_BUILD")
    public_opponent = public_opponent or {}
    projected = []
    next_states = []
    legal_signatures: list[tuple[str, ...] | None] = []
    certificates: set[str] = set()
    cleared_boosts: set[tuple[tuple[str, int], ...]] = set()
    try:
        for state in states:
            boost_stage = None
            try:
                _require_quiet_state(state)
                certificate = CERTIFICATE
                cleared = ()
            except PublicEventProjectionError:
                boost_stage = _require_boosted_double_switch_state(state)
                certificate = BOOSTED_DOUBLE_SWITCH_CERTIFICATE
                cleared = (
                    (
                        ("specialattack", boost_stage),
                        ("specialdefense", boost_stage),
                    )
                    if observer_side == "SideOne"
                    else ()
                )
            switches = _classify_actions(state, side_one_action, side_two_action)
            targets = _target_indices(state, switches)
            if boost_stage is not None:
                _require_boosted_switch_targets(
                    state, switches, targets, public_opponent, observer_side
                )
            debug_result = engine.step_with_uniform_debug(
                state, side_one_action, side_two_action, u
            )
            projected.append(
                _events_for_world(
                    state,
                    debug_result,
                    switches,
                    targets,
                    public_opponent,
                    observer_side,
                    boost_stage,
                )
            )
            next_states.append(debug_result.state)
            options = engine.root_options(state=debug_result.state)[
                0 if observer_side == "SideOne" else 1
            ]
            legal_actions = tuple(
                sorted(_canonical_action(action) for action in options)
            )
            if not legal_actions or len(set(legal_actions)) != len(legal_actions):
                raise PublicEventProjectionError("UNSUPPORTED_ENGINE_DELTA")
            legal_signatures.append(legal_actions)
            certificates.add(certificate)
            cleared_boosts.add(cleared)
    except Exception as exc:
        if isinstance(exc, PublicEventProjectionError):
            raise PublicEventProjectionError("UNSUPPORTED_INFORMATION_SET") from None
        raise PublicEventProjectionError("UNSUPPORTED_INFORMATION_SET") from None
    signature = tuple(event.public_dict() for event in projected[0])
    if any(tuple(event.public_dict() for event in events) != signature for events in projected[1:]):
        raise PublicEventProjectionError("NONINTERFERENCE_FAILURE")
    if (
        len(certificates) != 1
        or len(cleared_boosts) != 1
        or any(legal != legal_signatures[0] for legal in legal_signatures[1:])
    ):
        raise PublicEventProjectionError("NONINTERFERENCE_FAILURE")
    return InformationSetSwitchProjection(
        projected[0],
        tuple(next_states),
        legal_actions=legal_signatures[0],
        cleared_self_boosts=next(iter(cleared_boosts)),
        certificate=next(iter(certificates)),
    )


def _is_double_switch(side_one_action: str, side_two_action: str) -> bool:
    return (
        SWITCH_ACTION.fullmatch(side_one_action) is not None
        and SWITCH_ACTION.fullmatch(side_two_action) is not None
    )


def _is_switch_only_pair(side_one_action: str, side_two_action: str) -> bool:
    """Recognize ordinary and forced replacement switch-only transitions."""
    actions = (side_one_action, side_two_action)
    switch_count = sum(SWITCH_ACTION.fullmatch(action) is not None for action in actions)
    return switch_count == 2 or (
        switch_count == 1
        and any(_canonical_action(action) == "nomove" for action in actions)
    )


def _require_switch_transition_prestate(
    state: Any,
    side_one_action: str,
    side_two_action: str,
    public_opponent: Mapping[str, Mapping[str, Any]],
) -> None:
    switches = _classify_actions(state, side_one_action, side_two_action)
    targets = _target_indices(state, switches)
    try:
        _require_quiet_state(state)
    except PublicEventProjectionError:
        _require_boosted_double_switch_state(state)
        _require_boosted_switch_targets(
            state, switches, targets, public_opponent
        )


def private_transition_blockers(
    state: Any,
    side_one_action: str,
    side_two_action: str,
    *,
    public_opponent: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    if _is_switch_only_pair(side_one_action, side_two_action):
        try:
            _require_switch_transition_prestate(
                state,
                side_one_action,
                side_two_action,
                public_opponent or {},
            )
        except Exception:
            pass
        else:
            return ()
    return private_basic_move_blockers(
        state,
        side_one_action,
        side_two_action,
        public_opponent=public_opponent,
    )


def project_information_set_transition(
    engine: Any,
    states: Sequence[Any],
    side_one_action: str,
    side_two_action: str,
    u: float,
    *,
    observer_side: str = "SideOne",
    public_opponent: Mapping[str, Mapping[str, Any]] | None = None,
) -> InformationSetSwitchProjection | BasicMoveProjection:
    if _is_switch_only_pair(side_one_action, side_two_action):
        return project_information_set_switch(
            engine,
            states,
            side_one_action,
            side_two_action,
            u,
            observer_side=observer_side,
            public_opponent=public_opponent,
        )
    return project_information_set_basic_move(
        engine,
        states,
        side_one_action,
        side_two_action,
        u,
        observer_side=observer_side,
        public_opponent=public_opponent,
    )


def private_transition_diagnostic(
    engine: Any,
    state: Any,
    side_one_action: str,
    side_two_action: str,
    u: float,
    *,
    observer_side: str = "SideOne",
    public_opponent: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    if not _is_switch_only_pair(side_one_action, side_two_action):
        return private_basic_move_diagnostic(
            engine,
            state,
            side_one_action,
            side_two_action,
            u,
            observer_side=observer_side,
            public_opponent=public_opponent,
        )
    try:
        project_information_set_switch(
            engine,
            [state],
            side_one_action,
            side_two_action,
            u,
            observer_side=observer_side,
            public_opponent=public_opponent,
        )
    except PublicEventProjectionError:
        return "UNSUPPORTED_ENGINE_DELTA"
    except Exception:
        return "ENGINE_OR_BINDING_ERROR"
    return "ADMITTED"


def _boost_compatible_prestate(state: Any) -> bool:
    if (
        state.team_preview
        or _norm(state.weather) != "none"
        or _norm(state.terrain) != "none"
        or state.trick_room
    ):
        return False
    for side in (state.side_one, state.side_two):
        if any(
            not _quiet_value(getattr(side, field))
            for field in SIDE_FIELDS
            if field not in SIDE_BOOST_FIELDS
        ):
            return False
        if any(
            getattr(side.side_conditions, field) != 0
            for field in SIDE_CONDITION_FIELDS
        ) or any(
            getattr(side.volatile_status_durations, field) != 0
            for field in VOLATILE_DURATION_FIELDS
        ):
            return False
        active = _active(side)
        if (
            active.hp <= 0
            or _norm(active.status) != "none"
            or active.terastallized
            or tuple(active.types) != tuple(active.base_types)
        ):
            return False
    return True


def _declarative_boost_subset(state: Any, actions: Sequence[str]) -> bool:
    return all(_norm(action) in SELF_BOOST_RULES for action in actions) and (
        _boost_compatible_prestate(state)
    )


def _mixed_boost_switch_subset(
    state: Any,
    actions: Sequence[str],
    public_opponent: Mapping[str, Mapping[str, Any]],
    observer_side: str = "SideOne",
) -> bool:
    if len(actions) != 2 or _norm(actions[0]) not in SELF_BOOST_RULES:
        return False
    switch = SWITCH_ACTION.fullmatch(actions[1])
    if switch is None or not _boost_compatible_prestate(state):
        return False
    species = switch.group(1)
    known = public_opponent.get(species)
    if (
        species not in MIXED_BOOST_SWITCH_TARGETS
        or observer_side not in {"SideOne", "SideTwo"}
        or observer_side == "SideOne"
        and known is None
    ):
        return False
    matches = [
        (index, pokemon)
        for index, pokemon in enumerate(state.side_two.pokemon)
        if pokemon.hp > 0 and _norm(pokemon.id) == species
    ]
    if len(matches) != 1 or matches[0][0] == int(state.side_two.active_index):
        return False
    incoming = matches[0][1]
    engine_status = ENGINE_STATUS.get(_norm(incoming.status), _norm(incoming.status))
    return (
        _norm(incoming.ability) != "illusion"
        and incoming.maxhp > 0
        and not incoming.terastallized
        and tuple(incoming.types) == tuple(incoming.base_types)
        and (
            observer_side == "SideTwo"
            or (
                int(known.get("level", -1)) == int(incoming.level)
                and _norm(str(known.get("status", ""))) == engine_status
                and math.isclose(
                    float(known.get("hp_fraction", 0.0)),
                    float(incoming.hp) / incoming.maxhp,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
        )
    )


def _silent_mechanics_subset(state: Any, actions: Sequence[str]) -> bool:
    return (
        tuple(_norm(action) for action in actions) == ("calmmind", "bulkup")
        and _declarative_boost_subset(state, actions)
        and _norm(_active(state.side_one).item) == "leftovers"
        and _norm(_active(state.side_one).ability) == "protosynthesis"
        and _norm(_active(state.side_two).item) == "leftovers"
        and _norm(_active(state.side_two).ability) == "sapsipper"
    )


def _require_basic_move_state(
    state: Any,
    actions: Sequence[str],
    public_opponent: Mapping[str, Mapping[str, Any]],
    observer_side: str = "SideOne",
) -> str:
    if _silent_mechanics_subset(state, actions):
        return SILENT_MECHANICS_CERTIFICATE
    if _declarative_boost_subset(state, actions):
        return DECLARATIVE_BOOST_CERTIFICATE
    if _mixed_boost_switch_subset(
        state, actions, public_opponent, observer_side
    ):
        return MIXED_BOOST_SWITCH_CERTIFICATE
    if (
        len(actions) == 2
        and _norm(actions[0]) in SELF_BOOST_RULES
        and SWITCH_ACTION.fullmatch(actions[1]) is not None
    ):
        # This pair is only certified through the mixed boost/switch subset
        # above.  Do not let a public-registry mismatch fall through to the
        # generic single-switch semantic trace.
        raise PublicEventProjectionError("UNSUPPORTED_ACTION_PAIR")
    if len(actions) != 2 or any(action.endswith("-tera") for action in actions):
        raise PublicEventProjectionError("UNSUPPORTED_ACTION_PAIR")
    switches = sum(SWITCH_ACTION.fullmatch(action) is not None for action in actions)
    if switches > 1 or any(
        SWITCH_ACTION.fullmatch(action) is None
        and _canonical_action(action) in {"", "nomove"}
        for action in actions
    ):
        raise PublicEventProjectionError("UNSUPPORTED_ACTION_PAIR")
    if any(_active(side).hp <= 0 for side in (state.side_one, state.side_two)):
        raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_PRESTATE")
    return SEMANTIC_TRACE_CERTIFICATE


def private_basic_move_blockers(
    state: Any,
    side_one_action: str,
    side_two_action: str,
    *,
    public_opponent: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    """Return fixed aggregate-only blocker categories for private diagnostics."""
    blockers: set[str] = set()
    for side in (state.side_one, state.side_two):
        if _active(side).hp <= 0:
            blockers.add("ACTIVE_FAINTED")
    for action in (side_one_action, side_two_action):
        if action.endswith("-tera"):
            blockers.add("ACTION_TERA")
        elif SWITCH_ACTION.fullmatch(action) is None and _canonical_action(action) in {
            "",
            "nomove",
        }:
            blockers.add("ACTION_UNLISTED")
    return tuple(code for code in BASIC_MOVE_PRIVATE_BLOCKER_CODES if code in blockers)


def _actor(side: str, observer_side: str) -> str:
    expected = "side_one" if observer_side == "SideOne" else "side_two"
    return "self" if side == expected else "opponent"


def _public_hp_fraction(pokemon: Any, actor: str) -> float:
    if pokemon.maxhp <= 0:
        raise PublicEventProjectionError("UNSUPPORTED_ENGINE_DELTA")
    if actor == "self":
        return float(pokemon.hp) / float(pokemon.maxhp)
    if pokemon.hp <= 0:
        return 0.0
    displayed = math.ceil(float(pokemon.hp) * 100.0 / float(pokemon.maxhp))
    if displayed == 100 and pokemon.hp < pokemon.maxhp:
        displayed = 99
    return displayed / 100.0


def _own_delta(state: Any, observer_side: str) -> OwnPrivateDelta:
    side = state.side_one if observer_side == "SideOne" else state.side_two
    pokemon = _active(side)
    return OwnPrivateDelta(
        active_species=_norm(pokemon.id),
        hp=int(pokemon.hp),
        max_hp=int(pokemon.maxhp),
        status=ENGINE_STATUS.get(_norm(pokemon.status), _norm(pokemon.status)),
        moves=tuple(
            (_norm(move.id), int(move.pp), bool(move.disabled)) for move in pokemon.moves
        ),
    )


def _basic_public_events(
    result: Any,
    observer_side: str,
    public_opponent: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, ...]:
    if result.unaccounted_instruction_kinds:
        raise PublicEventProjectionError("UNACCOUNTED_INSTRUCTION")
    events = []
    hp_sides: set[str] = set()
    selected_actions = []
    for event in result.events:
        actor = _actor(event.side, observer_side) if event.side is not None else None
        if event.kind == "action_executed":
            move_id = _norm(event.move_id)
            if not move_id:
                raise PublicEventProjectionError("UNSUPPORTED_EXECUTED_ACTION")
            selected_actions.append((actor, move_id))
            events.append(PublicMoveEvent(actor=actor, move_id=move_id))
        elif event.kind in {"damage", "heal"}:
            if event.side in hp_sides:
                raise PublicEventProjectionError("UNSUPPORTED_HP_SEQUENCE")
            hp_sides.add(event.side)
            side = result.state.side_one if event.side == "side_one" else result.state.side_two
            pokemon = side.pokemon[int(event.pokemon_index)]
            events.append(
                PublicHPEvent(
                    actor=actor,
                    hp_fraction=_public_hp_fraction(pokemon, actor),
                    fainted=pokemon.hp <= 0,
                )
            )
        elif event.kind == "status_changed":
            status = ENGINE_STATUS.get(_norm(event.detail))
            if status is None:
                raise PublicEventProjectionError("UNSUPPORTED_STATUS")
            events.append(PublicStatusEvent(actor=actor, status=status))
        elif event.kind == "boost_changed":
            events.append(
                PublicBoostEvent(
                    actor=actor,
                    stat=str(event.detail),
                    amount=int(event.amount),
                )
            )
        elif event.kind == "item_activated":
            item_id = _norm(event.detail)
            if not item_id:
                raise PublicEventProjectionError("UNSUPPORTED_ITEM_ACTIVATION")
            events.append(
                PublicItemEvent(
                    actor=actor,
                    item_id=item_id,
                    active_after=item_id,
                )
            )
        elif event.kind == "switch":
            side = (
                result.state.side_one
                if event.side == "side_one"
                else result.state.side_two
            )
            pokemon = side.pokemon[int(event.pokemon_index)]
            species = _norm(pokemon.id)
            known = public_opponent.get(species) if actor == "opponent" else None
            events.append(
                PublicSwitchEvent(
                    actor=actor,
                    species=species,
                    level=(int(known["level"]) if known is not None else int(pokemon.level)),
                    hp_fraction=(
                        float(known["hp_fraction"])
                        if known is not None
                        else float(pokemon.hp) / float(pokemon.maxhp)
                    ),
                    status=(
                        str(known["status"])
                        if known is not None
                        else ENGINE_STATUS.get(
                            _norm(pokemon.status), _norm(pokemon.status)
                        )
                    ),
                    previously_revealed=known is not None or actor == "self",
                )
            )
        else:
            raise PublicEventProjectionError("UNSUPPORTED_SEMANTIC_EVENT")
    return tuple(events)


def _project_basic_move_world(
    engine: Any,
    state: Any,
    side_one_action: str,
    side_two_action: str,
    u: float,
    observer_side: str,
    public_opponent: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[Any, ...], OwnPrivateDelta, tuple[str, ...], Any, str]:
    certificate = _require_basic_move_state(
        state,
        (side_one_action, side_two_action),
        public_opponent,
        observer_side,
    )
    result = engine.step_with_uniform_r1_semantic(
        state, side_one_action, side_two_action, u
    )
    events = _basic_public_events(result, observer_side, public_opponent)
    if certificate in {
        SILENT_MECHANICS_CERTIFICATE,
        DECLARATIVE_BOOST_CERTIFICATE,
    }:
        expected_moves = Counter(
            {
                (_actor("side_one", observer_side), _norm(side_one_action)): 1,
                (_actor("side_two", observer_side), _norm(side_two_action)): 1,
            }
        )
        expected_boosts = Counter(
            (actor, stat, amount)
            for actor, action in (
                (_actor("side_one", observer_side), side_one_action),
                (_actor("side_two", observer_side), side_two_action),
            )
            for stat, amount in SELF_BOOST_RULES[_norm(action)]
        )
        base_event_count = 2 + sum(expected_boosts.values())
        base_events = events[:base_event_count]
        actual_moves = Counter(
            (event.actor, event.move_id)
            for event in base_events
            if isinstance(event, PublicMoveEvent)
        )
        actual_boosts = Counter(
            (event.actor, _norm(event.stat), event.amount)
            for event in base_events
            if isinstance(event, PublicBoostEvent)
        )
        if (
            len(base_events) != base_event_count
            or actual_moves != expected_moves
            or actual_boosts != expected_boosts
        ):
            raise PublicEventProjectionError("UNSUPPORTED_MECHANIC_ACTIVATION")
        extras = events[base_event_count:]
        if extras:
            if len(extras) not in {2, 4}:
                raise PublicEventProjectionError("UNSUPPORTED_MECHANIC_ACTIVATION")
            actors = set()
            for offset in range(0, len(extras), 2):
                item_event, hp_event = extras[offset : offset + 2]
                if (
                    not isinstance(item_event, PublicItemEvent)
                    or item_event.item_id != "leftovers"
                    or item_event.active_after != "leftovers"
                    or not isinstance(hp_event, PublicHPEvent)
                    or hp_event.actor != item_event.actor
                    or item_event.actor in actors
                ):
                    raise PublicEventProjectionError(
                        "UNSUPPORTED_MECHANIC_ACTIVATION"
                    )
                actors.add(item_event.actor)
            certificate = LEFTOVERS_ACTIVATION_CERTIFICATE
    elif certificate == MIXED_BOOST_SWITCH_CERTIFICATE:
        expected_species = SWITCH_ACTION.fullmatch(side_two_action).group(1)
        expected_moves = Counter(
            {(_actor("side_one", observer_side), _norm(side_one_action)): 1}
        )
        expected_boosts = Counter(
            (
                _actor("side_one", observer_side),
                stat,
                amount,
            )
            for stat, amount in SELF_BOOST_RULES[_norm(side_one_action)]
        )
        base_event_count = 2 + sum(expected_boosts.values())
        base_events = events[:base_event_count]
        switches = [
            event for event in base_events if isinstance(event, PublicSwitchEvent)
        ]
        actual_moves = Counter(
            (event.actor, event.move_id)
            for event in base_events
            if isinstance(event, PublicMoveEvent)
        )
        actual_boosts = Counter(
            (event.actor, _norm(event.stat), event.amount)
            for event in base_events
            if isinstance(event, PublicBoostEvent)
        )
        if (
            len(base_events) != base_event_count
            or len(switches) != 1
            or switches[0].actor != _actor("side_two", observer_side)
            or switches[0].species != expected_species
            or actual_moves != expected_moves
            or actual_boosts != expected_boosts
        ):
            raise PublicEventProjectionError("UNSUPPORTED_MECHANIC_ACTIVATION")
        extras = events[base_event_count:]
        if extras:
            if len(extras) != 2:
                raise PublicEventProjectionError("UNSUPPORTED_MECHANIC_ACTIVATION")
            item_event, hp_event = extras
            if (
                not isinstance(item_event, PublicItemEvent)
                or item_event.actor != _actor("side_one", observer_side)
                or item_event.item_id != "leftovers"
                or item_event.active_after != "leftovers"
                or not isinstance(hp_event, PublicHPEvent)
                or hp_event.actor != _actor("side_one", observer_side)
            ):
                raise PublicEventProjectionError("UNSUPPORTED_MECHANIC_ACTIVATION")
            certificate = LEFTOVERS_ACTIVATION_CERTIFICATE
    side_one_options, side_two_options = engine.root_options(state=result.state)
    observer_options = side_one_options if observer_side == "SideOne" else side_two_options
    legal_actions = tuple(sorted(_canonical_action(action) for action in observer_options))
    if not legal_actions or len(set(legal_actions)) != len(legal_actions):
        raise PublicEventProjectionError("UNSUPPORTED_ENGINE_DELTA")
    return (
        events,
        _own_delta(result.state, observer_side),
        legal_actions,
        result.state,
        certificate,
    )


def private_basic_move_diagnostic(
    engine: Any,
    state: Any,
    side_one_action: str,
    side_two_action: str,
    u: float,
    *,
    observer_side: str = "SideOne",
    public_opponent: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Return one fixed private diagnostic code without exception details."""
    if observer_side not in {"SideOne", "SideTwo"}:
        return "ENGINE_OR_BINDING_ERROR"
    try:
        _project_basic_move_world(
            engine,
            state,
            side_one_action,
            side_two_action,
            u,
            observer_side,
            public_opponent or {},
        )
    except PublicEventProjectionError as exc:
        if exc.code in BASIC_MOVE_PRIVATE_OUTCOME_CODES:
            return exc.code
        return "ENGINE_OR_BINDING_ERROR"
    except Exception:
        return "ENGINE_OR_BINDING_ERROR"
    return "ADMITTED"


def project_information_set_basic_move(
    engine: Any,
    states: Sequence[Any],
    side_one_action: str,
    side_two_action: str,
    u: float,
    *,
    observer_side: str = "SideOne",
    public_opponent: Mapping[str, Mapping[str, Any]] | None = None,
) -> BasicMoveProjection:
    """Partition worlds by authorized public events and own-private deltas."""
    if not states or observer_side not in {"SideOne", "SideTwo"}:
        raise PublicEventProjectionError("UNSUPPORTED_INFORMATION_SET")
    try:
        contract = engine.r1_semantic_contract()
    except Exception:
        raise PublicEventProjectionError("UNPINNED_ENGINE_BUILD") from None
    if contract != R1_SEMANTIC_CONTRACT:
        raise PublicEventProjectionError("UNPINNED_ENGINE_BUILD")
    public_opponent = public_opponent or {}

    grouped: dict[
        tuple[Any, ...],
        tuple[
            tuple[Any, ...],
            OwnPrivateDelta,
            tuple[str, ...],
            list[Any],
            list[int],
        ],
    ] = {}
    certificates: set[str] = set()
    try:
        for source_world_index, state in enumerate(states):
            events, own_delta, legal_actions, next_state, certificate = (
                _project_basic_move_world(
                    engine,
                    state,
                    side_one_action,
                    side_two_action,
                    u,
                    observer_side,
                    public_opponent,
                )
            )
            certificates.add(certificate)
            signature = events + (own_delta, legal_actions)
            if signature not in grouped:
                grouped[signature] = (
                    events,
                    own_delta,
                    legal_actions,
                    [],
                    [],
                )
            grouped[signature][3].append(next_state)
            grouped[signature][4].append(source_world_index)
    except Exception as exc:
        if isinstance(exc, PublicEventProjectionError):
            raise PublicEventProjectionError("UNSUPPORTED_INFORMATION_SET") from None
        raise PublicEventProjectionError("UNSUPPORTED_INFORMATION_SET") from None
    if len(certificates) != 1:
        raise PublicEventProjectionError("UNSUPPORTED_INFORMATION_SET")
    certificate = next(iter(certificates))
    classes = tuple(
        BasicMoveObservationClass(
            events,
            own_delta,
            legal_actions,
            tuple(next_states),
            tuple(source_world_indices),
            certificate=certificate,
        )
        for (
            events,
            own_delta,
            legal_actions,
            next_states,
            source_world_indices,
        ) in grouped.values()
    )
    return BasicMoveProjection(classes, certificate=certificate)


def _history_target(observation_space: Any) -> Any:
    current = observation_space
    while hasattr(current, "base_obs_space"):
        current = current.base_obs_space
    return current


def _seed_history(observation_space: Any, history: Mapping[str, Any]) -> None:
    target = _history_target(observation_space)
    target.any_opponent_asleep = bool(history["any_opponent_asleep"])
    target.any_opponent_frozen = bool(history["any_opponent_frozen"])
    target.revealed_opponents = set(history["revealed_opponents"])


@dataclass
class _TeamRecord:
    slot: str
    active: bool
    pokemon: Any
    previous_move: Any
    revealed_item: str | None = None


class R1SwitchTracker:
    """Player-information tracker for the certified pure-switch subset."""

    def __init__(self, state: Any, player_team: list[_TeamRecord], opponent_team: list[_TeamRecord], observation_space: Any):
        self.state = state
        self.player_team = player_team
        self.opponent_team = opponent_team
        self.observation_space = observation_space

    def fork(self) -> "R1SwitchTracker":
        return copy.deepcopy(self)

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any], observation_space_template: Any):
        import numpy as np

        from metamon.interface import UniversalMove, UniversalPokemon, UniversalState

        information = copy.deepcopy(snapshot["player_information_state"])
        state = UniversalState.from_dict(information["universal_state"])

        def records(rows: Sequence[Mapping[str, Any]]) -> list[_TeamRecord]:
            result = []
            for row in rows:
                pokemon = UniversalPokemon.from_dict(copy.deepcopy(row["pokemon"]))
                item = _norm(pokemon.item)
                result.append(
                    _TeamRecord(
                        slot=str(row["slot"]),
                        active=bool(row["active"]),
                        pokemon=pokemon,
                        previous_move=UniversalMove(
                            **copy.deepcopy(row["previous_move"])
                        ),
                        revealed_item=(
                            None if item in {"", "unknownitem"} else item
                        ),
                    )
                )
            return result

        root_space = copy.deepcopy(observation_space_template)
        root_space.reset()
        _seed_history(root_space, snapshot["player_observation_history"])
        root_obs = root_space.state_to_obs(state)
        if (
            root_obs["text_tokens"].tolist() != snapshot["text_tokens"]
            or not np.array_equal(
                np.nan_to_num(np.asarray(root_obs["numbers"], dtype=np.float32)),
                np.asarray(snapshot["numbers"], dtype=np.float32),
            )
        ):
            raise PublicEventProjectionError("ROOT_OBSERVATION_MISMATCH")

        continuation_space = copy.deepcopy(observation_space_template)
        continuation_space.reset()
        _seed_history(continuation_space, snapshot["continuation_observation_history"])
        return cls(
            state,
            records(information["player_team"]),
            records(information["opponent_public_team"]),
            continuation_space,
        )

    def _switch(self, event: PublicSwitchEvent) -> None:
        from metamon.backend.replay_parser.replay_state import Pokemon as ReplayPokemon
        from metamon.interface import UniversalMove, UniversalPokemon

        team = self.player_team if event.actor == "self" else self.opponent_team
        matches = [record for record in team if _norm(record.pokemon.name) == event.species]
        if len(matches) > 1:
            raise PublicEventProjectionError("AMBIGUOUS_PUBLIC_SLOT")
        if matches:
            incoming = matches[0]
        elif event.actor == "opponent" and not event.previously_revealed:
            replay_pokemon = ReplayPokemon(event.species, event.level, 9)
            replay_pokemon.current_hp = 100
            replay_pokemon.max_hp = 100
            incoming = _TeamRecord(
                slot=f"public-{len(team)}",
                active=False,
                pokemon=UniversalPokemon.from_ReplayPokemon(replay_pokemon),
                previous_move=UniversalMove.blank_move(),
            )
            team.append(incoming)
        else:
            raise PublicEventProjectionError("AMBIGUOUS_PUBLIC_SLOT")
        outgoing = [record for record in team if record.active]
        if len(outgoing) != 1:
            raise PublicEventProjectionError("AMBIGUOUS_PUBLIC_SLOT")
        outgoing[0].active = False
        outgoing[0].pokemon.effect = "noeffect"
        for field in (
            "atk_boost", "spa_boost", "def_boost", "spd_boost", "spe_boost",
            "accuracy_boost", "evasion_boost",
        ):
            setattr(outgoing[0].pokemon, field, 0)
        incoming.active = True
        incoming.pokemon.hp_pct = event.hp_fraction
        incoming.pokemon.status = event.status
        if event.actor == "self":
            self.state.player_active_pokemon = incoming.pokemon
            self.state.player_prev_move = incoming.previous_move
            self.state.available_switches = [
                record.pokemon
                for record in team
                if not record.active and record.pokemon.hp_pct > 0
            ]
        else:
            self.state.opponent_active_pokemon = incoming.pokemon
            self.state.opponent_prev_move = incoming.previous_move

    def apply_switch_projection(
        self, projection: InformationSetSwitchProjection
    ) -> dict[str, Any]:
        if projection.certificate not in {
            CERTIFICATE,
            BOOSTED_DOUBLE_SWITCH_CERTIFICATE,
        }:
            raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_EVENT")
        if projection.certificate == BOOSTED_DOUBLE_SWITCH_CERTIFICATE:
            expected = dict(projection.cleared_self_boosts)
            active = self._active_record(self.player_team).pokemon
            actual = {
                "attack": active.atk_boost,
                "specialattack": active.spa_boost,
                "defense": active.def_boost,
                "specialdefense": active.spd_boost,
                "speed": active.spe_boost,
                "accuracy": active.accuracy_boost,
                "evasion": active.evasion_boost,
            }
            if any(
                int(value) != int(expected.get(stat, 0))
                for stat, value in actual.items()
            ):
                raise PublicEventProjectionError("PUBLIC_PRESTATE_MISMATCH")
        return self.apply_switch_events(
            projection.events, projection.legal_actions
        )

    def apply_switch_events(
        self,
        events: Sequence[PublicSwitchEvent],
        certified_legal_actions: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        for event in events:
            if event.schema != EVENT_SCHEMA or event.kind != "switch":
                raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_EVENT")
            self._switch(event)
        self.state.forced_switch = False
        observation = self.observation_space.state_to_obs(self.state)
        illegal, name_table = self._next_mask_and_names(certified_legal_actions)
        return {
            "text_tokens": observation["text_tokens"].tolist(),
            "numbers": np.nan_to_num(
                np.asarray(observation["numbers"], dtype=np.float32)
            ).tolist(),
            "illegal_actions": illegal,
            "name_table": name_table,
        }

    def _active_record(self, team: Sequence[_TeamRecord]) -> _TeamRecord:
        active = [record for record in team if record.active]
        if len(active) != 1:
            raise PublicEventProjectionError("AMBIGUOUS_PUBLIC_SLOT")
        return active[0]

    def apply_basic_move_class(self, observation_class: BasicMoveObservationClass) -> dict[str, Any]:
        from metamon.backend.replay_parser.replay_state import Move as ReplayMove
        from metamon.interface import UniversalMove

        if observation_class.certificate not in {
            BASIC_MOVE_CERTIFICATE,
            SILENT_MECHANICS_CERTIFICATE,
            DECLARATIVE_BOOST_CERTIFICATE,
            MIXED_BOOST_SWITCH_CERTIFICATE,
            LEFTOVERS_ACTIVATION_CERTIFICATE,
            SEMANTIC_TRACE_CERTIFICATE,
        }:
            raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_EVENT")
        self_switches = [
            event
            for event in observation_class.events
            if isinstance(event, PublicSwitchEvent) and event.actor == "self"
        ]
        if len(self_switches) > 1:
            raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_EVENT")
        for event in self_switches:
            self._switch(event)
        own_record = self._active_record(self.player_team)
        if _norm(own_record.pokemon.name) != observation_class.own_delta.active_species:
            raise PublicEventProjectionError("OWN_PRIVATE_DELTA_MISMATCH")
        if observation_class.own_delta.max_hp <= 0:
            raise PublicEventProjectionError("OWN_PRIVATE_DELTA_MISMATCH")
        own_record.pokemon.hp_pct = (
            float(observation_class.own_delta.hp) / observation_class.own_delta.max_hp
        )
        own_record.pokemon.status = observation_class.own_delta.status
        opponent_record = self._active_record(self.opponent_team)
        self.state.opponent_active_pokemon = opponent_record.pokemon
        own_moves = {_norm(move.name): move for move in own_record.pokemon.moves}
        for move_id, pp, disabled in observation_class.own_delta.moves:
            if move_id not in own_moves or pp < 0:
                raise PublicEventProjectionError("OWN_PRIVATE_DELTA_MISMATCH")
            own_moves[move_id].current_pp = pp
        self.state.player_active_pokemon = own_record.pokemon

        boost_fields = {
            "attack": "atk_boost",
            "specialattack": "spa_boost",
            "defense": "def_boost",
            "specialdefense": "spd_boost",
            "speed": "spe_boost",
            "accuracy": "accuracy_boost",
            "evasion": "evasion_boost",
        }
        for event in observation_class.events:
            if isinstance(event, PublicSwitchEvent):
                if event.schema != EVENT_SCHEMA or event.kind != "switch":
                    raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_EVENT")
                if event.actor != "self":
                    self._switch(event)
            elif isinstance(event, PublicMoveEvent):
                if event.actor == "self":
                    if event.move_id not in own_moves:
                        raise PublicEventProjectionError("OWN_PRIVATE_DELTA_MISMATCH")
                    own_record.previous_move = own_moves[event.move_id]
                    self.state.player_prev_move = own_record.previous_move
                else:
                    opponent = self._active_record(self.opponent_team)
                    known_moves = {_norm(move.name): move for move in opponent.pokemon.moves}
                    if event.move_id not in known_moves:
                        known_moves[event.move_id] = UniversalMove.from_ReplayMove(
                            ReplayMove(event.move_id, gen=9)
                        )
                        opponent.pokemon.moves.append(known_moves[event.move_id])
                    opponent.previous_move = known_moves[event.move_id]
                    self.state.opponent_prev_move = opponent.previous_move
            elif isinstance(event, PublicHPEvent):
                team = self.player_team if event.actor == "self" else self.opponent_team
                record = self._active_record(team)
                was_fainted = record.pokemon.hp_pct <= 0
                record.pokemon.hp_pct = event.hp_fraction
                if event.fainted:
                    record.pokemon.status = "fnt"
                    if event.actor == "opponent" and not was_fainted:
                        self.state.opponents_remaining -= 1
                if event.actor == "self":
                    self.state.player_active_pokemon = record.pokemon
                else:
                    self.state.opponent_active_pokemon = record.pokemon
            elif isinstance(event, PublicStatusEvent):
                team = self.player_team if event.actor == "self" else self.opponent_team
                record = self._active_record(team)
                record.pokemon.status = event.status
            elif isinstance(event, PublicBoostEvent):
                field = boost_fields.get(_norm(event.stat))
                if field is None:
                    raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_EVENT")
                pokemon = (
                    self.state.player_active_pokemon
                    if event.actor == "self"
                    else self.state.opponent_active_pokemon
                )
                setattr(pokemon, field, getattr(pokemon, field) + event.amount)
            elif isinstance(event, PublicItemEvent):
                team = self.player_team if event.actor == "self" else self.opponent_team
                record = self._active_record(team)
                if event.item_id != "leftovers" or event.active_after != "leftovers":
                    raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_EVENT")
                record.revealed_item = event.item_id
                record.pokemon.item = event.active_after
            else:
                raise PublicEventProjectionError("UNSUPPORTED_PUBLIC_EVENT")

        self.state.forced_switch = own_record.pokemon.hp_pct <= 0
        self.state.battle_lost = self.state.forced_switch and not any(
            not record.active and record.pokemon.hp_pct > 0 for record in self.player_team
        )
        self.state.battle_won = self.state.opponents_remaining <= 0
        observation = self.observation_space.state_to_obs(self.state)
        terminal = bool(self.state.battle_won or self.state.battle_lost)
        automatic_action = None
        if terminal:
            illegal, name_table = [True] * 13, self._name_table()
        elif tuple(
            _canonical_action(action) for action in observation_class.legal_actions
        ) == ("nomove",):
            illegal, name_table = [True] * 13, self._name_table()
            automatic_action = "nomove"
        else:
            illegal, name_table = self._next_mask_and_names(
                observation_class.legal_actions
            )
        import numpy as np

        return {
            "text_tokens": observation["text_tokens"].tolist(),
            "numbers": np.nan_to_num(
                np.asarray(observation["numbers"], dtype=np.float32)
            ).tolist(),
            "illegal_actions": illegal,
            "name_table": name_table,
            "terminal": terminal,
            "automatic_action": automatic_action,
        }

    def public_opponent_registry(self) -> dict[str, dict[str, Any]]:
        return {
            _norm(record.pokemon.name): {
                "level": int(record.pokemon.lvl),
                "hp_fraction": float(record.pokemon.hp_pct),
                "status": str(record.pokemon.status),
                "item": str(record.pokemon.item),
                "revealed_item": record.revealed_item,
            }
            for record in self.opponent_team
        }

    def _next_mask_and_names(
        self, certified_legal_actions: Sequence[str] | None = None
    ) -> tuple[list[bool], dict[str, int]]:
        from metamon.backend.replay_parser.replay_state import Pokemon as ReplayPokemon
        from metamon.interface import consistent_move_order, consistent_pokemon_order

        opponent = self.state.opponent_active_pokemon
        known_ability = _norm(opponent.ability)
        possible_abilities = {
            _norm(ability)
            for ability in ReplayPokemon._lookup_pokedex_info(opponent.name, 9)
            .get("abilities", {})
            .values()
        }
        if certified_legal_actions is None and not self.state.forced_switch and (
            known_ability in TRAPPING_ABILITIES
            or (
                known_ability in {"", "unknownability"}
                and possible_abilities.intersection(TRAPPING_ABILITIES)
            )
        ):
            raise PublicEventProjectionError("NEXT_MASK_UNCERTIFIED")
        if certified_legal_actions is None and (
            not self.state.forced_switch
            and _norm(self.state.player_active_pokemon.item) == "assaultvest"
        ):
            raise PublicEventProjectionError("NEXT_MASK_UNCERTIFIED")
        moves = consistent_move_order(self.state.player_active_pokemon.moves)[:4]
        switches = consistent_pokemon_order(
            [
                record.pokemon
                for record in self.player_team
                if not record.active and record.pokemon.hp_pct > 0
            ]
        )[:5]
        self.state.available_switches = switches
        illegal = [True] * 13
        names = {}
        for index, move in enumerate(moves):
            action = _norm(move.name)
            names[action] = index
            names[f"{action}-tera"] = index + 9
            if not self.state.forced_switch and move.current_pp > 0:
                illegal[index] = False
                illegal[index + 9] = not self.state.can_tera
        for index, pokemon in enumerate(switches):
            action = f"switch {_norm(pokemon.name)}"
            names[action] = index + 4
            illegal[index + 4] = False
        if certified_legal_actions is not None:
            legal = tuple(_canonical_action(action) for action in certified_legal_actions)
            if not legal or len(set(legal)) != len(legal) or any(
                action not in names for action in legal
            ):
                raise PublicEventProjectionError("NEXT_MASK_UNCERTIFIED")
            illegal = [True] * 13
            for action in legal:
                illegal[names[action]] = False
        if all(illegal):
            raise PublicEventProjectionError("NEXT_MASK_UNCERTIFIED")
        return illegal, names

    def _name_table(self) -> dict[str, int]:
        from metamon.interface import consistent_move_order, consistent_pokemon_order

        names = {}
        for index, move in enumerate(
            consistent_move_order(self.state.player_active_pokemon.moves)[:4]
        ):
            action = _norm(move.name)
            names[action] = index
            names[f"{action}-tera"] = index + 9
        switches = consistent_pokemon_order(
            [
                record.pokemon
                for record in self.player_team
                if not record.active and record.pokemon.hp_pct > 0
            ]
        )[:5]
        for index, pokemon in enumerate(switches):
            names[f"switch {_norm(pokemon.name)}"] = index + 4
        return names


def _transformer_observation(payload: Mapping[str, Any]) -> TransformerObservation:
    """Validate and copy the narrow continuation ABI consumed by r1."""
    try:
        text_tokens = tuple(payload["text_tokens"])
        numbers = tuple(float(value) for value in payload["numbers"])
        illegal_actions = tuple(payload["illegal_actions"])
        name_table = tuple(sorted(dict(payload["name_table"]).items()))
        terminal = payload.get("terminal", False)
        automatic_action = payload.get("automatic_action")
    except (KeyError, TypeError, ValueError, OverflowError):
        raise PublicEventProjectionError("INVALID_TRANSFORMER_OBSERVATION") from None
    if (
        not text_tokens
        or not all(isinstance(token, int) and not isinstance(token, bool) for token in text_tokens)
        or not numbers
        or not all(math.isfinite(value) for value in numbers)
        or len(illegal_actions) != 13
        or not all(isinstance(flag, bool) for flag in illegal_actions)
        or not isinstance(terminal, bool)
        or automatic_action not in {None, "nomove"}
        or (all(illegal_actions) != (terminal or automatic_action == "nomove"))
        or (terminal and automatic_action is not None)
        or not all(
            isinstance(action, str)
            and action
            and isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < 13
            for action, index in name_table
        )
    ):
        raise PublicEventProjectionError("INVALID_TRANSFORMER_OBSERVATION")
    return TransformerObservation(
        text_tokens=text_tokens,
        numbers=numbers,
        illegal_actions=illegal_actions,
        name_table=name_table,
        terminal=terminal,
        automatic_action=automatic_action,
    )


def project_information_set_observations(
    engine: Any,
    states: Sequence[Any],
    tracker: R1SwitchTracker,
    side_one_action: str,
    side_two_action: str,
    u: float,
    *,
    observer_side: str = "SideOne",
    public_opponent: Mapping[str, Mapping[str, Any]] | None = None,
) -> InformationSetTransformerProjection:
    """Advance search worlds while exposing only player-information observations.

    Mechanical child states remain attached for the simulator, with repr disabled.
    Crucially, they are stripped before a projection is passed to the tracker, so
    neither the transformer adapter nor its observation-space code can inspect a
    sampled world's hidden fields.
    """
    projection = project_information_set_transition(
        engine,
        states,
        side_one_action,
        side_two_action,
        u,
        observer_side=observer_side,
        public_opponent=public_opponent,
    )
    classes: list[TransformerObservationClass] = []
    if isinstance(projection, InformationSetSwitchProjection):
        child_tracker = tracker.fork()
        sanitized = InformationSetSwitchProjection(
            events=projection.events,
            next_states=(),
            legal_actions=projection.legal_actions,
            cleared_self_boosts=projection.cleared_self_boosts,
            certificate=projection.certificate,
        )
        observation = _transformer_observation(
            child_tracker.apply_switch_projection(sanitized)
        )
        classes.append(
            TransformerObservationClass(
                observation=observation,
                source_world_indices=tuple(range(len(states))),
                next_states=projection.next_states,
                tracker=child_tracker,
            )
        )
    else:
        for observation_class in projection.observation_classes:
            child_tracker = tracker.fork()
            sanitized = BasicMoveObservationClass(
                events=observation_class.events,
                own_delta=observation_class.own_delta,
                legal_actions=observation_class.legal_actions,
                next_states=(),
                source_world_indices=observation_class.source_world_indices,
                certificate=observation_class.certificate,
            )
            observation = _transformer_observation(
                child_tracker.apply_basic_move_class(sanitized)
            )
            classes.append(
                TransformerObservationClass(
                    observation=observation,
                    source_world_indices=observation_class.source_world_indices,
                    next_states=observation_class.next_states,
                    tracker=child_tracker,
                )
            )

    lineage = [index for item in classes for index in item.source_world_indices]
    if (
        not classes
        or any(
            len(item.source_world_indices) != len(item.next_states)
            for item in classes
        )
        or sorted(lineage) != list(range(len(states)))
    ):
        raise PublicEventProjectionError("INVALID_INFORMATION_SET_LINEAGE")
    return InformationSetTransformerProjection(
        observation_classes=tuple(classes),
        certificate=projection.certificate,
    )
