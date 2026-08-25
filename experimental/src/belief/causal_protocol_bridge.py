"""Causal Showdown protocol facts for a leak-free search-state bridge.

This parser intentionally has no simulator dependency.  It never consumes the
transformer's reconstructed opponent team: opponent facts come only from
chronological public protocol events.  The own private request is accepted only
to authenticate the observer role and exact own action boundary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class CausalProtocolBridgeError(ValueError):
    pass


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


_SPECIES_ALIASES = {
    "morpekohangry": "morpeko",
    "eiscuenoice": "eiscue",
    "mimikyubusted": "mimikyu",
    "mimikyubustedtotem": "mimikyu",
    "palafinhero": "palafin",
    "wishiwashischool": "wishiwashi",
    "aegislashblade": "aegislash",
    "darmanitanzen": "darmanitan",
    "darmanitangalarzen": "darmanitangalar",
    "zygardecomplete": "zygarde",
}
_SAWSBUCK_FORMS = {
    "sawsbuckspring", "sawsbucksummer", "sawsbuckautumn", "sawsbuckwinter",
}
_MINIOR_FORMS = {
    "miniormeteor", "miniorred", "miniororange", "minioryellow",
    "miniorgreen", "miniorblue", "miniorindigo", "miniorviolet",
}


def canonical_species(value: Any) -> str:
    """Apply only the base/battle-form aliases frozen in Cycle 3."""
    species = norm(value)
    if species.startswith("alcremie"):
        return "alcremie"
    if species in _SAWSBUCK_FORMS:
        return "sawsbuck"
    if species in _MINIOR_FORMS:
        return "minior"
    return _SPECIES_ALIASES.get(species, species)


@dataclass
class MutableReveal:
    species: str
    level: int | None = None
    moves: set[str] = field(default_factory=set)
    ability: str | None = None
    current_item: str | None = None
    item_status_revealed: bool = False
    historically_revealed_items: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class OpponentReveal:
    species: str
    level: int | None
    moves: tuple[str, ...]
    ability: str | None
    current_item: str | None
    item_status_revealed: bool
    historically_revealed_items: tuple[str, ...]


@dataclass(frozen=True)
class CausalProtocolFacts:
    observer_role: str
    opponent_role: str
    opponent: tuple[OpponentReveal, ...]
    opponent_active_species: str
    public_event_lines: int
    request_lines_ignored_for_opponent: int


@dataclass(frozen=True)
class ReconciledPublicState:
    state: Any
    reveal_bits: int
    archival_repairs: tuple[str, ...]
    historical_item_sidecar: tuple[tuple[str, tuple[str, ...]], ...]


def _side(ident: str) -> str:
    match = re.match(r"^(p[12])", str(ident or ""))
    return match.group(1) if match else ""


def _details_species(details: str) -> str:
    return canonical_species(str(details or "").split(",", 1)[0])


def _details_level(details: str) -> int | None:
    match = re.search(r"(?:^|,\s*)L(\d+)(?:,|$)", str(details or ""))
    return int(match.group(1)) if match else None


def _request_from_prefix(prefix: Sequence[str]) -> list[Mapping[str, Any]]:
    requests: list[Mapping[str, Any]] = []
    for line in prefix:
        if not line.startswith("|request|"):
            continue
        try:
            value = json.loads(line.removeprefix("|request|"))
        except json.JSONDecodeError as exc:
            raise CausalProtocolBridgeError("invalid private request JSON") from exc
        if not isinstance(value, Mapping):
            raise CausalProtocolBridgeError("private request is not an object")
        requests.append(value)
    return requests


def _merge_reveal(target: MutableReveal, source: MutableReveal) -> None:
    if target.level is None:
        target.level = source.level
    target.moves.update(source.moves)
    if source.ability is not None:
        target.ability = source.ability
    if source.item_status_revealed:
        target.item_status_revealed = True
        target.current_item = source.current_item
    target.historically_revealed_items.update(source.historically_revealed_items)


def parse_causal_protocol(
    protocol_prefix: Sequence[str],
    *,
    player_role: str,
    private_request: Mapping[str, Any],
) -> CausalProtocolFacts:
    """Compile opponent facts without reading reconstructed or hidden fields."""
    if player_role not in {"p1", "p2"}:
        raise CausalProtocolBridgeError("invalid player role")
    if not isinstance(protocol_prefix, Sequence) or isinstance(protocol_prefix, (str, bytes)):
        raise CausalProtocolBridgeError("protocol prefix is not a line sequence")
    if any(not isinstance(line, str) for line in protocol_prefix):
        raise CausalProtocolBridgeError("protocol prefix contains a non-string line")
    side = private_request.get("side") if isinstance(private_request, Mapping) else None
    if not isinstance(side, Mapping) or side.get("id") != player_role:
        raise CausalProtocolBridgeError("private request role mismatch")
    requests = _request_from_prefix(protocol_prefix)
    if not requests:
        raise CausalProtocolBridgeError("protocol prefix lacks private request")
    if requests[-1] != private_request:
        raise CausalProtocolBridgeError("protocol/private request mismatch")

    opponent_role = "p2" if player_role == "p1" else "p1"
    active: dict[str, str] = {"p1": "", "p2": ""}
    activations: dict[tuple[str, str], int] = {}
    reveals: dict[str, MutableReveal] = {}
    public_lines = 0

    def ensure(species: str, level: int | None = None) -> MutableReveal:
        if not species:
            raise CausalProtocolBridgeError("public event lacks species identity")
        reveal = reveals.setdefault(species, MutableReveal(species=species))
        if level is not None:
            reveal.level = level
        return reveal

    def actor_species(ident: str) -> tuple[str, str]:
        actor_side = _side(ident)
        return actor_side, active.get(actor_side, "")

    for line in protocol_prefix:
        if line == "":
            # Foul Play preserves Showdown's empty framing separator.
            continue
        if not line.startswith("|"):
            raise CausalProtocolBridgeError("malformed protocol line")
        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag == "request":
            continue
        public_lines += 1

        if tag in {"switch", "drag", "replace", "detailschange"} and len(parts) >= 4:
            event_side = _side(parts[2])
            species = _details_species(parts[3])
            level = _details_level(parts[3])
            if not event_side or not species:
                raise CausalProtocolBridgeError("invalid public species event")
            old = active.get(event_side, "")
            active[event_side] = species
            if tag in {"switch", "drag"}:
                key = (event_side, species)
                activations[key] = activations.get(key, 0) + 1
            if event_side == opponent_role:
                target = ensure(species, level)
                # `replace` reveals Illusion.  Earlier moves/items/ability made
                # by the disguised active belong to the newly revealed species.
                if tag == "replace" and old and old != species and old in reveals:
                    if activations.get((event_side, old), 0) > 1:
                        raise CausalProtocolBridgeError(
                            "ambiguous repeated species before Illusion replace"
                        )
                    _merge_reveal(target, reveals.pop(old))
            continue

        if tag == "move" and len(parts) >= 4:
            event_side, species = actor_species(parts[2])
            move = norm(parts[3])
            if event_side == opponent_role:
                if not species or not move:
                    raise CausalProtocolBridgeError("opponent move lacks active identity")
                ensure(species).moves.add(move)

        elif tag == "-item" and len(parts) >= 4:
            event_side, species = actor_species(parts[2])
            item = norm(parts[3])
            if event_side == opponent_role:
                if not species or not item:
                    raise CausalProtocolBridgeError("opponent item lacks identity")
                reveal = ensure(species)
                reveal.current_item = item
                reveal.item_status_revealed = True
                reveal.historically_revealed_items.add(item)

        elif tag == "-enditem" and len(parts) >= 4:
            event_side, species = actor_species(parts[2])
            item = norm(parts[3])
            if event_side == opponent_role:
                if not species or not item:
                    raise CausalProtocolBridgeError("opponent end-item lacks identity")
                reveal = ensure(species)
                reveal.current_item = "none"
                reveal.item_status_revealed = True
                reveal.historically_revealed_items.add(item)

        elif tag == "-ability" and len(parts) >= 4:
            event_side, species = actor_species(parts[2])
            ability = norm(parts[3])
            if event_side == opponent_role:
                if not species or not ability:
                    raise CausalProtocolBridgeError("opponent ability lacks identity")
                ensure(species).ability = ability

        # A source such as Rough Skin/Rocky Helmet belongs to `[of]`, not
        # necessarily to the event's affected actor.
        source_ident = ""
        for value in parts[3:]:
            stripped = value.strip()
            if stripped.lower().startswith("[of] "):
                source_ident = stripped[5:].strip()
        source_side, source_species = actor_species(source_ident or (parts[2] if len(parts) > 2 else ""))
        if source_side == opponent_role and source_species:
            for value in parts[3:]:
                lowered = value.strip().lower()
                if lowered.startswith("[from] item:") or lowered.startswith("item:"):
                    item = norm(value.split(":", 1)[1])
                    if item:
                        reveal = ensure(source_species)
                        reveal.current_item = item
                        reveal.item_status_revealed = True
                        reveal.historically_revealed_items.add(item)
                elif lowered.startswith("[from] ability:") or lowered.startswith("ability:"):
                    ability = norm(value.split(":", 1)[1])
                    if ability:
                        ensure(source_species).ability = ability

    opponent = tuple(
        OpponentReveal(
            species=species,
            level=reveal.level,
            moves=tuple(sorted(reveal.moves)),
            ability=reveal.ability,
            current_item=reveal.current_item,
            item_status_revealed=reveal.item_status_revealed,
            historically_revealed_items=tuple(sorted(reveal.historically_revealed_items)),
        )
        for species, reveal in sorted(reveals.items())
    )
    if not opponent:
        raise CausalProtocolBridgeError("protocol contains no revealed opponent species")
    return CausalProtocolFacts(
        observer_role=player_role,
        opponent_role=opponent_role,
        opponent=opponent,
        opponent_active_species=active[opponent_role],
        public_event_lines=public_lines,
        request_lines_ignored_for_opponent=len(requests),
    )


def _clone_conditions(engine: Any, value: Any) -> Any:
    names = (
        "spikes", "toxic_spikes", "stealth_rock", "sticky_web", "tailwind",
        "lucky_chant", "lunar_dance", "reflect", "light_screen", "aurora_veil",
        "crafty_shield", "safeguard", "mist", "protect", "healing_wish",
        "mat_block", "quick_guard", "toxic_count", "wide_guard",
    )
    return engine.SideConditions(**{name: int(getattr(value, name)) for name in names})


def _clone_durations(engine: Any, value: Any) -> Any:
    names = ("confusion", "encore", "lockedmove", "slowstart", "taunt", "yawn")
    return engine.VolatileStatusDurations(
        **{name: int(getattr(value, name)) for name in names}
    )


def _clone_move(engine: Any, move: Any, *, move_id: str | None = None) -> Any:
    pp = int(getattr(move, "pp"))
    if move_id is not None and pp <= 0:
        pp = 1
    return engine.Move(
        id=move_id if move_id is not None else move.id,
        pp=pp,
        disabled=bool(move.disabled),
    )


def _clone_pokemon(
    engine: Any,
    pokemon: Any,
    *,
    item: str | None = None,
    ability: str | None = None,
    moves: Sequence[Any] | None = None,
) -> Any:
    return engine.Pokemon(
        id=pokemon.id,
        level=int(pokemon.level),
        types=tuple(pokemon.types),
        base_types=tuple(pokemon.base_types),
        hp=int(pokemon.hp),
        maxhp=int(pokemon.maxhp),
        ability=ability if ability is not None else pokemon.ability,
        base_ability=pokemon.base_ability,
        item=item if item is not None else pokemon.item,
        nature=pokemon.nature,
        evs=tuple(pokemon.evs),
        attack=int(pokemon.attack),
        defense=int(pokemon.defense),
        special_attack=int(pokemon.special_attack),
        special_defense=int(pokemon.special_defense),
        speed=int(pokemon.speed),
        status=pokemon.status,
        rest_turns=int(pokemon.rest_turns),
        sleep_turns=int(pokemon.sleep_turns),
        weight_kg=float(pokemon.weight_kg),
        moves=list(moves) if moves is not None else [
            _clone_move(engine, move) for move in pokemon.moves
        ],
        terastallized=bool(pokemon.terastallized),
        tera_type=pokemon.tera_type,
    )


def _clone_side(engine: Any, side: Any, pokemon: Sequence[Any]) -> Any:
    return engine.Side(
        pokemon=list(pokemon),
        side_conditions=_clone_conditions(engine, side.side_conditions),
        active_index=side.active_index,
        baton_passing=bool(side.baton_passing),
        shed_tailing=bool(side.shed_tailing),
        volatile_status_durations=_clone_durations(engine, side.volatile_status_durations),
        wish=tuple(side.wish),
        future_sight=tuple(side.future_sight),
        force_switch=bool(side.force_switch),
        force_trapped=bool(side.force_trapped),
        slow_uturn_move=bool(side.slow_uturn_move),
        volatile_statuses=set(side.volatile_statuses),
        substitute_health=int(side.substitute_health),
        attack_boost=int(side.attack_boost),
        defense_boost=int(side.defense_boost),
        special_attack_boost=int(side.special_attack_boost),
        special_defense_boost=int(side.special_defense_boost),
        speed_boost=int(side.speed_boost),
        accuracy_boost=int(side.accuracy_boost),
        evasion_boost=int(side.evasion_boost),
        last_used_move=side.last_used_move,
        switch_out_move_second_saved_move=side.switch_out_move_second_saved_move,
    )


def _clone_state(engine: Any, state: Any, side_two_pokemon: Sequence[Any], bits: int) -> Any:
    return engine.State(
        side_one=_clone_side(
            engine, state.side_one,
            [_clone_pokemon(engine, pokemon) for pokemon in state.side_one.pokemon],
        ),
        side_two=_clone_side(engine, state.side_two, side_two_pokemon),
        weather=state.weather,
        weather_turns_remaining=int(state.weather_turns_remaining),
        terrain=state.terrain,
        terrain_turns_remaining=int(state.terrain_turns_remaining),
        trick_room=bool(state.trick_room),
        trick_room_turns_remaining=int(state.trick_room_turns_remaining),
        team_preview=bool(state.team_preview),
        s1_threat=float(state.s1_threat),
        s2_threat=float(state.s2_threat),
        scout_value=float(state.scout_value),
        threat_matrix=list(state.threat_matrix),
        wincon_matrix=list(state.wincon_matrix),
        s1_public_reveals=int(bits),
        s2_public_reveals=int(state.s2_public_reveals),
    )


def reconcile_causal_facts(state: Any, engine: Any, facts: CausalProtocolFacts) -> ReconciledPublicState:
    """Attach already-public facts to engine slots and repair stale placeholders.

    The caller supplies an engine state, but this function never uses a field
    merely because it is populated.  It reads IDs/move IDs only after a public
    event has provided the identity to locate.
    """
    opponent = list(state.side_two.pokemon)
    slots: dict[str, int] = {}
    ambiguous: set[str] = set()
    for index, pokemon in enumerate(opponent):
        species = canonical_species(pokemon.id)
        if species in {"", "none"}:
            continue
        if species in slots:
            ambiguous.add(species)
        slots[species] = index
    for species in ambiguous:
        slots.pop(species, None)

    bits = 0
    repairs: list[str] = []
    item_sidecar: list[tuple[str, tuple[str, ...]]] = []
    claimed: set[int] = set()
    for reveal in facts.opponent:
        slot = slots.get(canonical_species(reveal.species))
        if slot is None or slot in claimed:
            raise CausalProtocolBridgeError(
                f"public species cannot map uniquely: {reveal.species}"
            )
        claimed.add(slot)
        bits |= 1 << slot
        original = opponent[slot]
        moves = [_clone_move(engine, move) for move in original.moves]
        for public_move in reveal.moves:
            matches = [index for index, move in enumerate(moves) if norm(move.id) == public_move]
            if len(matches) > 1:
                raise CausalProtocolBridgeError(
                    f"public move maps ambiguously: {reveal.species}/{public_move}"
                )
            if not matches:
                unknown = [
                    index for index, move in enumerate(moves)
                    if norm(move.id) in {"", "none", "nomove"}
                ]
                if not unknown:
                    raise CausalProtocolBridgeError(
                        f"public move has no deterministic slot: {reveal.species}/{public_move}"
                    )
                move_slot = unknown[0]
                moves[move_slot] = _clone_move(engine, moves[move_slot], move_id=public_move)
                repairs.append(f"move:{reveal.species}:{public_move}")
            else:
                move_slot = matches[0]
            bits |= 1 << (6 + slot * 4 + move_slot)

        item = None
        if reveal.item_status_revealed:
            item = reveal.current_item or "none"
            bits |= 1 << (30 + slot)
            if norm(original.item) != norm(item):
                repairs.append(f"item:{reveal.species}:{norm(original.item)}->{item}")
        ability = None
        if reveal.ability is not None:
            ability = reveal.ability
            bits |= 1 << (36 + slot)
            if norm(original.ability) != norm(ability):
                repairs.append(f"ability:{reveal.species}:{norm(original.ability)}->{ability}")
        if reveal.historically_revealed_items:
            item_sidecar.append((reveal.species, reveal.historically_revealed_items))
        opponent[slot] = _clone_pokemon(
            engine, original, item=item, ability=ability, moves=moves
        )

    if bits <= 0 or bits & ~((1 << 42) - 1):
        raise CausalProtocolBridgeError("invalid causal reveal mask")
    return ReconciledPublicState(
        state=_clone_state(engine, state, opponent, bits),
        reveal_bits=bits,
        archival_repairs=tuple(repairs),
        historical_item_sidecar=tuple(sorted(item_sidecar)),
    )
