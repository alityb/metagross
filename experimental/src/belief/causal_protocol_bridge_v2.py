"""Cycle 4 bridge using the source-pinned Showdown public-form contract."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from belief.causal_protocol_bridge import (
    CausalProtocolBridgeError,
    CausalProtocolFacts,
    ReconciledPublicState,
    _clone_move,
    _clone_pokemon,
    _clone_state,
    norm,
    parse_causal_protocol,
)
from belief.public_form_contract import PublicFormContract


_SPECIES_DETAIL_TAGS = {"switch", "drag", "replace", "detailschange"}


def _canonical_protocol_lines(
    protocol_prefix: Sequence[str], contract: PublicFormContract
) -> list[str]:
    """Canonicalize only public species-detail fields using Showdown authority."""
    result: list[str] = []
    for line in protocol_prefix:
        if not line.startswith("|"):
            result.append(line)
            continue
        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag in _SPECIES_DETAIL_TAGS and len(parts) >= 4:
            detail = parts[3].split(",", 1)
            detail[0] = contract.canonical(detail[0])
            parts[3] = ",".join(detail)
            line = "|".join(parts)
        result.append(line)
    return result


def parse_causal_protocol_v2(
    protocol_prefix: Sequence[str],
    *,
    player_role: str,
    private_request: Mapping[str, Any],
    contract: PublicFormContract,
) -> CausalProtocolFacts:
    return parse_causal_protocol(
        _canonical_protocol_lines(protocol_prefix, contract),
        player_role=player_role,
        private_request=private_request,
    )


def reconcile_causal_facts_v2(
    state: Any,
    engine: Any,
    facts: CausalProtocolFacts,
    contract: PublicFormContract,
) -> ReconciledPublicState:
    """Reconcile with source-pinned public identity; never inspect hidden extras."""
    opponent = list(state.side_two.pokemon)
    slots: dict[str, int] = {}
    ambiguous: set[str] = set()
    for index, pokemon in enumerate(opponent):
        species = contract.canonical(pokemon.id)
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
        species = contract.canonical(reveal.species)
        slot = slots.get(species)
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
