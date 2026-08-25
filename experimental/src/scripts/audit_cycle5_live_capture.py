#!/usr/bin/env python3
"""Run the frozen Cycle 5 production causal-ledger admission audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.run_public_search_state_gate_a import hidden_perturbation, sha256
from search.public_search_state_v1 import canonical_bytes, extract_public_search_state
from srcs.metagross.causal_reveal_ledger import (
    CausalRevealLedger,
    CausalRevealLedgerError,
    LEDGER_ATTRIBUTE,
    attached_ledger,
    causal_engine_payload,
    canonical_species,
    clear_public_protocol_lines,
    compile_reveal_bits,
    convert_battle_with_causal_ledger,
    freeze_and_attach_battle_ledger,
    freeze_ledger,
    install_observer_mask,
    norm,
    record_public_protocol_lines,
    serialization_without_masks,
    verify_sampled_ledgers,
)


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_CAPTURE_SHA256 = "7fd73b24cb7f91a8d88c26b1337084bde0be5d26d46189606800a45131f52056"
WORLD_COUNT = 8
SEED = 2026081505


class AdmissionError(RuntimeError):
    pass


def load_fixture() -> dict[str, Any]:
    paths = [
        ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-a-decisions.jsonl.dual-r1-roots.jsonl",
        ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-b-decisions.jsonl.dual-r1-roots.jsonl",
    ]
    matches = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("capture_sha256") == FIXTURE_CAPTURE_SHA256:
                    matches.append(row)
    if len(matches) != 1:
        raise AdmissionError("frozen fixture is missing or ambiguous")
    return matches[0]


def public_lines(snapshot: dict[str, Any]) -> list[str]:
    return [
        line for line in snapshot["protocol_prefix"]
        if line and not line.startswith("|request|")
    ]


def public_levels(lines: Sequence[str], opponent_role: str) -> dict[str, int]:
    levels: dict[str, int] = {}
    for line in lines:
        parts = line.split("|")
        if len(parts) < 4 or parts[1] not in {"switch", "drag", "replace", "detailschange", "poke"}:
            continue
        side = parts[2][:2]
        if side != opponent_role:
            continue
        details = parts[3]
        species = canonical_species(details.split(",", 1)[0])
        level = next(
            (int(part.strip()[1:]) for part in details.split(",") if part.strip().startswith("L") and part.strip()[1:].isdigit()),
            None,
        )
        if level is not None:
            levels[species] = level
    return levels


def build_fixture_battle(ledger: CausalRevealLedger, request: dict[str, Any], lines: Sequence[str]) -> Any:
    from constants import BattleType, UNKNOWN_ITEM
    from fp.battle import Battle, Pokemon

    levels = public_levels(lines, ledger.opponent_role)
    battle = Battle(ledger.battle_tag)
    battle.battle_type = BattleType.RANDOM_BATTLE
    battle.pokemon_format = "gen9randombattle"
    battle.generation = "gen9"
    battle.user.name = ledger.observer_role
    battle.opponent.name = ledger.opponent_role
    battle.user.initialize_first_turn_user_from_json(request)
    mons = {}
    for fact in ledger.facts:
        level = levels.get(fact.species)
        if level is None:
            raise AdmissionError(f"fixture has no public level for {fact.species}")
        mon = Pokemon(fact.species, level)
        for move in fact.moves:
            if mon.add_move(move) is None:
                raise AdmissionError(f"unknown public move in fixture: {move}")
        mon.ability = fact.ability
        if fact.item_status_revealed:
            mon.item = None if fact.current_item == "none" else fact.current_item
            if fact.current_item == "none" and fact.consumed_items:
                mon.removed_item = fact.consumed_items[-1]
        else:
            mon.item = UNKNOWN_ITEM
        mons[fact.species] = mon
    active = mons.pop(ledger.opponent_active_species, None)
    if active is None:
        raise AdmissionError("fixture active is not in causal facts")
    battle.opponent.active = active
    battle.opponent.reserve = [mons[name] for name in sorted(mons)]
    battle.started = True
    battle.rqid = request["rqid"]
    setattr(battle, LEDGER_ATTRIBUTE, ledger.to_payload())
    return battle


def instruction_signature(value: Any) -> tuple[float, tuple[str, ...]]:
    return float(value.percentage), tuple(str(row) for row in value.instruction_list)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def field_totals(ledger: CausalRevealLedger) -> Counter[str]:
    result: Counter[str] = Counter()
    for fact in ledger.facts:
        result["species"] += 1
        result["move"] += len(fact.moves)
        result["item"] += int(fact.item_status_revealed)
        result["ability"] += int(fact.ability is not None)
    return result


def field_typed_counts(state: Any, ledger: CausalRevealLedger, *, swap: bool) -> Counter[str]:
    opponent = state.side_one if swap else state.side_two
    bits = int(state.s2_public_reveals if swap else state.s1_public_reveals)
    result: Counter[str] = Counter()
    for fact in ledger.facts:
        matches = [
            index for index, pokemon in enumerate(opponent.pokemon)
            if canonical_species(pokemon.id) == fact.species
        ]
        if len(matches) != 1:
            continue
        slot = matches[0]
        result["species"] += int(bool(bits & (1 << slot)))
        for move in fact.moves:
            move_slots = [index for index, row in enumerate(opponent.pokemon[slot].moves) if norm(row.id) == move]
            if len(move_slots) == 1:
                result["move"] += int(bool(bits & (1 << (6 + slot * 4 + move_slots[0]))))
        result["item"] += int(fact.item_status_revealed and bool(bits & (1 << (30 + slot))))
        result["ability"] += int(fact.ability is not None and bool(bits & (1 << (36 + slot))))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = run_dir / "cycle5-live-capture-admission.json"
    if output.exists():
        raise AdmissionError("Cycle 5 report already exists")
    frozen = json.loads((run_dir / "FROZEN_INPUTS.json").read_text())
    for relative, expected in frozen["files"].items():
        if sha256(ROOT / relative) != expected:
            raise AdmissionError(f"frozen input hash mismatch: {relative}")

    import poke_engine

    # The frozen fixture builder uses the same Foul Play Battle/Pokemon classes as
    # production, so make the pinned vendor tree importable before constructing it.
    foul_play = ROOT / "srcs/vendor/foul-play"
    sys.path.insert(0, str(foul_play))

    fixture = load_fixture()
    snapshot = fixture["r1_policy_snapshot"]
    lines = public_lines(snapshot)
    request = snapshot["player_information_state"]["private_request"]
    tag = fixture["identity"]["battle_tag"]

    # Controlled production receive-tee smoke: requests are explicitly excluded.
    clear_public_protocol_lines(tag)
    record_public_protocol_lines(tag, [*lines, "|request|{\"private\":true}"])
    battle = build_fixture_battle(
        freeze_ledger(tag, snapshot["player_role"], lines), request, lines
    )
    delattr(battle, LEDGER_ATTRIBUTE)
    ledger = freeze_and_attach_battle_ledger(battle)
    if ledger.canonical_bytes() != freeze_ledger(tag, snapshot["player_role"], lines).canonical_bytes():
        raise AdmissionError("controlled live capture changed causal ledger")

    previous_cwd = Path.cwd()
    os.chdir(foul_play)
    try:
        from data.pkmn_sets import RandomBattleTeamDatasets
        from fp.search import main as search_main

        RandomBattleTeamDatasets.initialize("gen9")
        worlds = search_main.prepare_random_battles(
            battle, WORLD_COUNT, rng=random.Random(SEED)
        )
        verify_sampled_ledgers(battle, worlds)
        expected_per_world = field_totals(ledger)
        totals: Counter[str] = Counter()
        typed: Counter[str] = Counter()
        swap_typed: Counter[str] = Counter()
        masks = []
        swap_masks = []
        overhead_ms = []
        converted = []
        for sampled, _weight in worlds:
            raw = search_main.battle_to_poke_engine_state(copy.deepcopy(sampled))
            before = serialization_without_masks(raw)
            started = time.perf_counter()
            bits = compile_reveal_bits(raw, ledger, swap=False)
            state = install_observer_mask(raw, bits, swap=False, engine=poke_engine)
            overhead_ms.append((time.perf_counter() - started) * 1000.0)
            if serialization_without_masks(state) != before:
                raise AdmissionError("normal conversion changed non-mask bytes")
            expected_bits = compile_reveal_bits(raw, ledger, swap=False)
            if int(state.s1_public_reveals) != expected_bits or int(state.s2_public_reveals) != 0:
                raise AdmissionError("normal mask has spurious or missing bits")
            converted.append(state)
            masks.append(int(state.s1_public_reveals))
            totals.update(expected_per_world)
            typed.update(field_typed_counts(state, ledger, swap=False))

            raw_swap = search_main.battle_to_poke_engine_state(copy.deepcopy(sampled), swap=True)
            swap_before = serialization_without_masks(raw_swap)
            swap_bits = compile_reveal_bits(raw_swap, ledger, swap=True)
            swapped = install_observer_mask(raw_swap, swap_bits, swap=True, engine=poke_engine)
            if serialization_without_masks(swapped) != swap_before:
                raise AdmissionError("swapped conversion changed non-mask bytes")
            if int(swapped.s1_public_reveals) != 0 or int(swapped.s2_public_reveals) != swap_bits:
                raise AdmissionError("swapped mask has spurious or missing bits")
            swap_masks.append(int(swapped.s2_public_reveals))
            swap_typed.update(field_typed_counts(swapped, ledger, swap=True))

        # Actual wrapper must produce byte-identical installed states.
        for (sampled, _weight), expected in zip(worlds, converted, strict=True):
            wrapped = convert_battle_with_causal_ledger(
                copy.deepcopy(sampled), search_main.battle_to_poke_engine_state,
                poke_engine, swap=False,
            )
            if wrapped.to_string() != expected.to_string():
                raise AdmissionError("production converter wrapper disagrees")
    finally:
        os.chdir(previous_cwd)

    fields = ("species", "move", "item", "ability")
    recall = {field: typed[field] / totals[field] for field in fields}
    swap_recall = {field: swap_typed[field] / totals[field] for field in fields}
    if any(value != 1.0 for value in [*recall.values(), *swap_recall.values()]):
        raise AdmissionError("typed recall is below 100%")

    # Public noninterference and exact apply/reverse/deterministic replay.
    state = converted[0]
    public = canonical_bytes(extract_public_search_state(state, poke_engine))
    perturbation = hidden_perturbation(state, poke_engine)
    if perturbation is not None and canonical_bytes(
        extract_public_search_state(perturbation, poke_engine)
    ) != public:
        raise AdmissionError("hidden completion changed public projection")
    first_actions, second_actions = poke_engine.root_options(state)
    first = poke_engine.step_with_uniform_r1_semantic(state, first_actions[0], second_actions[0], 0.25)
    second = poke_engine.step_with_uniform_r1_semantic(state, first_actions[0], second_actions[0], 0.25)
    if (
        first.state.to_string() != second.state.to_string()
        or instruction_signature(first.selected_instructions) != instruction_signature(second.selected_instructions)
    ):
        raise AdmissionError("deterministic replay mismatch")
    restored = first.state.reverse_instructions(first.selected_instructions)
    if restored.to_string() != state.to_string() or canonical_bytes(
        extract_public_search_state(restored, poke_engine)
    ) != public:
        raise AdmissionError("apply/reverse did not restore causal state")

    payload = causal_engine_payload(state, ledger, swap=False)
    if json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":"))) != payload:
        raise AdmissionError("combined engine/sidecar payload did not round-trip")
    consumed_smoke = freeze_ledger(
        "battle-consumed-sidecar", "p1", [
            "|switch|p2a: Foe|Snorlax, L80|100/100",
            "|-enditem|p2a: Foe|Sitrus Berry|[eat]",
        ],
    )
    consumed_roundtrip = CausalRevealLedger.from_payload(
        json.loads(consumed_smoke.canonical_bytes())
    )
    if consumed_roundtrip.facts[0].consumed_items != ("sitrusberry",):
        raise AdmissionError("consumed-item sidecar identity was not preserved")

    failure_counts: Counter[str] = Counter()
    for mutation in ("missing_move", "missing_species"):
        payload_copy = ledger.to_payload()
        if mutation == "missing_move":
            payload_copy["facts"][0]["moves"].append("definitelynotamove")
        else:
            payload_copy["facts"][0]["species"] = "definitelynotaspecies"
        try:
            compile_reveal_bits(state, CausalRevealLedger.from_payload(payload_copy), swap=False)
        except CausalRevealLedgerError:
            failure_counts[mutation] += 1
        else:
            raise AdmissionError(f"{mutation} did not fail closed")

    freeze_ms = []
    for _ in range(50):
        started = time.perf_counter()
        freeze_ledger(tag, snapshot["player_role"], lines)
        freeze_ms.append((time.perf_counter() - started) * 1000.0)
    freeze_p95 = percentile(freeze_ms, 95)
    overhead_p95 = percentile(overhead_ms, 95)
    gates = {
        "typed_recall_100_percent": all(value == 1.0 for value in recall.values()),
        "swapped_typed_recall_100_percent": all(value == 1.0 for value in swap_recall.values()),
        "zero_spurious_bits": True,
        "non_mask_serialization_parity": True,
        "hidden_noninterference": True,
        "apply_reverse_parity": True,
        "perspective_swap": True,
        "serialization_roundtrip": True,
        "deterministic_replay": True,
        "failure_accounting": failure_counts == {"missing_move": 1, "missing_species": 1},
        "conversion_overhead_p95_le_5ms": overhead_p95 <= 5.0,
        "ledger_freeze_p95_le_5ms": freeze_p95 <= 5.0,
    }
    status = "pass" if all(gates.values()) else "fail"
    report = {
        "schema": "metagross-cycle5-live-capture-admission/v1",
        "status": status,
        "gates": gates,
        "fixture": {"capture_sha256": FIXTURE_CAPTURE_SHA256, "worlds": WORLD_COUNT, "seed": SEED},
        "typed_totals": dict(totals), "typed_preserved": dict(typed),
        "typed_recall": recall, "swapped_typed_preserved": dict(swap_typed),
        "swapped_typed_recall": swap_recall,
        "masks": masks, "swapped_masks": swap_masks,
        "failure_counts": dict(failure_counts),
        "performance_ms": {
            "conversion_added_samples": overhead_ms,
            "conversion_added_mean": statistics.fmean(overhead_ms),
            "conversion_added_p95": overhead_p95,
            "ledger_freeze_mean": statistics.fmean(freeze_ms),
            "ledger_freeze_p95": freeze_p95,
        },
        "sidecar_boundary": {
            "consumed_item_identity_preserved": True,
            "json_roundtrip": True,
            "rust_interior_sidecar_updates_proven": False,
            "newly_simulated_consumed_item_updates_proven": False,
            "deployed_rust_interior_inference_authorized": False,
        },
        "authorization": {
            "freeze_target_collection_permission": status == "pass",
            "target_collection_started": False, "training": False,
            "h2h": False, "deployment": False, "sealed_confirmation": False,
        },
        "sealed_confirmation_panel_rows_read": 0,
        "local_cpu_only": True, "paid_compute_usd": 0,
        "hashes": {
            "protocol_sha256": frozen["protocol_sha256"],
            "frozen_inputs_sha256": sha256(run_dir / "FROZEN_INPUTS.json"),
            "engine_binding_sha256": frozen["files"]["experimental/engine/pe_v3_learned_priors/poke-engine-py/python/poke_engine/poke_engine.cpython-311-darwin.so"],
        },
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    clear_public_protocol_lines(tag)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
