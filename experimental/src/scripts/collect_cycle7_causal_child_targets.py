#!/usr/bin/env python3
"""Run the frozen Cycle 7 causal child-target collector mechanics gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from belief.causal_protocol_bridge import _clone_move, _clone_pokemon, _clone_state, norm
from belief.causal_protocol_bridge_v2 import parse_causal_protocol_v2, reconcile_causal_facts_v2
from belief.public_form_contract import load_public_form_contract
from scripts.audit_cycle5_live_capture import build_fixture_battle, public_lines
from scripts.audit_public_search_state_gate_a_selection import select_without_representation
from scripts.run_causal_protocol_bridge_cycle4 import request_actions_exact
from scripts.run_causal_protocol_bridge_gate import _schedule
from scripts.run_public_search_state_gate_a import hidden_perturbation, load_rows, sha256
from search.causal_child_target_v1 import (
    CausalChildTargetError,
    child_information_state,
    group_target_members,
    public_fingerprint,
    snapshot_teacher_result,
)
from search.public_search_state_v1 import canonical_bytes
from srcs.metagross.causal_reveal_ledger import (
    CausalRevealLedger,
    canonical_species,
    compile_reveal_bits,
    freeze_ledger,
    install_observer_mask,
    verify_sampled_ledgers,
)


ROOT = Path(__file__).resolve().parents[3]
BASE_SEED = 2026081507
WORLDS = 4
ITERATIONS = 2048
WORKERS = 8
ROOT_CAP = 44
MIN_ROOTS = 40
SPLIT_COUNTS = {"train": 26, "validation": 9, "test": 9}


class CollectorError(RuntimeError):
    pass


def stable_u64(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def instruction_signature(value: Any) -> tuple[float, tuple[str, ...]]:
    return float(value.percentage), tuple(str(row) for row in value.instruction_list)


def event_payload(event: Any) -> dict[str, Any]:
    return {
        "kind": str(event.kind),
        "side": None if event.side is None else str(event.side),
        "pokemon_index": None if event.pokemon_index is None else str(event.pokemon_index),
        "move_id": None if event.move_id is None else str(event.move_id),
        "amount": None if event.amount is None else int(event.amount),
        "detail": None if event.detail is None else str(event.detail),
    }


def _clone_known_pokemon(engine: Any, root: Any, donor: Any, bits: int, slot: int) -> Any:
    moves = [_clone_move(engine, move) for move in donor.moves]
    for root_index, root_move in enumerate(root.moves):
        if not bits & (1 << (6 + slot * 4 + root_index)):
            continue
        matches = [index for index, move in enumerate(moves) if norm(move.id) == norm(root_move.id)]
        if len(matches) != 1:
            raise CollectorError(f"sampled donor lost certified move: {root.id}/{root_move.id}")
        moves[matches[0]] = engine.Move(
            id=root_move.id, pp=int(root_move.pp), disabled=bool(root_move.disabled)
        )
    maxhp = int(donor.maxhp)
    if maxhp <= 0 or int(root.maxhp) <= 0:
        raise CollectorError("invalid known Pokemon max HP")
    public_percent = 0 if int(root.hp) == 0 else math.ceil(100 * int(root.hp) / int(root.maxhp))
    hp = 0 if public_percent == 0 else max(1, math.floor(public_percent * maxhp / 100))
    return engine.Pokemon(
        id=root.id,
        level=int(root.level),
        types=tuple(root.types),
        base_types=tuple(root.base_types),
        hp=hp,
        maxhp=maxhp,
        ability=root.ability if bits & (1 << (36 + slot)) else donor.ability,
        base_ability=donor.base_ability,
        item=root.item if bits & (1 << (30 + slot)) else donor.item,
        nature=donor.nature,
        evs=tuple(donor.evs),
        attack=int(donor.attack),
        defense=int(donor.defense),
        special_attack=int(donor.special_attack),
        special_defense=int(donor.special_defense),
        speed=int(donor.speed),
        status=root.status,
        rest_turns=int(root.rest_turns),
        sleep_turns=int(root.sleep_turns),
        weight_kg=float(root.weight_kg),
        moves=moves,
        terastallized=bool(root.terastallized),
        tera_type=root.tera_type,
    )


def merge_hidden_completion(engine: Any, root: Any, donor: Any, ledger: CausalRevealLedger) -> Any:
    # Bit positions are completion-local because sampled move and reserve slots
    # can differ. Equality is semantic (ledger facts), never raw packed bits.
    compile_reveal_bits(donor, ledger, swap=False)
    root_bits = compile_reveal_bits(root, ledger, swap=False)
    donor_rows = list(donor.side_two.pokemon)
    used: set[int] = set()
    merged: list[Any | None] = [None] * 6
    for slot, root_pokemon in enumerate(root.side_two.pokemon):
        if not root_bits & (1 << slot):
            continue
        matches = [
            index for index, pokemon in enumerate(donor_rows)
            if index not in used and canonical_species(pokemon.id) == canonical_species(root_pokemon.id)
        ]
        if len(matches) != 1:
            raise CollectorError(f"sampled donor species mapping failed: {root_pokemon.id}")
        donor_slot = matches[0]
        used.add(donor_slot)
        merged[slot] = _clone_known_pokemon(
            engine, root_pokemon, donor_rows[donor_slot], root_bits, slot
        )
    remaining = [
        _clone_pokemon(engine, pokemon)
        for index, pokemon in enumerate(donor_rows) if index not in used
    ]
    for slot in range(6):
        if merged[slot] is None:
            if not remaining:
                raise CollectorError("sampled donor has too few hidden Pokemon")
            merged[slot] = remaining.pop(0)
    if remaining:
        raise CollectorError("sampled donor has extra hidden Pokemon")
    state = _clone_state(engine, root, merged, root_bits)
    compiled = compile_reveal_bits(state, ledger, swap=False)
    state = _clone_state(engine, root, merged, compiled)
    if int(state.s1_public_reveals) != compiled:
        raise CollectorError("merged world lost certified reveal bits")
    return state


def child_reveal_sidecar(root: Any, step: Any, ledger: CausalRevealLedger) -> dict[str, Any]:
    child = step.state
    before = int(root.s1_public_reveals)
    after = int(child.s1_public_reveals)
    if before & ~after:
        raise CollectorError("semantic child cleared a root reveal")
    events = [event_payload(event) for event in step.events]
    additions = []
    for offset in range(42):
        bit = 1 << offset
        if not after & bit or before & bit:
            continue
        if offset < 6:
            slot, field, value = offset, "species", str(child.side_two.pokemon[offset].id)
        elif offset < 30:
            slot = (offset - 6) // 4
            move_slot = (offset - 6) % 4
            field = "move"
            value = str(child.side_two.pokemon[slot].moves[move_slot].id)
            if norm(value) in {"", "none", "nomove"}:
                raise CollectorError("semantic child revealed an absent move")
        elif offset < 36:
            slot, field = offset - 30, "item"
            current = norm(child.side_two.pokemon[slot].item)
            activations = [
                event["detail"] for event in events
                if event["kind"] == "item_activated"
                and event["side"] == "side_two"
                and event["pokemon_index"] == str(slot)
                and event["detail"]
            ]
            if current in {"", "none"}:
                if len(set(map(norm, activations))) != 1:
                    raise CollectorError("new consumed-item reveal lacks unique event identity")
                value = activations[0]
            else:
                value = str(child.side_two.pokemon[slot].item)
        else:
            slot, field, value = offset - 36, "ability", str(child.side_two.pokemon[offset - 36].ability)
            if norm(value) in {"", "none"}:
                raise CollectorError("semantic child revealed an absent ability")
        additions.append({"slot": slot, "field": field, "value": value})
    return {
        "root_ledger": ledger.to_payload(),
        "root_ledger_sha256": hashlib.sha256(ledger.canonical_bytes()).hexdigest(),
        "root_mask": before,
        "child_mask": after,
        "added_reveals": additions,
        "semantic_events": events,
        "unaccounted_instruction_kinds": list(step.unaccounted_instruction_kinds),
        "simulated_consumed_item_identity_recorded_in_python_sidecar": any(
            row["field"] == "item" and norm(row["value"]) not in {"", "none"}
            for row in additions
        ),
        "rust_interior_sidecar_update_proven": False,
    }


def _teacher_task(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import poke_engine

        state = poke_engine.State.from_string(str(payload["state"]))
        first, second = map(list, poke_engine.root_options(state))
        s1_priors = [(action, 1.0 / len(first)) for action in first]
        s2_priors = [(action, 1.0 / len(second)) for action in second]

        def run() -> dict[str, Any]:
            result = poke_engine.monte_carlo_tree_search(
                state,
                duration_ms=0,
                iterations=ITERATIONS,
                threads=1,
                s1_priors=s1_priors,
                s2_priors=s2_priors,
                c_puct=2.0,
                seed=int(payload["teacher_seed"]),
            )
            return snapshot_teacher_result(result)

        started = time.perf_counter()
        teacher = run()
        repeated = None
        if payload["repeat_canary"]:
            repeated = run()
            if repeated != teacher:
                raise CollectorError("seeded teacher repeat disagreed")
        return {
            "target_id": payload["target_id"],
            "teacher": teacher,
            "repeat_canary": bool(payload["repeat_canary"]),
            "repeat_exact": repeated == teacher if repeated is not None else None,
            "elapsed_seconds": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        return {
            "target_id": payload["target_id"],
            "teacher": None,
            "repeat_canary": bool(payload.get("repeat_canary")),
            "repeat_exact": False if payload.get("repeat_canary") else None,
            "elapsed_seconds": 0.0,
            "error": f"{type(exc).__name__}:{exc}",
        }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report_path = run_dir / "cycle7-collector-mechanics-report.json"
    member_path = run_dir / "cycle7-target-members.jsonl"
    group_path = run_dir / "cycle7-target-groups.jsonl"
    if any(path.exists() for path in (report_path, member_path, group_path)):
        raise CollectorError("Cycle 7 output already exists")
    frozen = json.loads((run_dir / "FROZEN_INPUTS.json").read_text())
    for relative, expected in frozen["files"].items():
        if sha256(ROOT / relative) != expected:
            raise CollectorError(f"frozen input hash mismatch: {relative}")

    import poke_engine

    source_paths = [ROOT / path for path in frozen["source_paths"]]
    rows = load_rows(source_paths[0]) + load_rows(source_paths[1])
    selected, selection_rejections = select_without_representation(rows, poke_engine)
    first_by_battle = {}
    for row in selected:
        first_by_battle.setdefault(str(row["identity"]["battle_tag"]), row)
    battles = sorted(first_by_battle, key=lambda tag: stable_u64("cycle7-split-v1", tag))
    if not MIN_ROOTS <= len(battles) <= ROOT_CAP:
        raise CollectorError(f"root inventory {len(battles)} is outside [{MIN_ROOTS},{ROOT_CAP}]")
    split_by_battle = {}
    cursor = 0
    for split, count in SPLIT_COUNTS.items():
        for battle in battles[cursor : cursor + count]:
            split_by_battle[battle] = split
        cursor += count
    if cursor != len(battles):
        raise CollectorError("frozen 26/9/9 battle split does not match inventory")
    selected_roots = [first_by_battle[battle] for battle in battles]

    contract = load_public_form_contract()
    foul_play = ROOT / "srcs/vendor/foul-play"
    sys.path.insert(0, str(foul_play))
    previous_cwd = Path.cwd()
    os.chdir(foul_play)
    from data.pkmn_sets import RandomBattleTeamDatasets
    from fp.search import main as search_main

    RandomBattleTeamDatasets.initialize("gen9")
    scheduled = len(selected_roots) * WORLDS * 8
    schedule_supported = 0
    eligible = 0
    terminal_or_automatic = 0
    failures: Counter[str] = Counter()
    tasks = []
    public_root_by_id = {}
    collection_started = time.perf_counter()
    try:
        for root in selected_roots:
            battle_tag = str(root["identity"]["battle_tag"])
            split = split_by_battle[battle_tag]
            snapshot = root["r1_policy_snapshot"]
            lines = public_lines(snapshot)
            request = snapshot["player_information_state"]["private_request"]
            schedule = _schedule(root)
            try:
                if request_actions_exact(request) != set(root["_side_one_actions"]):
                    raise CollectorError("request/root legality mismatch")
                facts = parse_causal_protocol_v2(
                    snapshot["protocol_prefix"], player_role=snapshot["player_role"],
                    private_request=request, contract=contract,
                )
                reconciled = reconcile_causal_facts_v2(root["_state"], poke_engine, facts, contract)
                if any(repair.startswith("move:") for repair in reconciled.archival_repairs):
                    raise CollectorError("root reconciliation required placeholder move/PP")
                root_state = reconciled.state
                ledger = freeze_ledger(battle_tag, snapshot["player_role"], lines)
                compiled = compile_reveal_bits(root_state, ledger, swap=False)
                if compiled != int(root_state.s1_public_reveals):
                    raise CollectorError("Cycle4 and live-ledger masks disagree")
                battle = build_fixture_battle(ledger, request, lines)
                world_seed = stable_u64(BASE_SEED, root["capture_sha256"], "worlds")
                worlds = search_main.prepare_random_battles(
                    battle, WORLDS, rng=random.Random(world_seed)
                )
                verify_sampled_ledgers(battle, worlds)
                if len(worlds) != WORLDS:
                    raise CollectorError("sampler returned wrong world count")
            except Exception as exc:
                failures[f"root:{type(exc).__name__}:{exc}"] += WORLDS * len(schedule)
                continue

            root_public_values = []
            merged_worlds = []
            try:
                for sampled, weight in worlds:
                    donor_raw = search_main.battle_to_poke_engine_state(copy.deepcopy(sampled))
                    donor_bits = compile_reveal_bits(donor_raw, ledger, swap=False)
                    donor = install_observer_mask(
                        donor_raw, donor_bits, swap=False, engine=poke_engine
                    )
                    merged = merge_hidden_completion(poke_engine, root_state, donor, ledger)
                    public_root = child_information_state(merged, poke_engine)
                    root_public_values.append(canonical_bytes(public_root))
                    merged_worlds.append((merged, float(weight)))
                if len(set(root_public_values)) != 1:
                    raise CollectorError("hidden worlds changed sanitized root public bytes")
                public_root_by_id[root["capture_sha256"]] = hashlib.sha256(root_public_values[0]).hexdigest()
            except Exception as exc:
                failures[f"merge:{type(exc).__name__}:{exc}"] += WORLDS * len(schedule)
                continue

            for world_index, (world, world_weight) in enumerate(merged_worlds):
                for left, right, uniform in schedule:
                    target_id = hashlib.sha256(
                        json.dumps(
                            [root["capture_sha256"], world_index, left, right, uniform],
                            separators=(",", ":"),
                        ).encode("ascii")
                    ).hexdigest()
                    try:
                        root_string = world.to_string()
                        root_public = canonical_bytes(child_information_state(world, poke_engine))
                        first = poke_engine.step_with_uniform_r1_semantic(world, left, right, uniform)
                        second = poke_engine.step_with_uniform_r1_semantic(world, left, right, uniform)
                        if first.state.to_string() != second.state.to_string():
                            raise CollectorError("semantic step state nondeterminism")
                        if instruction_signature(first.selected_instructions) != instruction_signature(second.selected_instructions):
                            raise CollectorError("semantic instruction nondeterminism")
                        restored = first.state.reverse_instructions(first.selected_instructions)
                        if restored.to_string() != root_string:
                            raise CollectorError("semantic reverse mismatch")
                        if canonical_bytes(child_information_state(restored, poke_engine)) != root_public:
                            raise CollectorError("public reverse mismatch")
                        sidecar = child_reveal_sidecar(world, first, ledger)
                        public = child_information_state(first.state, poke_engine)
                        public_bytes = canonical_bytes(public)
                        perturbation = hidden_perturbation(first.state, poke_engine)
                        if perturbation is not None and canonical_bytes(
                            child_information_state(perturbation, poke_engine)
                        ) != public_bytes:
                            raise CollectorError("hidden completion changed child information state")
                        schedule_supported += 1
                        terminal = float(poke_engine.terminal_value(first.state)) != 0.0
                        side_one, side_two = map(list, poke_engine.root_options(first.state))
                        automatic = (
                            not side_one or not side_two
                            or any(norm(action) == "nomove" for action in [*side_one, *side_two])
                        )
                        if terminal or automatic:
                            terminal_or_automatic += 1
                            continue
                        eligible += 1
                        teacher_seed = stable_u64(
                            BASE_SEED, root["capture_sha256"], world_index,
                            left, right, uniform, "teacher",
                        ) % (2**32)
                        tasks.append({
                            "target_id": target_id,
                            "state": first.state.to_string(),
                            "teacher_seed": teacher_seed,
                            "repeat_canary": False,
                            "split": split,
                            "battle_tag": battle_tag,
                            "capture_sha256": root["capture_sha256"],
                            "world_index": world_index,
                            "world_weight": world_weight,
                            "branch_probability": float(first.branch_probability),
                            "weight": world_weight * float(first.branch_probability) * 0.5,
                            "schedule": {"side_one": left, "side_two": right, "uniform": uniform},
                            "public_state": public,
                            "public_fingerprint": public_fingerprint(public),
                            "legal_actions": side_one,
                            "child_sidecar": sidecar,
                        })
                    except Exception as exc:
                        failures[f"child:{type(exc).__name__}:{exc}"] += 1
    finally:
        os.chdir(previous_cwd)

    for task in sorted(tasks, key=lambda row: row["target_id"])[:16]:
        task["repeat_canary"] = True
    teacher_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        teacher_rows = list(executor.map(_teacher_task, tasks, chunksize=1))
    teacher_wall_seconds = time.perf_counter() - teacher_started
    teacher_by_id = {row["target_id"]: row for row in teacher_rows}
    members = []
    teacher_failures: Counter[str] = Counter()
    for task in tasks:
        result = teacher_by_id[task["target_id"]]
        if result["error"] is not None:
            teacher_failures[result["error"]] += 1
            continue
        members.append({
            key: value for key, value in task.items()
            if key not in {"state", "repeat_canary", "teacher_seed"}
        } | {
            "teacher": result["teacher"],
            "teacher_seed": task["teacher_seed"],
            "repeat_canary": result["repeat_canary"],
            "repeat_exact": result["repeat_exact"],
            "teacher_elapsed_seconds": result["elapsed_seconds"],
        })
    groups = group_target_members(members)
    split_battles = {
        split: sorted(battle for battle, assigned in split_by_battle.items() if assigned == split)
        for split in SPLIT_COUNTS
    }
    if any(
        set(split_battles[left]) & set(split_battles[right])
        for left in split_battles for right in split_battles if left < right
    ):
        raise CollectorError("battle split overlap")
    unique_by_split = Counter(group["split"] for group in groups)
    repeat_rows = [row for row in teacher_rows if row["repeat_canary"]]
    schedule_coverage = schedule_supported / scheduled
    target_coverage = len(members) / eligible if eligible else 0.0
    unique_gate = (
        len(groups) >= 160
        and unique_by_split["train"] >= 80
        and unique_by_split["validation"] >= 20
        and unique_by_split["test"] >= 20
    )
    finite_aggregates = all(
        math.isfinite(float(group["effective_sample_size"]))
        and float(group["effective_sample_size"]) > 0
        and math.isclose(math.fsum(group["visit_policy"].values()), 1.0, abs_tol=1e-9)
        for group in groups
    )
    semantic_leakage_tokens = (
        "placeholder", "certified", "reveal bits", "hidden completion",
        "consumed-item", "mapping", "ledger masks", "public bytes",
    )
    zero_leakage_failures = not any(
        any(token in key.lower() for token in semantic_leakage_tokens)
        for key in failures
    )
    gates = {
        "zero_leakage_spurious_placeholders_split_overlap": zero_leakage_failures,
        "deterministic_repeat": len(repeat_rows) == 16 and all(row["repeat_exact"] for row in repeat_rows),
        "schedule_support_ge_0.95": schedule_coverage >= 0.95,
        "eligible_teacher_support_ge_0.95": target_coverage >= 0.95,
        "unique_public_states": unique_gate,
        "finite_normalized_aggregates_positive_ess": finite_aggregates,
    }
    status = "pass" if all(gates.values()) else "fail"
    write_jsonl(member_path, members)
    write_jsonl(group_path, groups)
    total_wall_seconds = time.perf_counter() - collection_started
    searches = len(members) + sum(bool(row["repeat_canary"]) for row in teacher_rows if row["error"] is None)
    projected_searches = 200 * WORLDS * 8
    report = {
        "schema": "metagross-cycle7-causal-child-collector-mechanics/v1",
        "status": status,
        "gates": gates,
        "configuration": {
            "root_cap": ROOT_CAP, "minimum_roots": MIN_ROOTS,
            "worlds": WORLDS, "children_per_world": 8,
            "teacher_iterations": ITERATIONS, "teacher_workers": WORKERS,
            "split_battle_counts": SPLIT_COUNTS,
        },
        "counts": {
            "source_rows": len(rows), "corrected_selected_roots": len(selected),
            "physical_battles": len(battles), "collector_roots": len(selected_roots),
            "scheduled_children": scheduled, "schedule_supported": schedule_supported,
            "schedule_coverage": schedule_coverage,
            "terminal_or_automatic": terminal_or_automatic,
            "eligible_children": eligible, "teacher_targets": len(members),
            "teacher_target_coverage": target_coverage,
            "unique_public_states": len(groups),
            "unique_by_split": dict(unique_by_split),
            "argmax_disagreement_groups": sum(group["hidden_world_argmax_disagreement"] for group in groups),
            "repeat_canaries": len(repeat_rows),
        },
        "failures": dict(failures.most_common()),
        "teacher_failures": dict(teacher_failures.most_common()),
        "selection_rejections": dict(selection_rejections.most_common()),
        "split_battles": split_battles,
        "root_public_fingerprints": public_root_by_id,
        "ess": {
            "min": min((group["effective_sample_size"] for group in groups), default=0.0),
            "mean": math.fsum(group["effective_sample_size"] for group in groups) / len(groups) if groups else 0.0,
            "max": max((group["effective_sample_size"] for group in groups), default=0.0),
        },
        "throughput": {
            "teacher_wall_seconds": teacher_wall_seconds,
            "total_wall_seconds": total_wall_seconds,
            "searches_including_repeats": searches,
            "searches_per_second": searches / teacher_wall_seconds if teacher_wall_seconds else 0.0,
            "supported_children_per_second": schedule_supported / total_wall_seconds if total_wall_seconds else 0.0,
            "member_bytes": member_path.stat().st_size,
            "group_bytes": group_path.stat().st_size,
            "projected_200_root_searches": projected_searches,
            "projected_200_root_cpu_wall_seconds_same_8_workers": projected_searches / (searches / teacher_wall_seconds) if searches else None,
            "projected_200_root_member_bytes": int(member_path.stat().st_size * 200 / len(selected_roots)),
            "projected_200_root_group_bytes": int(group_path.stat().st_size * 200 / len(selected_roots)),
        },
        "target_semantics": {
            "full_legal_visits": True, "full_legal_q": True,
            "completed_q_available": False,
            "same_teacher_one_hot_control_derivable": True,
            "control_independent": False,
            "python_child_semantic_sidecar": True,
            "rust_interior_sidecar_updates_proven": False,
            "full_causal_history_in_public_fingerprint": False,
            "joint_action_chance_schedule_is_posterior_reach_weighted": False,
            "aggregates_authorized_as_learnable_targets": False,
        },
        "methodological_limitations": [
            "The frozen public fingerprint excludes full causal history and the child event sidecar, so distinct revealed-item/cause histories or posteriors can collide.",
            "The fixed equal joint-action/uniform schedule is a mechanics probe, not a posterior reach distribution; cross-path aggregates are schedule averages.",
            "Every raw path/world row is preserved, but neither raw nor aggregate rows are authorized for model training in Cycle 7.",
        ],
        "authorization": {
            "freeze_mechanics_collection_plan": status == "pass",
            "learnable_target_collection": False,
            "training": False, "h2h": False, "deployment": False,
            "sealed_confirmation": False,
        },
        "artifacts": {
            "members": member_path.name, "members_sha256": sha256(member_path),
            "groups": group_path.name, "groups_sha256": sha256(group_path),
        },
        "hashes": {
            "protocol_sha256": frozen["protocol_sha256"],
            "frozen_inputs_sha256": sha256(run_dir / "FROZEN_INPUTS.json"),
            "engine_binding_sha256": sha256(Path(poke_engine.poke_engine.__file__)),
        },
        "sealed_confirmation_panel_rows_read": 0,
        "local_cpu_only": True, "paid_compute_usd": 0,
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
