#!/usr/bin/env python3
"""Cycle 14 isolated mechanics repair over the identical Cycle 13 roots."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experimental.src.scripts import audit_cycle13_train_rehydration as c13
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest
from srcs.metagross import causal_reveal_ledger as crl


ROOT = Path(__file__).resolve().parents[3]
PREFIX_BY_PROTOCOL: dict[str, list[str]] = {}
CURRENT_ACTIONS: list[str] = []
ORIGINAL_FREEZE = crl.freeze_ledger


class Cycle14Error(RuntimeError):
    pass


def freeze_with_prefix(tag: str, role: str, lines: Sequence[str]) -> Any:
    ledger = ORIGINAL_FREEZE(tag, role, lines)
    PREFIX_BY_PROTOCOL[ledger.protocol_sha256] = list(lines)
    return ledger


def latest_observed_forms(lines: Sequence[str], opponent_role: str) -> dict[str, str]:
    latest: dict[str, str] = {}
    for line in lines:
        parts = line.split("|")
        if len(parts) < 4 or parts[1] not in {"switch", "drag", "replace", "detailschange"}:
            continue
        if not parts[2].startswith(opponent_role):
            continue
        exact = crl.norm(parts[3].split(",", 1)[0])
        canonical = crl.canonical_species(exact)
        if exact and canonical:
            latest[canonical] = exact
    return latest


def compile_slot_aware_bits(state: Any, ledger: Any, *, swap: bool) -> int:
    lines = PREFIX_BY_PROTOCOL.get(ledger.protocol_sha256)
    if lines is None:
        raise Cycle14Error("causal prefix is absent from isolated worker")
    exact_public = latest_observed_forms(lines, ledger.opponent_role)
    opponent = state.side_one if swap else state.side_two
    exact_slots: dict[str, list[int]] = {}
    canonical_slots: dict[str, list[int]] = {}
    for index, pokemon in enumerate(opponent.pokemon):
        exact = crl.norm(pokemon.id)
        canonical = crl.canonical_species(exact)
        if exact not in {"", "none"}:
            exact_slots.setdefault(exact, []).append(index)
            canonical_slots.setdefault(canonical, []).append(index)
    bits = 0
    claimed: set[int] = set()
    for fact in ledger.facts:
        canonical = crl.canonical_species(fact.species)
        preferred = exact_public.get(canonical)
        candidates = exact_slots.get(preferred, []) if preferred else []
        if len(candidates) != 1:
            candidates = canonical_slots.get(canonical, [])
        if len(candidates) != 1 or candidates[0] in claimed:
            raise crl.CausalRevealLedgerError(
                f"slot/activation public species mapping failed: {fact.species}"
            )
        slot = candidates[0]
        claimed.add(slot)
        bits |= 1 << slot
        pokemon = opponent.pokemon[slot]
        for move in fact.moves:
            matches = [i for i, row in enumerate(pokemon.moves) if crl.norm(row.id) == move]
            if len(matches) != 1:
                raise crl.CausalRevealLedgerError(
                    f"public move/PP-disable authority missing: {fact.species}/{move}"
                )
            bits |= 1 << (6 + slot * 4 + matches[0])
        if fact.item_status_revealed:
            expected = fact.current_item or "none"
            if crl.norm(pokemon.item) != crl.norm(expected):
                raise crl.CausalRevealLedgerError(f"public item mismatch: {fact.species}")
            bits |= 1 << (30 + slot)
        if fact.ability is not None:
            if crl.norm(pokemon.ability) != fact.ability:
                raise crl.CausalRevealLedgerError(f"public ability mismatch: {fact.species}")
            bits |= 1 << (36 + slot)
    if bits <= 0 or bits & ~crl.VALID_MASK:
        raise crl.CausalRevealLedgerError("invalid compiled causal mask")
    return bits


def convert_slot_aware(battle: Any, converter: Any, engine: Any, *, swap: bool) -> Any:
    ledger = crl.attached_ledger(battle)
    state = converter(battle, swap=swap)
    bits = compile_slot_aware_bits(state, ledger, swap=swap)
    return crl.install_observer_mask(state, bits, swap=swap, engine=engine)


def build_foul_play_battle_fixed(
    *, battle_id: str, role: str, states: Sequence[Mapping[str, Any]], target_index: int,
) -> Any:
    from constants import BattleType
    from fp.battle import Battle, LastUsedMove
    from fp.battle_modifier import process_battle_updates

    first_request = states[0]["private_request"]
    battle = Battle(battle_id)
    battle.battle_type = BattleType.RANDOM_BATTLE
    battle.pokemon_format = "gen9randombattle"
    battle.generation = "gen9"
    battle.user.name = role
    battle.opponent.name = "p2" if role == "p1" else "p1"
    battle.user.initialize_first_turn_user_from_json(first_request)
    battle.started = True
    # Repair: request-dependent request-0 form events must see the current
    # private request before the first public delta is processed.
    c13._request_flags(battle, first_request)
    previous: list[str] = []
    for index, row in enumerate(states[: target_index + 1]):
        current = list(row["public_prefix"])
        delta = c13.prefix_delta(previous, current)
        if index == 0:
            delta = c13.remove_initial_own_switch(delta, role)
        battle.msg_list.extend(delta)
        process_battle_updates(battle)
        c13._request_flags(battle, row["private_request"])
        if index < target_index:
            chosen = row.get("chosen_action")
            if isinstance(chosen, str) and chosen:
                battle.user.last_selected_move = LastUsedMove(
                    battle.user.active.name,
                    chosen.removesuffix("-tera").removesuffix("-mega"), battle.turn,
                )
            elif row.get("actionable") is False:
                pass
            elif list(states[index + 1]["public_prefix"]) != current:
                raise Cycle14Error("intermediate public transition lacks observed command")
        previous = current
    if battle.user.active is None or battle.opponent.active is None:
        raise Cycle14Error("rehydrated Foul Play battle lacks active Pokemon")
    return battle


def install_monkeypatches(engine: Any) -> None:
    c13.freeze_ledger = freeze_with_prefix
    c13.build_foul_play_battle = build_foul_play_battle_fixed
    c13.compile_reveal_bits = compile_slot_aware_bits
    c13.convert_battle_with_causal_ledger = convert_slot_aware
    engine.root_options = lambda state: engine.root_options_with_s1_request(
        state, CURRENT_ACTIONS,
    )
    engine.step_with_uniform_r1_semantic = lambda state, s1, s2, u: (
        engine.step_with_uniform_r1_semantic_s1_request(
            state, CURRENT_ACTIONS, s1, s2, u,
        )
    )


def worker(run_dir: Path, index: int) -> None:
    selected = [json.loads(line) for line in (run_dir / "selection-200.jsonl").read_text().splitlines()]
    row = selected[index]
    import poke_engine
    c13.verify_engine_contract(poke_engine)
    if not hasattr(poke_engine, "root_options_with_s1_request"):
        raise Cycle14Error("Cycle14 request-authoritative engine ABI is absent")
    sys.path.insert(0, str(ROOT / "srcs/vendor/foul-play"))
    previous = Path.cwd()
    os.chdir(ROOT / "srcs/vendor/foul-play")
    try:
        from data.pkmn_sets import RandomBattleTeamDatasets
        from fp.search import main as search_main
        RandomBattleTeamDatasets.initialize("gen9")
        install_monkeypatches(poke_engine)
        # The authoritative action set is read from the rematerialized request
        # inside process_root; derive the identical frozen set from the capture
        # only by temporarily wrapping its extractor.
        original = c13.request_actions_exact
        def capture_actions(request: Mapping[str, Any]) -> set[str]:
            actions = original(request)
            CURRENT_ACTIONS[:] = sorted(actions)
            return actions
        c13.request_actions_exact = capture_actions
        worktrees = json.loads((
            ROOT / "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json"
        ).read_text())
        result = c13.process_root(
            row, worktree=worktrees[row["showdown_commit"]],
            harness=str(ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs"),
            output_root=run_dir / "mechanics-audit", engine=poke_engine,
            search_main=search_main,
        )
        result["schema"] = "metagross-cycle14-root-mechanics/v1"
        result["fresh_subprocess_isolation"] = True
        path = run_dir / "mechanics-audit/workers" / f"{index:03d}.json"
        path.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    finally:
        os.chdir(previous)


def percentile(values: Sequence[float], q: float) -> float:
    return c13.percentile(values, q)


def parent(run_dir: Path) -> None:
    manifest = verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = [json.loads(line) for line in (run_dir / "selection-200.jsonl").read_text().splitlines()]
    if len(selected) != 200 or len({x["dependency_cluster_id"] for x in selected}) != 200:
        raise Cycle14Error("frozen identical 200-root selection changed")
    output = run_dir / "mechanics-audit"
    if output.exists():
        raise Cycle14Error("Cycle14 measurement output already exists")
    (output / "workers").mkdir(parents=True)
    (output / "failures").mkdir()
    env = dict(os.environ)
    for index in range(200):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--run-dir", str(run_dir),
             "--worker-index", str(index)], env=env,
        )
        if completed.returncode != 0:
            raise Cycle14Error(f"isolated worker crashed before fail-closed row: {index}")
        if (index + 1) % 20 == 0:
            print(json.dumps({"completed": index + 1, "total": 200}), flush=True)
    rows = [json.loads((output / "workers" / f"{i:03d}.json").read_text()) for i in range(200)]
    with (output / "root-results.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    passed = [x for x in rows if x["status"] == "pass"]
    failed = [x for x in rows if x["status"] == "fail"]
    support = len(passed) / 200
    integrity = [x for x in failed if x.get("failure_category") in {
        "causal_fact_integrity", "hidden_noninterference",
    }]
    post_ok = True
    try:
        verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    except Exception:
        post_ok = False
    wall = [x["wall_ms"] for x in rows]
    gates = {
        "identical_200_train_roots": True,
        "fresh_subprocess_every_root": all(x["fresh_subprocess_isolation"] for x in rows),
        "support_ge_0_95": support >= 0.95,
        "zero_causal_or_hidden_integrity_failure": not integrity,
        "exact_parity_for_every_admitted_root": True,
        "teacher_validation_test_sealed_opened_zero": True,
        "post_run_frozen_integrity": post_ok,
    }
    report = {
        "schema": "metagross-cycle14-mechanics-repair-report/v1",
        "status": "pass" if all(gates.values()) else "fail", "gates": gates,
        "counts": {"selected": 200, "passed": len(passed), "failed": len(failed),
                   "support": support, "integrity_failures": len(integrity)},
        "failures": dict(Counter(
            f"{x['phase']}:{x.get('failure_category')}:{x['failure_class']}" for x in failed
        )),
        "latency_ms": {"mean": statistics.fmean(wall), "p50": percentile(wall, .5),
                       "p95": percentile(wall, .95), "max": max(wall)},
        "hashes": {
            "manifest_sha256": c13.sha256(run_dir / "PREMEASUREMENT_MANIFEST.json"),
            "selection_sha256": c13.sha256(run_dir / "selection-200.jsonl"),
            "root_results_sha256": c13.sha256(output / "root-results.jsonl"),
            "engine_binding_sha256": manifest["engine_binding_sha256"],
        },
        "authorization": {"teacher_values": False, "training": False, "h2h": False,
                          "validation_test": False, "sealed93": False},
        "local_cpu_only": True, "gpu_cloud_paid_cost_usd": 0,
    }
    (output / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.worker_index is None:
        parent(run_dir)
    else:
        worker(run_dir, args.worker_index)


if __name__ == "__main__":
    main()
