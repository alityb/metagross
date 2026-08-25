#!/usr/bin/env python3
"""Audit Cycle 13 TRAIN-only replay-to-production-search mechanics without labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experimental.src.scripts import cycle12_replay_audit as v12
from experimental.src.scripts import run_cycle10_full_corpus_index as v10
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest
from experimental.src.search.public_search_state_v1 import canonical_action_table, canonical_bytes, extract_public_search_state
from srcs.metagross.causal_reveal_ledger import (
    LEDGER_ATTRIBUTE,
    canonical_species,
    compile_reveal_bits,
    convert_battle_with_causal_ledger,
    freeze_ledger,
    install_observer_mask,
    norm,
    serialization_without_masks,
    verify_sampled_ledgers,
)


ROOT = Path(__file__).resolve().parents[3]
WORLD_COUNT = 8
SCHEDULE_COUNT = 2
BASE_SEED = 2026081513


class Cycle13Error(RuntimeError):
    pass


def request_actions_exact(request: Mapping[str, Any]) -> set[str]:
    """Frozen own-private 13-action semantics without public-form aliases."""
    # Old pinned Showdown commits can legitimately omit ``rqid``.  The command
    # linkage is frozen separately by Cycle 12's request/input indices, so rqid
    # is not part of action legality and must not be fabricated here.
    forced_rows = request.get("forceSwitch", [False])
    if not isinstance(forced_rows, list) or not forced_rows or not isinstance(forced_rows[0], bool):
        raise Cycle13Error("invalid private request forceSwitch")
    forced = forced_rows[0]
    active_rows = request.get("active") or []
    if not isinstance(active_rows, list):
        raise Cycle13Error("invalid private request active rows")
    active = active_rows[0] if active_rows else {}
    if not isinstance(active, Mapping):
        raise Cycle13Error("invalid private request active row")
    trapped = active.get("trapped", False)
    if not isinstance(trapped, bool):
        raise Cycle13Error("invalid private request trapped flag")
    can_tera = bool(active.get("canTerastallize", False))
    actions: set[str] = set()
    if not forced:
        moves = active.get("moves", [])
        if not isinstance(moves, list):
            raise Cycle13Error("invalid private request moves")
        for move in moves:
            if not isinstance(move, Mapping):
                raise Cycle13Error("invalid private request move")
            move_id = norm(move.get("id", ""))
            if move_id and not move.get("disabled", False) and move.get("pp") != 0:
                actions.add(move_id)
                if can_tera:
                    actions.add(move_id + "-tera")
    side = request.get("side")
    party = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(party, list):
        raise Cycle13Error("invalid private request party")
    if forced or not trapped:
        for pokemon in party:
            if not isinstance(pokemon, Mapping):
                raise Cycle13Error("invalid private request Pokemon")
            if pokemon.get("active") is True:
                continue
            condition, details = pokemon.get("condition"), pokemon.get("details")
            if not isinstance(condition, str) or not isinstance(details, str):
                raise Cycle13Error("invalid private request Pokemon fields")
            hp = condition.split(" ", 1)[0].split("/", 1)[0]
            if condition.endswith(" fnt") or hp == "0":
                continue
            actions.add("switch " + norm(details.split(",", 1)[0]))
    if not actions:
        raise Cycle13Error("private request has no legal action")
    return actions


def stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def verify_engine_contract(engine: Any) -> None:
    """Require the combined Gen9-Tera and Cycle6 native-mask ABI."""
    first = engine.Pokemon(
        id="pikachu", level=80, moves=[engine.Move(id="thunderbolt")],
        tera_type="electric",
    )
    reserve = engine.Pokemon(
        id="eevee", level=80, moves=[engine.Move(id="tackle")],
        tera_type="normal",
    )
    state = engine.State(
        side_one=engine.Side(pokemon=[first, reserve]),
        side_two=engine.Side(pokemon=[first, reserve]),
    )
    side_one, _ = engine.root_options(state)
    names = {str(action) for action in side_one}
    if "thunderbolt-tera" not in names:
        raise Cycle13Error("engine contract lacks Gen9 Tera actions")
    if not all(hasattr(state, name) for name in (
        "with_side_one_public_reveals", "with_side_two_public_reveals",
    )):
        raise Cycle13Error("engine contract lacks symmetric Cycle6 native masks")
    masked = state.with_side_one_public_reveals(1).with_side_two_public_reveals(2)
    if int(masked.s1_public_reveals) != 1 or int(masked.s2_public_reveals) != 2:
        raise Cycle13Error("engine native mask contract failed preflight")
    if not hasattr(engine, "step_with_uniform_r1_semantic"):
        raise Cycle13Error("engine contract lacks seeded semantic stepping")


def failure_category(phase: str, exc: BaseException) -> str:
    detail = str(exc).lower()
    if phase == "capture":
        return "source_replay"
    if phase == "compact_parity":
        return "frozen_provenance_or_ledger"
    if phase == "foul_play_rehydration":
        return "causal_rehydration"
    if any(token in detail for token in (
        "public species mapping", "public move/pp", "public item mismatch",
        "public ability mismatch", "causal public fact", "causal public move",
        "reveal mask", "sampler changed causal ledger", "spurious bits",
    )):
        return "causal_fact_integrity"
    if "public projection" in detail or "hidden completion" in detail:
        return "hidden_noninterference"
    if "action" in detail or "root options" in detail:
        return "action_mapping"
    if "apply/reverse" in detail or "non-mask byte" in detail:
        return "state_byte_parity"
    if "nondetermin" in detail or "repeat disagreed" in detail:
        return "determinism"
    if "sampler" in detail or "world count" in detail:
        return "production_sampler"
    return "internal_unclassified"


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    rows = sorted(values)
    index = (len(rows) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return float(rows[lo])
    return float(rows[lo] + (rows[hi] - rows[lo]) * (index - lo))


def prefix_delta(previous: Sequence[str], current: Sequence[str]) -> list[str]:
    if list(current[:len(previous)]) != list(previous):
        raise Cycle13Error("causal public prefix regressed during Foul Play replay")
    return list(current[len(previous):])


def remove_initial_own_switch(lines: Sequence[str], role: str) -> list[str]:
    result = []
    removed = False
    for line in lines:
        fields = line.split("|")
        if (
            not removed and len(fields) >= 3 and fields[1] in {"switch", "drag"}
            and fields[2].startswith(role)
        ):
            removed = True
            continue
        result.append(line)
    if not removed:
        raise Cycle13Error("initial own switch is absent from causal prefix")
    return result


def _request_flags(battle: Any, request: Mapping[str, Any]) -> None:
    battle.request_json = copy.deepcopy(request)
    battle.rqid = request.get("rqid")
    battle.force_switch = bool(request.get("forceSwitch"))
    battle.wait = bool(request.get("wait", False))
    battle.user.update_from_request_json(request)


def build_foul_play_battle(
    *, battle_id: str, role: str, states: Sequence[Mapping[str, Any]], target_index: int,
) -> Any:
    from constants import BattleType
    from fp.battle import Battle, LastUsedMove
    from fp.battle_modifier import process_battle_updates

    if not 0 <= target_index < len(states):
        raise Cycle13Error("target request index is outside rehydrated states")
    first_request = states[0]["private_request"]
    battle = Battle(battle_id)
    battle.battle_type = BattleType.RANDOM_BATTLE
    battle.pokemon_format = "gen9randombattle"
    battle.generation = "gen9"
    battle.user.name = role
    battle.opponent.name = "p2" if role == "p1" else "p1"
    battle.user.initialize_first_turn_user_from_json(first_request)
    battle.started = True
    previous: list[str] = []
    for index, state in enumerate(states[:target_index + 1]):
        current = list(state["public_prefix"])
        delta = prefix_delta(previous, current)
        if index == 0:
            delta = remove_initial_own_switch(delta, role)
        battle.msg_list.extend(delta)
        process_battle_updates(battle)
        _request_flags(battle, state["private_request"])
        # The live bot records its selected action after receiving this request.
        # Reproduce that causal private fact only when replaying onward to a
        # later decision.  Never install the target decision's observed command
        # into the target root.
        if index < target_index:
            chosen = state.get("chosen_action")
            if isinstance(chosen, str) and chosen:
                battle.user.last_selected_move = LastUsedMove(
                    battle.user.active.name,
                    chosen.removesuffix("-tera").removesuffix("-mega"),
                    battle.turn,
                )
            elif state.get("actionable") is False:
                # A genuine ``wait: true`` request advances only through the
                # opponent/chance observation; there is no own command to add.
                pass
            elif list(states[index + 1]["public_prefix"]) != current:
                # A request can be superseded before any command while the
                # public prefix remains unchanged.  It cannot causally explain
                # a later public transition without an observed command.
                raise Cycle13Error("intermediate public transition lacks its observed command")
        previous = current
    if battle.user.active is None or battle.opponent.active is None:
        raise Cycle13Error("rehydrated Foul Play battle lacks an active Pokemon")
    return battle


def reconcile_public_facts_into_battle(battle: Any, ledger: Any) -> None:
    """Install event-certified opponent facts before production sampling.

    This is not hidden-team hydration: every installed value is carried by the
    immutable public ledger.  Revealed moves must already have causal PP state
    in Foul Play; we fail closed rather than inventing PP/disable metadata.
    """
    roster = [battle.opponent.active, *battle.opponent.reserve]
    slots: dict[str, Any] = {}
    ambiguous: set[str] = set()
    for pokemon in roster:
        species = canonical_species(getattr(pokemon, "name", ""))
        if not species:
            continue
        if species in slots:
            ambiguous.add(species)
        slots[species] = pokemon
    for species in ambiguous:
        slots.pop(species, None)
    for fact in ledger.facts:
        pokemon = slots.get(canonical_species(fact.species))
        if pokemon is None:
            raise Cycle13Error("causal public fact does not map uniquely into Foul Play")
        known_moves = [norm(getattr(move, "name", "")) for move in pokemon.moves]
        if any(known_moves.count(move) != 1 for move in fact.moves):
            raise Cycle13Error("causal public move lacks exact Foul Play PP state")
        if fact.item_status_revealed:
            expected = None if fact.current_item in {None, "none"} else fact.current_item
            pokemon.item = expected
            if fact.consumed_items:
                if len(fact.consumed_items) != 1:
                    raise Cycle13Error("multiple consumed items lack a root sidecar contract")
                pokemon.removed_item = fact.consumed_items[0]
        if fact.ability is not None:
            pokemon.ability = fact.ability


def _compact_state(derived: Mapping[str, Any], pov: Mapping[str, Any], request_index: int) -> dict[str, Any]:
    rows = v10.compact_states(derived, pov)
    matches = [row for row in rows if row["request_index"] == request_index]
    if len(matches) != 1:
        raise Cycle13Error("selected request does not map to one compact state")
    return matches[0]


def _state_parity(selected: Mapping[str, Any], compact: Mapping[str, Any]) -> None:
    for key in (
        "role", "request_index", "command_input_index", "public_event_index",
        "private_request_sha256",
        "causal_prefix_sha256", "legal_action_contract_sha256",
        "pp_disable_sidecar_sha256",
    ):
        if selected[key] != compact[key]:
            raise Cycle13Error(f"Cycle12 compact parity mismatch:{key}")


def _world_payload(
    sampled: Any, ledger: Any, derived_actions: set[str], engine: Any, search_main: Any,
) -> tuple[dict[str, Any], float, float]:
    started = time.perf_counter()
    raw = search_main.battle_to_poke_engine_state(copy.deepcopy(sampled), swap=False)
    before = serialization_without_masks(raw)
    bits = compile_reveal_bits(raw, ledger, swap=False)
    state = install_observer_mask(raw, bits, swap=False, engine=engine)
    wrapped = convert_battle_with_causal_ledger(
        copy.deepcopy(sampled), search_main.battle_to_poke_engine_state,
        engine, swap=False,
    )
    conversion_ms = (time.perf_counter() - started) * 1000.0
    if state.to_string() != wrapped.to_string():
        raise Cycle13Error("production converter disagrees with native mask installation")
    if serialization_without_masks(state) != before:
        raise Cycle13Error("observer mask changed a non-mask byte")
    if int(state.s1_public_reveals) != bits or int(state.s2_public_reveals) != 0:
        raise Cycle13Error("observer reveal mask has missing or spurious bits")

    raw_swap = search_main.battle_to_poke_engine_state(copy.deepcopy(sampled), swap=True)
    swap_before = serialization_without_masks(raw_swap)
    swap_bits = compile_reveal_bits(raw_swap, ledger, swap=True)
    swapped = install_observer_mask(raw_swap, swap_bits, swap=True, engine=engine)
    if serialization_without_masks(swapped) != swap_before:
        raise Cycle13Error("swapped reveal mask changed a non-mask byte")
    if int(swapped.s1_public_reveals) != 0 or int(swapped.s2_public_reveals) != swap_bits:
        raise Cycle13Error("swapped reveal mask has missing or spurious bits")

    side_one_raw, side_two = map(lambda rows: [str(row) for row in rows], engine.root_options(state))
    # The frozen Cycle 6 binding predates request-level Tera availability.
    # Production therefore intersects engine roots with the authoritative live
    # Showdown request before selection.  Audit that exact production contract:
    # every request action must be mechanically supported, while engine-only
    # Tera variants are not legal roots for this observer.
    side_one = [action for action in side_one_raw if action in derived_actions]
    if set(side_one) != derived_actions:
        raise Cycle13Error("exact request actions disagree with engine root options")
    action_table = canonical_action_table(side_one)
    if set(action_table["name_table"]) != derived_actions:
        raise Cycle13Error("canonical 13-action mapping lost an exact request action")
    if not side_one or not side_two or any(str(row).lower() == "nomove" for row in [*side_one, *side_two]):
        raise Cycle13Error("selected actionable root is terminal or automatic in engine")

    root_string = state.to_string()
    first = engine.step_with_uniform_r1_semantic(state, side_one[0], side_two[0], 0.25)
    second = engine.step_with_uniform_r1_semantic(state, side_one[0], side_two[0], 0.25)
    signature = lambda step: (
        step.state.to_string(), float(step.selected_instructions.percentage),
        tuple(str(row) for row in step.selected_instructions.instruction_list),
    )
    if signature(first) != signature(second):
        raise Cycle13Error("seeded semantic step is nondeterministic")
    restored = first.state.reverse_instructions(first.selected_instructions)
    if restored.to_string() != root_string:
        raise Cycle13Error("semantic apply/reverse byte parity failed")
    public = extract_public_search_state(state, engine, observer="side_one")
    public_bytes = canonical_bytes(public)
    payload = {
        "state_sha256": hashlib.sha256(root_string.encode("utf-8")).hexdigest(),
        "public_sha256": hashlib.sha256(public_bytes).hexdigest(),
        "observer_mask": bits, "swapped_mask": swap_bits,
        "legal_action_contract_sha256": hash_json(action_table),
        "step_sha256": hashlib.sha256(json.dumps(signature(first), default=str).encode("utf-8")).hexdigest(),
    }
    return payload, conversion_ms, len(public_bytes)


def process_root(
    selected: Mapping[str, Any], *, worktree: str, harness: str, output_root: Path,
    engine: Any, search_main: Any,
) -> dict[str, Any]:
    root_id = selected["model_information_fingerprint_sha256"][:16]
    tmp_parent = output_root / ".tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=root_id + "-", dir=tmp_parent))
    capture = temp / "capture"
    phase = "capture"
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    try:
        if sha256(Path(selected["raw_path"])) != selected["raw_sha256"]:
            raise Cycle13Error("selected raw replay hash changed")
        subprocess.run([
            "node", harness, "--showdown", worktree,
            "--input", selected["raw_path"], "--out-dir", str(capture),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        raw = json.loads(Path(selected["raw_path"]).read_text())
        public = json.loads((capture / "public.json").read_text())
        pov = json.loads((capture / f"{selected['role']}.json").read_text())
        phase = "compact_parity"
        derived = v12.materialize_role(
            battle_id=selected["battle_id"], role=selected["role"],
            public_capture=public, pov_capture=pov, inputlog=raw["inputlog"],
            showdown_commit=selected["showdown_commit"],
        )
        compact = _compact_state(derived, pov, selected["request_index"])
        _state_parity(selected, compact)
        target = derived["states"][selected["request_index"]]
        if not target["actionable"] or target["pp_disable_sidecar"].get("revival_prompt"):
            raise Cycle13Error("selected state is nonordinary actionable semantics")
        derived_actions = request_actions_exact(target["private_request"])
        if derived_actions != set(target["legal_actions"]):
            raise Cycle13Error("exact request action extractor disagrees with frozen bridge")
        ledger = freeze_ledger(selected["battle_id"], selected["role"], target["public_prefix"])
        corrected_ledger_sha256 = hash_json(ledger.to_payload())
        if corrected_ledger_sha256 != compact["typed_reveal_ledger_sha256"]:
            raise Cycle13Error("typed causal ledger disagrees with corrected rematerialization")
        corrected_information_sha256 = compact["model_information_fingerprint_sha256"]

        phase = "foul_play_rehydration"
        battle = build_foul_play_battle(
            battle_id=selected["battle_id"], role=selected["role"],
            states=derived["states"], target_index=selected["request_index"],
        )
        reconcile_public_facts_into_battle(battle, ledger)
        setattr(battle, LEDGER_ATTRIBUTE, ledger.to_payload())

        phase = "production_schedules"
        schedule_rows = []
        conversion_ms: list[float] = []
        sampler_ms: list[float] = []
        public_sizes: list[float] = []
        for schedule_index in range(SCHEDULE_COUNT):
            seed = stable_seed(BASE_SEED, corrected_information_sha256, schedule_index)
            schedule_payloads = []
            for repeat in range(2):
                sampler_started = time.perf_counter()
                worlds = search_main.prepare_random_battles(
                    copy.deepcopy(battle), WORLD_COUNT, rng=random.Random(seed),
                )
                sampler_ms.append((time.perf_counter() - sampler_started) * 1000.0)
                if len(worlds) != WORLD_COUNT:
                    raise Cycle13Error("production sampler returned wrong world count")
                verify_sampled_ledgers(battle, worlds)
                payloads = []
                weights = []
                for sampled, weight in worlds:
                    payload, elapsed_ms, public_size = _world_payload(
                        sampled, ledger, derived_actions, engine, search_main,
                    )
                    payloads.append(payload)
                    weights.append(float(weight))
                    conversion_ms.append(elapsed_ms)
                    public_sizes.append(public_size)
                schedule_payloads.append({"worlds": payloads, "weights": weights})
            if schedule_payloads[0] != schedule_payloads[1]:
                raise Cycle13Error("seeded production schedule repeat disagreed")
            public_hashes = {row["public_sha256"] for row in schedule_payloads[0]["worlds"]}
            if len(public_hashes) != 1:
                raise Cycle13Error("hidden completions changed observer public projection")
            schedule_rows.append({
                "schedule_index": schedule_index,
                "seed": seed,
                "schedule_sha256": hash_json(schedule_payloads[0]),
                "public_sha256": next(iter(public_hashes)),
                "world_count": WORLD_COUNT,
                "production_weights_preserved_but_not_interpreted_as_posterior": True,
            })
        if len({row["public_sha256"] for row in schedule_rows}) != 1:
            raise Cycle13Error("independent schedules changed observer public projection")
        result = {
            "schema": "metagross-cycle13-root-mechanics/v1", "status": "pass",
            "dependency_cluster_id": selected["dependency_cluster_id"],
            "model_information_fingerprint_sha256": corrected_information_sha256,
            "cycle12_legacy_model_information_fingerprint_sha256": selected[
                "model_information_fingerprint_sha256"
            ],
            "role": selected["role"], "request_index": selected["request_index"],
            "showdown_commit": selected["showdown_commit"],
            "full_causal_observation_identity": {
                key: selected[key] for key in (
                    "private_request_sha256", "causal_prefix_sha256",
                    "legal_action_contract_sha256",
                    "public_event_index", "command_input_index",
                )
            },
            "schedules": schedule_rows,
            "sampler_ms": sampler_ms, "conversion_ms": conversion_ms,
            "public_projection_bytes": public_sizes,
            "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
            "cpu_ms": (time.process_time() - cpu_started) * 1000.0,
            "teacher_values_opened": 0,
        }
        result["full_causal_observation_identity"].update({
            "typed_reveal_ledger_sha256": corrected_ledger_sha256,
            "cycle12_legacy_typed_reveal_ledger_sha256": selected[
                "typed_reveal_ledger_sha256"
            ],
        })
        shutil.rmtree(temp)
        return result
    except BaseException as exc:
        failure = output_root / "failures" / root_id
        failure.parent.mkdir(parents=True, exist_ok=True)
        if failure.exists():
            shutil.rmtree(failure)
        shutil.move(str(temp), failure)
        detail = f"{type(exc).__name__}:{exc}"
        row = {
            "schema": "metagross-cycle13-root-mechanics/v1", "status": "fail",
            "dependency_cluster_id": selected["dependency_cluster_id"],
            "model_information_fingerprint_sha256": selected["model_information_fingerprint_sha256"],
            "phase": phase, "failure_class": type(exc).__name__,
            "failure_category": failure_category(phase, exc),
            "failure_detail_sha256": hashlib.sha256(detail.encode("utf-8")).hexdigest(),
            "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
            "cpu_ms": (time.process_time() - cpu_started) * 1000.0,
            "teacher_values_opened": 0,
        }
        (failure / "FAILURE.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_root = run_dir / "mechanics-audit"
    if output_root.exists():
        raise Cycle13Error("Cycle 13 output already exists")
    output_root.mkdir()
    manifest = verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    selected = [json.loads(line) for line in (run_dir / "selection-200.jsonl").read_text().splitlines() if line]
    if len(selected) != 200 or any(row["split"] != "train" for row in selected):
        raise Cycle13Error("frozen Cycle 13 selection changed")
    if len({row["dependency_cluster_id"] for row in selected}) != 200:
        raise Cycle13Error("Cycle 13 selection is not cluster unique")
    worktrees = json.loads((ROOT / "experimental/runs/search_native_v2_cycle8_observed_transitions_20260815/replay-worktrees.json").read_text())
    import poke_engine
    engine = poke_engine
    binding_path = Path(engine.poke_engine.__file__).resolve()
    expected_binding = Path(manifest["engine_binding_path"]).resolve()
    if binding_path != expected_binding:
        raise Cycle13Error("wrong Cycle 13 engine binding imported")
    if sha256(binding_path) != manifest["engine_binding_sha256"]:
        raise Cycle13Error("Cycle 6 engine binding hash changed")
    verify_engine_contract(engine)

    foul_play = ROOT / "srcs/vendor/foul-play"
    sys.path.insert(0, str(foul_play))
    previous_cwd = Path.cwd()
    os.chdir(foul_play)
    try:
        from data.pkmn_sets import RandomBattleTeamDatasets
        from fp.search import main as search_main
        RandomBattleTeamDatasets.initialize("gen9")
        rows = []
        for index, row in enumerate(selected, start=1):
            rows.append(process_root(
                row, worktree=worktrees[row["showdown_commit"]],
                harness=str((ROOT / "experimental/src/scripts/replay_cycle8_inputlog.cjs").resolve()),
                output_root=output_root, engine=engine, search_main=search_main,
            ))
            if index % 20 == 0:
                print(json.dumps({
                    "completed": index, "total": len(selected),
                    "passed": sum(item["status"] == "pass" for item in rows),
                    "failed": sum(item["status"] == "fail" for item in rows),
                }, sort_keys=True), flush=True)
    finally:
        os.chdir(previous_cwd)
    with (output_root / "root-results.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    passed = [row for row in rows if row["status"] == "pass"]
    failed = [row for row in rows if row["status"] == "fail"]
    support = len(passed) / len(rows)
    failures = Counter(
        f"{row['phase']}:{row['failure_category']}:{row['failure_class']}"
        for row in failed
    )
    all_sampler = [value for row in passed for value in row["sampler_ms"]]
    all_conversion = [value for row in passed for value in row["conversion_ms"]]
    all_wall = [row["wall_ms"] for row in rows]
    gates = {
        "train_only_cluster_unique_selection": True,
        "end_to_end_support_ge_0_95": support >= 0.95,
        "zero_hidden_or_fabricated_fact_failure": not any(
            row["failure_category"] in {
                "causal_fact_integrity", "hidden_noninterference",
            }
            for row in failed
        ),
        "exact_action_mask_and_byte_parity_for_every_admitted_root": True,
        "deterministic_repeat_for_every_admitted_root": all(
            len(row["schedules"]) == 2 for row in passed
        ),
        "validation_dev_test_teacher_labels_opened_zero": True,
        "teacher_values_opened_zero": True,
    }
    post_integrity = "pass"
    post_detail = None
    try:
        verify_manifest(run_dir / "PREMEASUREMENT_MANIFEST.json")
    except BaseException as exc:
        post_integrity = "fail"
        post_detail = hashlib.sha256(f"{type(exc).__name__}:{exc}".encode()).hexdigest()
    gates["post_run_frozen_integrity"] = post_integrity == "pass"
    status = "pass" if all(gates.values()) else "fail"
    latency = lambda values: {
        "count": len(values), "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.5), "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99), "max": max(values, default=0.0),
    }
    report = {
        "schema": "metagross-cycle13-train-rehydration-report/v1",
        "status": status, "gates": gates,
        "counts": {
            "selected": len(rows), "passed": len(passed), "failed": len(failed),
            "support": support, "dependency_clusters": len({row["dependency_cluster_id"] for row in rows}),
            "production_schedules_per_root": SCHEDULE_COUNT,
            "worlds_per_schedule": WORLD_COUNT,
            "world_conversions_including_exact_repeats": len(passed) * SCHEDULE_COUNT * WORLD_COUNT * 2,
        },
        "failures": dict(failures.most_common()),
        "latency_ms": {
            "root_end_to_end": latency(all_wall),
            "eight_world_sampler_call": latency(all_sampler),
            "world_conversion_mask_projection_step": latency(all_conversion),
        },
        "methodology": {
            "one_state_per_dependency_cluster": True,
            "split": "train", "validation_index_files_opened": 0,
            "dev_test_index_files_opened": 0,
            "full_causal_observation_identity_preserved": True,
            "cross_path_or_world_aggregation": False,
            "schedules_called_posterior_weights": False,
            "teacher_q_visit_outcome_fields_opened": 0,
        },
        "authorization": {
            "cycle14_teacher_budget_stability_protocol": status == "pass",
            "teacher_execution": False, "training": False, "h2h": False,
            "sealed_confirmation": False,
        },
        "hashes": {
            "manifest_sha256": sha256(run_dir / "PREMEASUREMENT_MANIFEST.json"),
            "selection_sha256": sha256(run_dir / "selection-200.jsonl"),
            "root_results_sha256": sha256(output_root / "root-results.jsonl"),
            "engine_binding_sha256": sha256(binding_path),
        },
        "post_run_integrity": post_integrity,
        "post_run_integrity_detail_sha256": post_detail,
        "sealed_93_rows_read": 0, "local_cpu_only": True,
        "gpu_cloud_paid_cost_usd": 0,
    }
    report_path = output_root / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
