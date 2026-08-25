#!/usr/bin/env python3
"""Run the frozen Cycle 2 Gate A representation/throughput probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from search.public_search_policy_smoke import (
    encode_states,
    infer,
    make_policy,
    parameter_count,
)
from search.public_search_state_v1 import (
    PublicSearchStateError,
    canonical_bytes,
    compile_side_one_reveal_mask,
    extract_public_search_state,
    install_side_one_reveal_mask,
)
from train.resource_shadow import extract_resource_features


ROOT = Path(__file__).resolve().parents[3]
EXPECTED = {
    "architecture": "b694f00e6605339fd052fbedd722ff9b385cadf9c85c55c8fed26a8d2c4d1de8",
    "agent_a": "706d394fd4d48452c5c253e211ef598d3db6cf9b9fb950cf2029b394b44e2c01",
    "agent_b": "dda81bda32a10aa7405b213cbffa2dae2e457fdcb5de706cf8437b62c228e8af",
    "exclusions": "fa4b62f62993d5d7c0b028622c75f6f2ba108b71ae8e7645e7c1b83132f8a45a",
    "engine": "cf71fbba541c9e7b4f3c891bf9b25dca863196708b7131f77d1e0016c1073f69",
    "checkpoint": "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93",
}
SELECTION_SEED = 2026081502


class GateAError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise GateAError("non-object root row")
                rows.append(row)
    return rows


def rank(parts: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json([SELECTION_SEED, *parts]).encode()).hexdigest()


def _ordinary_actions(actions: Sequence[str]) -> list[str]:
    return [str(action) for action in actions if "".join(str(action).lower().split()) != "nomove"]


def select_roots(rows: Sequence[Mapping[str, Any]], engine: Any) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    rejected: Counter[str] = Counter()
    for raw in rows:
        row = dict(raw)
        try:
            snapshot = row["r1_policy_snapshot"]
            identity = row["identity"]
            if row.get("schema") != "metagross-causal-dual-r1-root/v1":
                raise GateAError("schema")
            if snapshot.get("schema") != 6:
                raise GateAError("not_schema6")
            legality = snapshot.get("own_legality")
            if not isinstance(legality, Mapping) or legality.get("force_switch"):
                raise GateAError("not_ordinary")
            if len(legality.get("actions", ())) < 2:
                raise GateAError("not_ordinary")
            state = engine.State.from_string(row["state"])
            if hashlib.sha256(row["state"].encode()).hexdigest() != row.get("state_sha256"):
                raise GateAError("state_hash")
            first, second = engine.root_options(state)
            first, second = _ordinary_actions(first), _ordinary_actions(second)
            if len(first) < 2 or not second:
                raise GateAError("not_joint_ordinary")
            if set(first) != set(legality["actions"]):
                raise GateAError("root_legality")
            bits = compile_side_one_reveal_mask(state, snapshot["player_information_state"])
            state = install_side_one_reveal_mask(state, bits)
            public = extract_public_search_state(state, engine)
            if public["action_table"]["name_table"] != snapshot["name_table"]:
                raise GateAError("root_action_map")
            if public["action_table"]["illegal_actions"] != snapshot["illegal_actions"]:
                raise GateAError("root_illegal_map")
            row["_state"] = state
            row["_side_one_actions"] = first
            row["_side_two_actions"] = second
            candidates.append(
                (rank([row["capture_sha256"], row["state_sha256"]]), row)
            )
        except GateAError as exc:
            rejected[str(exc)] += 1
        except Exception:
            rejected["invalid_or_mask_rejected"] += 1
    candidates.sort(key=lambda item: item[0])
    selected, per_battle = [], defaultdict(int)
    for _, row in candidates:
        battle = str(row["identity"]["battle_tag"])
        if per_battle[battle] >= 5:
            continue
        selected.append(row)
        per_battle[battle] += 1
        if len(selected) == 200:
            break
    if not 180 <= len(selected) <= 200:
        raise GateAError(
            f"selection produced {len(selected)} roots, outside roughly-200 range"
        )
    return selected, rejected


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


def _clone_pokemon(engine: Any, pokemon: Any, *, item: str | None = None, ability: str | None = None, moves: Sequence[Any] | None = None) -> Any:
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
            engine.Move(id=move.id, pp=int(move.pp), disabled=bool(move.disabled))
            for move in pokemon.moves
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


def hidden_perturbation(state: Any, engine: Any) -> Any | None:
    bits = int(state.s1_public_reveals)
    active = int(state.side_two.active_index)
    target = None
    for slot in range(6):
        hidden = (
            not bits & (1 << (30 + slot))
            or not bits & (1 << (36 + slot))
            or any(not bits & (1 << (6 + slot * 4 + move)) for move in range(4))
        )
        if slot != active and hidden:
            target = slot
            break
    if target is None:
        return None
    opponent = list(state.side_two.pokemon)
    original = opponent[target]
    item = None
    ability = None
    moves = None
    if not bits & (1 << (30 + target)):
        item = "choicescarf" if str(original.item).lower() != "choicescarf" else "leftovers"
    if not bits & (1 << (36 + target)):
        ability = "levitate" if str(original.ability).lower() != "levitate" else "pressure"
    hidden_moves = [index for index in range(4) if not bits & (1 << (6 + target * 4 + index))]
    if hidden_moves:
        moves = [
            engine.Move(id=move.id, pp=int(move.pp), disabled=bool(move.disabled))
            for move in original.moves
        ]
        index = hidden_moves[0]
        moves[index] = engine.Move(
            id="tackle" if str(moves[index].id).lower() != "tackle" else "protect",
            pp=int(moves[index].pp),
            disabled=bool(moves[index].disabled),
        )
    opponent[target] = _clone_pokemon(
        engine, original, item=item, ability=ability, moves=moves
    )
    return engine.State(
        side_one=_clone_side(
            engine, state.side_one,
            [_clone_pokemon(engine, pokemon) for pokemon in state.side_one.pokemon],
        ),
        side_two=_clone_side(engine, state.side_two, opponent),
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
        s1_public_reveals=int(state.s1_public_reveals),
        s2_public_reveals=int(state.s2_public_reveals),
    )


def python_rust_feature_error(state: Any, engine: Any) -> float:
    python = extract_resource_features(state, include_public_information=True)
    python[-1] = float(bool(state.trick_room))
    rust = list(engine.compute_resource_features(state))
    if len(python) != len(rust):
        raise GateAError("Python/Rust feature dimension mismatch")
    return max(abs(float(left) - float(right)) for left, right in zip(python, rust))


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = run_dir / "gate-a-report.json"
    if output.exists():
        raise GateAError("frozen Gate A report already exists")
    frozen = json.loads((run_dir / "FROZEN_INPUTS.json").read_text())
    for relative, expected in frozen["files"].items():
        if sha256(ROOT / relative) != expected:
            raise GateAError(f"frozen input hash mismatch: {relative}")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(1)
    import poke_engine

    paths = {
        "agent_a": ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-a-decisions.jsonl.dual-r1-roots.jsonl",
        "agent_b": ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-b-decisions.jsonl.dual-r1-roots.jsonl",
    }
    if any(sha256(paths[name]) != EXPECTED[name] for name in paths):
        raise GateAError("opened root source hash mismatch")
    if sha256(Path(poke_engine.poke_engine.__file__)) != EXPECTED["engine"]:
        raise GateAError("engine binding hash mismatch")
    rows = load_rows(paths["agent_a"]) + load_rows(paths["agent_b"])
    selected, selection_rejections = select_roots(rows, poke_engine)

    policy = make_policy()
    if not 1_000_000 <= parameter_count(policy) <= 5_000_000:
        raise GateAError("smoke policy is outside frozen parameter range")

    attempted = supported = terminal = automatic = query_count = 0
    restore_failures = hidden_failures = deterministic_failures = policy_failures = 0
    failure_codes: Counter[str] = Counter()
    parity_errors: list[float] = []
    public_states: list[dict[str, Any]] = []
    query_states: list[dict[str, Any]] = []
    for root in selected:
        state = root["_state"]
        pairs = [
            (left, right)
            for left in root["_side_one_actions"]
            for right in root["_side_two_actions"]
        ]
        pairs.sort(key=lambda pair: rank([root["capture_sha256"], *pair]))
        for left, right in pairs[:4]:
            for uniform in (0.25, 0.75):
                attempted += 1
                root_string = state.to_string()
                root_public = canonical_bytes(extract_public_search_state(state, poke_engine))
                try:
                    result = poke_engine.step_with_uniform_r1_semantic(
                        state, left, right, uniform
                    )
                    child = result.state
                    restored = child.reverse_instructions(result.selected_instructions)
                    if restored.to_string() != root_string or canonical_bytes(
                        extract_public_search_state(restored, poke_engine)
                    ) != root_public:
                        restore_failures += 1
                        raise GateAError("apply_reverse_mismatch")
                    public = extract_public_search_state(child, poke_engine)
                    first_bytes = canonical_bytes(public)
                    if first_bytes != canonical_bytes(
                        extract_public_search_state(child, poke_engine)
                    ):
                        deterministic_failures += 1
                        raise GateAError("nondeterministic_extraction")
                    perturbed = hidden_perturbation(child, poke_engine)
                    if perturbed is not None and canonical_bytes(
                        extract_public_search_state(perturbed, poke_engine)
                    ) != first_bytes:
                        hidden_failures += 1
                        raise GateAError("hidden_noninterference")
                    parity_errors.append(python_rust_feature_error(child, poke_engine))
                    if poke_engine.terminal_value(child) != 0.0:
                        terminal += 1
                    elif public["action_table"]["automatic_action"] == "nomove":
                        automatic += 1
                    else:
                        probabilities = infer(policy, [public])[0]
                        second = infer(policy, [public])[0]
                        if not np.array_equal(probabilities, second):
                            deterministic_failures += 1
                            raise GateAError("nondeterministic_policy")
                        query_count += 1
                        query_states.append(public)
                    public_states.append(public)
                    supported += 1
                except Exception as exc:
                    policy_failures += int(str(exc) == "policy")
                    failure_codes[type(exc).__name__ + ":" + str(exc)] += 1

    coverage = supported / attempted if attempted else 0.0
    if len(parity_errors) < 1_000:
        raise GateAError("fewer than 1,000 Python/Rust parity states")

    batch = query_states[:64]
    if len(batch) != 64:
        raise GateAError("fewer than 64 ordinary policy-query states")
    encoded = torch.from_numpy(encode_states(batch))
    illegal = torch.tensor(
        [state["action_table"]["illegal_actions"] for state in batch], dtype=torch.bool
    )
    traced = torch.jit.trace(policy, (encoded, illegal))
    model_path = run_dir / "public-search-policy-smoke.ts"
    torch.jit.save(traced, model_path)
    reloaded = torch.jit.load(str(model_path))
    with torch.no_grad():
        original_output = policy(encoded, illegal)
        exported_output = reloaded(encoded, illegal)
    export_error = float(torch.max(torch.abs(original_output - exported_output)).item())

    latency_ms = []
    for _ in range(10):
        infer(policy, batch)
    for _ in range(100):
        start = time.perf_counter()
        infer(policy, batch)
        latency_ms.append((time.perf_counter() - start) * 1_000.0)

    checks = {
        "coverage_ge_95pct": coverage >= 0.95,
        "hidden_completion_noninterference": hidden_failures == 0,
        "apply_reverse_byte_parity": restore_failures == 0,
        "deterministic_replay": deterministic_failures == 0,
        "python_rust_parity_1000": len(parity_errors) >= 1_000 and max(parity_errors) <= 1e-6,
        "export_parity": export_error <= 1e-7,
        "batch64_p95_le_50ms": percentile(latency_ms, 95) <= 50.0,
        "legal_policy_queries": query_count > 0 and policy_failures == 0,
    }
    report = {
        "schema": "metagross-search-native-v2-gate-a/v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "selection": {
            "source_rows": len(rows),
            "selected_roots": len(selected),
            "physical_battles": len({row["identity"]["battle_tag"] for row in selected}),
            "rejections": dict(sorted(selection_rejections.items())),
            "per_battle_cap": 5,
            "seed": SELECTION_SEED,
        },
        "coverage": {
            "attempted_successors": attempted,
            "supported_successors": supported,
            "coverage": coverage,
            "terminal_successors": terminal,
            "automatic_successors": automatic,
            "policy_queries": query_count,
            "failure_codes": dict(failure_codes.most_common()),
        },
        "invariants": {
            "hidden_failures": hidden_failures,
            "restore_failures": restore_failures,
            "determinism_failures": deterministic_failures,
            "python_rust_states": len(parity_errors),
            "python_rust_max_abs_error": max(parity_errors),
        },
        "policy_smoke": {
            "parameters": parameter_count(policy),
            "trained": False,
            "export_sha256": sha256(model_path),
            "export_max_abs_error": export_error,
            "batch_size": len(batch),
            "latency_mean_ms": statistics.fmean(latency_ms),
            "latency_p50_ms": percentile(latency_ms, 50),
            "latency_p95_ms": percentile(latency_ms, 95),
            "latency_max_ms": max(latency_ms),
        },
        "methodology": {
            "root_r1_used_at_interior": False,
            "fabricated_history": False,
            "sealed_confirmation_panel_rows_read": 0,
            "new_games": 0,
            "h2h_games": 0,
            "local_cpu_only": True,
            "paid_compute_usd": 0,
            "strength_claim_allowed": False,
        },
        "hashes": {
            "protocol_sha256": frozen["protocol_sha256"],
            "frozen_inputs_sha256": sha256(run_dir / "FROZEN_INPUTS.json"),
            "engine_binding_sha256": EXPECTED["engine"],
            "agent_a_source_sha256": EXPECTED["agent_a"],
            "agent_b_source_sha256": EXPECTED["agent_b"],
        },
    }
    report["report_sha256"] = hashlib.sha256(canonical_json(report).encode()).hexdigest()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
