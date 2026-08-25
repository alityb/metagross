#!/usr/bin/env python3
"""Audit event-certified facts through the accepted production sampler path."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from belief.causal_protocol_bridge import canonical_species, norm, parse_causal_protocol
from scripts.audit_public_search_state_gate_a_selection import select_without_representation
from scripts.run_public_search_state_gate_a import EXPECTED, GateAError, load_rows, rank, sha256


ROOT = Path(__file__).resolve().parents[3]
SEED = 2026081503
WORLD_COUNT = 8


def _converter_explicitly_omits_masks(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "battle_to_poke_engine_state":
            continue
        for call in (child for child in ast.walk(node) if isinstance(child, ast.Call)):
            name = call.func.id if isinstance(call.func, ast.Name) else ""
            if name != "PokeEngineState":
                continue
            keys = {keyword.arg for keyword in call.keywords}
            return "s1_public_reveals" not in keys and "s2_public_reveals" not in keys
    raise GateAError("could not locate production engine constructor")


def _fixture_battle(facts: Any, request: dict[str, Any], battle_tag: str) -> Any:
    from constants import BattleType, UNKNOWN_ITEM
    from fp.battle import Battle, Pokemon

    battle = Battle(battle_tag)
    battle.battle_type = BattleType.RANDOM_BATTLE
    battle.pokemon_format = "gen9randombattle"
    battle.generation = "gen9"
    battle.user.name = facts.observer_role
    battle.opponent.name = facts.opponent_role
    battle.user.initialize_first_turn_user_from_json(request)
    pokemon = {}
    for reveal in facts.opponent:
        if reveal.level is None:
            raise GateAError(f"fixture species lacks public level: {reveal.species}")
        mon = Pokemon(reveal.species, reveal.level)
        for move in reveal.moves:
            if mon.add_move(move) is None:
                raise GateAError(f"fixture has unknown public move: {move}")
        mon.ability = reveal.ability
        if reveal.item_status_revealed:
            mon.item = None if reveal.current_item == "none" else reveal.current_item
            if reveal.current_item == "none" and reveal.historically_revealed_items:
                mon.removed_item = reveal.historically_revealed_items[-1]
        else:
            mon.item = UNKNOWN_ITEM
        pokemon[reveal.species] = mon
    active = pokemon.pop(facts.opponent_active_species, None)
    if active is None:
        raise GateAError("fixture has no causal opponent active")
    battle.opponent.active = active
    battle.opponent.reserve = [pokemon[name] for name in sorted(pokemon)]
    battle.started = True
    battle.rqid = request["rqid"]
    return battle


def _find_engine_slot(state: Any, species: str) -> int | None:
    matches = [
        index for index, pokemon in enumerate(state.side_two.pokemon)
        if canonical_species(pokemon.id) == canonical_species(species)
    ]
    return matches[0] if len(matches) == 1 else None


def _score_world(state: Any, facts: Any) -> tuple[Counter[str], Counter[str], Counter[str]]:
    raw_total: Counter[str] = Counter()
    raw_kept: Counter[str] = Counter()
    typed_kept: Counter[str] = Counter()
    bits = int(getattr(state, "s1_public_reveals", 0))
    for reveal in facts.opponent:
        raw_total["species"] += 1
        slot = _find_engine_slot(state, reveal.species)
        if slot is None:
            continue
        raw_kept["species"] += 1
        typed_kept["species"] += int(bool(bits & (1 << slot)))
        pokemon = state.side_two.pokemon[slot]
        for move in reveal.moves:
            raw_total["move"] += 1
            matches = [index for index, row in enumerate(pokemon.moves) if norm(row.id) == move]
            if len(matches) == 1:
                raw_kept["move"] += 1
                typed_kept["move"] += int(bool(bits & (1 << (6 + slot * 4 + matches[0]))))
        if reveal.item_status_revealed:
            raw_total["item"] += 1
            expected = reveal.current_item or "none"
            if norm(pokemon.item) == norm(expected):
                raw_kept["item"] += 1
            typed_kept["item"] += int(bool(bits & (1 << (30 + slot))))
        if reveal.ability is not None:
            raw_total["ability"] += 1
            if norm(pokemon.ability) == reveal.ability:
                raw_kept["ability"] += 1
            typed_kept["ability"] += int(bool(bits & (1 << (36 + slot))))
    return raw_total, raw_kept, typed_kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = run_dir / "production-sampler-preservation.json"
    if output.exists():
        raise GateAError("sampler preservation report already exists")
    frozen = json.loads((run_dir / "FROZEN_INPUTS.json").read_text())
    for relative, expected in frozen["files"].items():
        if sha256(ROOT / relative) != expected:
            raise GateAError(f"frozen input hash mismatch: {relative}")

    import poke_engine

    if sha256(Path(poke_engine.poke_engine.__file__)) != EXPECTED["engine"]:
        raise GateAError("engine binding hash mismatch")
    paths = [
        ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-a-decisions.jsonl.dual-r1-roots.jsonl",
        ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-b-decisions.jsonl.dual-r1-roots.jsonl",
    ]
    rows = load_rows(paths[0]) + load_rows(paths[1])
    selected, _ = select_without_representation(rows, poke_engine)
    candidates = []
    parse_failures: Counter[str] = Counter()
    for root in selected:
        snapshot = root["r1_policy_snapshot"]
        try:
            facts = parse_causal_protocol(
                snapshot["protocol_prefix"],
                player_role=snapshot["player_role"],
                private_request=snapshot["player_information_state"]["private_request"],
            )
            field_types = {
                "species",
                *("move" for reveal in facts.opponent if reveal.moves),
                *("item" for reveal in facts.opponent if reveal.item_status_revealed),
                *("ability" for reveal in facts.opponent if reveal.ability is not None),
            }
            if field_types == {"species", "move", "item", "ability"}:
                candidates.append((rank([root["capture_sha256"], "sampler-fixture"]), root, facts))
        except Exception as exc:
            parse_failures[f"{type(exc).__name__}:{exc}"] += 1
    if not candidates:
        report = {
            "schema": "metagross-production-sampler-public-reveal-audit/v1",
            "status": "blocked",
            "reason": "no selected causal fixture certifies all four field types",
            "parse_failures": dict(parse_failures.most_common()),
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    _, root, facts = sorted(candidates, key=lambda row: row[0])[0]

    foul_play = ROOT / "srcs/vendor/foul-play"
    sys.path.insert(0, str(foul_play))
    old_cwd = Path.cwd()
    os.chdir(foul_play)
    try:
        from data.pkmn_sets import RandomBattleTeamDatasets
        from fp.search import main as search_main

        RandomBattleTeamDatasets.initialize("gen9")
        battle = _fixture_battle(
            facts,
            root["r1_policy_snapshot"]["player_information_state"]["private_request"],
            str(root["identity"]["battle_tag"]),
        )
        worlds = search_main.prepare_random_battles(
            battle, WORLD_COUNT, rng=random.Random(SEED)
        )
        raw_total: Counter[str] = Counter()
        raw_kept: Counter[str] = Counter()
        typed_kept: Counter[str] = Counter()
        masks = []
        for sampled, _weight in worlds:
            state = search_main.battle_to_poke_engine_state(sampled)
            total, kept, typed = _score_world(state, facts)
            raw_total.update(total)
            raw_kept.update(kept)
            typed_kept.update(typed)
            masks.append(int(getattr(state, "s1_public_reveals", 0)))
        sampler_path = Path(search_main.prepare_random_battles.__code__.co_filename).resolve()
        converter_path = Path(search_main.battle_to_poke_engine_state.__code__.co_filename).resolve()
    finally:
        os.chdir(old_cwd)

    fields = ("species", "move", "item", "ability")
    raw_recall = {
        field: raw_kept[field] / raw_total[field] if raw_total[field] else None
        for field in fields
    }
    typed_recall = {
        field: typed_kept[field] / raw_total[field] if raw_total[field] else None
        for field in fields
    }
    live_bug = any(
        typed_kept[field] != raw_total[field]
        for field in fields if raw_total[field]
    )
    report = {
        "schema": "metagross-production-sampler-public-reveal-audit/v1",
        "status": "live_capture_contract_bug" if live_bug else "pass",
        "fixture": {
            "battle_tag": root["identity"]["battle_tag"],
            "capture_sha256": root["capture_sha256"],
            "worlds": WORLD_COUNT,
            "seed": SEED,
        },
        "raw_fact_total": dict(raw_total),
        "raw_fact_preserved": dict(raw_kept),
        "raw_fact_recall": raw_recall,
        "typed_fact_preserved": dict(typed_kept),
        "typed_fact_recall": typed_recall,
        "sampled_engine_masks": masks,
        "converter_omits_typed_mask_keywords": _converter_explicitly_omits_masks(converter_path),
        "interpretation": (
            "A missing typed fact in any sampled engine world is a live search capture-contract bug; "
            "raw Pokemon fields alone do not authorize an interior policy to observe it."
        ),
        "hashes": {
            "accepted_runner_sha256": sha256(ROOT / "srcs/metagross/run_foul_play.py"),
            "sampler_module_sha256": sha256(sampler_path),
            "converter_module_sha256": sha256(converter_path),
            "engine_binding_sha256": EXPECTED["engine"],
        },
        "local_cpu_only": True,
        "paid_compute_usd": 0,
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
