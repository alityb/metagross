from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "srcs/vendor/foul-play"))

from constants import BattleType  # noqa: E402
from data import all_move_json  # noqa: E402
from fp.battle import Battle, Pokemon  # noqa: E402
from fp.battle_modifier import move, update_ability  # noqa: E402


def pp_battle(role: str = "p1", actor: str = "oricoriosensu") -> Battle:
    value = Battle("cycle39-test")
    value.user.name = role
    value.opponent.name = "p2" if role == "p1" else "p1"
    value.generation = "gen9"
    value.battle_type = BattleType.RANDOM_BATTLE
    value.user.active = Pokemon("suicune", 80)
    value.user.active.nickname = "Pressure"
    value.user.active.ability = "pressure"
    value.opponent.active = Pokemon(actor, 80)
    value.opponent.active.nickname = "Actor"
    value.turn = 1
    return value


def execute(value: Battle, move_name: str, target: str, *attrs: str):
    actor_role = value.opponent.name
    line = ["", "move", f"{actor_role}a: Actor", move_name, target, *attrs]
    move(value, line)


@pytest.mark.parametrize("role", ["p1", "p2"])
def test_self_moves_do_not_pay_pressure_and_normal_foe_moves_do(role: str) -> None:
    value = pp_battle(role)
    execute(value, "Roost", f"{value.opponent.name}a: Actor")
    execute(value, "Hurricane", f"{value.user.name}a: Pressure")
    roost = value.opponent.active.get_move("roost")
    hurricane = value.opponent.active.get_move("hurricane")
    assert (roost.current_pp, roost.max_pp) == (7, 8)
    assert (hurricane.current_pp, hurricane.max_pp) == (14, 16)
    assert roost.metagross_causal_pp_events[0]["pressure_extra"] == 0
    assert hurricane.metagross_causal_pp_events[0] == {
        "sequence": 0,
        "turn": 1,
        "executed_move": "hurricane",
        "charged_move": "hurricane",
        "called_by": None,
        "base_cost": 1,
        "pressure_extra": 1,
        "total_cost": 2,
        "pressure_authority": "observer_private_request",
        "target_semantics": "any",
        "mustpressure": False,
    }


def test_spread_adjacent_and_mustpressure_foeside_follow_pinned_semantics() -> None:
    value = pp_battle()
    execute(value, "Surf", f"{value.user.name}a: Pressure")
    execute(value, "Thunderbolt", f"{value.user.name}a: Pressure")
    execute(value, "Spikes", f"{value.user.name}: Pressure")
    assert value.opponent.active.get_move("surf").current_pp == 22  # 24 - 2
    assert value.opponent.active.get_move("thunderbolt").current_pp == 22
    spikes = value.opponent.active.get_move("spikes")
    assert spikes.current_pp == 30  # 32 - 2; foeSide + mustpressure
    assert spikes.metagross_causal_pp_events[0]["target_semantics"] == "foeSide"
    assert spikes.metagross_causal_pp_events[0]["mustpressure"] is True


def test_called_attack_charges_only_callers_pressure_surcharge() -> None:
    value = pp_battle()
    execute(value, "Sleep Talk", f"{value.opponent.name}a: Actor")
    execute(
        value, "Hurricane", f"{value.user.name}a: Pressure",
        "[from] move: Sleep Talk",
    )
    sleep_talk = value.opponent.active.get_move("sleeptalk")
    hurricane = value.opponent.active.get_move("hurricane")
    assert (sleep_talk.current_pp, sleep_talk.max_pp) == (14, 16)
    assert [row["total_cost"] for row in sleep_talk.metagross_causal_pp_events] == [1, 1]
    assert sleep_talk.metagross_causal_pp_events[-1]["called_by"] == "sleeptalk"
    assert hurricane.current_pp == hurricane.max_pp


def test_pressure_suppression_and_switch_change_event_cost_causally() -> None:
    value = pp_battle()
    value.opponent.active.ability = "neutralizinggas"
    execute(value, "Hurricane", f"{value.user.name}a: Pressure")
    assert value.opponent.active.get_move("hurricane").current_pp == 15

    value.opponent.active = Pokemon("oricoriosensu", 80)
    value.opponent.active.nickname = "Actor"
    value.user.active.volatile_statuses.append("gastroacid")
    execute(value, "Hurricane", f"{value.user.name}a: Pressure")
    assert value.opponent.active.get_move("hurricane").current_pp == 15

    value.opponent.active = Pokemon("oricoriosensu", 80)
    value.opponent.active.nickname = "Actor"
    value.user.active = Pokemon("pikachu", 80)
    value.user.active.nickname = "NoPressure"
    value.user.active.ability = "static"
    execute(value, "Hurricane", f"{value.user.name}a: NoPressure")
    assert value.opponent.active.get_move("hurricane").current_pp == 15


def test_public_certification_required_when_pressure_is_on_opponent_side() -> None:
    value = pp_battle()
    value.user.active, value.opponent.active = value.opponent.active, value.user.active
    value.user.active.nickname = "Actor"
    value.opponent.active.nickname = "Pressure"
    # Actor is the observer/user, so an opponent default/sample is insufficient.
    move(value, [
        "", "move", f"{value.user.name}a: Actor", "Hurricane",
        f"{value.opponent.name}a: Pressure",
    ])
    assert value.user.active.get_move("hurricane").current_pp == 15

    value.user.active = Pokemon("oricoriosensu", 80)
    value.user.active.nickname = "Actor"
    update_ability(value, [
        "", "-ability", f"{value.opponent.name}a: Pressure", "Pressure",
    ])
    move(value, [
        "", "move", f"{value.user.name}a: Actor", "Hurricane",
        f"{value.opponent.name}a: Pressure",
    ])
    assert value.user.active.get_move("hurricane").current_pp == 14


def test_exact_zero_is_valid_and_never_clamped() -> None:
    value = pp_battle()
    for turn in range(8):
        value.turn = turn + 1
        execute(value, "Roost", f"{value.opponent.name}a: Actor")
    roost = value.opponent.active.get_move("roost")
    assert roost.current_pp == 0
    assert sum(row["total_cost"] for row in roost.metagross_causal_pp_events) == 8


def test_foul_play_target_metadata_matches_pinned_showdown_contract() -> None:
    payload = json.loads(subprocess.check_output(
        ["node", str(ROOT / "srcs/metagross/export_showdown_pressure_target_contract.cjs")],
        text=True,
    ))
    assert payload["showdown_commit"] == "f8ac14003a5f27e1bdc8d8c59608a773c1cb96e5"
    assert payload["row_count"] >= 950
    contract = {row["id"]: row for row in payload["rows"]}
    for move_id in (
        "roost", "hurricane", "surf", "thunderbolt", "spikes", "sleeptalk",
    ):
        assert all_move_json[move_id]["target"] == contract[move_id]["target"]
        assert bool(all_move_json[move_id].get("flags", {}).get("mustpressure")) \
            is contract[move_id]["mustpressure"]


def test_tampered_or_missing_event_receipts_fail_closed() -> None:
    from types import SimpleNamespace
    from srcs.metagross.causal_reveal_ledger import (
        CausalRevealLedgerError, bind_live_move_states, freeze_ledger,
    )

    ledger = freeze_ledger("cycle39-tamper", "p1", [
        "|switch|p1a: Own|Suicune, L80|100/100",
        "|-ability|p1a: Own|Pressure",
        "|switch|p2a: Foe|Oricorio-Sensu, L80|100/100",
        "|move|p2a: Foe|Roost|p2a: Foe",
    ])
    row = SimpleNamespace(
        name="roost", current_pp=7, max_pp=8, disabled=False,
        metagross_causal_pp_events=[],
    )
    source = SimpleNamespace(opponent=SimpleNamespace(
        active=SimpleNamespace(name="oricoriosensu", moves=[row]), reserve=[],
    ))
    with pytest.raises(CausalRevealLedgerError, match="invalid causal"):
        bind_live_move_states(source, ledger)


def test_preserved_oricorio_pressure_transcript_reaches_exact_zero() -> None:
    import os
    from experimental.src.scripts import cycle12_replay_audit as v12
    from experimental.src.scripts import audit_cycle13_train_rehydration as c13
    from experimental.src.scripts import audit_cycle14_mechanics_repair as c14
    from srcs.metagross import causal_reveal_ledger as crl
    from data.pkmn_sets import RandomBattleTeamDatasets

    selection = ROOT / (
        "experimental/runs/search_native_v2_cycle37_own_active_resolver_20260816/"
        "resolver-selection.jsonl"
    )
    selected = next(
        json.loads(line) for line in selection.read_text().splitlines()
        if "gen9randombattle-2641485593" in line
    )
    capture = ROOT / (
        "experimental/runs/search_native_v2_cycle37_own_active_resolver_20260816/"
        "mechanics-audit/failures/cc266a54937a40d8/capture"
    )
    raw = json.loads(Path(selected["raw_path"]).read_text())
    public = json.loads((capture / "public.json").read_text())
    pov = json.loads((capture / "p2.json").read_text())
    derived = v12.materialize_role(
        battle_id=selected["battle_id"], role="p2", public_capture=public,
        pov_capture=pov, inputlog=raw["inputlog"],
        showdown_commit=selected["showdown_commit"],
    )
    target = derived["states"][selected["request_index"]]
    ledger = crl.freeze_ledger(selected["battle_id"], "p2", target["public_prefix"])
    RandomBattleTeamDatasets.initialize("gen9")
    previous = Path.cwd()
    os.chdir(ROOT / "srcs/vendor/foul-play")
    try:
        value = c14.build_foul_play_battle_fixed(
            battle_id=selected["battle_id"], role="p2", states=derived["states"],
            target_index=selected["request_index"],
        )
        c13.reconcile_public_facts_into_battle(value, ledger)
        bound = crl.bind_live_move_states(value, ledger)
    finally:
        os.chdir(previous)
    states = {
        state.move: state
        for fact in bound.facts if fact.species == "oricoriosensu"
        for state in fact.move_states
    }
    assert (states["roost"].current_pp, states["roost"].max_pp) == (0, 8)
    assert (states["quiverdance"].current_pp, states["quiverdance"].max_pp) == (26, 32)
    assert (states["hurricane"].current_pp, states["hurricane"].max_pp) == (8, 16)
    assert all(
        state.authority == "causal_event_counted_public_tracker"
        for state in states.values()
    )
