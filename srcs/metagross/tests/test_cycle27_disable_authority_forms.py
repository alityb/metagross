from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from srcs.metagross.causal_reveal_ledger import (
    CausalRevealLedgerError,
    LEDGER_ATTRIBUTE,
    MOVE_RECEIPT_ATTRIBUTE,
    bind_live_move_states,
    freeze_ledger,
    verify_sampled_move_states,
    _emit_move_world_receipt,
)
from experimental.src.scripts.monitor_cycle27_second_root_smoke import (
    selected_action_window,
    validate_move_receipts,
)


def move(name: str, pp: int = 7, max_pp: int = 8, disabled: bool = False):
    spent = max_pp - pp
    return SimpleNamespace(
        name=name, current_pp=pp, max_pp=max_pp, disabled=disabled,
        metagross_causal_pp_events=[
            {
                "sequence": index,
                "total_cost": 1,
                "pressure_extra": 0,
                "pressure_authority": None,
            }
            for index in range(spent)
        ],
    )


def battle(species: str, moves):
    return SimpleNamespace(
        opponent=SimpleNamespace(
            active=SimpleNamespace(name=species, moves=list(moves)), reserve=[]
        )
    )


def test_explicit_cursed_body_disable_is_causal_and_invariant() -> None:
    ledger = freeze_ledger("battle-disable", "p1", [
        "|switch|p1a: Own|Froslass, L80|100/100",
        "|switch|p2a: Foe|Zangoose, L80|100/100",
        "|move|p2a: Foe|Knock Off|p1a: Own",
        "|-start|p2a: Foe|Disable|Knock Off|[from] ability: Cursed Body|[of] p1a: Own",
    ])
    source = battle("zangoose", [move("knockoff", disabled=True)])
    bound = bind_live_move_states(source, ledger)
    state = bound.facts[0].move_states[0]
    assert (state.disable_authority, state.disabled) == ("causal_disable", True)
    setattr(source, LEDGER_ATTRIBUTE, bound.to_payload())
    good = copy.deepcopy(source)
    verify_sampled_move_states(source, [(good, 1.0)])
    receipt = getattr(good, MOVE_RECEIPT_ATTRIBUTE)["moves"][0]
    assert receipt["disable_authority"] == "causal_disable"
    assert receipt["causal_disabled"] is True
    assert receipt["world_disabled"] is True
    bad = copy.deepcopy(source)
    bad.opponent.active.moves[0].disabled = False
    with pytest.raises(CausalRevealLedgerError, match="changed causal PP-disable"):
        verify_sampled_move_states(source, [(bad, 1.0)])


def test_public_disable_end_and_switch_clear_are_causal_false() -> None:
    base = [
        "|switch|p1a: Own|Froslass, L80|100/100",
        "|switch|p2a: Foe|Zangoose, L80|100/100",
        "|move|p2a: Foe|Knock Off|p1a: Own",
        "|-start|p2a: Foe|Disable|Knock Off|[from] ability: Cursed Body|[of] p1a: Own",
    ]
    ended = freeze_ledger("battle-disable-end", "p1", [*base, "|-end|p2a: Foe|Disable"])
    state = bind_live_move_states(
        battle("zangoose", [move("knockoff", disabled=False)]), ended
    ).facts[0].move_states[0]
    assert (state.disable_authority, state.disabled) == (
        "world_mechanical_disable", None
    )
    assert state.causal_disable_lifecycle_state is False

    switched = freeze_ledger("battle-disable-switch", "p1", [
        *base,
        "|switch|p2a: Other|Mew, L80|100/100",
        "|switch|p2a: Foe|Zangoose, L80|100/100",
    ])
    zangoose = next(fact for fact in switched.facts if fact.species == "zangoose")
    assert zangoose.disable_history[-1].disabled is False
    assert zangoose.disable_history[-1].protocol_tag == "switch"


def test_world_mechanical_choice_lock_is_preserved_in_receipt() -> None:
    ledger = freeze_ledger("battle-choice", "p1", [
        "|switch|p1a: Own|Pikachu, L80|100/100",
        "|switch|p2a: Foe|Barraskewda, L80|100/100",
        "|move|p2a: Foe|Flip Turn|p1a: Own",
    ])
    source = battle("barraskewda", [move("flipturn", pp=31, max_pp=32)])
    bound = bind_live_move_states(source, ledger)
    setattr(source, LEDGER_ATTRIBUTE, bound.to_payload())
    sampled = copy.deepcopy(source)
    sampled.opponent.active.moves[0].disabled = True
    verify_sampled_move_states(source, [(sampled, 0.5)])
    state = bound.facts[0].move_states[0]
    receipt = getattr(sampled, MOVE_RECEIPT_ATTRIBUTE)["moves"][0]
    assert (state.disable_authority, state.disabled) == (
        "world_mechanical_disable", None
    )
    assert receipt["world_disabled"] is True
    assert receipt["causal_disabled"] is None


def test_conversion_receipt_is_bound_to_engine_world_and_context(
    tmp_path, monkeypatch
) -> None:
    ledger = freeze_ledger("battle-receipt", "p1", [
        "|switch|p1a: Own|Pikachu, L80|100/100",
        "|switch|p2a: Foe|Barraskewda, L80|100/100",
        "|move|p2a: Foe|Flip Turn|p1a: Own",
    ])
    source = battle("barraskewda", [move("flipturn", pp=31, max_pp=32)])
    bound = bind_live_move_states(source, ledger)
    setattr(source, LEDGER_ATTRIBUTE, bound.to_payload())
    sampled = copy.deepcopy(source)
    sampled.opponent.active.moves[0].disabled = True
    verify_sampled_move_states(source, [(sampled, 1.0)])
    engine_move = SimpleNamespace(id="flipturn", pp=31, disabled=True)
    engine_mon = SimpleNamespace(id="barraskewda", moves=[engine_move])
    none = SimpleNamespace(id="none", moves=[])
    state = SimpleNamespace(
        side_one=SimpleNamespace(pokemon=[none] * 6),
        side_two=SimpleNamespace(pokemon=[engine_mon, *([none] * 5)]),
    )
    context = {
        "phase": "equal8192_candidate", "cohort": "fixed_two_by_eight",
        "battle_tag": "battle-receipt", "rqid": 3, "decision_index": 1,
        "root_id": "root", "declared_world_count": 16,
        "conversion_index": 0, "schedule_index": 0, "world_index": 0,
    }
    monkeypatch.setenv("METAGROSS_CAUSAL_MOVE_RECEIPT_DIR", str(tmp_path))
    monkeypatch.setenv("METAGROSS_PRIOR_NAMESPACE", "agent_a")
    _emit_move_world_receipt(
        sampled, bound, state, swap=False, receipt_context=context
    )
    rows = [
        __import__("json").loads(line)
        for line in next(tmp_path.glob("agenta-*.jsonl")).read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["execution_context"] == context
    assert rows[0]["move_receipt"]["moves"][0]["world_disabled"] is True
    assert rows[0]["move_receipt"]["moves"][0]["disable_authority"] == (
        "world_mechanical_disable"
    )


def test_morpeko_hangry_activation_preserves_exact_public_form() -> None:
    ledger = freeze_ledger("battle-morpeko", "p1", [
        "|switch|p1a: Own|Umbreon, L80|100/100",
        "|switch|p2a: Morpeko|Morpeko, L88|100/100",
        "|-formechange|p2a: Morpeko|Morpeko-Hangry||[from] ability: Hunger Switch",
        "|move|p2a: Morpeko|Knock Off|p1a: Own",
    ])
    fact = ledger.facts[0]
    assert fact.species == "morpeko"
    assert fact.exact_public_species == "morpekohangry"
    assert fact.form_history[-1].exact_public_species == "morpekohangry"
    assert fact.form_history[-1].source_ability == "hungerswitch"
    assert fact.current_ability == "hungerswitch"
    bind_live_move_states(
        battle("morpekohangry", [move("knockoff", pp=31, max_pp=32)]), ledger
    )


def test_minior_activation_preserves_color_history_and_current_mechanics() -> None:
    ledger = freeze_ledger("battle-minior", "p1", [
        "|switch|p1a: Own|Meganium, L80|100/100",
        "|switch|p2a: Minior|Minior-Yellow, L79|100/100",
        "|-formechange|p2a: Minior|Minior-Meteor||[from] ability: Shields Down",
        "|-formechange|p2a: Minior|Minior-Yellow||[from] ability: Shields Down",
    ])
    fact = ledger.facts[0]
    assert fact.species == "minior"
    assert [event.exact_public_species for event in fact.form_history] == [
        "miniormeteor", "minioryellow"
    ]
    assert fact.exact_public_species == "minioryellow"


def test_unpinned_battle_form_transition_fails_closed() -> None:
    with pytest.raises(CausalRevealLedgerError, match="unsupported public battle-form"):
        freeze_ledger("battle-form-bad", "p1", [
            "|switch|p1a: Own|Pikachu, L80|100/100",
            "|switch|p2a: Foe|Morpeko, L80|100/100",
            "|-formechange|p2a: Foe|Morpeko-Hangry||[from] ability: Illusion",
        ])


def _conversion_receipt(context: dict, *, disabled: bool = False, stamp: int = 10):
    move_row = {
        "exact_public_species": "mienshao",
        "move": "closecombat",
        "current_pp": 7,
        "max_pp": 8,
        "disable_authority": "world_mechanical_disable",
        "causal_disabled": None,
        "causal_disable_lifecycle_state": None,
        "world_disabled": disabled,
    }
    return {
        "schema": "metagross-causal-move-conversion-receipt/v1",
        "battle_tag": context["battle_tag"],
        "observer_role": "p1",
        "protocol_sha256": "b" * 64,
        "swap": False,
        "execution_context": context,
        "move_receipt": {
            "schema": "metagross-causal-move-world-receipts/v1",
            "battle_tag": context["battle_tag"],
            "protocol_sha256": "b" * 64,
            "moves": [move_row],
        },
        "receipt_time_ns": stamp,
    }


def test_second_root_monitor_reconciles_typed_two_by_eight(tmp_path) -> None:
    identity = {
        "battle_tag": "battle-cycle27",
        "rqid": 3,
        "decision_index": 1,
        "root_id": "a" * 64,
    }
    rows = []
    for index in range(16):
        rows.append(_conversion_receipt({
            **identity,
            "phase": "production_control",
            "cohort": "adaptive_root_search",
            "declared_world_count": 16,
            "conversion_index": index,
            "schedule_index": None,
            "world_index": None,
        }))
    for schedule in range(2):
        for world in range(8):
            rows.append(_conversion_receipt({
                **identity,
                "phase": "equal8192_candidate",
                "cohort": "fixed_two_by_eight",
                "declared_world_count": 16,
                "conversion_index": schedule * 8 + world,
                "schedule_index": schedule,
                "world_index": world,
            }, disabled=(world % 2 == 0)))
    directory = tmp_path / "move-receipts"
    directory.mkdir()
    path = directory / "agenta-1.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = validate_move_receipts(
        tmp_path, identity, {"public_execution_time_ns": 100}
    )
    assert result["candidate_receipts"] == 16
    assert result["production_receipts"] == 16
    assert result["disable_authority_counts"]["world_mechanical_disable"] == 32


def test_second_root_action_window_uses_send_after_decision() -> None:
    protocol = [
        {"time_ns": 1, "direction": "received", "message": "|player|p1|Agent|\n|request|{\"rqid\":1}"},
        {"time_ns": 3, "direction": "sent", "messages": ["/choose move surf", "1"]},
        {"time_ns": 4, "direction": "received", "message": "|move|p1a: Agent|Surf|p2a: Foe"},
        {"time_ns": 5, "direction": "received", "message": "|request|{\"rqid\":2}"},
        {"time_ns": 8, "direction": "sent", "messages": ["/choose move roost", "2"]},
        {"time_ns": 9, "direction": "received", "message": "|move|p1a: Agent|Roost|p1a: Agent"},
    ]
    result = selected_action_window(protocol, 7, "agent", "roost")
    assert result["rqid"] == "2"
    assert result["send_time_ns"] == 8
    assert result["public_execution_time_ns"] == 9
