from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from belief.causal_protocol_bridge_v2 import (
    parse_causal_protocol_v2,
    reconcile_causal_facts_v2,
)
from belief.public_form_contract import load_public_form_contract
from scripts.run_causal_protocol_bridge_cycle4 import (
    instruction_signature,
    request_actions_exact,
)


def private_request() -> dict:
    return {
        "rqid": 1,
        "active": [{"moves": [{"id": "Tackle", "pp": 10, "disabled": False}]}],
        "side": {
            "id": "p1",
            "pokemon": [
                {"active": True, "condition": "100/100", "details": "Pikachu, L80"},
                {"active": False, "condition": "100/100", "details": "Sawsbuck-Summer, L80"},
                {"active": False, "condition": "100/100", "details": "Minior-Violet, L80"},
                {"active": False, "condition": "100/100", "details": "Alcremie-Caramel-Swirl, L80"},
            ],
        },
    }


def test_own_switch_actions_preserve_exact_forms() -> None:
    assert request_actions_exact(private_request()) == {
        "tackle", "switch sawsbucksummer", "switch miniorviolet",
        "switch alcremiecaramelswirl",
    }


def test_showdown_contract_is_systematic_and_covers_cycle3_failure() -> None:
    contract = load_public_form_contract()
    assert len(contract.mapping) == 236
    assert contract.canonical("Ogerpon-Wellspring-Tera") == "ogerponwellspring"
    assert contract.canonical("Mimikyu-Busted-Totem") == "mimikyutotem"
    assert contract.canonical("Sawsbuck-Summer") == "sawsbuck"
    assert contract.authority["ogerponwellspringtera"] == "battleOnly"
    assert contract.authority["sawsbucksummer"] == "cosmeticFormes"


def test_protocol_species_uses_source_contract_not_private_request() -> None:
    request = private_request()
    prefix = [
        "|switch|p2a: Ogerpon|Ogerpon-Wellspring-Tera, L80|100/100",
        "|request|" + json.dumps(request, separators=(",", ":")),
    ]
    facts = parse_causal_protocol_v2(
        prefix, player_role="p1", private_request=request,
        contract=load_public_form_contract(),
    )
    assert facts.opponent[0].species == "ogerponwellspring"


def test_instruction_signature_uses_supported_binding_shape() -> None:
    opaque = SimpleNamespace(
        percentage=85.0,
        instruction_list=["Damage SideTwo: 10", "PublicReveal SideOne P0 Move(M0)"],
    )
    assert instruction_signature(opaque) == (
        85.0, ("Damage SideTwo: 10", "PublicReveal SideOne P0 Move(M0)"),
    )


def test_instruction_signature_and_reverse_against_frozen_binding() -> None:
    import poke_engine

    source = Path(
        "experimental/runs/schema6_local_5000_20260814_r1/peer/"
        "agent-a-decisions.jsonl.dual-r1-roots.jsonl"
    )
    row = json.loads(source.read_text().splitlines()[0])
    snapshot = row["r1_policy_snapshot"]
    contract = load_public_form_contract()
    facts = parse_causal_protocol_v2(
        snapshot["protocol_prefix"], player_role=snapshot["player_role"],
        private_request=snapshot["player_information_state"]["private_request"],
        contract=contract,
    )
    state = reconcile_causal_facts_v2(
        poke_engine.State.from_string(row["state"]), poke_engine, facts, contract
    ).state
    root_string = state.to_string()
    first_actions, second_actions = poke_engine.root_options(state)
    first = poke_engine.step_with_uniform_r1_semantic(
        state, first_actions[0], second_actions[0], 0.25
    )
    second = poke_engine.step_with_uniform_r1_semantic(
        state, first_actions[0], second_actions[0], 0.25
    )
    assert instruction_signature(first.selected_instructions) == instruction_signature(
        second.selected_instructions
    )
    assert first.state.to_string() == second.state.to_string()
    assert first.state.reverse_instructions(first.selected_instructions).to_string() == root_string
