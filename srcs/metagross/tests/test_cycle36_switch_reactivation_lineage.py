from __future__ import annotations

import copy

import pytest

from srcs.metagross.causal_reveal_ledger import (
    CausalRevealLedger,
    CausalRevealLedgerError,
    freeze_ledger,
    form_ability_contract,
)


def _fact(lines: list[str], role: str = "p2"):
    ledger = freeze_ledger("battle-cycle36", role, lines)
    matches = [fact for fact in ledger.facts if fact.ability_history]
    assert len(matches) == 1
    return ledger, matches[0]


def test_preserved_cycle35_morpeko_toggle_return_recertifies_base_form() -> None:
    ledger, fact = _fact([
        "|switch|p1a: Morpeko|Morpeko, L88, F|100/100",
        "|-formechange|p1a: Morpeko|Morpeko-Hangry||[from] ability: Hunger Switch",
        "|-formechange|p1a: Morpeko|Morpeko||[from] ability: Hunger Switch",
        "|-formechange|p1a: Morpeko|Morpeko-Hangry||[from] ability: Hunger Switch",
        "|switch|p1a: Other|Umbreon, L80|100/100",
        "|switch|p1a: Morpeko|Morpeko, L88, F|72/100",
    ])
    assert fact.exact_public_species == "morpeko"
    assert fact.current_ability == "hungerswitch"
    assert fact.current_ability_authority == "rule_implied_switch_reactivation"
    assert fact.ability_history[-1].exact_public_species == "morpeko"
    assert fact.ability_history[-1].protocol_tag == "switch"
    assert CausalRevealLedger.from_payload(ledger.to_payload()) == ledger


@pytest.mark.parametrize("role,opponent", [("p1", "p2"), ("p2", "p1")])
def test_repeated_switch_and_drag_reactivation_both_roles(role: str, opponent: str) -> None:
    ledger, fact = _fact([
        f"|switch|{opponent}a: Morpeko|Morpeko, L88|100/100",
        f"|-ability|{opponent}a: Morpeko|Hunger Switch",
        f"|switch|{opponent}a: Other|Umbreon, L80|100/100",
        f"|drag|{opponent}a: Morpeko|Morpeko, L88|80/100",
        f"|switch|{opponent}a: Other|Umbreon, L80|90/100",
        f"|switch|{opponent}a: Morpeko|Morpeko, L88|60/100",
    ], role)
    reactivations = [
        event for event in fact.ability_history
        if event.authority == "rule_implied_switch_reactivation"
    ]
    assert [event.protocol_tag for event in reactivations] == ["drag", "switch"]
    assert all(event.exact_public_species == "morpeko" for event in reactivations)
    assert ledger.opponent_role == opponent


@pytest.mark.parametrize(
    "first_exact,changed_exact,returned_exact,ability",
    [
        ("Aegislash", "Aegislash-Blade", "Aegislash", "stancechange"),
        ("Minior-Meteor", "Minior-Yellow", "Minior-Meteor", "shieldsdown"),
        (
            "Ogerpon-Wellspring",
            "Ogerpon-Wellspring-Tera",
            "Ogerpon-Wellspring-Tera",
            "embodyaspectwellspring",
        ),
        ("Terapagos", "Terapagos-Terastal", "Terapagos-Terastal", "terashell"),
    ],
)
def test_systematic_exact_form_reactivation_contract(
    first_exact: str, changed_exact: str, returned_exact: str, ability: str
) -> None:
    # detailschange is the existing certified form transition for families that
    # change mechanical exact identity in the request/state contract.
    ledger, fact = _fact([
        f"|switch|p1a: Foe|{first_exact}, L80|100/100",
        f"|detailschange|p1a: Foe|{changed_exact}, L80",
        "|switch|p1a: Other|Umbreon, L80|100/100",
        f"|switch|p1a: Foe|{returned_exact}, L80|90/100",
    ])
    exact = "".join(ch for ch in returned_exact.lower() if ch.isalnum())
    assert form_ability_contract()[exact] == ability
    assert fact.exact_public_species == exact
    assert fact.current_ability == ability
    assert fact.current_ability_authority == "rule_implied_switch_reactivation"
    assert fact.ability_history[-1].exact_public_species == exact
    assert CausalRevealLedger.from_payload(ledger.to_payload()) == ledger


def test_skill_swap_is_not_carried_across_switch_reactivation() -> None:
    _ledger, fact = _fact([
        "|switch|p1a: Morpeko|Morpeko, L88|100/100",
        "|-ability|p1a: Morpeko|Levitate|[from] move: Skill Swap",
        "|switch|p1a: Other|Umbreon, L80|100/100",
        "|switch|p1a: Morpeko|Morpeko, L88|100/100",
    ])
    assert [event.ability for event in fact.ability_history] == [
        "levitate", "hungerswitch"
    ]
    assert fact.current_ability == "hungerswitch"
    assert fact.current_ability_authority == "rule_implied_switch_reactivation"


def test_transform_remains_fail_closed() -> None:
    with pytest.raises(
        CausalRevealLedgerError,
        match="unsupported public ability-changing transform event",
    ):
        _fact([
            "|switch|p1a: Ditto|Ditto, L88|100/100",
            "|-transform|p1a: Ditto|p2a: Morpeko",
            "|switch|p1a: Other|Umbreon, L80|100/100",
            "|switch|p1a: Ditto|Ditto, L88|100/100",
        ])


def test_ability_suppression_does_not_replace_exact_form_ability_on_return() -> None:
    _ledger, fact = _fact([
        "|switch|p1a: Morpeko|Morpeko, L88|100/100",
        "|-ability|p1a: Morpeko|Hunger Switch",
        "|-endability|p1a: Morpeko|Hunger Switch|[from] move: Gastro Acid",
        "|switch|p1a: Other|Umbreon, L80|100/100",
        "|switch|p1a: Morpeko|Morpeko, L88|100/100",
    ])
    assert fact.current_ability == "hungerswitch"
    assert fact.ability_history[-1].authority == "rule_implied_switch_reactivation"


def test_ambiguous_contract_clear_preserves_history(monkeypatch: pytest.MonkeyPatch) -> None:
    import srcs.metagross.causal_reveal_ledger as module

    contract = dict(form_ability_contract())
    contract.pop("morpeko")
    monkeypatch.setattr(module, "_FORM_ABILITY_CONTRACT", contract)
    ledger, fact = _fact([
        "|switch|p1a: Morpeko|Morpeko, L88|100/100",
        "|-ability|p1a: Morpeko|Hunger Switch",
        "|switch|p1a: Other|Umbreon, L80|100/100",
        "|switch|p1a: Morpeko|Morpeko, L88|100/100",
    ])
    assert fact.current_ability is None
    assert fact.current_ability_authority is None
    assert fact.ability_history[-1].ability == "hungerswitch"
    assert CausalRevealLedger.from_payload(copy.deepcopy(ledger.to_payload())) == ledger
