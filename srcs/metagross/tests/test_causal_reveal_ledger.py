from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from srcs.metagross.causal_reveal_ledger import (
    CausalRevealLedger,
    CausalRevealLedgerError,
    LEDGER_ATTRIBUTE,
    attached_ledger,
    canonical_species,
    compile_reveal_bits,
    clear_public_protocol_lines,
    freeze_ledger,
    form_ability_contract,
    protocol_lines_for_battle,
    parse_state_serialization,
    record_public_protocol_lines,
    serialization_without_masks,
    serialize_state_fields,
    verify_sampled_ledgers,
)


PUBLIC_LINES = [
    "|switch|p1a: Own|Pikachu, L80|100/100",
    "|switch|p2a: Mask|Ogerpon-Wellspring-Tera, L80|100/100",
    "|move|p2a: Mask|Ivy Cudgel|p1a: Own",
    "|-ability|p2a: Mask|Embody Aspect (Wellspring)",
    "|-enditem|p2a: Mask|Wellspring Mask|[from] move: Knock Off|[of] p1a: Own",
]


def test_public_ledger_is_causal_json_safe_and_preserves_consumed_item() -> None:
    ledger = freeze_ledger("battle-test", "p1", PUBLIC_LINES)
    assert ledger.facts[0].species == "ogerponwellspring"
    assert ledger.facts[0].exact_public_species == "ogerponwellspringtera"
    assert ledger.facts[0].moves == ("ivycudgel",)
    assert ledger.facts[0].ability == "embodyaspectwellspring"
    assert ledger.facts[0].current_ability_authority == "explicit_public_event"
    assert [event.ability for event in ledger.facts[0].ability_history] == [
        "embodyaspectwellspring"
    ]
    assert ledger.facts[0].current_item == "none"
    assert ledger.facts[0].consumed_items == ("wellspringmask",)
    payload = json.loads(ledger.canonical_bytes())
    assert CausalRevealLedger.from_payload(payload).canonical_bytes() == ledger.canonical_bytes()


def test_consumed_item_cause_does_not_resurrect_current_item() -> None:
    ledger = freeze_ledger("battle-berry", "p1", [
        "|switch|p1a: Own|Pikachu, L80|100/100",
        "|switch|p2a: Ice|Eiscue, L80|100/100",
        "|-enditem|p2a: Ice|Sitrus Berry|[eat]",
        "|-heal|p2a: Ice|76/100|[from] item: Sitrus Berry",
    ])
    assert ledger.facts[0].current_item == "none"
    assert ledger.facts[0].consumed_items == ("sitrusberry",)


def test_source_pinned_form_contract_handles_battle_and_cosmetic_forms() -> None:
    assert canonical_species("Ogerpon-Wellspring-Tera") == "ogerponwellspring"
    assert canonical_species("Sawsbuck-Summer") == "sawsbuck"
    assert canonical_species("Minior-Violet") == "minior"
    contract = form_ability_contract()
    assert contract["terapagosterastal"] == "terashell"
    assert contract["terapagosstellar"] == "teraformzero"
    assert contract["ogerponwellspringtera"] == "embodyaspectwellspring"


def test_receive_capture_excludes_request_and_deduplicates_full_reconnect() -> None:
    tag = "battle-capture"
    clear_public_protocol_lines(tag)
    record_public_protocol_lines(tag, [PUBLIC_LINES[0], "|request|{\"secret\":1}"])
    record_public_protocol_lines(tag, PUBLIC_LINES)
    assert protocol_lines_for_battle(tag) == tuple(PUBLIC_LINES)
    assert all(not line.startswith("|request|") for line in protocol_lines_for_battle(tag))
    clear_public_protocol_lines(tag)


def test_sampler_must_preserve_identical_sidecar() -> None:
    ledger = freeze_ledger("battle-test", "p1", PUBLIC_LINES)
    source = SimpleNamespace(**{LEDGER_ATTRIBUTE: ledger.to_payload()})
    good = SimpleNamespace(**{LEDGER_ATTRIBUTE: json.loads(ledger.canonical_bytes())})
    verify_sampled_ledgers(source, [(good, 1.0)])
    bad_payload = ledger.to_payload()
    bad_payload["facts"][0]["moves"] = []
    bad = SimpleNamespace(**{LEDGER_ATTRIBUTE: bad_payload})
    with pytest.raises(CausalRevealLedgerError, match="sampler changed"):
        verify_sampled_ledgers(source, [(bad, 1.0)])


def test_missing_certified_move_fails_closed_without_hydration() -> None:
    ledger = freeze_ledger("battle-test", "p1", PUBLIC_LINES)
    pokemon = SimpleNamespace(
        id="ogerponwellspringtera", item="none", ability="embodyaspectwellspring",
        moves=[SimpleNamespace(id="none") for _ in range(4)],
    )
    none = SimpleNamespace(id="none", item="none", ability="none", moves=[])
    state = SimpleNamespace(
        side_one=SimpleNamespace(pokemon=[none] * 6),
        side_two=SimpleNamespace(pokemon=[pokemon, *([none] * 5)]),
    )
    with pytest.raises(CausalRevealLedgerError, match="PP-disable authority missing"):
        compile_reveal_bits(state, ledger, swap=False)


def test_attached_ledger_rejects_missing_payload() -> None:
    with pytest.raises(CausalRevealLedgerError, match="lacks causal ledger"):
        attached_ledger(SimpleNamespace())


def _nonzero_matrix_state():
    import poke_engine

    fields = parse_state_serialization(poke_engine.State().to_string())
    fields["s1_threat"] = "1.25"
    fields["s2_threat"] = "-2.5"
    fields["scout_value"] = "3.75"
    fields["threat_matrix"] = ";".join(str(index + 1) for index in range(36))
    fields["wincon_matrix"] = ";".join(str(-(index + 1)) for index in range(36))
    return poke_engine.State.from_string(serialize_state_fields(fields))


def test_named_13_field_grammar_roundtrips_and_rejects_field_count() -> None:
    state = _nonzero_matrix_state()
    serialized = state.to_string()
    assert serialize_state_fields(parse_state_serialization(serialized)) == serialized
    with pytest.raises(CausalRevealLedgerError, match="expected 13"):
        parse_state_serialization(serialized + "/extra")
    fields = parse_state_serialization(serialized)
    fields.pop("wincon_matrix")
    with pytest.raises(CausalRevealLedgerError, match="field names disagree"):
        serialize_state_fields(fields)


@pytest.mark.parametrize("bits", [1, 3, (1 << 17) | 5, (1 << 42) - 1])
def test_native_mask_setters_change_only_named_mask_and_clear_exactly(bits: int) -> None:
    state = _nonzero_matrix_state()
    original = parse_state_serialization(state.to_string())
    side_one = state.with_side_one_public_reveals(bits)
    one = parse_state_serialization(side_one.to_string())
    assert one == {**original, "s1_public_reveals": str(bits)}
    side_two = state.with_side_two_public_reveals(bits)
    two = parse_state_serialization(side_two.to_string())
    assert two == {**original, "s2_public_reveals": str(bits)}
    both = side_one.with_side_two_public_reveals(bits)
    restored = both.with_side_one_public_reveals(0).with_side_two_public_reveals(0)
    assert restored.to_string() == state.to_string()
    assert serialization_without_masks(both) == state.to_string()
