from types import SimpleNamespace

from experimental.src.scripts.audit_cycle14_mechanics_repair import (
    PREFIX_BY_PROTOCOL, compile_slot_aware_bits, freeze_with_prefix,
    latest_observed_forms,
)


def _pokemon(species, *, move="none", item="none", ability="none"):
    return SimpleNamespace(
        id=species, item=item, ability=ability,
        moves=[SimpleNamespace(id=move), *[SimpleNamespace(id="none") for _ in range(3)]],
    )


def test_slot_aware_form_selects_revealed_cosmetic_form_not_hidden_base_collision():
    lines = [
        "|switch|p1a: Own|Pikachu, L80|100/100",
        "|switch|p2a: Deer|Sawsbuck-Winter, L80|100/100",
        "|move|p2a: Deer|Swords Dance|p1a: Own",
    ]
    ledger = freeze_with_prefix("battle", "p1", lines)
    none = _pokemon("none")
    state = SimpleNamespace(
        side_one=SimpleNamespace(pokemon=[none] * 6),
        side_two=SimpleNamespace(pokemon=[
            _pokemon("sawsbuckwinter", move="swordsdance"),
            _pokemon("sawsbuck", move="tackle"), *([none] * 4),
        ]),
    )
    bits = compile_slot_aware_bits(state, ledger, swap=False)
    assert bits & 1
    assert bits & (1 << 6)
    assert not bits & (1 << 1)


def test_latest_observed_form_tracks_form_change_for_same_canonical_species():
    forms = latest_observed_forms([
        "|switch|p2a: Ice|Eiscue, L80|100/100",
        "|detailschange|p2a: Ice|Eiscue-Noice, L80",
    ], "p2")
    assert forms["eiscue"] == "eiscuenoice"
