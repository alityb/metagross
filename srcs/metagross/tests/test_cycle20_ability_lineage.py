from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from srcs.metagross.causal_reveal_ledger import (
    CausalRevealLedger,
    CausalRevealLedgerError,
    CausalMoveState,
    compile_reveal_bits,
    freeze_ledger,
)


ROOT = Path(__file__).resolve().parents[3]
CYCLE19_FAILURE_PROTOCOL = ROOT / (
    "experimental/runs/search_native_v2_cycle19_operational_repair_20260815/"
    "h2h-logs/c19h2hx002cb97.protocol.jsonl"
)


def public_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text().splitlines():
        row = json.loads(raw)
        if row.get("direction") not in {"received", "reconnect_received"}:
            continue
        lines.extend(
            line
            for line in str(row.get("message", "")).splitlines()
            if line.startswith("|") and not line.startswith("|request|")
        )
    return lines


def fake_state_for_ledger(ledger, hidden_suffix: str = "a"):
    pokemon = []
    for fact in ledger.facts:
        moves = [SimpleNamespace(id=move, pp=16, disabled=False) for move in fact.moves]
        moves.extend(SimpleNamespace(id="none") for _ in range(4 - len(moves)))
        pokemon.append(
            SimpleNamespace(
                id=fact.exact_public_species,
                item=fact.current_item or f"hiddenitem{hidden_suffix}",
                ability=fact.current_ability or f"hiddenability{hidden_suffix}",
                moves=moves,
            )
        )
    while len(pokemon) < 6:
        index = len(pokemon)
        pokemon.append(
            SimpleNamespace(
                id=f"hiddenmon{hidden_suffix}{index}",
                item=f"hiddenitem{hidden_suffix}{index}",
                ability=f"hiddenability{hidden_suffix}{index}",
                moves=[SimpleNamespace(id="none") for _ in range(4)],
            )
        )
    own = SimpleNamespace(
        pokemon=[
            SimpleNamespace(id=f"own{index}", item="none", ability="none", moves=[])
            for index in range(6)
        ]
    )
    return SimpleNamespace(side_one=own, side_two=SimpleNamespace(pokemon=pokemon))


def test_full_cycle19_failure_transcript_tracks_form_phase_and_compiles() -> None:
    ledger = freeze_ledger(
        "battle-cycle19-regression", "p1", public_lines(CYCLE19_FAILURE_PROTOCOL)
    )
    terapagos = next(fact for fact in ledger.facts if fact.species == "terapagos")
    assert terapagos.exact_public_species == "terapagosterastal"
    assert [event.ability for event in terapagos.ability_history] == [
        "terashift",
        "terashell",
    ]
    assert [event.authority for event in terapagos.ability_history] == [
        "explicit_public_event",
        "rule_implied_form_transition",
    ]
    assert terapagos.current_ability == "terashell"
    assert terapagos.current_ability_authority == "rule_implied_form_transition"
    payload = json.loads(ledger.canonical_bytes())
    assert payload["schema"] == "metagross-causal-public-reveal-ledger/v4"
    assert "ability" not in payload["facts"][0]
    assert CausalRevealLedger.from_payload(payload).canonical_bytes() == ledger.canonical_bytes()

    # Cycle 26 makes exact live PP/disable state mandatory before engine-mask
    # compilation.  This historical transcript fixture supplies an explicit
    # synthetic tracker state only for its hidden-noninterference assertion.
    ledger = replace(ledger, facts=tuple(
        replace(fact, move_states=tuple(
            CausalMoveState(move=move, current_pp=16, max_pp=16, disabled=False)
            for move in fact.moves
        ))
        for fact in ledger.facts
    ))

    first = compile_reveal_bits(fake_state_for_ledger(ledger, "a"), ledger, swap=False)
    second = compile_reveal_bits(fake_state_for_ledger(ledger, "b"), ledger, swap=False)
    assert first == second


def test_v2_payload_requires_ordered_history_for_current_ability() -> None:
    ledger = freeze_ledger("battle-history-contract", "p1", [
        "|switch|p1a: Own|Pikachu, L80|100/100",
        "|switch|p2a: Foe|Mew, L80|100/100",
        "|-ability|p2a: Foe|Levitate|[from] move: Skill Swap|[of] p1a: Own",
    ])
    payload = ledger.to_payload()
    payload["facts"][0]["ability_history"] = []
    with pytest.raises(CausalRevealLedgerError, match="history presence"):
        CausalRevealLedger.from_payload(payload)


@pytest.mark.parametrize(
    ("initial", "target", "ability"),
    [
        ("Aegislash", "Aegislash-Blade", "stancechange"),
        ("Minior-Meteor", "Minior-Violet", "shieldsdown"),
        ("Ogerpon-Wellspring", "Ogerpon-Wellspring-Tera", "embodyaspectwellspring"),
        ("Terapagos-Terastal", "Terapagos-Stellar", "teraformzero"),
    ],
)
def test_systematic_exact_form_transition_uses_unique_pinned_ability(
    initial: str, target: str, ability: str
) -> None:
    ledger = freeze_ledger("battle-form", "p1", [
        "|switch|p1a: Own|Pikachu, L80|100/100",
        f"|switch|p2a: Form|{initial}, L80|100/100",
        f"|detailschange|p2a: Form|{target}, L80",
    ])
    fact = ledger.facts[0]
    assert fact.exact_public_species.replace("-", "") == target.lower().replace("-", "")
    assert fact.current_ability == ability
    assert fact.ability_history[-1].authority == "rule_implied_form_transition"


def test_terapagos_retains_historical_shift_then_current_shell() -> None:
    ledger = freeze_ledger("battle-terapagos", "p1", [
        "|switch|p1a: Own|Pikachu, L80|100/100",
        "|switch|p2a: Pagos|Terapagos, L77|100/100",
        "|-activate|p2a: Pagos|ability: Tera Shift",
        "|detailschange|p2a: Pagos|Terapagos-Terastal, L77",
    ])
    fact = ledger.facts[0]
    assert fact.species == "terapagos"
    assert fact.exact_public_species == "terapagosterastal"
    assert [(event.ability, event.authority) for event in fact.ability_history] == [
        ("terashift", "explicit_public_event"),
        ("terashell", "rule_implied_form_transition"),
    ]
    assert fact.current_ability == "terashell"


def test_unique_species_ability_is_not_inferred_without_public_form_transition() -> None:
    ledger = freeze_ledger("battle-no-default", "p1", [
        "|switch|p1a: Own|Pikachu, L80|100/100",
        "|switch|p2a: Pagos|Terapagos-Terastal, L77|100/100",
    ])
    fact = ledger.facts[0]
    assert fact.exact_public_species == "terapagosterastal"
    assert fact.current_ability is None
    assert fact.ability_history == ()


def test_explicit_skill_swap_ability_is_supported_and_ordered() -> None:
    ledger = freeze_ledger("battle-skill-swap", "p1", [
        "|switch|p1a: Own|Bronzong, L80|100/100",
        "|switch|p2a: Foe|Mew, L80|100/100",
        "|-ability|p2a: Foe|Levitate|[from] move: Skill Swap|[of] p1a: Own",
    ])
    fact = ledger.facts[0]
    assert fact.current_ability == "levitate"
    assert fact.ability_history[-1].authority == "explicit_public_event"


def test_unsupported_ability_changing_transform_fails_closed() -> None:
    with pytest.raises(CausalRevealLedgerError, match="ability-changing transform"):
        freeze_ledger("battle-transform", "p1", [
            "|switch|p1a: Own|Mew, L80|100/100",
            "|switch|p2a: Ditto|Ditto, L80|100/100",
            "|-transform|p2a: Ditto|p1a: Own",
        ])


def test_form_transition_without_unique_pinned_ability_fails_closed() -> None:
    with pytest.raises(CausalRevealLedgerError, match="unsupported public form"):
        freeze_ledger("battle-unsupported-form", "p1", [
            "|switch|p1a: Own|Pikachu, L80|100/100",
            "|switch|p2a: Foe|Charizard-Mega-X, L80|100/100",
            "|detailschange|p2a: Foe|Charizard, L80",
        ])
