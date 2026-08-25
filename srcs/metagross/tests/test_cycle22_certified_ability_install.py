from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import poke_engine
from srcs.metagross.causal_reveal_ledger import (
    CausalRevealLedgerError,
    LEDGER_ATTRIBUTE,
    compile_reveal_bits,
    convert_battle_with_causal_ledger,
    freeze_ledger,
    hydrate_certified_abilities,
    norm,
)


ROOT = Path(__file__).resolve().parents[3]
CYCLE21_PROTOCOL = ROOT / (
    "experimental/runs/search_native_v2_cycle21_registered_form_smoke_20260815/"
    "smoke-logs/c21smkx0014ab0.protocol.jsonl"
)


def public_lines(path: Path) -> list[str]:
    result = []
    for raw in path.read_text().splitlines():
        row = json.loads(raw)
        if row.get("direction") not in {"received", "reconnect_received"}:
            continue
        result.extend(
            line for line in str(row.get("message", "")).splitlines()
            if line.startswith("|") and not line.startswith("|request|")
        )
    return result


def mon(
    species: str,
    ability: str,
    base_ability: str,
    move: str = "darkpulse",
    *,
    item: str = "none",
) -> poke_engine.Pokemon:
    return poke_engine.Pokemon(
        id=species, hp=100, maxhp=100, ability=ability,
        base_ability=base_ability, item=item,
        moves=[poke_engine.Move(move)],
    )


def cycle21_state(*, swap: bool = False, hidden_ability: str = "static"):
    own = mon("urshifu", "unseenfist", "unseenfist", "closecombat")
    target = mon("terapagosterastal", "terashift", "terashift")
    hidden = mon("pikachu", hidden_ability, hidden_ability, "tackle")
    opponent = poke_engine.Side(pokemon=[target, hidden])
    observer = poke_engine.Side(pokemon=[own])
    if swap:
        return poke_engine.State(side_one=opponent, side_two=observer)
    return poke_engine.State(side_one=observer, side_two=opponent)


def cycle21_ledger():
    return freeze_ledger("battle-cycle21-preserved", "p1", public_lines(CYCLE21_PROTOCOL))


def test_exact_cycle21_stale_state_hydrates_current_and_base() -> None:
    ledger = cycle21_ledger()
    before_ledger = ledger.canonical_bytes()
    state = cycle21_state()
    before = state.to_string()
    hydrated = hydrate_certified_abilities(state, ledger, swap=False)
    target = hydrated.side_two.pokemon[0]
    assert norm(target.id) == "terapagosterastal"
    assert norm(target.ability) == "terashell"
    assert norm(target.base_ability) == "terashell"
    assert ledger.canonical_bytes() == before_ledger
    assert before.replace("TERASHIFT,TERASHIFT", "TERASHELL,TERASHELL", 1) == hydrated.to_string()
    assert compile_reveal_bits(hydrated, ledger, swap=False) == (1 | (1 << 36))


def test_hidden_completion_perturbation_does_not_change_patch_or_bits() -> None:
    ledger = cycle21_ledger()
    left = hydrate_certified_abilities(cycle21_state(hidden_ability="static"), ledger, swap=False)
    right = hydrate_certified_abilities(cycle21_state(hidden_ability="lightningrod"), ledger, swap=False)
    for state in (left, right):
        target = state.side_two.pokemon[0]
        assert tuple(map(norm, (target.id, target.ability, target.base_ability))) == (
            "terapagosterastal", "terashell", "terashell"
        )
    assert compile_reveal_bits(left, ledger, swap=False) == compile_reveal_bits(right, ledger, swap=False)
    assert norm(left.side_two.pokemon[1].ability) == "static"
    assert norm(right.side_two.pokemon[1].ability) == "lightningrod"


def test_request_actions_roundtrip_and_real_apply_reverse_are_exact() -> None:
    ledger = cycle21_ledger()
    state = cycle21_state()
    request_actions = ["closecombat"]
    before = poke_engine.root_options_with_s1_request(state, request_actions)
    hydrated = hydrate_certified_abilities(state, ledger, swap=False)
    assert poke_engine.root_options_with_s1_request(hydrated, request_actions) == before
    serialized = hydrated.to_string()
    assert poke_engine.State.from_string(serialized).to_string() == serialized
    instructions = poke_engine.generate_instructions(hydrated, "closecombat", "darkpulse")[0]
    advanced = hydrated.apply_instructions(instructions)
    assert advanced.reverse_instructions(instructions).to_string() == serialized


def test_perspective_swap_installs_only_side_one_exact_slot() -> None:
    original = cycle21_ledger()
    swapped = type(original)(
        battle_tag=original.battle_tag,
        observer_role="p2", opponent_role="p1",
        opponent_active_species=original.opponent_active_species,
        facts=original.facts, protocol_sha256=original.protocol_sha256,
    )
    state = cycle21_state(swap=True)
    hydrated = hydrate_certified_abilities(state, swapped, swap=True)
    assert norm(hydrated.side_one.pokemon[0].ability) == "terashell"
    assert norm(hydrated.side_one.pokemon[0].base_ability) == "terashell"
    assert norm(hydrated.side_two.pokemon[0].ability) == "unseenfist"


def test_explicit_skill_swap_changes_current_but_preserves_base() -> None:
    ledger = freeze_ledger("battle-skill-swap-cycle22", "p1", [
        "|switch|p1a: Own|Bronzong, L80|100/100",
        "|switch|p2a: Foe|Mew, L80|100/100",
        "|-ability|p2a: Foe|Levitate|[from] move: Skill Swap|[of] p1a: Own",
    ])
    state = poke_engine.State(
        side_one=poke_engine.Side(pokemon=[mon("bronzong", "levitate", "levitate")]),
        side_two=poke_engine.Side(pokemon=[mon("mew", "synchronize", "synchronize")]),
    )
    hydrated = hydrate_certified_abilities(state, ledger, swap=False)
    assert norm(hydrated.side_two.pokemon[0].ability) == "levitate"
    assert norm(hydrated.side_two.pokemon[0].base_ability) == "synchronize"


@pytest.mark.parametrize(
    ("initial", "target", "stale", "expected"),
    [
        ("Aegislash", "Aegislash-Blade", "stancechange", "stancechange"),
        ("Minior-Meteor", "Minior-Violet", "shieldsdown", "shieldsdown"),
        ("Ogerpon-Wellspring", "Ogerpon-Wellspring-Tera", "waterabsorb", "embodyaspectwellspring"),
        ("Terapagos-Terastal", "Terapagos-Stellar", "terashell", "teraformzero"),
    ],
)
def test_rule_implied_forms_install_current_and_base(initial, target, stale, expected) -> None:
    ledger = freeze_ledger("battle-systematic-cycle22", "p1", [
        "|switch|p1a: Own|Pikachu, L80|100/100",
        f"|switch|p2a: Form|{initial}, L80|100/100",
        f"|detailschange|p2a: Form|{target}, L80",
    ])
    exact = ledger.facts[0].exact_public_species
    state = poke_engine.State(
        side_one=poke_engine.Side(pokemon=[mon("pikachu", "static", "static", "tackle")]),
        side_two=poke_engine.Side(pokemon=[mon(exact, stale, stale)]),
    )
    hydrated = hydrate_certified_abilities(state, ledger, swap=False)
    assert norm(hydrated.side_two.pokemon[0].ability) == expected
    assert norm(hydrated.side_two.pokemon[0].base_ability) == expected


def test_missing_or_ambiguous_exact_form_fails_closed() -> None:
    ledger = cycle21_ledger()
    missing = poke_engine.State(
        side_one=poke_engine.Side(pokemon=[mon("urshifu", "unseenfist", "unseenfist")]),
        side_two=poke_engine.Side(pokemon=[mon("terapagos", "terashift", "terashift")]),
    )
    with pytest.raises(CausalRevealLedgerError, match="not unique"):
        hydrate_certified_abilities(missing, ledger, swap=False)
    duplicate = poke_engine.State(
        side_one=missing.side_one,
        side_two=poke_engine.Side(pokemon=[
            mon("terapagosterastal", "terashift", "terashift"),
            mon("terapagosterastal", "terashift", "terashift"),
        ]),
    )
    with pytest.raises(CausalRevealLedgerError, match="not unique"):
        hydrate_certified_abilities(duplicate, ledger, swap=False)


def test_conversion_hydrates_then_installs_mask_without_other_changes() -> None:
    ledger = cycle21_ledger()
    battle = SimpleNamespace(**{LEDGER_ATTRIBUTE: ledger.to_payload()})
    original = cycle21_state()
    converted = convert_battle_with_causal_ledger(
        battle, lambda _battle, swap=False: original, poke_engine, swap=False
    )
    assert norm(converted.side_two.pokemon[0].ability) == "terashell"
    assert norm(converted.side_two.pokemon[0].base_ability) == "terashell"
    assert converted.s1_public_reveals == (1 | (1 << 36))
    assert converted.s2_public_reveals == 0


def test_live_receipt_records_only_certified_post_installation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("METAGROSS_CAUSAL_ABILITY_RECEIPT_DIR", str(tmp_path))
    monkeypatch.setenv("METAGROSS_PRIOR_NAMESPACE", "agent_a")
    ledger = cycle21_ledger()
    hydrate_certified_abilities(cycle21_state(), ledger, swap=False)
    paths = list(tmp_path.glob("agenta-*.jsonl"))
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text())
    assert payload["schema"] == "metagross-certified-ability-installation/v1"
    assert payload["installations"] == [{
        "authority": "rule_implied_form_transition",
        "exact_public_species": "terapagosterastal",
        "installed_base_ability": "terashell",
        "installed_current_ability": "terashell",
        "slot": 0,
        "update_base": True,
    }]
    assert "before" not in paths[0].read_text()
