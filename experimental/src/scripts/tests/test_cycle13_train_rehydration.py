import pytest
from types import SimpleNamespace

from experimental.src.scripts.audit_cycle13_train_rehydration import (
    Cycle13Error, failure_category, prefix_delta, remove_initial_own_switch,
    reconcile_public_facts_into_battle, request_actions_exact, stable_seed,
    verify_engine_contract,
)
from srcs.metagross.causal_reveal_ledger import freeze_ledger


def test_prefix_delta_requires_causal_extension():
    assert prefix_delta(["a"], ["a", "b"]) == ["b"]
    with pytest.raises(Cycle13Error):
        prefix_delta(["a"], ["x", "b"])


def test_remove_initial_own_switch_removes_only_first_observer_switch():
    rows = [
        "|switch|p1a: A|A, L80|100/100",
        "|switch|p2a: B|B, L80|100/100",
        "|switch|p1a: C|C, L80|100/100",
    ]
    assert remove_initial_own_switch(rows, "p1") == rows[1:]


def test_stable_seed_is_repeatable_and_schedule_specific():
    assert stable_seed("state", 0) == stable_seed("state", 0)
    assert stable_seed("state", 0) != stable_seed("state", 1)


def test_request_actions_accepts_old_request_without_rqid():
    request = {
        "active": [{
            "moves": [
                {"id": "thunderbolt", "pp": 8, "disabled": False},
                {"id": "protect", "pp": 0, "disabled": False},
            ],
            "canTerastallize": "Electric",
        }],
        "side": {"pokemon": [
            {"active": True, "condition": "100/100", "details": "Raichu, L80"},
            {"active": False, "condition": "90/100", "details": "Sawsbuck-Winter, L80"},
            {"active": False, "condition": "0 fnt", "details": "Mew, L80"},
        ]},
    }
    assert request_actions_exact(request) == {
        "thunderbolt", "thunderbolt-tera", "switch sawsbuckwinter",
    }


def test_reconcile_installs_only_public_facts_and_requires_existing_move_pp():
    ledger = freeze_ledger("battle", "p1", [
        "|switch|p1a: Own|Raichu, L80|100/100",
        "|switch|p2a: Doll|Banette, L80|100/100",
        "|move|p2a: Doll|Gunk Shot|p1a: Own",
        "|-start|p1a: Own|Disable|Thunderbolt|[from] ability: Cursed Body|[of] p2a: Doll",
        "|-enditem|p2a: Doll|Life Orb|[damage]",
    ])
    banette = SimpleNamespace(
        name="banette", moves=[SimpleNamespace(name="gunkshot", current_pp=6)],
        item="unknownitem", removed_item=None, ability=None,
    )
    battle = SimpleNamespace(opponent=SimpleNamespace(active=banette, reserve=[]))
    reconcile_public_facts_into_battle(battle, ledger)
    assert banette.ability == "cursedbody"
    assert banette.item is None
    assert banette.removed_item == "lifeorb"
    banette.moves = []
    with pytest.raises(Cycle13Error, match="PP state"):
        reconcile_public_facts_into_battle(battle, ledger)


def test_combined_engine_contract_has_tera_masks_and_seeded_step():
    import poke_engine

    verify_engine_contract(poke_engine)


def test_failure_categories_keep_causal_breaches_explicit():
    assert failure_category(
        "production_schedules", RuntimeError("public ability mismatch: banette")
    ) == "causal_fact_integrity"
    assert failure_category(
        "production_schedules", RuntimeError("hidden completions changed public projection")
    ) == "hidden_noninterference"
    assert failure_category(
        "production_schedules", RuntimeError("exact request actions disagree")
    ) == "action_mapping"
