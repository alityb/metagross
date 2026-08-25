from __future__ import annotations

import copy
import json
from types import SimpleNamespace

from srcs.metagross.causal_reveal_ledger import (
    LEDGER_ATTRIBUTE,
    MOVE_RECEIPT_ATTRIBUTE,
    bind_live_move_states,
    freeze_ledger,
)
from srcs.metagross.run_foul_play import (
    prepare_production_random_battles_with_causal_move_receipts,
)


def move(name: str, pp: int = 7, max_pp: int = 8, disabled: bool = False):
    return SimpleNamespace(
        name=name, current_pp=pp, max_pp=max_pp, disabled=disabled
    )


def battle(species: str, moves):
    return SimpleNamespace(
        opponent=SimpleNamespace(
            active=SimpleNamespace(name=species, moves=list(moves)), reserve=[]
        )
    )


def attach(source, lines):
    ledger = bind_live_move_states(
        source, freeze_ledger("battle-cycle28", "p1", lines)
    )
    setattr(source, LEDGER_ATTRIBUTE, ledger.to_payload())
    return ledger


def deterministic_sampler(source, count, rng=None):
    return [(copy.deepcopy(source), (index + 1) / count) for index in range(count)]


def test_opening_root_receipt_is_valid_and_empty() -> None:
    source = battle("mienshao", [])
    ledger = attach(source, [
        "|switch|p1a: Own|Empoleon, L84|100/100",
        "|switch|p2a: Foe|Mienshao, L83|100/100",
    ])
    worlds = prepare_production_random_battles_with_causal_move_receipts(
        deterministic_sampler, source, 16
    )
    assert len(worlds) == 16
    for world, _weight in worlds:
        receipt = getattr(world, MOVE_RECEIPT_ATTRIBUTE)
        assert receipt == {
            "schema": "metagross-causal-move-world-receipts/v1",
            "battle_tag": "battle-cycle28",
            "protocol_sha256": ledger.protocol_sha256,
            "moves": [],
            "derived_executions": [],
        }


def test_wrapper_preserves_world_identity_order_weights_and_mechanics() -> None:
    source = battle("mienshao", [move("closecombat")])
    attach(source, [
        "|switch|p1a: Own|Empoleon, L84|100/100",
        "|switch|p2a: Foe|Mienshao, L83|100/100",
        "|move|p2a: Foe|Close Combat|p1a: Own",
    ])
    worlds = prepare_production_random_battles_with_causal_move_receipts(
        deterministic_sampler, source, 32
    )
    assert [weight for _world, weight in worlds] == [
        (index + 1) / 32 for index in range(32)
    ]
    for world, _weight in worlds:
        assert world.opponent.active.name == "mienshao"
        sampled_move = world.opponent.active.moves[0]
        assert (
            sampled_move.name,
            sampled_move.current_pp,
            sampled_move.max_pp,
            sampled_move.disabled,
        ) == ("closecombat", 7, 8, False)
        receipt = getattr(world, MOVE_RECEIPT_ATTRIBUTE)
        assert receipt["moves"][0]["disable_authority"] == (
            "world_mechanical_disable"
        )


def test_derived_execution_is_receipted_but_not_intrinsic() -> None:
    source = battle("hatterene", [])
    attach(source, [
        "|switch|p1a: Own|Ting-Lu, L80|100/100",
        "|switch|p2a: Foe|Hatterene, L80|100/100",
        "|move|p2a: Foe|Stealth Rock|p1a: Own|[from] ability: Magic Bounce",
    ])
    [(world, _weight)] = prepare_production_random_battles_with_causal_move_receipts(
        deterministic_sampler, source, 1
    )
    receipt = getattr(world, MOVE_RECEIPT_ATTRIBUTE)
    assert receipt["moves"] == []
    assert receipt["derived_executions"] == [{
        "event_index": 2,
        "exact_public_species": "hatterene",
        "move": "stealthrock",
        "authority": "derived_public_execution",
        "derived_cause": "ability: Magic Bounce",
    }]


def test_receipt_sidecar_is_not_part_of_mechanical_projection() -> None:
    source = battle("mienshao", [move("closecombat", disabled=True)])
    attach(source, [
        "|switch|p1a: Own|Empoleon, L84|100/100",
        "|switch|p2a: Foe|Mienshao, L83|100/100",
        "|move|p2a: Foe|Close Combat|p1a: Own",
    ])
    [(plain, plain_weight)] = deterministic_sampler(source, 1)
    [(hooked, hooked_weight)] = prepare_production_random_battles_with_causal_move_receipts(
        deterministic_sampler, source, 1
    )
    delattr(hooked, MOVE_RECEIPT_ATTRIBUTE)
    assert plain_weight == hooked_weight
    assert json.dumps(plain, default=lambda value: vars(value), sort_keys=True) == (
        json.dumps(hooked, default=lambda value: vars(value), sort_keys=True)
    )
