from __future__ import annotations

import copy
from collections import defaultdict, namedtuple
from types import SimpleNamespace

from srcs.metagross.causal_reveal_ledger import (
    LEDGER_ATTRIBUTE,
    MOVE_RECEIPT_ATTRIBUTE,
)
from srcs.metagross.production_sampler_projection import (
    canonical_mechanical_projection,
    mechanical_projection_sha256,
)


Range = namedtuple("Range", ["min", "max"])


def fixture():
    move = SimpleNamespace(name="closecombat", current_pp=7, max_pp=8, disabled=False)
    pokemon = SimpleNamespace(
        name="mienshao", hp=201, max_hp=240, item="lifeorb",
        ability="regenerator", speed_range=Range(100, float("inf")), moves=[move],
        boosts=defaultdict(lambda: 0, {"attack": -1}), volatile_statuses=set(),
    )
    battle = SimpleNamespace(
        battle_tag="battle-cycle29", rqid=2, request_json={
            "rqid": 2,
            "active": [{"trapped": False, "moves": [{"id": "closecombat", "pp": 7, "disabled": False}]}],
        },
        force_switch=False, wait=False,
        opponent=SimpleNamespace(active=pokemon, reserve=[], trapped=False, side_conditions=defaultdict(lambda: 0)),
        user=SimpleNamespace(active=copy.deepcopy(pokemon), reserve=[], trapped=False, side_conditions=defaultdict(lambda: 0)),
    )
    setattr(battle, LEDGER_ATTRIBUTE, {"schema": "ledger", "protocol_sha256": "a" * 64})
    return battle


def test_projection_is_deterministic_and_handles_dynamic_range() -> None:
    value = fixture()
    assert canonical_mechanical_projection(value) == canonical_mechanical_projection(copy.deepcopy(value))
    assert mechanical_projection_sha256(value) == mechanical_projection_sha256(copy.deepcopy(value))


def test_receipt_is_separate_but_ledger_is_mechanical() -> None:
    value = fixture()
    baseline = mechanical_projection_sha256(value)
    setattr(value, MOVE_RECEIPT_ATTRIBUTE, {"moves": [], "derived_executions": []})
    assert mechanical_projection_sha256(value) == baseline
    getattr(value, MOVE_RECEIPT_ATTRIBUTE)["moves"].append({"move": "closecombat"})
    assert mechanical_projection_sha256(value) == baseline
    getattr(value, LEDGER_ATTRIBUTE)["protocol_sha256"] = "b" * 64
    assert mechanical_projection_sha256(value) != baseline


def test_projection_is_sensitive_to_sampling_search_fields() -> None:
    base = fixture()
    baseline = mechanical_projection_sha256(base)
    mutations = [
        lambda value: setattr(value.opponent.active, "hp", 200),
        lambda value: setattr(value.opponent.active, "item", "choicescarf"),
        lambda value: setattr(value.opponent.active.moves[0], "current_pp", 6),
        lambda value: setattr(value.opponent.active.moves[0], "disabled", True),
        lambda value: value.request_json["active"][0].__setitem__("trapped", True),
        lambda value: value.request_json["active"][0]["moves"][0].__setitem__("disabled", True),
        lambda value: value.opponent.side_conditions.__setitem__("spikes", 1),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(base)
        mutate(changed)
        assert mechanical_projection_sha256(changed) != baseline
