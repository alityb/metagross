from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "srcs/vendor/foul-play"))

from fp.battle import Battle, Pokemon  # noqa: E402
from fp.battle_modifier import form_change, switch  # noqa: E402


def roster_row(
    species: str, *, role: str, nickname: str, active: bool = True,
    condition: str = "80/200", stats_seed: int = 100,
):
    return {
        "ident": f"{role}: {nickname}",
        "details": f"{species}, L80",
        "condition": condition,
        "active": active,
        "stats": {
            "atk": stats_seed,
            "def": stats_seed + 1,
            "spa": stats_seed + 2,
            "spd": stats_seed + 3,
            "spe": stats_seed + 4,
        },
        "moves": ["tackle"],
        "baseAbility": "shieldsdown",
        "ability": "shieldsdown",
        "item": "",
        "reviving": False,
    }


def request(active_row, *, force_switch: bool = False):
    return {
        "side": {"pokemon": [active_row]},
        "active": [{
            "moves": [{
                "id": "tackle", "move": "Tackle", "pp": 56,
                "maxpp": 56, "disabled": False, "target": "normal",
            }]
        }],
        "forceSwitch": [force_switch] if force_switch else None,
    }


def battle(role: str = "p1", *, species: str = "minioryellow", nickname: str = "Core"):
    value = Battle("cycle38-test")
    value.user.name = role
    value.opponent.name = "p2" if role == "p1" else "p1"
    value.user.active = Pokemon(species, 80)
    value.user.active.nickname = nickname
    value.user.active.hp = 100
    value.user.active.max_hp = 200
    return value


@pytest.mark.parametrize("role", ["p1", "p2"])
@pytest.mark.parametrize(
    "start,public_form,request_form",
    [
        ("minioryellow", "Minior-Meteor", "Minior-Yellow"),
        ("morpeko", "Morpeko-Hangry", "Morpeko"),
        ("terapagos", "Terapagos-Terastal", "Terapagos"),
    ],
)
def test_stale_request_never_hydrates_public_form_actor(
    role: str, start: str, public_form: str, request_form: str,
) -> None:
    value = battle(role, species=start)
    value.request_json = request(roster_row(
        "Skeledirge", role=role, nickname="Other", condition="1/999",
        stats_seed=900,
    ))
    form_change(value, ["", "-formechange", f"{role}a: Core", public_form])
    assert value.user.active.name != "skeledirge"
    assert value.user.active.hp == 100
    assert value.user.pending_form_request_hydration == {
        "public_ident": "Core",
        "request_ident": "Other",
        "exact_public_form": public_form.lower().replace("-", ""),
    }

    matching = request(roster_row(
        request_form, role=role, nickname="Core", condition="61/200", stats_seed=210,
    ))
    value.user.update_from_request_json(matching)
    assert value.user.pending_form_request_hydration is None
    assert value.user.active.hp == 61
    assert value.user.active.stats["attack"] == 210
    # Request identity is used for hydration only; it cannot rewrite the exact
    # public mechanical form already installed on the active object.
    assert value.user.active.name == public_form.lower().replace("-", "")


def test_deferred_delivery_matches_in_order_delivery_at_decision_boundary() -> None:
    matching = request(roster_row(
        "Minior-Yellow", role="p1", nickname="Core", condition="73/200",
        stats_seed=310,
    ))

    in_order = battle()
    in_order.request_json = copy.deepcopy(matching)
    form_change(in_order, ["", "detailschange", "p1a: Core", "Minior-Meteor"])
    in_order.user.update_from_request_json(copy.deepcopy(matching))

    delayed = battle()
    delayed.request_json = request(roster_row(
        "Skeledirge", role="p1", nickname="Other", condition="1/999",
        stats_seed=900,
    ))
    form_change(delayed, ["", "detailschange", "p1a: Core", "Minior-Meteor"])
    delayed.user.update_from_request_json(copy.deepcopy(matching))

    assert delayed.user.active.__dict__ == in_order.user.active.__dict__
    assert delayed.user.pending_form_request_hydration is None


def test_repeated_stale_requests_defer_but_strict_boundary_still_fails_closed() -> None:
    value = battle()
    stale = request(roster_row(
        "Skeledirge", role="p1", nickname="Other", condition="1/999",
    ), force_switch=True)
    value.request_json = stale
    form_change(value, ["", "-formechange", "p1a: Core", "Minior-Meteor"])
    first = copy.deepcopy(value.user.pending_form_request_hydration)
    form_change(value, ["", "-formechange", "p1a: Core", "Minior-Yellow"])
    assert value.user.pending_form_request_hydration == first | {
        "exact_public_form": "minioryellow"
    }
    with pytest.raises(ValueError, match="ident mismatch"):
        value.user.update_from_request_json(stale)
    assert value.user.pending_form_request_hydration is not None


def test_hidden_reserve_or_opponent_changes_cannot_affect_defer_decision() -> None:
    base = battle()
    base.request_json = request(roster_row(
        "Skeledirge", role="p1", nickname="Other", condition="1/999",
    ))
    perturbed = copy.deepcopy(base)
    perturbed.opponent.reserve = [Pokemon("mewtwo", 100), Pokemon("ditto", 100)]
    for value in (base, perturbed):
        form_change(value, ["", "-formechange", "p1a: Core", "Minior-Meteor"])
    assert base.user.active.__dict__ == perturbed.user.active.__dict__
    assert (
        base.user.pending_form_request_hydration
        == perturbed.user.pending_form_request_hydration
    )


def test_ambiguous_private_request_is_not_silently_deferred() -> None:
    value = battle()
    first = roster_row("Minior-Yellow", role="p1", nickname="Core")
    second = roster_row("Skeledirge", role="p1", nickname="Other")
    value.request_json = request(first)
    value.request_json["side"]["pokemon"].append(second)
    with pytest.raises(ValueError, match="exactly one"):
        form_change(value, ["", "-formechange", "p1a: Core", "Minior-Meteor"])


@pytest.mark.parametrize("crowned", ["Zacian-Crowned", "Zamazenta-Crowned"])
def test_crowned_public_switch_defers_stale_previous_active_request(crowned: str) -> None:
    value = battle(species="indeedeef", nickname="Indeedee")
    value.generation = "gen9"
    incoming = Pokemon(crowned.lower().replace("-", ""), 80)
    incoming.nickname = crowned.split("-", 1)[0]
    value.user.reserve = [incoming]
    value.request_json = request(roster_row(
        "Indeedee-F", role="p1", nickname="Indeedee", condition="71/200",
    ))

    switch(value, [
        "", "switch", f"p1a: {incoming.nickname}", f"{crowned}, L80", "100/100",
    ])

    assert value.user.active.name == crowned.lower().replace("-", "")
    assert value.user.pending_form_request_hydration == {
        "public_ident": incoming.nickname,
        "request_ident": "Indeedee",
        "exact_public_form": crowned.lower().replace("-", ""),
    }
