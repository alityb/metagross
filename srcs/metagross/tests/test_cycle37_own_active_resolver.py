from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "srcs/vendor/foul-play"))

from fp import battle as battle_module  # noqa: E402
from fp.battle import Pokemon, resolve_own_active_request_row  # noqa: E402


def row(species: str, *, role: str = "p1", nickname: str = "Foe", active=True):
    return {
        "ident": f"{role}: {nickname}",
        "details": f"{species}, L80",
        "condition": "100/100",
        "active": active,
        "stats": {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        "moves": ["tackle"],
        "baseAbility": "pressure",
        "ability": "pressure",
        "item": "",
        "reviving": False,
    }


def request(active_row, *reserve):
    return {"side": {"pokemon": [active_row, *reserve]}, "active": [{"moves": []}]}


def mon(species: str, nickname: str = "Foe") -> Pokemon:
    value = Pokemon(species, 80)
    value.nickname = nickname
    return value


@pytest.mark.parametrize(
    "tracked,requested",
    [
        ("terapagos", "Terapagos-Stellar"),
        ("terapagosterastal", "Terapagos"),
        ("terapagosstellar", "Terapagos-Terastal"),
        ("morpekohangry", "Morpeko"),
        ("miniormeteor", "Minior-Yellow"),
        ("minioryellow", "Minior-Meteor"),
        ("ogerponwellspringtera", "Ogerpon-Wellspring"),
        ("ogerponwellspring", "Ogerpon-Wellspring-Tera"),
        ("aegislashblade", "Aegislash"),
        ("aegislash", "Aegislash-Blade"),
        ("alcremiematchacream", "Alcremie"),
    ],
)
def test_systematic_battle_form_identity_selects_exact_active_row(
    tracked: str, requested: str
) -> None:
    active = mon(tracked)
    before = copy.deepcopy(active.__dict__)
    wanted = row(requested)
    selected = resolve_own_active_request_row(
        active, request(wanted, row("Pikachu", nickname="Bench", active=False))
    )
    assert selected is wanted
    assert active.__dict__ == before


@pytest.mark.parametrize("role", ["p1", "p2"])
def test_ident_authority_works_for_both_roles(role: str) -> None:
    active = mon("terapagos", "Shell")
    wanted = row("Terapagos-Stellar", role=role, nickname="Shell")
    assert resolve_own_active_request_row(active, request(wanted)) is wanted


def test_cycle36_preserved_stellar_request_reinitializes_without_form_rewrite() -> None:
    battler = battle_module.Battler()
    battler.active = mon("terapagos")
    battler.active.stats = {"attack": 1, "defense": 1, "special-attack": 1,
                            "special-defense": 1, "speed": 1}
    wanted = row("Terapagos-Stellar")
    wanted["condition"] = "41/373"
    wanted["stats"] = {"atk": 206, "def": 214, "spa": 245, "spd": 214, "spe": 175}
    battler.re_initialize_active_pokemon_from_request_json(request(wanted))
    assert battler.active.name == "terapagos"
    assert battler.active.base_name == "terapagos"
    assert battler.active.hp == 41
    assert battler.active.stats == {
        "attack": 206,
        "defense": 214,
        "special-attack": 245,
        "special-defense": 214,
        "speed": 175,
    }


def test_zero_or_multiple_active_rows_fail_closed() -> None:
    active = mon("terapagos")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_own_active_request_row(active, request(row("Terapagos", active=False)))
    with pytest.raises(ValueError, match="exactly one"):
        resolve_own_active_request_row(
            active, request(row("Terapagos"), row("Terapagos-Stellar"))
        )


def test_ident_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="ident mismatch"):
        resolve_own_active_request_row(
            mon("terapagos", "Expected"), request(row("Terapagos", nickname="Other"))
        )


def test_transform_or_unexplained_illusion_identity_mismatch_fails_closed() -> None:
    transformed = mon("ditto")
    transformed.volatile_statuses.append("transform")
    with pytest.raises(ValueError, match="public/private identity mismatch"):
        resolve_own_active_request_row(transformed, request(row("Mew")))
    disguised = mon("pikachu")
    disguised.zoroark_disguised_as = "pikachu"
    with pytest.raises(ValueError, match="Illusion identity mismatch"):
        resolve_own_active_request_row(disguised, request(row("Zoroark")))


def test_contract_is_deterministic_and_sensitive() -> None:
    first = battle_module._public_form_contract()
    second = battle_module._public_form_contract()
    assert first is second
    assert len(first) >= 100
    assert first["terapagosstellar"] == "terapagos"
    assert first["morpekohangry"] == "morpeko"
    assert first["aegislashblade"] == "aegislash"
