#!/usr/bin/env python3
"""Differential probe harness for recovering the damage-race v2 eval delta.

Builds controlled battle states isolating each suspected feature, runs
deterministic iterative-deepening search (id) on each, prints JSON.
Run under two different poke_engine wheels and diff the outputs:

  .venv-fp-priors/bin/python engine/probe_harness.py > /tmp/probe_patched.json
  .venv-pe-stock/bin/python  engine/probe_harness.py > /tmp/probe_stock.json
"""
import json
import sys

import poke_engine as pe


def mk_move(mid, pp=32):
    return pe.Move(id=mid, disabled=False, pp=pp)


def mk_mon(
    mid="dragonite", level=80, types=("dragon", "flying"), hp=300, maxhp=300,
    ability="multiscale", item="none", attack=250, defense=200,
    special_attack=220, special_defense=220, speed=180, status="none",
    moves=("tackle",), pps=None,
):
    pps = pps or [32] * len(moves)
    types = tuple(types) if len(types) == 2 else (types[0], "typeless")
    return pe.Pokemon(
        id=mid, level=level, types=tuple(types), base_types=tuple(types),
        hp=hp, maxhp=maxhp, ability=ability, base_ability=ability, item=item,
        attack=attack, defense=defense, special_attack=special_attack,
        special_defense=special_defense, speed=speed, status=status,
        rest_turns=0, sleep_turns=0, weight_kg=50.0,
        terastallized=False,
        moves=[mk_move(m, p) for m, p in zip(moves, pps)],
    )


def filler(idx):
    return mk_mon(mid=f"pikachu", types=("electric",), hp=0, maxhp=200,
                  ability="static", moves=("thundershock",))


def mk_side(active, bench=None, atk_boost=0, def_boost=0, spa_boost=0,
            spd_boost=0, spe_boost=0):
    mons = [active] + (bench or [])
    while len(mons) < 6:
        mons.append(filler(len(mons)))
    return pe.Side(
        pokemon=mons,
        attack_boost=atk_boost, defense_boost=def_boost,
        special_attack_boost=spa_boost, special_defense_boost=spd_boost,
        speed_boost=spe_boost,
    )


def mk_state(s1, s2):
    return pe.State(side_one=s1, side_two=s2)


def probe(state, ms=40):
    r = pe.id(state, ms)
    return {
        "depth": r.depth_searched,
        "s1": list(r.s1),
        "s2": list(r.s2),
        "matrix": [round(v, 4) for v in r.matrix],
    }


def main():
    probes = {}

    # A neutral attacker/defender pair used across probes
    def atk_mon(**kw):
        base = dict(mid="garchomp", types=("dragon", "ground"), hp=300,
                    maxhp=300, ability="roughskin", attack=280, defense=220,
                    special_attack=180, special_defense=200, speed=240,
                    moves=("earthquake", "outrage"), item="none")
        base.update(kw)
        return mk_mon(**base)

    def def_mon(**kw):
        base = dict(mid="corviknight", types=("flying", "steel"), hp=320,
                    maxhp=320, ability="pressure", attack=200, defense=270,
                    special_attack=140, special_defense=220, speed=150,
                    moves=("bravebird", "bodypress"), item="none")
        base.update(kw)
        return mk_mon(**base)

    # 1. baseline
    probes["baseline"] = mk_state(mk_side(atk_mon()), mk_side(def_mon()))

    # 2. fixed-damage: seismic toss attacker vs high-def wall
    probes["fixed_dmg_seismictoss"] = mk_state(
        mk_side(atk_mon(mid="blissey", types=("normal","typeless"), attack=20,
                        special_attack=140, defense=100, special_defense=300,
                        hp=550, maxhp=550, ability="naturalcure",
                        moves=("seismictoss", "softboiled"))),
        mk_side(def_mon()),
    )
    probes["no_fixed_dmg_control"] = mk_state(
        mk_side(atk_mon(mid="blissey", types=("normal","typeless"), attack=20,
                        special_attack=140, defense=100, special_defense=300,
                        hp=550, maxhp=550, ability="naturalcure",
                        moves=("pound", "softboiled"))),
        mk_side(def_mon()),
    )

    # 3. boosts: +2 attack vs 0
    probes["boost_plus2_atk"] = mk_state(
        mk_side(atk_mon(), atk_boost=2), mk_side(def_mon()))

    # 4. items
    for item in ("choiceband", "choicespecs", "lifeorb", "leftovers",
                 "blacksludge", "choicescarf"):
        probes[f"item_{item}"] = mk_state(
            mk_side(atk_mon(item=item)), mk_side(def_mon()))

    # 5. unaware defender vs boosted attacker
    probes["unaware_vs_boost"] = mk_state(
        mk_side(atk_mon(), atk_boost=2),
        mk_side(def_mon(ability="unaware")))
    probes["unaware_no_boost"] = mk_state(
        mk_side(atk_mon()),
        mk_side(def_mon(ability="unaware")))

    # 6. toxic status
    probes["toxic_on_attacker"] = mk_state(
        mk_side(atk_mon(status="toxic")), mk_side(def_mon()))
    probes["toxic_on_defender"] = mk_state(
        mk_side(atk_mon()), mk_side(def_mon(status="toxic")))

    # 7. PP stall: attacker with 1pp / 0pp moves
    probes["pp_low"] = mk_state(
        mk_side(atk_mon(pps=[1, 1])), mk_side(def_mon()))
    probes["pp_zero_one_move"] = mk_state(
        mk_side(atk_mon(pps=[0, 16])), mk_side(def_mon()))

    # 8. recovery on defender
    probes["defender_recovery"] = mk_state(
        mk_side(atk_mon()),
        mk_side(def_mon(moves=("bravebird", "roost"))))

    # 9. hp asymmetries (race position)
    probes["attacker_low_hp"] = mk_state(
        mk_side(atk_mon(hp=60)), mk_side(def_mon()))
    probes["defender_low_hp"] = mk_state(
        mk_side(atk_mon()), mk_side(def_mon(hp=60)))

    out = {}
    for name, st in probes.items():
        try:
            out[name] = probe(st)
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": str(e)}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
