from __future__ import annotations
import copy
from types import SimpleNamespace
from srcs.metagross.causal_reveal_ledger import LEDGER_ATTRIBUTE,MOVE_RECEIPT_ATTRIBUTE,bind_live_move_states,freeze_ledger,verify_sampled_move_states
def move(name,pp=1,max_pp=16,disabled=False):
    return SimpleNamespace(name=name,current_pp=pp,max_pp=max_pp,disabled=disabled,metagross_causal_pp_events=[])
def battle(pokemon,reserve=()):return SimpleNamespace(opponent=SimpleNamespace(active=pokemon,reserve=list(reserve)))
def trace(actor="p2a: Noivern"):
    return ["|switch|p1a: Ting-Lu|Ting-Lu, L80|100/100","|switch|p2a: Noivern|Noivern, L80|100/100",f"|move|{actor}|Struggle|p1a: Ting-Lu"]
def test_public_struggle_is_history_not_intrinsic_moveset_or_pp():
    ledger=freeze_ledger("battle-cycle43-preserved","p1",trace());fact=ledger.facts[0]
    assert fact.moves==() and fact.move_states==()
    assert [(x.move,x.authority,x.derived_cause) for x in fact.move_events]==[("struggle","derived_public_execution","mechanic: Struggle")]
    tracked=SimpleNamespace(name="noivern",moves=[move("hurricane",0),move("roost",0),move("dracometeor",0),move("uturn",0)])
    bound=bind_live_move_states(battle(tracked),ledger)
    assert bound.facts[0].move_states==() and [x.name for x in tracked.moves]==["hurricane","roost","dracometeor","uturn"]
def test_sampled_worlds_keep_actual_availability_and_never_gain_struggle():
    ledger=freeze_ledger("battle-worlds","p1",trace());source=battle(SimpleNamespace(name="noivern",moves=[]));bound=bind_live_move_states(source,ledger);setattr(source,LEDGER_ATTRIBUTE,bound.to_payload())
    exhausted=copy.deepcopy(source); exhausted.opponent.active.moves=[move("roost",0,16,True)]
    available=copy.deepcopy(source); available.opponent.active.moves=[move("roost",8,16,False)]
    worlds=[(exhausted,.4),(available,.6)];verify_sampled_move_states(source,worlds)
    assert [x.name for x in exhausted.opponent.active.moves]==["roost"] and [x.name for x in available.opponent.active.moves]==["roost"]
    assert all("struggle" not in [x.name for x in world.opponent.active.moves] for world,_ in worlds)
    assert all(getattr(world,MOVE_RECEIPT_ATTRIBUTE)["derived_executions"][0]["move"]=="struggle" for world,_ in worlds)
def test_hidden_completion_and_role_do_not_change_struggle_classification():
    left=freeze_ledger("battle-a","p1",trace());right=freeze_ledger("battle-a","p1",trace())
    assert left.canonical_bytes()==right.canonical_bytes()
    opposite=freeze_ledger("battle-b","p2",["|switch|p2a: Ting-Lu|Ting-Lu, L80|100/100","|switch|p1a: Noivern|Noivern, L80|100/100","|move|p1a: Noivern|Struggle|p2a: Ting-Lu"])
    assert opposite.facts[0].move_events[0].derived_cause=="mechanic: Struggle"
def test_intrinsic_pressure_disable_choice_encore_taunt_moves_remain_normal():
    for line in ("|move|p2a: Noivern|Hurricane|p1a: Ting-Lu","|move|p2a: Noivern|Roost|p2a: Noivern"):
        ledger=freeze_ledger("battle-control","p1",trace()[:2]+[line]);assert ledger.facts[0].moves
