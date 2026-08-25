from __future__ import annotations
import json
import pytest
from srcs.metagross.terminal_mcts_one_deviation import (
    EQUAL8192_CONTRACT,EQUAL8192_SCHEMA,OneDeviationController,assignment_manifest,
)
SEED="202642160842"; PREFIX="c42od"
def username(index:int)->str:return f"{PREFIX}{'x' if index%2 else 'y'}{index:03d}abcd"
def teacher(production="move 1",selected="switch 2",decision="override"):
    return {"schema":"metagross-terminal-mcts-live-decision/v1","controller_schema":"metagross-cycle19-equal8192-production-selector/v1","decision":decision,"production_action":production,"selected_action":selected if decision=="override" else production,"reason":"frozen_equal8192_production_considered_visit_policy","iterations_per_world":8192,"schedule_count":2,"world_count":16,"receipts":[{"total_visits":8192} for _ in range(16)]}
def controller():return OneDeviationController(seed=SEED,username_prefix=PREFIX,teacher_contract=EQUAL8192_CONTRACT)
def test_assignment_is_frozen_10_10_and_5_5():
    rows=assignment_manifest(SEED)["assignments"]
    assert sum(r["arm"]=="teacher" for r in rows)==10
    assert sum(r["arm"]=="teacher" and r["pair_leg"]==1 for r in rows)==5
    assert sum(r["arm"]=="teacher" and r["pair_leg"]==2 for r in rows)==5
    assert all({r["arm"] for r in rows if r["pair_index"]==p}=={"teacher","production"} for p in range(1,11))
def test_equal8192_first_disagreement_then_permanent_lock():
    c=controller(); i=next(r["game_index"] for r in assignment_manifest(SEED)["assignments"] if r["arm"]=="teacher"); u=username(i)
    action,row=c.observe(battle_tag="battle-fresh",username=u,decision_index=1,production_choice="move 1",teacher=teacher(selected="move 1",decision="abstain"))
    assert action=="move 1" and not row["eligible"] and not row["locked_after_decision"]
    action,row=c.observe(battle_tag="battle-fresh",username=u,decision_index=2,production_choice="move 1",teacher=teacher())
    assert action=="switch 2" and row["eligible"] and row["intervention_applied"]
    assert row["schema"]==EQUAL8192_SCHEMA and not c.should_query("battle-fresh",u)
def test_production_arm_observes_same_opportunity_but_keeps_production():
    c=controller(); i=next(r["game_index"] for r in assignment_manifest(SEED)["assignments"] if r["arm"]=="production"); u=username(i)
    action,row=c.observe(battle_tag="battle-fresh",username=u,decision_index=2,production_choice="move 1",teacher=teacher())
    assert action=="move 1" and row["eligible"] and not row["intervention_applied"]
def test_baseline_must_equal_actual_production_and_receipts_are_exact():
    for mutate in (lambda t:t.update(production_action="move 9"),lambda t:t.update(world_count=15),lambda t:t["receipts"][0].update(total_visits=8191),lambda t:t.update(controller_schema="wrong")):
        c=controller(); t=teacher(); mutate(t)
        action,row=c.observe(battle_tag="battle-fresh",username=username(1),decision_index=1,production_choice="move 1",teacher=t)
        assert action=="move 1" and row["integrity_failure"]=="invalid_certified_deviation" and not c.should_query("battle-fresh",username(1))
def test_randomization_schedule_hash_is_deterministic():
    assert json.dumps(assignment_manifest(SEED),sort_keys=True)==json.dumps(assignment_manifest(SEED),sort_keys=True)
