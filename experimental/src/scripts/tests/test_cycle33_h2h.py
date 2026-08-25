import hashlib

from experimental.src.scripts.summarize_cycle33_h2h import identity, wilson
from experimental.src.scripts.watch_cycle33_registrations import validate


def pair():
    one="alpha|packed"; two="beta|packed"
    return {"pair_id":"pair","pair_index":1,"battle_seed":"1,2,3,4","team_1_packed":one,"team_2_packed":two,"team_1_sha256":hashlib.sha256(one.encode()).hexdigest(),"team_2_sha256":hashlib.sha256(two.encode()).hexdigest()}


def test_registration_assignments_cover_both_public_roles():
    frozen=pair()
    common={"schema_version":1,"pair_id":"pair","leg":1,"format":"gen9randombattle","battle_seed":"1,2,3,4","team_1_sha256":frozen["team_1_sha256"],"team_2_sha256":frozen["team_2_sha256"]}
    p1=validate({**common,"assigned_team_sha256":frozen["team_1_sha256"],"packed_team":frozen["team_1_packed"]},frozen,"c33h2hx1234567")
    p2=validate({**common,"assigned_team_sha256":frozen["team_2_sha256"],"packed_team":frozen["team_2_packed"]},frozen,"c33h2hy1234567")
    assert (p1["side"],p2["side"])==("p1","p2")


def test_receipt_identity_uses_immutable_decision_coordinates():
    row={"context":{"tag":"battle-test","rqid":7,"decision_idx":3}}
    tag,rqid,index,root=identity(row)
    assert (tag,rqid,index)==("battle-test",7,3)
    assert root==hashlib.sha256(b"terminal-mcts-live\0battle-test\0"+b"3").hexdigest()


def test_stage_one_wilson_is_descriptive_not_success_boundary():
    low,high=wilson(13,20)
    assert low < .5 < high
