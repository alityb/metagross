from __future__ import annotations
import hashlib,json
from pathlib import Path
from experimental.src.scripts.cycle33_canonical_h2h import identity
from experimental.src.scripts.summarize_cycle42_one_deviation import fisher
ROOT=Path(__file__).resolve().parents[4]; RUN=ROOT/"experimental/runs/search_native_v2_cycle42_one_deviation_20260816"
def load(name):return json.loads((RUN/name).read_text())
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"))
def test_pair_config_and_identity_freshness():
    p=load("h2h-result.json.pairs.json"); r=load("PRIOR_IDENTITY_REGISTRY.json"); rows=p["pairs"]; path=RUN/"h2h-result.json.pairs.json"; h=hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity(RUN/"CANONICAL_H2H_ARGV.json","prepare")==identity(RUN/"CANONICAL_H2H_ARGV.json","live",h)==p["config_sha256"]
    pairs={canon(sorted((x["team_1_sha256"],x["team_2_sha256"]))) for x in rows}; teams={y for x in rows for y in(x["team_1_sha256"],x["team_2_sha256"])};ids={x["pair_id"] for x in rows};seeds={canon(x["battle_seed"]) for x in rows}
    assert (len(rows),len(pairs),len(teams),len(ids),len(seeds))==(10,10,20,10,10)
    assert not pairs&set(r["unordered_team_pairs"]);assert not teams&set(r["individual_team_sha256"]);assert not ids&set(r["pair_ids"]);assert not seeds&set(r["battle_seeds"])
def test_run_identity_and_registration_domain_fresh():
    r=load("PRIOR_IDENTITY_REGISTRY.json")
    assert "202642081616" not in r["mirror_seeds"] and "4242424242424242424242424242424242424242424242424242424242424242" not in r["production_run_seeds"]
    assert "cycle42-first-disagreement-one-deviation" not in r["run_ids"] and "c42od" not in r["username_prefixes"]
    assert not any(x.startswith("c42od") for x in r["usernames"]);assert list((RUN/"h2h-registrations").iterdir())==[]
def test_fisher_is_one_sided_greater():
    assert fisher(6,7,1,7)<fisher(4,7,3,7)
