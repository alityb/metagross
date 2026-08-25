from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experimental.src.scripts.cycle33_canonical_h2h import identity
from experimental.src.scripts.watch_cycle40_registrations import validate


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "experimental/runs/search_native_v2_cycle40_integrated_h2h_20260816"


def load_pairs() -> list[dict]:
    return json.loads((RUN / "h2h-result.json.pairs.json").read_text())["pairs"]


def test_prepare_and_live_config_identity_match_pair_manifest() -> None:
    canonical = RUN / "CANONICAL_H2H_ARGV.json"
    pair_path = RUN / "h2h-result.json.pairs.json"
    pair_sha = hashlib.sha256(pair_path.read_bytes()).hexdigest()
    payload = json.loads(pair_path.read_text())
    assert identity(canonical, "prepare") == identity(canonical, "live", pair_sha)
    assert identity(canonical, "prepare") == payload["config_sha256"]


def test_current_pairs_teams_ids_and_seeds_are_all_fresh() -> None:
    registry = json.loads((RUN / "PRIOR_IDENTITY_REGISTRY_V2.json").read_text())
    old_pairs = set(registry["unordered_team_pairs"])
    old_teams = set(registry["individual_team_sha256"])
    old_ids = set(registry["pair_ids"])
    old_seeds = set(registry["battle_seeds"])
    current_pairs = set()
    current_teams = set()
    current_ids = set()
    current_seeds = set()
    for row in load_pairs():
        current_pairs.add(json.dumps(sorted((row["team_1_sha256"], row["team_2_sha256"])), separators=(",", ":")))
        current_teams.update((row["team_1_sha256"], row["team_2_sha256"]))
        current_ids.add(row["pair_id"])
        current_seeds.add(json.dumps(row["battle_seed"], sort_keys=True, separators=(",", ":")))
    assert len(load_pairs()) == len(current_pairs) == len(current_ids) == len(current_seeds) == 10
    assert len(current_teams) == 20
    assert not current_pairs & old_pairs
    assert not current_teams & old_teams
    assert not current_ids & old_ids
    assert not current_seeds & old_seeds


def test_schedule_and_username_domain_are_fresh() -> None:
    registry = json.loads((RUN / "PRIOR_IDENTITY_REGISTRY_V2.json").read_text())
    assert "202640391638" not in registry["mirror_seeds"]
    assert "4040404040404040404040404040404040404040404040404040404040404040" not in registry["production_run_seeds"]
    assert "cycle40-integrated-equal8192-h2h" not in registry["run_ids"]
    assert "c40h2h" not in registry["username_prefixes"]
    assert not any(name.startswith("c40h2h") for name in registry["usernames"])


@pytest.mark.parametrize("suffix,side", [("x001abcd", "p1"), ("y001abcd", "p2")])
def test_registration_identity_and_packed_team(suffix: str, side: str) -> None:
    pair = load_pairs()[0]
    payload = {
        "schema_version": 1,
        "pair_id": pair["pair_id"],
        "format": "gen9randombattle",
        "battle_seed": pair["battle_seed"],
        "team_1_sha256": pair["team_1_sha256"],
        "team_2_sha256": pair["team_2_sha256"],
        "leg": 1,
        "assigned_team_sha256": pair["team_1_sha256"],
        "packed_team": pair["team_1_packed"],
    }
    assert validate(payload, pair, "c40h2h" + suffix)["side"] == side


def test_registration_rejects_prior_cycle_identity() -> None:
    with pytest.raises(RuntimeError, match="unexpected Cycle40"):
        validate({}, load_pairs()[0], "c35h2hx001abcd")


def test_registration_domain_is_empty_before_launch() -> None:
    assert list((RUN / "h2h-registrations").iterdir()) == []
