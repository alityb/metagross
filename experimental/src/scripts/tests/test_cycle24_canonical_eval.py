import hashlib
import json

from experimental.src.scripts.cycle24_canonical_eval import config_identity, derived_argv
from experimental.src.scripts.watch_cycle24_registrations import validate_registration


def test_prepare_and_live_projection_have_identical_evaluator_identity() -> None:
    canonical = __import__("pathlib").Path(
        "experimental/runs/search_native_v2_cycle24_canonical_prepare_20260815/CANONICAL_EVAL_ARGV.json"
    ).resolve()
    pair_sha = "a" * 64
    assert config_identity(canonical, "prepare") == config_identity(canonical, "live", pair_sha)
    _, prepare = derived_argv(canonical, "prepare")
    _, live = derived_argv(canonical, "live", pair_sha)
    assert prepare[-1] == "--prepare-mirrored-pairs-only"
    assert live[-2:] == ["--pair-manifest-sha256", pair_sha]


def test_cycle24_registration_binds_fresh_identity() -> None:
    team1, team2 = "Pikachu||||tackle", "Terapagos||||terastarstorm"
    pair = {
        "pair_id": "pair24", "battle_seed": "1,2,3,4",
        "team_1_packed": team1, "team_2_packed": team2,
        "team_1_sha256": hashlib.sha256(team1.encode()).hexdigest(),
        "team_2_sha256": hashlib.sha256(team2.encode()).hexdigest(),
    }
    payload = {
        "schema_version": 1, "pair_id": "pair24", "leg": 1,
        "format": "gen9randombattle", "battle_seed": "1,2,3,4",
        "team_1_sha256": pair["team_1_sha256"],
        "team_2_sha256": pair["team_2_sha256"],
        "assigned_team_sha256": pair["team_2_sha256"], "packed_team": team2,
    }
    assert validate_registration(payload, pair, "c24smky0000001") == "p2"
