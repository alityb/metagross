import json
from pathlib import Path

import pytest

from experimental.src.scripts import cycle11_replay_audit as v11
from experimental.src.scripts.run_cycle11_full_corpus_index import deterministic_match, digest
from experimental.src.scripts.select_cycle11_corpus import format_id


COMMIT = "f8ac14003a5f27e1bdc8d8c59608a773c1cb96e5"


def test_sethp_normalizes_only_pokemon_hp_pairs():
    rows = v11.canonical_public_lines(
        ["|-sethp|p1a: A|170/317|p2a: B|1/3|[from] move: Pain Split|[silent]"],
        inputlog="", showdown_commit=COMMIT,
    )
    assert rows == ["|-sethp|p1a: A|54/100|p2a: B|34/100|[from] move: Pain Split|[silent]"]


def test_sethp_preserves_public_percent_and_unrelated_fields():
    line = "|-sethp|p1a: A|54/100|[from] 170/317|[silent]"
    assert v11.canonical_public_lines([line], inputlog="", showdown_commit=COMMIT) == [line]


def test_name_change_forfeit_requires_exact_loser_and_supported_version():
    lines = ["|player|p1|Alice", "|player|p2|Bob",
             "|-message|Bob forfeited by changing their name.", "|win|Alice"]
    assert "|-message|Bob forfeited by changing their name." not in v11.canonical_public_lines(
        lines, inputlog=">forcelose p2", showdown_commit=COMMIT,
    )
    assert "|-message|Alice forfeited by changing their name." in v11.canonical_public_lines(
        ["|player|p1|Alice", "|player|p2|Bob",
         "|-message|Alice forfeited by changing their name.", "|win|Alice"],
        inputlog=">forcelose p2", showdown_commit=COMMIT,
    )


def test_unknown_commit_fails_closed():
    with pytest.raises(v11.ReplayAuditError):
        v11.canonical_public_lines([], inputlog="", showdown_commit="0" * 40)


def test_utf8_digest_accepts_non_ascii():
    assert digest("Poké🙂") == digest("Poké🙂".encode("utf-8"))


def test_determinism_accepts_exact_admission_or_exact_fail_closed_classification():
    admitted = {
        "status": "pass", "compact_sha256": "a",
        "canonical_public_sha256": "b", "execution_sha256": "c",
    }
    abstained = {
        "status": "fail", "failure_class": "causal_ledger_fail_closed",
        "failure_detail_sha256": "d", "relative_index": None,
    }
    assert deterministic_match(admitted, dict(admitted))
    assert deterministic_match(abstained, dict(abstained))
    assert not deterministic_match(abstained, {**abstained, "failure_detail_sha256": "x"})
    assert not deterministic_match(admitted, abstained)


def test_format_id_reads_only_start_format(tmp_path: Path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps({
        "inputlog": '>start {"formatid":"gen9randombattle"}\n>win',
        "log": "|win|name",
    }))
    assert format_id(str(path)) == "gen9randombattle"
