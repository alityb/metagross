import json

import pytest

from experimental.src.scripts.monitor_cycle25_attributed_smoke import validate_attributed_receipts
from srcs.metagross.causal_reveal_ledger import hydrate_certified_abilities
from srcs.metagross.tests.test_cycle22_certified_ability_install import cycle21_ledger, cycle21_state


IDENTITY = {
    "battle_tag": "battle-cycle25",
    "rqid": 2,
    "decision_index": 0,
    "root_id": "a" * 64,
}


def context(phase: str, index: int, schedule=None, world=None, declared=16) -> dict:
    return {
        **IDENTITY,
        "phase": phase,
        "cohort": "adaptive_root_search" if phase == "production_control" else "fixed_two_by_eight",
        "declared_world_count": declared,
        "conversion_index": index,
        "schedule_index": schedule,
        "world_index": world,
    }


def test_execution_marker_changes_no_state_or_ledger_bytes(tmp_path, monkeypatch) -> None:
    ledger = cycle21_ledger()
    identity = {**context("equal8192_candidate", 0, 0, 0), "battle_tag": ledger.battle_tag}
    state = cycle21_state()
    unmarked = hydrate_certified_abilities(state, ledger, swap=False)
    monkeypatch.setenv("METAGROSS_CAUSAL_ABILITY_RECEIPT_DIR", str(tmp_path))
    monkeypatch.setenv("METAGROSS_PRIOR_NAMESPACE", "agent_a")
    marked = hydrate_certified_abilities(state, ledger, swap=False, receipt_context=identity)
    assert marked.to_string() == unmarked.to_string()
    assert ledger.canonical_bytes() == cycle21_ledger().canonical_bytes()
    row = json.loads(next(tmp_path.glob("agenta-*.jsonl")).read_text())
    assert row["schema"] == "metagross-certified-ability-installation/v2"
    assert row["execution_context"] == identity
    assert "execution_context" not in marked.to_string()
    assert "execution_context" not in ledger.canonical_bytes().decode()


def receipt(ctx: dict, stamp: int = 10) -> dict:
    return {
        "schema": "metagross-certified-ability-installation/v2",
        "battle_tag": IDENTITY["battle_tag"], "observer_role": "p1",
        "protocol_sha256": "b" * 64, "swap": False,
        "receipt_time_ns": stamp, "execution_context": ctx,
        "installations": [{
            "authority": "rule_implied_form_transition",
            "exact_public_species": "terapagosterastal", "slot": 0,
            "installed_base_ability": "terashell",
            "installed_current_ability": "terashell", "update_base": True,
        }],
    }


def write_complete(path, production_count=32, late=False) -> None:
    rows = [receipt(context("production_control", i, declared=production_count)) for i in range(production_count)]
    rows += [receipt(context("equal8192_candidate", s * 8 + w, s, w)) for s in range(2) for w in range(8)]
    if late:
        rows[-1]["receipt_time_ns"] = 101
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_classifier_reconciles_named_adaptive_and_two_by_eight_cohorts(tmp_path) -> None:
    directory = tmp_path / "ability-receipts"
    directory.mkdir()
    write_complete(directory / "agenta-1.jsonl")
    result = validate_attributed_receipts(tmp_path, IDENTITY, "b" * 64, {"public_execution_time_ns": 100})
    assert result["production_receipts"] == 32
    assert result["candidate_receipts"] == 16


def test_classifier_rejects_any_post_execution_receipt(tmp_path) -> None:
    directory = tmp_path / "ability-receipts"
    directory.mkdir()
    write_complete(directory / "agenta-1.jsonl", late=True)
    with pytest.raises(RuntimeError, match="after public execution"):
        validate_attributed_receipts(tmp_path, IDENTITY, "b" * 64, {"public_execution_time_ns": 100})
