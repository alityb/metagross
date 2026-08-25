from pathlib import Path

from experimental.src.scripts.monitor_cycle19_operational_smoke import rows
from experimental.src.scripts.monitor_cycle20_form_smoke import validate_form_transition


ROOT = Path(__file__).resolve().parents[4]
PRESERVED = ROOT / (
    "experimental/runs/search_native_v2_cycle19_operational_repair_20260815/"
    "h2h-logs/c19h2hx002cb97.protocol.jsonl"
)


def test_preserved_cycle19_protocol_satisfies_exact_live_lineage_contract() -> None:
    protocol = rows(PRESERVED)
    result = validate_form_transition(protocol, "c19h2hx002cb97", 2**63 - 1)
    assert result["observer_role"] == "p1"
    assert result["exact_public_species"] == "terapagosterastal"
    assert result["current_ability"] == "terashell"
    assert result["ability_history_tail"] == [
        ("terashift", "explicit_public_event"),
        ("terashell", "rule_implied_form_transition"),
    ]
    assert result["shift_event_index"] < result["detailschange_event_index"]
