"""Prefreeze tests for the Cycle48 Gate A observed-state pipeline."""

from types import SimpleNamespace

import pytest

from experimental.src.scripts import audit_cycle14_mechanics_repair as c14
from experimental.src.scripts import run_cycle48_gateA_observed_states as gate
from experimental.src.scripts.select_cycle48_gateA_observed_states import (
    CANDIDATE_FIELDS, rank, slot_base_positions,
)


# --- selection helpers -------------------------------------------------------

def test_slot_base_positions_are_distinct_chronological_and_spread():
    assert slot_base_positions(8) == [0, 1, 2, 3, 4, 5, 6, 7]
    for count in (8, 9, 10, 17, 40, 113):
        bases = slot_base_positions(count)
        assert len(bases) == 8 == len(set(bases))
        assert bases == sorted(bases)
        assert bases[0] == 0 and bases[-1] == count - 1
    with pytest.raises(ValueError):
        slot_base_positions(7)


def test_selection_rank_is_deterministic_and_label_blind():
    assert rank("d", "cluster") == rank("d", "cluster")
    assert rank("d", "a") != rank("d", "b")
    assert rank("d1", "a") != rank("d2", "a")


def test_selection_candidate_fields_exclude_observed_behavior():
    assert not any(field.startswith("observed") for field in CANDIDATE_FIELDS)
    assert "chosen_action" not in CANDIDATE_FIELDS


# --- frozen slot rule --------------------------------------------------------

def _states(count, bad=()):
    return [
        {"actionable": True, "chosen_action": "tackle",
         "pp_disable_sidecar": {"revival_prompt": index in bad}}
        for index in range(count)
    ]


def test_resolve_slots_takes_bases_when_all_ordinary():
    states = _states(8)
    assert gate.resolve_slots(list(range(8)), states, list(range(8))) == list(range(8))


def test_resolve_slots_skips_non_ordinary_forward_then_backward():
    states = _states(10, bad={4})
    resolved = gate.resolve_slots(list(range(10)), states, [0, 1, 2, 4, 5, 6, 8, 9])
    assert resolved == [0, 1, 2, 5, 6, 7, 8, 9]
    states = _states(9, bad={7, 8})
    resolved = gate.resolve_slots(list(range(9)), states, [0, 1, 2, 3, 4, 5, 7, 8])
    assert resolved == [0, 1, 2, 3, 4, 5, 6, None]


def test_resolve_slots_never_reuses_a_state():
    states = _states(8, bad={6, 7})
    resolved = gate.resolve_slots(list(range(8)), states, [0, 1, 2, 3, 4, 5, 6, 7])
    filled = [value for value in resolved if value is not None]
    assert len(filled) == len(set(filled)) == 6
    assert resolved[6] is None and resolved[7] is None


def test_ordinary_state_matches_admitted_cycle13_semantics_plus_command():
    assert gate.ordinary_state(
        {"actionable": True, "chosen_action": "tackle", "pp_disable_sidecar": {}})
    assert gate.ordinary_state({"actionable": True, "chosen_action": "tackle"})
    assert not gate.ordinary_state({"actionable": False, "chosen_action": "tackle"})
    assert not gate.ordinary_state(
        {"actionable": True, "chosen_action": "tackle",
         "pp_disable_sidecar": {"revival_prompt": True}})
    # A dangling request the human never answered has no behavior anchor.
    assert not gate.ordinary_state({"actionable": True})
    assert not gate.ordinary_state({"actionable": True, "chosen_action": None})
    assert not gate.ordinary_state({"actionable": True, "chosen_action": ""})


# --- aggregation and metrics -------------------------------------------------

def test_aggregate_policy_uses_raw_posterior_weights():
    rows = [
        {"weight": 3.0, "result": {"total_visits": 10, "side_one": [
            {"action": "a", "N": 10}, {"action": "b", "N": 0}]}},
        {"weight": 1.0, "result": {"total_visits": 10, "side_one": [
            {"action": "a", "N": 0}, {"action": "b", "N": 10}]}},
    ]
    out = gate.aggregate_policy(rows, ["a", "b"])
    assert out["policy"] == {"a": 0.75, "b": 0.25}
    assert out["top1"] == "a"


def test_aggregate_policy_rejects_zero_mass_and_breaks_ties_lexically():
    with pytest.raises(gate.Cycle48Error):
        gate.aggregate_policy([{"weight": 0.0, "result": {
            "total_visits": 1, "side_one": [{"action": "a", "N": 1}]}}], ["a"])
    rows = [{"weight": 1.0, "result": {"total_visits": 2, "side_one": [
        {"action": "b", "N": 1}, {"action": "a", "N": 1}]}}]
    assert gate.aggregate_policy(rows, ["a", "b"])["top1"] == "a"


def test_effective_sample_size():
    assert gate.effective_sample_size([1.0] * 8) == pytest.approx(8.0)
    assert gate.effective_sample_size([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    with pytest.raises(gate.Cycle48Error):
        gate.effective_sample_size([0.0, 0.0])


def test_jsd_identity_and_symmetry():
    a = {"x": 0.75, "y": 0.25}
    b = {"x": 0.25, "y": 0.75}
    assert gate.jsd(a, a) == 0
    assert gate.jsd(a, b) == pytest.approx(gate.jsd(b, a))
    assert gate.jsd(a, b) > 0


def test_top1_of_breaks_ties_lexically():
    assert gate.top1_of({"b": 0.5, "a": 0.5}) == "a"
    assert gate.top1_of({"b": 0.6, "a": 0.4}) == "b"


# --- request authority (Cycle46 lesson) -------------------------------------

def test_install_request_authority_sets_current_actions_before_engine_use():
    previous = list(c14.CURRENT_ACTIONS)
    try:
        ordered = gate.install_request_authority({"tackle", "switch pikachu"})
        assert ordered == ["switch pikachu", "tackle"]
        assert c14.CURRENT_ACTIONS == ordered
        with pytest.raises(gate.Cycle48Error):
            gate.install_request_authority(set())
    finally:
        c14.CURRENT_ACTIONS[:] = previous


# --- teacher search validation ----------------------------------------------

def _engine_result(total, side_one):
    rows = [SimpleNamespace(move_choice=action, visits=n, total_score=w)
            for action, n, w in side_one]
    return SimpleNamespace(total_visits=total, side_one=rows, side_two=[])


def _engine(result):
    return SimpleNamespace(
        monte_carlo_tree_search_with_s1_request=lambda *args, **kwargs: result)


def test_run_teacher_search_accepts_exact_support_and_preserves_null_q():
    result = _engine_result(4, [("a", 3, 1.5), ("b", 1, 0.0), ("c", 0, 0.0)])
    out = gate.run_teacher_search(_engine(result), None, ["a", "b", "c"], 4, 7)
    by = {row["action"]: row for row in out["result"]["side_one"]}
    assert by["a"]["Q"] == pytest.approx(0.5)
    assert by["c"]["Q"] is None
    assert out["result"]["total_visits"] == 4


def test_run_teacher_search_rejects_visit_or_support_mismatch():
    with pytest.raises(gate.Cycle48Error):
        gate.run_teacher_search(
            _engine(_engine_result(3, [("a", 3, 1.0)])), None, ["a"], 4, 7)
    with pytest.raises(gate.Cycle48Error):
        gate.run_teacher_search(
            _engine(_engine_result(4, [("a", 4, 1.0)])), None, ["a", "b"], 4, 7)
    with pytest.raises(gate.Cycle48Error):
        gate.run_teacher_search(
            _engine(_engine_result(4, [("a", 4, float("nan"))])), None, ["a"], 4, 7)


# --- failure accounting ------------------------------------------------------

def test_slot_failures_cover_exactly_sixteen_unique_rows():
    rows = gate.slot_failures(3, 5, "resolve", gate.SlotUnfilledError("x"))
    assert len(rows) == 16
    assert len({row["row"] for row in rows}) == 16
    assert all(row["failure_class"] == "SlotUnfilledError" for row in rows)


def test_carried_failure_preserves_original_class():
    row = gate.failure_row(0, 0, 0, 0, "r1_control",
                           gate.CarriedFailure("HumanAnchorError", "carried"))
    assert row["failure_class"] == "HumanAnchorError"


# --- verify_candidates -------------------------------------------------------

def _candidate(request_index=0, **overrides):
    base = {
        "role": "p1", "request_index": request_index, "command_input_index": 1,
        "public_event_index": 2, "private_request_sha256": "aa",
        "causal_prefix_sha256": "bb", "legal_action_contract_sha256": "cc",
        "pp_disable_sidecar_sha256": "dd",
        "model_information_fingerprint_sha256": "ee",
        "typed_reveal_ledger_sha256": "ff",
    }
    base.update(overrides)
    return base


def test_verify_candidates_detects_parity_drift():
    selected = {"candidates": [_candidate()]}
    compacts = {0: _candidate()}
    first = gate.verify_candidates(selected, compacts)
    assert first == gate.verify_candidates(selected, compacts)
    with pytest.raises(Exception):
        gate.verify_candidates(selected, {0: _candidate(private_request_sha256="zz")})
    with pytest.raises(Exception):
        gate.verify_candidates(selected, {})


# --- gate evaluation ---------------------------------------------------------

def _synthetic_results(clusters=64):
    selection, results = [], []
    for cluster in range(clusters):
        selection.append({
            "split": "train",
            "dependency_cluster_id": f"execution:synthetic-{cluster:03d}",
        })
        raw_rows, cells = [], []
        for slot in range(8):
            fingerprint = f"fp-{cluster:03d}-{slot}"
            for schedule in range(2):
                for world in range(8):
                    raw_rows.append({
                        "row": gate.row_id(cluster, slot, schedule, world),
                        "arms": {"equal8192_a": {"latency_ms": 1.0},
                                 "equal8192_b": {"latency_ms": 1.0},
                                 "equal20000": {"latency_ms": 2.0}},
                    })
                policy = {"a": 0.9, "b": 0.1}
                cells.append({
                    "cluster_index": cluster,
                    "dependency_cluster_id": selection[-1]["dependency_cluster_id"],
                    "slot": slot, "schedule_index": schedule,
                    "information_fingerprint": fingerprint,
                    "effective_sample_size": 7.5,
                    "aggregates": {
                        "equal8192_a": {"policy": policy, "top1": "a"},
                        "equal8192_b": {"policy": policy, "top1": "a"},
                        "equal20000": {"policy": policy, "top1": "a"},
                    },
                    "repeat_jsd": 0.001,
                    "agreement_8192_20000": True,
                    "human_action": "a", "human_top1_match_20000": True,
                    "r1_top1": "a", "r1_top1_match_20000": True,
                })
        results.append({"cluster_index": cluster, "scheduled": 128,
                        "raw_rows": raw_rows, "cells": cells, "failures": []})
    return selection, results


def test_evaluate_passes_on_full_synthetic_support():
    selection, results = _synthetic_results()
    report = gate.evaluate(selection, results, dev=True)
    assert report["status"] == "pass"
    assert report["counts"]["scheduled"] == 8192
    assert report["counts"]["unique_fingerprints"] == 512
    assert report["metrics"]["coverage"] == 1.0
    assert report["gates"]["zero_hidden_sensitivity"]
    assert report["authorization"]["gateB_tiny_cpu_training"] is False  # dev smoke
    assert report["authorization"]["sealed93"] is False


def test_evaluate_fails_when_one_state_loses_all_cells():
    selection, results = _synthetic_results()
    lost = results[0]
    lost["cells"] = [cell for cell in lost["cells"] if cell["slot"] != 0]
    lost["raw_rows"] = lost["raw_rows"][16:]
    lost["failures"] = gate.slot_failures(0, 0, "resolve", gate.SlotUnfilledError("x"))
    report = gate.evaluate(selection, results, dev=True)
    assert report["counts"]["unique_fingerprints"] == 511
    assert not report["gates"]["unique_fingerprints_ge512"]
    assert report["gates"]["coverage_ge95"]  # 8176/8192 still above 95%
    assert report["status"] == "fail"


def test_evaluate_fails_closed_on_hidden_sensitivity():
    selection, results = _synthetic_results()
    target = results[1]
    target["raw_rows"] = target["raw_rows"][1:]
    target["failures"] = [gate.failure_row(
        1, 0, 0, 0, "production_schedules",
        gate.Cycle48Error("hidden completions changed observer public projection"),
    )]
    report = gate.evaluate(selection, results, dev=True)
    assert not report["gates"]["zero_hidden_sensitivity"]
    assert report["status"] == "fail"


def test_evaluate_rejects_unbalanced_row_accounting():
    selection, results = _synthetic_results()
    results[2]["raw_rows"] = results[2]["raw_rows"][:-1]
    with pytest.raises(gate.Cycle48Error):
        gate.evaluate(selection, results, dev=True)


def test_evaluate_reports_schedule_half_jsd_and_ess():
    selection, results = _synthetic_results()
    report = gate.evaluate(selection, results, dev=True)
    assert report["metrics"]["schedule_half_soft_policy_jsd_median"] == pytest.approx(0.0)
    assert report["metrics"]["effective_sample_size_mean"] == pytest.approx(7.5)
    assert report["metrics"]["human_top1_match_20000"] == pytest.approx(1.0)
    assert report["metrics"]["r1_top1_match_20000"] == pytest.approx(1.0)


def test_evaluate_coverage_threshold_is_frozen_at_95():
    selection, results = _synthetic_results()
    # Fail 6 whole clusters (768 rows): coverage 7424/8192 = 90.6% < 95%.
    for cluster in range(6):
        target = results[cluster]
        target["raw_rows"], target["cells"] = [], []
        target["failures"] = [
            row for slot in range(8)
            for row in gate.slot_failures(cluster, slot, "resolve",
                                          gate.CarriedFailure("Cycle48Error"))
        ]
    report = gate.evaluate(selection, results, dev=True)
    assert not report["gates"]["coverage_ge95"]
    assert report["status"] == "fail"
