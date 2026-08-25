#!/usr/bin/env python3
"""Correct Gate A coverage without conditioning selection on extractor support."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.run_public_search_state_gate_a import (
    EXPECTED,
    GateAError,
    _ordinary_actions,
    hidden_perturbation,
    load_rows,
    rank,
    sha256,
)
from search.public_search_policy_smoke import infer, make_policy
from search.public_search_state_v1 import (
    canonical_action_table,
    canonical_bytes,
    compile_side_one_reveal_mask,
    extract_public_search_state,
    install_side_one_reveal_mask,
)


ROOT = Path(__file__).resolve().parents[3]


def select_without_representation(
    rows: Sequence[Mapping[str, Any]], engine: Any
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    rejected: Counter[str] = Counter()
    for raw in rows:
        row = dict(raw)
        try:
            snapshot = row["r1_policy_snapshot"]
            if row.get("schema") != "metagross-causal-dual-r1-root/v1":
                raise GateAError("schema")
            if snapshot.get("schema") != 6:
                raise GateAError("not_schema6")
            legality = snapshot.get("own_legality")
            if not isinstance(legality, Mapping) or legality.get("force_switch"):
                raise GateAError("not_ordinary")
            if len(legality.get("actions", ())) < 2:
                raise GateAError("not_ordinary")
            state = engine.State.from_string(row["state"])
            if hashlib.sha256(row["state"].encode()).hexdigest() != row.get("state_sha256"):
                raise GateAError("state_hash")
            first, second = engine.root_options(state)
            first, second = _ordinary_actions(first), _ordinary_actions(second)
            if len(first) < 2 or not second:
                raise GateAError("not_joint_ordinary")
            if set(first) != set(legality["actions"]):
                raise GateAError("root_legality")
            table = canonical_action_table(first)
            if table["name_table"] != snapshot["name_table"]:
                raise GateAError("root_action_map")
            if table["illegal_actions"] != snapshot["illegal_actions"]:
                raise GateAError("root_illegal_map")
            row["_state"] = state
            row["_side_one_actions"] = first
            row["_side_two_actions"] = second
            candidates.append(
                (rank([row["capture_sha256"], row["state_sha256"]]), row)
            )
        except GateAError as exc:
            rejected[str(exc)] += 1
        except Exception:
            rejected["invalid_pre_representation_input"] += 1
    candidates.sort(key=lambda item: item[0])
    selected, per_battle = [], defaultdict(int)
    for _, row in candidates:
        battle = str(row["identity"]["battle_tag"])
        if per_battle[battle] >= 5:
            continue
        selected.append(row)
        per_battle[battle] += 1
        if len(selected) == 200:
            break
    if len(selected) != 200:
        raise GateAError(f"corrected selector produced {len(selected)} roots")
    return selected, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = run_dir / "corrected-gate-a-coverage-report.json"
    if output.exists():
        raise GateAError("corrected report already exists")
    frozen = json.loads((run_dir / "CORRECTED_COVERAGE_INPUTS.json").read_text())
    for relative, expected in frozen["files"].items():
        if sha256(ROOT / relative) != expected:
            raise GateAError(f"corrected frozen hash mismatch: {relative}")

    import poke_engine

    paths = [
        ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-a-decisions.jsonl.dual-r1-roots.jsonl",
        ROOT / "experimental/runs/schema6_local_5000_20260814_r1/peer/agent-b-decisions.jsonl.dual-r1-roots.jsonl",
    ]
    if sha256(paths[0]) != EXPECTED["agent_a"] or sha256(paths[1]) != EXPECTED["agent_b"]:
        raise GateAError("source hash mismatch")
    if sha256(Path(poke_engine.poke_engine.__file__)) != EXPECTED["engine"]:
        raise GateAError("engine hash mismatch")
    rows = load_rows(paths[0]) + load_rows(paths[1])
    selected, rejections = select_without_representation(rows, poke_engine)
    policy = make_policy()

    attempted = supported = 0
    failures: Counter[str] = Counter()
    for root in selected:
        pairs = [
            (left, right)
            for left in root["_side_one_actions"]
            for right in root["_side_two_actions"]
        ]
        pairs.sort(key=lambda pair: rank([root["capture_sha256"], *pair]))
        schedule = [(left, right, u) for left, right in pairs[:4] for u in (0.25, 0.75)]
        attempted += len(schedule)
        try:
            bits = compile_side_one_reveal_mask(
                root["_state"],
                root["r1_policy_snapshot"]["player_information_state"],
            )
            state = install_side_one_reveal_mask(root["_state"], bits)
        except Exception as exc:
            failures[f"root_mask:{type(exc).__name__}:{exc}"] += len(schedule)
            continue
        for left, right, uniform in schedule:
            try:
                root_string = state.to_string()
                root_public = canonical_bytes(extract_public_search_state(state, poke_engine))
                step = poke_engine.step_with_uniform_r1_semantic(
                    state, left, right, uniform
                )
                child = step.state
                restored = child.reverse_instructions(step.selected_instructions)
                if restored.to_string() != root_string:
                    raise GateAError("engine_reverse_mismatch")
                if canonical_bytes(extract_public_search_state(restored, poke_engine)) != root_public:
                    raise GateAError("public_reverse_mismatch")
                public = extract_public_search_state(child, poke_engine)
                first_bytes = canonical_bytes(public)
                perturbation = hidden_perturbation(child, poke_engine)
                if perturbation is not None and canonical_bytes(
                    extract_public_search_state(perturbation, poke_engine)
                ) != first_bytes:
                    raise GateAError("hidden_noninterference")
                if poke_engine.terminal_value(child) == 0.0 and public["action_table"]["automatic_action"] is None:
                    probabilities = infer(policy, [public])[0]
                    if not np.array_equal(probabilities, infer(policy, [public])[0]):
                        raise GateAError("policy_nondeterminism")
                supported += 1
            except Exception as exc:
                failures[f"successor:{type(exc).__name__}:{exc}"] += 1

    coverage = supported / attempted
    report = {
        "schema": "metagross-search-native-v2-gate-a-corrected-coverage/v1",
        "status": "pass" if coverage >= 0.95 else "fail",
        "gate": "supported_successors_divided_by_all_pre_representation_scheduled_successors_ge_0.95",
        "source_rows": len(rows),
        "selected_roots": len(selected),
        "physical_battles": len({row["identity"]["battle_tag"] for row in selected}),
        "scheduled_successors": attempted,
        "supported_successors": supported,
        "unsupported_successors": attempted - supported,
        "coverage": coverage,
        "failures": dict(failures.most_common()),
        "pre_representation_rejections": dict(sorted(rejections.items())),
        "initial_pass_quarantined": True,
        "root_r1_used_at_interior": False,
        "fabricated_history": False,
        "sealed_confirmation_panel_rows_read": 0,
        "new_games": 0,
        "h2h_games": 0,
        "local_cpu_only": True,
        "paid_compute_usd": 0,
        "hashes": {
            "corrected_protocol_sha256": frozen["protocol_sha256"],
            "corrected_inputs_sha256": sha256(run_dir / "CORRECTED_COVERAGE_INPUTS.json"),
            "agent_a_source_sha256": EXPECTED["agent_a"],
            "agent_b_source_sha256": EXPECTED["agent_b"],
            "engine_binding_sha256": EXPECTED["engine"],
        },
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
