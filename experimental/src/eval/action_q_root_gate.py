"""Paired independent-battle gate for action-Q-guided root search."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from eval.neural_value_root_gate import (
    NeuralValueRootGateError,
    RootResult,
    _result as _neural_result,
    evaluate_gate as _evaluate_neural_gate,
)


SCHEMA = "metagross-action-q-root-result/v1"


class ActionQRootGateError(NeuralValueRootGateError):
    pass


def _result(row: dict[str, Any]) -> RootResult:
    if row.get("schema") != SCHEMA:
        raise ActionQRootGateError("invalid action-Q root-result schema")
    queries = row.get("guidance_queries")
    legal = row.get("legal_action_queries")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (queries, legal)) or queries > legal:
        raise ActionQRootGateError("invalid action-Q guidance census")
    if row.get("arm") == "baseline" and queries != 0:
        raise ActionQRootGateError("baseline cannot query the action-Q model")
    translated = dict(row)
    translated.update({
        "schema": "metagross-neural-value-root-result/v2",
        "value_head_sha256": row.get("action_q_model_sha256"),
        "certified_neural_leaves": queries,
        "total_leaf_evaluations": legal,
    })
    try:
        return _neural_result(translated)
    except NeuralValueRootGateError as exc:
        raise ActionQRootGateError(str(exc)) from exc


def load_results(paths: Iterable[Path]) -> dict[str, dict[str, RootResult]]:
    pairs: dict[str, dict[str, RootResult]] = {}
    for path in map(Path, paths):
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    result = _result(json.loads(line))
                except (json.JSONDecodeError, ActionQRootGateError) as exc:
                    raise ActionQRootGateError(f"{path}:{line_number}: {exc}") from exc
                arms = pairs.setdefault(result.pair_id, {})
                if result.arm in arms:
                    raise ActionQRootGateError(f"duplicate {result.pair_id}/{result.arm}")
                arms[result.arm] = result
    if not pairs or any(set(arms) != {"baseline", "candidate"} for arms in pairs.values()):
        raise ActionQRootGateError("results are not complete paired arms")
    oracle_hashes = {row.oracle_artifact_sha256 for arms in pairs.values() for row in arms.values()}
    model_hashes = {arms["candidate"].value_head_sha256 for arms in pairs.values()}
    if len(oracle_hashes) != 1 or len(model_hashes) != 1:
        raise ActionQRootGateError("gate mixes oracle or action-Q artifacts")
    for pair_id, arms in pairs.items():
        baseline, candidate = arms["baseline"], arms["candidate"]
        if (
            baseline.budget_ms != candidate.budget_ms
            or baseline.root_id != candidate.root_id
            or baseline.battle_id != candidate.battle_id
            or (baseline.oracle_action, baseline.oracle_best_value)
            != (candidate.oracle_action, candidate.oracle_best_value)
        ):
            raise ActionQRootGateError(f"pair {pair_id} is not common-budget/common-oracle")
    return pairs


def evaluate_gate(pairs: dict[str, dict[str, RootResult]], **kwargs: Any) -> dict[str, Any]:
    base = _evaluate_neural_gate(
        pairs,
        min_certified_leaf_fraction=1.0,
        **kwargs,
    )
    checks = dict(base["checks"])
    checks["action_q_prior_coverage"] = checks.pop("certified_neural_coverage")
    metrics = dict(base["metrics"])
    metrics["action_q_prior_coverage"] = metrics.pop("certified_neural_leaf_fraction")
    report = {
        **base,
        "schema": "metagross-action-q-root-gate/v1",
        "action_q_model_sha256": base.pop("value_head_sha256"),
        "checks": checks,
        "metrics": metrics,
    }
    report.pop("report_sha256", None)
    report["passed"] = all(checks.values())
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_gate(load_results(args.results))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
