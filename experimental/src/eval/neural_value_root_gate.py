"""Paired, equal-wall-budget gate for neural-guided fixed-root search."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "metagross-neural-value-root-result/v2"
ARMS = ("baseline", "candidate")


class NeuralValueRootGateError(ValueError):
    pass


@dataclass(frozen=True)
class RootResult:
    pair_id: str
    arm: str
    budget_ms: int
    elapsed_ms: float
    selected_action: str
    oracle_action: str
    oracle_best_value: float
    selected_oracle_value: float
    oracle_artifact_sha256: str
    value_head_sha256: str | None
    certified_neural_leaves: int
    total_leaf_evaluations: int
    root_id: str | None = None
    battle_id: str | None = None

    @property
    def regret(self) -> float:
        return self.oracle_best_value - self.selected_oracle_value

    @property
    def oracle_top1(self) -> bool:
        return self.selected_action == self.oracle_action


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise NeuralValueRootGateError(f"{field} must be lowercase SHA-256")
    return value


def _result(row: dict[str, Any]) -> RootResult:
    if row.get("schema") != SCHEMA or row.get("arm") not in ARMS:
        raise NeuralValueRootGateError("invalid root-result schema or arm")
    pair_id = row.get("pair_id")
    root_id = row.get("root_id")
    battle_id = row.get("battle_id")
    selected, oracle = row.get("selected_action"), row.get("oracle_action")
    if not all(
        isinstance(value, str) and value
        for value in (pair_id, root_id, battle_id, selected, oracle)
    ):
        raise NeuralValueRootGateError("invalid root-result identity or action")
    budget = row.get("budget_ms")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise NeuralValueRootGateError("invalid wall budget")
    try:
        elapsed = float(row["elapsed_ms"])
        best = float(row["oracle_best_value"])
        chosen = float(row["selected_oracle_value"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise NeuralValueRootGateError("invalid numeric root result") from None
    if not all(math.isfinite(value) for value in (elapsed, best, chosen)) or elapsed < 0 or not 0 <= chosen <= best <= 1:
        raise NeuralValueRootGateError("out-of-range numeric root result")
    certified, leaves = row.get("certified_neural_leaves"), row.get("total_leaf_evaluations")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (certified, leaves)) or certified > leaves:
        raise NeuralValueRootGateError("invalid neural leaf census")
    head = row.get("value_head_sha256")
    if row["arm"] == "candidate":
        head = _sha(head, "value_head_sha256")
    elif head is not None:
        raise NeuralValueRootGateError("baseline cannot load a value head")
    return RootResult(
        pair_id=pair_id,
        arm=row["arm"],
        budget_ms=budget,
        elapsed_ms=elapsed,
        selected_action=selected,
        oracle_action=oracle,
        oracle_best_value=best,
        selected_oracle_value=chosen,
        oracle_artifact_sha256=_sha(row.get("oracle_artifact_sha256"), "oracle_artifact_sha256"),
        value_head_sha256=head,
        certified_neural_leaves=certified,
        total_leaf_evaluations=leaves,
        root_id=root_id,
        battle_id=battle_id,
    )


def load_results(paths: Iterable[Path]) -> dict[str, dict[str, RootResult]]:
    pairs: dict[str, dict[str, RootResult]] = {}
    for path in map(Path, paths):
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    result = _result(json.loads(line))
                except (json.JSONDecodeError, NeuralValueRootGateError) as exc:
                    raise NeuralValueRootGateError(f"{path}:{line_number}: {exc}") from exc
                arms = pairs.setdefault(result.pair_id, {})
                if result.arm in arms:
                    raise NeuralValueRootGateError(f"duplicate {result.pair_id}/{result.arm}")
                arms[result.arm] = result
    if not pairs or any(set(arms) != set(ARMS) for arms in pairs.values()):
        raise NeuralValueRootGateError("results are not complete paired arms")
    oracle_hashes = {result.oracle_artifact_sha256 for arms in pairs.values() for result in arms.values()}
    head_hashes = {arms["candidate"].value_head_sha256 for arms in pairs.values()}
    if len(oracle_hashes) != 1 or len(head_hashes) != 1:
        raise NeuralValueRootGateError("gate mixes oracle or value-head artifacts")
    for pair_id, arms in pairs.items():
        baseline, candidate = arms["baseline"], arms["candidate"]
        if baseline.budget_ms != candidate.budget_ms or (
            baseline.oracle_action,
            baseline.oracle_best_value,
        ) != (candidate.oracle_action, candidate.oracle_best_value):
            raise NeuralValueRootGateError(f"pair {pair_id} is not common-budget/common-oracle")
        if baseline.root_id != candidate.root_id:
            raise NeuralValueRootGateError(f"pair {pair_id} mixes root identities")
        if baseline.battle_id != candidate.battle_id:
            raise NeuralValueRootGateError(f"pair {pair_id} mixes source battles")
    return pairs


def _bootstrap_lower(values: list[float], seed: int, samples: int = 20_000) -> float:
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(samples))
    return means[int(0.025 * samples)]


def evaluate_gate(
    pairs: dict[str, dict[str, RootResult]],
    *,
    min_pairs: int = 100,
    min_roots: int = 50,
    min_battles: int = 50,
    max_budget_overrun_fraction: float = 0.05,
    min_certified_leaf_fraction: float = 0.05,
    top1_noninferiority_margin: float = 0.02,
    bootstrap_seed: int = 20260813,
) -> dict[str, Any]:
    if len(pairs) < min_pairs:
        raise NeuralValueRootGateError(f"gate requires at least {min_pairs} paired roots")
    ordered = [pairs[key] for key in sorted(pairs)]
    baseline = [arms["baseline"] for arms in ordered]
    candidate = [arms["candidate"] for arms in ordered]
    budget_ok = all(
        result.elapsed_ms <= result.budget_ms * (1 + max_budget_overrun_fraction)
        for result in baseline + candidate
    )
    certified = sum(result.certified_neural_leaves for result in candidate)
    leaves = sum(result.total_leaf_evaluations for result in candidate)
    coverage = certified / leaves if leaves else 0.0
    improvements = [left.regret - right.regret for left, right in zip(baseline, candidate, strict=True)]
    mean_improvement = sum(improvements) / len(improvements)
    by_root: dict[str, list[float]] = {}
    root_battle: dict[str, str] = {}
    for left, value in zip(baseline, improvements, strict=True):
        root_id = left.root_id or left.pair_id
        battle_id = left.battle_id or root_id
        previous = root_battle.setdefault(root_id, battle_id)
        if previous != battle_id:
            raise NeuralValueRootGateError(f"root {root_id} appears in multiple source battles")
        by_root.setdefault(root_id, []).append(value)
    if len(by_root) < min_roots:
        raise NeuralValueRootGateError(f"gate requires at least {min_roots} independent roots")
    by_battle: dict[str, list[float]] = {}
    for root_id, values in by_root.items():
        by_battle.setdefault(root_battle[root_id], []).append(sum(values) / len(values))
    if len(by_battle) < min_battles:
        raise NeuralValueRootGateError(
            f"gate requires at least {min_battles} independent source battles"
        )
    battle_improvements = [sum(values) / len(values) for values in by_battle.values()]
    lower = _bootstrap_lower(battle_improvements, bootstrap_seed)
    baseline_top1 = sum(result.oracle_top1 for result in baseline) / len(baseline)
    candidate_top1 = sum(result.oracle_top1 for result in candidate) / len(candidate)
    checks = {
        "equal_500ms_budget": budget_ok and {result.budget_ms for result in baseline + candidate} == {500},
        "certified_neural_coverage": coverage >= min_certified_leaf_fraction,
        "paired_regret_improvement_ci": lower > 0.0,
        "oracle_top1_noninferiority": candidate_top1 >= baseline_top1 - top1_noninferiority_margin,
    }
    report = {
        "schema": "metagross-neural-value-root-gate/v2",
        "pairs": len(pairs),
        "independent_roots": len(by_root),
        "independent_battles": len(by_battle),
        "oracle_artifact_sha256": baseline[0].oracle_artifact_sha256,
        "value_head_sha256": candidate[0].value_head_sha256,
        "metrics": {
            "baseline_mean_regret": sum(value.regret for value in baseline) / len(baseline),
            "candidate_mean_regret": sum(value.regret for value in candidate) / len(candidate),
            "paired_mean_regret_improvement": mean_improvement,
            "paired_bootstrap_95_lower": lower,
            "bootstrap_unit": "independent_source_battle",
            "baseline_oracle_top1": baseline_top1,
            "candidate_oracle_top1": candidate_top1,
            "certified_neural_leaf_fraction": coverage,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="JSONL result files containing paired arms")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--min-roots", type=int, default=50)
    parser.add_argument("--min-battles", type=int, default=50)
    args = parser.parse_args()
    report = evaluate_gate(
        load_results(args.results),
        min_pairs=args.min_pairs,
        min_roots=args.min_roots,
        min_battles=args.min_battles,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
