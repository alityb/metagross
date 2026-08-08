from __future__ import annotations

import json
import math
import random
import unittest

from srcs.metagross.holdout_metrics import compute_holdout_metrics


def _hash(number: int) -> str:
    return f"{number:064x}"


def _row(
    delta: float,
    *,
    pairs: int = 100,
    candidate_catastrophes: int = 0,
    baseline_catastrophes: int = 0,
    candidate_severity: float = 0.0,
    baseline_severity: float = 0.0,
    within_variance: float = 0.0,
) -> dict[str, object]:
    baseline_mean = 0.5 - delta / 2
    candidate_mean = 0.5 + delta / 2
    better = pairs if delta > 0 else 0
    worse = pairs if delta < 0 else 0
    return {
        "pairs": pairs,
        "baseline_sum": baseline_mean * pairs,
        "candidate_sum": candidate_mean * pairs,
        "delta_sum": delta * pairs,
        "delta_squared_sum": (delta * delta + within_variance) * pairs,
        "catastrophic_count": candidate_catastrophes,
        "baseline_catastrophic_count": baseline_catastrophes,
        "candidate_catastrophic_severity_sum": candidate_severity,
        "baseline_catastrophic_severity_sum": baseline_severity,
        "candidate_better_count": better,
        "baseline_better_count": worse,
        "equal_count": pairs - better - worse,
    }


def _v5_row(
    delta: float,
    *,
    pairs: int = 100,
    baseline_terminal_count: int = 0,
    candidate_terminal_count: int = 0,
    baseline_nonterminal_evaluation_delta_sum: float = 0.0,
    candidate_nonterminal_evaluation_delta_sum: float = 0.0,
    **kwargs: object,
) -> dict[str, object]:
    row = _row(delta, pairs=pairs, **kwargs)
    row.update(
        {
            "candidate_catastrophic_count": row["catastrophic_count"],
            "baseline_terminal_count": baseline_terminal_count,
            "candidate_terminal_count": candidate_terminal_count,
            "baseline_nonterminal_evaluation_delta_sum": (
                baseline_nonterminal_evaluation_delta_sum
            ),
            "candidate_nonterminal_evaluation_delta_sum": (
                candidate_nonterminal_evaluation_delta_sum
            ),
            "baseline_nonterminal_count": pairs - baseline_terminal_count,
            "candidate_nonterminal_count": pairs - candidate_terminal_count,
            "continuation_iterations_executed": 17,
        }
    )
    return row


def _compute(
    rows: list[dict[str, object]],
    weights: list[float] | None = None,
    states: list[str] | None = None,
    clusters: list[str] | None = None,
    **kwargs: object,
) -> dict[str, object]:
    size = len(rows)
    alpha = kwargs.pop("alpha", 0.05)
    return compute_holdout_metrics(
        rows,
        weights or [1.0] * size,
        states or [_hash(index + 100) for index in range(size)],
        clusters or [_hash(index + 200) for index in range(size)],
        alpha=alpha,
        **kwargs,
    )


def _swap(row: dict[str, object]) -> dict[str, object]:
    swapped = dict(row)
    swapped["baseline_sum"], swapped["candidate_sum"] = (
        row["candidate_sum"],
        row["baseline_sum"],
    )
    swapped["delta_sum"] = -float(row["delta_sum"])
    swapped["catastrophic_count"], swapped["baseline_catastrophic_count"] = (
        row["baseline_catastrophic_count"],
        row["catastrophic_count"],
    )
    (
        swapped["candidate_catastrophic_severity_sum"],
        swapped["baseline_catastrophic_severity_sum"],
    ) = (
        row["baseline_catastrophic_severity_sum"],
        row["candidate_catastrophic_severity_sum"],
    )
    swapped["candidate_better_count"], swapped["baseline_better_count"] = (
        row["baseline_better_count"],
        row["candidate_better_count"],
    )
    if "candidate_catastrophic_count" in row:
        swapped["candidate_catastrophic_count"] = row[
            "baseline_catastrophic_count"
        ]
    for suffix in (
        "terminal_count",
        "nonterminal_count",
        "nonterminal_evaluation_delta_sum",
    ):
        baseline_name = f"baseline_{suffix}"
        candidate_name = f"candidate_{suffix}"
        if baseline_name in row and candidate_name in row:
            swapped[baseline_name], swapped[candidate_name] = (
                row[candidate_name],
                row[baseline_name],
            )
    return swapped


class HoldoutMetricsTest(unittest.TestCase):
    def test_exact_v5_rows_produce_weighted_nonterminal_evaluator_means(self):
        rows = [
            _v5_row(
                0.1,
                pairs=4,
                baseline_terminal_count=1,
                candidate_terminal_count=2,
                baseline_nonterminal_evaluation_delta_sum=1.5,
                candidate_nonterminal_evaluation_delta_sum=-0.5,
            ),
            _v5_row(
                -0.2,
                pairs=8,
                baseline_terminal_count=4,
                candidate_terminal_count=2,
                baseline_nonterminal_evaluation_delta_sum=-1.0,
                candidate_nonterminal_evaluation_delta_sum=3.0,
            ),
        ]
        self.assertEqual([len(row) for row in rows], [20, 20])

        summary = _compute(rows, [1.0, 3.0])
        baseline_mass = 0.25 * 3 / 4 + 0.75 * 4 / 8
        candidate_mass = 0.25 * 2 / 4 + 0.75 * 6 / 8
        expected_baseline = (0.25 * 1.5 / 4 + 0.75 * -1.0 / 8) / baseline_mass
        expected_candidate = (0.25 * -0.5 / 4 + 0.75 * 3.0 / 8) / candidate_mass

        self.assertAlmostEqual(
            summary["weighted_baseline_nonterminal_evaluation_delta_mean"],
            expected_baseline,
        )
        self.assertAlmostEqual(
            summary["weighted_candidate_nonterminal_evaluation_delta_mean"],
            expected_candidate,
        )
        self.assertAlmostEqual(
            summary["weighted_nonterminal_evaluation_delta_mean_difference"],
            expected_candidate - expected_baseline,
        )

    def test_v5_alias_partition_and_nonfinite_evaluator_sums_are_rejected(self):
        valid = _v5_row(0.1, pairs=10)
        mutations = (
            {"candidate_catastrophic_count": 1},
            {"baseline_nonterminal_count": 9},
            {"candidate_terminal_count": 1},
            {"baseline_nonterminal_evaluation_delta_sum": float("nan")},
            {"candidate_nonterminal_evaluation_delta_sum": float("inf")},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                _compute([{**valid, **mutation}])

        misspelled_alias = dict(valid)
        del misspelled_alias["candidate_catastrophic_count"]
        misspelled_alias["candidate_catastrophe_count"] = 0
        with self.assertRaises(ValueError):
            _compute([misspelled_alias])

    def test_nonterminal_evaluator_outputs_are_arm_swap_symmetric(self):
        rows = [
            _v5_row(
                0.2,
                pairs=10,
                baseline_terminal_count=2,
                candidate_terminal_count=5,
                baseline_nonterminal_evaluation_delta_sum=1.0,
                candidate_nonterminal_evaluation_delta_sum=-2.0,
            ),
            _v5_row(
                -0.1,
                pairs=20,
                baseline_terminal_count=10,
                candidate_terminal_count=4,
                baseline_nonterminal_evaluation_delta_sum=-3.0,
                candidate_nonterminal_evaluation_delta_sum=4.0,
            ),
        ]
        original = _compute(rows, [1.0, 2.0])
        swapped = _compute([_swap(row) for row in rows], [1.0, 2.0])

        self.assertAlmostEqual(
            original["weighted_candidate_nonterminal_evaluation_delta_mean"],
            swapped["weighted_baseline_nonterminal_evaluation_delta_mean"],
        )
        self.assertAlmostEqual(
            original["weighted_baseline_nonterminal_evaluation_delta_mean"],
            swapped["weighted_candidate_nonterminal_evaluation_delta_mean"],
        )
        self.assertAlmostEqual(
            original["weighted_nonterminal_evaluation_delta_mean_difference"],
            -swapped["weighted_nonterminal_evaluation_delta_mean_difference"],
        )

    def test_zero_nonterminal_mass_returns_none_symmetrically(self):
        row = _v5_row(
            0.1,
            pairs=10,
            candidate_terminal_count=10,
            baseline_nonterminal_evaluation_delta_sum=2.0,
        )
        original = _compute([row])
        swapped = _compute([_swap(row)])

        self.assertIsNone(
            original["weighted_candidate_nonterminal_evaluation_delta_mean"]
        )
        self.assertIsNotNone(
            original["weighted_baseline_nonterminal_evaluation_delta_mean"]
        )
        self.assertIsNone(
            original["weighted_nonterminal_evaluation_delta_mean_difference"]
        )
        self.assertIsNone(
            swapped["weighted_baseline_nonterminal_evaluation_delta_mean"]
        )
        self.assertIsNone(
            swapped["weighted_nonterminal_evaluation_delta_mean_difference"]
        )

    def test_exact_duplicate_clusters_are_aggregated(self):
        summary = _compute(
            [_row(0.2), _row(-0.1), _row(0.4)],
            [2.0, 1.0, 1.0],
            clusters=[_hash(1), _hash(1), _hash(2)],
        )

        self.assertEqual(summary["cluster_count"], 2)
        first = summary["cluster_aggregates"][0]
        self.assertEqual(first["world_count"], 2)
        self.assertAlmostEqual(first["normalized_weight"], 0.75)
        self.assertAlmostEqual(first["delta_mean"], 0.1)
        self.assertAlmostEqual(summary["effective_clusters"], 1.6)
        self.assertAlmostEqual(summary["positive_cluster_mass"], 1.0)

    def test_unequal_weights_and_pairs_drive_weighted_rates_and_effective_pairs(self):
        summary = _compute(
            [
                _row(
                    0.2,
                    pairs=10,
                    candidate_catastrophes=2,
                    candidate_severity=1.0,
                ),
                _row(
                    -0.2,
                    pairs=40,
                    baseline_catastrophes=4,
                    baseline_severity=3.0,
                ),
            ],
            [1.0, 3.0],
        )

        self.assertEqual(summary["normalized_weights"], [0.25, 0.75])
        self.assertAlmostEqual(summary["weighted_mean_delta"], -0.1)
        self.assertAlmostEqual(summary["candidate_catastrophe_rate"], 0.05)
        self.assertAlmostEqual(summary["baseline_catastrophe_rate"], 0.075)
        self.assertAlmostEqual(summary["candidate_catastrophe_severity_mean"], 0.5)
        self.assertAlmostEqual(summary["baseline_catastrophe_severity_mean"], 0.75)
        expected_pairs = 1.0 / (0.25**2 / 10 + 0.75**2 / 40)
        self.assertAlmostEqual(summary["effective_pairs"], expected_pairs)
        self.assertAlmostEqual(summary["candidate_better_rate"], 0.25)
        self.assertAlmostEqual(summary["baseline_better_rate"], 0.75)

    def test_between_and_within_variance_both_enter_standard_error(self):
        summary = _compute(
            [
                _row(-0.2, pairs=20, within_variance=0.16),
                _row(0.2, pairs=20, within_variance=0.16),
            ]
        )

        self.assertAlmostEqual(summary["between_cluster_variance"], 0.08)
        self.assertAlmostEqual(summary["between_mean_variance"], 0.04)
        self.assertAlmostEqual(summary["within_world_mean_variance"], 0.004)
        self.assertAlmostEqual(summary["standard_error"], math.sqrt(0.044))

    def test_permutation_invariance_includes_serialized_summary(self):
        rows = [_row(0.3), _row(-0.2, pairs=50), _row(0.1, pairs=25)]
        weights = [1.0, 4.0, 2.0]
        states = [_hash(13), _hash(11), _hash(12)]
        clusters = [_hash(23), _hash(21), _hash(21)]
        expected = compute_holdout_metrics(
            rows, weights, states, clusters, alpha=0.025, mom_groups=2
        )
        order = list(range(3))
        random.Random(7).shuffle(order)
        actual = compute_holdout_metrics(
            [rows[index] for index in order],
            [weights[index] for index in order],
            [states[index] for index in order],
            [clusters[index] for index in order],
            alpha=0.025,
            mom_groups=2,
        )

        self.assertEqual(
            json.dumps(actual, sort_keys=True, allow_nan=False),
            json.dumps(expected, sort_keys=True, allow_nan=False),
        )

    def test_arm_swap_symmetry(self):
        rows = [
            _row(0.4, candidate_catastrophes=2, candidate_severity=1.2),
            _row(-0.1, baseline_catastrophes=3, baseline_severity=0.9),
            _row(0.2),
        ]
        original = _compute(rows, [1.0, 2.0, 3.0], mom_groups=3)
        swapped = _compute([_swap(row) for row in rows], [1.0, 2.0, 3.0], mom_groups=3)

        self.assertAlmostEqual(
            original["weighted_candidate_mean"], swapped["weighted_baseline_mean"]
        )
        self.assertAlmostEqual(
            original["weighted_mean_delta"], -swapped["weighted_mean_delta"]
        )
        self.assertAlmostEqual(
            original["candidate_tail_upper_confidence_bound"],
            swapped["baseline_tail_upper_confidence_bound"],
        )
        self.assertAlmostEqual(
            original["candidate_lower_tail_cvar"],
            swapped["baseline_lower_tail_cvar"],
        )
        self.assertAlmostEqual(
            original["candidate_median_of_means"],
            swapped["baseline_median_of_means"],
        )
        self.assertAlmostEqual(original["standard_error"], swapped["standard_error"])

    def test_zero_events_retain_positive_cluster_aware_uncertainty(self):
        rows = [_row(0.1, pairs=10_000) for _ in range(4)]
        summary = _compute(rows)
        expected = math.sqrt(math.log(20.0) / (2.0 * 4.0))

        self.assertEqual(summary["candidate_catastrophe_rate"], 0.0)
        self.assertEqual(summary["candidate_catastrophe_severity_mean"], 0.0)
        self.assertAlmostEqual(summary["tail_effective_sample_size"], 4.0)
        self.assertAlmostEqual(
            summary["candidate_tail_upper_confidence_bound"], expected
        )
        self.assertGreater(summary["candidate_tail_upper_confidence_bound"], 0.0)

    def test_weighted_cvar_uses_fractional_boundary_mass_in_both_directions(self):
        summary = _compute(
            [_row(-0.8), _row(-0.2), _row(0.4)],
            [0.1, 0.3, 0.6],
            cvar_tail_mass=0.25,
        )

        self.assertAlmostEqual(
            summary["candidate_lower_tail_cvar"], (-0.8 * 0.1 - 0.2 * 0.15) / 0.25
        )
        self.assertAlmostEqual(summary["baseline_lower_tail_cvar"], -0.4)

    def test_median_of_means_uses_hash_ordered_cluster_buckets(self):
        summary = _compute(
            [_row(-0.9), _row(0.1), _row(0.2), _row(0.8), _row(0.3)],
            clusters=[_hash(1), _hash(2), _hash(3), _hash(4), _hash(5)],
            mom_groups=3,
        )

        # Round-robin hash buckets are [-0.9, 0.8], [0.1, 0.3], and [0.2].
        for actual, expected in zip(
            summary["candidate_mom_group_means"], [-0.05, 0.2, 0.2], strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(summary["candidate_median_of_means"], 0.2)
        self.assertAlmostEqual(summary["baseline_median_of_means"], -0.2)

    def test_malformed_input_fails_closed(self):
        valid = _row(0.1)
        cases = {
            "missing catastrophe arm": {
                **valid,
                "baseline_catastrophic_count": None,
            },
            "inconsistent sums": {**valid, "delta_sum": 99.0},
            "inconsistent moments": {**valid, "delta_squared_sum": 0.0},
            "comparison counts": {**valid, "equal_count": 1},
            "severity without events": {
                **valid,
                "candidate_catastrophic_severity_sum": 0.1,
            },
            "unknown field": {**valid, "typo": 0},
        }
        for name, row in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                _compute([row])

        for weights in ([float("nan")], [-1.0], [0.0]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                _compute([valid], weights)
        with self.assertRaises(ValueError):
            _compute([valid], states=["not-a-hash"])
        with self.assertRaises(ValueError):
            _compute([valid], clusters=["A" * 64])
        with self.assertRaises(ValueError):
            compute_holdout_metrics(
                [valid, valid],
                [1.0, 1.0],
                [_hash(1), _hash(1)],
                [_hash(2), _hash(3)],
                alpha=0.05,
            )
        for alpha in (0.0, 1.0, float("nan")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                _compute([valid], alpha=alpha)

    def test_summary_is_strict_json_serializable(self):
        summary = _compute([_row(0.0, within_variance=0.2)])
        self.assertIsNone(
            summary["weighted_candidate_nonterminal_evaluation_delta_mean"]
        )
        self.assertIsNone(
            summary["weighted_baseline_nonterminal_evaluation_delta_mean"]
        )
        self.assertIsNone(
            summary["weighted_nonterminal_evaluation_delta_mean_difference"]
        )
        payload = json.dumps(summary, sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(payload)["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
