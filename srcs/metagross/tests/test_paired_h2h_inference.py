from __future__ import annotations

import unittest

from srcs.metagross import paired_h2h_inference


def game(pair, leg, winner, **overrides):
    row = {
        "agent_a": "candidate",
        "agent_b": "comparator",
        "pair_id": pair,
        "pair_leg": leg,
        "winner": winner,
        "void": False,
        "error": None,
    }
    row.update(overrides)
    return row


class PairedH2hInferenceTest(unittest.TestCase):
    def test_pair_scores_use_complete_mirrored_pairs(self):
        payload = {
            "games": [
                game("a", 1, "agent_a"),
                game("a", 2, "agent_a"),
                game("b", 1, "agent_a"),
                game("b", 2, "agent_b"),
                game("c", 1, "agent_b"),
                game("c", 2, "agent_b"),
            ]
        }
        self.assertEqual(
            paired_h2h_inference.pair_scores(payload, "candidate", "comparator"),
            [1.0, 0.5, 0.0],
        )

    def test_incomplete_void_and_wrong_treatment_fail_closed(self):
        invalid = (
            {"games": [game("a", 1, "agent_a")]},
            {
                "games": [
                    game("a", 1, "agent_a", void=True),
                    game("a", 2, "agent_b"),
                ]
            },
            {
                "games": [
                    game("a", 1, "agent_a", agent_a="other"),
                    game("a", 2, "agent_b"),
                ]
            },
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                paired_h2h_inference.pair_scores(
                    payload, "candidate", "comparator"
                )

    def test_bootstrap_is_deterministic_and_claims_require_250_pairs(self):
        games = []
        for index in range(10):
            games.extend(
                [
                    game(str(index), 1, "agent_a"),
                    game(str(index), 2, "agent_a"),
                ]
            )
        first = paired_h2h_inference.analyze(
            {"games": games},
            candidate="candidate",
            comparator="comparator",
            resamples=1_000,
            seed=7,
        )
        second = paired_h2h_inference.analyze(
            {"games": games},
            candidate="candidate",
            comparator="comparator",
            resamples=1_000,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertTrue(first["claims"]["positive_point_estimate"])
        self.assertFalse(
            first["claims"]["statistically_better_than_search_guided_r1"]
        )


if __name__ == "__main__":
    unittest.main()
