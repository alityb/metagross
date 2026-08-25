from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from srcs.metagross.known_team_belief_eval import (
    _score,
    _summary,
    _mixture,
    battle_split,
    load_corpus,
    truth_candidate_id,
)


def corpus_row(battle_id="known-team-1"):
    team = [
        {
            "species": f"Species {index}",
            "level": 80,
            "moves": ["Tackle"],
            "ability": "Pressure",
            "item": "Leftovers",
            "teraType": "Water",
        }
        for index in range(6)
    ]
    view = {
        "chunks": ["|request|{}"],
        "decisions": [{"chunk_count": 1, "action": "tackle"}],
    }
    return {
        "schema": "metagross-known-team-battle/v1",
        "battle_id": battle_id,
        "teams": {"p1": {"sets": team}, "p2": {"sets": team}},
        "views": {"p1": view, "p2": view},
    }


class KnownTeamBeliefEvalTest(unittest.TestCase):
    def test_load_corpus_rejects_duplicate_battles(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.jsonl"
            row = json.dumps(corpus_row())
            path.write_text(f"{row}\n{row}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate battle"):
                load_corpus(path)

    def test_load_corpus_rejects_nonmonotone_decisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.jsonl"
            row = corpus_row()
            row["views"]["p1"] = {
                "chunks": ["one", "two"],
                "decisions": [
                    {"chunk_count": 2, "action": "tackle"},
                    {"chunk_count": 1, "action": "tackle"},
                ],
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid p1 decisions"):
                load_corpus(path)

    def test_truth_identity_is_order_invariant_for_moves(self):
        left = {
            "species": "Ho-Oh",
            "level": 80,
            "moves": ["Brave Bird", "Recover"],
            "ability": "Regenerator",
            "item": "Heavy-Duty Boots",
            "teraType": "Ground",
        }
        right = {**left, "moves": list(reversed(left["moves"]))}
        self.assertEqual(truth_candidate_id(left), truth_candidate_id(right))

    def test_battle_split_is_deterministic_and_known(self):
        self.assertEqual(battle_split("known-team-000000"), "test")
        self.assertEqual(
            battle_split("known-team-000001"), battle_split("known-team-000001")
        )

    def test_score_accounts_for_truth_outside_support(self):
        score = _score({"other": 1.0}, "truth")
        self.assertEqual(score["truth_probability"], 0.0)
        self.assertIsNone(score["nll"])
        self.assertEqual(score["brier"], 2.0)
        self.assertIsNone(score["rank"])

    def test_mixture_falls_back_to_available_normalized_belief(self):
        proposed = {"truth": 0.75, "other": 0.25}
        self.assertEqual(_mixture({}, proposed, 0.25), proposed)
        self.assertEqual(_mixture(proposed, {}, 0.25), proposed)
        mixed = _mixture({"truth": 0.5, "other": 0.5}, proposed, 0.25)
        self.assertAlmostEqual(sum(mixed.values()), 1.0)
        self.assertAlmostEqual(mixed["truth"], 0.5625)

    def test_summary_uses_battle_cluster_bootstrap(self):
        rows = []
        for battle_id, proposed_probability, current_probability in (
            ("a", 0.8, 0.6),
            ("b", 0.7, 0.5),
        ):
            row = {
                    "battle_id": battle_id,
                    "proposed": _score(
                        {"truth": proposed_probability, "other": 1 - proposed_probability},
                        "truth",
                    ),
                    "current": _score(
                        {"truth": current_probability, "other": 1 - current_probability},
                        "truth",
                    ),
                }
            for alpha in (0.25, 0.5, 0.75):
                row[f"mixture_{int(alpha * 100):02d}"] = _score(
                    _mixture(
                        {"truth": current_probability, "other": 1 - current_probability},
                        {"truth": proposed_probability, "other": 1 - proposed_probability},
                        alpha,
                    ),
                    "truth",
                )
            rows.append(row)
        summary = _summary(rows)
        self.assertAlmostEqual(
            summary["paired_delta_proposed_minus_current"]["mean_truth_probability"],
            0.2,
        )
        self.assertIsNotNone(
            summary["battle_cluster_bootstrap_ci95"]["proposed"][
                "mean_truth_probability"
            ]
        )
        self.assertIsNotNone(
            summary["battle_cluster_bootstrap_ci95"]["mixture_25"]["brier"]
        )


if __name__ == "__main__":
    unittest.main()
