from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.summarize_counterbalanced_ab import CounterbalanceError, summarize


def games(winners: list[str]) -> dict[str, object]:
    rows = []
    for index, winner in enumerate(winners, 1):
        pair_index = (index + 1) // 2
        rows.append({
            "game_index": index,
            "pair_index": pair_index,
            "pair_id": f"pair-{pair_index}",
            "pair_leg": 1 if index % 2 else 2,
            "battle_seed": f"seed-{pair_index}",
            "team_1_sha256": f"team-a-{pair_index}",
            "team_2_sha256": f"team-b-{pair_index}",
            "winner": winner,
            "void": False,
            "error": None,
        })
    return {"games": rows}


class CounterbalancedSummaryTests(unittest.TestCase):
    def test_scores_g4_by_model_slot(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            as_a = root / "a.json"
            as_b = root / "b.json"
            as_a.write_text(json.dumps(games(["agent_a", "agent_b"])))
            as_b.write_text(json.dumps(games(["agent_b", "agent_b"])))
            result = summarize(as_a, as_b, 2)
            self.assertEqual(result["g4_wins"], 3)
            self.assertEqual(result["r1_wins"], 1)
            self.assertEqual(result["g4_as_agent_a"]["wins"], 1)
            self.assertEqual(result["g4_as_agent_b"]["wins"], 2)

    def test_rejects_mismatched_matchups(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            as_a = root / "a.json"
            as_b = root / "b.json"
            first = games(["agent_a", "agent_b"])
            second = games(["agent_b", "agent_a"])
            second["games"][0]["battle_seed"] = "different"
            as_a.write_text(json.dumps(first))
            as_b.write_text(json.dumps(second))
            with self.assertRaisesRegex(CounterbalanceError, "identical matchups"):
                summarize(as_a, as_b, 2)


if __name__ == "__main__":
    unittest.main()
