from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from srcs.metagross.known_team_decision_v2 import (
    CORPUS_SCHEMA,
    action_rows,
    aggregate_visits,
    argmax,
    canonical_json,
    load_corpus_v2,
    observer_for,
    OBSERVED_PARTIAL_CANDIDATE,
    partition_corpus,
    root_beliefs,
    wilson_interval,
)
from srcs.metagross.run_foul_play import _authorized_action_name


def corpus_row(index: int, uid: str | None = None) -> dict:
    uid = uid or f"{index:024x}"
    team = [{"species": f"Species {slot}"} for slot in range(6)]
    view = {
        "chunks": ["|request|{}"],
        "decisions": [{"chunk_count": 1, "action": "tackle"}],
    }
    return {
        "schema": CORPUS_SCHEMA,
        "corpus_uid": uid,
        "battle_id": f"known-team-v2-{uid}",
        "battle_index": index,
        "teams": {"p1": {"sets": team}, "p2": {"sets": team}},
        "views": {"p1": view, "p2": view},
    }


class KnownTeamDecisionV2Test(unittest.TestCase):
    def test_missing_history_distribution_falls_back_to_current(self):
        class Member:
            name = "mew"
            def is_alive(self):
                return True

        state = SimpleNamespace(
            _metagross_public_events=(),
            _metagross_random_battle_sets={"mew": ()},
            opponent=SimpleNamespace(active=Member(), reserve=[]),
        )
        # No candidates is the exact observed-partial production state.
        current, history, tv, affected = root_beliefs(state, {"mew": ()})
        self.assertEqual(current, history)
        self.assertEqual(current["mew"], {OBSERVED_PARTIAL_CANDIDATE: 1.0})
        self.assertEqual(tv, 0.0)
        self.assertEqual(affected, 0)

    def test_canonical_json_rejects_nonfinite_values(self):
        with self.assertRaises(ValueError):
            canonical_json({"bad": float("nan")})

    def test_loader_rejects_cross_corpus_identity_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.jsonl"
            first = corpus_row(1)
            second = corpus_row(2, first["corpus_uid"])
            path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate v2 corpus identity"):
                load_corpus_v2(path)

    def test_partition_and_observer_are_order_invariant(self):
        rows = [corpus_row(index) for index in range(400)]
        representative, sensitivity = partition_corpus(rows)
        reverse_representative, reverse_sensitivity = partition_corpus(list(reversed(rows)))
        self.assertEqual(
            [row["corpus_uid"] for row in representative],
            [row["corpus_uid"] for row in reverse_representative],
        )
        self.assertEqual(
            [row["corpus_uid"] for row in sensitivity],
            [row["corpus_uid"] for row in reverse_sensitivity],
        )
        self.assertEqual(observer_for(rows[0]), observer_for(dict(rows[0])))

    def test_partition_preregisters_one_extension(self):
        rows = [corpus_row(index) for index in range(600)]
        representative, sensitivity = partition_corpus(rows)
        self.assertEqual(len(representative), 375)
        self.assertEqual(len(sensitivity), 225)
        self.assertEqual(
            {row["battle_index"] for row in representative + sensitivity},
            set(range(600)),
        )

    def test_partition_rejects_unregistered_corpus_size(self):
        with self.assertRaisesRegex(ValueError, "exactly 400"):
            partition_corpus([corpus_row(index) for index in range(401)])

    def test_action_rows_filter_illegal_variants_and_compute_q(self):
        rows = [
            SimpleNamespace(move_choice="tackle", visits=10, total_score=6.0),
            SimpleNamespace(move_choice="tackle-tera", visits=30, total_score=21.0),
        ]
        result = SimpleNamespace(side_one=rows)
        parsed = action_rows(result, {"tackle"})
        self.assertEqual(parsed["tackle"]["visits"], 10)
        self.assertAlmostEqual(parsed["tackle"]["q"], 0.6)

    def test_minior_engine_form_maps_to_request_cosmetic_form(self):
        allowed = {"switch miniorviolet", "switch thundurus"}
        self.assertEqual(
            _authorized_action_name("switch miniormeteor", allowed),
            "switch miniorviolet",
        )

    def test_visit_aggregation_and_argmax_are_canonical(self):
        searches = [
            {"a": {"visits": 8}, "b": {"visits": 2}},
            {"a": {"visits": 1}, "b": {"visits": 9}},
        ]
        policy = aggregate_visits(searches)
        self.assertAlmostEqual(sum(policy.values()), 1.0)
        self.assertEqual(argmax({"z": 0.5, "a": 0.5}), "a")

    def test_wilson_interval_handles_zero_events(self):
        lower, upper = wilson_interval(0, 200)
        self.assertEqual(lower, 0.0)
        self.assertLess(upper, 0.05)


if __name__ == "__main__":
    unittest.main()
