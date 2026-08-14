from __future__ import annotations

import unittest

from experimental.src.scripts import verify_r1_history_dump_parity


class HistoryDumpParityTests(unittest.TestCase):
    def test_groups_interleaved_dump_by_namespace_and_tag(self):
        rows = [
            {"namespace": "a", "tag": "battle-1", "decision_idx": 0},
            {"namespace": "a", "tag": "battle-2", "decision_idx": 0},
            {"namespace": "a", "tag": "battle-1", "decision_idx": 1},
        ]

        grouped = verify_r1_history_dump_parity.group_session_rows(rows)

        self.assertEqual([identity for identity, _rows in grouped], [
            ("a", "battle-1"),
            ("a", "battle-2"),
        ])
        self.assertEqual(
            [[row["decision_idx"] for row in session] for _identity, session in grouped],
            [[0, 1], [0]],
        )

    def test_rejects_missing_session_identity(self):
        with self.assertRaisesRegex(
            verify_r1_history_dump_parity.HistoryParityError,
            "session identity",
        ):
            verify_r1_history_dump_parity.group_session_rows([
                {"namespace": "a", "tag": ""}
            ])


if __name__ == "__main__":
    unittest.main()
