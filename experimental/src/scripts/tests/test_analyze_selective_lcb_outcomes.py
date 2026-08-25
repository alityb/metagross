from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analyze_selective_lcb_outcomes import AnalysisError, analyze, render_markdown  # noqa: E402


def _record(*, triggered: bool, lcb: float | None = None, overridden: bool = False) -> dict:
    return {
        "triggered": triggered,
        "paired_lcb": lcb,
        "paired_evaluation_complete": triggered,
        "overridden": overridden,
    }


class AnalyzeSelectiveLcbOutcomesTests(unittest.TestCase):
    def _write_run(self, root: Path) -> None:
        games = [
            {"game_index": 1, "winner": "agent_a", "void": False},
            {"game_index": 2, "winner": "agent_b", "void": False},
            {"game_index": 3, "winner": "agent_a", "void": False},
            {"game_index": 4, "winner": "agent_b", "void": False},
        ]
        (root / "result.json.progress.json").write_text(
            json.dumps({"games": games}), encoding="ascii"
        )
        logs = root / "logs"
        logs.mkdir()
        records = {
            1: [_record(triggered=True, lcb=0.02, overridden=True)],
            2: [_record(triggered=True, lcb=0.04)],
            3: [_record(triggered=True, lcb=-0.01)],
            4: [_record(triggered=False)],
        }
        for game, entries in records.items():
            text = "".join(
                f"INFO SELECTIVE_SHARED {json.dumps(entry)}\n" for entry in entries
            )
            (logs / f"gatex{game:03d}abcd.log").write_text(text, encoding="ascii")

    def test_analysis_separates_positive_lcb_from_actual_override(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root)
            report = analyze(root)
        self.assertEqual(report["telemetry"]["triggered_events"], 3)
        self.assertEqual(report["telemetry"]["positive_lcb_events"], 2)
        self.assertEqual(report["telemetry"]["override_events"], 1)
        classes = report["exclusive_game_classes"]
        self.assertEqual(classes["actual_override"]["wins"], 1)
        self.assertEqual(classes["positive_lcb_retained_baseline"]["losses"], 1)
        self.assertEqual(classes["triggered_without_positive_lcb"]["wins"], 1)
        self.assertEqual(classes["no_trigger"]["losses"], 1)
        self.assertEqual(report["override_lcb_bins"]["0.01_to_0.025"]["events"], 1)
        self.assertIn("descriptive only", render_markdown(report))

    def test_incomplete_treatment_and_missing_telemetry_fail_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root)
            log = root / "logs" / "gatex001abcd.log"
            bad = _record(triggered=True, lcb=0.02, overridden=True)
            bad["paired_evaluation_complete"] = False
            log.write_text(f"SELECTIVE_SHARED {json.dumps(bad)}\n", encoding="ascii")
            with self.assertRaisesRegex(AnalysisError, "incomplete paired evaluation"):
                analyze(root)
            log.unlink()
            with self.assertRaisesRegex(AnalysisError, "lack candidate telemetry"):
                analyze(root)


if __name__ == "__main__":
    unittest.main()
