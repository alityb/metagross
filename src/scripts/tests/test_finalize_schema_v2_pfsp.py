from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.finalize_schema_v2_pfsp import finalize
from scripts.parse_randbats_replays import parse_replay_dir


def write_strict_shard(root: Path, name: str = "shard_00") -> Path:
    shard = root / "round" / name
    replay_dir = shard / "replays"
    replay_dir.mkdir(parents=True)
    (replay_dir / "battle-one_capture.json").write_text(
        json.dumps({"id": "battle-one", "players": ["learner", "opponent"], "_our_name": "learner"})
    )
    (shard / "agent_a_decisions.jsonl").write_text(
        json.dumps(
            {
                "record_type": "decision",
                "battle_tag": "battle-one",
                "username": "learner",
                "learner_pov": "learner",
                "mcts_schema_version": 2,
                "mcts_decision_seq": 0,
                "mcts_visits": {"tackle": 1.0},
                "selected_action": "tackle",
                "root_prior_count": 1,
                "opponent_prior_count": 1,
                "canonical_selected_action_index": 0,
                "mcts_visit_target_13": [1.0] + [0.0] * 12,
            }
        )
        + "\n"
    )
    return shard


class FinalizeSchemaV2PFSPTests(unittest.TestCase):
    def test_finalizes_nested_shard_idempotently(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            raw_root = base / "raw"
            shard = write_strict_shard(raw_root)
            pool = base / "pool.json"
            pool.write_text("{}")
            parsed_root = base / "parsed"
            learner_root = base / "learner"
            identity = base / "identity.jsonl"
            sidecar = base / "sidecar.jsonl"
            report_path = base / "report.json"

            def parse(replay_dir, out_dir, pool_path, workers):
                self.assertEqual(workers, 1)
                self.assertEqual(replay_dir, (shard / "replays").resolve())
                out_dir.mkdir(parents=True, exist_ok=True)
                return {"already_parsed": 0, "replays_to_parse": 1, "parsed_ok": 1, "failed": 0, "total_pov_trajectories": 2}

            def filter_povs(raw_dir, parsed_dir, out_dir):
                out_dir.mkdir(parents=True, exist_ok=True)
                return {"raw_learner_povs": 1, "parsed_input": 2, "learner_trajectories": 1, "malformed_parsed_names": 0}

            def index(root, output):
                output.write_text('{"identity":true}\n')
                return {"trajectories": 1}

            def build(logs, root, output, index_path):
                self.assertEqual(logs, [(shard / "agent_a_decisions.jsonl").resolve()])
                output.write_text('{"target":true}\n')
                return {"accepted": 1, "rejected": 0, "invalid_rows": 0}

            with (
                patch("scripts.finalize_schema_v2_pfsp.parse_replay_dir", side_effect=parse),
                patch("scripts.finalize_schema_v2_pfsp.filter_learner_povs", side_effect=filter_povs),
                patch("scripts.finalize_schema_v2_pfsp.build_trajectory_index", side_effect=index),
                patch("scripts.finalize_schema_v2_pfsp.build_sidecar", side_effect=build),
            ):
                first = finalize(raw_root, parsed_root, learner_root, identity, sidecar, pool, report_path)
                second = finalize(raw_root, parsed_root, learner_root, identity, sidecar, pool, report_path)

            self.assertTrue(first["ok"], first)
            self.assertTrue(second["ok"], second)
            self.assertEqual(json.loads(report_path.read_text())["stages"]["sidecar"]["accepted"], 1)
            self.assertEqual(identity.read_text(), '{"identity":true}\n')
            self.assertEqual(sidecar.read_text(), '{"target":true}\n')

    def test_writes_a_failure_report_before_rejecting_invalid_shard(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            raw_root = base / "raw"
            shard = write_strict_shard(raw_root)
            rows = json.loads((shard / "agent_a_decisions.jsonl").read_text())
            rows["root_prior_count"] = 0
            (shard / "agent_a_decisions.jsonl").write_text(json.dumps(rows) + "\n")
            pool = base / "pool.json"
            pool.write_text("{}")
            report_path = base / "report.json"

            with patch("scripts.finalize_schema_v2_pfsp.parse_replay_dir") as parse:
                report = finalize(
                    raw_root,
                    base / "parsed",
                    base / "learner",
                    base / "identity.jsonl",
                    base / "sidecar.jsonl",
                    pool,
                    report_path,
                )

            self.assertFalse(report["ok"])
            self.assertEqual(report["stages"]["validation"]["failed"], 1)
            self.assertTrue(json.loads(report_path.read_text())["integrity_errors"])
            parse.assert_not_called()

    def test_parser_rejects_partial_existing_povs_without_starting_a_worker(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            replay_dir = base / "replays"
            replay_dir.mkdir()
            (replay_dir / "capture.json").write_text(json.dumps({"id": "battle-one"}))
            pool = base / "pool.json"
            pool.write_text("{}")
            out_dir = base / "parsed"
            out_dir.mkdir()
            (out_dir / "battle-one_Unrated_a_vs_b.json.lz4").write_bytes(b"partial")

            with self.assertRaisesRegex(ValueError, "incomplete or ambiguous"):
                parse_replay_dir(replay_dir, out_dir, pool, workers=1)


if __name__ == "__main__":
    unittest.main()
