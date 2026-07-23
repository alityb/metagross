from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
REPRESENTATIVE_REPLAY = (
    PROJECT_ROOT
    / "experiments/mcts_high_budget_verified/raw/w1/mcts_high_budget_v2_final_w1/raw"
    / "accepted_r1_vs_accepted_r1_peer/shard_00/replays"
    / "battle-gen9randombattle-18826_63cc6091x023e311.json"
)
RANDBATS_POOL = PROJECT_ROOT / "data/randbats_pools/gen9randombattle_pool_50000.json"
sys.path.insert(0, str(ROOT))

from scripts.finalize_schema_v2_pfsp import finalize
from scripts.parse_randbats_replays import parse_replay_dir
from scripts import parse_randbats_replays


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


def write_replay_admission_shard(root: Path, battles: list[str]) -> Path:
    shard = root / "round" / "shard_00"
    replay_dir = shard / "replays"
    replay_dir.mkdir(parents=True)
    rows = []
    for battle in battles:
        (replay_dir / f"{battle}_capture.json").write_text(
            json.dumps({"id": battle, "players": ["learner", "opponent"], "_our_name": "learner"})
        )
        rows.append(
            {
                "record_type": "decision",
                "battle_tag": battle,
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
    (shard / "agent_a_decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return shard


class FinalizeSchemaV2PFSPTests(unittest.TestCase):
    def _run_replay_admission(self, battles, povs_by_battle, build_results=None):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        raw_root = base / "raw"
        write_replay_admission_shard(raw_root, battles)
        pool = base / "pool.json"
        pool.write_text("{}")
        parsed_root = base / "parsed"
        learner_root = base / "learner"
        identity = base / "identity.jsonl"
        sidecar = base / "sidecar.jsonl"
        report_path = base / "report.json"

        def parse(_replay_dir, out_dir, _pool_path, workers):
            self.assertEqual(workers, 1)
            out_dir.mkdir(parents=True)
            for battle, povs in povs_by_battle.items():
                for pov in povs:
                    (out_dir / f"{battle}_capture_Unrated_{pov}_vs_other.json.lz4").write_bytes(b"fixture")
            valid = sum(len(povs) == 2 for povs in povs_by_battle.values())
            return {
                "already_parsed": 0,
                "replays_to_parse": len(battles),
                "parsed_ok": valid,
                "failed": len(battles) - valid,
                "total_pov_trajectories": sum(len(povs) for povs in povs_by_battle.values()),
            }

        def index(root, output):
            output.write_text('{"identity":true}\n')
            return {"trajectories": len(list(root.rglob("*.json.lz4")))}

        calls = []

        def build(logs, _root, output, _index):
            rows = [json.loads(line) for line in logs[0].read_text().splitlines()]
            calls.append(rows)
            output.write_text('{"target":true}\n')
            if build_results:
                return build_results[len(calls) - 1]
            return {"accepted": len(rows), "rejected": 0, "invalid_rows": 0, "rejected_povs": []}

        with (
            patch("scripts.finalize_schema_v2_pfsp.parse_replay_dir", side_effect=parse),
            patch("scripts.finalize_schema_v2_pfsp.build_trajectory_index", side_effect=index),
            patch("scripts.finalize_schema_v2_pfsp.build_sidecar", side_effect=build),
        ):
            report = finalize(
                raw_root,
                parsed_root,
                learner_root,
                identity,
                sidecar,
                pool,
                report_path,
                replay_admission=True,
            )
        return report, calls

    def test_replay_admission_isolates_parser_failures(self):
        report, calls = self._run_replay_admission(
            ["battle-good", "battle-parser-failed"],
            {"battle-good": ["learner", "opponent"], "battle-parser-failed": []},
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["stages"]["admission"]["raw"], 2)
        self.assertEqual(report["stages"]["admission"]["parser_valid"], 1)
        self.assertEqual(report["stages"]["admission"]["target_valid"], 1)
        self.assertEqual(report["stages"]["admission"]["exclusions_by_reason"], {"parser_output_count": 1})
        self.assertEqual([row["battle_tag"] for row in calls[-1]], ["battle-good"])

    def test_replay_admission_isolates_missing_learner_pov(self):
        report, calls = self._run_replay_admission(
            ["battle-good", "battle-wrong-pov"],
            {"battle-good": ["learner", "opponent"], "battle-wrong-pov": ["other", "opponent"]},
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["stages"]["admission"]["parser_valid"], 2)
        self.assertEqual(report["stages"]["admission"]["learner_valid"], 1)
        self.assertEqual(report["stages"]["admission"]["exclusions_by_reason"], {"missing_or_ambiguous_learner_pov": 1})
        self.assertEqual([row["battle_tag"] for row in calls[-1]], ["battle-good"])

    def test_replay_admission_isolates_target_failures(self):
        report, calls = self._run_replay_admission(
            ["battle-good", "battle-bad-target"],
            {"battle-good": ["learner", "opponent"], "battle-bad-target": ["learner", "opponent"]},
            [
                {
                    "accepted": 1,
                    "rejected": 1,
                    "invalid_rows": 0,
                    "rejected_povs": [
                        {
                            "battle_tag": "battle-bad-target",
                            "learner_pov": "learner",
                            "reason": "explicit_illegal_or_unmappable_target",
                        }
                    ],
                },
                {"accepted": 1, "rejected": 0, "invalid_rows": 0, "rejected_povs": []},
            ],
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["stages"]["admission"]["target_valid"], 1)
        self.assertEqual(
            report["stages"]["admission"]["exclusions_by_reason"],
            {"target_explicit_illegal_or_unmappable_target": 1},
        )
        self.assertEqual([row["battle_tag"] for row in calls[0]], ["battle-bad-target", "battle-good"])
        self.assertEqual([row["battle_tag"] for row in calls[1]], ["battle-good"])

    def _parse_with_emitted_povs(self, emitted_povs: int):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        replay_dir = base / "replays"
        replay_dir.mkdir()
        (replay_dir / "battle-one_capture.json").write_text(json.dumps({"id": "battle-one"}))
        pool = base / "pool.json"
        pool.write_text("{}")
        out_dir = base / "parsed"

        class Parser:
            def parse_replay(self, _path):
                for index in range(emitted_povs):
                    (out_dir / f"battle-one_Unrated_p{index}_vs_other.json.lz4").write_bytes(b"fixture")

        def init_worker(_out_dir, _pool_path, _seed):
            parse_randbats_replays._worker_parser = Parser()

        with patch("scripts.parse_randbats_replays._init_worker", side_effect=init_worker):
            summary = parse_replay_dir(replay_dir, out_dir, pool, workers=1)
        return summary, out_dir

    def test_parser_rejects_metamon_style_silent_zero_pov_return(self):
        summary, out_dir = self._parse_with_emitted_povs(0)

        self.assertEqual(summary["parsed_ok"], 0)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["total_pov_trajectories"], 0)
        self.assertEqual(list(out_dir.glob("*.json.lz4")), [])

    def test_parser_rejects_and_cleans_a_single_pov_return(self):
        summary, out_dir = self._parse_with_emitted_povs(1)

        self.assertEqual(summary["parsed_ok"], 0)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["total_pov_trajectories"], 0)
        self.assertEqual(list(out_dir.glob("*.json.lz4")), [])

    def test_parser_counts_exactly_two_povs_as_success(self):
        summary, out_dir = self._parse_with_emitted_povs(2)

        self.assertEqual(summary["parsed_ok"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["total_pov_trajectories"], 2)
        self.assertEqual(len(list(out_dir.glob("*.json.lz4"))), 2)

    @unittest.skipUnless(
        REPRESENTATIVE_REPLAY.is_file() and RANDBATS_POOL.is_file(),
        "requires the local verified replay and RandBats pool",
    )
    def test_parser_emits_two_unknown_item_povs_for_jumpluff_replay(self):
        import lz4.frame

        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            replay_dir = base / "replays"
            replay_dir.mkdir()
            shutil.copy2(REPRESENTATIVE_REPLAY, replay_dir / REPRESENTATIVE_REPLAY.name)
            out_dir = base / "parsed"

            summary = parse_replay_dir(replay_dir, out_dir, RANDBATS_POOL, workers=1)

            self.assertEqual(summary["parsed_ok"], 1)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["total_pov_trajectories"], 2)
            jumpluff_items = set()
            for path in out_dir.glob("*.json.lz4"):
                with lz4.frame.open(path, "rb") as handle:
                    payload = json.loads(handle.read())
                for state in payload["states"]:
                    pokemon = [
                        state["player_active_pokemon"],
                        state["opponent_active_pokemon"],
                        *state["available_switches"],
                    ]
                    jumpluff_items.update(
                        mon["item"] for mon in pokemon if mon["name"] == "jumpluff"
                    )
            self.assertIn("unknownitem", jumpluff_items)

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
