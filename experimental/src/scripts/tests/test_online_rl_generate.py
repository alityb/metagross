from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.online_rl_generate import (
    CollectionError,
    _completed_chunk,
    load_plan,
    main,
    outcome_totals,
    policy_command,
    write_battle_ledger,
)


LOCAL = {
    "kind": "local",
    "run_dir": "srcs/models",
    "run_name": "randbats_exit_r1",
    "checkpoint": 5,
    "checkpoint_sha256": "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93",
}


class OnlineRLGenerateTests(unittest.TestCase):
    def test_local_policy_command_pins_checkpoint_and_trajectory_output(self):
        command = policy_command(
            python="python",
            profile=LOCAL,
            username="learner",
            opponent_username="opponent",
            role="acceptor",
            battle_format="gen9randombattle",
            battles=4,
            results_dir=Path("results"),
            trajectories_dir=Path("trajectories"),
            showdown_port=8011,
        )
        self.assertEqual(command[command.index("--local-run-name") + 1], "randbats_exit_r1")
        self.assertEqual(command[command.index("--checkpoint") + 1], "5")
        self.assertEqual(command[command.index("--save-trajectories-to") + 1], "trajectories")
        self.assertEqual(command[command.index("--showdown-port") + 1], "8011")

    def test_empty_alias_preserves_native_randbats_format_token(self):
        command = policy_command(
            python="python",
            profile={**LOCAL, "alias_to": ""},
            username="learner",
            opponent_username="opponent",
            role="acceptor",
            battle_format="gen9randombattle",
            battles=1,
            results_dir=Path("results"),
            trajectories_dir=None,
            showdown_port=8000,
        )
        self.assertEqual(command[command.index("--alias-to") + 1], "")

    def test_load_plan_rejects_unpinned_local_profile(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / "pool.json"
            schedule = root / "schedule.json"
            pool.write_text(json.dumps({"profiles": {"learner": {"kind": "local"}}}))
            schedule.write_text(json.dumps({"learner": "learner", "opponents": ["learner"]}))
            with self.assertRaisesRegex(CollectionError, "missing run_dir, run_name, checkpoint, checkpoint_sha256"):
                load_plan(pool, schedule)

    def test_dry_run_is_role_balanced_and_writes_atomic_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / "pool.json"
            schedule = root / "schedule.json"
            output = root / "out"
            pool.write_text(json.dumps({
                "format": "gen9randombattle",
                "profiles": {"learner": LOCAL, "frozen": LOCAL},
            }))
            schedule.write_text(json.dumps({
                "learner": "learner",
                "opponents": ["frozen"] * 5,
            }))
            argv = [
                "online_rl_generate.py", "--pool", str(pool), "--schedule", str(schedule),
                "--out-dir", str(output), "--chunk-games", "5", "--dry-run",
            ]
            with patch.object(sys, "argv", argv):
                main()
            manifest = json.loads((output / "MANIFEST.json").read_text())
            phases = manifest["shards"][0]["phases"]
            self.assertEqual(
                [(phase["learner_role"], phase["requested_battles"]) for phase in phases],
                [("acceptor", 2), ("challenger", 3)],
            )
            self.assertEqual(manifest["failed_shards"], 0)
            self.assertFalse((output / "MANIFEST.json.tmp").exists())
            self.assertTrue((output / "chunk_00000_learner_vs_frozen" / "MANIFEST.json").is_file())

    def test_safe_defaults_make_one_unique_chunk_per_game(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / "pool.json"
            schedule = root / "schedule.json"
            output = root / "out"
            pool.write_text(json.dumps({"profiles": {"learner": LOCAL, "frozen": LOCAL}}))
            schedule.write_text(json.dumps({"learner": "learner", "opponents": ["frozen"] * 3}))
            with patch.object(sys, "argv", [
                "online_rl_generate.py", "--pool", str(pool), "--schedule", str(schedule),
                "--out-dir", str(output), "--dry-run",
            ]):
                main()
            manifest = json.loads((output / "MANIFEST.json").read_text())
            self.assertEqual([chunk["requested_battles"] for chunk in manifest["chunks"]], [1, 1, 1])
            self.assertEqual(len(list(output.glob("chunk_*/MANIFEST.json"))), 3)

    def test_rejects_ties_and_unknown_outcomes(self):
        for rows in (["TIE"], ["WIN", "UNKNOWN"]):
            with self.subTest(rows=rows), self.assertRaisesRegex(CollectionError, "ties or unknown"):
                outcome_totals(rows)

    def test_completed_chunk_is_resumable_only_with_exact_totals(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "MANIFEST.json"
            chunk = {
                "requested_battles": 2,
                "phases": [{
                    "completed_battles": 2, "learner_wins": 1, "learner_losses": 1,
                    "learner_trajectory_count": 2,
                }],
            }
            path.write_text(json.dumps(chunk))
            self.assertEqual(_completed_chunk(path, 2), chunk)
            chunk["phases"][0]["learner_trajectory_count"] = 1
            path.write_text(json.dumps(chunk))
            self.assertIsNone(_completed_chunk(path, 2))

    def test_battle_ledger_preserves_csv_rows_and_is_immutable(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "results.csv"
            csv_path.write_text("battle,opponent,format,outcome\nb1,g3,randbats,WIN\n")
            chunks = [{
                "chunk_index": 0,
                "opponent": "g3",
                "phases": [{
                    "phase": "learner_acceptor", "learner_role": "acceptor",
                    "completed_battles": 1, "learner_result_csv": "results.csv",
                }],
            }]
            metadata = write_battle_ledger(root, chunks)
            row = json.loads((root / metadata["path"]).read_text())
            self.assertEqual(row["result"]["battle"], "b1")
            csv_path.write_text("battle,opponent,format,outcome\nb2,g3,randbats,LOSS\n")
            with self.assertRaisesRegex(CollectionError, "immutable battle ledger changed"):
                write_battle_ledger(root, chunks)


if __name__ == "__main__":
    unittest.main()
