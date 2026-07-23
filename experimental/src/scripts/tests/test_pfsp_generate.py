from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.pfsp_generate import main


class PFSPGenerateTests(unittest.TestCase):
    def test_records_failed_shard_and_continues_remaining_jobs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / "pool.json"
            pool.write_text(
                json.dumps(
                    {
                        "format": "gen9randombattle",
                        "profiles": {
                            "learner": {"agent": "learner-agent"},
                            "opponent": {"agent": "opponent-agent"},
                        },
                    }
                )
            )
            schedule = root / "schedule.json"
            schedule.write_text(json.dumps({"learner": "learner", "opponents": ["opponent"] * 5}))
            out_dir = root / "out"
            calls: list[str] = []

            def run(cmd, check, env):
                output = cmd[cmd.index("--json-out") + 1]
                calls.append(output)
                if output.endswith("shard_01/result.json"):
                    raise RuntimeError("isolated eval failure")

            argv = [
                "pfsp_generate.py",
                "--pool", str(pool),
                "--schedule", str(schedule),
                "--out-dir", str(out_dir),
                "--shards-per-matchup", "3",
            ]
            with patch("scripts.pfsp_generate.subprocess.run", side_effect=run), patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(SystemExit, "1 PFSP shard"):
                    main()

            self.assertEqual(len(calls), 3)
            manifest = json.loads((out_dir / "MANIFEST.json").read_text())
            self.assertEqual([(row["opponent"], row["shard_index"]) for row in manifest["matchups"]], [
                ("opponent", 0), ("opponent", 1), ("opponent", 2),
            ])
            failed = manifest["matchups"][1]
            self.assertEqual(failed["requested_games"], 2)
            self.assertEqual(failed["paired_games"], 2)
            self.assertEqual(failed["out"], str(out_dir / "learner_vs_opponent" / "shard_01"))
            self.assertEqual(failed["error"], "isolated eval failure")
            self.assertNotIn("error", manifest["matchups"][0])
            self.assertNotIn("error", manifest["matchups"][2])

    def test_assigns_unique_prior_namespace_per_concurrent_shard(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / "pool.json"
            pool.write_text(json.dumps({
                "format": "gen9randombattle",
                "profiles": {"learner": {"agent": "learner-agent"}, "opponent": {"agent": "opponent-agent"}},
            }))
            schedule = root / "schedule.json"
            schedule.write_text(json.dumps({"learner": "learner", "opponents": ["opponent"] * 4}))
            seen: list[str] = []

            def run(cmd, check, env):
                seen.append(env["METAGROSS_PRIOR_NAMESPACE"])

            argv = [
                "pfsp_generate.py", "--pool", str(pool), "--schedule", str(schedule),
                "--out-dir", str(root / "out"), "--shards-per-matchup", "2", "--workers", "2",
            ]
            with patch("scripts.pfsp_generate.subprocess.run", side_effect=run), patch.dict(
                os.environ, {"METAGROSS_PRIOR_NAMESPACE": "w1"}, clear=False
            ), patch.object(sys, "argv", argv):
                main()

            self.assertEqual(sorted(seen), ["w1-opponent-00", "w1-opponent-01"])
            manifest = json.loads((root / "out" / "MANIFEST.json").read_text())
            self.assertEqual(
                [row["prior_namespace"] for row in manifest["matchups"]],
                ["w1-opponent-00", "w1-opponent-01"],
            )


if __name__ == "__main__":
    unittest.main()
