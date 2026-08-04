from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from srcs.metagross import ladder_supervisor


class LadderSupervisorTest(unittest.TestCase):
    def test_requires_distinct_accounts_and_bounded_blocks(self):
        args = ladder_supervisor.parse_args(
            ["--g3-username", "g3bot", "--g4-username", "g4bot"]
        )
        self.assertEqual(args.block_games, 25)
        self.assertEqual(args.cycles, 0)
        for extra in (
            ["--block-games", "0"],
            ["--block-games", "101"],
            ["--g3-username", "same", "--g4-username", "same"],
        ):
            base = ["--g3-username", "g3bot", "--g4-username", "g4bot"]
            if extra[0] == "--g3-username":
                base = []
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ladder_supervisor.parse_args([*base, *extra])

    def test_g3_only_does_not_require_g4_account(self):
        args = ladder_supervisor.parse_args(
            ["--g3-username", "g3bot", "--g3-only"]
        )
        self.assertEqual(
            ladder_supervisor.configured_profiles(args),
            [("g3", "g3bot")],
        )

    def test_g4_account_required_by_default(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ladder_supervisor.parse_args(["--g3-username", "g3bot"])

    def test_r1_only_does_not_require_candidate_accounts(self):
        args = ladder_supervisor.parse_args(["--r1-username", "r1bot"])
        self.assertEqual(
            ladder_supervisor.configured_profiles(args),
            [("r1", "r1bot")],
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ladder_supervisor.parse_args(
                ["--r1-username", "r1bot", "--g3-username", "g3bot", "--g3-only"]
            )

    def test_launcher_command_contains_no_password(self):
        command = ladder_supervisor.launcher_command("g3", "bot", 25, Path("runs"), 30, 5, 8978)
        self.assertIn("--confirm-candidate-continuation", command)
        self.assertEqual(command[command.index("--search-parallelism") + 1], "5")
        self.assertEqual(command[command.index("--port") + 1], "8978")
        self.assertNotIn("password", " ".join(command).lower())

    def test_r1_launcher_command_omits_candidate_acknowledgement(self):
        command = ladder_supervisor.launcher_command("r1", "bot", 25, Path("runs"), 30)
        self.assertNotIn("--confirm-candidate-continuation", command)
        self.assertNotIn("password", " ".join(command).lower())

    def test_fatal_log_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prior.log").write_text("PRIOR_SERVER ready\n", encoding="utf-8")
            (root / "client.log").write_text("INFO completed\n", encoding="utf-8")
            self.assertIsNone(ladder_supervisor.find_fatal_log_line(root))
            (root / "client.log").write_text("ERROR invalid choice\n", encoding="utf-8")
            self.assertIn("client.log:1", ladder_supervisor.find_fatal_log_line(root))


if __name__ == "__main__":
    unittest.main()
