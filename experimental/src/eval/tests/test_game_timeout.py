"""Every wait in the per-game path must be bounded.

Regression tests for the 2026-08-26 league wedge (pair 17 / seed 2026082600):
eval.run sat alive for 18 hours with --game-timeout-seconds 900 never
producing a failed game. These tests pin the repaired behavior: a client that
ignores SIGTERM, a client that can never be reaped (lost child-watcher
wakeup), and a game whose collection path never returns all end as a bounded
FAILED GAME, not a wedged run.
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval import run as eval_run  # noqa: E402


class FakeUnreapableProc:
    """Duck-typed asyncio subprocess whose wait() future never resolves —
    the observed child-watcher pathology: the child is dead or unkillable,
    signals do nothing, and proc.wait() never wakes."""

    def __init__(self) -> None:
        self.pid = 999999
        self.returncode = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class BoundedReapTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_term = eval_run.REAP_TERM_GRACE_SECONDS
        self._old_kill = eval_run.REAP_KILL_GRACE_SECONDS
        eval_run.REAP_TERM_GRACE_SECONDS = 0.2
        eval_run.REAP_KILL_GRACE_SECONDS = 0.2

    def tearDown(self) -> None:
        eval_run.REAP_TERM_GRACE_SECONDS = self._old_term
        eval_run.REAP_KILL_GRACE_SECONDS = self._old_kill

    def test_terminate_process_returns_even_when_child_is_unreapable(self):
        proc = FakeUnreapableProc()
        started = time.monotonic()
        asyncio.run(asyncio.wait_for(eval_run.terminate_process(proc), timeout=10))
        self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(proc.terminated)
        self.assertTrue(proc.killed)

    def test_wait_for_foul_play_raises_within_bound_for_unreapable_child(self):
        proc = FakeUnreapableProc()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "client.log"
            log_path.write_text("no winner here\n")
            log_file = log_path.open("a")

            async def scenario() -> None:
                await asyncio.wait_for(
                    eval_run.wait_for_foul_play(proc, log_path, log_file, 1),
                    timeout=10,
                )

            started = time.monotonic()
            with self.assertRaises(eval_run.FoulPlayError):
                asyncio.run(scenario())
            self.assertLess(time.monotonic() - started, 8)

    def test_wait_for_foul_play_kill_escalation_beats_sigterm_ignorer(self):
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                log_path = Path(tmp) / "client.log"
                log_file = log_path.open("w")
                proc = await asyncio.create_subprocess_exec(
                    "/bin/sh", "-c", 'trap "" TERM; sleep 60',
                    stdout=log_file,
                    stderr=asyncio.subprocess.STDOUT,
                )
                started = time.monotonic()
                with self.assertRaises(eval_run.FoulPlayError):
                    await eval_run.wait_for_foul_play(proc, log_path, log_file, 1)
                self.assertLess(time.monotonic() - started, 8)
                self.assertIsNotNone(proc.returncode)

        asyncio.run(asyncio.wait_for(scenario(), timeout=20))


class HardDeadlineTest(unittest.TestCase):
    def _make_args(self, timeout_seconds: int = 1) -> argparse.Namespace:
        return argparse.Namespace(
            agent_a="production_r1_search_first",
            agent_b="production_r1_search_first",
            game_timeout_seconds=timeout_seconds,
            fail_fast=False,
        )

    def test_hard_timeout_includes_direct_r1_readiness_slack(self):
        args = self._make_args(900)
        base = eval_run.game_hard_timeout_seconds(args)
        args.agent_b = "direct_r1"
        self.assertGreater(eval_run.game_hard_timeout_seconds(args), base)

    def test_run_scheduled_game_voids_a_game_whose_collection_never_returns(self):
        old_slack = eval_run.GAME_HARD_TIMEOUT_SLACK_SECONDS
        old_play = eval_run.play_one_game
        eval_run.GAME_HARD_TIMEOUT_SLACK_SECONDS = 0.3

        async def wedged_play_one_game(*_args, **_kwargs):
            await asyncio.Event().wait()

        eval_run.play_one_game = wedged_play_one_game
        try:
            args = self._make_args(0)
            started = time.monotonic()
            result = asyncio.run(
                asyncio.wait_for(
                    eval_run.run_scheduled_game(
                        args, None, 33, "agent_a", "agent_b", Path(".")
                    ),
                    timeout=15,
                )
            )
            self.assertLess(time.monotonic() - started, 10)
            self.assertTrue(result.void)
            self.assertIn("hard deadline", result.error)
            self.assertIsNone(eval_run._GAME_SENTINEL.get())
        finally:
            eval_run.GAME_HARD_TIMEOUT_SLACK_SECONDS = old_slack
            eval_run.play_one_game = old_play

    def test_fail_fast_raises_instead_of_voiding(self):
        old_slack = eval_run.GAME_HARD_TIMEOUT_SLACK_SECONDS
        old_play = eval_run.play_one_game
        eval_run.GAME_HARD_TIMEOUT_SLACK_SECONDS = 0.3

        async def wedged_play_one_game(*_args, **_kwargs):
            await asyncio.Event().wait()

        eval_run.play_one_game = wedged_play_one_game
        try:
            args = self._make_args(0)
            args.fail_fast = True
            with self.assertRaises(eval_run.FoulPlayError):
                asyncio.run(
                    asyncio.wait_for(
                        eval_run.run_scheduled_game(
                            args, None, 33, "agent_a", "agent_b", Path(".")
                        ),
                        timeout=15,
                    )
                )
        finally:
            eval_run.GAME_HARD_TIMEOUT_SLACK_SECONDS = old_slack
            eval_run.play_one_game = old_play


class MirroredPairAgentMatrixTest(unittest.TestCase):
    def _parse(self, tmp: str, agent_a: str, agent_b: str) -> argparse.Namespace:
        return eval_run.parse_args([
            "--mode", "h2h", "--server", "local",
            "--mirrored-pairs", "--fail-fast", "--n-games", "2",
            "--agent-a", agent_a, "--agent-b", agent_b,
            "--json-out", f"{tmp}/result.json",
            "--log-dir", f"{tmp}/logs",
            "--pair-registration-dir", f"{tmp}/registrations",
        ])

    def test_poke_env_opponent_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._parse(tmp, "foul_play", "max_damage")
            self.assertTrue(args.mirrored_pairs)

    def test_two_poke_env_agents_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._parse(tmp, "random", "max_damage")

    def test_direct_r1_against_poke_env_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._parse(tmp, "direct_r1", "max_damage")


class RegistrationConsumptionTest(unittest.TestCase):
    def test_leftover_registrations_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(pair_registration_dir=tmp)
            (Path(tmp) / "lg00x033abcd.json").write_text("{}")
            leftover = eval_run.unconsumed_pair_registrations(
                args, ("lg00x033ABCD", "lg00y033efgh")
            )
            self.assertEqual([path.name for path in leftover], ["lg00x033abcd.json"])
            (Path(tmp) / "lg00x033abcd.json").unlink()
            self.assertEqual(
                eval_run.unconsumed_pair_registrations(
                    args, ("lg00x033ABCD", "lg00y033efgh")
                ),
                [],
            )


class SentinelTest(unittest.TestCase):
    def test_sentinel_kills_registered_clients_when_deadline_breached(self):
        victim = subprocess.Popen(["/bin/sleep", "60"])

        class Registered:
            pid = victim.pid
            returncode = None

        sentinel = eval_run.GameHardDeadline(1, timeout_seconds=0.3, grace_seconds=600)
        sentinel.register(Registered())
        try:
            with sentinel:
                deadline = time.monotonic() + 10
                while victim.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
            self.assertEqual(victim.poll(), -signal.SIGKILL)
        finally:
            if victim.poll() is None:
                victim.kill()
            victim.wait()

    def test_sentinel_disarms_cleanly_when_game_finishes_in_time(self):
        sentinel = eval_run.GameHardDeadline(2, timeout_seconds=30)
        with sentinel:
            pass
        self.assertFalse(sentinel._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
