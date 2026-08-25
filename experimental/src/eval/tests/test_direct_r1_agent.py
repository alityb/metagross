from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import poke_env.player  # noqa: F401
except ModuleNotFoundError:
    poke_env = ModuleType("poke_env")
    player = ModuleType("poke_env.player")
    ps_client = ModuleType("poke_env.ps_client")
    server_configuration = ModuleType("poke_env.ps_client.server_configuration")
    player.MaxBasePowerPlayer = type("MaxBasePowerPlayer", (), {})
    player.Player = type("Player", (), {})
    player.RandomPlayer = type("RandomPlayer", (), {})
    ps_client.AccountConfiguration = type("AccountConfiguration", (), {})
    server_configuration.ServerConfiguration = type("ServerConfiguration", (), {})
    sys.modules.setdefault("poke_env", poke_env)
    sys.modules.setdefault("poke_env.player", player)
    sys.modules.setdefault("poke_env.ps_client", ps_client)
    sys.modules.setdefault("poke_env.ps_client.server_configuration", server_configuration)

from eval import run  # noqa: E402
from scripts import run_direct_r1_challenge as direct  # noqa: E402


class DirectR1AgentTests(unittest.TestCase):
    def test_frozen_identity_and_checkpoint_mismatch_fail_closed(self):
        self.assertEqual(direct.R1_RUN_NAME, "randbats_exit_r1")
        self.assertEqual(direct.R1_CHECKPOINT, 5)
        self.assertEqual(
            direct.R1_SHA256,
            "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = (
                root
                / direct.R1_RUN_NAME
                / "ckpts"
                / "policy_weights"
                / f"policy_epoch_{direct.R1_CHECKPOINT}.pt"
            )
            path.parent.mkdir(parents=True)
            path.write_bytes(b"wrong checkpoint")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                direct.verify_r1_checkpoint(root)

    def test_native_token_and_deterministic_action_contract(self):
        class Tokenizer:
            def __init__(self):
                self._initial_ids = {"<gen9ou>": 17}

            def __getitem__(self, token):
                return self._initial_ids.get(token, -1)

        experiment = SimpleNamespace(sample_actions_val=True)
        model = SimpleNamespace(tokenizer=Tokenizer())
        direct.configure_direct_experiment(experiment, model)
        self.assertFalse(experiment.sample_actions_val)
        self.assertEqual(model.tokenizer["<gen9randombattle>"], 17)
        self.assertEqual(direct.R1_ACTION_CONTRACT, "masked-legal-argmax-lowest-index-v1")

    def test_metamon_legal_mask_and_forced_switch_contract(self):
        try:
            import numpy as np
            from metamon.interface import UniversalAction
            from metamon.rl.metamon_to_amago import MetamonAMAGOWrapper
        except (ImportError, ValueError) as exc:
            self.skipTest(f"Metamon runtime unavailable: {exc}")

        state = SimpleNamespace(
            forced_switch=True,
            player_active_pokemon=SimpleNamespace(moves={"a": object(), "b": object()}),
            can_tera=True,
            available_switches=[object(), object()],
        )
        self.assertEqual(
            {action.action_idx for action in UniversalAction.maybe_valid_actions(state)},
            {4, 5},
        )
        wrapper = SimpleNamespace(action_space=SimpleNamespace(n=14))
        observation = {}
        MetamonAMAGOWrapper.add_illegal_action_mask_to_obs(
            wrapper, observation, {"legal_actions": [4, 5]}
        )
        expected = np.ones(14, dtype=bool)
        expected[[4, 5]] = False
        np.testing.assert_array_equal(observation["illegal_actions"], expected)

    def test_invalid_action_fallback_is_disabled(self):
        wrappers = SimpleNamespace(PokeEnvWrapper=type("Wrapper", (), {}))
        direct.install_fail_closed_invalid_action(wrappers)
        with self.assertRaisesRegex(SystemExit, "outside the legal mask"):
            wrappers.PokeEnvWrapper().on_invalid_order(None)

    def test_mirrored_direct_r1_vs_root_priors_is_accepted(self):
        base = [
            "--mirrored-pairs",
            "--n-games",
            "2",
            "--agent-a",
            "direct_r1",
            "--agent-b",
            "foul_play_root_priors",
            "--json-out",
            "/tmp/direct-r1.json",
            "--log-dir",
            "/tmp/direct-r1-logs",
            "--pair-registration-dir",
            "/tmp/direct-r1-pairs",
            "--fail-fast",
        ]
        parsed = run.parse_args(base)
        self.assertTrue(parsed.mirrored_pairs)
        self.assertEqual(run.DIRECT_R1_SHA256, direct.R1_SHA256)
        unsupported = list(base)
        unsupported[4] = "random"
        with self.assertRaisesRegex(ValueError, "registration-aware"):
            run.parse_args(unsupported)

    def test_direct_command_is_frozen_and_uses_existing_challenge_harness(self):
        parsed = run.parse_args(["--agent-a", "direct_r1"])
        server = SimpleNamespace(websocket_url="ws://localhost:8000/showdown/websocket")
        command = run.direct_r1_command(
            parsed, server, "DirectA", "TeacherB", "challenger", Path("/tmp/results")
        )
        self.assertIn("run_direct_r1_challenge.py", command[1])
        self.assertEqual(command[command.index("--battle-format") + 1], "gen9randombattle")
        self.assertEqual(command[command.index("--role") + 1], "challenger")
        self.assertNotIn("--checkpoint", command)

    def test_mirrored_pair_dispatches_complete_external_game(self):
        parsed = run.parse_args(["--agent-a", "direct_r1", "--agent-b", "foul_play_root_priors"])
        expected = run.GameResult(
            1, parsed.agent_a, parsed.agent_b, "agent_a", "agent_b", "agent_a", "a", None
        )
        with patch.object(run, "play_external_vs_external", new=AsyncMock(return_value=expected)) as play:
            result = asyncio.run(
                run.play_one_game(
                    parsed,
                    SimpleNamespace(),
                    1,
                    "agent_a",
                    "agent_b",
                    Path("/tmp"),
                )
            )
        self.assertIs(result, expected)
        play.assert_awaited_once()

    def test_direct_acceptor_readiness_waits_for_explicit_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "direct.log"
            log.write_text("loading checkpoint\nMade Challenge Env (acceptor): a vs b\n")
            process = SimpleNamespace(returncode=None)
            asyncio.run(run.wait_for_direct_r1_acceptor_ready(process, log, 1.0))


if __name__ == "__main__":
    unittest.main()
