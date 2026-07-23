from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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

from eval.run import foul_play_env  # noqa: E402


def args() -> SimpleNamespace:
    return SimpleNamespace(
        randbats_belief_pool="/tmp/pool.json",
        randbats_conditional_script="/tmp/conditional.js",
        randbats_conditional_samples=8,
        randbats_conditional_max_teams=100,
        randbats_conditional_max_ms=1000,
        randbats_conditional_timeout_seconds=5,
        format="gen9randombattle",
        prior_server_url="http://localhost:8001",
        cpuct=2.0,
        tauros_kind_model="/tmp/tauros.json",
        tauros_kind_threshold=0.5,
        tauros_kind_min_policy_frac=0.75,
        tauros_kind_allowed_kinds="switch",
        value_shield_margin=0.15,
        value_shield_min_support=0.1,
        value_shield_close_policy_frac=0.75,
        value_shield_log=None,
        learned_value_model=None,
    )


class SelectiveSharedRootAgentTests(unittest.TestCase):
    def test_default_audit_and_agent_isolation(self):
        with patch.dict(os.environ, {}, clear=True):
            selective = foul_play_env(args(), "foul_play_selective_shared_root_opp")
            baseline = foul_play_env(args(), "foul_play_action_belief_root_priors_opp")

        self.assertEqual(selective["METAGROSS_SELECTIVE_SHARED_ROOT_MODE"], "audit")
        self.assertEqual(selective["METAGROSS_SHARED_ROOT_SEARCH"], "1")
        self.assertEqual(selective["METAGROSS_ACTION_CONDITIONED_BELIEF"], "1")
        self.assertIn("METAGROSS_PRIOR_SERVER", selective)
        self.assertNotIn("METAGROSS_SELECTIVE_SHARED_ROOT_MODE", baseline)
        self.assertNotIn("METAGROSS_SHARED_ROOT_SEARCH", baseline)

    def test_override_is_selective_agent_only(self):
        with patch.dict(
            os.environ, {"METAGROSS_SELECTIVE_SHARED_ROOT_MODE": "override"}, clear=True
        ):
            selective = foul_play_env(args(), "foul_play_selective_shared_root_opp")
            shared_only = foul_play_env(args(), "foul_play_shared_root_action_belief_opp")

        self.assertEqual(selective["METAGROSS_SELECTIVE_SHARED_ROOT_MODE"], "override")
        self.assertNotIn("METAGROSS_SELECTIVE_SHARED_ROOT_MODE", shared_only)
        self.assertEqual(shared_only["METAGROSS_SHARED_ROOT_SEARCH"], "1")


if __name__ == "__main__":
    unittest.main()
