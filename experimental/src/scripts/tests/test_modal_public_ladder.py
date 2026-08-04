from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experimental.src.scripts import modal_public_ladder


class ModalPublicLadderTest(unittest.TestCase):
    def test_launcher_command_preserves_profile_safety_flags(self):
        root = Path("/models")
        output = Path("/runs")
        r1 = modal_public_ladder._launcher_command("r1", "r1bot", 25, root, output)
        g3 = modal_public_ladder._launcher_command("g3", "g3bot", 25, root, output)
        canary = modal_public_ladder._launcher_command("g3", "g3bot", 3, root, output)
        self.assertNotIn("--confirm-candidate-continuation", r1)
        self.assertIn("--confirm-candidate-continuation", g3)
        self.assertIn("--confirm-g3-canary", canary)
        for command in (r1, g3, canary):
            self.assertNotIn("password", " ".join(command).lower())
            self.assertIn("--checkpoint-root", command)
            self.assertEqual(
                command[command.index("--search-parallelism") + 1],
                "8",
            )
            self.assertEqual(command[command.index("--search-threads") + 1], "1")
            self.assertEqual(
                command[command.index("--foul-play-python") + 1],
                "/workspace/.venv-fp-priors/bin/python",
            )

    def test_image_uses_websockets_with_proxy_support(self):
        source = Path(modal_public_ladder.__file__).read_text(encoding="utf-8")
        self.assertIn("websockets==15.0.1", source)
        environment = modal_public_ladder._proxy_environment("http://proxy")
        self.assertEqual(environment["HTTPS_PROXY"], "http://proxy")
        self.assertEqual(environment["wss_proxy"], "http://proxy")
        self.assertEqual(environment["NO_PROXY"], "127.0.0.1,localhost")

    def test_sticky_proxy_url_adds_us_session_without_exposing_password(self):
        url = modal_public_ladder._sticky_proxy_url(
            "http://proxyuser:password@residential.byteful.com:8000", "metagrossr1"
        )
        self.assertEqual(
            url,
            "http://proxyuser_c_us_s_metagrossr1_ttl_24h:password@"
            "residential.byteful.com:8000",
        )

    def test_block_proxy_sessions_are_stable_and_distinct(self):
        first = modal_public_ladder._block_proxy_session("experiment", "r1", 0)
        self.assertEqual(
            first, modal_public_ladder._block_proxy_session("experiment", "r1", 0)
        )
        self.assertNotEqual(first, modal_public_ladder._block_proxy_session("experiment", "r1", 1))
        self.assertNotEqual(first, modal_public_ladder._block_proxy_session("experiment", "g3", 0))

    def test_campaign_proxy_uses_static_profile_url_without_rewriting(self):
        environment = {
            "METAGROSS_PROXY_URL": "http://fallback:pass@residential.byteful.com:8000",
            "METAGROSS_R1_PROXY_URL": "http://user:pass@98.159.236.28:61234",
        }
        self.assertEqual(
            modal_public_ladder._campaign_proxy_url(environment, "experiment", "r1", 0),
            environment["METAGROSS_R1_PROXY_URL"],
        )
        self.assertIn(
            "_s_",
            modal_public_ladder._campaign_proxy_url(environment, "experiment", "g3", 0),
        )

    def test_campaign_can_force_rotating_proxy(self):
        environment = {
            "METAGROSS_PROXY_URL": "http://fallback:pass@residential.byteful.com:8000",
            "METAGROSS_R1_PROXY_URL": "http://user:pass@98.159.236.28:61234",
        }
        url = modal_public_ladder._campaign_proxy_url(
            environment,
            "experiment",
            "r1",
            0,
            use_profile_proxy=False,
        )
        self.assertIn("residential.byteful.com", url)
        self.assertIn("_s_", url)

    def test_games_to_request_stops_at_exact_rating_target(self):
        self.assertEqual(modal_public_ladder._rated_games({"w": 381, "l": 194}), 575)
        self.assertEqual(
            modal_public_ladder._games_to_request({"w": 381, "l": 194}, 600, 25),
            25,
        )
        self.assertEqual(
            modal_public_ladder._games_to_request({"w": 390, "l": 198}, 600, 25),
            12,
        )
        self.assertEqual(
            modal_public_ladder._games_to_request({"w": 400, "l": 200}, 600, 25),
            0,
        )
        self.assertEqual(modal_public_ladder._rated_games({"w": None, "l": None}), 0)

    def test_checkpoint_path_matches_production_layout(self):
        path = modal_public_ladder._checkpoint_path(Path("/models"), "r1")
        self.assertEqual(
            path,
            Path("/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"),
        )

    def test_created_run_dir_requires_exactly_one_new_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = set()
            (root / "run").mkdir()
            self.assertEqual(
                modal_public_ladder._created_run_dir(root, before),
                root / "run",
            )

    def test_identifiers_are_restricted(self):
        self.assertEqual(
            modal_public_ladder._validate_identifier("paired-r1.g3_001", "id"),
            "paired-r1.g3_001",
        )
        with self.assertRaises(ValueError):
            modal_public_ladder._validate_identifier("../escape", "id")

    def test_artifact_inventory_records_size_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "nested" / "rows.jsonl").write_bytes(b"row\n")
            inventory = modal_public_ladder._artifact_inventory(root)
            self.assertEqual(inventory["nested/rows.jsonl"]["bytes"], 4)
            self.assertEqual(
                inventory["nested/rows.jsonl"]["sha256"],
                "83ad05a6ffdb5c97fb81a8501561e30cc3458bed5a83525e931acb0f8486a393",
            )


if __name__ == "__main__":
    unittest.main()
