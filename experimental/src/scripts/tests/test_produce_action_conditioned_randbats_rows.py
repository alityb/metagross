from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from belief.action_conditioned_randbats import load_generator_pool_active_candidates  # noqa: E402
from scripts.produce_action_conditioned_randbats_rows import replay_state_from_public_state, rows_from_replay  # noqa: E402


class ReplayRowProducerTests(unittest.TestCase):
    def test_prefix_candidates_exclude_future_move_and_suffix_sets_label(self):
        fixture = Path(__file__).parent / "fixtures" / "action_conditioned_randbats_replay.json"
        raw = json.loads(fixture.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            pool_path = Path(directory) / "pool.json"
            # Use a temporary pool schema accepted by the production pool loader.
            pool_path.write_text(json.dumps({"candidates": [
                {"candidate_id": "flame-air", "speciesId": "charizard", "level": 80, "moves": ["flamethrower", "airslash"], "ability": "Blaze", "item": "Heavy-Duty Boots", "teraType": "Fire"},
                {"candidate_id": "flame-pulse", "speciesId": "charizard", "level": 80, "moves": ["flamethrower", "dragonpulse"], "ability": "Blaze", "item": "Heavy-Duty Boots", "teraType": "Fire"},
            ]}), encoding="utf-8")
            rows = rows_from_replay(raw, load_generator_pool_active_candidates(pool_path))
        first = rows[0]
        self.assertEqual({candidate["candidate_id"] for candidate in first["active_candidates"]}, {"flame-air", "flame-pulse"})
        self.assertNotIn("airslash", first["public_state"]["protocol_prefix"][-1])
        self.assertEqual(first["label"], "flame-air")
        self.assertNotIn("label", rows[1])
        replay_state = replay_state_from_public_state(first["public_state"])
        self.assertEqual(replay_state.opponent_active_pokemon.had_name, "Charizard")


if __name__ == "__main__":
    unittest.main()
