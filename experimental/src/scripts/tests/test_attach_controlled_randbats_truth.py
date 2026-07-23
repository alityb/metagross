from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.attach_controlled_randbats_truth import attach  # noqa: E402


SET = {"speciesId": "charizard", "level": 80, "moves": ["Air Slash", "Flamethrower", "Roost", "Dragon Pulse"], "ability": "Blaze", "item": "Heavy-Duty Boots", "teraType": "Fire", "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 84, "spd": 0, "spe": 172}}


class ControlledTruthTests(unittest.TestCase):
    def _files(self, directory: Path, candidates=None, manifest_captures=1, active_candidates=None):
        replay_dir, manifest_dir = directory / "replays", directory / "manifests"
        replay_dir.mkdir()
        manifest_dir.mkdir()
        raw = {"id": "battle-1", "players": ["Alice", "Bob"], "log": "|player|p1|Alice\n|player|p2|Bob\n|switch|p2a: Charizard|Charizard, L80|100/100\n|turn|1"}
        (replay_dir / "battle-1.json").write_text(json.dumps(raw), encoding="utf-8")
        manifests = []
        for capture in range(manifest_captures):
            manifests.extend([{"capture_id": f"capture-{capture}", "side": "p1", "player": "Alice", "team": []}, {"capture_id": f"capture-{capture}", "side": "p2", "player": "Bob", "team": [SET]}])
        (manifest_dir / "truth.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifests), encoding="utf-8")
        pool = {"candidates": candidates or [SET]}
        pool_path = directory / "pool.json"
        pool_path.write_text(json.dumps(pool), encoding="utf-8")
        row = {"replay_id": "battle-1", "active_candidates": active_candidates if active_candidates is not None else [{"candidate_id": self._candidate_id(pool_path)}], "public_state": {"acting_side": "p2", "protocol_prefix": [["player", "p1", "Alice"], ["player", "p2", "Bob"], ["switch", "p2a: Charizard", "Charizard, L80", "100/100"]]}}
        rows_path = directory / "rows.jsonl"
        rows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return rows_path, replay_dir, manifest_dir, pool_path

    def _candidate_id(self, pool_path: Path) -> str:
        from belief.action_conditioned_randbats import load_generator_pool_active_candidates
        return load_generator_pool_active_candidates(pool_path)[0].candidate_id

    def test_player_pair_join_exact_hash_and_private_data_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = self._files(directory)
            output, report = directory / "out.jsonl", directory / "report.json"
            attach(*paths, output, report)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["label"], self._candidate_id(paths[3]))
            self.assertNotIn("team", row["public_state"])
            self.assertFalse(any("manifest" in key for key in row["public_state"]))

    def test_ambiguous_player_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = self._files(directory, manifest_captures=2)
            with self.assertRaisesRegex(ValueError, "no labeled rows"):
                attach(*paths, directory / "out.jsonl", directory / "report.json")
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["rejection_reasons"]["manifest_capture_pair_not_unique"], 1)

    def test_label_not_in_prefix_candidates_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            other = dict(SET, item="Leftovers")
            pool_path = directory / "placeholder.json"
            pool_path.write_text(json.dumps({"candidates": [SET, other]}), encoding="utf-8")
            from belief.action_conditioned_randbats import load_generator_pool_active_candidates
            other_id = load_generator_pool_active_candidates(pool_path)[1].candidate_id
            paths = self._files(directory, candidates=[SET, other], active_candidates=[{"candidate_id": other_id}])
            with self.assertRaisesRegex(ValueError, "no labeled rows"):
                attach(*paths, directory / "out.jsonl", directory / "report.json")
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["rejection_reasons"]["label_not_in_active_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
