from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experimental.src.scripts.run_cycle10_full_corpus_index import (
    assign_clusters,
    classify_error,
    compact_states,
    deterministic_gzip_write,
    raw_metadata,
    read_gzip_json,
    terminal_provenance,
    unordered_mechanical_team_pair_sha256,
)
from experimental.src.scripts.cycle9_replay_audit import ReplayAuditError


class Cycle10FullCorpusIndexTest(unittest.TestCase):
    def test_raw_seed_metadata_hashes_without_emitting_seeds(self) -> None:
        raw = {"inputlog": "\n".join((
            '>start {"seed":"start-secret"}',
            '>player p1 {"name":"A","seed":"team-a"}',
            '>player p2 {"name":"B","seed":"team-b"}',
        ))}
        first = raw_metadata(raw, "commit")
        reversed_players = {"inputlog": "\n".join((
            '>start {"seed":"start-secret"}',
            '>player p2 {"name":"B","seed":"team-b"}',
            '>player p1 {"name":"A","seed":"team-a"}',
        ))}
        self.assertEqual(
            first["unordered_player_seed_pair_sha256"],
            raw_metadata(reversed_players, "commit")["unordered_player_seed_pair_sha256"],
        )
        self.assertFalse(any("secret" in value or "team-" in value for value in first.values()))

    def test_mechanical_team_pair_ignores_role_identity_but_binds_team(self) -> None:
        def pov(role: str, species: str) -> dict:
            return {"requests": [{"request": {"side": {
                "id": role, "name": "Player " + role,
                "pokemon": [{
                    "ident": f"{role}: {species}", "details": species,
                    "moves": ["tackle"], "ability": "pressure", "item": "leftovers",
                    "condition": "100/100", "active": True,
                }],
            }}}]}
        first = unordered_mechanical_team_pair_sha256(pov("p1", "Zapdos"), pov("p2", "Moltres"), "c")
        swapped = unordered_mechanical_team_pair_sha256(pov("p2", "Moltres"), pov("p1", "Zapdos"), "c")
        changed = unordered_mechanical_team_pair_sha256(pov("p1", "Zapdos"), pov("p2", "Articuno"), "c")
        self.assertEqual(first, swapped)
        self.assertNotEqual(first, changed)

    def test_compact_state_keeps_hashes_not_payloads(self) -> None:
        derived = {"states": [{
            "role": "p1", "request_index": 0, "actionable": True,
            "public_event_index": 12,
            "model_information_fingerprint_sha256": "a" * 64,
            "private_request": {"side": {"id": "p1", "name": "secret"}},
            "public_prefix": ["|turn|1"],
            "legal_actions": ["tackle"], "action_table": {"tackle": 0},
            "action_semantics": {"tackle": "move"},
            "typed_reveal_ledger": {"facts": []},
            "pp_disable_sidecar": {"moves": []},
            "chosen_action": "tackle", "chosen_action_index": 0,
            "chosen_action_semantics": "move",
        }]}
        pov = {"commands": [{
            "preceding_request_index": 0, "input_index": 7,
            "command": "move tackle",
        }]}
        row = compact_states(derived, pov)[0]
        self.assertEqual(row["observed_command"], "move tackle")
        self.assertEqual(row["command_input_index"], 7)
        self.assertNotIn("private_request", row)
        self.assertNotIn("public_prefix", row)
        self.assertNotIn("secret", json.dumps(row))
        for key, value in row.items():
            if key.endswith("_sha256"):
                self.assertEqual(len(value), 64)

    def test_terminal_outcome_is_role_provenance(self) -> None:
        log = "\n".join((
            "|player|p1|Alice|1|1500", "|player|p2|Bob|2|1500", "|win|Bob",
        ))
        terminal = terminal_provenance(log)
        self.assertEqual(terminal["kind"], "win")
        self.assertEqual(terminal["winner_role"], "p2")
        self.assertNotIn("Bob", json.dumps(terminal))

    def test_deterministic_gzip_and_cluster_union(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = []
            for index, public in enumerate(("p", "p", "q")):
                relative = Path("states") / f"{index}.json.gz"
                deterministic_gzip_write(root / relative, {"schema": "x", "states": []})
                results.append({
                    "panel_index": index, "status": "pass", "battle_id": f"b{index}",
                    "commit": "c", "state_count": 2,
                    "canonical_public_sha256": public * 64,
                    "execution_sha256": str(index) * 64,
                    "start_seed_sha256": str(index + 3) * 64,
                    "unordered_player_seed_pair_sha256": str(index + 6) * 64,
                    "unordered_mechanical_team_pair_sha256": str(index + 7) * 64,
                    "relative_index": str(relative),
                })
            report = assign_clusters(results, root)
            self.assertEqual(report["clusters"], 2)
            self.assertEqual(report["duplicate_clusters"], 1)
            self.assertEqual(report["battles_in_duplicate_clusters"], 2)
            self.assertEqual(results[0]["dependency_cluster_id"], results[1]["dependency_cluster_id"])
            self.assertEqual(results[0]["split"], results[1]["split"])
            self.assertEqual(report["cross_split_cluster_leakage"], 0)
            payload = read_gzip_json(root / results[0]["relative_index"])
            self.assertTrue(payload["states_inherit_battle_split"])

    def test_unknown_errors_are_never_silent(self) -> None:
        known, known_detail = classify_error(ReplayAuditError("normalized public replay mismatch"))
        unknown, unknown_detail = classify_error(RuntimeError("unexpected bug"))
        unknown_semantic, _ = classify_error(ReplayAuditError("new unmapped semantic"))
        self.assertEqual(known, "public_replay_mismatch")
        self.assertEqual(unknown, "internal_unclassified:RuntimeError")
        self.assertEqual(unknown_semantic, "unknown_semantic:ReplayAuditError")
        self.assertEqual(len(known_detail), 64)
        self.assertEqual(len(unknown_detail), 64)


if __name__ == "__main__":
    unittest.main()
