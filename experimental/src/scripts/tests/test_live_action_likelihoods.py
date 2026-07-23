from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.prior_server import action_likelihoods_from_public_payload  # noqa: E402


class _Adapter:
    def __init__(self):
        self.calls = []

    def action_likelihoods(self, public_state, candidates, action):
        self.calls.append((public_state, candidates, action))
        return {candidate.candidate_id: 0.0 if action == "move surf-tera" else 0.4 for candidate in candidates}


class _Server:
    def __init__(self):
        self.action_likelihood_adapter = _Adapter()


class LiveActionLikelihoodTests(unittest.TestCase):
    def payload(self, **extra):
        payload = {
            "protocol_prefix": [["start"], ["switch", "p1a: A", "A, L80"], ["switch", "p2a: B", "B, L80"]],
            "active_candidates": [{"candidate_id": "set", "species": "b", "level": 80, "moves": ["surf"], "ability": "torrent", "item": "leftovers", "teraType": "water"}],
            "observed_action": "move surf-tera",
            "public_metadata": {"format": "gen9randombattle", "observer_side": "p1"},
        }
        payload.update(extra)
        return payload

    def test_tera_is_canonical_and_reconstruction_path_is_used(self):
        server = _Server()
        replay = types.SimpleNamespace(
            force_switch=False, active_pokemon=object(), available_switches=[object()],
            can_tera=True, opponent_active_pokemon=object(),
        )
        with patch("scripts.produce_action_conditioned_randbats_rows.replay_state_from_public_state", return_value=replay) as rebuild, \
             patch("scripts.produce_action_conditioned_randbats_rows._pokemon_facts", return_value={}), \
             patch("scripts.produce_action_conditioned_randbats_rows._matches", return_value=True):
            result = action_likelihoods_from_public_payload(server, self.payload())
        self.assertTrue(result["available"])
        self.assertEqual(server.action_likelihood_adapter.calls[0][2], "move surf-tera")
        self.assertEqual(server.action_likelihood_adapter.calls[0][0]["public_opponent_remaining"], 2)
        rebuild.assert_called_once()
        self.assertEqual(result["likelihoods"], {"set": 0.0})

    def test_request_and_malformed_payload_are_unavailable(self):
        server = _Server()
        self.assertFalse(action_likelihoods_from_public_payload(server, self.payload(protocol_prefix=[["request", "private"]]))["available"])
        self.assertFalse(action_likelihoods_from_public_payload(server, {"protocol_prefix": []})["available"])

    def test_forced_switch_is_excluded(self):
        server = _Server()
        replay = types.SimpleNamespace(force_switch=True)
        with patch("scripts.produce_action_conditioned_randbats_rows.replay_state_from_public_state", return_value=replay):
            result = action_likelihoods_from_public_payload(server, self.payload())
        self.assertFalse(result["available"])
        self.assertEqual(server.action_likelihood_adapter.calls, [])


if __name__ == "__main__":
    unittest.main()
