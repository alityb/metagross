from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.mcts_policy_distillation import (
    ActionMappingError,
    add_distillation_loss,
    build_sidecar,
    build_trajectory_index,
    explicit_policy_target,
    foul_play_action_to_index,
    remapped_explicit_policy_target,
    visit_distribution_to_target,
)


class Move:
    def __init__(self, name):
        self.name = name


class Pokemon:
    def __init__(self, name, moves=()):
        self.name = name
        self.moves = {move.name: move for move in moves}


class State:
    def __init__(self, *, can_tera=True, forced_switch=False):
        self.player_active_pokemon = Pokemon("Pikachu", [Move("Thunderbolt"), Move("Volt Switch")])
        self.available_switches = [Pokemon("Zapdos"), Pokemon("Amoonguss")]
        self.can_tera = can_tera
        self.forced_switch = forced_switch


class MCTSPolicyDistillationTests(unittest.TestCase):
    def test_maps_moves_tera_and_switches_in_metamon_order(self):
        state = State()
        self.assertEqual(foul_play_action_to_index("thunderbolt", state), 0)
        self.assertEqual(foul_play_action_to_index("voltswitch-tera", state), 10)
        self.assertEqual(foul_play_action_to_index("switch zapdos", state), 5)

    def test_normalizes_visit_mass(self):
        target = visit_distribution_to_target(
            {"thunderbolt": 2, "voltswitch-tera": 6, "switch zapdos": 2}, State()
        )
        self.assertAlmostEqual(sum(target), 1.0)
        self.assertEqual(target[0], 0.2)
        self.assertEqual(target[10], 0.6)
        self.assertEqual(target[5], 0.2)

    def test_rejects_unknown_and_illegal_actions(self):
        with self.assertRaises(ActionMappingError):
            foul_play_action_to_index("surf", State())
        with self.assertRaises(ActionMappingError):
            foul_play_action_to_index("thunderbolt-tera", State(can_tera=False))
        with self.assertRaises(ActionMappingError):
            foul_play_action_to_index("thunderbolt", State(forced_switch=True))
        with self.assertRaises(ActionMappingError):
            foul_play_action_to_index("switch missingno", State())

    def test_disabled_auxiliary_loss_is_an_exact_noop(self):
        import torch

        total = torch.tensor(3.0, requires_grad=True)
        probs = torch.tensor([[[[0.5, 0.5]]]])
        targets = torch.tensor([[[[1.0, 0.0]]]])
        masks = {
            "valid": torch.ones((1, 1, 1, 1), dtype=torch.bool),
            "illegal_actions": torch.zeros((1, 1, 1, 2), dtype=torch.bool),
        }
        self.assertIs(add_distillation_loss(total, probs, targets, masks, 0.0), total)

    def test_explicit_capture_target_is_legal_and_normalized(self):
        target = explicit_policy_target(
            {
                "canonical_selected_action_index": 0,
                "mcts_visit_target_13": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
            },
            State(),
        )
        self.assertEqual(target[0], 0.2)
        self.assertEqual(target[9], 0.8)
        with self.assertRaises(ActionMappingError):
            explicit_policy_target(
                {"canonical_selected_action_index": 0, "mcts_visit_target_13": [0.0] * 13}, State()
            )

    def test_remaps_capture_strings_when_foul_play_slots_differ(self):
        state = State()
        target = remapped_explicit_policy_target(
            {
                # Foul Play omitted Amoonguss, so it placed Zapdos in slot 4.
                "canonical_selected_action_index": 4,
                "mcts_visit_target_13": [0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 8,
                "selected_action": "switch zapdos",
                "mcts_visits": {"switch zapdos": 2.0, "thunderbolt": 1.0},
            },
            state,
            5,
        )
        self.assertEqual(target[5], 2 / 3)
        self.assertEqual(target[0], 1 / 3)

    def test_remap_rejects_unmappable_positive_mass_or_selected_mismatch(self):
        record = {
            "canonical_selected_action_index": 4,
            "mcts_visit_target_13": [0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 8,
            "selected_action": "switch zapdos",
            "mcts_visits": {"switch missingno": 1.0},
        }
        with self.assertRaises(ActionMappingError):
            remapped_explicit_policy_target(record, State(), 5)
        record["mcts_visits"] = {"switch zapdos": 1.0}
        with self.assertRaises(ActionMappingError):
            remapped_explicit_policy_target(record, State(), 4)

    def test_remap_ignores_unmappable_zero_mass_actions(self):
        target = remapped_explicit_policy_target(
            {
                "canonical_selected_action_index": 4,
                "mcts_visit_target_13": [0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 8,
                "selected_action": "switch zapdos",
                "mcts_visits": {"switch zapdos": 1.0, "switch missingno": 0.0},
            },
            State(),
            5,
        )
        self.assertEqual(target[5], 1.0)

    def test_explicit_capture_and_parser_identity_build_a_sidecar(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "parsed"
            trajectory_dir = root / "gen9randombattle"
            trajectory_dir.mkdir(parents=True)
            trajectory = trajectory_dir / "battle-smoke_learner_Unrated_learner_vs_opponent.json"
            trajectory.write_text(json.dumps({"states": [{}, {}], "actions": [0, 0]}))
            identity = Path(temporary) / "trajectory_identity.jsonl"
            self.assertEqual(build_trajectory_index(root, identity), {"trajectories": 1})
            decisions = Path(temporary) / "decisions.jsonl"
            decisions.write_text(
                json.dumps(
                    {
                        "record_type": "decision",
                        "battle_tag": "battle-smoke",
                        "username": "learner",
                        "mcts_schema_version": 2,
                        "mcts_decision_seq": 0,
                        "canonical_selected_action_index": 0,
                        "mcts_visit_target_13": [1.0] + [0.0] * 12,
                    }
                )
                + "\n"
            )
            output = Path(temporary) / "targets.jsonl"
            with patch("train.mcts_policy_distillation._load_states", return_value=[State()]):
                result = build_sidecar([decisions], root, output, identity)
            self.assertEqual(result["accepted"], 1, result)
            self.assertEqual(json.loads(output.read_text())["timestep"], 0)

    def test_explicit_target_failure_rejects_the_whole_learner_trajectory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "parsed"
            trajectory_dir = root / "gen9randombattle"
            trajectory_dir.mkdir(parents=True)
            trajectory = trajectory_dir / "battle-smoke_learner_Unrated_learner_vs_opponent.json"
            trajectory.write_text(json.dumps({"states": [{}, {}, {}], "actions": [0, 0, 0]}))
            identity = Path(temporary) / "trajectory_identity.jsonl"
            build_trajectory_index(root, identity)
            decisions = Path(temporary) / "decisions.jsonl"
            decisions.write_text(
                "".join(
                    json.dumps(
                        {
                            "record_type": "decision",
                            "battle_tag": "battle-smoke",
                            "username": "learner",
                            "mcts_schema_version": 2,
                            "mcts_decision_seq": timestep,
                            "canonical_selected_action_index": 0,
                            "mcts_visit_target_13": target,
                        }
                    )
                    + "\n"
                    for timestep, target in enumerate(([1.0] + [0.0] * 12, [0.0] * 13))
                )
            )
            output = Path(temporary) / "targets.jsonl"
            with patch("train.mcts_policy_distillation._load_states", return_value=[State(), State()]):
                result = build_sidecar([decisions], root, output, identity)

            self.assertEqual(result["accepted"], 0, result)
            self.assertEqual(result["rejected"], 2, result)
            self.assertEqual(result["rejected_povs"][0]["reason"], "explicit_illegal_or_unmappable_target")
            self.assertEqual(output.read_text(), "")

    def test_explicit_target_fallback_remaps_the_whole_learner_trajectory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "parsed"
            trajectory_dir = root / "gen9randombattle"
            trajectory_dir.mkdir(parents=True)
            trajectory = trajectory_dir / "battle-smoke_learner_Unrated_learner_vs_opponent.json"
            trajectory.write_text(json.dumps({"states": [{}, {}], "actions": [5, 0]}))
            identity = Path(temporary) / "trajectory_identity.jsonl"
            build_trajectory_index(root, identity)
            decisions = Path(temporary) / "decisions.jsonl"
            decisions.write_text(
                json.dumps(
                    {
                        "record_type": "decision",
                        "battle_tag": "battle-smoke",
                        "username": "learner",
                        "mcts_schema_version": 2,
                        "mcts_decision_seq": 0,
                        "canonical_selected_action_index": 4,
                        "mcts_visit_target_13": [0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 8,
                        "selected_action": "switch zapdos",
                        "mcts_visits": {"switch zapdos": 1.0},
                    }
                )
                + "\n"
            )
            output = Path(temporary) / "targets.jsonl"
            with patch("train.mcts_policy_distillation._load_states", return_value=[State()]):
                result = build_sidecar([decisions], root, output, identity)

            self.assertEqual(result["accepted"], 1, result)
            self.assertEqual(result["remapped_replays"], 1, result)
            self.assertEqual(result["remapped_records"], 1, result)
            self.assertEqual(json.loads(output.read_text())["target"][5], 1.0)


if __name__ == "__main__":
    unittest.main()
