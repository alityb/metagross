from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from belief.action_conditioned_randbats import CandidateValidationError  # noqa: E402
from scripts.attach_action_conditioned_likelihoods import attach, load_frozen_r1_adapter  # noqa: E402


class FakeAdapter:
    def action_likelihoods(self, public_state, candidates, observed_action):
        if public_state["replay_state"] != "state":
            raise CandidateValidationError("missing reconstructed state")
        return {candidate.candidate_id: 0.5 for candidate in candidates}


def valid_row(candidates=None, **extra):
    row = {
        "active_candidates": candidates or [
            {"candidate_id": "a", "prior_weight": 2},
            {"candidate_id": "b", "prior_weight": 1},
        ],
        "legal_actions": ["move surf"],
        "observed_action": "move surf",
        "public_state": {"protocol_prefix": []},
        "metadata": {"unchanged": True},
    }
    row.update(extra)
    return row


class AttachLikelihoodTests(unittest.TestCase):
    def run_attach(self, rows, cap=10):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        input_path, output_path, report_path = root / "input.jsonl", root / "output.jsonl", root / "report.json"
        input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        report = attach(input_path, output_path, report_path, FakeAdapter(), batch_row_cap=cap, state_builder=lambda _: "state")
        return report, [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()], json.loads(report_path.read_text(encoding="utf-8"))

    def test_retains_row_and_only_adds_likelihoods(self):
        source = valid_row()
        report, output, _ = self.run_attach([source])
        self.assertEqual(report["output_rows"], 1)
        self.assertEqual({key: value for key, value in output[0].items() if key != "action_likelihoods"}, source)
        self.assertEqual(output[0]["action_likelihoods"], {"a": 0.5, "b": 0.5})

    def test_rejects_invalid_candidate_and_reports_error(self):
        bad = valid_row([{"candidate_id": "a", "prior_weight": 0}])
        with self.assertRaisesRegex(ValueError, "no likelihood rows output"):
            self.run_attach([bad])
        # Re-run with a valid row so the report can be inspected despite the required nonzero condition.
        report, _, persisted = self.run_attach([bad, valid_row()])
        self.assertEqual(report["rejected_rows"], 1)
        self.assertEqual(persisted["row_results"][0]["status"], "rejected")

    def test_cap_is_deterministic_and_protects_label(self):
        candidates = [
            {"candidate_id": "z", "prior_weight": 2},
            {"candidate_id": "a", "prior_weight": 2},
            {"candidate_id": "b", "prior_weight": 1},
        ]
        report, output, persisted = self.run_attach([valid_row(candidates)], cap=2)
        self.assertEqual([candidate["candidate_id"] for candidate in output[0]["active_candidates"]], ["a", "z"])
        self.assertEqual(report["capped_rows"], 1)
        self.assertEqual(persisted["row_results"], [{"line": 1, "status": "capped"}])
        protected = valid_row(candidates, label="b")
        with self.assertRaisesRegex(ValueError, "no likelihood rows output"):
            self.run_attach([protected], cap=2)

    def test_loader_uses_local_finetuned_model_without_checkpoint_files(self):
        captured = {}

        class Model:
            observation_space = object()

            def __init__(self, **kwargs):
                captured.update(kwargs)

            def initialize_agent(self, checkpoint, log):
                class Agent:
                    def eval(self):
                        return None

                    def parameters(self):
                        return iter([types.SimpleNamespace(device="cpu")])

                return types.SimpleNamespace(policy=Agent())

        metamon = types.ModuleType("metamon")
        metamon.__path__ = []
        rl = types.ModuleType("metamon.rl")
        rl.__path__ = []
        pretrained = types.ModuleType("metamon.rl.pretrained")
        pretrained.Kakuna = "base"
        pretrained.LocalFinetunedModel = Model
        metamon.rl = rl
        rl.pretrained = pretrained
        with patch.dict(sys.modules, {
            "metamon": metamon,
            "metamon.rl": rl,
            "metamon.rl.pretrained": pretrained,
        }), patch("scripts.attach_action_conditioned_likelihoods.make_frozen_r1_adapter", return_value="adapter") as factory:
            self.assertEqual(load_frozen_r1_adapter(Path("checkpoint-dir"), "run", 5, "Kakuna"), "adapter")
        self.assertEqual(captured, {
            "base_model": "base", "amago_ckpt_dir": "checkpoint-dir", "model_name": "run", "default_checkpoint": 5,
        })
        factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
