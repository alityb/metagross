#!/usr/bin/env python3
"""Attach frozen-r1 candidate-conditioned action likelihoods to Randbats JSONL."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from belief.action_conditioned_randbats import Candidate, CandidateValidationError, load_active_candidates
from belief.action_likelihood_adapter import FrozenR1CandidatePolicyLikelihoodAdapter, make_frozen_r1_adapter
from scripts.benchmark_action_conditioned_randbats import validate_row
from scripts.produce_action_conditioned_randbats_rows import replay_state_from_public_state

FROZEN_R1_RUN_DIR = ROOT / "nets/checkpoints/randbats_full"
FROZEN_R1_RUN_NAME = "randbats_exit_r1"
FROZEN_R1_CHECKPOINT = 5


def _limited_candidates(candidates: Sequence[Candidate], cap: int) -> tuple[Candidate, ...]:
    return tuple(sorted(candidates, key=lambda candidate: (-candidate.prior_weight, candidate.candidate_id))[:cap])


def _attach_row(
    row: Any,
    adapter: FrozenR1CandidatePolicyLikelihoodAdapter,
    candidate_cap: int,
    state_builder: Callable[[Mapping[str, Any]], Any],
) -> tuple[dict[str, Any], bool]:
    if not isinstance(row, Mapping):
        raise CandidateValidationError("row must be an object")
    if not isinstance(row.get("public_state"), Mapping):
        raise CandidateValidationError("row missing public_state object")
    candidates = load_active_candidates(row.get("active_candidates"))
    selected = _limited_candidates(candidates, candidate_cap)
    capped = len(selected) != len(candidates)
    label = row.get("label")
    if label is not None and label not in {candidate.candidate_id for candidate in selected}:
        raise CandidateValidationError("label would be excluded by candidate cap")

    # Keep source values intact; replay_state is transient adapter input only.
    public_state = dict(row["public_state"])
    public_state["replay_state"] = state_builder(public_state)
    likelihoods = adapter.action_likelihoods(public_state, selected, row.get("observed_action"))
    attached = dict(row)
    if capped:
        selected_ids = {candidate.candidate_id for candidate in selected}
        attached["active_candidates"] = [
            candidate for candidate in row["active_candidates"] if candidate["candidate_id"] in selected_ids
        ]
        # The benchmark consumes candidates in any order; make cap output stable.
        attached["active_candidates"].sort(key=lambda candidate: (-candidate["prior_weight"], candidate["candidate_id"]))
    attached["action_likelihoods"] = dict(likelihoods)
    validate_row(attached)
    return attached, capped


def attach(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    adapter: FrozenR1CandidatePolicyLikelihoodAdapter,
    *,
    batch_row_cap: int,
    max_rows: int | None = None,
    state_builder: Callable[[Mapping[str, Any]], Any] = replay_state_from_public_state,
) -> dict[str, Any]:
    """Process input JSONL, retaining only rows with real, validated likelihoods."""
    if batch_row_cap < 1:
        raise ValueError("batch_row_cap must be positive")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    report: dict[str, Any] = {"input_rows": 0, "output_rows": 0, "rejected_rows": 0, "capped_rows": 0, "row_results": []}
    output_rows: list[dict[str, Any]] = []
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read input: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if max_rows is not None and report["input_rows"] >= max_rows:
            break
        report["input_rows"] += 1
        try:
            if not line.strip():
                raise CandidateValidationError("blank JSONL row")
            row = json.loads(line)
            attached, capped = _attach_row(row, adapter, batch_row_cap, state_builder)
        except (json.JSONDecodeError, CandidateValidationError, ValueError, TypeError, KeyError) as exc:
            report["rejected_rows"] += 1
            report["row_results"].append({"line": line_number, "status": "rejected", "error": str(exc)})
            continue
        output_rows.append(attached)
        report["output_rows"] += 1
        if capped:
            report["capped_rows"] += 1
            report["row_results"].append({"line": line_number, "status": "capped"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
    report["output"] = str(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if not output_rows:
        raise ValueError("no likelihood rows output")
    return report


def load_frozen_r1_adapter(local_run_dir: Path, local_run_name: str, checkpoint: int, local_base_model: str) -> FrozenR1CandidatePolicyLikelihoodAdapter:
    """Load r1 directly; this script never starts a PriorServer or network service."""
    os.environ.setdefault("METAMON_CACHE_DIR", str(ROOT / "external" / "metamon_cache"))
    os.environ.setdefault("WANDB_MODE", "disabled")
    # Match prior_server.py: MPS Inductor cannot compile r1's full transformer.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import metamon.rl.pretrained as pretrained

    try:
        base_model = getattr(pretrained, local_base_model)
    except AttributeError as exc:
        raise ValueError(f"unknown local base model: {local_base_model}") from exc
    model = pretrained.LocalFinetunedModel(
        base_model=base_model,
        amago_ckpt_dir=str(local_run_dir),
        model_name=local_run_name,
        default_checkpoint=checkpoint,
    )
    experiment = model.initialize_agent(checkpoint=checkpoint, log=False)
    agent = experiment.policy
    agent.eval()
    return make_frozen_r1_adapter(agent, model.observation_space, next(agent.parameters()).device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL from produce_action_conditioned_randbats_rows.py")
    parser.add_argument("output", type=Path, help="JSONL consumable by benchmark_action_conditioned_randbats.py")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--local-run-dir", type=Path, default=FROZEN_R1_RUN_DIR)
    parser.add_argument("--local-run-name", default=FROZEN_R1_RUN_NAME)
    parser.add_argument("--checkpoint", type=int, default=FROZEN_R1_CHECKPOINT)
    parser.add_argument("--local-base-model", default="Kakuna")
    parser.add_argument("--batch-row-cap", type=int, default=256, help="maximum candidates evaluated per row")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    try:
        adapter = load_frozen_r1_adapter(args.local_run_dir, args.local_run_name, args.checkpoint, args.local_base_model)
        print(json.dumps(attach(args.input, args.output, args.report, adapter, batch_row_cap=args.batch_row_cap, max_rows=args.max_rows), sort_keys=True))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
