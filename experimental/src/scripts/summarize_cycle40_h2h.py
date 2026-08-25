#!/usr/bin/env python3
"""Apply the frozen fail-closed H2H scorer and label Cycle40's report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experimental.src.scripts.summarize_cycle33_h2h import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run.resolve(), args.manifest.resolve())
    report["schema"] = "metagross-cycle40-integrated-h2h-result/v1"
    report["cycle39_target_aware_pp_admitted"] = True
    report["cycle38c_temporal_request_lineage_admitted"] = True
    report["cycle34_causal_disable_admitted"] = True
    report["cycle32_authenticated_identity_admitted"] = True
    report["prior_h2h_or_smoke_identity_reused"] = False
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
