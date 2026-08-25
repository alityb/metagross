#!/usr/bin/env python3
"""Run the frozen Cycle33-equivalent scorer and relabel the Cycle35 report."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from experimental.src.scripts.summarize_cycle33_h2h import summarize

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run.resolve(), args.manifest.resolve())
    report["schema"] = "metagross-cycle35-h2h-result/v1"
    report["cycle33_results_reused"] = False
    report["cycle34_repair_admitted"] = True
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))

if __name__ == "__main__": main()
