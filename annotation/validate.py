from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .rlm_annotator import load_pool


def iter_annotation_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(root.glob("*.json"))


def validate_annotation(annotation: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    decisions = annotation.get("decisions")
    if not isinstance(decisions, dict):
        return ["missing decisions object"]
    for turn, decision in decisions.items():
        posterior = decision.get("posterior", {})
        if not isinstance(posterior, dict):
            errors.append(f"turn {turn}: posterior is not object")
            continue
        for slot, entries in posterior.items():
            if not isinstance(entries, list):
                errors.append(f"turn {turn} {slot}: posterior is not list")
                continue
            if not entries:
                continue
            probability = sum(float(entry.get("probability", 0.0)) for entry in entries if isinstance(entry, dict))
            if abs(probability - 1.0) > 0.05:
                errors.append(f"turn {turn} {slot}: posterior probability sums to {probability:.3f}")
        value = decision.get("V_rlm", 0.0)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            errors.append(f"turn {turn}: V_rlm is not numeric")
            continue
        if not -1.0 <= value_float <= 1.0:
            errors.append(f"turn {turn}: V_rlm outside [-1, 1]")
    return errors


def ground_truth_missing_rate(annotation: dict[str, Any]) -> tuple[int, int]:
    missing = 0
    checked = 0
    for decision in annotation.get("decisions", {}).values():
        truth = decision.get("ground_truth") or {}
        posterior = decision.get("posterior") or {}
        for slot, true_set in truth.items():
            checked += 1
            entries = posterior.get(slot, [])
            true_index = true_set.get("set_index") if isinstance(true_set, dict) else true_set
            if not any(entry.get("set_index") == true_index for entry in entries if isinstance(entry, dict)):
                missing += 1
    return missing, checked


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Phase 0 annotations")
    parser.add_argument("--annotations", default="data/annotations")
    parser.add_argument("--pool", default="data/gen9_random_pool.json")
    parser.add_argument("--fail-on-errors", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _pool = load_pool(args.pool)
    errors: dict[str, list[str]] = {}
    missing = 0
    checked = 0
    paths = iter_annotation_paths(args.annotations)
    for path in paths:
        annotation = json.loads(path.read_text())
        path_errors = validate_annotation(annotation)
        if path_errors:
            errors[str(path)] = path_errors
        path_missing, path_checked = ground_truth_missing_rate(annotation)
        missing += path_missing
        checked += path_checked
    report = {
        "files": len(paths),
        "files_with_errors": len(errors),
        "errors": errors,
        "ground_truth_checked": checked,
        "ground_truth_missing": missing,
        "ground_truth_missing_rate": (missing / checked) if checked else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_errors and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
