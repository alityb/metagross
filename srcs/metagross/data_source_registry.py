"""Fail-closed validation for the frozen Metagross data-source registry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "metagross-data-source-registry/v1"
TRAINABLE_STATUSES = {
    "approved",
    "approved_after_audit",
    "approved_noncommercial_research",
}
NONTRAINING_STATUSES = {
    "development_only",
    "provenance_only",
    "quarantined_pending_license",
    "evaluation_only",
}


def load_registry(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        registry = json.load(handle)
    validate_registry(registry)
    return registry


def _require_strings(row: Mapping[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list) or not value or any(not isinstance(v, str) for v in value):
        raise ValueError(f"{key} must be a nonempty string list")
    return value


def validate_registry(registry: Mapping[str, Any]) -> None:
    if registry.get("schema") != SCHEMA:
        raise ValueError("unsupported data-source registry schema")
    if registry.get("identity", {}).get("split_unit") != "battle":
        raise ValueError("all splits must be battle-grouped")
    if registry.get("identity", {}).get("derived_rows_inherit_split") is not True:
        raise ValueError("derived rows must inherit their source battle split")

    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("registry must contain sources")
    by_id: dict[str, Mapping[str, Any]] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("source rows must be objects")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id or source_id in by_id:
            raise ValueError("source ids must be unique nonempty strings")
        status = source.get("status")
        if status not in TRAINABLE_STATUSES | NONTRAINING_STATUSES:
            raise ValueError(f"unknown status for {source_id}: {status}")
        _require_strings(source, "formats")
        _require_strings(source, "uses")
        _require_strings(source, "forbidden_uses")
        if str(source.get("location", "")).startswith("hf://") and not source.get("revision"):
            raise ValueError(f"remote source {source_id} is not revision-pinned")
        for artifact in source.get("selected_artifacts", []):
            if not isinstance(artifact, Mapping):
                raise ValueError(f"selected artifacts for {source_id} must be objects")
            sha256 = artifact.get("sha256")
            size = artifact.get("size_bytes")
            if not isinstance(sha256, str) or len(sha256) != 64:
                raise ValueError(f"selected artifact for {source_id} lacks a SHA-256")
            if not isinstance(size, int) or size <= 0:
                raise ValueError(f"selected artifact for {source_id} lacks a positive size")
        by_id[source_id] = source

    stages = registry.get("training_stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("registry must contain training stages")
    for stage in stages:
        rows = stage.get("sources") if isinstance(stage, Mapping) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("training stages must contain sources")
        total = 0.0
        seen: set[str] = set()
        for row in rows:
            source_id = row.get("id")
            if source_id not in by_id or source_id in seen:
                raise ValueError(f"invalid or duplicate stage source: {source_id}")
            seen.add(source_id)
            source = by_id[source_id]
            if source["status"] not in TRAINABLE_STATUSES:
                raise ValueError(f"non-trainable source {source_id} appears in training stage")
            weight = row.get("weight")
            if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"invalid weight for {source_id}")
            total += float(weight)
            filters = _require_strings(row, "format_filter")
            source_formats = set(source["formats"])
            if not set(filters).issubset(source_formats):
                raise ValueError(f"stage requests unavailable format from {source_id}")
            objectives = _require_strings(row, "objectives")
            forbidden = set(source["forbidden_uses"])
            if forbidden.intersection(objectives):
                raise ValueError(f"stage requests forbidden objective from {source_id}")
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"stage weights must sum to 1.0, got {total}")

    confirmation = registry.get("splits", {}).get("fresh_confirmation", {})
    if confirmation.get("may_select_checkpoints") is not False:
        raise ValueError("fresh confirmation data may not select checkpoints")
    h2h = registry.get("splits", {}).get("h2h_only", {})
    if h2h.get("may_select_checkpoints") is not False:
        raise ValueError("H2H data may not become training/checkpoint-selection examples")


def summarize_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for source in registry["sources"]:
        statuses[source["status"]] = statuses.get(source["status"], 0) + 1
    return {
        "schema": registry["schema"],
        "target_format": registry["target_format"],
        "sources": len(registry["sources"]),
        "training_stages": [stage["id"] for stage in registry["training_stages"]],
        "statuses": statuses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_registry(load_registry(args.registry)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
