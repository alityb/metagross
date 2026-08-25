"""Pinned, opt-in downloader for Metamon artifacts approved by the registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from srcs.metagross.data_source_registry import TRAINABLE_STATUSES, load_registry


DEFAULT_REGISTRY = Path("experimental/configs/data_sources_v1.json")


def describe_source(registry: dict[str, Any], source_id: str) -> dict[str, Any]:
    source = next((row for row in registry["sources"] if row["id"] == source_id), None)
    if source is None:
        raise ValueError(f"unknown source: {source_id}")
    if source["status"] not in TRAINABLE_STATUSES:
        raise ValueError(f"source is not approved for training: {source_id}")
    if not source.get("selected_artifacts"):
        raise ValueError(f"source has no selected downloadable artifacts: {source_id}")
    return source


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(source: dict[str, Any], cache_dir: Path) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("huggingface_hub is required; use .venv-metamon") from exc

    repo_id = source["location"].split("@", 1)[0].removeprefix("hf://datasets/")
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for artifact in source["selected_artifacts"]:
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=source["revision"],
                filename=artifact["path"],
                local_dir=cache_dir,
            )
        )
        actual = sha256_path(path)
        if actual != artifact["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {artifact['path']}")
        manifest.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": actual})
    (cache_dir / "METAGROSS_SOURCE_MANIFEST.json").write_text(
        json.dumps({"source": source, "files": manifest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--source", default="metamon_human_v6")
    parser.add_argument("--cache-dir", type=Path, default=Path("external/metamon_cache/datasets"))
    parser.add_argument("--download", action="store_true", help="perform the large download")
    args = parser.parse_args()

    source = describe_source(load_registry(args.registry), args.source)
    total = sum(row["size_bytes"] for row in source["selected_artifacts"])
    if not args.download:
        print(json.dumps({
            "source": source["id"],
            "status": source["status"],
            "revision": source["revision"],
            "download_required": True,
            "download_size_bytes": total,
            "artifacts": source["selected_artifacts"],
            "next": "rerun with --download after confirming disk and CC-BY-NC-4.0 use",
        }, indent=2, sort_keys=True))
        return
    print(json.dumps(fetch(source, args.cache_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
