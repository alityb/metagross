"""Package historical online-RL anchors inside their source Modal Volume.

Run this against the source workspace only. It performs no training and writes
two deterministic archives back to the existing `metagross-exit-r2` Volume.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import modal


APP = modal.App("metagross-pack-online-rl-anchors")
app = APP
VOLUME = modal.Volume.from_name("metagross-exit-r2")
FORMAT = "gen9randombattle"
IMAGE = modal.Image.debian_slim(python_version="3.11").apt_install("pigz")


def _package_shard(
    source: Path,
    destination: Path,
    dataset_name: str,
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    paths = sorted((source / FORMAT).rglob("*.json.lz4"))
    if not paths:
        raise ValueError(f"no trajectories under {source / FORMAT}")
    selected = paths[shard_index::shard_count]
    if not selected:
        raise ValueError("empty trajectory shard")
    file_list = Path(f"/tmp/{dataset_name}-{shard_index:03d}.files")
    file_list.write_bytes(
        b"\0".join(str(path.relative_to(source)).encode() for path in selected) + b"\0"
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    subprocess.run(
        [
            "tar",
            "--create",
            f"--file={temporary}",
            "--use-compress-program=pigz -1 -p 4",
            f"--transform=s,^,online_rl/{dataset_name}/,",
            "--directory",
            str(source),
            "--null",
            f"--files-from={file_list}",
        ],
        check=True,
    )
    temporary.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "dataset": dataset_name,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "trajectories": len(selected),
        "bytes": destination.stat().st_size,
        "sha256": digest.hexdigest(),
        "path": str(destination),
    }


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=3 * 3600, cpu=4)
def pack_shard(dataset: str, shard_index: int, shard_count: int) -> dict[str, object]:
    sources = {
        "legacy": Path("/data/selfplay_parsed_indexed"),
        "human": Path("/data/parsed_replays"),
    }
    if dataset not in sources:
        raise ValueError(f"unsupported dataset: {dataset}")
    output = Path("/data/online_rl_transfer")
    output.mkdir(parents=True, exist_ok=True)
    report = _package_shard(
        sources[dataset],
        output / f"{dataset}-{shard_index:03d}-of-{shard_count:03d}.tar.gz",
        dataset,
        shard_index,
        shard_count,
    )
    VOLUME.commit()
    return report


@APP.function(timeout=300)
def launch_shards(shard_count: int = 8) -> dict[str, object]:
    calls = [
        pack_shard.spawn(dataset, shard_index, shard_count)
        for dataset in ("legacy", "human")
        for shard_index in range(shard_count)
    ]
    return {"shard_count": shard_count, "function_call_ids": [call.object_id for call in calls]}


@APP.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(launch_shards.remote(), indent=2, sort_keys=True))
