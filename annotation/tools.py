from __future__ import annotations

import subprocess
from pathlib import Path


def READ(path: str, offset: int = 0, length: int = 4096) -> str:
    with open(path, encoding="utf-8") as handle:
        handle.seek(max(0, offset))
        return handle.read(max(0, length))


def GREP(path: str, pattern: str) -> str:
    result = subprocess.run(
        ["grep", "-n", pattern, path],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def BASH(script: str, timeout: int = 60) -> str:
    result = subprocess.run(
        ["python3", "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return result.stdout + result.stderr


def iter_replay_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    suffixes = {".log", ".txt", ".json"}
    return sorted(file for file in root.rglob("*") if file.suffix.lower() in suffixes)
