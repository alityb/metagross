#!/usr/bin/env python3
"""Temporarily recreate the frozen production Python MCTS ABI for a Linux build.

The checked source tree intentionally contains newer seeded-MCTS experiments,
while the admitted production controller is pinned to the earlier ABI without
the public ``seed`` argument.  The local macOS capture runner already enforces
that split with independent source and binary hashes.  Modal uses this helper
to build the equivalent Linux binary, then restores the source byte-for-byte.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path("/workspace/srcs/vendor/poke-engine")
BACKUP = Path("/tmp/metagross-production-capture-backup")


RUST_REPLACEMENTS = (
    (
        "use poke_engine::mcts::{perform_mcts_with_optional_seed, MctsResult, MctsSideResult};",
        "use poke_engine::mcts::{perform_mcts, MctsResult, MctsSideResult};",
    ),
    (
        "#[pyo3(signature = (py_state, duration_ms, iterations, threads, s1_priors=None, s2_priors=None, c_puct=2.0, seed=None))]",
        "#[pyo3(signature = (py_state, duration_ms, iterations, threads, s1_priors=None, s2_priors=None, c_puct=2.0))]",
    ),
    (
        "    c_puct: f32,\n    seed: Option<u64>,\n) -> PyResult<PyMctsResult> {",
        "    c_puct: f32,\n) -> PyResult<PyMctsResult> {",
    ),
    (
        "    if threads > 1 && seed.is_some() {\n        return Err(pyo3::exceptions::PyValueError::new_err(\n            \"seeded MCTS requires threads=1\",\n        ));\n    }\n",
        "",
    ),
    (
        "        perform_mcts_with_optional_seed(\n",
        "        perform_mcts(\n",
    ),
    (
        "            c_puct,\n            seed,\n        )",
        "            c_puct,\n        )",
    ),
)

PYTHON_REPLACEMENTS = (
    (
        "    c_puct: float = 2.0,\n    seed: int | None = None,\n) -> MctsResult:",
        "    c_puct: float = 2.0,\n) -> MctsResult:",
    ),
    (
        "            c_puct=c_puct,\n            seed=seed,\n        )",
        "            c_puct=c_puct,\n        )",
    ),
)

TARGETS = {
    Path("poke-engine-py/src/lib.rs"): RUST_REPLACEMENTS,
    Path("python/poke_engine/__init__.py"): PYTHON_REPLACEMENTS,
}


def apply() -> None:
    if BACKUP.exists():
        raise RuntimeError(f"backup already exists: {BACKUP}")
    BACKUP.mkdir()
    for relative, replacements in TARGETS.items():
        target = ROOT / relative
        original = target.read_bytes()
        text = original.decode("utf-8")
        for newer, production in replacements:
            count = text.count(newer)
            if count != 1:
                raise RuntimeError(
                    f"expected exactly one source pattern, found {count}: {newer[:80]!r}"
                )
            text = text.replace(newer, production)
        backup = BACKUP / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(original)
        target.write_text(text, encoding="utf-8")


def restore() -> None:
    if not BACKUP.is_dir():
        raise RuntimeError(f"backup does not exist: {BACKUP}")
    for relative in TARGETS:
        (ROOT / relative).write_bytes((BACKUP / relative).read_bytes())
    import shutil

    shutil.rmtree(BACKUP)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("apply", "restore"))
    args = parser.parse_args()
    if args.mode == "apply":
        apply()
    else:
        restore()


if __name__ == "__main__":
    main()
