from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


class PokeEnginePool:
    """Manages per-generation poke-engine builds.

    Python can only import one installed `poke_engine` extension at a time in
    this scaffold. The pool records available wheels and returns the currently
    installed module, falling back to gen9 when a specific build is unavailable.
    """

    def __init__(self, wheels_dir: str = "wheels"):
        self.wheels_dir = Path(wheels_dir)
        self._engines: dict[str, Any] = {}
        self._available_wheels: dict[str, Path] = {}
        self._load_available(self.wheels_dir)

    def _load_available(self, wheels_dir: Path) -> None:
        for path in wheels_dir.glob("poke_engine_gen*.whl") if wheels_dir.exists() else []:
            gen = self._extract_gen(path.name)
            self._available_wheels[gen] = path
        try:
            engine = importlib.import_module("poke_engine")
        except Exception as exc:  # pragma: no cover - optional Rust extension
            LOGGER.warning("poke_engine is not importable: %s", exc)
            return
        self._engines["gen9"] = engine
        for gen in self._available_wheels:
            self._engines.setdefault(gen, engine)

    def get_engine(self, format_str: str) -> Any:
        gen = self._extract_gen(format_str)
        engine = self._engines.get(gen) or self._engines.get("gen9")
        if engine is None:
            raise RuntimeError("No poke_engine build is importable")
        return engine

    @staticmethod
    def _extract_gen(format_str: str) -> str:
        match = re.search(r"gen([1-9])", str(format_str))
        return f"gen{match.group(1)}" if match else "gen9"
