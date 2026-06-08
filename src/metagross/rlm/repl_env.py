from __future__ import annotations

import contextlib
import io
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional for the REPL wrapper
    np = None  # type: ignore[assignment]


@dataclass
class RLMConfig:
    max_iterations: int = 8
    max_sub_queries: int = 6
    truncate_len: int = 3000
    time_budget_ms: int | None = 500
    sub_model: str = "qwen3-0.6b-int4"


@dataclass
class ReplExecution:
    stdout: str
    error: str | None
    final: bool
    elapsed_ms: float


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(float(value) - max_value) for value in values]
    total = sum(exps)
    if total <= 0 or not math.isfinite(total):
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]


def normalize_policy(values: Any, size: int = 14) -> list[float]:
    if values is None:
        return [1.0 / size] * size
    try:
        policy = [max(0.0, float(value)) for value in list(values)[:size]]
    except (TypeError, ValueError):
        return [1.0 / size] * size
    if len(policy) < size:
        policy.extend([0.0] * (size - len(policy)))
    total = sum(policy)
    if total <= 0 or not math.isfinite(total):
        return [1.0 / size] * size
    return [value / total for value in policy]


def extract_python_code(text: str) -> str:
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    return text.strip()


class RecursiveRepl:
    """Persistent Python environment used by the root RLM loop."""

    def __init__(
        self,
        *,
        battle_log: str,
        pool: dict[str, Any],
        current_state: Any,
        base_policy: list[float],
        sub_rlm: Callable[[str], str],
        config: RLMConfig,
        extra_globals: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self._sub_rlm = sub_rlm
        self.sub_queries = 0
        self.started_at = time.monotonic()
        self.namespace: dict[str, Any] = self._initial_namespace(battle_log, pool, current_state, base_policy)
        if extra_globals:
            self.namespace.update(extra_globals)

    def _initial_namespace(
        self,
        battle_log: str,
        pool: dict[str, Any],
        current_state: Any,
        base_policy: list[float],
    ) -> dict[str, Any]:
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
        namespace: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "BATTLE_LOG": battle_log,
            "POOL": pool,
            "CURRENT_STATE": current_state,
            "BASE_POLICY": normalize_policy(base_policy),
            "POSTERIOR": {},
            "WIN_WEIGHTS": {},
            "STRATEGIC_PRIOR": [],
            "V_RLM": 0.0,
            "SPEED_CONSTRAINTS": {},
            "FINAL": False,
            "Counter": Counter,
            "defaultdict": defaultdict,
            "json": json,
            "math": math,
            "re": re,
            "statistics": statistics,
            "softmax": softmax,
            "normalize_policy": normalize_policy,
            "sub_rlm": self._budgeted_sub_rlm,
        }
        if np is not None:
            namespace["np"] = np
        return namespace

    def _budgeted_sub_rlm(self, prompt: str) -> str:
        if self.sub_queries >= self.config.max_sub_queries:
            return json.dumps({"error": "max_sub_queries_exceeded"})
        self.sub_queries += 1
        return self._sub_rlm(prompt)

    def run_code(self, code: str) -> ReplExecution:
        start = time.monotonic()
        stdout = io.StringIO()
        error: str | None = None
        with contextlib.redirect_stdout(stdout):
            try:
                exec(code, self.namespace, self.namespace)
            except Exception as exc:  # REPL errors are observations, not process failures.
                error = f"{type(exc).__name__}: {exc}"
                print(error)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        out = stdout.getvalue()
        if len(out) > self.config.truncate_len:
            out = out[: self.config.truncate_len] + "\n...<truncated>"
        return ReplExecution(stdout=out, error=error, final=bool(self.namespace.get("FINAL")), elapsed_ms=elapsed_ms)

    def time_exceeded(self) -> bool:
        if self.config.time_budget_ms is None:
            return False
        return (time.monotonic() - self.started_at) * 1000.0 >= self.config.time_budget_ms

    def snapshot(self) -> dict[str, Any]:
        posterior = self.namespace.get("POSTERIOR", {})
        return {
            "posterior_keys": list(posterior.keys()) if isinstance(posterior, dict) else [],
            "win_weight_keys": list(self.namespace.get("WIN_WEIGHTS", {}).keys())
            if isinstance(self.namespace.get("WIN_WEIGHTS"), dict)
            else [],
            "strategic_prior_len": len(self.namespace.get("STRATEGIC_PRIOR") or []),
            "v_rlm": self.namespace.get("V_RLM", 0.0),
            "sub_queries": self.sub_queries,
            "final": bool(self.namespace.get("FINAL")),
        }

    def output(self) -> tuple[list[float], float, dict[str, Any]]:
        prior = normalize_policy(self.namespace.get("STRATEGIC_PRIOR") or self.namespace.get("BASE_POLICY"), size=14)
        try:
            value = float(self.namespace.get("V_RLM", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        value = max(-1.0, min(1.0, value))
        posterior = self.namespace.get("POSTERIOR", {})
        if not isinstance(posterior, dict):
            posterior = {}
        return prior, value, posterior
