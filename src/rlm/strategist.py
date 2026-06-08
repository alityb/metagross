from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .prompts import root_iteration_prompt
from .repl_env import RLMConfig, RecursiveRepl, extract_python_code, normalize_policy, softmax


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class HeuristicLLMClient:
    """Deterministic fallback that exercises the RLM loop without model weights."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return """
import re, json
lines = BATTLE_LOG.split('\n')
revealed = {}
for line in lines:
    parts = line.split('|')
    if len(parts) >= 4 and parts[1] in ('switch', 'drag') and parts[2].startswith('p2'):
        slot = parts[2].split(':', 1)[0].strip()
        species = parts[3].split(',')[0].strip()
        revealed[slot] = species
POSTERIOR.update({slot: POOL.get(species, []) for slot, species in revealed.items()})
print({'revealed': revealed, 'posterior_slots': list(POSTERIOR)})
"""
        if self.calls == 2:
            return """
move_seen = {}
for line in BATTLE_LOG.split('\n'):
    parts = line.split('|')
    if len(parts) >= 4 and parts[1] == 'move' and parts[2].startswith('p2'):
        slot = parts[2].split(':', 1)[0].strip()
        move_seen.setdefault(slot, set()).add(parts[3].strip())
for slot, moves in move_seen.items():
    candidates = POSTERIOR.get(slot, [])
    if candidates:
        keep = []
        required = {re.sub(r'[^a-z0-9]+', '', m.lower()) for m in moves}
        for candidate in candidates:
            cmoves = {re.sub(r'[^a-z0-9]+', '', str(m).lower()) for m in candidate.get('moves', [])}
            if required.issubset(cmoves):
                keep.append(candidate)
        POSTERIOR[slot] = keep or candidates
print({'move_seen': {k: sorted(v) for k, v in move_seen.items()}})
"""
        return """
base = normalize_policy(BASE_POLICY)
STRATEGIC_PRIOR = base
V_RLM = 0.0
for slot, candidates in list(POSTERIOR.items()):
    if isinstance(candidates, list) and candidates:
        p = 1.0 / len(candidates)
        POSTERIOR[slot] = [dict(candidate, set_index=i, probability=p) for i, candidate in enumerate(candidates)]
FINAL = True
print({'final_prior': STRATEGIC_PRIOR, 'posterior_slots': list(POSTERIOR)})
"""


class HeuristicSubClient:
    def complete(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "score" in lowered or "win" in lowered:
            return json.dumps({"scores": [0.0] * 14, "win_probability": 0.0})
        return json.dumps([])


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1200):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("anthropic is not installed") from exc
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:  # pragma: no cover - networked path
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "\n".join(block.text for block in response.content if getattr(block, "type", None) == "text")


class LocalRLMClient:
    """Production RLM-Qwen3-8B client placeholder.

    The repo does not vendor the alexzhang13/rlm weights. This client makes the
    production path explicit and fails early unless a local model path exists.
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise NotImplementedError(f"RLM-Qwen3-8B weights not found at {self.model_path}")
        raise NotImplementedError("Local RLM-Qwen3-8B INT8 loading is not wired in this scaffold yet")

    def complete(self, prompt: str) -> str:  # pragma: no cover - constructor always raises for now
        raise NotImplementedError


class NullRLMStrategist:
    """Drop-in RLM stub that returns flat priors and V=0.

    Used when:
    - RLM-Qwen3-8B is not yet loaded (Phase 2 PPO, early Phase 3)
    - No-RLM ablation condition
    - CPU-only inference where the 8B model is unavailable

    The MCTS engine already accepts root_prior_rlm=None gracefully (falls back
    to pure PokeNet prior). This stub makes the interface explicit and lets
    agent code stay identical whether or not the real RLM is loaded.
    """

    def assess(
        self,
        *,
        log: str,
        state: Any,
        pool: dict[str, Any],
        base_policy: list[float],
    ) -> "RLMOutput":
        import time
        n = len(base_policy)
        return RLMOutput(
            pi_rlm=[1.0 / n] * n,
            v_rlm=0.0,
            refined_belief={},
            iterations=0,
            sub_queries=0,
            elapsed_ms=0.0,
            observations=["NullRLMStrategist: no RLM loaded"],
        )


@dataclass
class RLMOutput:
    pi_rlm: list[float]
    v_rlm: float
    refined_belief: dict[str, Any]
    iterations: int
    sub_queries: int
    elapsed_ms: float
    observations: list[str]


class RLMStrategist:
    def __init__(
        self,
        root_client: LLMClient | None = None,
        sub_client: LLMClient | None = None,
        config: RLMConfig | None = None,
    ) -> None:
        self.root_client = root_client or HeuristicLLMClient()
        self.sub_client = sub_client or HeuristicSubClient()
        self.config = config or RLMConfig()

    @classmethod
    def from_provider(
        cls,
        provider: str = "heuristic",
        model: str | None = None,
        config: RLMConfig | None = None,
    ) -> "RLMStrategist":
        if provider == "anthropic":
            return cls(root_client=AnthropicClient(model or "claude-sonnet-4-20250514"), config=config)
        if provider == "local":
            if not model:
                raise ValueError("--model must point to local RLM-Qwen3-8B weights when --provider local is used")
            return cls(root_client=LocalRLMClient(model), config=config)
        if provider not in {"heuristic", "null"}:
            raise ValueError(f"Unsupported RLM provider: {provider}")
        return cls(config=config)

    def assess(
        self,
        *,
        log: str,
        state: Any,
        pool: dict[str, Any],
        base_policy: list[float],
    ) -> RLMOutput:
        start = time.monotonic()
        if isinstance(self.root_client, HeuristicLLMClient):
            self.root_client.calls = 0

        def sub_rlm(prompt: str) -> str:
            return self.sub_client.complete(prompt)

        repl = RecursiveRepl(
            battle_log=log,
            pool=pool,
            current_state=state,
            base_policy=base_policy,
            sub_rlm=sub_rlm,
            config=self.config,
        )
        observations: list[str] = []
        metadata = self._metadata(log, state, base_policy)
        iterations = 0
        for iteration in range(self.config.max_iterations):
            if repl.time_exceeded():
                observations.append("time_budget_exceeded_before_iteration")
                break
            prompt = root_iteration_prompt(metadata, repl.snapshot(), observations)
            response = self.root_client.complete(prompt)
            code = extract_python_code(response)
            execution = repl.run_code(code)
            iterations = iteration + 1
            observation = execution.stdout.strip() or "<no stdout>"
            if execution.error:
                observation = f"ERROR {execution.error}\n{observation}"
            observations.append(observation)
            if execution.final or repl.time_exceeded():
                break
        prior, value, posterior = repl.output()
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return RLMOutput(
            pi_rlm=prior,
            v_rlm=value,
            refined_belief=posterior,
            iterations=iterations,
            sub_queries=repl.sub_queries,
            elapsed_ms=elapsed_ms,
            observations=observations,
        )

    @staticmethod
    def _metadata(log: str, state: Any, base_policy: list[float]) -> dict[str, Any]:
        turn = None
        for match in re.finditer(r"\|turn\|(\d+)", log):
            turn = int(match.group(1))
        policy = normalize_policy(base_policy)
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in policy)
        return {
            "log_chars": len(log),
            "turn": turn,
            "state_type": type(state).__name__,
            "available_actions": sum(1 for p in policy if p > 0),
            "base_policy_entropy": round(entropy, 4),
        }


def blend_root_prior(pi_net: list[float], pi_rlm: list[float], weight: float = 0.5) -> list[float]:
    net = normalize_policy(pi_net)
    rlm = normalize_policy(pi_rlm)
    logits = []
    for a, b in zip(net, rlm):
        logits.append((1.0 - weight) * math.log(max(a, 1e-12)) + weight * math.log(max(b, 1e-12)))
    return softmax(logits)
