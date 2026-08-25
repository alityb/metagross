"""Late-game prior temperature flattening (mechanism-synthesis prediction 2).

Entropy-matching schedule calibrated offline on the 699 paired causal/
stateless decisions of `r1_causal_vs_stateless_screen_20260814` (frozen in
the Belief-v1 temperature preregistration): per turn bucket, the temperature
that equates mean causal-prior entropy with the stateless profile.

Applied at serving time to the player root prior only, gated by
METAGROSS_PRIOR_TEMP_SCHEDULE (JSON mapping); absent env = byte-identical
behavior. Fail-open: any malformed schedule or prior leaves priors
untouched.
"""

from __future__ import annotations

import json
import os

CALIBRATED_SCHEDULE = {"0": 1.02, "10": 1.03, "20": 1.134, "30": 1.689}


def _schedule() -> dict[int, float] | None:
    raw = os.environ.get("METAGROSS_PRIOR_TEMP_SCHEDULE")
    if not raw:
        return None
    try:
        mapping = json.loads(raw)
        return {int(k): float(v) for k, v in mapping.items()}
    except Exception:
        return None


def temperature_for_turn(turn: int, schedule: dict[int, float]) -> float:
    tau = 1.0
    for threshold in sorted(schedule):
        if turn >= threshold:
            tau = schedule[threshold]
    return tau


def flatten_priors(priors: list[tuple[str, float]], turn: int | None,
                   mode: str | None = None):
    """Return temperature-flattened (action, p) pairs, or the input untouched
    when the schedule env is absent, the serving mode is known and not
    causal-history, or anything is unusable."""
    schedule = _schedule()
    if schedule is None or not priors or turn is None:
        return priors
    if mode is not None and mode != "causal-history":
        return priors
    try:
        tau = temperature_for_turn(int(turn), schedule)
        if tau <= 1.0:
            return priors
        raised = [(action, float(p) ** (1.0 / tau)) for action, p in priors]
        total = sum(p for _, p in raised)
        if total <= 0:
            return priors
        return [(action, p / total) for action, p in raised]
    except Exception:
        return priors
