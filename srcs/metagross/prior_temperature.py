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


_ACTIVE_LOGGED = {"done": False}


def _port_gate_passes() -> bool:
    """Per-arm gate: when METAGROSS_PRIOR_TEMP_PORTS is set, apply only in
    clients whose own prior-server port is listed. This is how one global env
    differentiates arms in a screen (the 2026-08-25 invalid run flattened
    BOTH arms because mode was never passed at the production install site)."""
    import re
    ports_raw = os.environ.get("METAGROSS_PRIOR_TEMP_PORTS")
    if not ports_raw:
        return True
    ports = {p.strip() for p in ports_raw.split(",") if p.strip()}
    if not ports or not all(p.isdigit() for p in ports):
        raise RuntimeError(
            f"METAGROSS_PRIOR_TEMP_PORTS is set but invalid: {ports_raw!r}")
    m = re.search(r":(\d+)", os.environ.get("METAGROSS_PRIOR_SERVER") or "")
    return bool(m and m.group(1) in ports)


def flatten_priors(priors: list[tuple[str, float]], turn: int | None,
                   mode: str | None = None):
    """Return temperature-flattened (action, p) pairs, or the input untouched
    when the schedule env is absent, the arm/mode gates exclude this client,
    or the inputs are unusable. Set-but-malformed env RAISES (fail-closed);
    activation is observable via a one-time ACTIVE line (the 2026-08-19 and
    2026-08-25 lessons: silent fail-open screens measure the wrong thing)."""
    raw = os.environ.get("METAGROSS_PRIOR_TEMP_SCHEDULE")
    if not raw:
        return priors
    schedule = _schedule()
    if schedule is None:
        raise RuntimeError(
            f"METAGROSS_PRIOR_TEMP_SCHEDULE is set but unparseable: {raw!r}")
    if not priors or turn is None:
        return priors
    if mode is not None and mode != "causal-history":
        return priors
    if not _port_gate_passes():
        return priors
    tau = temperature_for_turn(int(turn), schedule)
    if not _ACTIVE_LOGGED["done"]:
        import sys
        print(f"prior temperature schedule ACTIVE (tau={tau} at turn {turn})",
              file=sys.stderr, flush=True)
        print(f"prior temperature schedule ACTIVE (tau={tau} at turn {turn})",
              flush=True)
        _ACTIVE_LOGGED["done"] = True
    if tau <= 1.0:
        return priors
    raised = [(action, float(p) ** (1.0 / tau)) for action, p in priors]
    total = sum(p for _, p in raised)
    if total <= 0:
        return priors
    return [(action, p / total) for action, p in raised]
