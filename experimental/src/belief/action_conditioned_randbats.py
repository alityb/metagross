"""Action-conditioned posterior updates for active Randbats generator candidates.

The policy likelihood boundary deliberately accepts likelihoods from a caller.
Do not substitute a public-only opponent prior here: it is not P(action | set).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


class CandidateValidationError(ValueError):
    """Raised when an active candidate set is not safe to benchmark."""


class CandidatePolicyLikelihoodAdapter(Protocol):
    """Boundary for a future frozen candidate-state Metamon policy adapter.

    Implementations must evaluate the same public state once per candidate and
    return P(observed_action | candidate). Existing public-only priors are not
    valid implementations of this protocol.
    """

    def action_likelihoods(
        self, public_state: Mapping[str, Any], candidates: Sequence["Candidate"], observed_action: str
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class Candidate:
    """One active generator-pool set and its unnormalized generator weight."""

    candidate_id: str
    prior_weight: float = 1.0
    public_data: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Posterior:
    """Normalized prior and action-conditioned posterior for an active set."""

    candidates: tuple[Candidate, ...]
    prior: Mapping[str, float]
    likelihood: Mapping[str, float]
    posterior: Mapping[str, float]
    evidence: float

    def ranking(self, posterior: bool = True) -> list[tuple[str, float]]:
        """Deterministic probability ranking, suitable for top-k and calibration."""
        probabilities = self.posterior if posterior else self.prior
        return sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))

    def probability(self, candidate_id: str, posterior: bool = True) -> float:
        return (self.posterior if posterior else self.prior)[candidate_id]

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable probabilities and rankings for calibration analysis."""
        return {
            "evidence": self.evidence,
            "prior": dict(self.prior),
            "posterior": dict(self.posterior),
            "prior_ranking": self.ranking(posterior=False),
            "posterior_ranking": self.ranking(),
        }


def _positive_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateValidationError(f"{field} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise CandidateValidationError(f"{field} must be positive and finite")
    return value


def _nonnegative_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateValidationError(f"{field} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise CandidateValidationError(f"{field} must be non-negative and finite")
    return value


def _reject_label_keys(value: Any, location: str = "candidate") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if "label" in str(key).lower():
                raise CandidateValidationError(f"{location} contains forbidden label field {key!r}")
            _reject_label_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_label_keys(child, f"{location}[{index}]")


def load_active_candidates(records: Sequence[Mapping[str, Any]]) -> tuple[Candidate, ...]:
    """Load pre-action active candidates without allowing target-label leakage.

    Each record requires ``candidate_id`` and may include positive finite
    ``prior_weight``. All remaining fields are retained as public candidate
    data, but field names containing ``label`` are rejected at every depth.
    """
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise CandidateValidationError("active_candidates must be a non-empty list")
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CandidateValidationError(f"active_candidates[{index}] must be an object")
        _reject_label_keys(record, f"active_candidates[{index}]")
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id.strip() != candidate_id:
            raise CandidateValidationError(f"active_candidates[{index}].candidate_id must be a non-empty string")
        if candidate_id in seen:
            raise CandidateValidationError(f"duplicate candidate_id {candidate_id!r}")
        seen.add(candidate_id)
        prior_weight = _positive_finite(record.get("prior_weight", 1.0), f"candidate {candidate_id} prior_weight")
        public_data = {key: value for key, value in record.items() if key not in {"candidate_id", "prior_weight"}}
        candidates.append(Candidate(candidate_id, prior_weight, public_data))
    return tuple(candidates)


def load_generator_pool_active_candidates(pool_path: Path, active_candidate_ids: set[str] | None = None) -> tuple[Candidate, ...]:
    """Load and frequency-weight candidates from a generator-pool JSON file.

    The loader accepts either ``{"candidates": [...]}`` or Showdown-style
    ``{"teams": [[set, ...], ...]}`` pools. A set's explicit ``candidate_id``
    is preferred; otherwise a stable hash of its public JSON is used. Repeated
    pool entries are aggregated into generator prior weights.
    """
    try:
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateValidationError(f"cannot load generator pool {pool_path}: {exc}") from exc
    if not isinstance(pool, Mapping):
        raise CandidateValidationError("generator pool must be a JSON object")
    raw_candidates = pool.get("candidates")
    if raw_candidates is None:
        teams = pool.get("teams")
        if not isinstance(teams, list):
            raise CandidateValidationError("generator pool requires candidates or teams")
        raw_candidates = [set_ for team in teams if isinstance(team, list) for set_ in team]
    if not isinstance(raw_candidates, list):
        raise CandidateValidationError("generator pool candidates must be a list")

    counts: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, record in enumerate(raw_candidates):
        if not isinstance(record, Mapping):
            raise CandidateValidationError(f"generator candidate {index} must be an object")
        _reject_label_keys(record, f"generator candidate {index}")
        candidate_id = record.get("candidate_id")
        if candidate_id is None:
            # Generator metadata such as ``role`` is not a distinct active set.
            # Collapse repeated draws to the attributes that affect legal moves
            # and the policy observation, so a true manifest set has one label.
            raw_evs = record.get("evs")
            identity = {
                "species": _norm(record.get("speciesId") or record.get("species")),
                "level": record.get("level"),
                "moves": sorted(_norm(move) for move in record.get("moves", [])),
                "ability": _norm(record.get("ability")),
                "item": _norm(record.get("item")),
                "tera_type": _norm(record.get("teraType")),
                "evs": {
                    _norm(stat): value
                    for stat, value in sorted(raw_evs.items())
                } if isinstance(raw_evs, Mapping) else None,
            }
            canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            candidate_id = hashlib.sha256(canonical.encode("ascii")).hexdigest()
        if not isinstance(candidate_id, str) or not candidate_id:
            raise CandidateValidationError(f"generator candidate {index} has invalid candidate_id")
        if active_candidate_ids is not None and candidate_id not in active_candidate_ids:
            continue
        count, _ = counts.get(candidate_id, (0, record))
        counts[candidate_id] = (count + 1, record)
    records = []
    for candidate_id, (count, record) in counts.items():
        public_record = dict(record)
        public_record.pop("candidate_id", None)
        public_record.pop("prior_weight", None)
        records.append({"candidate_id": candidate_id, "prior_weight": count, **public_record})
    return load_active_candidates(records)


def update_from_action(candidates: Sequence[Candidate], likelihoods: Mapping[str, Any]) -> Posterior:
    """Compute P(candidate | action), allowing zero for impossible actions."""
    if not candidates:
        raise CandidateValidationError("at least one active candidate is required")
    ids = {candidate.candidate_id for candidate in candidates}
    if set(likelihoods) != ids:
        missing, extra = ids - set(likelihoods), set(likelihoods) - ids
        raise CandidateValidationError(f"likelihood ids must exactly match candidates; missing={sorted(missing)}, extra={sorted(extra)}")
    total_weight = sum(_positive_finite(candidate.prior_weight, f"candidate {candidate.candidate_id} prior_weight") for candidate in candidates)
    prior = {candidate.candidate_id: candidate.prior_weight / total_weight for candidate in candidates}
    likelihood = {candidate_id: _nonnegative_finite(value, f"likelihood for {candidate_id}") for candidate_id, value in likelihoods.items()}
    evidence = sum(prior[candidate_id] * likelihood[candidate_id] for candidate_id in ids)
    if not math.isfinite(evidence) or evidence <= 0.0:
        raise CandidateValidationError("action evidence must be positive and finite")
    posterior = {candidate_id: prior[candidate_id] * likelihood[candidate_id] / evidence for candidate_id in ids}
    return Posterior(tuple(candidates), prior, likelihood, posterior, evidence)
