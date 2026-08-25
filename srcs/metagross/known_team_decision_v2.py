"""Frozen Phase-1 headroom gate for the final known-team decision experiment.

This module deliberately contains no calibration gate.  It freezes B0, selects
truth-blind representative and sensitivity roots, evaluates B0 on current-belief
worlds, and measures its action-value regret against repeated known-team MCTS.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from statistics import stdev
from typing import Any, Iterable, Mapping, Sequence

from srcs.metagross import run_foul_play, shadow_replay
from srcs.metagross.history_belief import candidate_fields, candidate_id, compile_public_belief
from srcs.metagross.history_belief_replay import _current_strict_weights
from srcs.metagross.known_team_belief_eval import (
    _candidate_pool_ids,
    _pristine_candidates,
    _reconstruct_view,
    truth_candidate_id,
)
from srcs.metagross.known_team_decision_gate import _truth_battle
from srcs.metagross.public_history import normalize_id


ROOT = Path(__file__).resolve().parents[2]
ENGINE_PYTHON = (
    ROOT
    / "experimental/engine/pe_v3_learned_priors/poke-engine-py/python"
)
SCHEMA = "metagross-known-team-decision-phase1/v2"
MANIFEST_SCHEMA = "metagross-known-team-decision-manifest/v2"
PANEL_SCHEMA = "metagross-known-team-root-panel/v2"
CORPUS_SCHEMA = "metagross-known-team-battle/v2"
MASTER_SEED = "metagross-known-team-decision-v2-development-20260812"
REPRESENTATIVE_BATTLES = 250
SENSITIVITY_BATTLES = 150
EXTENSION_BATTLES = 200
EXTENSION_REPRESENTATIVE_BATTLES = 125
EXTENSION_SENSITIVITY_BATTLES = 75
REQUIRED_REPRESENTATIVE_ROOTS = 200
REQUIRED_SENSITIVITY_ROOTS = 80
BASELINE_WORLD_COUNT = 16
WORLD_ITERATIONS = 20_000
TEACHER_ITERATIONS = 200_000
TEACHER_ESCALATION_ITERATIONS = 1_000_000
TEACHER_REPEATS = 3
MEANINGFUL_HEADROOM = 0.02
TEACHER_HALF_WIDTH_LIMIT = 0.01
T_CRITICAL_DF2_95 = 4.30265272975
OBSERVED_PARTIAL_CANDIDATE = "__observed_partial__"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def seed(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).digest()[:8],
        "big",
    )


def rank(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).hexdigest()


def load_engine():
    engine_path = str(ENGINE_PYTHON)
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    import poke_engine

    if "seed" not in inspect.signature(poke_engine.monte_carlo_tree_search).parameters:
        raise RuntimeError("Phase 1 requires the seeded experimental Gen 9 engine")
    return poke_engine


def _git_output(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE
    ).stdout


def _dependency_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "srcs/metagross/public_history.py",
        ROOT / "srcs/metagross/history_belief.py",
        ROOT / "srcs/metagross/history_belief_replay.py",
        ROOT / "srcs/metagross/known_team_belief_eval.py",
        ROOT / "srcs/metagross/known_team_decision_gate.py",
        ROOT / "srcs/metagross/run_foul_play.py",
        ROOT / "srcs/metagross/shadow_replay.py",
        ROOT / "srcs/metagross/generate_known_team_randbats_v2.cjs",
        ROOT / "srcs/metagross/known_team_decision_v2_phase2.py",
        ROOT / "srcs/metagross/known_team_decision_v2_phase3.py",
        ROOT / "srcs/vendor/foul-play/fp/battle_modifier.py",
        ROOT / "srcs/vendor/foul-play/fp/search/poke_engine_helpers.py",
        ROOT / "srcs/vendor/foul-play/data/pkmn_sets_cache/gen9randombattle.json",
    ]
    return {
        str(path.relative_to(ROOT)): sha256_path(path)
        for path in paths
        if path.is_file()
    }


def build_manifest(corpus_path: Path | None = None) -> dict[str, Any]:
    poke_engine = load_engine()
    native = Path(inspect.getfile(poke_engine.poke_engine)).resolve()
    tracked_diff = _git_output("diff", "--binary", "HEAD")
    status = _git_output("status", "--porcelain=v1", "-z")
    commit = _git_output("rev-parse", "HEAD").decode("ascii").strip()
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "frozen_before_execution",
        "baseline": {
            "id": "B0",
            "belief": "current_strict_foul_play",
            "history_influence": 0.0,
            "world_count": BASELINE_WORLD_COUNT,
            "world_iterations": WORLD_ITERATIONS,
            "threads": 1,
            "request_authoritative_legality": True,
            "tie_break": "descending_mass_then_canonical_action",
        },
        "phase1": {
            "master_seed": MASTER_SEED,
            "representative_battles": REPRESENTATIVE_BATTLES,
            "sensitivity_battles": SENSITIVITY_BATTLES,
            "required_representative_roots": REQUIRED_REPRESENTATIVE_ROOTS,
            "required_sensitivity_roots": REQUIRED_SENSITIVITY_ROOTS,
            "teacher_iterations": TEACHER_ITERATIONS,
            "teacher_escalation_iterations": TEACHER_ESCALATION_ITERATIONS,
            "teacher_repeats": TEACHER_REPEATS,
            "teacher_half_width_limit": TEACHER_HALF_WIDTH_LIMIT,
            "meaningful_headroom": MEANINGFUL_HEADROOM,
            "kill_rules": {
                "representative_meaningful_headroom_wilson_upper_below": 0.05,
                "minimum_stable_meaningful_representative_roots": 10,
            },
        },
        "repository": {
            "commit": commit,
            "tracked_diff_sha256": sha256_bytes(tracked_diff),
            "porcelain_status_sha256": sha256_bytes(status),
        },
        "engine": {
            "python_package": str(Path(inspect.getfile(poke_engine)).resolve()),
            "native_path": str(native),
            "native_sha256": sha256_path(native),
            "seeded": True,
        },
        "dependencies": _dependency_hashes(),
        "existing_known_team_corpora": "development_only",
        "public_ladder_authorized": False,
    }
    if corpus_path is not None:
        manifest["corpus"] = {
            "path": str(corpus_path),
            "sha256": sha256_path(corpus_path),
        }
    unhashed = canonical_json(manifest).encode("ascii")
    manifest["manifest_sha256"] = sha256_bytes(unhashed)
    return manifest


def validate_manifest(manifest: Mapping[str, Any], corpus_path: Path | None = None) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("invalid Phase-1 manifest schema")
    unhashed = dict(manifest)
    claimed = unhashed.pop("manifest_sha256", None)
    if claimed != sha256_bytes(canonical_json(unhashed).encode("ascii")):
        raise ValueError("Phase-1 manifest content hash does not match")
    if corpus_path is not None:
        corpus = manifest.get("corpus") or {}
        if corpus.get("sha256") != sha256_path(corpus_path):
            raise ValueError("Phase-1 corpus differs from the frozen manifest")
    engine = manifest.get("engine") or {}
    native = Path(str(engine.get("native_path", "")))
    if not native.is_file() or engine.get("native_sha256") != sha256_path(native):
        raise ValueError("seeded engine differs from the frozen manifest")
    if manifest.get("dependencies") != _dependency_hashes():
        raise ValueError("experiment dependencies differ from the frozen manifest")


def load_corpus_v2(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_uids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid corpus JSON at line {line_number}") from exc
            battle_id = row.get("battle_id") if isinstance(row, dict) else None
            corpus_uid = row.get("corpus_uid") if isinstance(row, dict) else None
            if (
                row.get("schema") != CORPUS_SCHEMA
                or not isinstance(battle_id, str)
                or not isinstance(corpus_uid, str)
                or len(corpus_uid) != 24
            ):
                raise ValueError(f"invalid v2 corpus row at line {line_number}")
            if battle_id in seen_ids or corpus_uid in seen_uids:
                raise ValueError(f"duplicate v2 corpus identity at line {line_number}")
            seen_ids.add(battle_id)
            seen_uids.add(corpus_uid)
            for side in ("p1", "p2"):
                team = ((row.get("teams") or {}).get(side) or {}).get("sets")
                view = (row.get("views") or {}).get(side)
                if not isinstance(team, list) or len(team) != 6 or not isinstance(view, dict):
                    raise ValueError(f"invalid {side} truth/view at line {line_number}")
                chunks, decisions = view.get("chunks"), view.get("decisions")
                if not isinstance(chunks, list) or not isinstance(decisions, list):
                    raise ValueError(f"invalid {side} protocol at line {line_number}")
                counts = [decision.get("chunk_count") for decision in decisions]
                if (
                    any(not isinstance(value, int) or not 1 <= value <= len(chunks) for value in counts)
                    or counts != sorted(counts)
                    or len(counts) != len(set(counts))
                ):
                    raise ValueError(f"invalid {side} decisions at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("v2 corpus is empty")
    return rows


def partition_corpus(corpus: Sequence[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    initial_count = REPRESENTATIVE_BATTLES + SENSITIVITY_BATTLES
    allowed_counts = {initial_count, initial_count + EXTENSION_BATTLES}
    if len(corpus) not in allowed_counts:
        raise ValueError(
            f"Phase 1 requires exactly {initial_count} battles or its one "
            f"preregistered {EXTENSION_BATTLES}-battle extension"
        )
    by_index = {row.get("battle_index"): row for row in corpus}
    if set(by_index) != set(range(len(corpus))):
        raise ValueError("v2 corpus battle indices must be the contiguous frozen prefix")
    initial = sorted(
        (by_index[index] for index in range(initial_count)),
        key=lambda row: rank("panel", row["corpus_uid"]),
    )
    representative = initial[:REPRESENTATIVE_BATTLES]
    sensitivity = initial[REPRESENTATIVE_BATTLES:]
    if len(corpus) == initial_count + EXTENSION_BATTLES:
        extension = sorted(
            (by_index[index] for index in range(initial_count, len(corpus))),
            key=lambda row: rank("extension-panel", row["corpus_uid"]),
        )
        representative += extension[:EXTENSION_REPRESENTATIVE_BATTLES]
        sensitivity += extension[EXTENSION_REPRESENTATIVE_BATTLES:]
    return representative, sensitivity


def observer_for(row: Mapping[str, Any]) -> tuple[str, str]:
    observer = "p1" if seed("observer", row["corpus_uid"]) % 2 == 0 else "p2"
    return observer, "p2" if observer == "p1" else "p1"


def _normalized_counts(species: str, candidates: Sequence[Any]) -> dict[str, float]:
    rows = [
        (candidate_id(species, candidate_fields(candidate)), candidate.pkmn_set.count)
        for candidate in candidates
    ]
    total = math.fsum(count for _identity, count in rows)
    return {identity: count / total for identity, count in rows} if total > 0 else {}


def root_beliefs(
    state: Any, pristine: Mapping[str, tuple[Any, ...]]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], float, int]:
    compiled = compile_public_belief(state._metagross_public_events, pristine)
    history = {
        belief.species: {candidate.candidate_id: candidate.weight for candidate in belief.candidates}
        for belief in compiled.species
        if belief.status == "compiled"
    }
    members = list(state.opponent.reserve)
    if state.opponent.active is not None:
        members.append(state.opponent.active)
    current: dict[str, dict[str, float]] = {}
    total_tv = 0.0
    affected = 0
    snapshot = state._metagross_random_battle_sets
    for member in members:
        if not member.is_alive():
            continue
        species = normalize_id(member.name)
        current_weights = _current_strict_weights(
            species,
            member,
            list(snapshot.get(species, pristine.get(species, ()))),
        )
        if not current_weights:
            # This is the exact production B0 behavior: Foul Play leaves the
            # observed partial Pokémon untouched when strict filtering finds
            # no compatible frozen set.
            current_weights = {OBSERVED_PARTIAL_CANDIDATE: 1.0}
        current[species] = current_weights
        # A missing/unsupported history compilation is deployment fallback,
        # not a zero-mass alternative distribution.
        history_weights = history.get(species) or dict(current_weights)
        history[species] = history_weights
        tv = 0.5 * math.fsum(
            abs(current_weights.get(identity, 0.0) - history_weights.get(identity, 0.0))
            for identity in set(current_weights) | set(history_weights)
        )
        total_tv += tv
        affected += int(tv > 1e-12)
    return current, history, total_tv, affected


def select_panel(corpus: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pristine = _pristine_candidates()
    representative, sensitivity = partition_corpus(corpus)
    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for panel_name, battles in (("representative", representative), ("sensitivity", sensitivity)):
        for battle in battles:
            observer, opponent = observer_for(battle)
            truth_sets = battle["teams"][opponent]["sets"]
            truth_species = {
                normalize_id(row.get("species") or row.get("name")) for row in truth_sets
            }
            base = {
                "battle_id": battle["battle_id"],
                "corpus_uid": battle["corpus_uid"],
                "panel": panel_name,
                "observer": observer,
                "opponent": opponent,
            }
            if any(normalize_id(row.get("ability")) == "illusion" for row in truth_sets):
                exclusions.append({**base, "reason": "illusion_truth_unsupported"})
                continue
            if any(
                truth_candidate_id(row)[1]
                not in _candidate_pool_ids(normalize_id(row.get("species") or row.get("name")), pristine)
                for row in truth_sets
            ):
                exclusions.append({**base, "reason": "truth_set_out_of_pool"})
                continue
            states = _reconstruct_view(battle, observer)
            eligible = []
            for decision_idx, state in enumerate(states):
                allowed = run_foul_play.request_player_actions(state)
                if len(allowed) < 2:
                    continue
                active_request = (state.request_json.get("active") or [{}])[0]
                # Showdown advertises switches under ``maybeTrapped`` to avoid
                # leaking Shadow Tag/Arena Trap.  A truth-injected teacher can
                # know those switches are impossible, so this public flag is
                # the only truth-blind way to preserve exact common legality.
                if active_request.get("maybeTrapped") is True:
                    continue
                revealed_species = {
                    normalize_id(member.name)
                    for member in [state.opponent.active, *state.opponent.reserve]
                    if member is not None
                }
                if not revealed_species <= truth_species:
                    # The replay contains a currently unsupported Illusion
                    # attribution (often both disguise and real species).
                    continue
                _current, _history, belief_tv, affected = root_beliefs(state, pristine)
                if affected < 1:
                    continue
                # Reconstruction snapshots are request-indexed prefixes.  The
                # assertion below binds the selected index to an actual recorded
                # request and prevents a post-battle synthetic state from entering.
                decisions = battle["views"][observer]["decisions"]
                if decision_idx >= len(decisions):
                    raise ValueError("reconstruction produced a non-request root")
                eligible.append(
                    {
                        **base,
                        "decision_idx": decision_idx,
                        "decision_chunk_count": decisions[decision_idx]["chunk_count"],
                        "turn": int(state.turn),
                        "belief_total_variation": belief_tv,
                        "affected_living_species": affected,
                        "legal_actions": sorted(allowed),
                        "tera_available": any(action.endswith("-tera") for action in allowed),
                    }
                )
            if not eligible:
                exclusions.append({**base, "reason": "no_eligible_history_informative_root"})
                continue
            if panel_name == "representative":
                chosen = min(
                    eligible,
                    key=lambda root: rank("representative-root", battle["corpus_uid"], root["decision_idx"]),
                )
            else:
                chosen = sorted(
                    eligible,
                    key=lambda root: (-root["belief_total_variation"], root["decision_idx"]),
                )[0]
            selected.append(chosen)
    counts = {
        name: sum(root["panel"] == name for root in selected)
        for name in ("representative", "sensitivity")
    }
    extension_used = len(corpus) > REPRESENTATIVE_BATTLES + SENSITIVITY_BATTLES
    return {
        "schema": PANEL_SCHEMA,
        "selection_truth_blind": True,
        "counts": counts,
        "required": {
            "representative": REQUIRED_REPRESENTATIVE_ROOTS,
            "sensitivity": REQUIRED_SENSITIVITY_ROOTS,
        },
        "extension_used": extension_used,
        "extension_authorized": (not extension_used and (
            counts["representative"] < REQUIRED_REPRESENTATIVE_ROOTS
            or counts["sensitivity"] < REQUIRED_SENSITIVITY_ROOTS
        )),
        "extension_exhausted": extension_used,
        "roots": selected,
        "exclusions": exclusions,
    }


def _sample_baseline_worlds(
    state: Any,
    pristine: Mapping[str, tuple[Any, ...]],
    count: int,
    world_seed: int,
) -> list[dict[str, Any]]:
    if str(shadow_replay.FOUL_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(shadow_replay.FOUL_PLAY_ROOT))
    from data.pkmn_sets import RandomBattleTeamDatasets
    from fp.search.helpers import populate_pkmn_from_set
    from fp.search.poke_engine_helpers import battle_to_poke_engine_state
    from fp.search.random_battles import (
        get_all_remaining_sets_for_revealed_pkmn,
        populate_randombattle_unrevealed_pkmn,
    )

    rng = random.Random(world_seed)
    source = deepcopy(state)
    current_sets = RandomBattleTeamDatasets.pkmn_sets
    RandomBattleTeamDatasets.pkmn_sets = {
        name: list(rows) for name, rows in source._metagross_random_battle_sets.items()
    }
    try:
        revealed = get_all_remaining_sets_for_revealed_pkmn(deepcopy(source), rng=rng)
        worlds = []
        for index in range(count):
            world = deepcopy(source)
            selected = {}
            members = list(world.opponent.reserve)
            if world.opponent.active is not None:
                members.append(world.opponent.active)
            for pokemon in members:
                candidates = revealed.get(pokemon.name, [])
                if not candidates or not pokemon.is_alive():
                    continue
                species = normalize_id(pokemon.name)
                probabilities = _normalized_counts(species, candidates)
                candidate_by_id = {
                    candidate_id(species, candidate_fields(candidate)): candidate
                    for candidate in candidates
                }
                support = sorted(probabilities)
                identity = rng.choices(support, weights=[probabilities[row] for row in support])[0]
                populate_pkmn_from_set(pokemon, candidate_by_id[identity])
                selected[species] = identity
            populate_randombattle_unrevealed_pkmn(world, rng=rng)
            world.opponent.lock_moves()
            serialized = battle_to_poke_engine_state(world).to_string()
            worlds.append(
                {
                    "index": index,
                    "state": serialized,
                    "state_sha256": sha256_bytes(serialized.encode("ascii")),
                    "selected_candidates": selected,
                }
            )
        return worlds
    finally:
        RandomBattleTeamDatasets.pkmn_sets = current_sets


def action_rows(result: Any, allowed: set[str]) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    for raw in result.side_one:
        action = run_foul_play._authorized_action_name(raw.move_choice, allowed)
        if action is None or raw.visits <= 0:
            continue
        entry = rows.setdefault(action, {"visits": 0, "total_score": 0.0})
        entry["visits"] = int(entry["visits"]) + int(raw.visits)
        entry["total_score"] = float(entry["total_score"]) + float(raw.total_score)
    if not rows:
        raise ValueError("MCTS produced no request-authorized action results")
    for entry in rows.values():
        entry["q"] = float(entry["total_score"]) / int(entry["visits"])
    return dict(sorted(rows.items()))


def aggregate_visits(searches: Sequence[Mapping[str, Mapping[str, float | int]]]) -> dict[str, float]:
    mass: dict[str, float] = {}
    for search in searches:
        total = math.fsum(float(row["visits"]) for row in search.values())
        if total <= 0:
            raise ValueError("world search has zero authorized visits")
        for action, row in search.items():
            mass[action] = mass.get(action, 0.0) + float(row["visits"]) / total / len(searches)
    return dict(sorted(mass.items()))


def argmax(values: Mapping[str, float]) -> str:
    return sorted(values, key=lambda action: (-values[action], action))[0]


def teacher_values(
    engine: Any,
    truth_state: str,
    allowed: set[str],
    identity: Sequence[object],
    iterations: int,
) -> tuple[list[dict[str, float]], dict[str, float], dict[str, float]]:
    repeats = []
    for repeat in range(TEACHER_REPEATS):
        result = engine.monte_carlo_tree_search(
            engine.State.from_string(truth_state),
            duration_ms=0,
            iterations=iterations,
            threads=1,
            seed=seed(*identity, "teacher", iterations, repeat),
        )
        rows = action_rows(result, allowed)
        repeats.append({action: float(row["q"]) for action, row in rows.items()})
    support = sorted(set.intersection(*(set(row) for row in repeats)))
    if set(support) != allowed:
        missing = sorted(allowed - set(support))
        raise ValueError(f"teacher did not evaluate every authorized action: {missing}")
    means = {action: math.fsum(row[action] for row in repeats) / len(repeats) for action in support}
    half_widths = {
        action: T_CRITICAL_DF2_95 * stdev([row[action] for row in repeats]) / math.sqrt(len(repeats))
        for action in support
    }
    return repeats, means, half_widths


def _regret_half_width(repeats: Sequence[Mapping[str, float]], baseline: str, best: str) -> float:
    deltas = [row[best] - row[baseline] for row in repeats]
    return T_CRITICAL_DF2_95 * stdev(deltas) / math.sqrt(len(deltas))


def wilson_interval(successes: int, count: int, z: float = 1.95996398454) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 1.0
    probability = successes / count
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(probability * (1 - probability) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def run_phase1(
    corpus_path: Path,
    manifest_path: Path,
    panel_path: Path,
    *,
    max_roots: int | None = None,
    smoke: bool = False,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    engine = load_engine()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, corpus_path)
    corpus = load_corpus_v2(corpus_path)
    by_id = {row["battle_id"]: row for row in corpus}
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    if panel.get("schema") != PANEL_SCHEMA:
        raise ValueError("invalid Phase-1 root panel")
    if not smoke and panel.get("extension_authorized"):
        raise RuntimeError("root panel is undersized; generate the one allowed extension first")
    selected = list(panel["roots"])
    if max_roots is not None:
        selected = selected[:max_roots]
    panel_sha = sha256_path(panel_path)
    roots: list[dict[str, Any]] = []
    if resume and checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("schema") != SCHEMA
            or checkpoint.get("manifest_sha256") != manifest["manifest_sha256"]
            or checkpoint.get("corpus_sha256") != sha256_path(corpus_path)
            or checkpoint.get("panel_sha256") != panel_sha
            or bool(checkpoint.get("smoke")) != smoke
        ):
            raise ValueError("Phase-1 checkpoint does not match the frozen run")
        roots = list(checkpoint.get("roots") or [])
        expected_prefix = [
            (root["battle_id"], root["observer"], root["decision_idx"])
            for root in selected[: len(roots)]
        ]
        actual_prefix = [
            (root["battle_id"], root["observer"], root["decision_idx"])
            for root in roots
        ]
        if actual_prefix != expected_prefix:
            raise ValueError("Phase-1 checkpoint roots are not the frozen panel prefix")
    pristine = _pristine_candidates()
    for root in selected[len(roots):]:
        battle = by_id[root["battle_id"]]
        states = _reconstruct_view(battle, root["observer"])
        reconstructed = states[root["decision_idx"]]
        allowed = run_foul_play.request_player_actions(reconstructed)
        if set(root["legal_actions"]) != allowed:
            raise ValueError("root legality differs from frozen panel")
        worlds = _sample_baseline_worlds(
            reconstructed,
            pristine,
            2 if smoke else BASELINE_WORLD_COUNT,
            seed(root["corpus_uid"], root["observer"], root["decision_idx"], "b0-worlds"),
        )
        world_searches = []
        for world in worlds:
            result = engine.monte_carlo_tree_search(
                engine.State.from_string(world["state"]),
                duration_ms=0,
                iterations=256 if smoke else WORLD_ITERATIONS,
                threads=1,
                seed=seed(root["corpus_uid"], root["observer"], root["decision_idx"], "b0", world["index"]),
            )
            world_searches.append(action_rows(result, allowed))
        baseline_policy = aggregate_visits(world_searches)
        baseline_action = argmax(baseline_policy)

        truth = _truth_battle(reconstructed, battle["teams"][root["opponent"]]["sets"], pristine)
        if str(shadow_replay.FOUL_PLAY_ROOT) not in sys.path:
            sys.path.insert(0, str(shadow_replay.FOUL_PLAY_ROOT))
        from fp.search.poke_engine_helpers import battle_to_poke_engine_state

        truth_state = battle_to_poke_engine_state(truth).to_string()
        initial_iterations = 512 if smoke else TEACHER_ITERATIONS
        repeats, means, half_widths = teacher_values(
            engine,
            truth_state,
            allowed,
            (root["corpus_uid"], root["observer"], root["decision_idx"]),
            initial_iterations,
        )
        best_action = argmax(means)
        regret_half_width = _regret_half_width(repeats, baseline_action, best_action)
        escalated = False
        if not smoke and regret_half_width > TEACHER_HALF_WIDTH_LIMIT:
            escalated = True
            repeats, means, half_widths = teacher_values(
                engine,
                truth_state,
                allowed,
                (root["corpus_uid"], root["observer"], root["decision_idx"]),
                TEACHER_ESCALATION_ITERATIONS,
            )
            best_action = argmax(means)
            regret_half_width = _regret_half_width(repeats, baseline_action, best_action)
        headroom = means[best_action] - means[baseline_action]
        roots.append(
            {
                **root,
                "baseline_world_hashes": [world["state_sha256"] for world in worlds],
                "baseline_policy": baseline_policy,
                "baseline_action": baseline_action,
                "teacher_iterations": (
                    TEACHER_ESCALATION_ITERATIONS if escalated else initial_iterations
                ),
                "teacher_escalated": escalated,
                "teacher_repeats": repeats,
                "teacher_mean_q": means,
                "teacher_action_half_widths": half_widths,
                "teacher_best_action": best_action,
                "baseline_headroom": headroom,
                "headroom_half_width": regret_half_width,
                "headroom_stable": regret_half_width <= TEACHER_HALF_WIDTH_LIMIT,
                "meaningful_headroom": (
                    regret_half_width <= TEACHER_HALF_WIDTH_LIMIT
                    and headroom >= MEANINGFUL_HEADROOM
                ),
            }
        )
        if checkpoint_path is not None:
            write_json(
                checkpoint_path,
                {
                    "schema": SCHEMA,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "corpus_sha256": sha256_path(corpus_path),
                    "panel_sha256": panel_sha,
                    "smoke": smoke,
                    "complete": False,
                    "completed_roots": len(roots),
                    "total_roots": len(selected),
                    "roots": roots,
                },
            )
    representative = [root for root in roots if root["panel"] == "representative"]
    stable = [root for root in representative if root["headroom_stable"]]
    meaningful = sum(root["meaningful_headroom"] for root in representative)
    interval = wilson_interval(meaningful, len(representative))
    gate_passed = (
        bool(smoke)
        or (
            len(representative) >= REQUIRED_REPRESENTATIVE_ROOTS
            and len([root for root in roots if root["panel"] == "sensitivity"])
            >= REQUIRED_SENSITIVITY_ROOTS
            and interval[1] >= 0.05
            and meaningful >= 10
        )
    )
    return {
        "schema": SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "corpus_sha256": sha256_path(corpus_path),
        "panel_sha256": panel_sha,
        "smoke": smoke,
        "complete": True,
        "summary": {
            "roots": len(roots),
            "representative_roots": len(representative),
            "sensitivity_roots": sum(root["panel"] == "sensitivity" for root in roots),
            "stable_representative_roots": len(stable),
            "meaningful_representative_roots": meaningful,
            "meaningful_representative_frequency": meaningful / len(representative) if representative else 0.0,
            "meaningful_representative_wilson95": list(interval),
            "gate_passed": gate_passed,
            "decision": "continue_to_phase2" if gate_passed else "end_belief_research",
        },
        "roots": roots,
    }


def write_json(path: Path, value: Any) -> None:
    def json_safe(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: json_safe(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_safe(child) for child in item]
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--corpus", type=Path)
    panel_parser = subparsers.add_parser("panel")
    panel_parser.add_argument("--corpus", type=Path, required=True)
    panel_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--corpus", type=Path, required=True)
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--panel", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--max-roots", type=int)
    run_parser.add_argument("--smoke", action="store_true")
    run_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        write_json(args.output, build_manifest(args.corpus))
    elif args.command == "panel":
        write_json(args.output, select_panel(load_corpus_v2(args.corpus)))
    else:
        write_json(
            args.output,
            run_phase1(
                args.corpus,
                args.manifest,
                args.panel,
                max_roots=args.max_roots,
                smoke=args.smoke,
                checkpoint_path=args.output,
                resume=args.resume,
            ),
        )


if __name__ == "__main__":
    main()
