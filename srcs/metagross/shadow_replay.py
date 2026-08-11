#!/usr/bin/env python3
"""Replay captured production decisions through their frozen holdout policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

from srcs.metagross import run_foul_play
from srcs.metagross.decision_harness import (
    execute_recursive_shadow,
    plan_recursive_shadow,
)
from srcs.metagross.mcts_contract import (
    ENGINE_CONTRACT,
    ENGINE_SOURCE_SHA256,
    REQUEST_SCHEMA,
    validate_holdout_result_payload,
    validate_result_payload,
)
from srcs.metagross.world_provenance import (
    RNG_SCHEME,
    canonical_json,
    derive_seed,
    deterministic_request_id,
    read_ledger,
    semantic_capture_digest,
    state_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
FOUL_PLAY_ROOT = ROOT / "srcs" / "vendor" / "foul-play"
DEFAULT_CAPTURE = (
    ROOT
    / "srcs"
    / "runtime"
    / "local-r1-v3-jeanfan83"
    / "20260806T041728Z-r1-jeanfan83-28708"
)
PATHOLOGY_TURNS = {1, 5, 6, 26, 36, 46}


class _DatasetSnapshot(dict):
    """Immutable-by-convention candidate lists shared across battle deep copies."""

    def __deepcopy__(self, memo):
        return self


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decision_key(row: dict) -> tuple[str, int]:
    tag = row.get("tag")
    index = row.get("decision_idx")
    if not isinstance(tag, str) or not tag:
        raise ValueError("decision row has an invalid battle tag")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("decision row has an invalid decision index")
    return tag, index


def _search_key(row: dict) -> tuple[str, int]:
    context = row.get("context")
    if not isinstance(context, dict):
        raise ValueError("search row has no context")
    return _decision_key(context)


def load_capture(capture: Path) -> tuple[list[dict], dict[tuple[str, int], dict], dict]:
    manifest_path = capture / "manifest.json"
    decisions_path = capture / "decisions.jsonl"
    search_path = capture / "search.jsonl"
    protocol_path = capture / "protocol.jsonl"
    for path in (manifest_path, decisions_path, search_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decisions = _read_jsonl(decisions_path)
    searches = _read_jsonl(search_path)
    protocol = _read_jsonl(protocol_path)
    ledger_by_key = None
    ledger_path = None
    rng = manifest.get("rng")
    search_config = manifest.get("search") or {}
    remote_config = search_config.get("modal") or search_config.get("http") or {}
    v5_capture = remote_config.get("schema") == REQUEST_SCHEMA
    if v5_capture:
        if not isinstance(rng, dict) or rng.get("scheme") != RNG_SCHEME:
            raise ValueError("v5 capture manifest has an invalid RNG scheme")
        configured = (manifest.get("outputs") or {}).get("holdout_ledger")
        if not isinstance(configured, str) or not configured:
            raise ValueError("v5 capture manifest has no holdout ledger")
        ledger_path = capture / Path(configured).name
        if not ledger_path.is_file():
            raise FileNotFoundError(ledger_path)
        ledger_rows = read_ledger(ledger_path)
        if [row.get("sequence_index") for row in ledger_rows] != list(
            range(len(ledger_rows))
        ):
            raise ValueError("holdout ledger sequence is not contiguous")
        if any(
            row.get("certification") is not None
            and (
                not isinstance(row["certification"], dict)
                or row["certification"].get("alpha_sequence_index")
                != row.get("sequence_index")
            )
            for row in ledger_rows
        ):
            raise ValueError("holdout ledger alpha sequence is inconsistent")
        ledger_by_key = {
            _search_key(row): row
            for row in ledger_rows
        }
        if len(ledger_by_key) != len(ledger_rows):
            raise ValueError("duplicate holdout ledger keys")
    decision_by_key = {_decision_key(row): row for row in decisions}
    search_by_key = {_search_key(row): row for row in searches}
    if len(decision_by_key) != len(decisions):
        raise ValueError("duplicate decision keys")
    if len(search_by_key) != len(searches):
        raise ValueError("duplicate search keys")
    if set(decision_by_key) != set(search_by_key):
        missing_search = sorted(set(decision_by_key) - set(search_by_key))
        missing_decision = sorted(set(search_by_key) - set(decision_by_key))
        raise ValueError(
            f"decision/search join mismatch: search={missing_search}, decision={missing_decision}"
        )
    if ledger_by_key is not None and set(ledger_by_key) != set(search_by_key):
        raise ValueError("holdout ledger/search join mismatch")
    if any(row.get("mask_fallback") for row in decisions):
        raise ValueError("capture contains a legality-mask fallback")

    for key, search in search_by_key.items():
        samples = search.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"search {key} has no sampled worlds")
        indexes = []
        weight = 0.0
        for sample in samples:
            indexes.append(sample.get("index"))
            chance = sample.get("sample_chance")
            if (
                isinstance(chance, bool)
                or not isinstance(chance, (int, float))
                or not math.isfinite(chance)
                or chance <= 0
            ):
                raise ValueError(f"search {key} has an invalid sample weight")
            weight += chance
            validate_result_payload(sample.get("result"))
        if indexes != list(range(len(samples))):
            raise ValueError(f"search {key} has non-contiguous world indexes")
        if not math.isclose(weight, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"search {key} sample weights do not sum to one")
        if ledger_by_key is not None:
            ledger = ledger_by_key[key]
            override = search.get("choice_override") or {}
            selection = ledger.get("selection_cohort") or {}
            remote = search.get("remote_search") or {}
            if (
                ledger.get("final_choice") != search.get("choice")
                or ledger.get("final_reason") != override.get("reason")
                or ledger.get("baseline") != override.get("baseline")
                or ledger.get("candidate_panel") != override.get("candidate_panel")
                or canonical_json(ledger.get("certification"))
                != canonical_json(override.get("holdout_panel"))
                or selection.get("sampling_seed") != remote.get("sampling_seed")
                or selection.get("state_hashes") != remote.get("state_hashes")
                or selection.get("request_ids") != remote.get("request_ids")
                or selection.get("weights")
                != [float(sample["sample_chance"]) for sample in samples]
            ):
                raise ValueError(f"holdout ledger/search mismatch at {key}")

    files = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in (
            ("manifest", manifest_path),
            ("decisions", decisions_path),
            ("search", search_path),
            ("protocol", protocol_path),
        )
    }
    if ledger_path is not None:
        files["holdout_ledger"] = {
            "path": str(ledger_path),
            "sha256": _sha256(ledger_path),
        }
    digest = hashlib.sha256(
        "\n".join(files[name]["sha256"] for name in sorted(files)).encode("ascii")
    ).hexdigest()
    metadata = {
        "manifest": manifest,
        "files": files,
        "capture_digest": digest,
        "decisions": len(decisions),
        "searches": len(searches),
        "protocol_rows": len(protocol),
        "battle_tags": sorted({key[0] for key in decision_by_key}),
        "semantic_capture_digest": semantic_capture_digest(capture),
        "holdout_ledger_rows": ledger_by_key,
    }
    return protocol, search_by_key, {**metadata, "decision_rows": decision_by_key}


def _protocol_lines(message: str) -> tuple[str | None, list[str]]:
    lines = message.splitlines()
    tag = None
    if lines and lines[0].startswith(">battle-"):
        tag = lines.pop(0)[1:]
    return tag, lines


def reconstruct_battles(
    protocol: list[dict],
    search_by_key: dict[tuple[str, int], dict],
    username: str,
) -> dict[tuple[str, int], object]:
    if str(FOUL_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(FOUL_PLAY_ROOT))
    import constants
    from data.pkmn_sets import RandomBattleTeamDatasets
    from fp.battle import Battle, LastUsedMove
    from fp.battle_modifier import process_battle_updates, request, update_battle

    battles: dict[str, object] = {}
    next_index: dict[str, int] = {}
    reconstructed: dict[tuple[str, int], object] = {}

    def ensure_battle(tag: str):
        if tag not in battles:
            # Live Foul Play reloads the dataset for every random battle. Damage
            # inference then narrows its global candidate lists during that battle.
            RandomBattleTeamDatasets.initialize("gen9")
            battle = Battle(tag)
            battle.pokemon_format = "gen9randombattle"
            battle.generation = "gen9"
            battle.battle_type = constants.BattleType.RANDOM_BATTLE
            battles[tag] = battle
            next_index[tag] = 0
        return battles[tag]

    for row in protocol:
        if row.get("direction") != "received":
            continue
        message = row.get("message")
        if not isinstance(message, str):
            raise ValueError("received protocol row has no message")
        tag, lines = _protocol_lines(message)
        if tag is None:
            continue
        battle = ensure_battle(tag)

        for line in lines:
            parts = line.split("|")
            if len(parts) >= 4 and parts[1] == "player":
                role, account = parts[2], parts[3]
                if account.lower() == username.lower():
                    battle.user.name = role
                    battle.user.account_name = account
                else:
                    battle.opponent.name = role
                    battle.opponent.account_name = account

        # Production checks the whole websocket message for a terminal outcome
        # before applying any bundled request. Mirror that ordering so a final
        # request+win message cannot create a phantom decision.
        if any(line.startswith(("|win|", "|tie|")) for line in lines):
            continue

        first_request = battle.user.active is None
        if first_request:
            request_line = next(
                (line for line in lines if line.startswith("|request|")), None
            )
            for line in lines:
                if line == request_line:
                    continue
                parts = line.split("|")
                if (
                    len(parts) >= 3
                    and parts[1] == "switch"
                    and battle.user.name
                    and parts[2].startswith(battle.user.name)
                ):
                    continue
                battle.msg_list.append(line)
            if request_line is None:
                continue
            if not battle.user.name or not battle.opponent.name:
                raise ValueError(f"request for {tag} arrived before player roles")
            request_parts = request_line.split("|")
            request_json = json.loads(request_parts[2])
            battle.user.initialize_first_turn_user_from_json(request_json)
            request(battle, request_parts)
            process_battle_updates(battle)
            action_required = not battle.wait
        else:
            action_required = update_battle(battle, "\n".join(lines))

        if not action_required:
            continue
        index = next_index[tag]
        key = (tag, index)
        if key not in search_by_key:
            raise ValueError(f"protocol produced an uncaptured decision {key}")
        battle_copy = deepcopy(battle)
        if not battle_copy.team_preview:
            battle_copy.user.update_from_request_json(battle_copy.request_json)
        battle_copy._metagross_random_battle_sets = _DatasetSnapshot(
            {
                name: tuple(candidate_sets)
                for name, candidate_sets in RandomBattleTeamDatasets.pkmn_sets.items()
            }
        )
        reconstructed[key] = battle_copy

        selected = search_by_key[key].get("choice")
        if not isinstance(selected, str) or not selected:
            raise ValueError(f"search {key} has no selected action")
        battle.user.last_selected_move = LastUsedMove(
            battle.user.active.name,
            selected.removesuffix("-tera").removesuffix("-mega"),
            battle.turn,
        )
        next_index[tag] += 1

    if set(reconstructed) != set(search_by_key):
        missing = sorted(set(search_by_key) - set(reconstructed))
        extra = sorted(set(reconstructed) - set(search_by_key))
        raise ValueError(
            f"protocol reconstruction mismatch: missing={missing}, extra={extra}"
        )
    return reconstructed


def _priors(decision: dict) -> list[tuple[str, float]]:
    probabilities = decision.get("probs")
    names = decision.get("name_table")
    if not isinstance(probabilities, list) or not isinstance(names, dict):
        raise ValueError("decision has invalid policy outputs")
    priors = []
    for name, index in names.items():
        if (
            not isinstance(name, str)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(probabilities)
        ):
            raise ValueError("decision has an invalid name table")
        probability = probabilities[index]
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or probability < 0
        ):
            raise ValueError("decision has an invalid probability")
        priors.append((name, float(probability)))
    return priors


def _mcts_results(search: dict):
    import poke_engine

    return [
        (
            run_foul_play._mcts_result_from_payload(sample["result"], poke_engine),
            float(sample["sample_chance"]),
            int(sample["index"]),
        )
        for sample in search["samples"]
    ]


def _seed(digest: str, channel: str, tag: str, decision: int, world: int = 0) -> int:
    payload = f"shadow-v4\0{digest}\0{channel}\0{tag}\0{decision}\0{world}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _fresh_worlds(battle, count: int, seed: int):
    if str(FOUL_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(FOUL_PLAY_ROOT))
    from fp.search.poke_engine_helpers import battle_to_poke_engine_state
    from fp.search.random_battles import prepare_random_battles
    from data.pkmn_sets import RandomBattleTeamDatasets

    sampled_battle = deepcopy(battle)
    if sampled_battle.team_preview:
        sampled_battle.user.active = sampled_battle.user.reserve.pop(0)
        sampled_battle.opponent.active = sampled_battle.opponent.reserve.pop(0)
    snapshot = getattr(sampled_battle, "_metagross_random_battle_sets", None)
    current_sets = RandomBattleTeamDatasets.pkmn_sets
    if snapshot is not None:
        RandomBattleTeamDatasets.pkmn_sets = {
            name: list(candidate_sets) for name, candidate_sets in snapshot.items()
        }
    try:
        sampled = prepare_random_battles(
            sampled_battle, count, rng=random.Random(seed)
        )
    finally:
        RandomBattleTeamDatasets.pkmn_sets = current_sets
    states = [battle_to_poke_engine_state(world).to_string() for world, _ in sampled]
    weights = [float(weight) for _, weight in sampled]
    return states, weights


def _remote_holdout(
    states: list[str],
    baseline: str,
    candidate: str,
    capture_digest: str,
    tag: str,
    decision_index: int,
    function,
    native_sha256: str,
    continuation_steps: int,
    opponent_priors=None,
    run_seed: str | None = None,
    candidate_rank: int = 1,
    request_channel: str = "certification-request",
    tape_channel: str = "holdout-tape",
) -> tuple[list[dict], float]:
    requests = []
    for world, state in enumerate(states):
        request_id = (
            deterministic_request_id(
                run_seed,
                tag,
                decision_index,
                f"{candidate_rank}:{continuation_steps}:{world}",
                channel=request_channel,
            )
            if run_seed is not None
            else hashlib.sha256(
                f"{capture_digest}\0{tag}\0{decision_index}\0{world}".encode("utf-8")
            ).hexdigest()[:32]
        )
        requests.append(
            {
                "schema": REQUEST_SCHEMA,
                "operation": "paired_holdout",
                "request_id": request_id,
                "index": world,
                "state": state,
                "baseline_action": baseline,
                "candidate_action": candidate,
                "rollouts": run_foul_play.HOLDOUT_ROLLOUTS,
                "continuation_iterations": (
                    run_foul_play.HOLDOUT_CONTINUATION_ITERATIONS
                ),
                "continuation_steps": continuation_steps,
                "seed": (
                    derive_seed(run_seed, tape_channel, tag, decision_index, world)
                    if run_seed is not None
                    else _seed(
                        capture_digest, "holdout-tape", tag, decision_index, world
                    )
                ),
                "opponent_priors": (
                    [list(row) for row in opponent_priors]
                    if opponent_priors
                    else None
                ),
            }
        )
    started = time.monotonic()
    responses = function.remote(requests)
    elapsed_ms = round((time.monotonic() - started) * 1000, 3)
    if not isinstance(responses, list) or len(responses) != len(requests):
        raise RuntimeError("remote holdout returned the wrong batch size")
    results = []
    for request, response in zip(requests, responses, strict=True):
        if (
            not isinstance(response, dict)
            or response.get("schema") != REQUEST_SCHEMA
            or response.get("request_id") != request["request_id"]
            or response.get("index") != request["index"]
            or response.get("ok") is not True
        ):
            raise RuntimeError("remote holdout response failed correlation")
        engine = response.get("engine")
        if (
            not isinstance(engine, dict)
            or engine.get("contract") != ENGINE_CONTRACT
            or engine.get("source_sha256") != ENGINE_SOURCE_SHA256
            or engine.get("native_sha256") != native_sha256
        ):
            raise RuntimeError("remote holdout engine identity mismatch")
        results.append(
            validate_holdout_result_payload(
                response.get("result"),
                expected_pairs=request["rollouts"],
                maximum_executed=(
                    2
                    * request["rollouts"]
                    * request["continuation_iterations"]
                    * request["continuation_steps"]
                ),
            )
        )
    return results, elapsed_ms


def _recompute_captured_v5_panel(panel: object) -> dict[str, dict]:
    if not isinstance(panel, dict) or panel.get("complete") is not True:
        raise ValueError("captured v5 holdout panel is incomplete")
    candidate_panel = panel.get("candidate_panel")
    recorded_by_action = panel.get("certificates_by_action")
    cohort = panel.get("certification_cohort")
    if (
        not isinstance(candidate_panel, list)
        or not isinstance(recorded_by_action, dict)
        or not isinstance(cohort, dict)
    ):
        raise ValueError("captured v5 holdout panel is malformed")
    expected_actions = [row.get("action") for row in candidate_panel]
    if set(recorded_by_action) != set(expected_actions):
        raise ValueError("captured v5 holdout panel has inconsistent candidates")
    recomputed_by_action = {}
    for panel_row in candidate_panel:
        action = panel_row.get("action")
        rank = panel_row.get("rank")
        combined = recorded_by_action[action]
        if combined.get("candidate_rank") != rank:
            raise ValueError("captured v5 candidate rank changed")
        certificates = combined.get("certificates")
        if not isinstance(certificates, dict):
            raise ValueError("captured v5 candidate has no look certificates")
        recomputed_looks = {}
        for horizon, certificate in certificates.items():
            if certificate.get("alpha_sequence_index") != panel.get(
                "alpha_sequence_index"
            ):
                raise ValueError("captured v5 alpha sequence changed")
            recomputed = run_foul_play.recompute_robust_holdout_certificate(certificate)
            if canonical_json(recomputed) != canonical_json(certificate):
                raise ValueError("captured v5 look certificate does not recompute")
            if (
                recomputed["state_hashes"] != cohort.get("state_hashes")
                or recomputed["cluster_hashes"] != cohort.get("cluster_hashes")
                or recomputed["world_weights"] != cohort.get("weights")
            ):
                raise ValueError("captured v5 certificate cohort changed")
            recomputed_looks[int(horizon)] = recomputed
        recomputed_combined = run_foul_play.combined_robust_holdout_certificate(
            recomputed_looks
        )
        if canonical_json(recomputed_combined) != canonical_json(combined):
            raise ValueError("captured v5 combined certificate does not recompute")
        recomputed_by_action[action] = recomputed_combined
    qualified = [
        row["action"]
        for row in candidate_panel
        if recomputed_by_action[row["action"]]["qualified"] is True
    ]
    if qualified != panel.get("qualified_actions"):
        raise ValueError("captured v5 qualified action list changed")
    return recomputed_by_action


def run_replay(
    capture: Path,
    dry_run: bool = False,
    continuation_steps: int = run_foul_play.HOLDOUT_CONTINUATION_STEPS,
    captured_holdout: bool = False,
    recursive_shadow: bool = False,
) -> tuple[dict, list[dict]]:
    if not 1 <= continuation_steps <= 4:
        raise ValueError("continuation_steps must be between 1 and 4")
    if dry_run and captured_holdout:
        raise ValueError("dry-run and captured-holdout modes are mutually exclusive")
    if recursive_shadow and (dry_run or captured_holdout):
        raise ValueError("recursive shadow requires live remote replay")
    protocol, search_by_key, metadata = load_capture(capture)
    manifest = metadata["manifest"]
    username = manifest["ladder"]["username"]
    reconstructed = reconstruct_battles(protocol, search_by_key, username)
    decision_by_key = metadata.pop("decision_rows")
    ledger_by_key = metadata.pop("holdout_ledger_rows") or {}
    is_v5_capture = bool(ledger_by_key)
    if recursive_shadow and not is_v5_capture:
        raise ValueError("recursive shadow requires a v5 capture")

    remote_function = None
    engine = None
    if captured_holdout:
        first_search = next(iter(search_by_key.values()))
        engine = (first_search.get("remote_search") or {}).get("engine")
        if not isinstance(engine, dict):
            raise ValueError("capture has no remote engine identity")
    elif not dry_run:
        import modal

        app = "metagross-mcts-r1-p16"
        remote_function = modal.Function.from_name(app, "search_batch")
        engine = modal.Function.from_name(app, "engine_info").remote()
        if (
            engine.get("contract") != ENGINE_CONTRACT
            or engine.get("source_sha256") != ENGINE_SOURCE_SHA256
        ):
            raise RuntimeError("deployed v5 engine identity mismatch")

    histories: dict = {}
    rows = []
    for key in sorted(search_by_key):
        tag, decision_index = key
        battle = reconstructed[key]
        decision = decision_by_key[key]
        search = search_by_key[key]
        results = _mcts_results(search)
        priors = _priors(decision)
        _provisional_choice, provisional = run_foul_play.select_final_choice(
            battle,
            results,
            priors,
            histories,
            record_history=False,
        )
        baseline = provisional["baseline"]
        candidate = provisional["raw_choice"]
        frozen_panel = run_foul_play.freeze_holdout_candidate_panel(battle, provisional)
        if key in ledger_by_key and frozen_panel != ledger_by_key[key].get(
            "candidate_panel"
        ):
            raise ValueError(f"frozen candidate panel mismatch at {key}")
        if key in ledger_by_key:
            run_seed = (manifest.get("rng") or {}).get("run_seed")
            selection = ledger_by_key[key].get("selection_cohort") or {}
            expected_sampling_seed = derive_seed(
                run_seed, "selection-worlds", tag, decision_index, 0
            )
            selection_states, selection_weights = _fresh_worlds(
                battle, len(search["samples"]), expected_sampling_seed
            )
            expected_request_ids = [
                deterministic_request_id(
                    run_seed,
                    tag,
                    decision_index,
                    index,
                    channel="selection-search-request",
                )
                for index in range(len(selection_states))
            ]
            if (
                selection.get("sampling_seed") != expected_sampling_seed
                or selection.get("state_hashes")
                != [state_sha256(state) for state in selection_states]
                or selection.get("request_ids") != expected_request_ids
                or selection.get("weights") != selection_weights
            ):
                raise ValueError(f"selection cohort provenance mismatch at {key}")
        request_actions = set(provisional["request_actions"])
        no_op_reason = run_foul_play._known_noop_reason(battle, candidate)
        prediction_reason = run_foul_play._prediction_sensitive_reason(
            battle, candidate
        )
        eligible = (
            candidate != baseline
            and candidate in request_actions
            and no_op_reason is None
            and prediction_reason is None
        )
        if key in ledger_by_key:
            eligible = bool(frozen_panel)
        holdout = None
        holdout_panel = None
        evidence_by_action = {}
        certificates_by_action = {}
        recursive_inputs = None
        latency_ms = None
        failure = None
        if eligible and captured_holdout:
            if key in ledger_by_key:
                holdout_panel = ledger_by_key[key].get("certification")
                try:
                    evidence_by_action = _recompute_captured_v5_panel(holdout_panel)
                    holdout = evidence_by_action.get(candidate)
                    if holdout is None and frozen_panel:
                        holdout = evidence_by_action.get(frozen_panel[0]["action"])
                except (KeyError, TypeError, ValueError) as exc:
                    failure = type(exc).__name__
            else:
                holdout = (search.get("choice_override") or {}).get("holdout")
                if not isinstance(holdout, dict):
                    failure = "MissingCapturedHoldout"
                elif holdout.get("complete") is not True:
                    failure = str(holdout.get("error") or "IncompleteCapturedHoldout")
                else:
                    evidence_by_action = {candidate: holdout}
        elif eligible and not dry_run:
            try:
                world_count = len(search["samples"])
                recorded_panel = ledger_by_key.get(key, {}).get("certification")
                run_seed = (
                    (manifest.get("rng") or {}).get("run_seed")
                    if recorded_panel is not None
                    else None
                )
                sampling_seed = (
                    derive_seed(
                        run_seed,
                        "certification-worlds",
                        tag,
                        decision_index,
                        0,
                    )
                    if run_seed is not None
                    else _seed(
                        metadata["capture_digest"],
                        "fresh-worlds",
                        tag,
                        decision_index,
                    )
                )
                states, weights = _fresh_worlds(
                    battle,
                    world_count,
                    sampling_seed,
                )
                if recorded_panel is not None:
                    cohort = recorded_panel.get("certification_cohort") or {}
                    hashes = [state_sha256(state) for state in states]
                    if (
                        hashes != cohort.get("state_hashes")
                        or weights != cohort.get("weights")
                        or sampling_seed != cohort.get("sampling_seed")
                    ):
                        raise ValueError("replayed v5 certification cohort changed")
                    total_latency = 0.0
                    for panel_row in frozen_panel:
                        action = panel_row["action"]
                        rank = panel_row["rank"]
                        certificates = {}
                        for horizon_index, horizon in enumerate(
                            run_foul_play.HOLDOUT_CONTINUATION_HORIZONS
                        ):
                            holdout_results, elapsed = _remote_holdout(
                                states,
                                baseline,
                                action,
                                metadata["capture_digest"],
                                tag,
                                decision_index,
                                remote_function,
                                engine["native_sha256"],
                                horizon,
                                recorded_panel.get("opponent_priors"),
                                run_seed,
                                rank,
                            )
                            total_latency += elapsed
                            certificate = run_foul_play.robust_holdout_certificate(
                                holdout_results,
                                weights,
                                hashes,
                                hashes,
                                action,
                                baseline,
                                recorded_panel["alpha_sequence_index"],
                                rank,
                                horizon_index,
                            )
                            certificates[horizon] = certificate
                            if certificate["qualified"] is not True:
                                break
                        evidence_by_action[action] = (
                            run_foul_play.combined_robust_holdout_certificate(
                                certificates
                            )
                        )
                        certificates_by_action[action] = certificates
                    latency_ms = round(total_latency, 3)
                    holdout = evidence_by_action.get(candidate)
                    if holdout is None and frozen_panel:
                        holdout = evidence_by_action.get(frozen_panel[0]["action"])
                    holdout_panel = {
                        **recorded_panel,
                        "certificates_by_action": evidence_by_action,
                        "qualified_actions": [
                            row["action"]
                            for row in frozen_panel
                            if evidence_by_action[row["action"]]["qualified"] is True
                        ],
                    }
                    recursive_inputs = {
                        "states": states,
                        "weights": weights,
                        "hashes": hashes,
                        "run_seed": run_seed,
                        "recorded_panel": recorded_panel,
                    }
                else:
                    holdout_results, latency_ms = _remote_holdout(
                        states,
                        baseline,
                        candidate,
                        metadata["capture_digest"],
                        tag,
                        decision_index,
                        remote_function,
                        engine["native_sha256"],
                        continuation_steps,
                        search.get("opponent_priors"),
                    )
                    holdout = run_foul_play.independent_holdout_certificate(
                        holdout_results,
                        weights,
                        candidate,
                        baseline,
                        decision_index,
                    )
                    evidence_by_action = {candidate: holdout}
            except Exception as exc:  # Fail closed and preserve the audit row.
                failure = type(exc).__name__
                holdout = {
                    "candidate": candidate,
                    "baseline": baseline,
                    "evidence_kind": "independent_seeded_paired_holdout",
                    "complete": False,
                    "qualified": False,
                    "error": failure,
                }
        final_choice, final = run_foul_play.select_final_choice(
            battle,
            results,
            priors,
            histories,
            independent_evidence=evidence_by_action or None,
        )
        recursive_artifact = None
        if recursive_shadow:
            plan = plan_recursive_shadow(provisional, certificates_by_action)

            def allocate_recursive(shadow_plan):
                if recursive_inputs is None:
                    raise RuntimeError("recursive shadow has no v5 replay inputs")
                candidate_action = str(shadow_plan.candidate)
                panel_row = next(
                    row for row in frozen_panel if row["action"] == candidate_action
                )
                rank = int(panel_row["rank"])
                run_seed = recursive_inputs["run_seed"]
                if shadow_plan.operation == "horizon":
                    states = recursive_inputs["states"]
                    weights = recursive_inputs["weights"]
                    hashes = recursive_inputs["hashes"]
                    tape_channel = "holdout-tape"
                    sampling_seed = derive_seed(
                        run_seed, "certification-worlds", tag, decision_index, 0
                    )
                elif shadow_plan.operation == "worlds":
                    tape_channel = "recursive-shadow-tape"
                    sampling_seed = derive_seed(
                        run_seed, "recursive-shadow-worlds", tag, decision_index, 0
                    )
                    states, weights = _fresh_worlds(battle, 16, sampling_seed)
                    total_weight = math.fsum(weights)
                    if total_weight <= 0:
                        raise RuntimeError("recursive shadow has no world mass")
                    weights = [weight / total_weight for weight in weights]
                    hashes = [state_sha256(state) for state in states]
                else:
                    raise RuntimeError("unsupported recursive shadow operation")
                results, elapsed = _remote_holdout(
                    states,
                    baseline,
                    candidate_action,
                    metadata["capture_digest"],
                    tag,
                    decision_index,
                    remote_function,
                    engine["native_sha256"],
                    2,
                    recursive_inputs["recorded_panel"].get("opponent_priors"),
                    run_seed,
                    rank,
                    request_channel="recursive-shadow-request",
                    tape_channel=tape_channel,
                )
                certificate = run_foul_play.robust_holdout_certificate(
                    results,
                    weights,
                    hashes,
                    hashes,
                    candidate_action,
                    baseline,
                    recursive_inputs["recorded_panel"]["alpha_sequence_index"],
                    rank,
                    1,
                )
                return {
                    "sampling_seed": sampling_seed,
                    "worlds": len(states),
                    "continuation_steps": 2,
                    "state_hashes": hashes,
                    "latency_ms": elapsed,
                    "certificate": certificate,
                }

            recursive_artifact = execute_recursive_shadow(
                plan, final_choice, allocate_recursive
            )
            recursive_artifact.update(
                {
                    "context": {
                        "tag": tag,
                        "decision_idx": decision_index,
                        "battle_turn": decision["battle_turn"],
                    },
                    "baseline": baseline,
                    "search_top": candidate,
                }
            )
        selected_holdout = evidence_by_action.get(final_choice, holdout)
        holdout_admitted = bool(
            selected_holdout
            and selected_holdout.get("qualified")
            and final_choice != baseline
        )
        deterministic_correction = (
            provisional["final_choice"] != baseline
            and provisional["reason"] != "independent_holdout_qualified_search_override"
        )
        holdout_required_admission = holdout_admitted and not deterministic_correction
        recorded_request_actions = set(
            (search.get("choice_override") or {}).get("request_actions") or ()
        )
        if recorded_request_actions and request_actions != recorded_request_actions:
            raise ValueError(f"request-action reconstruction mismatch at {key}")
        rows.append(
            {
                "tag": tag,
                "decision_idx": decision_index,
                "battle_turn": decision["battle_turn"],
                "recorded_choice": search.get("choice"),
                "baseline": baseline,
                "candidate": candidate,
                "provisional_choice": provisional["final_choice"],
                "provisional_reason": provisional["reason"],
                "eligible_for_holdout": eligible,
                "known_noop_reason": no_op_reason,
                "prediction_sensitive_reason": prediction_reason,
                "holdout": holdout,
                "holdout_panel": holdout_panel,
                "holdout_latency_ms": latency_ms,
                "holdout_failure": failure,
                "shadow_choice": final_choice,
                "shadow_reason": final["reason"],
                "blocked_safeguard": final["blocked_safeguard"],
                "recursive_shadow": recursive_artifact,
                "holdout_admitted": holdout_admitted,
                "holdout_required_admission": holdout_required_admission,
                "deterministic_correction": deterministic_correction,
                "recorded_behavior_changed": final_choice != search.get("choice"),
                "request_actions": sorted(request_actions),
                "spotlight": (
                    tag == metadata["battle_tags"][0]
                    and decision["battle_turn"] in PATHOLOGY_TURNS
                ),
            }
        )

    eligible_rows = [row for row in rows if row["eligible_for_holdout"]]
    evaluated_rows = [row for row in eligible_rows if row["holdout"]]
    admitted_rows = [row for row in rows if row["holdout_admitted"]]
    required_admission_rows = [row for row in rows if row["holdout_required_admission"]]
    latencies = [
        row["holdout_latency_ms"]
        for row in rows
        if row["holdout_latency_ms"] is not None
    ]
    report = {
        "schema": 1,
        "mode": (
            "captured_v5_holdout"
            if captured_holdout and is_v5_capture
            else "captured_v4_holdout"
            if captured_holdout
            else "dry_run"
            if dry_run
            else "remote_v5_holdout"
            if is_v5_capture
            else "remote_v4_holdout"
        ),
        "capture": metadata,
        "engine": engine,
        "evaluation": {
            "rollouts": run_foul_play.HOLDOUT_ROLLOUTS,
            "continuation_iterations": run_foul_play.HOLDOUT_CONTINUATION_ITERATIONS,
            "continuation_steps": continuation_steps,
            "seed_scheme": (
                RNG_SCHEME
                if is_v5_capture
                else "sha256(capture,channel,tag,decision,world)"
            ),
        },
        "counts": {
            "decisions": len(rows),
            "policy_search_agreements": sum(
                row["candidate"] == row["baseline"] for row in rows
            ),
            "policy_search_disagreements": sum(
                row["candidate"] != row["baseline"] for row in rows
            ),
            "eligible_disagreements": len(eligible_rows),
            "evaluated_disagreements": len(evaluated_rows),
            "admitted_overrides": len(admitted_rows),
            "holdout_required_admissions": len(required_admission_rows),
            "deterministic_corrections": sum(
                row["deterministic_correction"] for row in rows
            ),
            "blocked_safeguards": sum(
                row["blocked_safeguard"] is not None for row in rows
            ),
            "recorded_behavior_changes": sum(
                row["recorded_behavior_changed"] for row in rows
            ),
            "holdout_failures": sum(bool(row["holdout_failure"]) for row in rows),
            "known_noop_candidates": sum(
                bool(row["known_noop_reason"]) for row in rows
            ),
            "prediction_sensitive_candidates": sum(
                bool(row["prediction_sensitive_reason"]) for row in rows
            ),
            "request_invalid_candidates": sum(
                row["candidate"] not in row["request_actions"] for row in rows
            ),
            "recursive_shadow_triggers": sum(
                bool((row["recursive_shadow"] or {}).get("triggered")) for row in rows
            ),
            "recursive_shadow_failures": sum(
                row["recursive_shadow"] is not None
                and row["recursive_shadow"].get("complete") is not True
                for row in rows
            ),
        },
        "latency_ms": {
            "total": round(sum(latencies), 3),
            "minimum": min(latencies) if latencies else None,
            "maximum": max(latencies) if latencies else None,
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
        },
        "admitted": [
            {
                "tag": row["tag"],
                "decision_idx": row["decision_idx"],
                "battle_turn": row["battle_turn"],
                "baseline": row["baseline"],
                "candidate": row["candidate"],
                "shadow_choice": row["shadow_choice"],
            }
            for row in admitted_rows
        ],
        "holdout_required_admitted": [
            {
                "tag": row["tag"],
                "decision_idx": row["decision_idx"],
                "battle_turn": row["battle_turn"],
                "baseline": row["baseline"],
                "candidate": row["candidate"],
                "shadow_choice": row["shadow_choice"],
            }
            for row in required_admission_rows
        ],
        "spotlight": [row for row in rows if row["spotlight"]],
    }
    return report, rows


def _write_report(output: Path, report: dict, rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def compare_horizon_reports(first: dict, second: dict) -> dict:
    first_digest = first.get("capture", {}).get("capture_digest")
    second_digest = second.get("capture", {}).get("capture_digest")
    if not first_digest or first_digest != second_digest:
        raise ValueError("horizon reports use different captures")
    for field in ("contract", "source_sha256", "native_sha256"):
        if first.get("engine", {}).get(field) != second.get("engine", {}).get(field):
            raise ValueError("horizon reports use different engine identities")
    first_horizon = first.get("evaluation", {}).get("continuation_steps")
    second_horizon = second.get("evaluation", {}).get("continuation_steps")
    if first_horizon == second_horizon:
        raise ValueError("horizon reports must use different continuation depths")

    def admission_map(report: dict) -> dict[tuple[str, int], dict]:
        rows = report.get("holdout_required_admitted")
        if not isinstance(rows, list):
            # The initial horizon-1 report predates the causal-admission summary.
            rows = [
                row
                for row in report.get("admitted", [])
                if not (
                    row.get("tag") == "battle-gen9randombattle-2660576873"
                    and row.get("battle_turn") == 6
                )
            ]
        return {(row["tag"], row["decision_idx"]): row for row in rows}

    first_admitted = admission_map(first)
    second_admitted = admission_map(second)
    stable_keys = sorted(set(first_admitted) & set(second_admitted))
    first_only = sorted(set(first_admitted) - set(second_admitted))
    second_only = sorted(set(second_admitted) - set(first_admitted))

    def rows(keys, mapping):
        return [mapping[key] for key in keys]

    return {
        "schema": 1,
        "capture_digest": first_digest,
        "engine": first["engine"],
        "horizons": sorted([first_horizon, second_horizon]),
        "counts": {
            "first_required_admissions": len(first_admitted),
            "second_required_admissions": len(second_admitted),
            "stable_required_admissions": len(stable_keys),
            "first_only": len(first_only),
            "second_only": len(second_only),
        },
        "stable_required_admitted": rows(stable_keys, first_admitted),
        "first_only": rows(first_only, first_admitted),
        "second_only": rows(second_only, second_admitted),
        "spotlight": {
            str(first_horizon): first.get("spotlight", []),
            str(second_horizon): second.get("spotlight", []),
        },
    }


def _write_comparison(output: Path, comparison: dict) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--captured-holdout", action="store_true")
    parser.add_argument("--recursive-shadow", action="store_true")
    parser.add_argument(
        "--continuation-steps", type=int, default=1, choices=range(1, 5)
    )
    parser.add_argument("--compare-reports", type=Path, nargs=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.compare_reports:
        if args.dry_run or args.recursive_shadow:
            raise ValueError(
                "--dry-run/--recursive-shadow cannot be combined with --compare-reports"
            )
        if args.captured_holdout:
            raise ValueError(
                "--captured-holdout cannot be combined with --compare-reports"
            )
        reports = [
            json.loads(path.resolve().read_text(encoding="utf-8"))
            for path in args.compare_reports
        ]
        comparison = compare_horizon_reports(*reports)
        _write_comparison(args.output.resolve(), comparison)
        print(json.dumps(comparison["counts"], sort_keys=True))
        print(args.output.resolve())
        return
    report, rows = run_replay(
        args.capture.resolve(),
        args.dry_run,
        args.continuation_steps,
        args.captured_holdout,
        args.recursive_shadow,
    )
    _write_report(args.output.resolve(), report, rows)
    print(json.dumps(report["counts"], sort_keys=True))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
