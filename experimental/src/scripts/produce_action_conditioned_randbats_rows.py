#!/usr/bin/env python3
"""Produce no-future-leak Randbats action rows from Showdown replay JSON.

Rows intentionally omit ``action_likelihoods``.  To attach them, replay
``public_state.protocol_prefix`` through Metamon's ``forward.SimProtocol`` and
construct the p1-observer ``ReplayState`` immediately before the recorded p2
action, then call ``FrozenR1CandidatePolicyLikelihoodAdapter``.  No backward
fill or team predictor is used here.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from belief.action_conditioned_randbats import Candidate, CandidateValidationError, load_generator_pool_active_candidates


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _messages(raw: Mapping[str, Any]) -> list[list[str]]:
    log = raw.get("log")
    if not isinstance(log, str):
        raise ValueError("replay has no string log")
    return [[part.strip() for part in line.split("|")[1:]] for line in log.splitlines() if line.replace("|", "").strip()]


def _pokemon_facts(pokemon: Any) -> dict[str, Any]:
    """Only facts already exposed by forward parsing, never filled predictions."""
    facts: dict[str, Any] = {"speciesId": _norm(pokemon.had_name or pokemon.name), "level": pokemon.lvl}
    moves = sorted({_norm(move.name) for move in pokemon.had_moves.values()})
    if moves:
        facts["moves"] = moves
    if pokemon.had_ability:
        facts["ability"] = _norm(pokemon.had_ability)
    if pokemon.had_item:
        facts["item"] = _norm(pokemon.had_item)
    if pokemon.tera_type:
        facts["teraType"] = _norm(pokemon.tera_type)
    return facts


def _matches(candidate: Candidate, facts: Mapping[str, Any]) -> bool:
    data = candidate.public_data or {}
    species = data.get("speciesId", data.get("species"))
    if _norm(species) != facts["speciesId"]:
        return False
    for key in ("ability", "item", "teraType", "level"):
        if key in facts and _norm(data.get(key)) != _norm(facts[key]):
            return False
    candidate_moves = {_norm(move) for move in data.get("moves", [])}
    return set(facts.get("moves", [])) <= candidate_moves


def _candidate_record(candidate: Candidate) -> dict[str, Any]:
    # Candidate data is pool data only. In particular, it never contains a label.
    return {"candidate_id": candidate.candidate_id, "prior_weight": candidate.prior_weight, **dict(candidate.public_data or {})}


def _is_forced_switch(turn: Any, side: int = 2) -> bool:
    return any(subturn.team == side and subturn.slot == 0 for subturn in turn.subturns)


def _is_tera_action(messages: list[list[str]], action_index: int, side: str) -> bool:
    """Tera declaration follows the move in protocol but is part of that action."""
    for message in messages[action_index + 1 :]:
        if not message:
            continue
        if message[0] in {"move", "switch", "drag", "turn", "cant"}:
            return False
        if message[0] == "-terastallize" and message[1].startswith(side):
            return True
    return False


def _legal_actions(candidates: Iterable[Candidate], turn: Any, can_tera: bool) -> list[str]:
    moves = sorted({_norm(move) for candidate in candidates for move in (candidate.public_data or {}).get("moves", [])})
    actions = [f"move {move}" for move in moves]
    if can_tera:
        actions.extend(f"move {move}-tera" for move in moves)
    actions.extend(f"switch {_norm(pokemon.had_name or pokemon.name)}" for pokemon in turn.available_switches_2)
    return list(dict.fromkeys(actions))


def _final_pokemon(replay: Any, unique_id: str) -> Any | None:
    for pokemon in replay.turnlist[-1].pokemon_2:
        if pokemon is not None and pokemon.unique_id == unique_id:
            return pokemon
    return None


def replay_state_from_public_state(public_state: Mapping[str, Any]) -> Any:
    """Rebuild the raw p1-observer ReplayState required by the likelihood adapter."""
    from metamon.backend.replay_parser import forward
    from metamon.backend.replay_parser.replay_state import ParsedReplay, ReplayState

    prefix = public_state.get("protocol_prefix")
    if not isinstance(prefix, list):
        raise ValueError("public_state.protocol_prefix must be a protocol-message list")
    replay = ParsedReplay(gameid="public-prefix", format=public_state.get("format"), time_played=datetime.now(timezone.utc))
    protocol = forward.SimProtocol(replay)
    for message in prefix:
        protocol.interpret_message(message)
    turn = protocol.curr_turn
    if turn.active_pokemon_1[0] is None or turn.active_pokemon_2[0] is None:
        raise ValueError("public prefix lacks an active Pokemon on one side")
    return ReplayState(
        format=replay.format,
        force_switch=turn.is_force_switch,
        active_pokemon=turn.active_pokemon_1[0],
        opponent_active_pokemon=turn.active_pokemon_2[0],
        opponent_team=turn.pokemon_2,
        available_switches=turn.available_switches_1,
        player_prev_move=turn.active_pokemon_1[0].last_used_move,
        opponent_prev_move=turn.active_pokemon_2[0].last_used_move,
        player_conditions=turn.conditions_1,
        opponent_conditions=turn.conditions_2,
        weather=turn.weather,
        battle_field=turn.battle_field,
        battle_won=False,
        battle_lost=False,
        can_tera=turn.can_tera_1,
        opponent_teampreview=turn.teampreview_2,
    )


def rows_from_replay(raw: Mapping[str, Any], pool: tuple[Candidate, ...], replay_id: str | None = None) -> list[dict[str, Any]]:
    """Forward-replay one raw replay and return p2 discretionary action rows."""
    from metamon.backend.replay_parser import forward
    from metamon.backend.replay_parser.replay_state import ParsedReplay

    messages = _messages(raw)
    gameid = replay_id or str(raw.get("id") or "unknown-replay")
    uploadtime = raw.get("uploadtime")
    when = float(uploadtime) if isinstance(uploadtime, (int, float)) else None
    replay = ParsedReplay(gameid=gameid, format=raw.get("formatid"), time_played=datetime.now(timezone.utc))
    protocol = forward.SimProtocol(replay)
    captures: list[tuple[dict[str, Any], str, dict[str, Any], bool]] = []
    p2_cant_pending = False

    for index, message in enumerate(messages):
        if message and message[0] == "cant" and message[1].startswith("p2"):
            # Showdown does not provide a discretionary choice for this boundary.
            p2_cant_pending = True
        if message and message[0] in {"move", "switch"} and message[1].startswith("p2"):
            turn = protocol.curr_turn
            active = turn.active_pokemon_2[0]
            forced = p2_cant_pending or (message[0] == "switch" and _is_forced_switch(turn))
            # A move has intent; drag, forced replacement, and cant do not.
            if active is not None and not forced:
                prefix_facts = _pokemon_facts(active)
                candidates = tuple(candidate for candidate in pool if _matches(candidate, prefix_facts))
                if candidates:
                    if message[0] == "move":
                        tera = _is_tera_action(messages, index, "p2")
                        observed = f"move {_norm(message[2])}{'-tera' if tera else ''}"
                    else:
                        observed = f"switch {_norm(message[2].split(',', 1)[0])}"
                    legal = _legal_actions(candidates, turn, turn.can_tera_2)
                    if observed in legal:
                        captures.append(({
                            "schema_version": 1,
                            "replay_id": gameid,
                            "time": when,
                            "action_index": index,
                            "active_candidates": [_candidate_record(candidate) for candidate in candidates],
                            "legal_actions": legal,
                            "observed_action": observed,
                            "public_state": {
                                "format": raw.get("formatid"),
                                "observer_side": "p1",
                                "acting_side": "p2",
                                "acting_can_tera": turn.can_tera_2,
                                # Team size and faints are public; unrevealed members
                                # remain counted without being invented as Pokemon.
                                "public_opponent_remaining": 6 - sum(
                                    pokemon is not None and str(pokemon.status).endswith("FNT")
                                    for pokemon in turn.pokemon_2
                                ),
                                "protocol_prefix": messages[:index],
                                "handoff": "Replay protocol prefix is exclusive of observed_action; replay it with forward.SimProtocol and build ReplayState immediately before the action.",
                            },
                        }, active.unique_id, prefix_facts, False))
        protocol.interpret_message(message)
        if message and (message[0] == "turn" or (message[0] in {"move", "switch"} and message[1].startswith("p2"))):
            p2_cant_pending = False

    rows: list[dict[str, Any]] = []
    for capture_index, (row, unique_id, prefix_facts, _) in enumerate(captures):
        final = _final_pokemon(replay, unique_id)
        if final is not None:
            # This is deliberately a separate suffix-only label pass. Candidate
            # construction above used prefix_facts before the observed action.
            suffix_facts = _pokemon_facts(final)
            candidates = row["active_candidates"]
            compatible = [candidate["candidate_id"] for candidate in candidates if _matches(
                Candidate(candidate["candidate_id"], candidate["prior_weight"], candidate), suffix_facts
            )]
            # Require a later action by this same Pokemon. This conservative
            # guard prevents its observed action from becoming its own label.
            later_same_actor = any(
                other_id == unique_id
                for _, other_id, _, _ in captures[capture_index + 1 :]
            )
            if len(compatible) == 1 and suffix_facts != prefix_facts and later_same_actor:
                row["label"] = compatible[0]
        if row["time"] is None:
            row.pop("time")
        rows.append(row)
    return rows


def produce(replay_path: Path, output: Path, pool_path: Path, max_replays: int | None = None) -> dict[str, Any]:
    if not pool_path.is_file():
        raise ValueError(f"pool does not exist: {pool_path}")
    paths = sorted(replay_path.glob("*.json")) if replay_path.is_dir() else [replay_path]
    if max_replays is not None:
        paths = paths[:max_replays]
    pool = load_generator_pool_active_candidates(pool_path)
    rows: list[dict[str, Any]] = []
    failed = 0
    for path in paths:
        try:
            rows.extend(rows_from_replay(json.loads(path.read_text(encoding="utf-8")), pool, path.stem))
        except Exception:  # A malformed protocol must not prevent other raw replays from producing rows.
            failed += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return {"replays": len(paths), "failed_replays": failed, "rows": len(rows), "labeled_rows": sum("label" in row for row in rows), "label_coverage": sum("label" in row for row in rows) / len(rows) if rows else 0.0, "likelihoods_attached": False, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path, help="Raw replay JSON file or directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pool", default=ROOT / "data/randbats_pools/gen9randombattle_pool_50000.json", type=Path)
    parser.add_argument("--max-replays", type=int)
    args = parser.parse_args()
    if args.max_replays is not None and args.max_replays < 1:
        parser.error("--max-replays must be positive")
    try:
        print(json.dumps(produce(args.replay, args.output, args.pool, args.max_replays), sort_keys=True))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
