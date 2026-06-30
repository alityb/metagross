#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import metamon.env
from metamon.env.wrappers import ChallengeByUsername
from metamon.rl.pretrained import get_pretrained_model


ROOT = Path(__file__).resolve().parents[1]
NUMERIC_FIELDS = ["active_hp", "opponent_hp", "player_alive", "opponent_alive", "turn_index"]
SLEEP_MOVES = {"sleeppowder", "hypnosis", "lovelykiss", "sing", "spore"}
PARALYSIS_MOVES = {"thunderwave", "bodyslam", "stunspore", "glare"}
BOOM_MOVES = {"explosion", "selfdestruct"}
RECOVERY_MOVES = {"recover", "softboiled", "rest"}


def pokemon_name(pokemon: dict[str, Any] | None) -> str:
    if not pokemon:
        return "none"
    return str(pokemon.get("name") or pokemon.get("species") or "none").lower()


def hp_pct(pokemon: dict[str, Any] | None) -> float:
    if not pokemon:
        return 0.0
    try:
        return float(pokemon.get("hp_pct", 0.0))
    except (TypeError, ValueError):
        return 0.0


def status(pokemon: dict[str, Any] | None) -> str | None:
    if not pokemon:
        return None
    value = pokemon.get("status")
    if value in {None, "", "none", "nostatus"}:
        return None
    return str(value).lower()


def move_names(pokemon: dict[str, Any] | None) -> list[str]:
    if not pokemon:
        return []
    return [str(move.get("name", "")).lower() for move in pokemon.get("moves", [])]


def alive_count(state: dict[str, Any], side: str) -> int:
    if side == "player":
        mons = [state.get("player_active_pokemon")] + list(state.get("available_switches", []))
        return sum(1 for mon in mons if hp_pct(mon) > 0)
    try:
        return int(state.get("opponents_remaining", 0))
    except (TypeError, ValueError):
        return 0


def hp_bin(value: Any) -> str:
    try:
        hp = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if hp <= 0:
        return "0"
    if hp <= 0.25:
        return "1-25"
    if hp <= 0.5:
        return "26-50"
    if hp <= 0.75:
        return "51-75"
    return "76-100"


def action_name(state: dict[str, Any], action_idx: int) -> str:
    if action_idx < 0:
        return "noop"
    active = state.get("player_active_pokemon") or {}
    moves = move_names(active)
    if 0 <= action_idx < len(moves):
        return f"move:{moves[action_idx]}"
    switch_idx = action_idx - 4
    switches = state.get("available_switches", [])
    if 0 <= switch_idx < len(switches):
        return f"switch:{pokemon_name(switches[switch_idx])}"
    return f"unknown:{action_idx}"


def bucket_state(state: dict[str, Any]) -> str:
    active = state.get("player_active_pokemon")
    opponent = state.get("opponent_active_pokemon")
    player_alive = alive_count(state, "player")
    opponent_alive = alive_count(state, "opponent")
    active_name = pokemon_name(active)
    opponent_name = pokemon_name(opponent)
    active_moves = set(move_names(active))

    if state.get("forced_switch"):
        return "forced_switch"
    if player_alive <= 2 or opponent_alive <= 2:
        if active_name == "tauros" or opponent_name == "tauros":
            return "tauros_endgame"
        return "low_hp_endgame"
    if active_moves & SLEEP_MOVES:
        return "sleep_pressure"
    if active_moves & PARALYSIS_MOVES:
        return "paralysis_spread"
    if active_moves & BOOM_MOVES:
        return "explosion_opportunity"
    if active_name == "chansey" and opponent_name == "chansey":
        return "chansey_mirror"
    if active_name == "snorlax" or opponent_name == "snorlax":
        return "snorlax_trade"
    if active_moves & RECOVERY_MOVES:
        return "recovery_loop"
    if status(active) is not None:
        return "statused_active"
    return "other"


def token_features(state: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    active = state.get("player_active_pokemon") or {}
    opponent = state.get("opponent_active_pokemon") or {}
    active_moves = move_names(active)
    opponent_moves = move_names(opponent)
    active_hp = hp_pct(active)
    opponent_hp = hp_pct(opponent)
    player_alive = alive_count(state, "player")
    opponent_alive = alive_count(state, "opponent")
    tokens = [
        f"bucket={bucket_state(state)}",
        f"active={pokemon_name(active)}",
        f"opponent={pokemon_name(opponent)}",
        f"active_status={status(active) or 'none'}",
        f"opponent_status={status(opponent) or 'none'}",
        f"player_alive={player_alive}",
        f"opponent_alive={opponent_alive}",
        f"forced_switch={bool(state.get('forced_switch'))}",
        f"has_sleep={bool(set(active_moves) & SLEEP_MOVES)}",
        f"has_para={bool(set(active_moves) & PARALYSIS_MOVES)}",
        f"has_boom={bool(set(active_moves) & BOOM_MOVES)}",
        f"has_recovery={bool(set(active_moves) & RECOVERY_MOVES)}",
        f"active_hp_bin={hp_bin(active_hp)}",
        f"opponent_hp_bin={hp_bin(opponent_hp)}",
    ]
    tokens.extend(f"active_move={move}" for move in active_moves)
    tokens.extend(f"opponent_move={move}" for move in opponent_moves)
    numeric = {
        "active_hp": active_hp,
        "opponent_hp": opponent_hp,
        "player_alive": float(player_alive),
        "opponent_alive": float(opponent_alive),
        "turn_index": 0.0,
    }
    return tokens, numeric


class LinearPolicy:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.model_type = payload.get("model_type", "linear")
        self.classes = payload["classes"]
        self.vocab = payload["vocab"]
        self.numeric_fields = payload["numeric_fields"]
        if self.model_type == "mlp_relu":
            self.w1 = payload["w1"]
            self.b1 = payload["b1"]
            self.w2 = payload["w2"]
            self.b2 = payload["b2"]
        else:
            self.weight = payload["weight"]
            self.bias = payload["bias"]

    def logits(self, state: dict[str, Any], turn_index: int) -> dict[str, float]:
        tokens, numeric = token_features(state)
        numeric["turn_index"] = turn_index / 200.0
        x = [0.0 for _ in range(len(self.vocab) + len(self.numeric_fields))]
        for token in tokens:
            idx = self.vocab.get(token)
            if idx is not None:
                x[idx] = 1.0
        offset = len(self.vocab)
        for idx, field in enumerate(self.numeric_fields):
            x[offset + idx] = float(numeric.get(field, 0.0))
        if self.model_type == "mlp_relu":
            hidden = [max(0.0, sum(w * value for w, value in zip(row, x)) + bias) for row, bias in zip(self.w1, self.b1)]
            return {
                label: sum(w * value for w, value in zip(self.w2[class_idx], hidden)) + self.b2[class_idx]
                for class_idx, label in enumerate(self.classes)
            }
        return {
            label: sum(w * value for w, value in zip(self.weight[class_idx], x)) + self.bias[class_idx]
            for class_idx, label in enumerate(self.classes)
        }

    def choose(self, state: dict[str, Any], legal_actions: list[int], turn_index: int) -> int:
        logits = self.logits(state, turn_index)
        legal = [(action, action_name(state, int(action))) for action in legal_actions]
        scored = [(action, logits.get(label, -1e9), label) for action, label in legal]
        scored = [entry for entry in scored if not entry[2].startswith("unknown:")]
        if not scored:
            return int(legal_actions[0])
        return int(max(scored, key=lambda entry: entry[1])[0])


def run(args: argparse.Namespace) -> None:
    pretrained = get_pretrained_model("TaurosV0")
    team_set = metamon.env.get_metamon_teams(args.battle_format, args.team_set)
    env = ChallengeByUsername(
        battle_format=args.battle_format,
        num_battles=args.n_games,
        observation_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=pretrained.reward_function,
        player_team_set=team_set,
        player_username=args.username,
        opponent_username=args.opponent_username,
        role=args.role,
        save_results_to=args.save_results_to,
        save_trajectories_to=args.save_trajectories_to,
        battle_backend="metamon",
        print_battle_bar=args.print_battle_bar,
    )
    policy = LinearPolicy(Path(args.model))
    obs, info = env.reset()
    turn_index = 0
    finished = 0
    try:
        while finished < args.n_games:
            state = env._most_recent_state.to_dict()
            legal_actions = [int(action) for action in info.get("legal_actions", [])]
            if not legal_actions:
                legal_actions = [0]
            action = policy.choose(state, legal_actions, turn_index)
            obs, reward, terminated, truncated, info = env.step(action)
            turn_index += 1
            if terminated or truncated:
                finished += 1
                if finished >= args.n_games:
                    break
                obs, info = env.reset()
                turn_index = 0
    finally:
        env.close(purge=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a distilled Tauros linear policy")
    parser.add_argument("--model", default=str(ROOT / "nets" / "checkpoints" / "tauros_exact_action_n100.json"))
    parser.add_argument("--username", required=True)
    parser.add_argument("--opponent_username", required=True)
    parser.add_argument("--role", choices=["challenger", "acceptor"], default="acceptor")
    parser.add_argument("--n-games", type=int, default=1)
    parser.add_argument("--battle-format", default="gen1ou")
    parser.add_argument("--team-set", default="competitive")
    parser.add_argument("--save-results-to", default=None)
    parser.add_argument("--save-trajectories-to", default=None)
    parser.add_argument("--print-battle-bar", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
