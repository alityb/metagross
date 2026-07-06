from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# Gen1 feature set (14 features) — includes speed/KO features the hand eval misses.
# Must match FEATURE_COUNT and extract_features() in
# engine/patches/poke-engine-0.0.47-learned-eval.patch.
FEATURE_NAMES = [
    "hp_frac_diff",                  # 0  side total HP fraction (s1 - s2)
    "alive_frac_diff",               # 1  alive count fraction (s1 - s2)
    "active_hp_frac_diff",           # 2  active mon HP fraction (s1 - s2)
    "status_frac_diff",              # 3  opponent status frac - own (s2 - s1)
    "attack_boost_diff",             # 4  (s1 - s2) / 6
    "defense_boost_diff",            # 5  (s1 - s2) / 6
    "special_attack_boost_diff",     # 6  (s1 - s2) / 6  [= special in gen1]
    "speed_boost_diff",              # 7  (s1 - s2) / 6
    "sub_diff",                      # 8  s1_has_sub - s2_has_sub
    "active_stat_total_diff",        # 9  active mon total stats normalized
    "team_stat_total_diff",          # 10 team total stats normalized
    "damage_ratio_diff",             # 11 NEW: (s1_best_dmg/s2_hp - s2_best_dmg/s1_hp)
    "speed_diff",                    # 12 NEW: (s1_eff_speed - s2_eff_speed) / 500
    "outspeeds",                     # 13 NEW: +1/-1/0 turn order
]

STAT_KEYS = {
    "atk": "atk",
    "def": "def",
    "spa": "spa",
    "spd": "spd",
    "spe": "spe",
}


@dataclass
class PublicMon:
    hp: int = 0
    maxhp: int = 100
    status: str = ""
    item_known: bool = False
    terastallized: bool = False


@dataclass
class PublicSide:
    active: Optional[str] = None
    mons: dict[str, PublicMon] = field(default_factory=dict)
    boosts: dict[str, int] = field(
        default_factory=lambda: {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
    )
    conditions: dict[str, int] = field(default_factory=dict)
    substitute: bool = False


@dataclass
class PublicBattle:
    sides: dict[str, PublicSide] = field(
        default_factory=lambda: {"p1": PublicSide(), "p2": PublicSide()}
    )
    winner_side: Optional[str] = None
    player_to_side: dict[str, str] = field(default_factory=dict)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def side_from_ident(value: str) -> Optional[str]:
    value = value.strip()
    if value.startswith("p1"):
        return "p1"
    if value.startswith("p2"):
        return "p2"
    return None


def species_from_ident(value: str) -> str:
    if ":" in value:
        value = value.split(":", 1)[1]
    return value.strip().split(",", 1)[0].strip()


def species_from_details(value: str) -> str:
    return value.strip().split(",", 1)[0].strip()


def parse_condition(value: str, previous_maxhp: int = 100) -> tuple[int, int, str]:
    value = value.strip()
    parts = value.split()
    hp_part = parts[0] if parts else "0"
    status = ""
    for part in parts[1:]:
        if part in {"brn", "par", "psn", "tox", "slp", "frz"}:
            status = part
    if hp_part == "0" or hp_part == "0/100" or "fnt" in parts:
        return 0, previous_maxhp, status
    if "/" in hp_part:
        hp_str, maxhp_str = hp_part.split("/", 1)
        return int(hp_str), max(1, int(maxhp_str)), status
    hp = int(hp_part)
    return hp, max(previous_maxhp, hp, 1), status


def get_active_mon(state: PublicBattle, side_name: str) -> Optional[PublicMon]:
    side = state.sides[side_name]
    if side.active is None:
        return None
    return side.mons.get(side.active)


def update_mon_condition(side: PublicSide, species: str, condition: str) -> None:
    mon = side.mons.setdefault(species, PublicMon())
    hp, maxhp, status = parse_condition(condition, mon.maxhp)
    mon.hp = hp
    mon.maxhp = maxhp
    if status:
        mon.status = status
    if hp == 0:
        mon.status = ""


def reset_active_boosts(side: PublicSide) -> None:
    for key in side.boosts:
        side.boosts[key] = 0
    side.substitute = False


def side_hp_fraction(side: PublicSide) -> float:
    return sum(max(0.0, min(1.0, mon.hp / max(1, mon.maxhp))) for mon in side.mons.values()) / 6.0


def side_alive_fraction(side: PublicSide) -> float:
    return sum(1 for mon in side.mons.values() if mon.hp > 0) / 6.0


def side_status_fraction(side: PublicSide) -> float:
    return sum(1 for mon in side.mons.values() if mon.hp > 0 and mon.status) / 6.0


def side_item_fraction(side: PublicSide) -> float:
    return sum(1 for mon in side.mons.values() if mon.hp > 0 and mon.item_known) / 6.0


def side_used_tera(side: PublicSide) -> float:
    return 1.0 if any(mon.terastallized for mon in side.mons.values()) else 0.0


def hp_fraction(mon: Optional[PublicMon]) -> float:
    if mon is None or mon.hp <= 0:
        return 0.0
    return max(0.0, min(1.0, mon.hp / max(1, mon.maxhp)))


def screen_score(side: PublicSide) -> float:
    return (
        side.conditions.get("reflect", 0)
        + side.conditions.get("lightscreen", 0)
        + side.conditions.get("auroraveil", 0) * 2
    ) / 8.0


def hazard_score(side: PublicSide) -> float:
    return (
        side.conditions.get("stealthrock", 0)
        + side.conditions.get("spikes", 0)
        + side.conditions.get("toxicspikes", 0)
        + side.conditions.get("stickyweb", 0) * 2
    ) / 8.0


def features_for_fixed_side(state: PublicBattle, fixed_side: str) -> list[float]:
    other_side = "p2" if fixed_side == "p1" else "p1"
    side_one = state.sides[fixed_side]
    side_two = state.sides[other_side]
    side_one_active = get_active_mon(state, fixed_side)
    side_two_active = get_active_mon(state, other_side)
    return [
        side_hp_fraction(side_one) - side_hp_fraction(side_two),
        side_alive_fraction(side_one) - side_alive_fraction(side_two),
        hp_fraction(side_one_active) - hp_fraction(side_two_active),
        side_status_fraction(side_two) - side_status_fraction(side_one),
        side_item_fraction(side_one) - side_item_fraction(side_two),
        side_used_tera(side_two) - side_used_tera(side_one),
        side_one.boosts["atk"] / 6.0 - side_two.boosts["atk"] / 6.0,
        side_one.boosts["def"] / 6.0 - side_two.boosts["def"] / 6.0,
        side_one.boosts["spa"] / 6.0 - side_two.boosts["spa"] / 6.0,
        side_one.boosts["spd"] / 6.0 - side_two.boosts["spd"] / 6.0,
        side_one.boosts["spe"] / 6.0 - side_two.boosts["spe"] / 6.0,
        screen_score(side_one) - screen_score(side_two),
        hazard_score(side_two) - hazard_score(side_one),
        0.0,
        0.0,
        (1.0 if side_one.substitute else 0.0) - (1.0 if side_two.substitute else 0.0),
    ]


def condition_name(value: str) -> str:
    value = normalize_name(value)
    aliases = {
        "moveauroraveil": "auroraveil",
        "movereflect": "reflect",
        "movelightscreen": "lightscreen",
        "movestealthrock": "stealthrock",
        "movespikes": "spikes",
        "movetoxicspikes": "toxicspikes",
        "movestickyweb": "stickyweb",
    }
    return aliases.get(value, value)


def apply_line(state: PublicBattle, line: str) -> None:
    if not line.startswith("|"):
        return
    parts = line.split("|")
    tag = parts[1] if len(parts) > 1 else ""
    if tag == "player" and len(parts) >= 4:
        side = parts[2]
        player = parts[3]
        if side in {"p1", "p2"}:
            state.player_to_side[player] = side
    elif tag in {"switch", "drag"} and len(parts) >= 5:
        side_name = side_from_ident(parts[2])
        if side_name:
            side = state.sides[side_name]
            species = species_from_details(parts[3])
            side.active = species
            side.mons.setdefault(species, PublicMon())
            reset_active_boosts(side)
            update_mon_condition(side, species, parts[4])
    elif tag in {"-damage", "-heal", "-sethp"} and len(parts) >= 4:
        side_name = side_from_ident(parts[2])
        if side_name:
            side = state.sides[side_name]
            species = species_from_ident(parts[2])
            update_mon_condition(side, species, parts[3])
    elif tag == "faint" and len(parts) >= 3:
        side_name = side_from_ident(parts[2])
        if side_name:
            species = species_from_ident(parts[2])
            state.sides[side_name].mons.setdefault(species, PublicMon()).hp = 0
    elif tag == "-status" and len(parts) >= 4:
        side_name = side_from_ident(parts[2])
        if side_name:
            species = species_from_ident(parts[2])
            state.sides[side_name].mons.setdefault(species, PublicMon()).status = parts[3]
    elif tag == "-curestatus" and len(parts) >= 3:
        side_name = side_from_ident(parts[2])
        if side_name:
            species = species_from_ident(parts[2])
            state.sides[side_name].mons.setdefault(species, PublicMon()).status = ""
    elif tag in {"-boost", "-unboost"} and len(parts) >= 5:
        side_name = side_from_ident(parts[2])
        stat = STAT_KEYS.get(parts[3])
        if side_name and stat:
            delta = int(parts[4]) * (1 if tag == "-boost" else -1)
            side = state.sides[side_name]
            side.boosts[stat] = max(-6, min(6, side.boosts[stat] + delta))
    elif tag == "-setboost" and len(parts) >= 5:
        side_name = side_from_ident(parts[2])
        stat = STAT_KEYS.get(parts[3])
        if side_name and stat:
            state.sides[side_name].boosts[stat] = max(-6, min(6, int(parts[4])))
    elif tag in {"-clearboost", "-clearnegativeboost"} and len(parts) >= 3:
        side_name = side_from_ident(parts[2])
        if side_name:
            reset_active_boosts(state.sides[side_name])
    elif tag == "-clearallboost":
        for side in state.sides.values():
            reset_active_boosts(side)
    elif tag == "-sidestart" and len(parts) >= 4:
        side_name = side_from_ident(parts[2])
        if side_name:
            condition = condition_name(parts[3])
            side = state.sides[side_name]
            if condition in {"reflect", "lightscreen", "auroraveil", "stealthrock", "stickyweb"}:
                side.conditions[condition] = 1
            elif condition in {"spikes", "toxicspikes"}:
                side.conditions[condition] = side.conditions.get(condition, 0) + 1
    elif tag == "-sideend" and len(parts) >= 4:
        side_name = side_from_ident(parts[2])
        if side_name:
            state.sides[side_name].conditions[condition_name(parts[3])] = 0
    elif tag == "-terastallize" and len(parts) >= 3:
        side_name = side_from_ident(parts[2])
        if side_name:
            species = species_from_ident(parts[2])
            state.sides[side_name].mons.setdefault(species, PublicMon()).terastallized = True
    elif tag in {"-item", "-enditem"} and len(parts) >= 3:
        side_name = side_from_ident(parts[2])
        if side_name:
            species = species_from_ident(parts[2])
            state.sides[side_name].mons.setdefault(species, PublicMon()).item_known = tag == "-item"
    elif tag in {"-start", "-end"} and len(parts) >= 4:
        side_name = side_from_ident(parts[2])
        if side_name and condition_name(parts[3]) == "substitute":
            state.sides[side_name].substitute = tag == "-start"
    elif tag == "win" and len(parts) >= 3:
        state.winner_side = state.player_to_side.get(parts[2])


def examples_from_replay(payload: dict) -> tuple[list[list[float]], list[int]]:
    log = payload.get("log", "")
    state = PublicBattle()
    snapshots: list[tuple[list[float], str]] = []
    for line in log.splitlines():
        apply_line(state, line)
        if line.startswith("|turn|") and get_active_mon(state, "p1") and get_active_mon(state, "p2"):
            snapshots.append((features_for_fixed_side(state, "p1"), "p1"))
            snapshots.append((features_for_fixed_side(state, "p2"), "p2"))
    if state.winner_side not in {"p1", "p2"}:
        return [], []
    features = [feature for feature, _ in snapshots]
    labels = [1 if fixed_side == state.winner_side else 0 for _, fixed_side in snapshots]
    return features, labels


def examples_from_decision_log(
    jsonl_paths: list[Path],
) -> tuple[list[list[float]], list[int], list[str]]:
    """Load examples from poke-engine decision log files (generated by generate_selfplay_data.sh).

    Returns (features, labels, battle_tags). Features are computed via the Rust
    compute_value_features binding so they EXACTLY match inference.
    """
    try:
        import poke_engine as _pe
        _compute = _pe.compute_value_features
    except (ImportError, AttributeError):
        raise RuntimeError(
            "poke_engine.compute_value_features not available; "
            "rebuild with the enriched patch before training"
        )
    features: list[list[float]] = []
    labels: list[int] = []
    tags: list[str] = []
    for path in jsonl_paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "state" not in row or "label" not in row or "battle_tag" not in row:
                continue
            state = _pe.State.from_string(row["state"])
            features.append(_compute(state))
            labels.append(int(row["label"]))
            tags.append(row["battle_tag"])
    return features, labels, tags


def fetch_replay_index(format_id: str, page: int) -> list[dict]:
    query = urllib.parse.urlencode({"format": format_id, "page": page})
    request = urllib.request.Request(
        f"https://replay.pokemonshowdown.com/search.json?{query}",
        headers={"User-Agent": "metagross-phase1/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_replay_payload(replay_id: str) -> dict:
    request = urllib.request.Request(
        f"https://replay.pokemonshowdown.com/{replay_id}.json",
        headers={"User-Agent": "metagross-phase1/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_replay_payloads(args: argparse.Namespace) -> Iterable[dict]:
    if args.replay_dir:
        for path in sorted(Path(args.replay_dir).glob("*.json")):
            yield json.loads(path.read_text(encoding="utf-8"))
        return

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    yielded = 0
    for page in range(1, args.download_pages + 1):
        for row in fetch_replay_index(args.format, page):
            if yielded >= args.max_replays:
                return
            replay_id = row["id"]
            cache_path = cache_dir / f"{replay_id}.json" if cache_dir else None
            if cache_path and cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                payload = fetch_replay_payload(replay_id)
                if cache_path:
                    cache_path.write_text(json.dumps(payload), encoding="utf-8")
            yielded += 1
            yield payload


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def train_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=x_train.shape[1]).astype(np.float64)
    bias = 0.0
    n = max(1, len(x_train))
    for _ in range(epochs):
        logits = bias + x_train @ weights
        probs = sigmoid_np(logits)
        error = probs - y_train
        grad_bias = float(error.mean())
        grad_weights = (x_train.T @ error) / n + l2 * weights
        bias -= lr * grad_bias
        weights -= lr * grad_weights
    return bias, weights


def binary_cross_entropy(labels: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return float(-(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)).mean())


def reliability(labels: np.ndarray, probs: np.ndarray, bins: int) -> list[dict]:
    rows = []
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        mask = (probs >= lo) & ((probs < hi) if i + 1 < bins else (probs <= hi))
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": i, "lo": lo, "hi": hi, "count": 0, "mean_pred": None, "winrate": None})
        else:
            rows.append(
                {
                    "bin": i,
                    "lo": lo,
                    "hi": hi,
                    "count": count,
                    "mean_pred": float(probs[mask].mean()),
                    "winrate": float(labels[mask].mean()),
                }
            )
    return rows


def write_model(path: Path, bias: float, weights: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "metagross_value_net_v1",
        "# fixed side_one perspective; direct win probability leaf value",
        "# features " + " ".join(FEATURE_NAMES),
        f"bias {bias:.9g}",
        "weights " + " ".join(f"{w:.9g}" for w in weights),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 1 fixed-side value net from gen9randombattle replays")
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--replay-dir", default=None)
    parser.add_argument("--cache-dir", default="external/replays/gen9randombattle")
    parser.add_argument("--download-pages", type=int, default=4)
    parser.add_argument("--max-replays", type=int, default=200)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-out", default="nets/checkpoints/phase1_value_logreg_gen9randombattle.txt")
    parser.add_argument("--metrics-out", default="nets/checkpoints/phase1_value_logreg_gen9randombattle.metrics.json")
    args = parser.parse_args()

    features: list[list[float]] = []
    labels: list[int] = []
    replay_count = 0
    for payload in iter_replay_payloads(args):
        replay_features, replay_labels = examples_from_replay(payload)
        if replay_features:
            features.extend(replay_features)
            labels.extend(replay_labels)
            replay_count += 1

    if not features:
        raise RuntimeError("no training examples extracted")

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    heldout_n = max(1, int(len(indices) * args.heldout_fraction))
    heldout_idx = indices[:heldout_n]
    train_idx = indices[heldout_n:]
    if len(train_idx) == 0:
        raise RuntimeError("not enough examples for train/heldout split")

    bias, weights = train_logistic(x[train_idx], y[train_idx], args.epochs, args.lr, args.l2, args.seed)
    heldout_probs = sigmoid_np(bias + x[heldout_idx] @ weights)
    heldout_labels = y[heldout_idx]
    train_probs = sigmoid_np(bias + x[train_idx] @ weights)
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": args.format,
        "model_type": "logistic_value_net",
        "perspective": "fixed side_one; no to-move sign flip",
        "feature_masking": "public replay information only at each turn",
        "replays_used": replay_count,
        "examples": int(len(x)),
        "train_examples": int(len(train_idx)),
        "heldout_examples": int(len(heldout_idx)),
        "train_cross_entropy": binary_cross_entropy(y[train_idx], train_probs),
        "heldout_cross_entropy": binary_cross_entropy(heldout_labels, heldout_probs),
        "heldout_brier": float(np.mean((heldout_probs - heldout_labels) ** 2)),
        "heldout_accuracy": float(np.mean((heldout_probs >= 0.5) == heldout_labels)),
        "heldout_base_rate": float(heldout_labels.mean()),
        "feature_names": FEATURE_NAMES,
        "reliability": reliability(heldout_labels, heldout_probs, 10),
    }
    write_model(Path(args.model_out), bias, weights)
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
