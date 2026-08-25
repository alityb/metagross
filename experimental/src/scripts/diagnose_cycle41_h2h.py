#!/usr/bin/env python3
"""Development-only, outcome-open attribution for the clean Cycle41 H2H."""
from __future__ import annotations

import collections
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

from experimental.src.scripts.monitor_cycle19_operational_smoke import (
    received_request,
    rows,
)
from experimental.src.scripts.monitor_cycle21_registered_form_smoke import player_roles

ROOT = Path(__file__).resolve().parents[3]
C40 = ROOT / "experimental/runs/search_native_v2_cycle40_integrated_h2h_20260816"
C41 = ROOT / "experimental/runs/search_native_v2_cycle41_blind_scorer_repair_20260816"
OUT = C41 / "diagnosis_v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entropy(policy: dict[str, float]) -> float:
    values = [float(value) for value in policy.values() if float(value) > 0]
    if len(values) <= 1:
        return 0.0
    total = math.fsum(values)
    return -math.fsum((value / total) * math.log(value / total) for value in values) / math.log(len(values))


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = math.fsum(weights.values())
    return {key: value / total for key, value in weights.items()} if total > 0 else {}


def top(policy: dict[str, float]) -> str | None:
    return max(policy, key=policy.get) if policy else None


def rank(policy: dict[str, float], action: str) -> int | None:
    ordered = sorted(policy, key=lambda key: (-policy[key], key))
    return ordered.index(action) + 1 if action in ordered else None


def aggregate_production(samples: list[dict]) -> dict[str, Any]:
    visits: dict[str, float] = collections.defaultdict(float)
    scores: dict[str, float] = collections.defaultdict(float)
    world_tops = []
    for sample in samples:
        chance = float(sample.get("sample_chance", 0.0))
        actions = (sample.get("result") or {}).get("side_one") or []
        if actions:
            world_tops.append(max(actions, key=lambda row: int(row.get("visits", 0)))["move_choice"])
        for row in actions:
            action = str(row["move_choice"])
            visits[action] += chance * int(row.get("visits", 0))
            scores[action] += chance * float(row.get("total_score", 0.0))
    policy = normalize(visits)
    q = {action: scores[action] / value for action, value in visits.items() if value > 0}
    return {"policy": policy, "q": q, "world_tops": world_tops}


def aggregate_teacher(teacher: dict, selected: str, production: str) -> dict[str, Any]:
    receipts = teacher["receipts"]
    by_schedule: dict[int, dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    w_sum: dict[str, float] = collections.defaultdict(float)
    n_sum: dict[str, float] = collections.defaultdict(float)
    world_tops = []
    for receipt in receipts:
        weight = float(receipt.get("weight", 0.0))
        actions = receipt.get("side_one") or []
        if actions:
            world_tops.append(max(actions, key=lambda row: int(row["N"]))["action"])
        for row in actions:
            action = str(row["action"])
            count = int(row["N"])
            by_schedule[int(receipt["schedule_index"])][action] += weight * count
            n_sum[action] += weight * count
            w_sum[action] += weight * float(row["W"])
    schedule_policies = {index: normalize(dict(values)) for index, values in by_schedule.items()}
    schedule_tops = {index: top(policy) for index, policy in schedule_policies.items()}
    q = {action: w_sum[action] / count for action, count in n_sum.items() if count > 0}
    policy = {str(key): float(value) for key, value in teacher["prefilter_aggregate_policy"].items()}
    others = [value for action, value in policy.items() if action != selected]
    return {
        "policy": policy,
        "q": q,
        "schedule_tops": schedule_tops,
        "schedule_agreement": len(set(schedule_tops.values())) == 1,
        "selected_schedule_agreement": all(value == selected for value in schedule_tops.values()),
        "world_selected_agreement": sum(value == selected for value in world_tops) / len(world_tops),
        "selected_mass": policy.get(selected, 0.0),
        "production_mass": policy.get(production, 0.0),
        "visit_margin_selected_minus_production": policy.get(selected, 0.0) - policy.get(production, 0.0),
        "q_margin_selected_minus_production": q.get(selected, math.nan) - q.get(production, math.nan),
        "selected_top_gap": policy.get(selected, 0.0) - max(others, default=0.0),
        "entropy": entropy(policy),
    }


def condition_fraction(condition: str) -> float | None:
    value = str(condition).split(" ", 1)[0]
    if value == "0" or "fnt" in str(condition):
        return 0.0
    if "/" in value:
        left, right = value.split("/", 1)
        try:
            return float(left) / float(right)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def public_state(protocol: list[dict], time_ns: int, role: str) -> dict[str, Any]:
    request = received_request(protocol, time_ns)
    roster = (request.get("side") or {}).get("pokemon") or []
    active = next((row for row in roster if row.get("active")), None)
    hazards = {"p1": collections.Counter(), "p2": collections.Counter()}
    revealed: dict[str, dict[str, bool]] = {"p1": {}, "p2": {}}
    boosts = {"p1": collections.Counter(), "p2": collections.Counter()}
    hazard_ids = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
    for row in protocol:
        if int(row.get("time_ns", 0)) >= time_ns or row.get("direction") not in {"received", "reconnect_received"}:
            continue
        for line in str(row.get("message", "")).splitlines():
            parts = line.split("|")
            if len(parts) >= 4 and parts[1] in {"switch", "drag", "replace"} and parts[2][:2] in {"p1", "p2"}:
                side = parts[2][:2]
                revealed[side][parts[2]] = True
                boosts[side].clear()
            elif len(parts) >= 3 and parts[1] == "faint" and parts[2][:2] in {"p1", "p2"}:
                revealed[parts[2][:2]][parts[2]] = False
            elif len(parts) >= 4 and parts[1] in {"-sidestart", "-sideend"}:
                side = parts[2][:2]
                effect = "".join(ch for ch in parts[3].lower() if ch.isalnum()).removeprefix("move")
                if side in hazards and effect in hazard_ids:
                    if parts[1] == "-sidestart":
                        hazards[side][effect] += 1
                    else:
                        hazards[side][effect] = 0
            elif len(parts) >= 5 and parts[1] in {"-boost", "-unboost", "-setboost"} and parts[2][:2] in boosts:
                side = parts[2][:2]
                amount = int(parts[4])
                boosts[side][parts[3]] = amount if parts[1] == "-setboost" else boosts[side][parts[3]] + (amount if parts[1] == "-boost" else -amount)
    opponent = "p2" if role == "p1" else "p1"
    return {
        "own_active_hp_fraction": condition_fraction(active.get("condition", "")) if active else None,
        "own_team_alive": sum("fnt" not in str(row.get("condition", "")) for row in roster),
        "opponent_public_revealed_alive": sum(revealed[opponent].values()),
        "own_hazard_layers": sum(hazards[role].values()),
        "opponent_hazard_layers": sum(hazards[opponent].values()),
        "own_abs_boost": sum(abs(value) for value in boosts[role].values()),
        "opponent_abs_boost": sum(abs(value) for value in boosts[opponent].values()),
    }


def move_metadata(actions: set[str]) -> dict[str, dict]:
    process = subprocess.run(
        ["node", str(ROOT / "experimental/src/scripts/export_cycle41_move_metadata.cjs")],
        input=json.dumps(sorted(action for action in actions if not action.startswith("switch "))),
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    return json.loads(process.stdout)


def semantic(action: str, metadata: dict[str, dict]) -> dict[str, Any]:
    if action.startswith("switch "):
        return {"class": "switch", "tera": False}
    row = metadata[action]
    tera = action.endswith("-tera")
    if row.get("category") != "Status" or float(row.get("basePower") or 0) > 0:
        kind = "attack"
    elif any(row.get(key) for key in ("boosts", "selfBoosts", "sideCondition", "slotCondition", "weather", "terrain", "pseudoWeather")):
        kind = "setup"
    else:
        kind = "status"
    return {"class": kind, "tera": tera, "move_target": row.get("target")}


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_records(records: list[dict]) -> dict[str, Any]:
    overrides = [row for row in records if row["overridden"]]
    passes = [row for row in records if not row["overridden"]]
    def cohort(rows_: list[dict]) -> dict[str, Any]:
        return {
            "decisions": len(rows_),
            "in_candidate_win_games": sum(row["candidate_game_win"] for row in rows_),
            "in_candidate_loss_games": sum(not row["candidate_game_win"] for row in rows_),
            "selected_classes": dict(collections.Counter(row["selected_semantic"]["class"] for row in rows_)),
            "selected_tera": sum(row["selected_semantic"]["tera"] for row in rows_),
            "mean_selected_r1_probability": mean([row["r1_selected_probability"] for row in rows_]),
            "mean_selected_r1_rank": mean([float(row["r1_selected_rank"]) for row in rows_ if row["r1_selected_rank"] is not None]),
            "mean_production_confidence_ratio": mean([row["production_confidence_ratio"] for row in rows_]),
            "mean_production_world_disagreement": mean([row["production_world_disagreement"] for row in rows_]),
            "mean_teacher_world_disagreement": mean([1.0 - row["teacher_world_selected_agreement"] for row in rows_]),
        }
    return {"all": cohort(records), "overrides": cohort(overrides), "pass_through": cohort(passes)}


def numeric_summary(rows_: list[dict], fields: list[str]) -> dict[str, float | None]:
    output = {}
    for field in fields:
        values = [
            float(row[field]) for row in rows_
            if row.get(field) is not None and math.isfinite(float(row[field]))
        ]
        output[field] = mean(values)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    result = json.loads((C40 / "h2h-result.json").read_text())
    games = {row["battle_tag"]: row for row in result["games"]}
    witness = json.loads((C40 / "REGISTRATION_CONSUMPTION.json").read_text())
    registration = {row["username"]: row for row in witness["registrations"]}
    candidates = []
    actions: set[str] = set()
    for path in sorted((C40 / "h2h-logs").glob("*.search.jsonl")):
        file_rows = rows(path)
        if not any((row.get("choice_override") or {}).get("terminal_mcts_teacher") for row in file_rows):
            continue
        external = path.name.removesuffix(".search.jsonl")
        protocol_path = path.with_name(external + ".protocol.jsonl")
        protocol = rows(protocol_path)
        role = player_roles(protocol)[external]
        if registration[external]["side"] != role:
            raise RuntimeError("registration/public role mismatch")
        for row in file_rows:
            teacher = (row["choice_override"] or {})["terminal_mcts_teacher"]
            production = str(teacher["production_action"])
            selected = str(teacher["selected_action"])
            actions.update((production, selected))
            candidates.append((path, protocol, role, row, teacher, production, selected))
    metadata = move_metadata(actions)
    records = []
    for path, protocol, role, row, teacher, production, selected in candidates:
        context = row["context"]
        game = games[context["tag"]]
        priors = {str(action): float(value) for action, value in row.get("player_priors") or []}
        prod = aggregate_production(row.get("samples") or [])
        teacher_metrics = aggregate_teacher(teacher, selected, production)
        prod_policy = prod["policy"]
        prod_max = max(prod_policy.values(), default=0.0)
        prod_others = [value for action, value in prod_policy.items() if action != production]
        record = {
            "battle_tag": context["tag"],
            "game_index": game["game_index"],
            "pair_id": game["pair_id"],
            "pair_index": game["pair_index"],
            "pair_leg": game["pair_leg"],
            "candidate_public_role": role,
            "candidate_game_win": game["winner"] == "agent_a",
            "battle_turn": context.get("battle_turn"),
            "turn_bucket": "early_1_10" if context.get("battle_turn", 0) <= 10 else "middle_11_25" if context.get("battle_turn", 0) <= 25 else "late_26_plus",
            "decision_index": context.get("decision_idx"),
            "production_action": production,
            "selected_action": selected,
            "overridden": teacher.get("decision") == "override",
            "production_semantic": semantic(production, metadata),
            "selected_semantic": semantic(selected, metadata),
            "r1_selected_probability": priors.get(selected, 0.0),
            "r1_production_probability": priors.get(production, 0.0),
            "r1_selected_rank": rank(priors, selected),
            "r1_production_rank": rank(priors, production),
            "r1_probability_gap_selected_minus_production": priors.get(selected, 0.0) - priors.get(production, 0.0),
            "production_policy": prod_policy,
            "production_q": prod["q"],
            "production_top_action": top(prod_policy),
            "production_action_mass": prod_policy.get(production, 0.0),
            "production_confidence_ratio": prod_policy.get(production, 0.0) / prod_max if prod_max else 0.0,
            "production_action_top_gap": prod_policy.get(production, 0.0) - max(prod_others, default=0.0),
            "production_entropy": entropy(prod_policy),
            "production_world_disagreement": 1.0 - sum(value == production for value in prod["world_tops"]) / len(prod["world_tops"]),
            "teacher_policy": teacher_metrics["policy"],
            "teacher_q": teacher_metrics["q"],
            "teacher_schedule_tops": teacher_metrics["schedule_tops"],
            "teacher_schedule_agreement": teacher_metrics["schedule_agreement"],
            "teacher_selected_schedule_agreement": teacher_metrics["selected_schedule_agreement"],
            "teacher_world_selected_agreement": teacher_metrics["world_selected_agreement"],
            "teacher_selected_mass": teacher_metrics["selected_mass"],
            "teacher_production_mass": teacher_metrics["production_mass"],
            "teacher_visit_margin_selected_minus_production": teacher_metrics["visit_margin_selected_minus_production"],
            "teacher_q_margin_selected_minus_production": teacher_metrics["q_margin_selected_minus_production"],
            "teacher_selected_top_gap": teacher_metrics["selected_top_gap"],
            "teacher_entropy": teacher_metrics["entropy"],
            "teacher_elapsed_ms": teacher.get("elapsed_ms"),
            "observable_state": public_state(protocol, int(row["time_ns"]), role),
        }
        records.append(record)
    if len(records) != 575 or sum(row["overridden"] for row in records) != 165:
        raise RuntimeError("Cycle41 decision denominator changed")
    decisions_path = OUT / "decisions.jsonl"
    decisions_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records))

    by_game = []
    for game_index, group in sorted(((key, list(value)) for key, value in __import__("itertools").groupby(sorted(records, key=lambda row: row["game_index"]), key=lambda row: row["game_index"])), key=lambda row: row[0]):
        by_game.append({
            "game_index": game_index,
            "pair_index": group[0]["pair_index"],
            "pair_leg": group[0]["pair_leg"],
            "candidate_win": group[0]["candidate_game_win"],
            "decisions": len(group),
            "overrides": sum(row["overridden"] for row in group),
            "switch_overrides": sum(row["overridden"] and row["selected_semantic"]["class"] == "switch" for row in group),
            "low_r1_overrides": sum(row["overridden"] and (row["r1_selected_probability"] < 0.05 or (row["r1_selected_rank"] or 99) >= 4) for row in group),
            "high_production_confidence_overrides": sum(row["overridden"] and row["production_confidence_ratio"] >= 0.9 for row in group),
        })
    by_pair = []
    for pair_index in range(1, 11):
        games_ = [row for row in by_game if row["pair_index"] == pair_index]
        by_pair.append({
            "pair_index": pair_index,
            "candidate_wins": sum(row["candidate_win"] for row in games_),
            "leg_decisions": [row["decisions"] for row in games_],
            "leg_overrides": [row["overrides"] for row in games_],
            "override_count_divergence": abs(games_[0]["overrides"] - games_[1]["overrides"]),
        })
    override_rows = [row for row in records if row["overridden"]]
    loss_overrides = [row for row in override_rows if not row["candidate_game_win"]]
    win_overrides = [row for row in override_rows if row["candidate_game_win"]]
    def rate(rows_: list[dict], predicate) -> float:
        return sum(predicate(row) for row in rows_) / len(rows_) if rows_ else math.nan
    game_win_rows = [row for row in by_game if row["candidate_win"]]
    game_loss_rows = [row for row in by_game if not row["candidate_win"]]
    transition_matrix = collections.Counter(
        f"{row['production_semantic']['class']}->{row['selected_semantic']['class']}"
        for row in override_rows
    )
    selected_actions = collections.Counter(row["selected_action"] for row in records)
    production_actions = collections.Counter(row["production_action"] for row in records)
    turn_buckets = {}
    for bucket in ("early_1_10", "middle_11_25", "late_26_plus"):
        subset = [row for row in records if row["turn_bucket"] == bucket]
        subset_overrides = [row for row in subset if row["overridden"]]
        turn_buckets[bucket] = {
            "decisions": len(subset),
            "overrides": len(subset_overrides),
            "override_rate": len(subset_overrides) / len(subset) if subset else None,
            "override_decisions_in_loss_games": sum(not row["candidate_game_win"] for row in subset_overrides),
            "override_decisions_in_win_games": sum(row["candidate_game_win"] for row in subset_overrides),
        }
    diagnostic_fields = [
        "r1_selected_probability", "r1_production_probability",
        "r1_probability_gap_selected_minus_production", "production_action_mass",
        "production_confidence_ratio", "production_action_top_gap",
        "production_entropy", "production_world_disagreement",
        "teacher_selected_mass", "teacher_production_mass",
        "teacher_visit_margin_selected_minus_production",
        "teacher_q_margin_selected_minus_production", "teacher_selected_top_gap",
        "teacher_entropy", "teacher_world_selected_agreement", "teacher_elapsed_ms",
    ]
    override_search_metrics = {
        "win_game_overrides": numeric_summary(win_overrides, diagnostic_fields),
        "loss_game_overrides": numeric_summary(loss_overrides, diagnostic_fields),
    }
    observable_fields = [
        "own_active_hp_fraction", "own_team_alive", "opponent_public_revealed_alive",
        "own_hazard_layers", "opponent_hazard_layers", "own_abs_boost",
        "opponent_abs_boost",
    ]
    observable_root_metrics = {
        label: numeric_summary([row["observable_state"] for row in subset], observable_fields)
        for label, subset in (
            ("win_game_overrides", win_overrides),
            ("loss_game_overrides", loss_overrides),
        )
    }
    report = {
        "schema": "metagross-cycle41-development-diagnosis/v1",
        "status": "complete_development_only",
        "methodology": {
            "outcome_open": True,
            "decision_to_game_outcome_is_confounded": True,
            "opened_cycle41_may_form_hypotheses_but_never_confirm_them": True,
            "no_action_counterfactual_hp_team_hazard_tempo_deltas_logged": True,
            "observable_root_state_features_reported_instead": True,
        },
        "counts": {"games": 20, "pairs": 10, "decisions": len(records), "overrides": len(override_rows), "pass_through": len(records) - len(override_rows)},
        "decision_cohorts": summarize_records(records),
        "candidate_vs_production_selector": {
            "selected_semantic_distribution": dict(collections.Counter(row["selected_semantic"]["class"] for row in records)),
            "production_semantic_distribution": dict(collections.Counter(row["production_semantic"]["class"] for row in records)),
            "override_transition_matrix": dict(transition_matrix),
            "top_selected_actions": selected_actions.most_common(20),
            "top_production_actions": production_actions.most_common(20),
            "candidate_tera_decisions": sum(row["selected_semantic"]["tera"] for row in records),
            "production_tera_decisions": sum(row["production_semantic"]["tera"] for row in records),
        },
        "turn_buckets": turn_buckets,
        "override_search_metrics_confounded": override_search_metrics,
        "observable_root_metrics_confounded": observable_root_metrics,
        "override_exploration": {
            "win_game_override_decisions": len(win_overrides),
            "loss_game_override_decisions": len(loss_overrides),
            "switch_rate_win_games": rate(win_overrides, lambda row: row["selected_semantic"]["class"] == "switch"),
            "switch_rate_loss_games": rate(loss_overrides, lambda row: row["selected_semantic"]["class"] == "switch"),
            "low_r1_rate_win_games": rate(win_overrides, lambda row: row["r1_selected_probability"] < 0.05 or (row["r1_selected_rank"] or 99) >= 4),
            "low_r1_rate_loss_games": rate(loss_overrides, lambda row: row["r1_selected_probability"] < 0.05 or (row["r1_selected_rank"] or 99) >= 4),
            "high_production_confidence_rate_win_games": rate(win_overrides, lambda row: row["production_confidence_ratio"] >= 0.9),
            "high_production_confidence_rate_loss_games": rate(loss_overrides, lambda row: row["production_confidence_ratio"] >= 0.9),
            "schedule_agreement_rate_win_games": rate(win_overrides, lambda row: row["teacher_schedule_agreement"]),
            "schedule_agreement_rate_loss_games": rate(loss_overrides, lambda row: row["teacher_schedule_agreement"]),
            "thresholds_are_exploratory_not_gates": {"low_r1": "probability<0.05 OR rank>=4", "high_production_confidence": "production action mass / max mass >=0.9"},
        },
        "game_level_confounded": {
            "winning_games": len(game_win_rows),
            "losing_games": len(game_loss_rows),
            "mean_overrides_winning_games": mean([float(row["overrides"]) for row in game_win_rows]),
            "mean_overrides_losing_games": mean([float(row["overrides"]) for row in game_loss_rows]),
            "decision_weighted_override_rate_winning_games": len(win_overrides) / sum(row["decisions"] for row in game_win_rows),
            "decision_weighted_override_rate_losing_games": len(loss_overrides) / sum(row["decisions"] for row in game_loss_rows),
            "mean_per_game_override_fraction_winning_games": mean([row["overrides"] / row["decisions"] for row in game_win_rows]),
            "mean_per_game_override_fraction_losing_games": mean([row["overrides"] / row["decisions"] for row in game_loss_rows]),
            "mean_low_r1_overrides_winning_games": mean([float(row["low_r1_overrides"]) for row in game_win_rows]),
            "mean_low_r1_overrides_losing_games": mean([float(row["low_r1_overrides"]) for row in game_loss_rows]),
            "mean_switch_overrides_winning_games": mean([float(row["switch_overrides"]) for row in game_win_rows]),
            "mean_switch_overrides_losing_games": mean([float(row["switch_overrides"]) for row in game_loss_rows]),
            "mean_high_production_confidence_overrides_winning_games": mean([float(row["high_production_confidence_overrides"]) for row in game_win_rows]),
            "mean_high_production_confidence_overrides_losing_games": mean([float(row["high_production_confidence_overrides"]) for row in game_loss_rows]),
        },
        "per_game": by_game,
        "per_pair": by_pair,
        "cycle17_exact_overlap": {
            "count": 0,
            "reason": "Cycle17 stores model-information fingerprints for historical roots while live Cycle41 stores request/protocol identities and per-world engine hashes; no shared exact root identifier or rematerialized byte-equality witness exists. Action-set similarity is not exact-root overlap.",
            "cycle17_global_context_only": {
                "equal_8192_difference_from_production_exact": 0.375,
                "equal_8192_schedule_half_top1_agreement": 0.9,
                "r1_20000_difference_from_production_exact": 0.2,
                "r1_20000_schedule_half_top1_agreement": 0.95,
            },
        },
        "hypothesis_ranking": [
            {"rank": 1, "id": "C", "hypothesis": "one-deviation outcome-grounded attribution with the same production continuation", "reason": "Only option that can establish whether a local override improves long-horizon terminal outcome instead of the same hand evaluator."},
            {"rank": 2, "id": "A", "hypothesis": "retain the R1 root prior and admit high-budget search only through a narrow correction gate", "reason": "Preserves the demonstrated production anchor; should be trained/frozen only from independently proven C-style deviations."},
            {"rank": 3, "id": "B", "hypothesis": "mixture/root-prior PUCT sweep", "reason": "Cheap opened-data ablation can test whether equal-prior erased useful policy information, but offline hand-Q is self-referential and any strength claim still needs fresh H2H."},
            {"rank": 4, "id": "D", "hypothesis": "search-native interior policy trained from current targets", "reason": "Rejected for now: the candidate teacher failed its prospective strength gate, so distilling its targets has no independent improvement justification."},
        ],
        "source_hashes": {
            "cycle41_result": sha(C41 / "RESULT_REPORT.json"),
            "cycle40_raw_result": sha(C40 / "h2h-result.json"),
            "decisions_jsonl": sha(decisions_path),
            "cycle17_report": sha(ROOT / "experimental/runs/search_native_v2_cycle17_teacher_stability_20260815/measurement/REPORT.json"),
        },
        "sealed93_rows_read": 0,
        "gpu_cloud_paid_cost_usd": 0,
    }
    (OUT / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "report": str((OUT / "REPORT.json").resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
