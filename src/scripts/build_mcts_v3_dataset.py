#!/usr/bin/env python3
"""Build a schema-v3 MCTS distillation dataset by joining Foul Play decision
logs against prior-server observation dumps.

Schema-v3 contract (docs/mcts_policy_distillation.md):
  - the prior server dumps one JSONL row per served /priors decision:
    {schema: 3, tag, decision_idx, battle_turn, username, text_tokens,
     numbers, illegal_actions, name_table, probs}
  - Foul Play decision rows record the echoed join key `prior_decision_idx`
    (schema version 3) plus raw `mcts_visits` keyed by poke-engine move
    strings.
  - This builder joins on (tag, username, decision_idx) and maps raw visit
    strings through the SERVER's name_table. No replay parsing, and no
    FP-side canonical index arithmetic, is used to build targets.

Fail-closed per battle POV: any decision in a (tag, username) group that
cannot be joined, mapped, or validated rejects the whole group, with the
reason recorded in the report.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

N_ACTIONS = 13
MASS_TOLERANCE = 1e-4


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def normalize_tag(tag: str) -> str:
    tag = str(tag or "").strip()
    if tag.startswith("battle-"):
        tag = tag[len("battle-"):]
    return tag


class GroupRejected(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def read_jsonl(path: Path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            yield line_no, json.loads(line)
        except json.JSONDecodeError:
            yield line_no, None


def load_dumps(paths: list[Path]) -> tuple[dict, Counter]:
    """Return {(tag, username, decision_idx): row} and stats. Duplicate keys
    are fatal for the affected group (recorded and both dropped)."""
    dumps: dict[tuple, dict] = {}
    duplicates: set[tuple] = set()
    stats: Counter = Counter()
    for path in paths:
        for line_no, row in read_jsonl(path):
            if row is None:
                stats["dump_invalid_json"] += 1
                continue
            if row.get("schema") != 3:
                stats["dump_wrong_schema"] += 1
                continue
            key = (
                normalize_tag(row.get("tag")),
                str(row.get("username")),
                row.get("decision_idx"),
            )
            if not key[0] or not isinstance(key[2], int):
                stats["dump_bad_key"] += 1
                continue
            if key in dumps:
                duplicates.add(key)
                stats["dump_duplicate_key"] += 1
                continue
            dumps[key] = row
            stats["dump_rows"] += 1
    for key in duplicates:
        dumps.pop(key, None)
    return dumps, stats


def load_decisions(paths: list[Path]) -> tuple[dict, dict, Counter]:
    """Return ({(tag, username): [rows]}, {(tag, username): label}, stats)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    labels: dict[tuple, int] = {}
    stats: Counter = Counter()
    for path in paths:
        for line_no, row in read_jsonl(path):
            if row is None:
                stats["decision_invalid_json"] += 1
                continue
            record_type = row.get("record_type")
            if record_type == "battle_result":
                tag = normalize_tag(row.get("battle_tag"))
                username = str(row.get("username"))
                if tag and row.get("label") in (0, 1):
                    labels[(tag, username)] = int(row["label"])
                continue
            if record_type != "decision":
                continue
            tag = normalize_tag(row.get("battle_tag"))
            username = str(row.get("username"))
            if not tag:
                stats["decision_missing_tag"] += 1
                continue
            groups[(tag, username)].append(row)
            stats["decision_rows"] += 1
    return dict(groups), labels, stats


def map_move_string(move: str, name_table: dict[str, int]) -> tuple[int | None, str]:
    """Map a poke-engine move string to a canonical action index using the
    server's name table. Returns (index, how) where how is
    'exact'/'norm'/'form'. Raises GroupRejected on ambiguity.

    The 'form' tier handles regional/alternate forms of BENCHED mons: the
    live Metamon tracker names them by base species (request ident) until
    they are revealed, while poke-engine uses the full form name
    (e.g. server 'switch weezing' vs FP 'switch weezinggalar'). Showdown's
    species clause makes the base-name prefix relation unique on a team;
    any ambiguity rejects the group."""
    if move in name_table:
        return int(name_table[move]), "exact"
    normalized = norm(move)
    candidates = {
        int(idx) for name, idx in name_table.items() if norm(name) == normalized
    }
    if len(candidates) == 1:
        return candidates.pop(), "norm"
    if len(candidates) > 1:
        raise GroupRejected("ambiguous_move_string", move)
    if move.startswith("switch "):
        query = norm(move[len("switch "):])
        form_candidates = {
            int(idx)
            for name, idx in name_table.items()
            if name.startswith("switch ")
            and query
            and (lambda k: k and (query.startswith(k) or k.startswith(query)))(
                norm(name[len("switch "):])
            )
        }
        if len(form_candidates) == 1:
            return form_candidates.pop(), "form"
        if len(form_candidates) > 1:
            raise GroupRejected("ambiguous_move_string", move)
    return None, "unmapped"


def build_group(
    key: tuple,
    rows: list[dict],
    dumps: dict,
    label: int | None,
    match_stats: Counter,
) -> list[dict]:
    tag, username = key
    for row in rows:
        if row.get("mcts_schema_version") != 3:
            raise GroupRejected("wrong_schema_version", str(row.get("mcts_schema_version")))
        if not isinstance(row.get("prior_decision_idx"), int):
            raise GroupRejected("missing_prior_decision_idx", f"turn={row.get('turn')}")

    idxs = [row["prior_decision_idx"] for row in rows]
    if len(set(idxs)) != len(idxs):
        raise GroupRejected("duplicate_decision_idx")
    if set(idxs) != set(range(len(idxs))):
        raise GroupRejected(
            "discontinuous_decision_idxs",
            f"got {sorted(idxs)[:5]}..max={max(idxs)} n={len(idxs)}",
        )

    out = []
    for row in sorted(rows, key=lambda r: r["prior_decision_idx"]):
        decision_idx = row["prior_decision_idx"]
        dump = dumps.get((tag, username, decision_idx))
        if dump is None:
            raise GroupRejected("missing_dump_row", f"decision_idx={decision_idx}")

        illegal = dump.get("illegal_actions")
        name_table = dump.get("name_table")
        if dump.get("mask_fallback"):
            # Legality validation is vacuous for these decisions; count them
            # so the collection report can quantify (and gate on) their rate.
            match_stats["mask_fallback_decisions"] += 1
        if not isinstance(illegal, list) or len(illegal) != N_ACTIONS:
            raise GroupRejected("bad_dump_illegal_actions", f"decision_idx={decision_idx}")
        if not isinstance(name_table, dict) or not name_table:
            raise GroupRejected("bad_dump_name_table", f"decision_idx={decision_idx}")
        if any(not isinstance(v, int) or not (0 <= v < N_ACTIONS) for v in name_table.values()):
            raise GroupRejected("bad_dump_name_table", f"decision_idx={decision_idx}")

        visits = row.get("mcts_visits")
        if not isinstance(visits, dict) or not visits:
            raise GroupRejected("missing_mcts_visits", f"decision_idx={decision_idx}")
        mass = sum(float(v) for v in visits.values())
        if not math.isfinite(mass) or abs(mass - 1.0) > MASS_TOLERANCE:
            raise GroupRejected("bad_visit_mass", f"decision_idx={decision_idx} mass={mass}")

        target = [0.0] * N_ACTIONS
        for move, weight in visits.items():
            weight = float(weight)
            if weight <= 0.0:
                continue
            idx, how = map_move_string(str(move), name_table)
            if idx is None:
                raise GroupRejected("unmappable_visit_string", f"{move!r} idx={decision_idx}")
            if illegal[idx]:
                raise GroupRejected(
                    "visit_mass_on_illegal_action", f"{move!r}->{idx} idx={decision_idx}"
                )
            match_stats[f"visit_match_{how}"] += 1
            target[idx] += weight

        total = sum(target)
        if abs(total - 1.0) > MASS_TOLERANCE:
            raise GroupRejected("target_mass_mismatch", f"idx={decision_idx} total={total}")
        target = [t / total for t in target]

        selected = row.get("selected_action")
        if not selected:
            raise GroupRejected("missing_selected_action", f"decision_idx={decision_idx}")
        selected_idx, how = map_move_string(str(selected), name_table)
        if selected_idx is None:
            raise GroupRejected("unmappable_selected_action", f"{selected!r} idx={decision_idx}")
        if illegal[selected_idx]:
            raise GroupRejected(
                "selected_action_illegal", f"{selected!r}->{selected_idx} idx={decision_idx}"
            )
        if target[selected_idx] <= 0.0:
            raise GroupRejected(
                "selected_action_zero_target", f"{selected!r}->{selected_idx} idx={decision_idx}"
            )
        match_stats[f"selected_match_{how}"] += 1

        # Diagnostic only: agreement with the legacy FP-side canonical mapping.
        fp_target = row.get("mcts_visit_target_13")
        if isinstance(fp_target, list) and len(fp_target) == N_ACTIONS:
            match_stats["fp_target_present"] += 1
            fp_top = max(range(N_ACTIONS), key=lambda i: fp_target[i])
            v3_top = max(range(N_ACTIONS), key=lambda i: target[i])
            if fp_top == v3_top:
                match_stats["fp_target_top1_agree"] += 1

        out.append(
            {
                "schema": 3,
                "battle_tag": tag,
                "username": username,
                "decision_idx": decision_idx,
                "battle_turn": dump.get("battle_turn"),
                "text_tokens": dump["text_tokens"],
                "numbers": dump["numbers"],
                "illegal_actions": [bool(x) for x in illegal],
                "visit_target_13": target,
                "selected_action_index": selected_idx,
                "selected_action": str(selected),
                "policy_probs": dump.get("probs"),
                "label": label,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-log", action="append", required=True, type=Path,
                        help="Decision JSONL (repeatable).")
    parser.add_argument("--prior-dump", action="append", required=True, type=Path,
                        help="Prior-server dump JSONL (repeatable).")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--require-labels", action="store_true",
                        help="Reject groups without a battle_result label.")
    parser.add_argument("--min-admission-rate", type=float, default=None,
                        help="Exit nonzero if admitted-group rate falls below this.")
    args = parser.parse_args()

    dumps, dump_stats = load_dumps(list(args.prior_dump))
    groups, labels, decision_stats = load_decisions(list(args.decision_log))

    match_stats: Counter = Counter()
    rejection_reasons: Counter = Counter()
    rejected_groups: list[dict] = []
    admitted = 0
    written = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for key in sorted(groups):
            rows = groups[key]
            label = labels.get(key)
            try:
                if args.require_labels and label is None:
                    raise GroupRejected("missing_battle_result_label")
                built = build_group(key, rows, dumps, label, match_stats)
            except GroupRejected as exc:
                rejection_reasons[exc.reason] += 1
                rejected_groups.append(
                    {"battle_tag": key[0], "username": key[1],
                     "reason": exc.reason, "detail": exc.detail}
                )
                continue
            for record in built:
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1
            admitted += 1

    total_groups = len(groups)
    admission_rate = admitted / total_groups if total_groups else 0.0
    report = {
        "ok": True,
        "mode": "schema_v3_join",
        "groups_total": total_groups,
        "groups_admitted": admitted,
        "groups_rejected": total_groups - admitted,
        "admission_rate": admission_rate,
        "targets_written": written,
        "dump_stats": dict(dump_stats),
        "decision_stats": dict(decision_stats),
        "match_stats": dict(match_stats),
        "rejection_reasons": dict(rejection_reasons),
        "rejected_groups": rejected_groups[:200],
        "output": str(args.output),
    }
    if args.min_admission_rate is not None and admission_rate < args.min_admission_rate:
        report["ok"] = False
        report["error"] = (
            f"admission rate {admission_rate:.4f} below required "
            f"{args.min_admission_rate:.4f}"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "ok", "groups_total", "groups_admitted", "targets_written",
        "rejection_reasons")}, indent=2))
    if not report["ok"]:
        raise SystemExit(report["error"])


if __name__ == "__main__":
    main()
