#!/usr/bin/env python3
"""Run the frozen Cycle 10 full-corpus replay coverage and compact index gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from experimental.src.scripts import cycle8_replay_audit as v8
from experimental.src.scripts import cycle9_replay_audit as v9
from experimental.src.scripts.run_cycle8_replay_audit import verify_manifest


SPLIT_DOMAIN = "cycle10-split-20260815"
SCHEMA = "metagross-cycle10-compact-battle-index/v1"
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
ERROR_CLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("source_or_provenance_integrity", (
        "selected raw file hash changed", "selected inputlog hash changed",
        "selected public-log hash changed", "selected start seed hash changed",
        "worktree commit mismatch", "player seed is absent",
        "battle seeds are incomplete", "stored public log lacks one attributable",
        "selected raw record is incomplete", "private team member is malformed",
        "first genuine private team request is absent",
    )),
    ("public_replay_mismatch", ("normalized public replay mismatch",)),
    ("terminal_mismatch", ("terminal result mismatch", "replay did not reach terminal")),
    ("request_linkage", (
        "command has null preceding request", "request was reused",
        "command request index is out of range", "invalid command row",
    )),
    ("illegal_or_unsupported_action", (
        "recorded action is illegal", "request contains no supported action",
        "unsupported recorded command", "move slot", "switch slot",
        "switch details", "empty command",
    )),
    ("team_preview_unsupported", ("team-preview command",)),
    ("revival_blessing_semantics", ("Revival Blessing",)),
    ("private_pov_violation", (
        "Showdown emitted a private error", "POV role mismatch",
        "opposite-side or missing private request", "POV capture is incomplete",
    )),
    ("causal_public_boundary", (
        "invalid public event boundary", "command-time public prefix regressed",
        "public capture lacks chunks", "malformed public chunk",
    )),
    ("wait_request_semantics", ("wait request received", "invalid wait metadata")),
    ("request_schema_semantics", (
        "invalid forceSwitch", "invalid active", "invalid move", "invalid own",
        "request lacks own side", "request exceeds", "invalid trapped",
        "invalid Tera", "request contains no legal",
        "invalid request row", "invalid PP/disable", "invalid own switch row",
    )),
    ("compact_index_invariant", (
        "compact battle contains no states", "compact state contains an invalid",
    )),
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("ascii")
    return hashlib.sha256(value).hexdigest()


def hash_json(value: object) -> str:
    return digest(canonical_json(value))


def source_tree_sha256(paths: Any) -> str:
    """Reproduce Cycle 8's relative-path/size/content source-tree hash."""
    result = hashlib.sha256()
    files = sorted({Path(path).resolve() for path in paths if Path(path).is_file()})
    for path in files:
        relative = path.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
        size = path.stat().st_size
        result.update(relative.encode())
        result.update(b"\0")
        result.update(str(size).encode())
        result.update(b"\0")
        result.update(v8.sha256_path(path).encode())
        result.update(b"\n")
    return result.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def deterministic_gzip_write(path: Path, value: object) -> str:
    payload = (canonical_json(value) + "\n").encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(payload)
    return digest(payload)


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="ascii") as handle:
        return json.load(handle)


def raw_metadata(raw: Mapping[str, Any], commit: str) -> dict[str, str]:
    inputlog = raw.get("inputlog")
    if not isinstance(inputlog, str):
        raise v9.ReplayAuditError("selected raw record is incomplete")
    start_seed = None
    player_seeds = []
    for line in inputlog.splitlines():
        if line.startswith(">start "):
            start_seed = json.loads(line.split(" ", 1)[1]).get("seed")
        elif line.startswith(">player "):
            payload = json.loads(line.split(" ", 2)[2])
            seed = payload.get("seed")
            if not isinstance(seed, str) or not seed:
                raise v9.ReplayAuditError("player seed is absent")
            player_seeds.append(seed)
    if start_seed is None or len(player_seeds) != 2:
        raise v9.ReplayAuditError("battle seeds are incomplete")
    team_pair_preimage = "\0".join(
        ("cycle10-team-pair-v1", commit, *sorted(player_seeds))
    )
    return {
        "start_seed_sha256": digest(str(start_seed)),
        "unordered_player_seed_pair_sha256": digest(team_pair_preimage),
    }


def _first_private_team(pov: Mapping[str, Any]) -> list[dict[str, Any]]:
    for row in pov.get("requests", []):
        request = row.get("request") if isinstance(row, Mapping) else None
        side = request.get("side") if isinstance(request, Mapping) else None
        pokemon = side.get("pokemon") if isinstance(side, Mapping) else None
        if isinstance(pokemon, list) and pokemon:
            result = []
            for member in pokemon:
                if not isinstance(member, Mapping):
                    raise v9.ReplayAuditError("private team member is malformed")
                sanitized = json.loads(canonical_json(member))
                sanitized.pop("ident", None)
                result.append(sanitized)
            return result
    raise v9.ReplayAuditError("first genuine private team request is absent")


def unordered_mechanical_team_pair_sha256(
    p1: Mapping[str, Any], p2: Mapping[str, Any], commit: str,
) -> str:
    team_hashes = sorted((hash_json(_first_private_team(p1)), hash_json(_first_private_team(p2))))
    return digest("\0".join(("cycle10-mechanical-team-pair-v1", commit, *team_hashes)))


def terminal_provenance(log: str) -> dict[str, Any]:
    players: dict[str, str] = {}
    terminals = []
    for line in log.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and fields[1] == "player" and fields[2] in {"p1", "p2"}:
            players[fields[3]] = fields[2]
        elif line.startswith("|win|"):
            winner = line[5:]
            terminals.append({
                "kind": "win", "winner_role": players.get(winner, "unknown"),
                "terminal_line_sha256": digest(line),
            })
        elif line == "|tie":
            terminals.append({
                "kind": "tie", "winner_role": None,
                "terminal_line_sha256": digest(line),
            })
    if len(terminals) != 1 or terminals[0]["winner_role"] == "unknown":
        raise v9.ReplayAuditError("stored public log lacks one attributable terminal result")
    return terminals[0]


def compact_states(derived: Mapping[str, Any], pov: Mapping[str, Any]) -> list[dict[str, Any]]:
    commands = {
        row["preceding_request_index"]: row
        for row in pov["commands"]
        if isinstance(row.get("preceding_request_index"), int)
    }
    compact = []
    for state in derived["states"]:
        command = commands.get(state["request_index"])
        legal_payload = {
            "legal_actions": state["legal_actions"],
            "action_table": state["action_table"],
            "action_semantics": state["action_semantics"],
        }
        row = {
            "role": state["role"],
            "request_index": state["request_index"],
            "actionable": state["actionable"],
            "command_input_index": command.get("input_index") if command else None,
            "public_event_index": state["public_event_index"],
            "model_information_fingerprint_sha256": state["model_information_fingerprint_sha256"],
            "private_request_sha256": hash_json(state["private_request"]),
            "causal_prefix_sha256": hash_json(state["public_prefix"]),
            "legal_action_contract_sha256": hash_json(legal_payload),
            "typed_reveal_ledger_sha256": hash_json(state["typed_reveal_ledger"]),
            "pp_disable_sidecar_sha256": hash_json(state["pp_disable_sidecar"]),
            "observed_command": command.get("command") if command else None,
            "observed_action": state["chosen_action"],
            "observed_action_index": state["chosen_action_index"],
            "observed_action_semantics": state["chosen_action_semantics"],
        }
        required_hashes = (
            "model_information_fingerprint_sha256", "private_request_sha256",
            "causal_prefix_sha256", "legal_action_contract_sha256",
            "typed_reveal_ledger_sha256", "pp_disable_sidecar_sha256",
        )
        if any(len(str(row[key])) != 64 for key in required_hashes):
            raise v9.ReplayAuditError("compact state contains an invalid required hash")
        compact.append(row)
    return compact


def classify_error(exc: BaseException) -> tuple[str, str]:
    detail = digest(str(exc))
    if isinstance(exc, subprocess.CalledProcessError):
        return "showdown_replay_process_error", detail
    message = str(exc)
    for category, prefixes in ERROR_CLASSES:
        if any(message.startswith(prefix) for prefix in prefixes):
            return category, detail
    # Explicit causal-ledger failures are expected semantic abstentions, but
    # their full text is hashed to avoid putting battle identity in summaries.
    if exc.__class__.__name__ == "CausalRevealLedgerError":
        return "causal_ledger_fail_closed", detail
    if isinstance(exc, v9.ReplayAuditError):
        return "unknown_semantic:ReplayAuditError", detail
    return "internal_unclassified:" + exc.__class__.__name__, detail


def verify_source_trees(corpus: list[dict[str, Any]], manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("source_tree_hashes")
    if not isinstance(expected, Mapping):
        raise RuntimeError("manifest lacks source tree hashes")
    directories = {}
    for row in corpus:
        directories.setdefault(row["source"], Path(row["raw_path"]).parent)
    actual_primary = source_tree_sha256(directories["primary"].glob("*.json"))
    actual_external = source_tree_sha256(directories["external"].glob("*.json"))
    if actual_primary != expected.get("raw_human_primary"):
        raise RuntimeError("primary raw source tree hash mismatch")
    if actual_external != expected.get("raw_human_external"):
        raise RuntimeError("external raw source tree hash mismatch")


def process_one(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        panel_index, row, worktree, harness, output_root, diagnostic_root,
        retain_index,
    ) = task
    raw_path = Path(row["raw_path"])
    raw = load_json(raw_path)
    inputlog = raw.get("inputlog")
    stored_log = raw.get("log")
    temp_parent = Path(output_root) / ".tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{panel_index:05d}-", dir=temp_parent))
    capture = temp_dir / "capture"
    relative_index: Path | None = None
    try:
        if v8.sha256_path(raw_path) != row["raw_sha256"]:
            raise v9.ReplayAuditError("selected raw file hash changed")
        if not isinstance(inputlog, str) or not isinstance(stored_log, str):
            raise v9.ReplayAuditError("selected raw record is incomplete")
        if digest(inputlog.encode()) != row["inputlog_sha256"]:
            raise v9.ReplayAuditError("selected inputlog hash changed")
        if digest(stored_log.encode()) != row["public_log_sha256"]:
            raise v9.ReplayAuditError("selected public-log hash changed")
        actual_commit = subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual_commit != row["showdown_commit"]:
            raise v9.ReplayAuditError("worktree commit mismatch")
        subprocess.run(
            [
                "node", str(harness), "--showdown", str(worktree),
                "--input", str(raw_path), "--out-dir", str(capture),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        public = load_json(capture / "public.json")
        if not public.get("replay_ended") or not isinstance(public.get("terminal"), Mapping):
            raise v9.ReplayAuditError("replay did not reach terminal state")
        regenerated = v9.public_lines_from_capture(
            public, inputlog=inputlog, showdown_commit=row["showdown_commit"],
        )
        stored = v9.canonical_public_lines(
            stored_log.splitlines(), inputlog=inputlog,
            showdown_commit=row["showdown_commit"],
        )
        if regenerated != stored:
            raise v9.ReplayAuditError("normalized public replay mismatch")
        terminal = terminal_provenance(stored_log)
        captured_winner = public["terminal"].get("winner", "")
        if terminal["kind"] == "win":
            names = {}
            for line in stored_log.splitlines():
                fields = line.split("|")
                if len(fields) >= 4 and fields[1] == "player":
                    names.setdefault(fields[2], fields[3])
            if captured_winner != names.get(terminal["winner_role"]):
                raise v9.ReplayAuditError("terminal result mismatch")
        elif captured_winner:
            raise v9.ReplayAuditError("terminal result mismatch")

        all_states = []
        role_capture_hashes = {}
        pov_by_role = {}
        for role in ("p1", "p2"):
            pov_path = capture / f"{role}.json"
            pov = load_json(pov_path)
            pov_by_role[role] = pov
            derived = v9.materialize_role(
                battle_id=str(raw["id"]), role=role, public_capture=public,
                pov_capture=pov, inputlog=inputlog,
                showdown_commit=row["showdown_commit"],
            )
            all_states.extend(compact_states(derived, pov))
            role_capture_hashes[role] = v8.sha256_path(pov_path)
        if not all_states:
            raise v9.ReplayAuditError("compact battle contains no states")
        seed_meta = raw_metadata(raw, row["showdown_commit"])
        if seed_meta["start_seed_sha256"] != row["start_seed_sha256"]:
            raise v9.ReplayAuditError("selected start seed hash changed")
        model_public = v8._model_public_prefix(stored)
        canonical_public_sha = hash_json(model_public)
        unordered_team_pair_sha = unordered_mechanical_team_pair_sha256(
            pov_by_role["p1"], pov_by_role["p2"], row["showdown_commit"],
        )
        execution_semantics = {
            "canonical_public_sha256": canonical_public_sha,
            "terminal": terminal,
            "states": [
                {
                    key: state[key] for key in (
                        "role", "request_index", "model_information_fingerprint_sha256",
                        "legal_action_contract_sha256", "observed_action",
                        "observed_action_index", "observed_action_semantics",
                    )
                }
                for state in all_states
            ],
        }
        execution_sha = hash_json(execution_semantics)
        battle = {
            "schema": SCHEMA,
            "source": row["source"],
            "raw_relative_path": row["raw_relative_path"],
            "battle_id": row["battle_id"],
            "raw_sha256": row["raw_sha256"],
            "inputlog_sha256": row["inputlog_sha256"],
            "public_log_sha256": row["public_log_sha256"],
            "showdown_commit": row["showdown_commit"],
            **seed_meta,
            "unordered_mechanical_team_pair_sha256": unordered_team_pair_sha,
            "canonical_public_sha256": canonical_public_sha,
            "canonical_public_provenance_sha256": hash_json(stored),
            "execution_sha256": execution_sha,
            "capture_provenance_sha256": hash_json(role_capture_hashes),
            "terminal_outcome_provenance": terminal,
            "state_count": len(all_states),
            "states": all_states,
        }
        compact_sha = hash_json(battle)
        shutil.rmtree(temp_dir)
        if retain_index:
            relative_index = Path("states") / row["raw_sha256"][:2] / (
                f"{panel_index:05d}-{row['raw_sha256'][:16]}.json.gz"
            )
            deterministic_gzip_write(Path(output_root) / relative_index, battle)
        return {
            "panel_index": panel_index, "status": "pass",
            "battle_id": row["battle_id"], "commit": row["showdown_commit"],
            "state_count": len(all_states), "compact_sha256": compact_sha,
            "canonical_public_sha256": canonical_public_sha,
            "execution_sha256": execution_sha,
            "start_seed_sha256": seed_meta["start_seed_sha256"],
            "unordered_player_seed_pair_sha256": seed_meta["unordered_player_seed_pair_sha256"],
            "unordered_mechanical_team_pair_sha256": unordered_team_pair_sha,
            "relative_index": str(relative_index) if relative_index else None,
        }
    except BaseException as exc:
        failure_class, failure_detail = classify_error(exc)
        if relative_index is not None:
            (Path(output_root) / relative_index).unlink(missing_ok=True)
        destination = Path(diagnostic_root) / f"panel-{panel_index:05d}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        if temp_dir.exists():
            shutil.move(str(temp_dir), destination)
        else:
            destination.mkdir(parents=True, exist_ok=True)
        (destination / "FAILURE.json").write_text(canonical_json({
            "schema": "metagross-cycle10-failure/v1",
            "panel_index": panel_index, "source": row["source"],
            "battle_id_sha256": digest(row["battle_id"]),
            "raw_sha256": row["raw_sha256"], "commit": row["showdown_commit"],
            "failure_class": failure_class,
            "failure_detail_sha256": failure_detail,
        }) + "\n")
        return {
            "panel_index": panel_index, "status": "fail",
            "battle_id": row["battle_id"], "commit": row["showdown_commit"],
            "failure_class": failure_class,
            "failure_detail_sha256": failure_detail,
        }


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def assign_clusters(results: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    passed = [row for row in results if row["status"] == "pass"]
    union = UnionFind(len(passed))
    owner: dict[str, int] = {}
    tokens_by_index = []
    for index, row in enumerate(passed):
        tokens = [
            "public:" + row["canonical_public_sha256"],
            "execution:" + row["execution_sha256"],
            "start:" + row["start_seed_sha256"],
            "seedpair:" + row["unordered_player_seed_pair_sha256"],
            "team:" + row["unordered_mechanical_team_pair_sha256"],
        ]
        tokens_by_index.append(tokens)
        for token in tokens:
            if token in owner:
                union.union(index, owner[token])
            else:
                owner[token] = index
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(passed)):
        members[union.find(index)].append(index)
    cluster_ids = {}
    for root, indices in members.items():
        cluster_ids[root] = min(
            token for index in indices for token in tokens_by_index[index]
        )

    split_counts: Counter[str] = Counter()
    state_split_counts: Counter[str] = Counter()
    master_rows = []
    cluster_split: dict[str, str] = {}
    for index, result in enumerate(passed):
        root = union.find(index)
        cluster = cluster_ids[root]
        residue = int(digest(SPLIT_DOMAIN + "\0" + cluster), 16) % 10
        split = "train" if residue <= 5 else "validation" if residue <= 7 else "test"
        prior = cluster_split.setdefault(cluster, split)
        if prior != split:
            raise RuntimeError("dependency cluster crossed splits")
        result["dependency_cluster_id"] = cluster
        result["split"] = split
        split_counts[split] += 1
        state_split_counts[split] += result["state_count"]
        state_path = output_root / result["relative_index"]
        payload = read_gzip_json(state_path)
        payload["dependency_cluster_id"] = cluster
        payload["split"] = split
        payload["states_inherit_battle_split"] = True
        payload_sha = deterministic_gzip_write(state_path, payload)
        master_rows.append({
            key: result[key] for key in (
                "panel_index", "battle_id", "commit", "state_count",
                "canonical_public_sha256", "execution_sha256",
                "start_seed_sha256", "unordered_player_seed_pair_sha256",
                "unordered_mechanical_team_pair_sha256",
                "relative_index", "dependency_cluster_id", "split",
            )
        } | {"index_payload_sha256": payload_sha})
    master_rows.sort(key=lambda row: row["panel_index"])
    master_path = output_root / "eligible-battles.jsonl.gz"
    with master_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for row in master_rows:
                handle.write((canonical_json(row) + "\n").encode("ascii"))
    duplicate_clusters = sum(len(indices) > 1 for indices in members.values())
    duplicate_battles = sum(len(indices) for indices in members.values() if len(indices) > 1)
    return {
        "clusters": len(members),
        "duplicate_clusters": duplicate_clusters,
        "battles_in_duplicate_clusters": duplicate_battles,
        "battle_split_counts": dict(sorted(split_counts.items())),
        "state_split_counts": dict(sorted(state_split_counts.items())),
        "cross_split_cluster_leakage": 0,
        "master_index_sha256": v8.sha256_path(master_path),
    }


def run_tasks(tasks: list[tuple[Any, ...]], workers: int, label: str) -> list[dict[str, Any]]:
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for count, result in enumerate(executor.map(process_one, tasks, chunksize=1), start=1):
            results.append(result)
            if count % 500 == 0 or count == len(tasks):
                print(json.dumps({
                    "stage": label, "completed": count, "total": len(tasks),
                    "passed": sum(row["status"] == "pass" for row in results),
                    "failed": sum(row["status"] == "fail" for row in results),
                }, sort_keys=True), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--spot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--showdown-repo", type=Path, required=True)
    parser.add_argument("--worktree-map", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers != 8:
        raise ValueError("frozen Cycle 10 worker count is exactly eight")
    manifest = verify_manifest(args.manifest)
    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines() if line]
    spot = [json.loads(line) for line in args.spot.read_text().splitlines() if line]
    positives = [row for row in corpus if row["commit_present"]]
    negatives = [row for row in corpus if not row["commit_present"]]
    if len(corpus) != 20633 or len(positives) != 20629 or len(negatives) != 4 or len(spot) != 256:
        raise ValueError("frozen corpus or spot cardinality changed")
    verify_source_trees(corpus, manifest)
    worktrees = load_json(args.worktree_map)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    diagnostic_root = args.output_dir / "failures"
    negative_passes = 0
    for row in negatives:
        present = subprocess.run(
            ["git", "-C", str(args.showdown_repo), "cat-file", "-e", f"{row['showdown_commit']}^{{commit}}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if not present and v8.sha256_path(Path(row["raw_path"])) == row["raw_sha256"]:
            negative_passes += 1

    main_tasks = [
        (
            index, row, worktrees[row["showdown_commit"]], str(args.harness.resolve()),
            str(args.output_dir), str(diagnostic_root), True,
        )
        for index, row in enumerate(corpus) if row["commit_present"]
    ]
    results = run_tasks(main_tasks, args.workers, "full_corpus")
    by_battle = {row["battle_id"]: row for row in results}

    spot_root = args.output_dir / "determinism-repeat"
    spot_tasks = [
        (
            index, row, worktrees[row["showdown_commit"]], str(args.harness.resolve()),
            str(spot_root), str(args.output_dir / "determinism-failures"), False,
        )
        for index, row in enumerate(spot)
    ]
    spot_results = run_tasks(spot_tasks, args.workers, "determinism_repeat")
    deterministic_pass = 0
    for row, repeated in zip(spot, spot_results, strict=True):
        original = by_battle.get(row["battle_id"])
        if (
            original and original["status"] == "pass"
            and repeated["status"] == "pass"
            and original["compact_sha256"] == repeated["compact_sha256"]
            and original["canonical_public_sha256"] == repeated["canonical_public_sha256"]
            and original["execution_sha256"] == repeated["execution_sha256"]
        ):
            deterministic_pass += 1
    if spot_root.exists():
        shutil.rmtree(spot_root)

    totals = Counter(row["commit"] for row in results)
    passed_counts = Counter(row["commit"] for row in results if row["status"] == "pass")
    by_commit = {
        commit: {
            "total": total, "passed": passed_counts[commit],
            "failed": total - passed_counts[commit],
            "coverage": passed_counts[commit] / total,
            "major": total >= 100,
        }
        for commit, total in sorted(totals.items())
    }
    failures = Counter(
        row["failure_class"] for row in results if row["status"] == "fail"
    )
    failure_details: dict[str, set[str]] = defaultdict(set)
    for row in results:
        if row["status"] == "fail":
            failure_details[row["failure_class"]].add(row["failure_detail_sha256"])
    unknown = sum(
        count for name, count in failures.items()
        if name.startswith(("internal_unclassified:", "unknown_semantic:"))
    )
    passed = sum(row["status"] == "pass" for row in results)
    coverage = passed / len(positives)
    commit_gate = all(
        row["coverage"] >= 0.99 for row in by_commit.values() if row["major"]
    )
    cluster_report = assign_clusters(results, args.output_dir)
    post_run_integrity = "pass"
    post_run_integrity_detail_sha256 = None
    try:
        post_manifest = verify_manifest(args.manifest)
        verify_source_trees(corpus, post_manifest)
    except BaseException as exc:
        post_run_integrity = "fail"
        post_run_integrity_detail_sha256 = digest(
            exc.__class__.__name__ + ":" + str(exc)
        )
    status = "pass" if (
        negative_passes == 4 and coverage >= 0.99 and commit_gate
        and unknown == 0 and deterministic_pass == 256
        and cluster_report["cross_split_cluster_leakage"] == 0
        and post_run_integrity == "pass"
    ) else "fail"
    report = {
        "schema": "metagross-cycle10-full-corpus-report/v1",
        "status": status,
        "corpus_rows": len(corpus), "positive_rows": len(positives),
        "positive_passed": passed, "positive_failed": len(positives) - passed,
        "overall_coverage": coverage,
        "negative_controls_passed": negative_passes,
        "by_commit": by_commit,
        "failure_classes": dict(sorted(failures.items())),
        "failure_detail_hashes_by_class": {
            key: sorted(values) for key, values in sorted(failure_details.items())
        },
        "unknown_failure_count": unknown,
        "determinism_spot_total": 256,
        "determinism_spot_passed": deterministic_pass,
        "indexed_states": sum(row.get("state_count", 0) for row in results),
        "dependency_index": cluster_report,
        "post_run_frozen_integrity": post_run_integrity,
        "post_run_integrity_detail_sha256": post_run_integrity_detail_sha256,
        "workers": args.workers,
        "manifest_sha256": v8.sha256_path(args.manifest),
        "corpus_sha256": v8.sha256_path(args.corpus),
        "spot_sha256": v8.sha256_path(args.spot),
        "teacher_q_visit_fields_opened": 0,
        "training_rows_written": 0,
        "sealed_93_rows_read": 0,
        "cloud_gpu_paid_cost_usd": 0,
    }
    report_path = args.output_dir / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
