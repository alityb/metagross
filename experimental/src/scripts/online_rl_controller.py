#!/usr/bin/env python3
"""Run resumable batched online-RL generations without manual phase changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SRC_ROOT.parents[1]
SCRIPTS = SRC_ROOT / "scripts"
R1 = {
    "id": "frozen_r1",
    "run_name": "randbats_exit_r1",
    "checkpoint": 5,
    "sha256": "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93",
}


class ControllerError(RuntimeError):
    pass


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0 or not 0 <= wins <= games:
        raise ValueError("invalid win count")
    rate = wins / games
    denominator = 1.0 + z * z / games
    center = (rate + z * z / (2.0 * games)) / denominator
    half = z * math.sqrt((rate * (1.0 - rate) + z * z / (4.0 * games)) / games) / denominator
    return center - half, center + half


def arena_decision(
    wins: int,
    games: int,
    *,
    lineage_floor: float,
    promotion_min_games: int,
) -> dict[str, Any]:
    low, high = wilson_interval(wins, games)
    rate = wins / games
    return {
        "wins": wins,
        "losses": games - wins,
        "games": games,
        "winrate": rate,
        "wilson95": [low, high],
        "advance_lineage": rate >= lineage_floor,
        "promote_accepted": games >= promotion_min_games and low > 0.5,
    }


def validate_config(config: dict[str, Any]) -> None:
    required_positive = (
        "generations", "collection_games", "arena_games", "learner_steps",
        "batch_size", "promotion_min_games",
    )
    for key in required_positive:
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ControllerError(f"{key} must be a positive integer")
    if config["arena_games"] < config["promotion_min_games"]:
        raise ControllerError("arena_games must be at least promotion_min_games")
    for key in ("workers", "chunk_games", "collector_torch_threads"):
        if not isinstance(config.get(key, 1), int) or config.get(key, 1) <= 0:
            raise ControllerError(f"{key} must be a positive integer")
    if config.get("retain_fresh_generations", "current") not in {"current", "all"}:
        raise ControllerError("retain_fresh_generations must be current or all")
    opponents = config.get("collection_opponents")
    if opponents is not None:
        if not isinstance(opponents, list) or not opponents:
            raise ControllerError("collection_opponents must be a non-empty list")
        total_weight = 0.0
        for opponent in opponents:
            if not isinstance(opponent, dict) or (opponent.get("source") == "current") == ("snapshot" in opponent):
                raise ControllerError("each collection opponent must select current or one snapshot")
            weight = opponent.get("base_weight")
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise ControllerError("collection opponent weights must be positive")
            if "snapshot" in opponent:
                snapshot = opponent["snapshot"]
                if not isinstance(snapshot, dict) or any(
                    key not in snapshot for key in ("id", "run_name", "checkpoint", "sha256")
                ):
                    raise ControllerError("collection opponent snapshot is incomplete")
            total_weight += float(weight)
        if not math.isclose(total_weight, 1.0, abs_tol=1e-9):
            raise ControllerError("collection opponent weights must sum to 1")
    if config.get("collection_backend", "local") not in {"local", "modal"}:
        raise ControllerError("collection_backend must be local or modal")
    if not isinstance(config.get("automatic_promotion", True), bool):
        raise ControllerError("automatic_promotion must be a boolean")


def profile(profile_id: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "local",
        "run_dir": "srcs/models",
        "run_name": snapshot["run_name"],
        "checkpoint": snapshot["checkpoint"],
        "checkpoint_sha256": snapshot["sha256"],
        "base_model": "Kakuna",
        "temperature": 1.0,
        "alias_to": "",
    }


def collection_pool(
    current: dict[str, Any],
    accepted: dict[str, Any],
    configured_opponents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    learner_id = current["id"]
    profiles = {learner_id: profile(learner_id, current)}
    if configured_opponents is None:
        members = [{"id": learner_id, "base_weight": 0.5}]
        if accepted["id"] != learner_id:
            profiles[accepted["id"]] = profile(accepted["id"], accepted)
            members.append({"id": accepted["id"], "base_weight": 0.5})
    else:
        members = []
        for configured in configured_opponents:
            snapshot = current if configured.get("source") == "current" else configured["snapshot"]
            snapshot_id = snapshot["id"]
            candidate_profile = profile(snapshot_id, snapshot)
            if snapshot_id in profiles and profiles[snapshot_id] != candidate_profile:
                raise ControllerError(f"conflicting collection snapshot id: {snapshot_id}")
            profiles[snapshot_id] = candidate_profile
            members.append({"id": snapshot_id, "base_weight": float(configured["base_weight"])})
    return {
        "schema_version": 1,
        "format": "gen9randombattle",
        "profiles": profiles,
        "pfsp": {
            "learner": learner_id,
            "pool": members,
            "target_winrate": [0.4, 0.6],
            "min_pool_weight": 0.05,
        },
    }


def arena_pool(candidate: dict[str, Any], accepted: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format": "gen9randombattle",
        "profiles": {
            candidate["id"]: profile(candidate["id"], candidate),
            accepted["id"]: profile(accepted["id"], accepted),
        },
        "pfsp": {
            "learner": candidate["id"],
            "pool": [{"id": accepted["id"], "base_weight": 1.0}],
            "target_winrate": [0.4, 0.6],
            "min_pool_weight": 0.05,
        },
    }


def score_manifest(path: Path) -> tuple[int, int]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    wins = games = 0
    if manifest.get("failed_shards"):
        raise ControllerError(f"arena contains failed shards: {path}")
    units = manifest.get("chunks", manifest.get("shards", []))
    losses = trajectories = 0
    for shard in units:
        for phase in shard.get("phases", []):
            completed = int(phase.get("completed_battles", 0))
            wins += int(phase.get("learner_wins", 0))
            losses += int(phase.get("learner_losses", 0))
            trajectories += int(phase.get("learner_trajectory_count", 0))
            games += completed
    if (
        manifest.get("collection_kind") != "arena"
        or games != manifest.get("completed_battles")
        or wins != manifest.get("learner_wins")
        or losses != manifest.get("learner_losses")
        or trajectories != manifest.get("learner_trajectory_count")
        or games != wins + losses
        or games != trajectories
    ):
        raise ControllerError(f"arena count mismatch: {path}")
    return wins, games


def admitted_collection_source(generation: int, root: Path) -> dict[str, Any]:
    path = root / "MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("collection_kind") != "fresh" or manifest.get("failed_shards"):
        raise ControllerError(f"fresh source is not an admitted collection: {path}")
    completed = int(manifest.get("completed_battles", -1))
    wins = int(manifest.get("learner_wins", -1))
    losses = int(manifest.get("learner_losses", -1))
    trajectories = int(manifest.get("learner_trajectory_count", -1))
    requested = int(manifest.get("requested_battles", -1))
    units = manifest.get("chunks", manifest.get("shards"))
    if not isinstance(units, list) or not units:
        raise ControllerError(f"fresh source has no collection chunks: {path}")
    derived = {
        "requested_battles": sum(int(unit.get("requested_battles", 0)) for unit in units),
        "completed_battles": 0,
        "learner_wins": 0,
        "learner_losses": 0,
        "learner_trajectory_count": 0,
    }
    for unit in units:
        if unit.get("error"):
            raise ControllerError(f"fresh source contains a failed chunk: {path}")
        for phase in unit.get("phases", []):
            for key in derived.keys() - {"requested_battles"}:
                derived[key] += int(phase.get(key, 0))
    declared = {
        "requested_battles": requested,
        "completed_battles": completed,
        "learner_wins": wins,
        "learner_losses": losses,
        "learner_trajectory_count": trajectories,
    }
    if (
        declared != derived
        or completed != requested
        or completed != wins + losses
        or completed != trajectories
    ):
        raise ControllerError(f"fresh source admission totals do not match: {path}")
    ledger = manifest.get("battle_ledger")
    if not isinstance(ledger, dict):
        raise ControllerError(f"fresh source has no immutable battle ledger: {path}")
    ledger_path = root / str(ledger.get("path", ""))
    if (
        not ledger_path.is_file()
        or ledger.get("records") != completed
        or ledger.get("sha256") != sha256(ledger_path)
        or sum(1 for line in ledger_path.read_text(encoding="utf-8").splitlines() if line) != completed
    ):
        raise ControllerError(f"fresh source battle ledger does not match: {path}")
    return {
        "generation": generation,
        "root": str(root.resolve()),
        "manifest": str(path.resolve()),
        "requested_battles": requested,
        "completed_battles": completed,
        "learner_wins": wins,
        "learner_losses": losses,
        "learner_trajectory_count": trajectories,
    }


class Controller:
    def __init__(self, config: dict[str, Any], run_dir: Path):
        validate_config(config)
        self.config = config
        self.run_dir = run_dir.resolve()
        self.state_path = self.run_dir / "STATE.json"
        self.showdown: subprocess.Popen | None = None
        self.showdown_port = int(config.get("showdown_port", 8000))
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if "fresh_collection_sources" not in self.state:
                retained = [
                    admitted_collection_source(record["generation"], Path(record["collection"]))
                    for record in self.state.get("generations", [])
                ]
                if config.get("retain_fresh_generations", "current") == "current":
                    retained = retained[-1:]
                self.state["fresh_collection_sources"] = retained
                self.save()
        else:
            current = dict(config["initial_current"])
            current.setdefault("id", "online_g1_smoke")
            self.state = {
                "schema_version": 1,
                "status": "READY",
                "next_generation": int(config.get("start_generation", 2)),
                "current": current,
                "accepted": dict(config.get("initial_accepted", R1)),
                "snapshots": [current],
                "generations": [],
                "fresh_collection_sources": [],
            }
            self.save()

    def save(self) -> None:
        atomic_json(self.state_path, self.state)

    def set_status(self, status: str, generation: int) -> None:
        self.state["status"] = status
        self.state["active_generation"] = generation
        self.save()

    def run_command(self, command: list[str]) -> None:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)

    def ensure_dnfcubes(self) -> None:
        result = subprocess.run(
            ["modal", "profile", "current"], cwd=WORKSPACE_ROOT, check=True,
            capture_output=True, text=True,
        )
        if result.stdout.strip() != "dnfcubes":
            raise ControllerError("Modal profile must be dnfcubes")

    def start_showdown(self) -> None:
        uri = f"ws://localhost:{self.showdown_port}/showdown/websocket"
        check_command = ["node", str(SCRIPTS / "check_showdown_ws.mjs"), uri, "2000"]
        if subprocess.run(check_command, capture_output=True).returncode == 0:
            return
        log_path = self.run_dir / "showdown.log"
        log = log_path.open("a", encoding="utf-8")
        self.showdown = subprocess.Popen(
            ["bash", str(SCRIPTS / "start_showdown.sh"), str(self.showdown_port)],
            cwd=WORKSPACE_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.showdown.poll() is not None:
                raise ControllerError(f"Showdown exited; see {log_path}")
            if subprocess.run(check_command, capture_output=True).returncode == 0:
                return
            time.sleep(1)
        raise ControllerError("Showdown did not become ready")

    def stop_showdown(self) -> None:
        if self.showdown is not None and self.showdown.poll() is None:
            self.showdown.terminate()
            try:
                self.showdown.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.showdown.kill()

    def plan_and_collect(
        self, pool: dict[str, Any], games: int, seed: int, output: Path, collection_kind: str
    ) -> None:
        manifest = output / "MANIFEST.json"
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            trajectory_count = sum(1 for path in output.glob("**/*.lz4") if path.is_file())
            if (
                payload.get("completed_battles") == games
                and not payload.get("failed_shards")
                and trajectory_count == payload.get("learner_trajectory_count")
            ):
                return
        output.mkdir(parents=True, exist_ok=True)
        pool_path = output / "pool.json"
        schedule_path = output / "schedule.json"
        atomic_json(pool_path, pool)
        self.run_command(
            [sys.executable, str(SCRIPTS / "pfsp_plan.py"), "--pool", str(pool_path),
             "--battles", str(games), "--seed", str(seed), "--out", str(schedule_path)]
        )
        if self.config.get("collection_backend", "local") == "modal" and collection_kind == "fresh":
            collection_id = f"{self.run_dir.name}-{output.parent.name}-{output.name}"
            self.run_command([
                "modal", "run", str(SCRIPTS / "modal_collect_online_rl.py"),
                "--pool", str(pool_path), "--schedule", str(schedule_path),
                "--collection-id", collection_id,
                "--local-out", str(output),
                "--workers", str(self.config.get("workers", 4)),
                "--chunk-games", str(self.config.get("chunk_games", 25)),
                "--torch-threads", str(self.config.get("collector_torch_threads", 2)),
            ])
        else:
            self.run_command(
                [sys.executable, str(SCRIPTS / "online_rl_generate.py"), "--pool", str(pool_path),
                 "--schedule", str(schedule_path), "--out-dir", str(output),
                 "--showdown-port", str(self.showdown_port),
                  "--workers", str(self.config.get("workers", 1)),
                  "--chunk-games", str(self.config.get("chunk_games", 1)),
                  "--torch-threads", str(self.config.get("collector_torch_threads", 4)),
                  "--collection-kind", collection_kind]
            )

    def train(self, generation: int, current: dict[str, Any], source_manifest: Path) -> dict[str, Any]:
        run_name = f"randbats_online_g{generation}_{self.config['run_tag']}"
        local_manifest = self.run_dir / f"generation_{generation:03d}" / "TRAINING_MANIFEST.json"
        model_dir = WORKSPACE_ROOT / "srcs" / "models" / run_name
        checkpoint = model_dir / "ckpts" / "policy_weights" / "policy_epoch_1.pt"
        if local_manifest.is_file() and checkpoint.is_file():
            return json.loads(local_manifest.read_text(encoding="utf-8"))
        command = [
            "modal", "run", str(SCRIPTS / "modal_train_online_rl.py"),
            "--fresh-source-manifest", str(source_manifest),
            "--reuse-anchor-artifact", self.config["anchor_artifact"],
            "--run-name", run_name,
            "--steps", str(self.config["learner_steps"]),
            "--batch-size", str(self.config["batch_size"]),
            "--base-run-dir", str(WORKSPACE_ROOT / "srcs" / "models" / current["run_name"]),
            "--base-run-name", current["run_name"],
            "--base-checkpoint", str(current["checkpoint"]),
            "--base-checkpoint-sha256", current["sha256"],
        ]
        self.run_command(command)
        remote_root = f"/online_rl/checkpoints/{run_name}"
        local_manifest.parent.mkdir(parents=True, exist_ok=True)
        self.run_command(
            ["modal", "volume", "get", "--force", "metagross-online-rl",
             f"{remote_root}/ONLINE_RL_MANIFEST.json", str(local_manifest)]
        )
        manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
        deployable = manifest.get("deployable", {})
        validation = manifest.get("validation", {})
        if not validation.get("passed") or not deployable.get("checkpoint_sha256"):
            raise ControllerError("training did not produce a validated deployable checkpoint")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.run_command(
            ["modal", "volume", "get", "--force", "metagross-online-rl",
             f"{remote_root}/ckpts/policy_weights/policy_epoch_1.pt", str(checkpoint)]
        )
        if sha256(checkpoint) != deployable["checkpoint_sha256"]:
            raise ControllerError("downloaded checkpoint SHA-256 mismatch")
        return manifest

    def run_generation(self, generation: int) -> None:
        generation_dir = self.run_dir / f"generation_{generation:03d}"
        current = dict(self.state["current"])
        accepted = dict(self.state["accepted"])
        self.set_status("COLLECTING", generation)
        collection = generation_dir / "collection"
        self.plan_and_collect(
            collection_pool(current, accepted, self.config.get("collection_opponents")),
            self.config["collection_games"],
            self.config["seed"] + generation * 10, collection, "fresh",
        )
        source = admitted_collection_source(generation, collection)
        retained = list(self.state.get("fresh_collection_sources", []))
        retained = [item for item in retained if item.get("generation") != generation]
        retained.append(source)
        retained.sort(key=lambda item: item["generation"])
        if self.config.get("retain_fresh_generations", "current") == "current":
            retained = retained[-1:]
        source_manifest = generation_dir / "FRESH_SOURCES.json"
        atomic_json(source_manifest, {
            "schema_version": 1,
            "record_type": "online_rl_fresh_sources",
            "retention": self.config.get("retain_fresh_generations", "current"),
            "sources": retained,
            "totals": {
                key: sum(item[key] for item in retained)
                for key in ("requested_battles", "completed_battles", "learner_wins", "learner_losses", "learner_trajectory_count")
            },
        })
        self.state["fresh_collection_sources"] = retained
        self.save()
        self.set_status("TRAINING", generation)
        training = self.train(generation, current, source_manifest)
        deployable = training["deployable"]
        candidate_suffix = self.config.get("candidate_suffix", "")
        candidate = {
            "id": f"online_g{generation}" + (f"_{candidate_suffix}" if candidate_suffix else ""),
            "run_name": training["run_name"],
            "checkpoint": 1,
            "sha256": deployable["checkpoint_sha256"],
        }
        self.set_status("ARENA", generation)
        arena = generation_dir / "arena_holdout"
        self.plan_and_collect(
            arena_pool(candidate, accepted), self.config["arena_games"],
            self.config["seed"] + generation * 10 + 1, arena, "arena",
        )
        atomic_json(arena / "HOLDOUT.json", {"excluded_from_training": True})
        wins, games = score_manifest(arena / "MANIFEST.json")
        decision = arena_decision(
            wins, games, lineage_floor=self.config["lineage_floor"],
            promotion_min_games=self.config["promotion_min_games"],
        )
        decision["automatic_promotion_enabled"] = self.config.get("automatic_promotion", True)
        decision["promotion_pending_external_gate"] = (
            decision["promote_accepted"] and not decision["automatic_promotion_enabled"]
        )
        record = {
            "generation": generation,
            "base": current,
            "candidate": candidate,
            "collection": str(collection),
            "arena": str(arena),
            "decision": decision,
        }
        self.state["snapshots"].append(candidate)
        if decision["advance_lineage"]:
            self.state["current"] = candidate
        if decision["promote_accepted"] and decision["automatic_promotion_enabled"]:
            self.state["accepted"] = candidate
        self.state["generations"].append(record)
        self.state["next_generation"] = generation + 1
        self.state["status"] = "READY"
        self.save()

    def run(self) -> None:
        self.ensure_dnfcubes()
        self.start_showdown()
        try:
            stop_generation = self.config["start_generation"] + self.config["generations"]
            while self.state["next_generation"] < stop_generation:
                self.run_generation(self.state["next_generation"])
            self.state["status"] = "COMPLETED"
            self.save()
        except Exception:
            self.state["status"] = "FAILED"
            self.save()
            raise
        finally:
            self.stop_showdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    Controller(config, args.run_dir).run()


if __name__ == "__main__":
    main()
