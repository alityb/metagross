#!/usr/bin/env python3
"""Verify all local N=100 runtime identities before any game starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

from experimental.src.eval.experiment_manifest import hash_tree
from srcs.metagross.h2h_audit import _sha256


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise ValueError(f"non-object response from {url}")
    return payload


def preflight(
    preregistration_path: Path,
    authorization_path: Path,
    runtime_manifest_path: Path,
    python_runtime_manifest_path: Path,
    showdown_dir: Path,
    pair_directory: Path,
    prior_urls: list[str],
    expected_nonces: list[str],
    remote_preflight_path: Path,
) -> dict[str, object]:
    prereg = json.loads(preregistration_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    python_manifest = json.loads(
        python_runtime_manifest_path.read_text(encoding="utf-8")
    )
    remote = json.loads(remote_preflight_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    file_paths = {
        "package_json": showdown_dir / "package.json",
        "package_lock": showdown_dir / "package-lock.json",
        "launcher": showdown_dir / "pokemon-showdown",
        "ladders_source": showdown_dir / "server" / "ladders.ts",
        "registration_source": showdown_dir / "server" / "eval-pair-registration.ts",
        "ladders_runtime": showdown_dir / "dist" / "server" / "ladders.js",
        "registration_runtime": showdown_dir / "dist" / "server" / "eval-pair-registration.js",
        "config": showdown_dir / "config" / "config.js",
        "team_generator": root / "experimental" / "src" / "scripts" / "generate_mirrored_randbats_pair.cjs",
    }
    observed_files = {name: _sha256(path) for name, path in file_paths.items()}
    node_path = shutil.which("node")
    if node_path is None:
        raise RuntimeError("node executable is unavailable")
    node_executable = Path(node_path).resolve()
    node_identity = json.loads(
        subprocess.check_output(
            [
                str(node_executable),
                "-p",
                "JSON.stringify({node:process.version,v8:process.versions.v8,modules:process.versions.modules,platform:process.platform,arch:process.arch})",
            ],
            text=True,
        )
    )
    node_identity["executable"] = str(node_executable)
    node_identity["executable_sha256"] = _sha256(node_executable)
    prior_health = [_get_json(url.rstrip("/") + "/health") for url in prior_urls]
    expected_checkpoint = prereg["source_identity"]["policy_checkpoint"]
    pair_entries = list(pair_directory.iterdir()) if pair_directory.exists() else []
    frozen_prior_servers = prereg.get("prior_servers", {})
    expected_prior_urls = [
        frozen_prior_servers[mode]["url"] for mode in ("candidate", "comparator")
    ]
    frozen_nonces = [
        frozen_prior_servers[mode]["nonce"] for mode in ("candidate", "comparator")
    ]
    expected_pair_directory = (
        root
        / prereg["execution"]["environment"]["METAGROSS_EVAL_PAIR_DIR"]
    ).resolve()
    expected_showdown_argv = manifest.get("execution_constraints", {}).get("argv")
    gate_paths = [
        prereg.get("artifacts", {}).get(name)
        for name in (
            "operational_gate_request",
            "operational_gate_approval",
            "operational_gate_review",
        )
    ]
    pair_manifest_path = (root / prereg["artifacts"]["pair_manifest"]).resolve()
    pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    frozen_eval_argv = prereg.get("execution", {}).get("evaluation_argv", [])
    fresh_paths = [
        root / prereg["artifacts"][name]
        for name in ("results", "log_directory")
    ] + [
        root / f"{prereg['artifacts']['results']}{suffix}"
        for suffix in (".progress.json", ".progress.jsonl")
    ] + [
        root / "experimental" / "runs" / "search_native_stage2_20260809" / prereg["artifacts"][name]
        for name in ("candidate_prior_dump", "comparator_prior_dump")
    ]
    gates = {
        "execution_explicitly_authorized": (
            authorization.get("status") == "authorized"
            and authorization.get("preregistration_sha256") == _sha256(preregistration_path)
            and authorization.get("local_n100_authorized") is True
            and authorization.get("public_ladder_authorized") is False
        ),
        "frozen_preregistration": prereg.get("status") == "frozen_before_games",
        "runtime_manifest_bound": prereg["source_identity"].get("showdown_runtime_identity") == _sha256(runtime_manifest_path),
        "corrected_showdown_security_mode": (
            expected_showdown_argv
            == prereg.get("execution", {}).get("showdown", {}).get("argv")
            and "--no-security" in expected_showdown_argv
        ),
        "showdown_supervisor_source_exact": _sha256(
            root / "srcs" / "metagross" / "showdown_runtime_server.py"
        ) == prereg.get("source_identity", {}).get("showdown_supervisor.py"),
        "foul_play_launcher_source_exact": _sha256(
            root / "srcs" / "metagross" / "run_foul_play.py"
        ) == prereg.get("source_identity", {}).get("run_foul_play.py"),
        "evaluation_source_exact": _sha256(
            root / "experimental" / "src" / "eval" / "run.py"
        ) == prereg.get("source_identity", {}).get("eval_run.py"),
        "preflight_source_exact": _sha256(Path(__file__).resolve())
        == prereg.get("source_identity", {}).get("local_preflight.py"),
        "inventory_implementation_exact": _sha256(
            root / "experimental" / "src" / "eval" / "experiment_manifest.py"
        ) == prereg.get("source_identity", {}).get("experiment_manifest.py"),
        "pair_manifest_exact": (
            _sha256(pair_manifest_path)
            == prereg.get("source_identity", {}).get("pair_manifest")
            and pair_manifest.get("config_sha256")
            == prereg.get("source_identity", {}).get("pair_config_sha256")
            and len(pair_manifest.get("pairs", [])) == 50
        ),
        "evaluation_contract_exact": (
            frozen_eval_argv[:3] == [".venv-metamon/bin/python", "-m", "eval.run"]
            and "--resume" not in frozen_eval_argv
            and "--prepare-mirrored-pairs-only" not in frozen_eval_argv
            and frozen_eval_argv.count("--operational-gate-prior-decisions") == 2
            and "--operational-gate-showdown-launch" in frozen_eval_argv
            and frozen_eval_argv.count("--foul-play-search-time-ms") == 1
            and frozen_eval_argv[
                frozen_eval_argv.index("--foul-play-search-time-ms") + 1
            ] == "500"
            and prereg.get("execution", {}).get("environment", {}).get(
                "METAGROSS_PAIR_MANIFEST_SHA256"
            )
            == prereg.get("source_identity", {}).get("pair_manifest")
            and prereg.get("execution", {}).get("environment", {}).get("PYTHONPATH")
            == "experimental/src"
        ),
        "fresh_evaluation_outputs": all(not path.exists() for path in fresh_paths),
        "python_runtime_manifest_bound": prereg["source_identity"].get("python_runtime_identity") == _sha256(python_runtime_manifest_path),
        "runtime_root_exact": showdown_dir.resolve() == (root / "external" / "pokemon-showdown").resolve(),
        "pair_directory_matches_preregistration": pair_directory == expected_pair_directory,
        "prior_urls_match_preregistration": prior_urls == expected_prior_urls,
        "prior_nonces_match_preregistration": expected_nonces == frozen_nonces,
        "file_hashes_exact": observed_files == manifest.get("files"),
        "dist_tree_exact": hash_tree(showdown_dir / "dist") == manifest.get("trees", {}).get("dist"),
        "node_modules_tree_exact": hash_tree(showdown_dir / "node_modules") == manifest.get("trees", {}).get("node_modules"),
        "node_identity_exact": node_identity == manifest.get("node"),
        "metamon_environment_exact": hash_tree(root / ".venv-metamon")
        == python_manifest.get("environments", {}).get(".venv-metamon"),
        "foul_play_environment_exact": hash_tree(root / ".venv-foul-play")
        == python_manifest.get("environments", {}).get(".venv-foul-play"),
        "audit_dependencies_exact": all(
            _sha256(root / path) == digest
            for path, digest in python_manifest.get("audit_dependencies", {}).items()
        ),
        "loopback_config": "exports.bindaddress = '127.0.0.1';" in (showdown_dir / "config" / "config.js").read_text(),
        "pair_directory_fresh_empty": not pair_entries,
        "operational_gate_paths_fresh": all(
            path and not (root / path).exists() for path in gate_paths
        ),
        "two_distinct_prior_urls": len(prior_urls) == len(set(prior_urls)) == 2,
        "two_distinct_prior_processes": len({health.get("identity", {}).get("pid") for health in prior_health}) == 2,
        "prior_nonces_exact": [health.get("identity", {}).get("nonce") for health in prior_health] == expected_nonces,
        "prior_checkpoints_exact": all(
            health.get("ok") is True
            and health.get("identity", {}).get("checkpoint_sha256") == expected_checkpoint
            for health in prior_health
        ),
        "prior_runtime_identities_exact": all(
            health.get("identity", {}).get("source_sha256")
            == prereg.get("source_identity", {}).get("prior_server.py")
            and health.get("identity", {}).get("python_executable_sha256")
            == python_manifest.get("python", {}).get("executable_sha256")
            and health.get("identity", {}).get("python_executable")
            == python_manifest.get("python", {}).get("resolved_executable")
            and health.get("identity", {}).get("python_prefix")
            == str((root / ".venv-metamon").resolve())
            and health.get("identity", {}).get("argv", [None])[0]
            == str((root / "srcs" / "metagross" / "prior_server.py").resolve())
            and health.get("identity", {}).get("argv", [])[1:]
            == frozen_prior_servers[mode]["arguments"]
            and health.get("identity", {}).get("host") == "127.0.0.1"
            and health.get("identity", {}).get("port")
            == int(frozen_prior_servers[mode]["url"].rsplit(":", 1)[1])
            and health.get("identity", {}).get("decision_dump")
            == str((root / "experimental" / "runs" / "search_native_stage2_20260809" / frozen_prior_servers[mode]["decision_dump"]).resolve())
            and health.get("identity", {}).get("environment")
            == frozen_prior_servers[mode]["environment"]
            for mode, health in zip(("candidate", "comparator"), prior_health, strict=True)
        ),
        "remote_preflight_passed": remote.get("ok") is True,
        "remote_preflight_coverage_exact": (
            remote.get("operations")
            == ["search", "paired_holdout", "shared_root"]
            and remote.get("search_durations_ms") == [250, 500]
        ),
        "remote_preflight_source_exact": remote.get("preflight_source_sha256")
        == prereg.get("source_identity", {}).get("remote_preflight.py"),
        "remote_identity_exact": remote.get("engine") == prereg.get("expected_engine_identity"),
        "remote_endpoint_exact": (
            remote.get("transport") == "modal"
            and remote.get("app") == prereg.get("remote_worker", {}).get("app")
            and remote.get("function") == prereg.get("remote_worker", {}).get("function")
            and remote.get("url") is None
        ),
        "remote_invocation_exact": (
            remote.get("arguments")
            == prereg.get("execution", {}).get("remote_preflight_argv", [])[3:]
            and remote.get("python_executable")
            == python_manifest.get("python", {}).get("resolved_executable")
            and remote.get("python_executable_sha256")
            == python_manifest.get("python", {}).get("executable_sha256")
            and remote.get("python_prefix") == str((root / ".venv-foul-play").resolve())
            and remote.get("environment")
            == prereg.get("execution", {}).get("remote_preflight_environment")
        ),
    }
    return {
        "schema_version": 1,
        "mode": "adaptive_ensemble_local_runtime_preflight",
        "preregistration_sha256": _sha256(preregistration_path),
        "authorization_artifact": {"path": str(authorization_path), "sha256": _sha256(authorization_path)},
        "runtime_manifest_sha256": _sha256(runtime_manifest_path),
        "python_runtime_manifest_sha256": _sha256(python_runtime_manifest_path),
        "preflight_source_sha256": _sha256(Path(__file__).resolve()),
        "showdown": {
            "cwd": str(showdown_dir),
            "argv": expected_showdown_argv,
            "host": "127.0.0.1",
            "port": 8010,
            "files": observed_files,
            "dist": hash_tree(showdown_dir / "dist"),
            "node_modules": hash_tree(showdown_dir / "node_modules"),
            "node": node_identity,
        },
        "pair_directory": {
            "path": str(pair_directory),
            "existed": pair_directory.exists(),
            "entries": sorted(path.name for path in pair_entries),
        },
        "prior_servers": [
            {"url": url, "health": health}
            for url, health in zip(prior_urls, prior_health, strict=True)
        ],
        "remote_preflight": {
            "path": str(remote_preflight_path),
            "sha256": _sha256(remote_preflight_path),
            "result": remote,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "authorization": {
            "local_screen_preflight_passed": all(gates.values()),
            "public_ladder_authorized": False,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--python-runtime-manifest", type=Path, required=True)
    parser.add_argument("--showdown-dir", type=Path, required=True)
    parser.add_argument("--pair-directory", type=Path, required=True)
    parser.add_argument("--prior-url", action="append", required=True)
    parser.add_argument("--expected-nonce", action="append", required=True)
    parser.add_argument("--remote-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.prior_url) != 2 or len(args.expected_nonce) != 2:
        parser.error("exactly two --prior-url and --expected-nonce values are required")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = preflight(
        args.preregistration.expanduser().resolve(),
        args.authorization.expanduser().resolve(),
        args.runtime_manifest.expanduser().resolve(),
        args.python_runtime_manifest.expanduser().resolve(),
        args.showdown_dir.expanduser().resolve(),
        args.pair_directory.expanduser().resolve(),
        args.prior_url,
        args.expected_nonce,
        args.remote_preflight.expanduser().resolve(),
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "gates": report["gates"]}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
