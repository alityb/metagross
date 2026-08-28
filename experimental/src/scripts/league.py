#!/usr/bin/env python3
"""League harness: evaluate one candidate serving configuration against a
frozen pool of diverse reference opponents, reporting the PER-OPPONENT vector.

Motivation (2026-08 campaign): population and head-to-head evaluations flip
signs — temperature flattening won its H2H vs stateless (55.5%) while losing
~8 GXE on the real ladder. A single-opponent H2H therefore cannot license
deployment. The league reproduces population diversity locally: mirrored
pairs against opponents spanning strength and style, one result vector.

Design rules (each earned by a recorded failure):
- Observable activation: a matchup involving a serving intervention is valid
  only if the ACTIVE line appears in the candidate arm's logs and does NOT
  appear in the opponent arm's.
- Idempotent/resumable: matchups with a result.json are skipped; eval.run
  runs with --resume after the first attempt. Safe to re-run after any crash.
- Per-arm gating by prior-server port (candidate 9023, opponent 9024), so a
  global env cannot contaminate the opponent arm.

Usage:
  league.py --config league.json --out <run-dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/alityb/projects/metagross")
PY = str(ROOT / ".venv-metamon/bin/python")
FOUL_PLAY_PY = str(ROOT / ".venv-foul-play/bin/python")
SHOWDOWN = ROOT / "external/pokemon-showdown"
SHOWDOWN_PORT = 8022
CAND_PORT = 9023
OPP_PORT = 9024
CHECKPOINT_SHA = "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
PRODUCTION_RUN_SEED = (
    "53b8bfbbeb2927291872d939575909c7fd3d4c328d42a1c4f81218ec6e711863")
ENGINE_SHA = "79bea0e467b32e2958bd5d39595fd728a3068be2950085f7b18fa69943f30d71"


def wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def port_free(port: int) -> bool:
    return subprocess.run(["/usr/sbin/lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                          capture_output=True).returncode != 0


def wait_ports(ports: list[int], timeout: int = 480) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(not port_free(p) for p in ports):
            return True
        time.sleep(2)
    return False


def kill_infra() -> None:
    subprocess.run(["pkill", "-f", "prior_server.py"], capture_output=True)
    # Orphaned clients from a killed eval.run keep playing (and holding
    # Showdown usernames) forever — reap them too.
    subprocess.run(["pkill", "-9", "-f", "srcs.metagross.run_foul_play"],
                   capture_output=True)
    for p in (CAND_PORT, OPP_PORT):
        out = subprocess.run(["/usr/sbin/lsof", "-ti", f"tcp:{p}"], capture_output=True,
                             text=True).stdout.split()
        for pid in out:
            subprocess.run(["kill", "-9", pid], capture_output=True)
    time.sleep(2)


def registration_dir(out_root: Path) -> Path:
    """One league-wide registration dir, shared by every matchup and by the
    Showdown process (via METAGROSS_EVAL_PAIR_DIR)."""
    return out_root / "registrations"


def start_showdown(out_dir: Path) -> None:
    if not port_free(SHOWDOWN_PORT):
        return
    os.makedirs(SHOWDOWN / "logs/repl", exist_ok=True)
    env = dict(os.environ)
    # Without METAGROSS_EVAL_PAIR_DIR, Showdown silently ignores the mirrored
    # pair registrations and every "mirrored" battle runs with random teams
    # and a random seed (the 2026-08-26 baseline played 33 such games while
    # recording manifest hashes). eval.run now fails a game whose
    # registrations were not consumed, so a misconfigured Showdown dies on
    # game 1 instead of banking fiction.
    env["METAGROSS_EVAL_PAIR_DIR"] = str(registration_dir(out_dir))
    # Under cron, PATH lacks node (Homebrew/nvm); every guardian relaunch of
    # Showdown died with `env: node: No such file or directory`.
    node_dirs = [str(p) for p in Path.home().glob(".nvm/versions/node/*/bin")]
    env["PATH"] = ":".join(["/opt/homebrew/bin", *sorted(node_dirs, reverse=True),
                            env.get("PATH", "/usr/bin:/bin")])
    subprocess.Popen(
        ["./pokemon-showdown", "start", "--no-security", str(SHOWDOWN_PORT)],
        cwd=SHOWDOWN, env=env, stdout=open(out_dir / "showdown.log", "a"),
        stderr=subprocess.STDOUT)


def start_prior_server(port: int, name: str, mode: str, env_extra: dict,
                       out_dir: Path) -> subprocess.Popen:
    env = dict(os.environ,
               METAMON_CACHE_DIR=str(ROOT / "external/metamon_cache"),
               TORCHDYNAMO_DISABLE="1", ACCELERATE_USE_CPU="true",
               PYTHONPATH=str(ROOT))
    env.update({str(k): str(v) for k, v in (env_extra or {}).items()})
    return subprocess.Popen(
        [PY, "-u", str(ROOT / "srcs/metagross/prior_server.py"),
         "--local-run-dir", str(ROOT / "srcs/models"),
         "--local-run-name", "randbats_exit_r1", "--checkpoint", "5",
         "--checkpoint-sha256", CHECKPOINT_SHA,
         "--port", str(port), "--username", name,
         "--trajectory-mode", mode],
        env=env, stdout=open(out_dir / f"prior_{name}.log", "a"),
        stderr=subprocess.STDOUT)


def run_matchup(candidate: dict, opp: dict, index: int, base_seed: int,
                out_root: Path, games: int) -> dict:
    mdir = out_root / f"m{index:02d}_{opp['name']}"
    mdir.mkdir(parents=True, exist_ok=True)
    result_path = mdir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    kill_infra()
    # Shared with the Showdown process (METAGROSS_EVAL_PAIR_DIR). Clear stale
    # registrations from earlier matchups/attempts so eval.run's fresh-dir
    # check and the per-game consumption check stay meaningful.
    reg_dir = registration_dir(out_root)
    reg_dir.mkdir(parents=True, exist_ok=True)
    for stale in reg_dir.glob("*.json"):
        stale.unlink()
    servers = [start_prior_server(
        CAND_PORT, "candidate", candidate.get("trajectory_mode", "causal-history"),
        candidate.get("prior_env") or {}, mdir)]
    ports = [SHOWDOWN_PORT, CAND_PORT]
    if opp.get("needs_prior_server", False):
        servers.append(start_prior_server(
            OPP_PORT, "opponent", opp.get("trajectory_mode", "causal-history"),
            opp.get("prior_env") or {}, mdir))
        ports.append(OPP_PORT)
    if not wait_ports(ports):
        kill_infra()
        raise RuntimeError(f"infrastructure not ready for {opp['name']}")

    env = dict(os.environ,
               PYTHONPATH=str(ROOT / "experimental/src"),
               METAGROSS_PINNED_ENGINE_IMPORT_ROOT=str(
                   ROOT / ".venv-foul-play/lib/python3.11/site-packages"),
               METAGROSS_PINNED_ENGINE_SHA256=ENGINE_SHA)
    # Candidate serving-time interventions, port-gated so they can never
    # touch the opponent arm even though the env is process-global.
    for k, v in (candidate.get("client_env") or {}).items():
        env[str(k)] = str(v)
    for gate in ("METAGROSS_PRIOR_TEMP_PORTS", "METAGROSS_GUMBEL_ROOT_PORTS"):
        if gate in env:
            env[gate] = str(CAND_PORT)
    if "METAGROSS_PRIOR_TEMP_SCHEDULE" in env:
        env.setdefault("METAGROSS_PRIOR_TEMP_PORTS", str(CAND_PORT))

    cmd = [PY, "-m", "eval.run",
           "--mode", "h2h", "--server", "local",
           "--format", "gen9randombattle",
           "--websocket-uri", f"ws://localhost:{SHOWDOWN_PORT}/showdown/websocket",
           "--showdown-dir", str(SHOWDOWN),
           "--paired", "--mirrored-pairs",
           "--mirror-seed", str(base_seed + index), "--fail-fast",
           "--mirrored-team-generator",
           str(ROOT / "experimental/src/scripts/generate_mirrored_randbats_pair.cjs"),
           "--pair-registration-dir", str(reg_dir),
           "--agent-a", candidate.get("agent", "production_r1_search_first"),
           "--agent-b", opp["agent"],
           "--agent-a-prior-server-url", f"http://127.0.0.1:{CAND_PORT}",
           "--agent-a-require-priors",
           "--foul-play-python", FOUL_PLAY_PY,
           "--foul-play-search-time-ms", "500",
           "--foul-play-search-parallelism", "8",
           "--foul-play-search-threads", "1",
           "--n-games", str(games), "--cpuct", "2.0",
           "--production-run-seed", PRODUCTION_RUN_SEED,
           "--username-prefix", f"lg{index:02d}",
           "--run-id", f"league-{out_root.name}-{opp['name']}",
           "--json-out", str(result_path), "--log-dir", str(mdir / "logs")]
    if opp.get("needs_prior_server", False):
        # Strict isolation needs BOTH per-slot prior flags; passing it for a
        # priorless opponent (vanilla foul-play, max_damage) fails parse_args
        # and made matchups m02/m03 unrunnable.
        cmd += ["--agent-b-prior-server-url", f"http://127.0.0.1:{OPP_PORT}",
                "--agent-b-require-priors", "--strict-isolated-priors"]

    for attempt in range(1, 9):
        run_cmd = list(cmd) + (["--resume"] if (mdir / "result.json.progress.json").exists() else [])
        with open(mdir / "eval.log", "a") as log:
            log.write(f"=== attempt {attempt} {time.strftime('%H:%M:%SZ', time.gmtime())}\n")
            log.flush()
            subprocess.run(run_cmd, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT)
        if result_path.exists():
            break
        time.sleep(15)
    kill_infra()
    if not result_path.exists():
        raise RuntimeError(f"matchup {opp['name']} gave up after retries")
    return json.loads(result_path.read_text())


def activation_check(mdir: Path, intervention_active: bool) -> dict:
    cand_active = opp_active = False
    for path in glob.glob(str(mdir / "logs" / "*.log")):
        text = Path(path).read_text(errors="replace")
        if "ACTIVE" in text and ("temperature" in text or "TRUNCATION" in text
                                 or "GUMBEL" in text):
            base = os.path.basename(path)
            # candidate clients talk to CAND_PORT; identify by server URL line
            if f":{CAND_PORT}" in text:
                cand_active = True
            elif f":{OPP_PORT}" in text:
                opp_active = True
            else:
                cand_active = True
    valid = (not intervention_active) or (cand_active and not opp_active)
    return {"candidate_active": cand_active, "opponent_active": opp_active,
            "valid": valid}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    candidate = cfg["candidate"]
    pool = cfg["pool"]
    base_seed = int(cfg.get("base_seed", 2026082600))
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, args.out / "league_config.json")

    if args.dry_run:
        print(f"candidate: {candidate['name']}")
        for i, opp in enumerate(pool):
            print(f"  m{i:02d}: vs {opp['name']} ({opp['agent']}, "
                  f"{opp.get('n_games', 40)} games, seed {base_seed + i}, "
                  f"prior_server={opp.get('needs_prior_server', False)})")
        return

    intervention = bool(candidate.get("client_env") or candidate.get("prior_env"))
    start_showdown(args.out)
    rows = []
    for i, opp in enumerate(pool):
        games = int(opp.get("n_games", 40))
        print(f"[league] m{i:02d} {candidate['name']} vs {opp['name']} "
              f"({games} games)", flush=True)
        result = run_matchup(candidate, opp, i, base_seed, args.out, games)
        s = result.get("summary", result)
        w = int(s.get("agent_a_wins", 0))
        n = int(s.get("decisive_games", s.get("completed_games", 0)) or 0)
        lo, hi = wilson(w, n)
        act = activation_check(args.out / f"m{i:02d}_{opp['name']}", intervention)
        rows.append({"opponent": opp["name"], "agent": opp["agent"],
                     "wins": w, "games": n,
                     "winrate": round(w / n, 4) if n else None,
                     "ci95": [round(lo, 4), round(hi, 4)],
                     "activation": act})
        print(f"[league]   -> {w}-{n - w} ({100 * w / max(1, n):.1f}%) "
              f"CI [{lo:.3f},{hi:.3f}] valid={act['valid']}", flush=True)

    report = {"candidate": candidate["name"],
              "candidate_config": candidate,
              "base_seed": base_seed,
              "vector": rows,
              "all_valid": all(r["activation"]["valid"] for r in rows)}
    (args.out / "league_report.json").write_text(json.dumps(report, indent=2))

    lines = [f"# League report: {candidate['name']}", "",
             "| Opponent | Record | Win% | 95% CI | Valid |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['opponent']} | {r['wins']}-{r['games'] - r['wins']} | "
            f"{(r['winrate'] or 0) * 100:.1f}% | "
            f"[{r['ci95'][0] * 100:.1f}, {r['ci95'][1] * 100:.1f}] | "
            f"{'yes' if r['activation']['valid'] else 'NO — INVALID'} |")
    (args.out / "league_report.md").write_text("\n".join(lines) + "\n")
    print(f"[league] report written to {args.out}/league_report.md", flush=True)


if __name__ == "__main__":
    main()
