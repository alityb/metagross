#!/usr/bin/env python3
"""
Overnight ladder runner — measures GXE/ELO for a Foul Play agent against
the live Pokemon Showdown human ladder.

Logs every game result append-only. Writes checkpoint summaries every
CHECKPOINT_EVERY games. Watchdog: if a game takes more than GAME_TIMEOUT_MINUTES
without completing, logs FATAL and aborts to prevent silent hangs.

Usage:
  python scripts/ladder_runner.py \
      --username metagrass_stock1 \
      --label "stock_foul_play" \
      --n-games 200 \
      --search-time-ms 100 \
      [--learned-value-model path/to/model.txt] \
      --log-file /path/to/ladder.log \
      [--run-count-start 0]
"""
import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"

ROOT_DIR = Path(__file__).resolve().parents[1]
FOUL_PLAY_DIR = ROOT_DIR / "external" / "foul-play"
VENV_PY = ROOT_DIR / ".venv-foul-play" / "bin" / "python"
LIVE_URI = "wss://sim3.psim.us/showdown/websocket"
GAME_TIMEOUT_MINUTES = 20
CHECKPOINT_EVERY = 20
BACKOFF_BASE = 30  # seconds before retry on connection error
BACKOFF_MAX = 300
BETWEEN_GAMES_DELAY = 15  # polite pause between games


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("ladder_runner")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def append_game(log_file: str, record: dict) -> None:
    """Append one game result as a JSONL line."""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def write_checkpoint(log_file: str, label: str, games: list, logger: logging.Logger) -> None:
    """Write a human-readable checkpoint summary line."""
    if not games:
        return
    wins   = sum(1 for g in games if g.get("result") == "win")
    losses = sum(1 for g in games if g.get("result") == "loss")
    total  = wins + losses
    elos   = [g["elo_after"] for g in games if g.get("elo_after") is not None]
    last_elo = elos[-1] if elos else None
    last_gxe = games[-1].get("gxe_after") if games else None
    logger.info(
        "CHECKPOINT | label=%s | games=%d | W=%d L=%d | elo=%s | gxe=%s",
        label, total, wins, losses, last_elo, last_gxe,
    )


def parse_rating_line(raw_line: str) -> tuple[Optional[int], Optional[float]]:
    """Parse '...rating: 1000 → <strong>1234</strong>...' to extract new ELO."""
    # ELO
    elo_match = re.search(r"<strong>(\d+)</strong>", raw_line)
    elo = int(elo_match.group(1)) if elo_match else None
    # GXE (appears in subsequent raw lines)
    gxe_match = re.search(r"GXE:\s*([\d.]+)", raw_line)
    gxe = float(gxe_match.group(1)) if gxe_match else None
    return elo, gxe


def fetch_gxe(username: str, pokemon_format: str) -> Optional[float]:
    """Fetch GXE from the PS user profile API.

    Endpoint: https://pokemonshowdown.com/users/{user_id}.json
    Returns the GXE for the given format, or None on failure.
    Only works for registered accounts with enough rated games.
    """
    try:
        user_id = re.sub(r"[^a-z0-9]", "", username.lower())
        url = f"https://pokemonshowdown.com/users/{user_id}.json"
        req = urllib.request.Request(url, headers={"User-Agent": "metagross-ladder/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        rating = data.get("ratings", {}).get(pokemon_format, {})
        gxe = rating.get("gxe")
        return float(gxe) if gxe is not None else None
    except Exception:
        return None


async def run_one_game(
    username: str,
    password: Optional[str],
    search_time_ms: int,
    model_path: Optional[str],
    logger: logging.Logger,
    game_index: int,
    pokemon_format: str = "gen9randombattle",
) -> dict:
    """
    Run one Foul Play search_ladder game via subprocess.
    Parse ELO/GXE from the output.
    Returns dict with keys: game_index, result, elo_after, gxe_after, turns, opponent, duration_s, error
    """
    env = dict(os.environ)
    env.pop("METAGROSS_DECISION_LOG", None)  # never log decisions during ladder measurement
    if model_path:
        env["METAGROSS_VALUE_MODEL"] = str(Path(model_path).resolve())
    else:
        env.pop("METAGROSS_VALUE_MODEL", None)

    cmd = [
        str(VENV_PY),
        str(ROOT_DIR / "scripts" / "run_foul_play.py"),
        "--websocket-uri", LIVE_URI,
        "--ps-username", username,
        "--bot-mode", "search_ladder",
        "--pokemon-format", pokemon_format,
        "--run-count", "1",
        "--search-time-ms", str(search_time_ms),
        "--search-parallelism", "1",
        "--search-threads", "1",
        "--log-level", "INFO",
    ]
    if password:
        cmd.extend(["--ps-password", password])

    t0 = time.time()
    record: dict = {
        "game_index": game_index,
        "result": None,
        "elo_after": None,
        "gxe_after": None,
        "turns": None,
        "opponent": None,
        "duration_s": None,
        "error": None,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(FOUL_PLAY_DIR),
                env=env,
            ),
            timeout=10,
        )
        stdout_lines = []
        while True:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=GAME_TIMEOUT_MINUTES * 60)  # type: ignore[union-attr]
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                record["error"] = f"game_timeout_{GAME_TIMEOUT_MINUTES}min"
                logger.error("GAME_TIMEOUT game=%d timeout after %d min", game_index, GAME_TIMEOUT_MINUTES)
                break
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            stdout_lines.append(decoded)
            # Parse useful lines
            if "Initialized battle" in decoded:
                m = re.search(r"Initialized .+? against:\s*(.+)", decoded)
                if m:
                    record["opponent"] = m.group(1).strip()
            elif "Turn:" in decoded:
                m = re.search(r"Turn:\s*(\d+)", decoded)
                if m:
                    record["turns"] = int(m.group(1))
            elif "Winner:" in decoded:
                winner = decoded.split("Winner:")[-1].strip()
                record["result"] = "win" if winner == username else "loss"
            elif "|raw|" in decoded or "&rarr;" in decoded:
                # Rating line format (both local and live server):
                #   "USERNAME's rating: OLD &rarr; <strong>NEW</strong><br />(+N for winning)"
                # The server sends BOTH players' rating lines.  We must only
                # extract the ELO from lines that contain OUR username.
                # Match is case-insensitive because PS normalises to lowercase IDs.
                username_id = re.sub(r"[^a-z0-9]", "", username.lower())
                # Extract the name before "'s rating:"
                name_m = re.search(r"([A-Za-z0-9 _-]+)'s rating:", decoded)
                if name_m:
                    line_owner_id = re.sub(r"[^a-z0-9]", "", name_m.group(1).lower())
                    if line_owner_id == username_id:
                        elo_m = re.search(r"<strong>(\d+)</strong>", decoded)
                        if elo_m:
                            record["elo_after"] = int(elo_m.group(1))
            elif "W:" in decoded and "L:" in decoded:
                pass  # aggregate counts, not per-game

        await proc.wait()
        if proc.returncode not in (0, None) and record["error"] is None:
            record["error"] = f"exit_{proc.returncode}"

    except asyncio.TimeoutError:
        record["error"] = "subprocess_start_timeout"
        logger.error("Failed to start subprocess for game=%d", game_index)
    except Exception as exc:
        record["error"] = str(exc)
        logger.error("Exception in game=%d: %s", game_index, exc)

    record["duration_s"] = round(time.time() - t0, 1)

    # Fetch GXE from the PS API (only meaningful for registered accounts
    # after Glicko-RD has converged, typically 30+ rated games).
    if not record.get("error") and password:
        gxe = fetch_gxe(username, pokemon_format)
        if gxe is not None:
            record["gxe_after"] = gxe

    return record


async def run_ladder(args: argparse.Namespace, logger: logging.Logger) -> None:
    games: list[dict] = []
    consecutive_errors = 0
    game_index = args.run_count_start

    logger.info("START | label=%s | username=%s | n_games=%d | search_ms=%d | model=%s",
                args.label, args.username, args.n_games, args.search_time_ms,
                args.learned_value_model or "stock")

    target = args.run_count_start + args.n_games
    while game_index < target:
        logger.info("GAME_START game=%d / %d", game_index + 1, target)

        record = await run_one_game(
            username=args.username,
            password=getattr(args, 'password', None),
            search_time_ms=args.search_time_ms,
            model_path=args.learned_value_model,
            logger=logger,
            game_index=game_index,
            pokemon_format=getattr(args, 'pokemon_format', 'gen9randombattle'),
        )

        record["username"] = args.username
        record["label"] = args.label
        append_game(args.log_file, record)
        games.append(record)

        if record.get("error"):
            consecutive_errors += 1
            logger.warning(
                "GAME_ERROR game=%d error=%s consecutive=%d",
                game_index, record["error"], consecutive_errors,
            )
        if consecutive_errors >= 5 and consecutive_errors < 15:
                backoff = min(BACKOFF_BASE * (2 ** min(consecutive_errors - 5, 5)), BACKOFF_MAX)
                logger.error(
                    "WATCHDOG: %d consecutive errors — backing off %ds before retry",
                    consecutive_errors, backoff,
                )
                await asyncio.sleep(backoff)
        if consecutive_errors >= 15:
                logger.critical(
                    "FATAL: 15 consecutive errors, aborting. Check connection and account status."
                )
                break
        else:
            consecutive_errors = 0
            logger.info(
                "GAME_DONE game=%d result=%s elo=%s gxe=%s opponent=%s turns=%s dur=%ss",
                game_index, record.get("result"), record.get("elo_after"),
                record.get("gxe_after"), record.get("opponent"),
                record.get("turns"), record.get("duration_s"),
            )

        game_index += 1

        # Checkpoint every N games
        completed = len([g for g in games if not g.get("error")])
        if completed > 0 and completed % CHECKPOINT_EVERY == 0:
            write_checkpoint(args.log_file, args.label, [g for g in games if not g.get("error")], logger)

        # Polite pause between games — avoids rapid reconnection throttling
        await asyncio.sleep(BETWEEN_GAMES_DELAY)

    # Final summary
    good = [g for g in games if not g.get("error")]
    wins   = sum(1 for g in good if g.get("result") == "win")
    losses = sum(1 for g in good if g.get("result") == "loss")
    elos   = [g["elo_after"] for g in good if g.get("elo_after") is not None]
    final_elo = elos[-1] if elos else None
    final_gxe = good[-1].get("gxe_after") if good else None

    logger.info(
        "FINAL SUMMARY | label=%s | completed=%d | W=%d L=%d | final_elo=%s | final_gxe=%s | errors=%d",
        args.label, len(good), wins, losses, final_elo, final_gxe,
        sum(1 for g in games if g.get("error")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight ladder measurement runner")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default=None, help="Registered account password (enables persistent ELO)")
    parser.add_argument("--label", required=True, help="Short label for log (e.g. stock_foul_play)")
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--search-time-ms", type=int, default=100)
    parser.add_argument("--learned-value-model", default=None)
    parser.add_argument("--pokemon-format", default="gen9randombattle")
    parser.add_argument("--log-file", required=True, help="Append-only log file path")
    parser.add_argument("--run-count-start", type=int, default=0,
                        help="Game index to start from (for resuming)")
    args = parser.parse_args()

    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log_file)
    asyncio.run(run_ladder(args, logger))


if __name__ == "__main__":
    main()
