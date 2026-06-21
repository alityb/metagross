#!/usr/bin/env python3
import asyncio
import multiprocessing as mp
import os
import sys
from pathlib import Path


def patch_foul_play_protocol_bugs() -> None:
    import fp.run_battle as run_battle

    if not hasattr(run_battle, "format_decision") or not callable(run_battle.format_decision):
        raise RuntimeError("Foul Play patch target fp.run_battle.format_decision is missing")

    original_format_decision = run_battle.format_decision

    def format_decision_with_default(battle, decision):
        if isinstance(decision, str) and decision.strip().lower() == "no move":
            return ["/choose default", str(battle.rqid)]
        return original_format_decision(battle, decision)

    run_battle.format_decision = format_decision_with_default


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    foul_play_dir = Path(os.environ.get("FOUL_PLAY_DIR", root_dir / "external" / "foul-play"))

    if sys.platform == "darwin":
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass

    os.chdir(foul_play_dir)
    sys.path.insert(0, str(foul_play_dir))

    patch_foul_play_protocol_bugs()

    from run import run_foul_play

    asyncio.run(run_foul_play())


if __name__ == "__main__":
    main()
