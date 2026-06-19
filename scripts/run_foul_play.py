#!/usr/bin/env python3
import asyncio
import multiprocessing as mp
import os
import sys
from pathlib import Path


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

    from run import run_foul_play

    asyncio.run(run_foul_play())


if __name__ == "__main__":
    main()
