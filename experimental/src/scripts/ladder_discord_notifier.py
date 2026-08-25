#!/usr/bin/env python3
"""Post roguefan23 ladder updates to a Discord webhook.

Tails the newest supervisor block's ratings.jsonl and posts on change (rate-
limited) plus a heartbeat every 30 minutes. Webhook URL is read from
~/.metagross_discord_webhook (single line, chmod 600, owner-created);
never logged. Near-zero CPU: one stat/read per poll.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path("/Users/alityb/projects/metagross")
SUP = ROOT / "experimental/runs/ladder_headline_20260818/supervisor"
HOOK_FILE = Path.home() / ".metagross_discord_webhook"
POLL_SECONDS = 60
HEARTBEAT_SECONDS = 1800


def newest_ratings() -> tuple[Path | None, dict | None]:
    blocks = sorted(SUP.glob("*/blocks/*/ratings.jsonl"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if not blocks:
        return None, None
    lines = blocks[0].read_text().splitlines()
    return blocks[0], (json.loads(lines[-1]) if lines else None)


def post(webhook: str, text: str) -> None:
    body = json.dumps({"content": text}).encode()
    request = urllib.request.Request(
        webhook, data=body, headers={"Content-Type": "application/json",
                                     "User-Agent": "metagross-ladder/1.0"})
    urllib.request.urlopen(request, timeout=10).read()


def fmt(r: dict, block: Path) -> str:
    return (f"**roguefan23** {r['w']}-{r['l']} | Elo {r['elo']:.0f} | "
            f"GXE {r['gxe']}% | Glicko {r['rpr']:.0f} ± {r['rprd']:.0f} | "
            f"block `{block.parent.name.rsplit('-', 1)[0]}`")


def main() -> None:
    webhook = HOOK_FILE.read_text().strip()
    last_key = None
    last_post = 0.0
    post(webhook, "ladder notifier online — watching roguefan23")
    while True:
        try:
            block, r = newest_ratings()
            if r is not None:
                key = (r["w"], r["l"], round(r["gxe"], 1))
                now = time.time()
                if key != last_key or now - last_post > HEARTBEAT_SECONDS:
                    post(webhook, fmt(r, block))
                    last_key, last_post = key, now
        except Exception as exc:  # keep the notifier alive across blips
            print(f"notifier warning: {type(exc).__name__}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
