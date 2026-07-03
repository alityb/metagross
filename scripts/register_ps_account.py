#!/usr/bin/env python3
"""Register a Pokémon Showdown account via the public API.

Flow: connect to the live sim websocket as a guest, grab the challstr, then
POST act=register with the standard captcha answer. Prints the result.

Usage:
  python scripts/register_ps_account.py --username NAME --password PW
"""
from __future__ import annotations

import argparse
import asyncio
import json
import urllib.parse
import urllib.request

WS = "wss://sim3.psim.us/showdown/websocket"
ACTION = "https://play.pokemonshowdown.com/api/register"


async def get_challstr() -> str:
    import websockets

    async with websockets.connect(WS) as ws:
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=30)
            for line in str(msg).split("\n"):
                if line.startswith("|challstr|"):
                    return line[len("|challstr|"):]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    challstr = asyncio.run(get_challstr())
    print(f"challstr acquired ({len(challstr)} chars)")

    data = urllib.parse.urlencode({
        "act": "register",
        "username": args.username,
        "password": args.password,
        "cpassword": args.password,
        "captcha": "pikachu",
        "challstr": challstr,
    }).encode()
    req = urllib.request.Request(
        ACTION, data=data,
        headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode()
    if body.startswith("]"):
        body = body[1:]
    try:
        parsed = json.loads(body)
        ok = bool(parsed.get("curuser", {}).get("loggedin") or parsed.get("assertion"))
        print("REGISTER_RESULT", json.dumps({
            "ok": ok,
            "actionsuccess": parsed.get("actionsuccess"),
            "error": parsed.get("actionerror"),
        }))
    except json.JSONDecodeError:
        print("REGISTER_RAW", body[:300])


if __name__ == "__main__":
    main()
