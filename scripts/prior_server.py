#!/usr/bin/env python3
"""Prior server: hosts the fine-tuned 142M policy + metamon live battle tracking,
and serves per-turn root priors to Foul Play over localhost HTTP.

Data flow (FP side patches in scripts/run_foul_play.py, METAGROSS_PRIOR_SERVER):
  POST /lines  {"tag": ..., "lines": [...]}   raw protocol lines (incl. |request|)
  GET  /priors?tag=...                        -> {"priors": {engine_move_str: prob}}
  POST /end    {"tag": ...}                   cleanup

The server infers FP's chosen actions from the protocol stream itself (our own
|move|/|switch| lines), so FP only needs to tee messages and ask for priors.

Run in .venv-metamon:
  METAMON_CACHE_DIR=external/metamon_cache TORCHDYNAMO_DISABLE=1 ACCELERATE_USE_CPU=true \
    .venv-metamon/bin/python scripts/prior_server.py \
      --local-run-dir nets/checkpoints/randbats_full --local-run-name randbats_D_hlgauss \
      --checkpoint 4 --port 8977
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class BattleSession:
    """Tracks one battle: metamon backend battle + obs/action/reward history."""

    def __init__(self, tag: str, username: str, server):
        from metamon.env.metamon_battle import MetamonBackendBattle

        self.tag = tag
        self.username = username
        self.server = server
        logger = logging.getLogger(f"prior.{tag}")
        logger.setLevel(logging.ERROR)
        self.battle = MetamonBackendBattle(
            tag, username, logger, save_replays=False, gen=9
        )
        # attrs only initialized by parse_request; guard the pre-request window
        if not hasattr(self.battle, "_reviving"):
            self.battle._reviving = False
        self.obs_hist: list[dict] = []      # tokenized obs per decision point
        self.action_hist: list[int] = []    # action idx actually taken (len = len(obs)-1)
        self.reward_hist: list[float] = []
        self.last_state = None
        self.last_name_table: dict[str, int] = {}  # engine_move_str -> action idx
        self.pending_request = False

    def feed_line(self, line: str) -> None:
        if not line.startswith("|"):
            return
        parts = line.split("|")
        # parts[0] == "" for battle lines
        if len(parts) < 2 or parts[0] != "":
            return
        msg_type = parts[1]
        if msg_type in ("win", "tie"):
            # battle over: free the session (200-game gate runs would otherwise
            # accumulate obs history forever)
            with self.server.lock:
                self.server.sessions.pop(self.tag, None)
            return
        if msg_type == "request":
            payload = "|".join(parts[2:]).strip()
            if payload:
                try:
                    self.battle.parse_request(json.loads(payload))
                    self.pending_request = True
                except Exception as e:  # noqa: BLE001
                    print(f"WARN request parse {self.tag}: {e!r}", flush=True)
            return
        if msg_type in ("move", "switch", "drag") and len(parts) >= 3:
            self._maybe_record_our_action(msg_type, parts)
        try:
            self.battle.parse_message(parts)
        except Exception as e:  # noqa: BLE001
            # SimProtocol can raise on exotic messages; never kill the stream
            print(f"WARN msg parse {self.tag} {msg_type}: {e!r}", flush=True)

    def _maybe_record_our_action(self, msg_type: str, parts: list[str]) -> None:
        """Infer FP's chosen action idx from our own move/switch lines."""
        if not self.last_name_table:
            return
        ident = parts[2]  # e.g. "p1a: Weezing"
        role = getattr(self.battle, "player_role", None) or getattr(
            self.battle, "_player_role", None
        )
        if role is None or not ident.startswith(role):
            return
        if msg_type == "move":
            key = norm(parts[3])
        else:
            # switch target species from details: parts[3] like "Weezing-Galar, L84"
            key = "switch " + norm(parts[3].split(",")[0])
        # match against last emitted name table (tera variant folds to base move)
        idx = None
        for name, i in self.last_name_table.items():
            base = name[:-5] if name.endswith("-tera") else name
            if norm(base) == key or norm(name) == key:
                idx = i
                if not name.endswith("-tera"):
                    break
        if idx is not None and len(self.action_hist) < len(self.obs_hist):
            self.action_hist.append(idx)

    def compute_priors(self) -> dict:
        import numpy as np
        import torch

        from metamon.interface import UniversalState, UniversalAction, consistent_move_order, consistent_pokemon_order

        state = UniversalState.from_Battle(self.battle)
        # reward for rl2s bookkeeping
        if self.last_state is not None:
            try:
                r = self.server.reward_fn(self.last_state, state)
            except Exception:
                r = 0.0
            self.reward_hist.append(float(r))
        self.last_state = state

        obs = self.server.obs_space.state_to_obs(state)
        # legality mask
        illegal = np.ones(13, dtype=bool)
        try:
            for a in UniversalAction.definitely_valid_actions(state, self.battle):
                illegal[a] = False
        except Exception:
            illegal[:] = False
        obs = dict(obs)
        obs["illegal_actions"] = illegal
        self.obs_hist.append(obs)
        # if action inference missed a turn, pad with a no-op guess (idx 0)
        while len(self.action_hist) < len(self.obs_hist) - 1:
            self.action_hist.append(0)
        while len(self.reward_hist) < len(self.obs_hist) - 1:
            self.reward_hist.append(0.0)

        T = len(self.obs_hist)
        A = 13
        text = torch.tensor(
            np.stack([o["text_tokens"] for o in self.obs_hist]), dtype=torch.int32
        ).unsqueeze(0)
        numbers = torch.tensor(
            np.stack([o["numbers"] for o in self.obs_hist]), dtype=torch.float32
        ).unsqueeze(0)
        ill = torch.tensor(
            np.stack([o["illegal_actions"] for o in self.obs_hist])
        ).unsqueeze(0)
        acts = torch.zeros((T, A))
        for i, a in enumerate(self.action_hist[: T - 1]):
            acts[i, a] = 1.0
        rews = torch.zeros((T, 1))
        for i, r in enumerate(self.reward_hist[: T - 1]):
            rews[i, 0] = r
        # rl2s[t] = (reward[t-1], action[t-1]); blank at t=0
        rl2s = torch.cat(
            [
                torch.cat([torch.zeros(1, 1), rews[: T - 1]], dim=0),
                torch.cat([torch.zeros(1, A), acts[: T - 1]], dim=0),
            ],
            dim=-1,
        ).unsqueeze(0)
        time_idxs = torch.arange(T).long().unsqueeze(0)
        obs_batch = {"text_tokens": text, "numbers": numbers, "illegal_actions": ill}

        agent = self.server.agent
        with torch.no_grad():
            emb, _ = agent.get_state_embedding(
                obs=obs_batch, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None
            )
            dists = agent.actor(
                emb,
                straight_from_obs={
                    k: obs_batch[k] for k in agent.pass_obs_keys_to_actor
                },
            )
            probs = dists.probs[0, -1, -1, :].cpu().numpy()  # last step, inference gamma

        probs = probs * (~illegal)
        if probs.sum() <= 0:
            probs = (~illegal).astype(float)
        probs = probs / probs.sum()

        # name table: action idx -> engine move string
        name_table: dict[str, int] = {}
        try:
            moves = consistent_move_order(
                list(self.battle.active_pokemon.moves.values())
            ) if self.battle.active_pokemon else []
        except Exception:
            moves = []
        try:
            bench = consistent_pokemon_order(
                [p for p in self.battle.team.values() if not p.fainted and not p.active]
            )
        except Exception:
            bench = []
        for i, mv in enumerate(moves[:4]):
            name_table[mv.id] = i
            name_table[f"{mv.id}-tera"] = i + 9
        for i, p in enumerate(bench[:5]):
            name_table[f"switch {norm(p.name)}"] = i + 4
        self.last_name_table = name_table

        priors = {}
        for name, idx in name_table.items():
            priors[name] = float(probs[idx])
        return {
            "priors": priors,
            "probs": [float(p) for p in probs],
            "turn": T,
        }


class PriorServer:
    def __init__(self, args):
        os.environ.setdefault("METAMON_CACHE_DIR", str(ROOT / "external" / "metamon_cache"))
        os.environ.setdefault("WANDB_MODE", "disabled")
        import metamon.rl.pretrained as _pt

        if args.local_run_dir:
            model = _pt.LocalFinetunedModel(
                base_model=getattr(_pt, args.local_base_model),
                amago_ckpt_dir=args.local_run_dir,
                model_name=args.local_run_name,
                default_checkpoint=args.checkpoint,
            )
            label = f"local:{args.local_run_name}@ckpt{args.checkpoint}"
        else:
            model = _pt.get_pretrained_model(args.agent)
            label = args.agent
        print(f"PRIOR_SERVER loading {label}", flush=True)
        experiment = model.initialize_agent(checkpoint=args.checkpoint, log=False)
        self.agent = experiment.policy
        self.agent.eval()
        self.obs_space = model.observation_space
        self.reward_fn = model.reward_function
        self.username = args.username
        self.sessions: dict[str, BattleSession] = {}
        self.lock = threading.Lock()
        print("PRIOR_SERVER ready", flush=True)

    def session(self, tag: str) -> BattleSession:
        with self.lock:
            if tag not in self.sessions:
                self.sessions[tag] = BattleSession(tag, self.username, self)
            return self.sessions[tag]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="Kakuna")
    parser.add_argument("--local-run-dir", default=None)
    parser.add_argument("--local-run-name", default=None)
    parser.add_argument("--local-base-model", default="Kakuna")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--username", required=True,
                        help="FP's showdown username (to identify our side)")
    parser.add_argument("--port", type=int, default=8977)
    args = parser.parse_args()

    server = PriorServer(args)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            tag = data.get("tag", "")
            if path == "/lines":
                sess = server.session(tag)
                for line in data.get("lines", []):
                    sess.feed_line(line)
                self._json(200, {"ok": True})
            elif path == "/end":
                with server.lock:
                    server.sessions.pop(tag, None)
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "unknown"})

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/priors":
                tag = parse_qs(parsed.query).get("tag", [""])[0]
                try:
                    result = server.session(tag).compute_priors()
                    self._json(200, result)
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    self._json(500, {"error": f"{type(e).__name__}: {e}"})
            elif parsed.path == "/health":
                self._json(200, {"ok": True, "sessions": len(server.sessions)})
            else:
                self._json(404, {"error": "unknown"})

    httpd = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"PRIOR_SERVER listening on 127.0.0.1:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
