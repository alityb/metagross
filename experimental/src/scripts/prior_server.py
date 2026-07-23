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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def action_likelihoods_from_public_payload(server, payload: object) -> dict:
    """Build action evidence from a pre-action public protocol prefix only."""
    from belief.action_conditioned_randbats import CandidateValidationError, load_active_candidates
    from scripts.produce_action_conditioned_randbats_rows import _matches, _pokemon_facts, replay_state_from_public_state

    def unavailable(reason: str) -> dict:
        return {"available": False, "reason": reason}

    if not isinstance(payload, dict):
        return unavailable("payload must be an object")
    prefix, records = payload.get("protocol_prefix"), payload.get("active_candidates")
    action, metadata = payload.get("observed_action"), payload.get("public_metadata")
    if not isinstance(prefix, list) or not prefix or not isinstance(records, list) or not 1 <= len(records) <= 32:
        return unavailable("invalid public payload")
    if not isinstance(metadata, dict) or set(metadata) - {"format", "observer_side"} or metadata.get("observer_side") != "p1" or not isinstance(metadata.get("format"), str):
        return unavailable("invalid public_metadata")
    if not isinstance(action, str) or not re.fullmatch(r"(?:move [a-z0-9]+(?:-tera)?|switch [a-z0-9]+)", action):
        return unavailable("invalid observed_action")
    if any(not isinstance(message, list) or not message or not all(isinstance(value, str) for value in message) or message[0] == "request" for message in prefix):
        return unavailable("invalid public protocol prefix")
    try:
        candidates = load_active_candidates(records)
        replay_state = replay_state_from_public_state({"protocol_prefix": prefix, "format": metadata["format"]})
        if replay_state.force_switch:
            return unavailable("forced switch has no discretionary action likelihood")
        facts = _pokemon_facts(replay_state.opponent_active_pokemon)
        if any(not _matches(candidate, facts) for candidate in candidates):
            candidate_species = sorted({
                norm((candidate.public_data or {}).get("speciesId") or (candidate.public_data or {}).get("species"))
                for candidate in candidates
            })
            return unavailable(
                "candidate conflicts with public active species "
                f"public={facts.get('speciesId')} candidates={','.join(candidate_species)}"
            )
        # The candidate is p2; the adapter's opponent count is the p1 observer
        # side. Derive it from replay reconstruction, never client state.
        observer_remaining = int(replay_state.active_pokemon is not None) + len(replay_state.available_switches)
        if not 1 <= observer_remaining <= 6:
            return unavailable("invalid public observer remaining count")
        likelihoods = server.action_likelihood_adapter.action_likelihoods({
            "replay_state": replay_state,
            "acting_can_tera": bool(replay_state.can_tera),
            "public_opponent_remaining": observer_remaining,
        }, candidates, action)
        if set(likelihoods) != {candidate.candidate_id for candidate in candidates}:
            return unavailable("incomplete likelihood result")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in likelihoods.values()):
            return unavailable("invalid likelihood result")
        return {"available": True, "likelihoods": likelihoods}
    except AttributeError as exc:
        import traceback
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        return unavailable(
            f"public reconstruction unavailable: AttributeError: "
            f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
        )
    except ValueError as exc:
        return unavailable(f"public reconstruction unavailable: ValueError: {exc}")
    except (CandidateValidationError, TypeError, KeyError) as exc:
        return unavailable(f"public reconstruction unavailable: {type(exc).__name__}")
    except Exception as exc:
        detail = f": {exc}" if isinstance(exc, (FileNotFoundError, AttributeError)) else ""
        if isinstance(exc, AttributeError) and not str(exc):
            import traceback
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            detail = f": {Path(frame.filename).name}:{frame.lineno}:{frame.name}"
        return unavailable(f"public reconstruction unavailable: {type(exc).__name__}{detail}")


def session_key(namespace: str, tag: str) -> str:
    """Namespace the live session without changing the dumped replay tag."""
    return f"{namespace}\0{tag}" if namespace else tag


class BattleSession:
    """Tracks one battle: metamon backend battle + obs/action/reward history."""

    def __init__(
        self, session_id: str, tag: str, namespace: str, username: str, server
    ):
        from metamon.env.metamon_battle import MetamonBackendBattle

        self.tag = tag
        self.session_id = session_id
        self.namespace = namespace
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
        # Schema-v3: monotone per-battle decision counter. Incremented once per
        # successful /priors response; echoed to FP and written to the dump so
        # MCTS visit targets can be joined to the exact observation served,
        # with no replay parsing.
        self.decision_idx = 0
        self.last_state = None
        self.last_name_table: dict[str, int] = {}  # engine_move_str -> action idx
        self.pending_request = False
        # /lines and /priors arrive on different HTTP threads. Keep each battle
        # state consistent while allowing unrelated battles to proceed.
        self.lock = threading.RLock()

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
                self.server.sessions.pop(self.session_id, None)
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

    def compute_priors(self, requester_username: str | None = None) -> dict:
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
        mask_fallback = False
        mask_fallback_error = None
        try:
            for a in UniversalAction.definitely_valid_actions(state, self.battle):
                illegal[a.action_idx] = False
        except Exception as exc:
            # Fail-open for serving (a missing mask must not stall the game),
            # but flag it so schema-v3 consumers can count/reject these rows:
            # their legality validation is vacuous.
            illegal[:] = False
            mask_fallback = True
            mask_fallback_error = f"{type(exc).__name__}: {exc}"
        obs = dict(obs)
        obs["illegal_actions"] = illegal
        # Stateless two-step inference is intentionally used here. The live
        # protocol occasionally misses an action boundary, which makes a
        # history sequence and its RL2 action/reward stream differ by one
        # timestep. The policy's current-state prior is still useful, while
        # this fixed shape avoids silently falling back to no-prior MCTS.
        current_obs = obs
        T = 2
        A = 13
        device = self.server.device
        text = torch.tensor(
            np.stack([np.zeros_like(current_obs["text_tokens"]), current_obs["text_tokens"]]),
            dtype=torch.int32,
            device=device,
        ).unsqueeze(0)
        numbers = torch.tensor(
            np.stack([np.zeros_like(current_obs["numbers"]), current_obs["numbers"]]),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        numbers = torch.nan_to_num(numbers)
        ill = torch.tensor(
            np.stack([np.ones(13, dtype=bool), current_obs["illegal_actions"]]), device=device
        ).unsqueeze(0)
        rl2s = torch.zeros((1, T, A + 1), device=device)
        # AMAGO's transformer squeezes the final dimension internally. Keep
        # it explicit so a one-turn history stays [B, L] rather than [B].
        time_idxs = torch.arange(T, device=device).long().unsqueeze(0).unsqueeze(-1)
        obs_batch = {"text_tokens": text, "numbers": numbers, "illegal_actions": ill}

        agent = self.server.agent
        with torch.no_grad():
            emb, _ = agent.get_state_embedding(
                obs=obs_batch, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None
            )
            dists = agent.actor(
                emb,
                straight_from_obs={
                    # The trajectory encoder emits one fewer transition than
                    # raw observations. Match its sequence length for actor
                    # side-channel tensors (especially numbers).
                    k: obs_batch[k][:, : emb.shape[1]]
                    for k in agent.pass_obs_keys_to_actor
                },
            )
            probs = dists.probs[0, -1, -1, :].cpu().numpy()  # last step, inference gamma

        if not np.isfinite(probs).all():
            raise RuntimeError("non-finite root policy probabilities")

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

        decision_idx = self.decision_idx
        battle_turn = getattr(self.battle, "turn", None)
        if self.server.dump_path:
            # Fail closed: if the dump write fails, the whole /priors request
            # fails (500), and METAGROSS_REQUIRE_PRIORS=1 discards the game.
            # A decision must never be played whose observation was not
            # durably recorded.
            dump_row = {
                "schema": 3,
                "tag": self.tag,
                "namespace": self.namespace,
                "decision_idx": decision_idx,
                "battle_turn": battle_turn,
                # The requester's actual PS username (per-game generated by
                # eval.run) is the join key against decision logs; the launch
                # --username is only a fallback label.
                "username": requester_username or self.username,
                # exactly what the policy consumed (numbers nan_to_num'ed to
                # match the inference path above)
                "text_tokens": current_obs["text_tokens"].tolist(),
                "numbers": np.nan_to_num(
                    np.asarray(current_obs["numbers"], dtype=np.float32)
                ).tolist(),
                "illegal_actions": [bool(x) for x in illegal],
                "mask_fallback": mask_fallback,
                "mask_fallback_error": mask_fallback_error,
                "name_table": name_table,
                "probs": [float(p) for p in probs],
            }
            line = json.dumps(dump_row, separators=(",", ":")) + "\n"
            with self.server.dump_lock:
                with open(self.server.dump_path, "a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
        self.decision_idx += 1

        return {
            "priors": priors,
            "opp_priors": self.compute_opponent_priors(),
            "probs": [float(p) for p in probs],
            "turn": T,
            "decision_idx": decision_idx,
            "battle_turn": battle_turn,
        }

    def compute_opponent_priors(self) -> dict:
        """Compute priors for the OPPONENT's moves from the opponent's POV.

        FP's modeled opponent currently sees our full team (mirror assumption).
        We bias the opponent's action distribution toward what a human would do,
        using the same 142M policy evaluated from the opponent's perspective.

        The opponent's POV = same game with sides swapped. We build a flipped
        UniversalState and run the policy on it. The opponent's legal moves
        are their active mon's moves + switches; we map those to engine move
        strings in the opponent's option order.
        """
        import numpy as np
        import torch
        from metamon.interface import UniversalState, UniversalAction, consistent_move_order, consistent_pokemon_order

        try:
            opp_battle = self._make_opp_battle()
            if opp_battle is None:
                return {}
            # Build a real opponent battle view instead of mutating obsolete
            # UniversalState field names. This gives state_to_obs the expected
            # player_active_pokemon / available_switches layout.
            flipped = UniversalState.from_Battle(opp_battle)
            obs = self.server.obs_space.state_to_obs(flipped)
            # opponent's legal actions from the flipped battle
            illegal = np.ones(13, dtype=bool)
            try:
                for a in UniversalAction.definitely_valid_actions(flipped, opp_battle):
                    illegal[a.action_idx] = False
            except Exception:
                illegal[:] = False
            obs = dict(obs)
            obs["illegal_actions"] = illegal

            # single-step inference (no history for opponent — we don't track
            # their action/reward sequence; this is a stateless prior).
            # state_to_obs returns 1D arrays for a single state.
            # The transformer requires T>=2, so pad with a blank first step.
            tt = obs["text_tokens"]  # (L,)
            nn = obs["numbers"]      # (N,)
            tt = np.stack([np.zeros_like(tt), tt])  # (2, L)
            nn = np.stack([np.zeros_like(nn), nn])  # (2, N)
            T = 2
            device = self.server.device
            ill_opp = np.ones((T, 13), dtype=bool)
            ill_opp[-1] = illegal  # only the real step has the mask
            text = torch.tensor(tt, dtype=torch.int32, device=device).unsqueeze(0)  # [1, 2, L]
            numbers = torch.tensor(nn, dtype=torch.float32, device=device).unsqueeze(0)
            numbers = torch.nan_to_num(numbers)
            ill_t = torch.tensor(ill_opp, device=device).unsqueeze(0)  # [1, 2, A]
            rl2s = torch.zeros((1, T, 14), device=device)
            time_idxs = torch.arange(T, device=device).long().unsqueeze(0).unsqueeze(-1)
            obs_batch = {"text_tokens": text, "numbers": numbers, "illegal_actions": ill_t}

            agent = self.server.agent
            with torch.no_grad():
                try:
                    emb, _ = agent.get_state_embedding(
                        obs=obs_batch, rl2s=rl2s, time_idxs=time_idxs, hidden_state=None
                    )
                    dists = agent.actor(
                        emb,
                        straight_from_obs={
                            k: obs_batch[k][:, : emb.shape[1]]
                            for k in agent.pass_obs_keys_to_actor
                        },
                    )
                    probs = dists.probs[0, -1, -1, :].cpu().numpy()
                    if not np.isfinite(probs).all():
                        raise RuntimeError("non-finite opponent policy probabilities")
                except (ValueError, RuntimeError) as e:
                    if "not enough values" in str(e) or "shape" in str(e).lower():
                        return {}
                    raise

            probs = probs * (~illegal)
            if probs.sum() <= 0:
                probs = (~illegal).astype(float)
            probs = probs / probs.sum()

            # map to opponent's engine move strings
            opp_name_table: dict[str, int] = {}
            try:
                opp_active = self.battle.opponent_active_pokemon
                opp_moves = consistent_move_order(
                    list(opp_active.moves.values())
                ) if opp_active else []
            except Exception:
                opp_moves = []
            try:
                opp_bench = consistent_pokemon_order(
                    [p for p in self.battle.opponent_team.values() if not p.fainted and not p.active]
                )
            except Exception:
                opp_bench = []
            for i, mv in enumerate(opp_moves[:4]):
                opp_name_table[mv.id] = i
                opp_name_table[f"{mv.id}-tera"] = i + 9
            for i, p in enumerate(opp_bench[:5]):
                opp_name_table[f"switch {norm(p.name)}"] = i + 4

            opp_priors = {}
            for name, idx in opp_name_table.items():
                opp_priors[name] = float(probs[idx])
            return opp_priors
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"WARN opponent priors failed: {e!r}", flush=True)
            return {}

    def _flip_state(self, state):
        """Swap player/opponent in a UniversalState for opponent-POV inference."""
        from metamon.interface import UniversalState
        # UniversalState fields: player_team, opponent_team, active_pokemon,
        # opponent_active_pokemon, etc. — swap them
        flipped = UniversalState.__new__(UniversalState)
        for attr in dir(state):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(state, attr)
                setattr(flipped, attr, val)
            except Exception:
                pass
        # swap player/opponent fields
        if hasattr(state, "player_team"):
            flipped.player_team = state.opponent_team
            flipped.opponent_team = state.player_team
        if hasattr(state, "active_pokemon"):
            flipped.active_pokemon = state.opponent_active_pokemon
            flipped.opponent_active_pokemon = state.active_pokemon
        if hasattr(state, "player_side_conditions"):
            flipped.player_side_conditions = state.opponent_side_conditions
            flipped.opponent_side_conditions = state.player_side_conditions
        return flipped

    def _make_opp_battle(self):
        """Create a minimal battle-like object for opponent legal-action check."""
        # The opponent's legal actions = their active mon's moves + switches
        # We can use the original battle but swap team/opponent_team refs
        class OppBattleView:
            pass
        view = OppBattleView()
        try:
            view.active_pokemon = self.battle.opponent_active_pokemon
            view.team = self.battle.opponent_team
            view.opponent_active_pokemon = self.battle.active_pokemon
            view.opponent_team = self.battle.team
            view.weather = self.battle.weather
            view.fields = self.battle.fields
            view.side_conditions = self.battle.opponent_side_conditions
            view.opponent_side_conditions = self.battle.side_conditions
            view.force_switch = False
            view.reviving = False
            view.won = False
            view.lost = False
            view.can_tera = True
            view.battle_tag = self.battle._battle_tag
            # Metamon only needs a species list for this field. The opponent's
            # original preview is not available from the flipped view, so use
            # the revealed team as a conservative proxy.
            view.teampreview_opponent_team = list(self.battle.team.values())
        except Exception:
            return None
        return view


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
        self.device = next(self.agent.parameters()).device
        self.obs_space = model.observation_space
        self.reward_fn = model.reward_function
        from belief.action_likelihood_adapter import make_frozen_r1_adapter
        self.action_likelihood_adapter = make_frozen_r1_adapter(
            self.agent, self.obs_space, self.device
        )
        self.username = args.username
        self.sessions: dict[str, BattleSession] = {}
        self.lock = threading.Lock()
        self.action_likelihood_stats = {
            "requests": 0,
            "available": 0,
            "unavailable": 0,
            "reasons": {},
        }
        # Schema-v3 observation dump (one JSONL row per /priors decision)
        dump = getattr(args, "decision_dump", None) or os.environ.get(
            "METAGROSS_PRIOR_DUMP"
        )
        self.dump_path = str(dump) if dump else None
        self.dump_lock = threading.Lock()
        if self.dump_path:
            Path(self.dump_path).parent.mkdir(parents=True, exist_ok=True)
            print(f"PRIOR_SERVER decision dump -> {self.dump_path}", flush=True)
        print("PRIOR_SERVER ready", flush=True)

    def session(self, session_id: str, tag: str, namespace: str) -> BattleSession:
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = BattleSession(
                    session_id, tag, namespace, self.username, self
                )
            return self.sessions[session_id]


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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--decision-dump",
        default=None,
        help="JSONL path: append one schema-v3 row (obs, legal mask, name table) "
        "per served /priors decision. Env fallback: METAGROSS_PRIOR_DUMP.",
    )
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
            namespace = data.get("namespace", "")
            if not isinstance(tag, str) or not isinstance(namespace, str):
                self._json(400, {"error": "tag and namespace must be strings"})
                return
            live_session = session_key(namespace, tag)
            if path == "/lines":
                sess = server.session(live_session, tag, namespace)
                with sess.lock:
                    for line in data.get("lines", []):
                        sess.feed_line(line)
                self._json(200, {"ok": True})
            elif path == "/end":
                with server.lock:
                    server.sessions.pop(live_session, None)
                self._json(200, {"ok": True})
            elif path == "/action-likelihoods":
                result = action_likelihoods_from_public_payload(server, data)
                with server.lock:
                    stats = server.action_likelihood_stats
                    stats["requests"] += 1
                    if result.get("available") is True:
                        stats["available"] += 1
                    else:
                        stats["unavailable"] += 1
                        reason = str(result.get("reason") or "unknown")
                        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                self._json(200, result)
            else:
                self._json(404, {"error": "unknown"})

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/priors":
                query = parse_qs(parsed.query)
                tag = query.get("tag", [""])[0]
                namespace = query.get("namespace", [""])[0]
                requester_username = query.get("username", [""])[0] or None
                if not isinstance(tag, str) or not isinstance(namespace, str):
                    self._json(400, {"error": "tag and namespace must be strings"})
                    return
                try:
                    sess = server.session(session_key(namespace, tag), tag, namespace)
                    with sess.lock:
                        result = sess.compute_priors(requester_username)
                    self._json(200, result)
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    self._json(500, {"error": f"{type(e).__name__}: {e}"})
            elif parsed.path == "/health":
                with server.lock:
                    action_stats = {
                        **server.action_likelihood_stats,
                        "reasons": dict(server.action_likelihood_stats["reasons"]),
                    }
                self._json(200, {"ok": True, "sessions": len(server.sessions), "action_likelihoods": action_stats})
            else:
                self._json(404, {"error": "unknown"})

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"PRIOR_SERVER listening on {args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
