#!/usr/bin/env python3
"""Run the frozen r1 policy directly, without Foul Play search.

Action contract: on every decision, Metamon constructs its definitely-valid
action set (including forced-switch handling), masks every other action, and
selects the lowest-index maximizer of the masked r1 policy logits.  There is no
sampling or random invalid-action fallback on a non-terminal decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
R1_RUN_NAME = "randbats_exit_r1"
R1_CHECKPOINT = 5
R1_SHA256 = "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93"
R1_BASE_MODEL = "Kakuna"
R1_OBSERVATION_FORMAT = "gen9randombattle"
R1_ACTION_CONTRACT = "masked-legal-argmax-lowest-index-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_r1_checkpoint(checkpoint_root: Path) -> Path:
    path = (
        checkpoint_root
        / R1_RUN_NAME
        / "ckpts"
        / "policy_weights"
        / f"policy_epoch_{R1_CHECKPOINT}.pt"
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen r1 checkpoint not found: {path}")
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, R1_SHA256):
        raise RuntimeError(
            f"frozen r1 checkpoint SHA-256 mismatch: expected {R1_SHA256}, got {actual}"
        )
    return path


def configure_direct_experiment(experiment, model) -> None:
    """Freeze eval selection and bind the native token to r1's trained Gen 9 row."""
    trained_token = "<gen9ou>"
    native_token = f"<{R1_OBSERVATION_FORMAT}>"
    trained_token_id = model.tokenizer[trained_token]
    if trained_token_id < 0:
        raise RuntimeError(f"frozen r1 tokenizer lacks required token {trained_token}")
    # Do this after model construction/checkpoint loading so the embedding shape
    # remains exactly the frozen checkpoint shape.
    model.tokenizer._initial_ids[native_token] = trained_token_id
    experiment.sample_actions_val = False


def install_fail_closed_invalid_action(metamon_wrappers) -> None:
    def fail_invalid_order(_self, _battle):
        # The AMAGO wrapper catches Exception and converts it to a truncated
        # episode. SystemExit deliberately escapes that recovery path.
        raise SystemExit("direct r1 selected an action outside the legal mask")

    metamon_wrappers.PokeEnvWrapper.on_invalid_order = fail_invalid_order


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--opponent-username", required=True)
    parser.add_argument("--role", choices=["challenger", "acceptor"], required=True)
    parser.add_argument("--battle-format", default=R1_OBSERVATION_FORMAT)
    parser.add_argument("--websocket-uri", default="ws://localhost:8000/showdown/websocket")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=WORKSPACE_ROOT / "srcs" / "models"
    )
    parser.add_argument("--save-results-to", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.battle_format != R1_OBSERVATION_FORMAT:
        parser.error(f"frozen direct r1 supports only {R1_OBSERVATION_FORMAT}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    checkpoint_path = verify_r1_checkpoint(checkpoint_root)
    args.save_results_to.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "disabled")

    from poke_env.ps_client.server_configuration import ServerConfiguration
    import poke_env.player.player as player_module

    server_configuration = ServerConfiguration(
        args.websocket_uri,
        "https://play.pokemonshowdown.com/action.php?",
    )
    player_module.LocalhostServerConfiguration = server_configuration

    from metamon import config as metamon_config

    # Preserve the real Random Battle token in UniversalState observations.
    metamon_config.FORMAT_ALIASES[args.battle_format] = R1_OBSERVATION_FORMAT

    import metamon.env.wrappers as metamon_wrappers
    import metamon.rl.pretrained as pretrained
    from metamon.rl.evaluate.__main__ import pretrained_vs_challenge

    metamon_wrappers.LocalhostServerConfiguration = server_configuration
    install_fail_closed_invalid_action(metamon_wrappers)
    model = pretrained.LocalFinetunedModel(
        base_model=getattr(pretrained, R1_BASE_MODEL),
        amago_ckpt_dir=str(checkpoint_root),
        model_name=R1_RUN_NAME,
        default_checkpoint=R1_CHECKPOINT,
    )
    initialize_agent = model.initialize_agent

    def initialize_deterministic_agent(*init_args, **init_kwargs):
        experiment = initialize_agent(*init_args, **init_kwargs)
        configure_direct_experiment(experiment, model)
        return experiment

    model.initialize_agent = initialize_deterministic_agent
    identity = {
        "run_name": R1_RUN_NAME,
        "checkpoint": R1_CHECKPOINT,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": R1_SHA256,
        "observation_format_token": R1_OBSERVATION_FORMAT,
        "action_contract": R1_ACTION_CONTRACT,
    }
    print(f"DIRECT_R1_IDENTITY {json.dumps(identity, sort_keys=True)}", flush=True)
    results = pretrained_vs_challenge(
        pretrained_model=model,
        username=args.username,
        opponent_username=args.opponent_username,
        role=args.role,
        battle_format=args.battle_format,
        team_set=None,
        total_battles=1,
        checkpoint=R1_CHECKPOINT,
        action_temperature=1.0,
        save_results_to=str(args.save_results_to),
    )

    csv_path = args.save_results_to / f"battle_log_{args.username}_{args.battle_format}.csv"
    rows = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, skipinitialspace=True))
        if len(rows) == 1 and rows[0].get("Result") in {"WIN", "LOSS"}:
            break
        time.sleep(0.5)
    if len(rows) != 1 or rows[0].get("Result") not in {"WIN", "LOSS"}:
        detail = csv_path.read_text(encoding="utf-8") if csv_path.exists() else "<missing>"
        raise RuntimeError(
            f"expected exactly one decisive direct-r1 result in {csv_path}; "
            f"rows={len(rows)} content={detail!r}"
        )
    winner = args.username if rows[0]["Result"] == "WIN" else args.opponent_username
    print(f"Winner: {winner}", flush=True)
    print(f"DIRECT_R1_RESULTS {json.dumps(results, sort_keys=True, default=str)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
