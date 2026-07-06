#!/usr/bin/env python3
"""Patch local Metamon checkout for CPU inference and public Showdown ladder.

This is intentionally a small, reversible local patch script.  It applies the
changes needed to run public Metamon checkpoints on this Mac/CPU setup:

- Replace FlashAttention gin bindings with VanillaAttention.
- Register AMAGO VanillaAttention as a gin configurable.
- Add PublicShowdownLadder to Metamon and expose `--eval_type public_ladder`.

Run after installing Metamon editable:

    METAMON_CACHE_DIR=external/metamon_cache .venv-metamon/bin/python \
      scripts/patch_metamon_cpu_public_ladder.py
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METAMON = ROOT / "external" / "metamon"
VENV = ROOT / ".venv-metamon"


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        return
    path.write_text(text.replace(old, new))
    print(f"patched {path.relative_to(ROOT)}")


def ensure_contains(path: Path, marker: str, insertion_point: str, insert: str) -> None:
    text = path.read_text()
    if marker in text:
        return
    if insertion_point not in text:
        raise RuntimeError(f"insertion point not found in {path}: {insertion_point!r}")
    path.write_text(text.replace(insertion_point, insertion_point + insert, 1))
    print(f"patched {path.relative_to(ROOT)}")


def insert_before(path: Path, marker: str, insertion_point: str, insert: str) -> None:
    text = path.read_text()
    if marker in text:
        return
    if insertion_point not in text:
        raise RuntimeError(f"insertion point not found in {path}: {insertion_point!r}")
    path.write_text(text.replace(insertion_point, insert + insertion_point, 1))
    print(f"patched {path.relative_to(ROOT)}")


def main() -> None:
    # CPU-safe attention in all Metamon gin configs.
    model_dir = METAMON / "metamon" / "rl" / "configs" / "models"
    for gin in model_dir.glob("*.gin"):
        text = gin.read_text()
        changed = False
        if "@transformer.FlashAttention" in text:
            text = text.replace("@transformer.FlashAttention", "@transformer.VanillaAttention")
            changed = True
        lines = []
        for line in text.splitlines():
            if "transformer.FlashAttention.window_size" in line:
                changed = True
                continue
            lines.append(line)
        if changed:
            gin.write_text("\n".join(lines) + "\n")
            print(f"patched {gin.relative_to(ROOT)}")

    # AMAGO's VanillaAttention exists but is not gin-configurable in v3.4.0.
    for transformer in VENV.glob("lib/python*/site-packages/amago/nets/transformer.py"):
        text = transformer.read_text()
        if "@gin.configurable\nclass VanillaAttention" not in text:
            replace(transformer, "class VanillaAttention(SelfAttention):", "@gin.configurable\nclass VanillaAttention(SelfAttention):")

    # AMAGO 3.4 can configure save_every as floats; Python 3.12's randint
    # requires integer bounds.
    for amago_env in VENV.glob("lib/python*/site-packages/amago/envs/amago_env.py"):
        replace(
            amago_env,
            "    def random_traj_length(self):\n        return random.randint(*self.save_every) if self.save_every else None\n",
            "    def random_traj_length(self):\n        if not self.save_every:\n            return None\n        low, high = (int(bound) for bound in self.save_every)\n        return random.randint(low, high)\n",
        )

    wrappers = METAMON / "metamon" / "env" / "wrappers.py"
    ensure_contains(
        wrappers,
        "PublicShowdownServerConfiguration",
        'PokeAgentServerConfiguration = ServerConfiguration(\n    "wss://battling.pokeagentchallenge.com/showdown/websocket",\n    "https://battling.pokeagentchallenge.com/action.php?",\n)\n',
        '\n\nPublicShowdownServerConfiguration = ServerConfiguration(\n    "wss://sim3.psim.us/showdown/websocket",\n    "https://play.pokemonshowdown.com/action.php?",\n)\n',
    )
    ensure_contains(
        wrappers,
        "class PublicShowdownLadder",
        "class PokeAgentLadder(QueueOnLocalLadder):\n",
        "",
    )
    if "class PublicShowdownLadder" not in wrappers.read_text():
        wrappers.write_text(wrappers.read_text() + '''\n\nclass PublicShowdownLadder(QueueOnLocalLadder):\n    \"\"\"Battle against the public Pokémon Showdown ladder.\"\"\"\n\n    _INIT_RETRIES = 3000\n\n    @property\n    def server_configuration(self):\n        return PublicShowdownServerConfiguration\n\n    def handle_ladder_start(self, n_challenges: int):\n        assert (\n            self.player_username is not None and self.player_password is not None\n        ), \"Username and password are required for public Showdown laddering\"\n        super().start_laddering(n_challenges)\n''')
        print(f"patched {wrappers.relative_to(ROOT)}")

    env_init = METAMON / "metamon" / "env" / "__init__.py"
    ensure_contains(
        env_init,
        "PublicShowdownLadder",
        "    PokeAgentLadder,\n",
        "    PublicShowdownServerConfiguration,\n    PublicShowdownLadder,\n",
    )

    metamon_to_amago = METAMON / "metamon" / "rl" / "metamon_to_amago.py"
    ensure_contains(
        metamon_to_amago,
        "PublicShowdownLadder",
        "    PokeAgentLadder,\n",
        "    PublicShowdownLadder,\n",
    )
    insert_before(
        metamon_to_amago,
        "def make_public_ladder_env",
        "\n\ndef make_challenge_env(*args, **kwargs):\n",
        '''\n\ndef make_public_ladder_env(*args, **kwargs):\n    \"\"\"Battle on the public Pokémon Showdown ladder.\"\"\"\n    _block_warnings()\n    menv = PublicShowdownLadder(*args, **kwargs)\n    print(\"Made Public Showdown Ladder Env\")\n    return PSLadderAMAGOWrapper(menv)\n''',
    )

    eval_main = METAMON / "metamon" / "rl" / "evaluate" / "__main__.py"
    ensure_contains(eval_main, "import os\n", "import functools\n", "import os\n")
    ensure_contains(
        eval_main,
        "make_public_ladder_env",
        "    make_pokeagent_ladder_env,\n",
        "    make_public_ladder_env,\n",
    )
    insert_before(
        eval_main,
        "def pretrained_vs_public_ladder",
        "\n\ndef pretrained_vs_challenge(\n",
        '''\n\ndef pretrained_vs_public_ladder(\n    pretrained_model: PretrainedModel,\n    username: str,\n    password: str,\n    battle_format: str,\n    team_set: metamon.env.TeamSet,\n    total_battles: int,\n    avatar: Optional[str] = None,\n    checkpoint: Optional[int] = None,\n    battle_backend: str = \"metamon\",\n    action_temperature: float = 1.0,\n    save_trajectories_to: Optional[str] = None,\n    save_results_to: Optional[str] = None,\n    log_to_wandb: bool = False,\n    team_preview_model: Optional[TeamPreviewModel] = None,\n) -> Dict[str, Any]:\n    return _pretrained_on_ladder(\n        pretrained_model=pretrained_model,\n        make_ladder=make_public_ladder_env,\n        total_battles=total_battles,\n        checkpoint=checkpoint,\n        log_to_wandb=log_to_wandb,\n        action_temperature=action_temperature,\n        team_preview_model=team_preview_model,\n        player_username=username,\n        player_password=password,\n        player_avatar=avatar,\n        player_team_set=team_set,\n        battle_backend=battle_backend,\n        battle_format=battle_format,\n        save_trajectories_to=save_trajectories_to,\n        save_results_to=save_results_to,\n    )\n''',
    )
    replace(eval_main, 'choices=["heuristic", "il", "ladder", "pokeagent", "challenge"]', 'choices=["heuristic", "il", "ladder", "pokeagent", "public_ladder", "challenge"]')
    replace(eval_main, '        "--password",\n        default=None,', '        "--password",\n        default=os.environ.get("METAMON_SHOWDOWN_PASSWORD"),')
    insert_before(
        eval_main,
        'elif args.eval_type == "public_ladder"',
        '    elif args.eval_type == "challenge":\n',
        '''    elif args.eval_type == "public_ladder":\n        base_eval_kwargs.update(\n            {\n                "username": args.username,\n                "password": args.password,\n                "avatar": args.avatar,\n            }\n        )\n        return pretrained_vs_public_ladder\n''',
    )


if __name__ == "__main__":
    main()
