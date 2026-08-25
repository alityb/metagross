#!/usr/bin/env python3
"""Belief-v2 cloud game farm — Modal app.

Milestone 1: Linux image building the VENDORED patched engine (the deployed
artifact: srcs/vendor/poke-engine, version-labelled 0.0.47 but carrying the
project's s1_priors/paired-root/shared-IS extensions) and probing its API
surface plus a search smoke. The deployed search is unseeded by design, so
farm validity rests on platform-internal pairing, not bit-equivalence.
"""

from __future__ import annotations

import json

import modal

app = modal.App("metagross-game-farm")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "git")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "pip install maturin==1.7.4",
    )
    .add_local_dir("/Users/alityb/projects/metagross/srcs/vendor/poke-engine",
                   "/build/poke-engine", copy=True,
                   ignore=["**/target/**", "**/.git/**", "**/__pycache__/**",
                           "**/dist/**"])
    .run_commands(
        ". $HOME/.cargo/env && cd /build/poke-engine/poke-engine-py && "
        "maturin build --release "
        "--features poke-engine/terastallization --no-default-features "
        "-o /dist && pip install /dist/*.whl",
    )
    .pip_install("requests==2.33.0", "websockets==14.1", "python-dateutil==2.8.0")
)


@app.function(image=image, timeout=900)
def engine_probe() -> dict:
    """API-surface parity with the deployed Mac engine + a search smoke."""
    import platform

    import poke_engine as e

    surface = sorted(n for n in dir(e) if not n.startswith("__"))
    first = e.Pokemon(id="pikachu", level=80,
                      moves=[e.Move(id="thunderbolt"), e.Move(id="voltswitch")],
                      tera_type="electric")
    reserve = e.Pokemon(id="eevee", level=80, moves=[e.Move(id="tackle")],
                        tera_type="normal")
    state = e.State(side_one=e.Side(pokemon=[first, reserve]),
                    side_two=e.Side(pokemon=[first, reserve]))
    result = e.monte_carlo_tree_search(state, duration_ms=0, iterations=4096,
                                       threads=1)
    visits = sorted(((str(r.move_choice), int(r.visits)) for r in result.side_one),
                    key=lambda x: -x[1])
    return {
        "platform": platform.platform(),
        "api_has": {name: hasattr(e, name) for name in (
            "monte_carlo_tree_search", "paired_root_policy_evaluation",
            "shared_information_set_root_search")},
        "total_visits": int(result.total_visits),
        "side_one_visits": visits,
        "api_surface_size": len(surface),
    }


REPO = "/Users/alityb/projects/metagross"


def _local_head(path: str) -> str:
    """Deploy-time capture of a local repo's HEAD; '' inside the container."""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                           capture_output=True, text=True)
        sha = r.stdout.strip()
        return sha if r.returncode == 0 and len(sha) == 40 else ""
    except OSError:
        return ""


# The provenance manifest shells out to `git rev-parse HEAD` in these roots,
# but code mounts strip .git. Capture the REAL local commits at deploy time;
# ladder runs reconstruct minimal ref-only .git stubs so the recorded commit
# is the actual commit the mounted (byte-identical) tree came from.
DEPLOY_HEADS = json.dumps({
    "srcs/vendor/foul-play": _local_head(f"{REPO}/srcs/vendor/foul-play"),
    "srcs/vendor/metamon": _local_head(f"{REPO}/srcs/vendor/metamon"),
    "srcs/metagross": _local_head(REPO),
    # battle-time cjs contract exporters run `git -C external/pokemon-showdown
    # rev-parse HEAD`; without a stub here they crash and the bot forfeits.
    "external/pokemon-showdown": _local_head(f"{REPO}/external/pokemon-showdown"),
})

CODE_IGNORE = ["**/.git/**", "**/__pycache__/**", "**/node_modules/**",
               "**/target/**", "**/*.log", "**/logs/**",
               # code mounts must never carry training artifacts: the 22GB of
               # nets/checkpoints stalled deploy mount uploads for ~45 min
               # (serving weights come from the metagross-farm-assets volume)
               "**/nets/checkpoints/**", "**/*.pt", "**/*.ckpt",
               "**/runs/**", "**/wandb/**", "**/selfplay_data_1k/**"]

assets = modal.Volume.from_name("metagross-farm-assets", create_if_missing=True)

stack_image = (
    image
    .apt_install("nodejs", "npm")
    .pip_install("torch==2.12.1", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "git+https://github.com/UT-Austin-RPL/amago@0974781a9096ff43df1b708312256f96fc2ab127",
        "gymnasium==0.29.1")
    .add_local_dir(f"{REPO}/srcs/vendor/metamon", "/repo/srcs/vendor/metamon",
                   copy=True, ignore=CODE_IGNORE)
    .run_commands("pip install -e /repo/srcs/vendor/metamon")
    .add_local_dir(f"{REPO}/external/pokemon-showdown", "/repo/external/pokemon-showdown",
                   copy=True, ignore=CODE_IGNORE)
    # The mounted dist/ is authoritative (built on the Mac); install ONLY
    # runtime deps (ts-chacha20 etc.) against it and FAIL CLOSED if the sim
    # can't load. Previously `npm ci && ./build || true` masked an install
    # failure, leaving dist present but node_modules empty -> the bot crashed
    # mid-battle on require("ts-chacha20") and forfeited every game.
    # npm install (not ci): the vendored lockfile is out of sync with
    # package.json (e.g. pg missing from the lock), and ci refuses on any
    # mismatch. We only need runtime deps for the mounted dist to load.
    .run_commands(
        "cd /repo/external/pokemon-showdown && "
        "npm install --omit=dev --no-audit --no-fund && "
        "node -e \"require('./dist/sim'); console.log('showdown sim loads OK')\"")
    .add_local_dir(f"{REPO}/srcs/vendor/foul-play", "/repo/srcs/vendor/foul-play",
                   copy=True, ignore=CODE_IGNORE)
    .add_local_dir(f"{REPO}/srcs/metagross", "/repo/srcs/metagross",
                   copy=True, ignore=CODE_IGNORE)
    .add_local_dir(f"{REPO}/experimental/src", "/repo/experimental/src",
                   copy=True, ignore=CODE_IGNORE)
    .add_local_dir(f"{REPO}/experimental/data/randbats_pools",
                   "/repo/experimental/data/randbats_pools", copy=True)
    .env({"TORCHDYNAMO_DISABLE": "1", "ACCELERATE_USE_CPU": "true",
          "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "2"})
    # The corrected causal stack refuses to serve without the experimental
    # mask-capable engine (State.s1/s2_public_reveals + with_s1_request root):
    # replace the vendored build with one from the pe_v3 crate, same feature
    # set as production gen9 serving.
    .add_local_dir(f"{REPO}/experimental/engine/pe_v3_learned_priors",
                   "/build/pe-v3", copy=True,
                   ignore=["**/target/**", "**/.git/**", "**/__pycache__/**",
                           "**/dist/**", "**/linux_wheels/**", "**/release/**"])
    .run_commands(
        ". $HOME/.cargo/env && cd /build/pe-v3/poke-engine-py && "
        "maturin build --release "
        "--features poke-engine/terastallization --no-default-features "
        "-o /dist-pe3 && pip install --force-reinstall /dist-pe3/*.whl",
    )
    .env({"METAGROSS_DEPLOY_HEADS": DEPLOY_HEADS})
    # Replicate the production venv's one-line amago patch (site-packages was
    # edited directly on the M4; upstream v3.4.0 lacks the gin registration
    # that superkazam.gin's @transformer.VanillaAttention reference needs).
    .run_commands(
        "sed -i '/^class VanillaAttention/i @gin.configurable' "
        "/usr/local/lib/python3.11/site-packages/amago/nets/transformer.py && "
        "python -c \"import gin, amago.nets.transformer;"
        "assert gin.config._REGISTRY.get_all_matches('VanillaAttention'),"
        " 'VanillaAttention not gin-registered'\"",
    )
    # Residential egress for the Showdown websocket: Tailscale userspace SOCKS
    # proxy with the owner's Mac as exit node (datacenter IPs are proxy-locked
    # by Showdown; TUN mode is unavailable in Modal, so userspace + SOCKS).
    .apt_install("curl", "ca-certificates", "iptables")
    .run_commands("curl -fsSL https://tailscale.com/install.sh | sh")
    .pip_install("python-socks==2.5.1", "async-timeout==4.0.3")
)


@app.function(image=stack_image, timeout=300)
def node_contract_probe() -> dict:
    """Reconstruct the git stubs (as the ladder run does) and run ALL three
    battle-time Showdown contract exporters, reporting each one's result."""
    import json as _j
    import os
    import subprocess
    from pathlib import Path as P
    for rel, sha in _j.loads(os.environ.get("METAGROSS_DEPLOY_HEADS", "{}")).items():
        gd = P("/repo") / rel / ".git"
        if sha and not gd.exists():
            (gd / "refs" / "heads").mkdir(parents=True, exist_ok=True)
            (gd / "objects").mkdir(exist_ok=True)
            (gd / "HEAD").write_text("ref: refs/heads/deploy\n")
            (gd / "refs" / "heads" / "deploy").write_text(sha + "\n")
    out = {"dist_sim": os.path.isdir("/repo/external/pokemon-showdown/dist/sim"),
           "dist_data": os.path.isfile(
               "/repo/external/pokemon-showdown/dist/data/moves.js")}
    for tool in ("node", "npm"):
        r = subprocess.run([tool, "--version"], capture_output=True, text=True)
        out[tool] = (r.stdout or r.stderr).strip()[:40]
    for cjs in ("export_showdown_public_form_contract.cjs",
                "export_showdown_form_ability_contract.cjs",
                "export_showdown_pressure_target_contract.cjs"):
        r = subprocess.run(["node", f"/repo/srcs/metagross/{cjs}"],
                           cwd="/repo", capture_output=True, text=True)
        out[cjs] = {"rc": r.returncode,
                    "stderr_tail": r.stderr.strip()[-300:] if r.returncode else "",
                    "ok_len": len(r.stdout) if r.returncode == 0 else 0}
    print("NODE_PROBE_RESULT " + _j.dumps(out), flush=True)
    return out


@app.function(image=stack_image, volumes={"/assets": assets}, timeout=1800, cpu=8)
def stack_smoke() -> dict:
    """M2: boot Showdown + one prior server in-container, verify both serve."""
    import os
    import subprocess
    import time
    import urllib.request

    os.makedirs("/repo/srcs/models/randbats_exit_r1/ckpts/policy_weights",
                exist_ok=True)
    os.symlink("/assets/r1/policy_epoch_5.pt",
               "/repo/srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt")
    env = dict(os.environ, PYTHONPATH="/repo:/repo/experimental/src",
               METAMON_CACHE_DIR="/tmp/metamon-cache")
    showdown = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security", "8000"],
        cwd="/repo/external/pokemon-showdown",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    prior = subprocess.Popen(
        ["python", "/repo/experimental/src/scripts/prior_server.py",
         "--local-run-dir", "/repo/srcs/models", "--local-run-name",
         "randbats_exit_r1", "--checkpoint", "5", "--port", "8977",
         "--username", "smoke"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = {"showdown": False, "prior": False}
    deadline = time.time() + 600
    while time.time() < deadline and not all(out.values()):
        time.sleep(5)
        for name, url in (("showdown", "http://127.0.0.1:8000/"),
                          ("prior", "http://127.0.0.1:8977/health")):
            if not out[name]:
                try:
                    with urllib.request.urlopen(url, timeout=3) as resp:
                        out[name] = resp.status == 200
                except Exception:
                    pass
        for name, proc in (("showdown_proc", showdown), ("prior_proc", prior)):
            if proc.poll() is not None:
                out[name + "_died"] = proc.stdout.read().decode()[-1500:]
                deadline = 0
    showdown.terminate()
    prior.terminate()
    return out


@app.function(image=stack_image, volumes={"/assets": assets}, timeout=43200, cpu=16)
def run_games(spec: dict) -> dict:
    """M3: one container = one lane. Boots Showdown + two isolated prior
    servers, plays `n_games` paired games of agent_a vs agent_b via the
    frozen eval harness, returns progress rows. Every lane is
    platform-internal: both arms share this container and budget."""
    import os
    import subprocess
    import time
    import urllib.request

    os.makedirs("/repo/srcs/models/randbats_exit_r1/ckpts/policy_weights",
                exist_ok=True)
    link = "/repo/srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"
    if not os.path.exists(link):
        os.symlink("/assets/r1/policy_epoch_5.pt", link)
    env = dict(os.environ, PYTHONPATH="/repo:/repo/experimental/src",
               METAMON_CACHE_DIR="/tmp/metamon-cache")
    # Client-side env for the eval harness (e.g. the D-gate temperature
    # schedule, which is mode-gated in the client so a global env cleanly
    # differentiates a causal candidate from a stateless baseline).
    env.update({str(k): str(v) for k, v in (spec.get("client_env") or {}).items()})
    env["METAGROSS_SEARCH_TELEMETRY_DIR"] = "/tmp/telemetry"
    # CODE_IGNORE strips **/logs/** from mounts, but Showdown scandirs
    # logs/repl at startup and dies without it.
    os.makedirs("/repo/external/pokemon-showdown/logs/repl", exist_ok=True)
    # The eval harness's mirrored-pair planner records the Showdown commit via
    # `git rev-parse HEAD`; mounts strip .git, so reconstruct the ref-only
    # stubs (real deploy-time HEADs) exactly as the ladder runner does.
    from pathlib import Path as _P
    for rel, sha in json.loads(
            os.environ.get("METAGROSS_DEPLOY_HEADS", "{}")).items():
        gd = _P("/repo") / rel / ".git"
        if sha and not gd.exists():
            (gd / "refs" / "heads").mkdir(parents=True, exist_ok=True)
            (gd / "objects").mkdir(exist_ok=True)
            (gd / "HEAD").write_text("ref: refs/heads/deploy\n")
            (gd / "refs" / "heads" / "deploy").write_text(sha + "\n")
    procs = [subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security", "8000"],
        cwd="/repo/external/pokemon-showdown",
        stdout=open("/tmp/showdown.log", "w"), stderr=subprocess.STDOUT)]
    # PRODUCTION prior servers (srcs/metagross) — trajectory-mode aware, and
    # the site of the C-gate truncation hook. Per-arm mode + env injection.
    for port, name, mode_key, env_key in (
            (8977, "lane-a", "agent_a_trajectory_mode", "prior_env_a"),
            (8978, "lane-b", "agent_b_trajectory_mode", "prior_env_b")):
        server_env = dict(env)
        server_env.update(
            {str(k): str(v) for k, v in (spec.get(env_key) or {}).items()})
        log_path = f"/tmp/prior-{name}.log"
        procs.append(subprocess.Popen(
            ["python", "-u", "/repo/srcs/metagross/prior_server.py",
             "--local-run-dir", "/repo/srcs/models", "--local-run-name",
             "randbats_exit_r1", "--checkpoint", "5",
             "--checkpoint-sha256",
             "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93",
             "--port", str(port), "--username", name,
             "--trajectory-mode", spec.get(mode_key, "causal-history")],
            env=server_env, stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT))
    deadline = time.time() + 600
    ready = set()
    while time.time() < deadline and len(ready) < 3:
        time.sleep(5)
        for port in (8000, 8977, 8978):
            if port not in ready:
                try:
                    url = (f"http://127.0.0.1:{port}/" if port == 8000
                           else f"http://127.0.0.1:{port}/health")
                    with urllib.request.urlopen(url, timeout=3) as resp:
                        if resp.status == 200:
                            ready.add(port)
                except Exception:
                    pass
    if len(ready) < 3:
        tails = {}
        for path in ("/tmp/showdown.log", "/tmp/prior-lane-a.log",
                     "/tmp/prior-lane-b.log"):
            try:
                tails[path] = open(path, errors="replace").read()[-1200:]
            except OSError:
                tails[path] = "(missing)"
        return {"error": f"infrastructure not ready: {sorted(ready)}",
                "log_tails": tails}
    out_dir = f"/tmp/lane-{spec['lane']}"
    os.makedirs(f"{out_dir}/pair-registration", exist_ok=True)
    os.makedirs(f"{out_dir}/logs", exist_ok=True)
    command = [
        "python", "-m", "eval.run", "--mode", "h2h", "--server", "local",
        "--format", "gen9randombattle",
        "--agent-a", spec["agent_a"], "--agent-b", spec["agent_b"],
        "--agent-a-prior-server-url", "http://127.0.0.1:8977",
        "--agent-b-prior-server-url", "http://127.0.0.1:8978",
        "--agent-a-require-priors", "--agent-b-require-priors",
        "--strict-isolated-priors",
        "--randbats-belief-pool",
        "/repo/experimental/data/randbats_pools/gen9randombattle_pool_50000.json",
        "--foul-play-python", "python",
        "--foul-play-search-time-ms", str(spec.get("search_time_ms", 500)),
        "--foul-play-search-parallelism", "8", "--foul-play-search-threads", "1",
        "--n-games", str(spec["n_games"]), "--paired", "--fail-fast",
        "--mirrored-pairs", "--mirror-seed", str(spec["mirror_seed"]),
        "--pair-registration-dir", f"{out_dir}/pair-registration",
        "--username-prefix", f"l{spec['lane']:03d}",
        "--run-id", spec["run_id"],
        "--json-out", f"{out_dir}/result.json", "--log-dir", f"{out_dir}/logs",
    ]
    completed = subprocess.run(command, env=env, cwd="/repo",
                               capture_output=True, text=True, timeout=40000)
    result = {"lane": spec["lane"], "exit": completed.returncode}
    for name in ("result.json", "result.json.progress.jsonl"):
        path = f"{out_dir}/{name}"
        if os.path.exists(path):
            result[name] = open(path).read()
    # Mandatory activation evidence (prereg): the candidate arm must have
    # logged its ACTIVE line — D in a client log, C in a prior-server log.
    activation = []
    import glob as _glob
    for path in _glob.glob("/tmp/prior-lane-*.log") + _glob.glob(f"{out_dir}/logs/**/*", recursive=True):
        try:
            if os.path.isfile(path):
                for line in open(path, errors="replace"):
                    if "ACTIVE" in line and ("temperature" in line or "TRUNCATION" in line):
                        activation.append(f"{os.path.basename(path)}: {line.strip()[:160]}")
                        break
        except OSError:
            pass
    result["activation_evidence"] = activation
    # Persist search telemetry + prior-server logs to the volume for the
    # cross-arm comparison analysis (prior/visit entropy by turn, flip rates).
    keep_dir = f"/assets/dc-gates/{spec.get('run_id', 'unlabeled')}"
    os.makedirs(keep_dir, exist_ok=True)
    import shutil as _shutil
    telemetry_summary = {}
    for path in _glob.glob("/tmp/telemetry/*.jsonl") + _glob.glob("/tmp/prior-lane-*.log"):
        try:
            _shutil.copy(path, keep_dir)
            if path.endswith(".jsonl"):
                n = flips = 0
                for line in open(path, errors="replace"):
                    n += 1
                    if '"flip": true' in line:
                        flips += 1
                telemetry_summary[os.path.basename(path)] = {
                    "decisions": n, "gumbel_flips": flips}
        except OSError:
            pass
    result["telemetry_summary"] = telemetry_summary
    for name in ("result.json", "result.json.progress.jsonl"):
        src = f"{out_dir}/{name}"
        if os.path.exists(src):
            try:
                _shutil.copy(src, keep_dir)
            except OSError:
                pass
    assets.commit()
    if completed.returncode != 0:
        result["stderr_tail"] = completed.stderr[-8000:]
    for proc in procs:
        proc.terminate()
    return result



_TS_SOCKS_PORT = 1055        # container-local Tailscale userspace SOCKS
_MAC_SOCKS_PORT = 1055       # microsocks on the Mac, bound to its tailnet IP
_RELAY_PORT = 1080           # local endpoint the Showdown client dials
_CLOUD_ASN_MARKERS = ("amazon", "aws", "google", "gcp", "microsoft", "azure",
                      "oracle", "digitalocean", "linode", "hetzner", "ovh",
                      "cloudflare", "vultr", "modal")


def _start_socks_relay(mac_ip: str) -> None:
    """Expose the Mac's tailnet-bound microsocks at 127.0.0.1:_RELAY_PORT.

    macOS Homebrew tailscaled advertises as an exit node but does not forward,
    so instead the Mac runs a plain SOCKS server and we reach it over the
    tailnet. In userspace mode the container cannot dial a 100.x address
    directly, so this byte-pipe forwards each local connection through the
    container's Tailscale SOCKS to the Mac's SOCKS. The Showdown client then
    speaks ordinary SOCKS5 to 127.0.0.1:_RELAY_PORT and its handshake
    terminates at the Mac -> egress is the home IP."""
    import asyncio
    import threading
    from python_socks.async_.asyncio import Proxy

    async def handle(reader, writer):
        up_writer = None
        try:
            proxy = Proxy.from_url(f"socks5://127.0.0.1:{_TS_SOCKS_PORT}")
            up_sock = await proxy.connect(dest_host=mac_ip, dest_port=_MAC_SOCKS_PORT)
            up_reader, up_writer = await asyncio.open_connection(sock=up_sock)

            async def pipe(r, w):
                try:
                    while True:
                        data = await r.read(65536)
                        if not data:
                            break
                        w.write(data)
                        await w.drain()
                except Exception:
                    pass
                finally:
                    try:
                        w.close()
                    except Exception:
                        pass

            await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))
        except Exception:
            for w in (writer, up_writer):
                try:
                    if w is not None:
                        w.close()
                except Exception:
                    pass

    async def serve():
        server = await asyncio.start_server(handle, "127.0.0.1", _RELAY_PORT)
        async with server:
            await server.serve_forever()

    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(serve())

    threading.Thread(target=run, daemon=True).start()


def _bring_up_residential_egress(mac_ip: str) -> str:
    """Join the tailnet, relay to the Mac's SOCKS, and FAIL CLOSED unless
    egress through the relay is a residential (non-cloud) IP distinct from this
    container's datacenter IP. Returns the socks5h URL for the Showdown client.
    Raises on any failure so a run can never silently leak the datacenter IP
    Showdown proxy-locks."""
    import json as _json
    import os
    import subprocess
    import time

    authkey = os.environ["TS_AUTHKEY"]
    subprocess.Popen(
        ["tailscaled", "--tun=userspace-networking",
         f"--socks5-server=localhost:{_TS_SOCKS_PORT}",
         "--state=/tmp/ts.state", "--socket=/tmp/tailscaled.sock"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)
    up = subprocess.run(
        ["tailscale", "--socket=/tmp/tailscaled.sock", "up",
         f"--authkey={authkey}",
         f"--hostname=metagross-{os.environ.get('METAGROSS_ARM_LABEL', 'arm')}",
         "--accept-dns=false"],
        capture_output=True, text=True, timeout=120)
    if up.returncode != 0:
        raise RuntimeError(f"tailscale up failed: {up.stderr.strip()[:300]}")
    # Warm the path to the Mac (DERP handshake) before relaying.
    subprocess.run(["tailscale", "--socket=/tmp/tailscaled.sock", "ping",
                    "-c", "2", mac_ip], capture_output=True, text=True, timeout=40)
    _start_socks_relay(mac_ip)
    time.sleep(2)

    # python_socks accepts 'socks5' (remote DNS by default), not curl's 'socks5h'.
    socks = f"socks5://127.0.0.1:{_RELAY_PORT}"
    direct = subprocess.run(["curl", "-s", "--max-time", "20",
                             "https://ipinfo.io/json"],
                            capture_output=True, text=True)
    proxied = None
    for _ in range(12):
        proxied = subprocess.run(
            ["curl", "-s", "--max-time", "25", "--socks5-hostname",
             f"127.0.0.1:{_RELAY_PORT}", "https://ipinfo.io/json"],
            capture_output=True, text=True)
        if proxied.returncode == 0 and proxied.stdout.strip():
            break
        time.sleep(6)
    try:
        d = _json.loads(direct.stdout or "{}")
        p = _json.loads(proxied.stdout or "{}")
    except ValueError:
        raise RuntimeError(f"egress probe returned non-JSON: "
                           f"direct={direct.stdout[:120]!r} "
                           f"proxied={(proxied.stdout if proxied else '')[:120]!r} "
                           f"(is microsocks running on the Mac?)")
    p_ip, d_ip = p.get("ip"), d.get("ip")
    p_org = (p.get("org") or "").lower()
    print(f"EGRESS_GATE direct_ip={d_ip} proxied_ip={p_ip} proxied_org={p_org!r} "
          f"city={p.get('city')!r}", flush=True)
    if not p_ip:
        raise RuntimeError("no egress IP through relay (is microsocks running on the Mac?)")
    if p_ip == d_ip:
        raise RuntimeError("proxied egress == datacenter egress; relay not routing")
    if any(m in p_org for m in _CLOUD_ASN_MARKERS):
        raise RuntimeError(f"proxied egress still a cloud network ({p_org!r})")
    return socks


def _ladder_run_impl(spec: dict) -> dict:
    """One public-ladder account campaign in one container.

    spec: {"arm": "a"|"b", "trajectory_mode": "causal-history"|"legacy-stateless",
           "blocks": int, "label": str}
    Each arm's function attaches ONLY its own secret (the two owner-created
    secrets both use PSA_-prefixed key names, so attaching both would collide);
    PSB_ names are also accepted in case the secret is later renamed.
    Results persist to /assets/ladder-results/<label>/ after every block.
    """
    import hashlib
    import importlib
    import os
    import subprocess
    from pathlib import Path as P

    username = os.environ.get("PSA_USERNAME") or os.environ["PSB_USERNAME"]
    password = os.environ.get("PSA_PASSWORD") or os.environ["PSB_PASSWORD"]

    # Residential egress for the Showdown websocket (fail-closed).
    os.environ["METAGROSS_ARM_LABEL"] = spec["label"]
    socks_url = _bring_up_residential_egress(spec["exit_node"])

    # Reconstruct ref-only .git stubs so provenance's `git rev-parse HEAD`
    # reports the real deploy-time commit of each mounted tree.
    import json as _json
    for rel, sha in _json.loads(os.environ.get("METAGROSS_DEPLOY_HEADS", "{}")).items():
        gd = P("/repo") / rel / ".git"
        if sha and not gd.exists():
            (gd / "refs" / "heads").mkdir(parents=True, exist_ok=True)
            (gd / "objects").mkdir(exist_ok=True)
            (gd / "HEAD").write_text("ref: refs/heads/deploy\n")
            (gd / "refs" / "heads" / "deploy").write_text(sha + "\n")
    os.makedirs("/repo/srcs/models/randbats_exit_r1/ckpts/policy_weights",
                exist_ok=True)
    link = "/repo/srcs/models/randbats_exit_r1/ckpts/policy_weights/policy_epoch_5.pt"
    if not os.path.exists(link):
        os.symlink("/assets/r1/policy_epoch_5.pt", link)
    native = P(importlib.import_module("poke_engine.poke_engine").__file__)
    env = dict(os.environ,
               PYTHONPATH="/repo:/repo/experimental/src",
               METAMON_CACHE_DIR="/tmp/metamon-cache",
               METAGROSS_SHOWDOWN_PASSWORD=password,
               METAGROSS_TRAJECTORY_MODE=spec["trajectory_mode"],
               METAGROSS_SEARCH_ITERATIONS_PER_500MS=str(spec.get("iterations_per_500ms", 472000)),
               METAGROSS_PINNED_ENGINE_IMPORT_ROOT=str(native.parents[1]),
               METAGROSS_WEBSOCKET_SOCKS=socks_url,
               METAGROSS_PINNED_ENGINE_SHA256=hashlib.sha256(
                   native.read_bytes()).hexdigest())
    out_root = f"/assets/ladder-results/{spec['label']}"
    os.makedirs(out_root, exist_ok=True)

    def _free_prior_port(port=8977):
        # A prior server stuck mid-model-load ignores SIGTERM from launch.py's
        # cleanup and keeps holding the port, so the next block's bind fails
        # with EADDRINUSE. SIGKILL any lingering prior server (dependency-free
        # /proc scan) and wait for the port to actually free.
        import glob
        import signal
        import socket
        import time as _t
        for cmdline in glob.glob("/proc/*/cmdline"):
            try:
                argv = open(cmdline, "rb").read().replace(b"\x00", b" ").decode(
                    "utf-8", "ignore")
                if "prior_server.py" in argv:
                    os.kill(int(cmdline.split("/")[2]), signal.SIGKILL)
            except (OSError, ValueError):
                pass
        for _ in range(30):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return
            _t.sleep(1)

    summary = []
    for block in range(int(spec["blocks"])):
        _free_prior_port()
        completed = subprocess.run(
            ["python", "-m", "srcs.metagross.launch",
             "--username", username, "--profile", "r1",
             "--games", "25",
             "--output-root", f"{out_root}/block-{block:02d}",
             "--foul-play-python", "python",
             "--metamon-python", "python",
             "--search-parallelism", "8",
             "--port", "8977"],
            cwd="/repo", env=env, capture_output=True, text=True,
            timeout=14000)
        summary.append({"block": block, "exit": completed.returncode,
                        "stderr_tail": completed.stderr[-500:]
                        if completed.returncode else ""})
        assets.commit()
    return {"label": spec["label"], "username_masked": username[:3] + "***",
            "blocks": summary}


@app.function(image=stack_image, volumes={"/assets": assets}, timeout=86000,
              cpu=8, secrets=[modal.Secret.from_name("metagross-ladder-a"),
                              modal.Secret.from_name("metagross-tailscale")])
def ladder_run_a(spec: dict) -> dict:
    return _ladder_run_impl(spec)


@app.function(image=stack_image, volumes={"/assets": assets}, timeout=86000,
              cpu=8, secrets=[modal.Secret.from_name("metagross-ladder-b"),
                              modal.Secret.from_name("metagross-tailscale")])
def ladder_run_b(spec: dict) -> dict:
    return _ladder_run_impl(spec)


@app.local_entrypoint()
def main(milestone: str = "1"):
    if milestone == "1":
        print(json.dumps(engine_probe.remote(), indent=1))
    elif milestone == "2":
        print(json.dumps(stack_smoke.remote(), indent=1))
    elif milestone == "3":
        result = run_games.remote({
            "lane": 0, "agent_a": "foul_play_randbats_conditional_root_priors_opp",
            "agent_b": "foul_play_randbats_conditional_root_priors_opp",
            "n_games": 2, "mirror_seed": 2026081799,
            "run_id": "farm-m3-smoke"})
        print(json.dumps({k: (v[:400] if isinstance(v, str) else v)
                          for k, v in result.items()}, indent=1))
