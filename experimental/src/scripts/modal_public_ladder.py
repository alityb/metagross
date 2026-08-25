"""Run a pinned public-ladder campaign on Modal CPU containers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import modal


SCRIPT_PATH = Path(__file__).resolve()
IS_LOCAL_CHECKOUT = len(SCRIPT_PATH.parents) > 3 and (SCRIPT_PATH.parents[3] / "srcs").is_dir()
ROOT = SCRIPT_PATH.parents[3] if IS_LOCAL_CHECKOUT else Path("/workspace")
AMAGO = ROOT / ".venv-metamon" / "lib" / "python3.11" / "site-packages" / "amago"
APP_NAME = "metagross-public-ladder"
VOLUME_NAME = "metagross-public-ladder"
SECRET_NAME = "metagross-public-ladder"
APP = modal.App(APP_NAME)
app = APP
VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
SECRET = modal.Secret.from_name(SECRET_NAME)
FORMAT = "gen9randombattle"
WEBSOCKET_URI = "wss://sim3.psim.us/showdown/websocket"
MAX_BLOCKS_PER_SEGMENT = 8
DEPLOYMENT_VARIANT = "r1-p16-cloud-max"
CLOUD_CPUS = 32.0
CLOUD_MEMORY_MIB = 32768
R1_SEARCH_PARALLELISM = 16
R1_SEARCH_THREADS = 1
ARTIFACT_COMMIT_SECONDS = 60
NETWORK_ERROR_MARKERS = (
    "websockets.exceptions.connectionclosed",
    "connectionclosederror",
    "no close frame received",
    "keepalive ping timeout",
    "connection reset by peer",
    "remote host closed",
    "opening handshake failed",
    "invalidproxystatus",
    "proxyerror",
)
PROFILES = {
    "r1": {
        "run_name": "randbats_exit_r1",
        "checkpoint": 5,
        "sha256": "c6a4c0f571b8066e7471727dc82598e3a825256ec5391fab4ea55a6f16781d93",
    },
    "g3": {
        "run_name": "randbats_online_g3_autonomous_freshfix_20260729",
        "checkpoint": 1,
        "sha256": "0c754bb96953b900e282de91c570aaae5c2c6f002dc2419e149d01132888815c",
    },
}


def _build_image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("build-essential", "curl", "git")
        .pip_install(
            "accelerate",
            "datasets",
            "einops",
            "gin-config",
            "gymnasium==0.29.1",
            "huggingface_hub",
            "lz4",
            "maturin>=1.0,<2.0",
            "numpy==2.4.6",
            "pandas",
            "ratarmountcore",
            "rich",
            "scipy",
            "termcolor",
            "torch==2.12.1",
            "tqdm",
            "wandb",
            "poke-env @ git+https://github.com/UT-Austin-RPL/poke-env.git@e1268d270c3f2bd32c7ff5713e01062302020579",
            "amago @ git+https://github.com/UT-Austin-RPL/amago@0974781a9096ff43df1b708312256f96fc2ab127",
        )
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y --profile minimal",
        )
    )
    if not IS_LOCAL_CHECKOUT:
        return image
    image = (
        image.add_local_dir(
            ROOT / "srcs" / "metagross",
            "/workspace/srcs/metagross",
            copy=True,
            ignore=["__pycache__", "*.pyc", "tests"],
        )
        .add_local_dir(
            ROOT / "srcs" / "vendor" / "metamon" / "metamon",
            "/usr/local/lib/python3.11/site-packages/metamon",
            copy=True,
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_dir(
            AMAGO,
            "/usr/local/lib/python3.11/site-packages/amago",
            copy=True,
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_dir(
            ROOT / "srcs" / "vendor" / "foul-play",
            "/workspace/srcs/vendor/foul-play",
            copy=True,
            ignore=[".git", "__pycache__", "*.pyc", "external", "tests"],
        )
        .add_local_dir(
            ROOT / "srcs" / "vendor" / "poke-engine",
            "/workspace/srcs/vendor/poke-engine",
            copy=True,
            ignore=[".git", "__pycache__", "*.pyc", "linux_wheels", "release", "target"],
        )
        .run_commands(
            "python -m venv --system-site-packages /workspace/.venv-fp-priors",
            "/workspace/.venv-fp-priors/bin/python -m pip install "
            "maturin requests==2.33.0 websockets==15.0.1 python-dateutil==2.8.0",
            "PATH=/root/.cargo/bin:$PATH CARGO_TARGET_DIR=/tmp/metagross-poke-engine "
            "/workspace/.venv-fp-priors/bin/python -m pip install --no-cache-dir "
            "/workspace/srcs/vendor/poke-engine "
            "--config-settings='build-args=--no-default-features "
            "--features poke-engine/gen9,poke-engine/terastallization'",
            "rm -rf /tmp/metagross-poke-engine",
        )
    )
    return image


IMAGE = _build_image()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_identifier(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{label} contains unsafe characters")
    return value


def _checkpoint_path(root: Path, profile: str) -> Path:
    config = PROFILES[profile]
    return (
        root
        / str(config["run_name"])
        / "ckpts"
        / "policy_weights"
        / f"policy_epoch_{config['checkpoint']}.pt"
    )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fetch_rating(username: str) -> dict[str, object]:
    url = f"https://pokemonshowdown.com/users/{username.lower()}.json?ts={time.time_ns()}"
    request = urllib.request.Request(url, headers={"User-Agent": "metagross-modal-ladder"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read())
    rating = (payload.get("ratings") or {}).get(FORMAT) or {}
    return {key: rating.get(key) for key in ("elo", "gxe", "rpr", "rprd", "w", "l")}


def _launcher_command(
    profile: str,
    username: str,
    games: int,
    checkpoint_root: Path,
    output_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "srcs.metagross.launch",
        "--username",
        username,
        "--profile",
        profile,
        "--websocket-uri",
        WEBSOCKET_URI,
        "--games",
        str(games),
        "--metamon-python",
        sys.executable,
        "--foul-play-python",
        "/workspace/.venv-fp-priors/bin/python",
        "--checkpoint-root",
        str(checkpoint_root),
        "--output-root",
        str(output_root),
        "--rating-poll-seconds",
        "90",
        "--search-parallelism",
        str(R1_SEARCH_PARALLELISM),
        "--search-threads",
        str(R1_SEARCH_THREADS),
    ]
    if profile == "g3":
        command.append(
            "--confirm-g3-canary" if games == 3 else "--confirm-candidate-continuation"
        )
    return command


def _created_run_dir(output_root: Path, before: set[Path]) -> Path:
    after = {path for path in output_root.iterdir() if path.is_dir()}
    created = after - before
    if len(created) != 1:
        raise RuntimeError(f"expected one launcher run directory, found {len(created)}")
    return created.pop()


def _artifact_inventory(run_dir: Path) -> dict[str, dict[str, object]]:
    return {
        str(path.relative_to(run_dir)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }


def _proxy_environment(proxy_url: str) -> dict[str, str]:
    environment = {
        name: proxy_url
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "wss_proxy",
        )
    }
    environment.update({"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
    return environment


def _sticky_proxy_url(proxy_url: str, session_id: str) -> str:
    parsed = urlsplit(proxy_url)
    if not parsed.hostname or parsed.username is None or parsed.password is None:
        raise ValueError("proxy URL must include username, password, and host")
    username = unquote(parsed.username)
    if "_s_" not in username.lower():
        username = f"{username}_c_us_s_{session_id}_ttl_24h"
    credentials = f"{quote(username, safe='_')}:{quote(unquote(parsed.password), safe='')}"
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{credentials}@{host}"
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _block_proxy_session(experiment_id: str, profile: str, block_index: int) -> str:
    value = f"{experiment_id}:{profile}:{block_index}".encode()
    return f"metagross{hashlib.sha256(value).hexdigest()[:20]}"


def _campaign_proxy_url(
    environment: dict[str, str],
    experiment_id: str,
    profile: str,
    block_index: int,
    *,
    use_profile_proxy: bool = True,
) -> str:
    if use_profile_proxy:
        variable = f"METAGROSS_{profile.upper()}_PROXY_URL"
        proxy_url = environment.get(variable)
        if not proxy_url:
            raise RuntimeError(f"Modal secret did not provide {variable}")
        return proxy_url
    else:
        proxy_url = environment.get("METAGROSS_PROXY_URL")
    if not proxy_url:
        raise RuntimeError(f"Modal secret did not provide a proxy URL for {profile}")
    if (urlsplit(proxy_url).hostname or "").lower() == "residential.byteful.com":
        return _sticky_proxy_url(
            proxy_url,
            _block_proxy_session(experiment_id, profile, block_index),
        )
    return proxy_url


def _rated_games(rating: dict[str, object]) -> int:
    return sum(int(rating.get(key) or 0) for key in ("w", "l"))


def _games_to_request(rating: dict[str, object], target_games: int, block_games: int) -> int:
    return min(block_games, max(0, target_games - _rated_games(rating)))


def _network_error(run_dir: Path) -> str | None:
    client_log = run_dir / "client.log"
    if not client_log.is_file():
        return "client log is missing"
    content = client_log.read_text(encoding="utf-8", errors="replace").lower()
    return next((marker for marker in NETWORK_ERROR_MARKERS if marker in content), None)


def _run_with_periodic_commits(
    command: list[str], environment: dict[str, str], wrapper_log_path: Path
) -> int:
    with wrapper_log_path.open("w", encoding="utf-8") as wrapper_log:
        process = subprocess.Popen(
            command,
            cwd="/workspace",
            env=environment,
            stdout=wrapper_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            while True:
                try:
                    return process.wait(timeout=ARTIFACT_COMMIT_SECONDS)
                except subprocess.TimeoutExpired:
                    VOLUME.commit()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


@APP.function(
    image=IMAGE,
    cpu=1.0,
    memory=1024,
    timeout=300,
    cloud="gcp",
    region="us-west",
    secrets=[SECRET],
)
def probe_showdown_proxy(
    username: str, profile: str = "r1", soak_seconds: int = 30
) -> dict[str, object]:
    _validate_identifier(username, "username")
    if profile not in PROFILES:
        raise ValueError(f"unsupported profile: {profile}")
    if not 0 <= soak_seconds <= 240:
        raise ValueError("soak_seconds must be between 0 and 240")
    password = os.environ.get("METAGROSS_SHOWDOWN_PASSWORD")
    if not password:
        raise RuntimeError("Modal secret is missing Showdown credentials")
    proxy_url = _campaign_proxy_url(
        os.environ, "probe", profile, 0, use_profile_proxy=True
    )
    environment = os.environ.copy()
    environment.update(_proxy_environment(proxy_url))
    environment["PYTHONPATH"] = "/workspace/srcs/vendor/foul-play"
    script = f'''import asyncio
import json
import os

import requests
import websockets
from fp.websocket_client import PSWebsocketClient

def indicators(messages):
    content = "\\n".join(messages).lower()
    return {{
        "popup": "|popup|" in content,
        "proxy": "proxy" in content,
        "locked": "lock" in content,
        "banned": "ban" in content,
        "nametaken": "|nametaken|" in content,
        "updateuser": "|updateuser|" in content,
    }}

async def probe():
    client = await PSWebsocketClient.create(
        {username!r},
        os.environ["METAGROSS_SHOWDOWN_PASSWORD"],
        {WEBSOCKET_URI!r},
    )
    try:
        client_id, challstr = await client.get_id_and_challstr()
        response = requests.post(
            client.login_uri,
            data={{
                "name": client.username,
                "pass": client.password,
                "challstr": "|".join([client_id, challstr]),
            }},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Showdown login HTTP status {{response.status_code}}")
        response_json = json.loads(response.text[1:])
        if "actionsuccess" not in response_json:
            raise RuntimeError("Showdown rejected account login")
        await client.send_message(
            "", ["/trn " + client.username + ",0," + response_json["assertion"]]
        )
        messages = []
        authenticated = False
        try:
            while len(messages) < 20:
                message = await asyncio.wait_for(client.receive_message(), timeout=2)
                messages.append(message)
                if "|updateuser|" in message and "|1|" in message:
                    authenticated = True
                    break
        except asyncio.TimeoutError:
            pass
        except websockets.exceptions.ConnectionClosed as exc:
            raise RuntimeError(
                "Showdown closed before authentication: "
                f"code={{exc.code}} indicators={{json.dumps(indicators(messages), sort_keys=True)}}"
            ) from exc
        if not authenticated:
            raise RuntimeError(
                "Showdown did not confirm authentication: "
                f"indicators={{json.dumps(indicators(messages), sort_keys=True)}}"
            )
        await client.update_team("None")
        deadline = asyncio.get_running_loop().time() + {soak_seconds!r}
        while asyncio.get_running_loop().time() < deadline:
            try:
                message = await asyncio.wait_for(client.receive_message(), timeout=1)
                messages.append(message)
                status = indicators(messages)
                if status["popup"] or status["nametaken"]:
                    raise RuntimeError(
                        "Showdown rejected the authenticated session: "
                        f"indicators={{json.dumps(status, sort_keys=True)}}"
                    )
            except asyncio.TimeoutError:
                pong = await client.websocket.ping()
                await asyncio.wait_for(pong, timeout=10)
            except websockets.exceptions.ConnectionClosed as exc:
                raise RuntimeError(
                    "Showdown closed during proxy soak: "
                    f"code={{exc.code}} indicators={{json.dumps(indicators(messages), sort_keys=True)}}"
                ) from exc
    finally:
        await client.close()

asyncio.run(probe())
'''
    result = subprocess.run(
        ["/workspace/.venv-fp-priors/bin/python", "-c", script],
        cwd="/workspace",
        env=environment,
        text=True,
        capture_output=True,
        timeout=240,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"proxied Showdown probe failed: {detail[-2000:]}")
    print(f"PROXY_PROBE_OK username={username} websocket_uri={WEBSOCKET_URI}", flush=True)
    return {
        "ok": True,
        "username": username,
        "websocket_uri": WEBSOCKET_URI,
        "soak_seconds": soak_seconds,
    }


@APP.function(
    image=IMAGE,
    cpu=(CLOUD_CPUS, CLOUD_CPUS),
    memory=(CLOUD_MEMORY_MIB, CLOUD_MEMORY_MIB),
    timeout=24 * 3600,
    max_containers=1,
    cloud="gcp",
    region="us-west",
    secrets=[SECRET],
    volumes={"/data": VOLUME},
)
def run_campaign_segment(
    experiment_id: str,
    profile: str,
    username: str,
    target_games: int,
    block_games: int,
    start_block: int = 0,
) -> dict[str, object]:
    _validate_identifier(experiment_id, "experiment_id")
    _validate_identifier(username, "username")
    if profile not in PROFILES:
        raise ValueError(f"unsupported profile: {profile}")
    if target_games <= 0 or block_games <= 0:
        raise ValueError("target_games and block_games must be positive")
    if start_block < 0:
        raise ValueError("start_block must be non-negative")
    if not os.environ.get("METAGROSS_SHOWDOWN_PASSWORD"):
        raise RuntimeError("Modal secret did not provide METAGROSS_SHOWDOWN_PASSWORD")

    VOLUME.reload()
    checkpoint_root = Path("/data/public_ladder/models")
    checkpoint = _checkpoint_path(checkpoint_root, profile)
    expected_sha = str(PROFILES[profile]["sha256"])
    actual_sha = _sha256(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"{profile} checkpoint SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )

    campaign_root = Path("/data/public_ladder/experiments") / experiment_id / profile
    output_root = campaign_root / "blocks"
    output_root.mkdir(parents=True, exist_ok=True)
    segment_index = start_block // MAX_BLOCKS_PER_SEGMENT
    segment_path = campaign_root / "segments" / f"segment_{segment_index:03d}.json"
    segment: dict[str, object] = {
        "schema": 1,
        "status": "running",
        "started_at": _utc_now(),
        "experiment_id": experiment_id,
        "profile": profile,
        "deployment_variant": DEPLOYMENT_VARIANT,
        "cloud_cpus": CLOUD_CPUS,
        "search_parallelism": R1_SEARCH_PARALLELISM,
        "username": username,
        "checkpoint_sha256": actual_sha,
        "target_games": target_games,
        "block_games": block_games,
        "start_block": start_block,
        "blocks": [],
    }
    _atomic_json(segment_path, segment)
    VOLUME.commit()

    end_block = start_block + MAX_BLOCKS_PER_SEGMENT
    target_reached = False
    try:
        for block_index in range(start_block, end_block):
            rating_before = _fetch_rating(username)
            games = _games_to_request(rating_before, target_games, block_games)
            if games == 0:
                target_reached = True
                break
            before = {path for path in output_root.iterdir() if path.is_dir()}
            command = _launcher_command(
                profile, username, games, checkpoint_root, output_root
            )
            wrapper_log_path = campaign_root / "wrapper_logs" / f"block_{block_index:03d}.log"
            wrapper_log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["METAGROSS_SHOWDOWN_PASSWORD"] = os.environ[
                "METAGROSS_SHOWDOWN_PASSWORD"
            ]
            environment["METAGROSS_WEBSOCKET_KEEPALIVE"] = "1"
            environment.update(
                _proxy_environment(
                    _campaign_proxy_url(
                        os.environ,
                        experiment_id,
                        profile,
                        block_index,
                        use_profile_proxy=True,
                    )
                )
            )
            if profile == "g3" and block_index == start_block:
                time.sleep(30)
            return_code = _run_with_periodic_commits(
                command, environment, wrapper_log_path
            )
            run_dir = _created_run_dir(output_root, before)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            network_error = _network_error(run_dir)
            try:
                rating = _fetch_rating(username)
            except Exception as exc:
                rating = {"error": f"{type(exc).__name__}: {exc}"}
            block_record = {
                "block_index": block_index,
                "completed_at": _utc_now(),
                "requested_games": games,
                "return_code": return_code,
                "manifest_status": manifest.get("status"),
                "network_error": network_error,
                "run_dir": str(run_dir),
                "rating_before": rating_before,
                "rating": rating,
                "artifacts": _artifact_inventory(run_dir),
            }
            segment["blocks"].append(block_record)
            segment["next_block"] = block_index + 1
            _atomic_json(segment_path, segment)
            VOLUME.commit()
            if return_code != 0 or manifest.get("status") != "completed" or network_error:
                raise RuntimeError(
                    "ladder block failed closed: "
                    f"return_code={return_code}, manifest_status={manifest.get('status')}, "
                    f"network_error={network_error}"
                )
            if _rated_games(rating) >= target_games:
                target_reached = True
                break
    except Exception as exc:
        segment["status"] = "failed"
        segment["finished_at"] = _utc_now()
        segment["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(segment_path, segment)
        VOLUME.commit()
        raise

    segment["status"] = "completed" if target_reached else "continuing"
    segment["finished_at"] = _utc_now()
    if not target_reached:
        continuation = run_campaign_segment.spawn(
            experiment_id,
            profile,
            username,
            target_games,
            block_games,
            end_block,
        )
        segment["continuation_call_id"] = continuation.object_id
        segment["next_block"] = end_block
    _atomic_json(segment_path, segment)
    VOLUME.commit()
    return segment


def _upload_experiment(experiment_id: str, assignments: dict[str, str]) -> None:
    remote_models = "/public_ladder/models"
    manifest = {
        "schema": 1,
        "created_at": _utc_now(),
        "experiment_id": experiment_id,
        "assignments": assignments,
        "deployment_variant": DEPLOYMENT_VARIANT,
        "cloud_cpus": CLOUD_CPUS,
        "search_parallelism": R1_SEARCH_PARALLELISM,
        "profiles": PROFILES,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        with VOLUME.batch_upload(force=True) as batch:
            for profile in assignments:
                config = PROFILES[profile]
                checkpoint = _checkpoint_path(ROOT / "srcs" / "models", profile)
                if _sha256(checkpoint) != config["sha256"]:
                    raise RuntimeError(f"local {profile} checkpoint SHA-256 mismatch")
                destination = (
                    f"{remote_models}/{config['run_name']}/ckpts/policy_weights/"
                    f"policy_epoch_{config['checkpoint']}.pt"
                )
                batch.put_file(str(checkpoint), destination)
            batch.put_file(
                handle.name,
                f"/public_ladder/experiments/{experiment_id}/EXPERIMENT.json",
            )


@APP.local_entrypoint()
def main(
    experiment_id: str,
    r1_username: str,
    target_games: int = 600,
    block_games: int = 25,
    wait: bool = False,
) -> None:
    _validate_identifier(experiment_id, "experiment_id")
    assignments = {"r1": _validate_identifier(r1_username, "r1_username")}
    if target_games <= 0 or block_games <= 0:
        raise ValueError("target_games and block_games must be positive")
    if block_games > 100:
        raise ValueError("block_games must not exceed 100")
    _upload_experiment(experiment_id, assignments)
    call = run_campaign_segment.spawn(
        experiment_id,
        "r1",
        assignments["r1"],
        target_games,
        block_games,
        0,
    )
    report: dict[str, object] = {
        "experiment_id": experiment_id,
        "assignments": assignments,
        "deployment_variant": DEPLOYMENT_VARIANT,
        "cloud_cpus": CLOUD_CPUS,
        "search_parallelism": R1_SEARCH_PARALLELISM,
        "target_games": target_games,
        "block_games": block_games,
        "function_call_id": call.object_id,
    }
    if wait:
        report["result"] = call.get()
    print(json.dumps(report, indent=2, sort_keys=True))
