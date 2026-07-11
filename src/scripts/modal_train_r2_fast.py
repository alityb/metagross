"""Train the first strong-search ExIt checkpoint on Modal H100.

Uses our patched local Metamon package rather than the incomplete PyPI package,
so the exact training path matches the working Mac/AWS setup.
"""
from __future__ import annotations

import io
import os
import tarfile

import modal


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP = modal.App("metagross-exit-r2-train")
app = APP  # Modal CLI discovers the conventional lowercase export.
VOLUME = modal.Volume.from_name("metagross-exit-r2", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl")
    .pip_install(
        "torch",
        "numpy",
        "gymnasium<=0.29.1",
        "gin-config",
        "wandb",
        "einops",
        "tqdm",
        "lz4",
        "termcolor",
        "rich",
        "huggingface_hub",
        "datasets",
        "pandas",
        "scipy",
        "ratarmountcore",
        "poke-env @ git+https://github.com/UT-Austin-RPL/poke-env.git",
        "amago @ git+https://github.com/UT-Austin-RPL/amago@0974781a9096ff43df1b708312256f96fc2ab127",
    )
    # Copy our full patched Metamon source, including tokenizer/config data.
    .add_local_dir(
        os.path.join(ROOT, "external", "metamon", "metamon"),
        "/usr/local/lib/python3.11/site-packages/metamon",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )
)


@APP.function(
    image=IMAGE,
    gpu="H100",
    timeout=3600,
    volumes={"/data": VOLUME},
)
def train(parsed_tarball: bytes, variant_script: bytes, toggles: bytes, gins: bytes) -> list[str]:
    import glob
    import subprocess
    import sys

    os.environ.update(
        METAMON_CACHE_DIR="/data/metamon_cache",
        HF_HOME="/data/hf_home",
        WANDB_MODE="disabled",
        TORCHDYNAMO_DISABLE="1",
    )
    os.makedirs("/data/metamon_cache", exist_ok=True)
    os.makedirs("/data/repo/src/train", exist_ok=True)
    os.makedirs("/data/repo/src/scripts", exist_ok=True)

    with tarfile.open(fileobj=io.BytesIO(parsed_tarball), mode="r:gz") as archive:
        archive.extractall("/data")

    with open("/data/repo/src/train/finetune_toggles.py", "wb") as f:
        f.write(toggles)
    with open("/data/repo/src/scripts/run_finetune_variant.py", "wb") as f:
        f.write(variant_script)
    with tarfile.open(fileobj=io.BytesIO(gins), mode="r:gz") as archive:
        archive.extractall("/data/repo/src/train")

    # The trainer runs in a fresh Python subprocess. Patch its installed Amago
    # source so superkazam.gin can resolve @transformer.VanillaAttention.
    transformer = "/usr/local/lib/python3.11/site-packages/amago/nets/transformer.py"
    with open(transformer) as f:
        transformer_source = f.read()
    if not transformer_source.startswith("import gin"):
        transformer_source = "import gin\n" + transformer_source
    if "@gin.configurable\nclass VanillaAttention" not in transformer_source:
        transformer_source = transformer_source.replace(
            "class VanillaAttention", "@gin.configurable\nclass VanillaAttention", 1
        )
    with open(transformer, "w") as f:
        f.write(transformer_source)

    for root, dirs, _ in os.walk("/usr/local/lib/python3.11/site-packages"):
        for directory in dirs:
            if directory == "__pycache__":
                import shutil
                shutil.rmtree(os.path.join(root, directory), ignore_errors=True)

    n = len(glob.glob("/data/selfplay_round2_parsed/gen9randombattle/*.lz4"))
    print(f"Training on {n} trajectories", flush=True)

    with open("/data/randbats_r2.yaml", "w") as f:
        f.write(
            "replay_weight: 0.0\n"
            "custom_replays:\n"
            "  - dir: /data/selfplay_round2_parsed\n"
            "    weight: 1.0\n"
            "formats:\n"
            "  - gen9randombattle\n"
        )

    env = os.environ | {"PYTHONPATH": "/data/repo/src"}
    command = [
        sys.executable,
        "/data/repo/src/scripts/run_finetune_variant.py",
        "--variant",
        "ALL",
        "--dataset-config",
        "/data/randbats_r2.yaml",
        "--save-dir",
        "/data/ckpts",
        "--epochs",
        "2",
        "--steps-per-epoch",
        "1000",
        "--batch-size",
        "24",
        "--dloader-workers",
        "8",
    ]
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, env=env, capture_output=True, text=True)

    ckpts = sorted(
        glob.glob("/data/ckpts/randbats_ALL/*/ckpts/policy_weights/policy_epoch_*.pt")
    )
    print("Checkpoints:", ckpts, flush=True)
    if result.returncode:
        print("TRAIN STDOUT:\n" + result.stdout[-12000:], flush=True)
        print("TRAIN STDERR:\n" + result.stderr[-30000:], flush=True)
        raise RuntimeError(f"finetune failed with exit code {result.returncode}")
    VOLUME.commit()
    return ckpts


@APP.local_entrypoint()
def main() -> None:
    with open("/tmp/randbats_r2_train.tgz", "rb") as f:
        parsed = f.read()
    with open(os.path.join(ROOT, "src", "scripts", "run_finetune_variant.py"), "rb") as f:
        variant_script = f.read()
    with open(os.path.join(ROOT, "src", "train", "finetune_toggles.py"), "rb") as f:
        toggles = f.read()

    gins_buf = io.BytesIO()
    with tarfile.open(fileobj=gins_buf, mode="w:gz") as archive:
        for name in ("metagross_ALL.gin", "metagross_B_klanchor.gin", "metagross_D_hlgauss.gin"):
            path = os.path.join(ROOT, "src", "train", "gins", name)
            archive.add(path, arcname=f"gins/{name}")

    print(f"Uploading {len(parsed) / 1e6:.1f}MB of trajectories", flush=True)
    call = train.spawn(parsed, variant_script, toggles, gins_buf.getvalue())
    print(f"Detached training call: {call.object_id}", flush=True)
