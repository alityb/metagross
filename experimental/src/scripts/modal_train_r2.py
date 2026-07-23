"""Modal training: upload data + train ExIt round-2 in one shot.

Usage:
  python3.11 src/scripts/modal_train_r2.py
"""
import modal
import os

vol = modal.Volume.from_name("metagross-exit-r2", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl")
    .pip_install(
        "torch", "numpy", "gymnasium<=0.29.1", "gin-config", "wandb",
        "einops", "tqdm", "lz4", "termcolor", "rich", "huggingface_hub",
        "datasets", "pandas", "scipy", "ratarmountcore",
        "poke-env @ git+https://github.com/UT-Austin-RPL/poke-env.git",
        "amago @ git+https://github.com/UT-Austin-RPL/amago@0974781a9096ff43df1b708312256f96fc2ab127",
    )
    .pip_install(
        "metamon @ git+https://github.com/UT-Austin-RPL/metamon.git@0a00a759c9a4382a2877088d828302ec294a05a5",
    )
    # Clear __pycache__ to avoid stale imports
    .run_commands("find /usr/local/lib/python3.11/site-packages/metamon -name '__pycache__' -exec rm -rf {} + 2>/dev/null; true")
)

app = modal.App("metagross-exit-r2")


@app.function(
    image=IMAGE,
    gpu="H100",
    timeout=3600,
    volumes={"/data": vol},
)
def train(parsed_tarball: bytes, repo_tarball: bytes, finetune_toggles_py: bytes, gins_tarball: bytes = b""):
    import subprocess, os, sys, tarfile, io, glob

    os.environ["METAMON_CACHE_DIR"] = "/data/metamon_cache"
    os.environ["HF_HOME"] = "/data/hf_home"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.makedirs("/data/metamon_cache", exist_ok=True)
    os.makedirs("/data/selfplay_round2_parsed", exist_ok=True)
    os.makedirs("/data/repo/src/train", exist_ok=True)

    # 1. Extract parsed replays
    print("Extracting parsed replays...", flush=True)
    with tarfile.open(fileobj=io.BytesIO(parsed_tarball), mode="r:gz") as tf:
        tf.extractall("/data/")
    n_traj = len(glob.glob("/data/selfplay_round2_parsed/*.lz4"))
    print(f"  {n_traj} trajectories", flush=True)

    # 2. Extract repo
    print("Extracting repo...", flush=True)
    with tarfile.open(fileobj=io.BytesIO(repo_tarball), mode="r:gz") as tf:
        tf.extractall("/data/repo/")
    # Write finetune_toggles.py
    with open("/data/repo/src/train/finetune_toggles.py", "wb") as f:
        f.write(finetune_toggles_py)
    # Extract gins if provided
    if gins_tarball:
        with tarfile.open(fileobj=io.BytesIO(gins_tarball), mode="r:gz") as tf:
            tf.extractall("/data/repo/src/train/")

    sys.path.insert(0, "/data/repo/src")

    # 3. Fix metamon imports (same patches as AWS)
    import amago.nets.transformer
    from amago.nets.transformer import VanillaAttention
    import gin
    try:
        gin.external_configurable(VanillaAttention, module="transformer")
    except Exception:
        pass

    # Fix gin config
    import metamon
    gin_dir = os.path.join(os.path.dirname(metamon.__file__), "rl", "configs", "models")
    gin_file = os.path.join(gin_dir, "superkazam.gin")
    if os.path.exists(gin_file):
        with open(gin_file) as f:
            content = f.read()
        if "@transformer.VanillaAttention" in content:
            content = content.replace(
                "@transformer.VanillaAttention",
                "@amago.nets.transformer.VanillaAttention",
            )
            with open(gin_file, "w") as f:
                f.write(content)
            print("Patched superkazam.gin", flush=True)

    # Fix LocalFinetunedModel
    pretrained_path = os.path.join(os.path.dirname(metamon.__file__), "rl", "pretrained.py")
    with open(pretrained_path) as f:
        content = f.read()
    if "gin_overrides=base_model.gin_overrides" not in content:
        content = content.replace(
            "            dataset_config=dataset_config,",
            "            dataset_config=dataset_config,\n            gin_overrides=base_model.gin_overrides,",
        )
        with open(pretrained_path, "w") as f:
            f.write(content)
        print("Patched pretrained.py (gin_overrides)", flush=True)

    # 4. Install finetune toggles (rating conditioning)
    from train.finetune_toggles import install_rating_conditioning
    install_rating_conditioning()
    print("Rating conditioning installed", flush=True)

    # 5. Create dataset config
    dataset_yaml = "/data/randbats_r2.yaml"
    with open(dataset_yaml, "w") as f:
        f.write("""replay_weight: 0.0
custom_replays:
  - dir: /data/selfplay_round2_parsed
    weight: 1.0
formats:
  - gen9randombattle
""")

    # 6. Train
    cmd = [
        sys.executable, "-m", "metamon.rl.finetune",
        "--run_name", "randbats_exit_r2",
        "--save_dir", "/data/ckpts",
        "--base_model", "Kakuna",
        "--dataset_config", dataset_yaml,
        "--epochs", "2",
        "--steps_per_epoch", "1000",
        "--batch_size_per_gpu", "24",
        "--ckpt_interval", "1",
        "--dloader_workers", "8",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=False, text=True)

    # 7. List + return checkpoint paths
    ckpts = sorted(glob.glob("/data/ckpts/randbats_exit_r2/*/ckpts/policy_weights/policy_epoch_*.pt"))
    print(f"\nCheckpoints: {ckpts}", flush=True)
    vol.commit()
    return ckpts


@app.local_entrypoint()
def main():
    # Read the tarballs locally and pass as bytes
    with open("/tmp/randbats_r2_parsed.tgz", "rb") as f:
        parsed_tarball = f.read()
    with open("/tmp/metamon_for_modal.tgz", "rb") as f:
        repo_tarball = f.read()
    with open("src/train/finetune_toggles.py", "rb") as f:
        finetune_toggles = f.read()

    # Create gins tarball
    import tarfile, io
    gins_buf = io.BytesIO()
    with tarfile.open(fileobj=gins_buf, mode="w:gz") as tf:
        for gin_file in ["metagross_D_hlgauss.gin", "metagross_B_klanchor.gin", "metagross_ALL.gin"]:
            path = f"src/train/gins/{gin_file}"
            if os.path.exists(path):
                tf.add(path, arcname=f"gins/{gin_file}")
    gins_tarball = gins_buf.getvalue()

    print(f"Parsed tarball: {len(parsed_tarball)/1e6:.1f}MB")
    print(f"Repo tarball: {len(repo_tarball)/1e6:.1f}MB")
    print(f"Finetune toggles: {len(finetune_toggles)/1e3:.1f}KB")
    print(f"Gins tarball: {len(gins_tarball)/1e3:.1f}KB")
    print("Launching training on H100...")

    result = train.remote(parsed_tarball, repo_tarball, finetune_toggles, gins_tarball)
    print(f"\nTraining complete! Checkpoints: {result}")
