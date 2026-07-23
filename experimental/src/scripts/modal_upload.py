"""Upload data to Modal volume, then trigger training."""
import modal
import os

vol = modal.Volume.from_name("metagross-data", create_if_missing=True)

app = modal.App("metagross-upload")

@app.function(volumes={"/data": vol})
def upload():
    import os, subprocess
    # Create dirs
    os.makedirs("/data/selfplay_round2_parsed", exist_ok=True)
    os.makedirs("/data/repo/src", exist_ok=True)
    os.makedirs("/data/metamon_cache", exist_ok=True)
    vol.commit()

@app.function(volumes={"/data": vol})
def upload_replays():
    """Upload parsed replays from local tarball."""
    import os, tarfile
    os.makedirs("/data/selfplay_round2_parsed", exist_ok=True)
    # The tarball is staged via modal put
    with tarfile.open("/tmp/randbats_r2_parsed.tgz") as tf:
        tf.extractall("/data/")
    print(f"Uploaded {len(os.listdir('/data/selfplay_round2_parsed'))} trajectory files")
    vol.commit()

@app.function(volumes={"/data": vol})
def upload_repo():
    """Upload repo (src/train + finetune toggles)."""
    import os, tarfile
    os.makedirs("/data/repo", exist_ok=True)
    with tarfile.open("/tmp/repo_for_modal.tgz") as tf:
        tf.extractall("/data/repo/")
    print("Repo uploaded")
    vol.commit()

@app.local_entrypoint()
def main():
    upload.remote()
    # Stage files on the volume by putting them into the function
    # Actually, modal doesn't support local file upload to volumes directly
    # We need to use modal put or embed in the function
    print("Use: modal volume put metagross-data <local_path> <remote_path>")
