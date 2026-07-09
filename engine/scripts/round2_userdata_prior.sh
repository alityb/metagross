#!/bin/bash
# User-data for g4dn prior-server instance. Runs as root on first boot.
set -ex
apt-get update -qq && apt-get install -y -qq python3.11 python3.11-venv git curl build-essential
# NVIDIA drivers are pre-installed on g4dn by default; install CUDA torch via pip
cd /opt
git clone https://github.com/alityb/metagross.git
cd metagross
python3.11 -m venv .venv-metamon
.venv-metamon/bin/pip install --upgrade pip -q
.venv-metamon/bin/pip install -q -r requirements.txt 2>/dev/null || true
# Metamon cache will auto-download from HuggingFace on first prior_server import
mkdir -p external/metamon_cache
echo "PRIOR_SETUP_DONE" > /tmp/setup_done
