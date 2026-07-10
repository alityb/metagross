#!/bin/bash
# User-data for g4dn prior-server instance.
set -ex
apt-get update -qq
apt-get install -y -qq software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y -qq python3.11 python3.11-venv python3.11-dev git curl build-essential

cd /opt
git clone https://github.com/alityb/metagross.git
cd metagross

python3.11 -m venv .venv-metamon
.venv-metamon/bin/pip install --upgrade pip -q

# Install metamon from git (pulls torch, gymnasium, amago, huggingface_hub, etc.)
.venv-metamon/bin/pip install -q "git+https://github.com/UT-Austin-RPL/metamon.git@0a00a759c9a4382a2877088d828302ec294a05a5#egg=metamon"

mkdir -p external/metamon_cache
echo "PRIOR_SETUP_DONE" > /tmp/setup_done
