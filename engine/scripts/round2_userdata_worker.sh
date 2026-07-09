#!/bin/bash
# User-data for c6i game-worker instances. Runs as root on first boot.
set -ex
apt-get update -qq && apt-get install -y -qq python3.11 python3.11-venv nodejs npm git curl build-essential
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y -qq nodejs
cd /opt
git clone https://github.com/alityb/metagross.git
cd metagross
python3.11 -m venv .venv-foul-play
.venv-foul-play/bin/pip install --upgrade pip -q
# Install foul-play
cd external
git clone https://github.com/pmariglia/foul-play.git
cd foul-play
git checkout e1e2ca650598621e85c3b6ab751c66e625489934
cd /opt/metagross
.venv-foul-play/bin/pip install -q -r external/foul-play/requirements.txt
# Install Showdown
cd external
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install --prefix . 2>/dev/null
# Start Showdown in background
cd /opt/metagross
node external/pokemon-showdown/pokemon-showdown start --no-security > /tmp/showdown.log 2>&1 &
echo "WORKER_SETUP_DONE" > /tmp/setup_done
