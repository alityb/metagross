#!/usr/bin/env bash
set -euo pipefail

instance_type=${1:-c7a.8xlarge}
repo=/opt/metagross/repo
venv=/opt/metagross/venv
wheel=srcs/vendor/poke-engine/linux_wheels/poke_engine-0.0.47-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_repo=${METAGROSS_REPO_SOURCE:-"${script_dir}/../.."}

fail() {
  printf 'metagross AWS install: %s\n' "$1" >&2
  exit 1
}

if [[ ! -d ${source_repo} ]]; then
  fail "repository input not found at ${source_repo}; set METAGROSS_REPO_SOURCE to a complete checkout"
fi
source_repo=$(cd -- "${source_repo}" && pwd -P)
for source in \
  srcs/metagross/__init__.py \
  srcs/metagross/aws_http_mcts.py \
  srcs/metagross/mcts_contract.py
do
  if [[ ! -f ${source_repo}/${source} ]]; then
    fail "repository input is incomplete: missing ${source_repo}/${source}"
  fi
done
source_wheel=${METAGROSS_POKE_ENGINE_WHEEL:-}
if [[ -z ${source_wheel} ]]; then
  fail "set METAGROSS_POKE_ENGINE_WHEEL to an explicitly built v5 CPython 3.11 Linux wheel"
fi
if [[ ! -s ${source_wheel} ]]; then
  fail "poke-engine wheel input not found at ${source_wheel}; build or copy the pinned CPython 3.11 Linux wheel first"
fi
if [[ ! -f ${script_dir}/metagross-mcts.service ]]; then
  fail "systemd unit input not found at ${script_dir}/metagross-mcts.service"
fi
if [[ ${EUID} -ne 0 ]]; then
  fail "run as root"
fi

install_repo_file() {
  local source=$1
  local destination=$2
  if [[ ${source} -ef ${destination} ]]; then
    chown ec2-user:ec2-user "${destination}"
    chmod 0644 "${destination}"
  else
    install -o ec2-user -g ec2-user -m 0644 "${source}" "${destination}"
  fi
}

dnf install -y python3.11 python3.11-pip
install -d -o ec2-user -g ec2-user /opt/metagross
install -d -o ec2-user -g ec2-user "${repo}/srcs/metagross"
install -d -o ec2-user -g ec2-user \
  "${repo}/srcs/vendor/poke-engine/linux_wheels"
install_repo_file "${source_repo}/srcs/metagross/__init__.py" \
  "${repo}/srcs/metagross/__init__.py"
install_repo_file "${source_repo}/srcs/metagross/aws_http_mcts.py" \
  "${repo}/srcs/metagross/aws_http_mcts.py"
install_repo_file "${source_repo}/srcs/metagross/mcts_contract.py" \
  "${repo}/srcs/metagross/mcts_contract.py"
install_repo_file "${source_wheel}" \
  "${repo}/${wheel}"

sudo -u ec2-user python3.11 -m venv "${venv}"
sudo -u ec2-user "${venv}/bin/pip" install --force-reinstall "${repo}/${wheel}"
sudo -u ec2-user env PYTHONPATH="${repo}" "${venv}/bin/python" -c \
  'from srcs.metagross.mcts_contract import engine_identity; engine_identity({"provider": "aws_ec2_install"})'

if [[ ! -s /etc/metagross-mcts.env ]]; then
  umask 077
  token=$(openssl rand -hex 32)
  printf 'METAGROSS_REMOTE_MCTS_TOKEN=%s\nMETAGROSS_AWS_INSTANCE_TYPE=%s\n' \
    "${token}" "${instance_type}" > /etc/metagross-mcts.env
fi

install -m 0644 "${script_dir}/metagross-mcts.service" \
  /etc/systemd/system/metagross-mcts.service
systemctl daemon-reload
systemctl enable metagross-mcts.service
systemctl restart metagross-mcts.service
