#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

KEY=~/.ssh/metagross-r2.pem
SG=sg-079cc5d3110984b39
AMI=ami-0d28727121d5d4a3c
WORKER_TYPE=c6i.16xlarge
PRIOR_TYPE=g4dn.xlarge
N_WORKERS=2

echo "=== launching prior server (g4dn.xlarge) ==="
PRIOR_ID=$(aws ec2 run-instances \
  --image-id "$AMI" --instance-type "$PRIOR_TYPE" \
  --key-name metagross-r2 --security-group-ids "$SG" \
  --user-data file://engine/scripts/round2_userdata_prior.sh \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --query "Instances[0].InstanceId" --output text)
echo "prior server: $PRIOR_ID"

echo "=== launching $N_WORKERS workers ($WORKER_TYPE) ==="
WORKER_IDS=""
for i in $(seq 1 $N_WORKERS); do
  WID=$(aws ec2 run-instances \
    --image-id "$AMI" --instance-type "$WORKER_TYPE" \
    --key-name metagross-r2 --security-group-ids "$SG" \
    --user-data file://engine/scripts/round2_userdata_worker.sh \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
    --query "Instances[0].InstanceId" --output text)
  echo "worker $i: $WID"
  WORKER_IDS="$WORKER_IDS $WID"
done

echo "=== waiting for instances to boot + run user-data ==="
ALL_IDS="$PRIOR_ID $WORKER_IDS"
aws ec2 wait instance-running --instance-ids $ALL_IDS 2>/dev/null || true
sleep 30

echo "=== getting IPs ==="
PRIOR_IP=$(aws ec2 describe-instances --instance-ids "$PRIOR_ID" \
  --query "Reservations[0].Instances[0].PrivateIpAddress" --output text)
PRIOR_PUB=$(aws ec2 describe-instances --instance-ids "$PRIOR_ID" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "prior server: private=$PRIOR_IP public=$PRIOR_PUB"

WORKER_PUBS=""
for WID in $WORKER_IDS; do
  WPUB=$(aws ec2 describe-instances --instance-ids "$WID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
  echo "worker $WID: public=$WPUB"
  WORKER_PUBS="$WORKER_PUBS $WPUB"
done

echo "=== waiting for user-data scripts to finish (checking /tmp/setup_done) ==="
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"
for IP in "$PRIOR_PUB" $WORKER_PUBS; do
  echo -n "  $IP: "
  for attempt in $(seq 1 30); do
    if $SSH ubuntu@$IP "test -f /tmp/setup_done" 2>/dev/null; then
      echo "READY"; break
    fi
    echo -n "."; sleep 20
  done
done

echo "=== transferring files ==="
# 1. Linux wheel to all workers
WHEEL=engine/pe_v2/linux_wheels/poke_engine-0.0.47-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
for WPUB in $WORKER_PUBS; do
  echo "  wheel -> worker $WPUB"
  scp -i $KEY -o StrictHostKeyChecking=no "$WHEEL" ubuntu@$WPUB:/opt/metagross/poke_engine.whl
  $SSH ubuntu@$WPUB ".venv-foul-play/bin/pip install -q --force-reinstall /opt/metagross/poke_engine.whl"
done

# 2. Base checkpoint to prior server (545MB)
echo "  checkpoint -> prior server $PRIOR_PUB"
scp -i $KEY -o StrictHostKeyChecking=no -r \
  src/nets/checkpoints/randbats_full/randbats_D_hlgauss \
  ubuntu@$PRIOR_PUB:/opt/metagross/src/nets/checkpoints/randbats_full/

echo "=== starting prior server on g4dn ==="
$SSH ubuntu@$PRIOR_PUB bash -c "'
cd /opt/metagross
export METAMON_CACHE_DIR=/opt/metagross/external/metamon_cache
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=src
nohup .venv-metamon/bin/python -u src/scripts/prior_server.py \
  --local-run-dir /opt/metagross/src/nets/checkpoints/randbats_full \
  --local-run-name randbats_D_hlgauss --checkpoint 4 \
  --host 0.0.0.0 --port 8977 --username gen \
  > /tmp/prior_server.log 2>&1 &
'"
echo "waiting for prior server to load..."
for attempt in $(seq 1 40); do
  if $SSH ubuntu@$PRIOR_PUB "grep -q 'PRIOR_SERVER ready' /tmp/prior_server.log" 2>/dev/null; then
    echo "PRIOR SERVER READY"; break
  fi
  echo -n "."; sleep 10
done

echo "=== starting generation on workers ==="
GAMES_PER_WORKER=$(( 100000 / N_WORKERS ))
for WPUB in $WORKER_PUBS; do
  echo "  worker $WPUB: $GAMES_PER_WORKER games"
  $SSH ubuntu@$WPUB bash -c "'
cd /opt/metagross
export PYTHONPATH=src
export PRIOR_SERVER_URL=http://$PRIOR_IP:8977
export SEARCH_TIME_MS=500
export SEARCH_PARALLELISM=8
export CONCURRENCY=8
export N_GAMES=$GAMES_PER_WORKER
export OUTPUT_DIR=/opt/metagross/data/selfplay_round2
nohup bash src/scripts/generate_round2.sh > /tmp/generation.log 2>&1 &
'"
  echo "  generation started on $WPUB"
done

echo "=== fleet launched ==="
echo "prior server: $PRIOR_PUB (private: $PRIOR_IP:8977)"
echo "workers: $WORKER_PUBS"
echo "monitor: ssh -i $KEY ubuntu@<ip> 'tail -f /tmp/generation.log'"
