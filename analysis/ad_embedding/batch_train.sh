#!/usr/bin/env bash
# Sequentially train SVDD + DSEBM on the remaining 8 D4RL datasets.
# Ordered smallest-first (replay < medium < medium-expert) for quicker early results.
# Run detached so it survives disconnect:
#   nohup bash analysis/ad_embedding/batch_train.sh > analysis/ad_embedding/logs/batch.log 2>&1 &
set -u
source /home/compu/anaconda3/etc/profile.d/conda.sh
conda activate sh_rlad
cd /home/sohyung/RLAD

ENVS=(
  hopper-medium-replay-v2
  walker2d-medium-replay-v2
  halfcheetah-medium-replay-v2
  walker2d-medium-v2
  halfcheetah-medium-v2
  hopper-medium-expert-v2
  walker2d-medium-expert-v2
  halfcheetah-medium-expert-v2
)
LOGDIR=analysis/ad_embedding/logs
mkdir -p "$LOGDIR"

for i in "${!ENVS[@]}"; do
  env="${ENVS[$i]}"
  echo "===== [$((i+1))/${#ENVS[@]}] $env  $(date '+%F %T') ====="
  python -u analysis/ad_embedding/train_ad.py --env "$env" --epochs 200 --batch 8192 --threads 8 \
      > "$LOGDIR/train_${env}.log" 2>&1
  rc=$?
  echo "----- $env done (exit $rc)  $(date '+%F %T') -----"
done
echo "===== BATCH COMPLETE  $(date '+%F %T') ====="
