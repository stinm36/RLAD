#!/usr/bin/env bash
#SBATCH -p main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -t 04:00:00
#SBATCH -J o2o
#SBATCH -o /data2/sohyung/runs_o2o/%x_%j.out
source /home/compu/anaconda3/etc/profile.d/conda.sh
conda activate /data2/sohyung/conda-envs/sh_rlad_o2o
cd /home/sohyung/RLAD
python -u analysis/o2o/rlad_o2o.py --env "${ENVV}" --init "${ARM}" \
  --offline_epochs "${OFF}" --online_epochs "${ON}" --seed "${SEED}" \
  ${SMOKE:+--smoke} --out /data2/sohyung/runs_o2o
