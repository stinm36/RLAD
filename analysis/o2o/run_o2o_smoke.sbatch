#!/usr/bin/env bash
#SBATCH -p main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -t 00:30:00
#SBATCH -J o2o_smoke
#SBATCH -o /data2/sohyung/runs_o2o/smoke_%j.out
source /home/compu/anaconda3/etc/profile.d/conda.sh
conda activate /data2/sohyung/conda-envs/sh_rlad_o2o
cd /home/sohyung/RLAD
python -u analysis/o2o/rlad_o2o.py --env hopper-medium-v2 --init ad_rlad \
    --offline_epochs 3 --online_epochs 3 --smoke --out /data2/sohyung/runs_o2o
