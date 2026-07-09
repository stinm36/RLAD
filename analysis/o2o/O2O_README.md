# RLAD Offline-to-Online (O2O) — run guide

Warm-start study: does AD-regularized offline pretraining give a better O2O
initializer than CQL / scratch? Online phase is plain SAC fine-tune for all arms
(penalty OFF during online), isolating init quality (hypothesis H-A).

## Files
- `rlad_o2o.py` — 2-phase harness (offline pretrain → online SAC). Arms: `ad_rlad`, `cql`, `scratch`.
- `run_o2o.sbatch` — parametrized SLURM job (env vars: `ENVV ARM OFF ON SEED [SMOKE=1]`).
- `collect_o2o.py` — aggregate progress.csv into warm-start metrics + comparison curves.
- `setup_env.sh` — one-shot env build (Ampere+ / A100 ready).
- weights: `weights_ad/{svdd,dsebm}/{env}/` (also archived at `/data2/sohyung/rlad_o2o_weights.tar.gz`, ~29MB).

## Setup on a fresh server (e.g. A100 VESSL)
```bash
git clone <repo> && cd RLAD && git checkout sh_ad-embedding-viz
bash analysis/o2o/setup_env.sh ./envs/sh_rlad_o2o https://download.pytorch.org/whl/cu121
# put AD weights in place (tar) OR retrain:
tar -xzf rlad_o2o_weights.tar.gz            # if transferred
```
Offline data auto-downloads from the HF D4RL mirror on first use (no d4rl pkg needed).
A100 is Ampere → torch cu117 or cu12x both work.

## Run
```bash
# smoke (few min)
ENVV=hopper-medium-v2 ARM=ad_rlad OFF=3 ON=3 SEED=0 SMOKE=1 sbatch analysis/o2o/run_o2o.sbatch
# real run (~1 h): offline 250 + online 250
ENVV=hopper-medium-v2 ARM=ad_rlad OFF=250 ON=250 SEED=0 sbatch analysis/o2o/run_o2o.sbatch
```
Arms: `ad_rlad` (AD penalty offline, off online), `cql`, `scratch` (online only).

## Collect
```bash
python analysis/o2o/collect_o2o.py --root /data2/sohyung/runs_o2o
# -> o2o_curves.png + o2o_metrics.json (offline_final, online_start, dip, online_final, online_auc)
```

## Status (validated on greenbeard)
- Env, GPU (Blackwell cu130), offline→buffer load, all 3 arms: smoke-passed end-to-end.
- Timing: full run (num_trains=1000) ≈ ~1 h/run. Matrix 3 arm × 3 env(medium) × 5 seed ≈ ~45 GPU-h.
- Known minor: two phases append to one progress.csv with a repeated header row — `collect_o2o.py` handles it.
