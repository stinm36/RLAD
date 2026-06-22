# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RLOCC** (Reinforcement Learning with One-Class Classification) — an offline RL research project that mitigates overestimation bias by integrating anomaly detection (AD) with Q-target computation. Built on top of RLKit (RAIL Berkeley's RL framework). Requires Python 3.8.11, PyTorch 1.13.1 with CUDA 11.7, MuJoCo, and D4RL.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Running Experiments

**RLOCC-SAC** (primary method):
```bash
python examples/RLAD.py \
  --env hopper-medium-v2 \
  --gpu 0 \
  --ad_module svdd \        # svdd | dagmm | dsebm | fanogan | mahal | gmm | maf | random
  --ad_train True \         # train AD from scratch; False loads weights from --ad_save_path
  --weight_function hill \  # hill/tanh_penalty (bounded) | identity (softplus applied)
  --penalty_coef 0.01 \     # ||Q^π − Q_pen^π||_∞ ≤ penalty_coef/(1−γ)
  --nepochs 3000 \
  --trial_name exp1         # optional label; appears in run dir name
```

**Other baselines and variants:**
```bash
python examples/RLAD_BEAR.py --env hopper-medium-v2   # RLOCC-BEAR
python examples/RLAD_CQL.py  --env hopper-medium-v2   # RLOCC-CQL
python examples/RLAD_IQL.py  --env hopper-medium-v2   # RLOCC-IQL
python examples/RL_CQL.py    --env hopper-medium-v2   # Baseline CQL (no AD)
python examples/RL_IQL.py    --env hopper-medium-v2   # Baseline IQL (no AD)
python examples/Train_AD.py  --env hopper-medium-v2 --ad_module svdd  # Standalone AD training
```

Batch experiment scripts are in `train_scripts/{svdd,dagmm,dsebm,fanogan,cql,iql,...}/`.

### Saved artefacts

```
weights_ad/
  svdd/{env}/
    pretrained_ae.pth   # AE encoder init weights + hypersphere center
    model[_tag].pth     # trained SVDD network
    metadata.json       # hyperparameters + timestamp
  {module}/{env}/       # dagmm, dsebm, fanogan, mahal, gmm, maf
    *.pth

runs/
  {env}/{ad_module}/{trial_name}_{datetime}/
    config.json         # full CLI args + timestamp (written at launch)
    variant.json        # rlkit variant (algorithm/hyperparams)
    progress.csv        # per-epoch metrics (written by rlkit logger)
    params.pkl          # final policy + Q-function weights
```

`--all_saves` (default `runs`) sets the root log directory. `--ad_save_path` (default `./weights_ad`) sets the root for AD model weights.

## Architecture

### 1. Training Loop — `rlkit/core/batch_rl_algorithm.py`
`BatchRLAlgorithm` orchestrates everything: loads the D4RL dataset into a replay buffer (`batch_rl=True`), iterates epochs, calls `trainer.train_from_torch(batch)`, and triggers periodic evaluation via `MdpPathCollector`. The `TorchBatchRLAlgorithm` wrapper in `rlkit/torch/torch_rl_algorithm.py` handles device placement and train/eval mode switching.

### 2. AD-Augmented Trainers — `rlkit/torch/sac/`

| File | Base algorithm | AD integration |
|------|---------------|----------------|
| `rlad.py` | SAC | Mode 3 penalty on Q-target |
| `bear_ad.py` | BEAR + MMD constraint | Mode 3 penalty on Q-target |
| `cql_ad.py` | CQL conservatism | Mode 3 penalty on Q-target |
| `iql_ad.py` | IQL value function | Mode 3 penalty on Q-target |
| `bear.py`, `cql.py`, `iql.py` | Baselines (no AD) | — |

All trainers inherit `TorchTrainer` and must implement `train_from_torch(batch)` and expose networks via the `networks` property. `self.discrete = False` and `eval_q_custom()` are required by `BatchRLAlgorithm` — keep them.

**Shared AD utilities in `rlad.py`** (imported by all `*_ad.py` trainers):
- `calc_anomaly_score(ad, state_action, c, ad_type)` — unified interface for all AD modules
- `setup_ad(ad, ad_module)` — loads weights and puts model in eval mode
- `make_weight_function(name)` — returns a callable weight transform

**Q-target formulation — Mode 3 (Penalised-reward MDP):**
```
q_target = r + γ·min(Q1',Q2') − α·log π(a'|s') − penalty_coef · g(score(s', a'))
```
The penalty propagates OOD pessimism backward through the Bellman chain.
Fixed point: `Q_pen^π = Q^π − penalty_coef/(1−γ) · E_π[discounted OOD frequency]`.
Bound: `||Q^π − Q_pen^π||_∞ ≤ penalty_coef / (1−γ)`.

**`--weight_function`** — maps raw AD score to penalty magnitude `g(·)`:
- `hill` **(default)**: `score/(score+T)`, bounded `[0,1)`. Penalty always ∈ `(0, penalty_coef)`.
- `tanh_penalty`: `tanh(score/T)`, bounded `[0,1)`. Similar behaviour to `hill`.
- `identity`: unbounded raw score; `softplus` applied automatically in the trainer to compress tails.

`T=5.0` is hardcoded in `make_weight_function`; tune to typical in-distribution AD scores.

**`--penalty_coef`** (default `0.01`) — penalty scale. With `hill`/`tanh_penalty`, penalty ∈ `(0, penalty_coef)` always. Bound: `||Q^π − Q_pen^π||_∞ ≤ penalty_coef/(1−γ)`.

**`--weight_actor` flag** — when `True`, weights the actor loss by `exp(−score)` per sample (SAC only). Off by default.

### 3. Anomaly Detection Modules

| Module | Directory | Key class/function | Weight path |
|--------|-----------|-------------------|-------------|
| Deep SVDD | `svdd/` | `TrainerDeepSVDD` in `svdd.py` | `weights_ad/svdd/{env}/` |
| DAGMM | `dagmm/` | `Solver` in `solver.py` | `weights_ad/dagmm/{env}/` |
| DSEBM | `DSEBM/` | `DSEBM` model, `Trainer` | `weights_ad/dsebm/{env}/` |
| f-AnoGAN | `fAnogan/` | `Generator`, `Discriminator`, `Encoder` | `weights_ad/fanogan/{env}/` |
| Mahalanobis | `mahal/` | `MahalanobisAD` | `weights_ad/mahal/{env}/` |
| GMM (GPU) | `gmm/` | `GPUGMM` | `weights_ad/gmm/{env}/` |
| MAF | `maf/` | `MAFAD` | `weights_ad/maf/{env}/` |
| Random | — | `None` (random scores) | — |

`D4RLDataset` in `svdd/dataset.py` wraps D4RL offline datasets as PyTorch `Dataset` objects; it is shared by all AD modules. Each AD module follows the same training pattern: instantiate → optionally train → passed into the trainer as `ad`.

### Key Files
- `examples/RLAD.py` — wires all components together; best place to understand the full pipeline
- `rlkit/torch/sac/rlad.py` — core SAC trainer and shared AD utilities (`calc_anomaly_score`, `setup_ad`)
- `rlkit/torch/networks.py` — `FlattenMlp`, Q-function, and policy network definitions
- `rlkit/data_management/gpu_replay_buffer.py` — GPU-resident replay buffer; `random_batch()` returns CUDA tensors directly, eliminating per-batch CPU→GPU conversion
- `rlkit/torch/pytorch_util.py` — CPU/GPU transfer utilities; `ptu.set_gpu_mode()` must be called before training

### Performance design

Four optimisations applied across all trainers and example scripts:

| Technique | Where | Benefit |
|-----------|-------|---------|
| **GPU Replay Buffer** | `gpu_replay_buffer.py` + example scripts | Eliminates 5000 `np→GPU` copies/epoch; all transition tensors live in CUDA memory permanently |
| **Mixed precision (AMP)** | `rlad.py` (SACTrainer) | FP16 forward/backward on tensor cores; AD scores kept FP32 via inner `autocast(enabled=False)` |
| **Merged QF backward** | all `*_ad.py` trainers | `(qf1_loss + qf2_loss).backward()` — single graph traversal instead of two; safe because qf1 ∩ qf2 params = ∅ and targets are `.detach()`ed |
| **Persistent DataLoader workers** | all example scripts | `persistent_workers=True, prefetch_factor=4` — avoids re-spawning 4 workers per AD epoch |

`TorchTrainer.train()` detects GPU-tensor batches (`isinstance(v, torch.Tensor)`) and skips `np_to_pytorch_batch()` automatically — no change needed to `BatchRLAlgorithm`.

## Analysis

Jupyter notebooks and scripts in `analysis/` cover four experiments:
- `exp1_critic_vs_both` — ablation: Q-weighting only vs. Q+actor weighting
- `exp2_epoch_sensitivity` — sensitivity to SVDD training epochs
- `exp3_score_distribution` — anomaly score distributions across datasets
- `exp4_td_correlation` — correlation between TD error and anomaly scores

## Benchmarks

D4RL MuJoCo environments (e.g., `hopper-medium-v2`, `walker2d-medium-expert-v2`) and Adroit robotic manipulation tasks. Environment names follow the D4RL convention: `{task}-{dataset}-v{version}`.
