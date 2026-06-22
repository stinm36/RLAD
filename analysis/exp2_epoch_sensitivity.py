"""Exp 2: Deep SVDD pretraining epoch sensitivity.

Reads progress.csv from:
  saves_exp2_svdd_epochs_identity/  (runs tagged e{N}_s{seed})

Outputs:
  results/exp2_epoch_sensitivity.csv   — mean ± std per epoch count
  results/exp2_epoch_sensitivity.png   — line plot (x: epochs, y: norm return)
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import d4rl  # noqa: F401

ENV = 'hopper-medium-v2'
SVDD_EPOCHS = [100, 300, 500, 1000]
N_SEEDS = 3
SAVE_ROOT = 'saves_exp2_svdd_epochs_identity'
EVAL_COL = 'evaluation/Average Returns'


def get_normalized_score(env_name, raw_return):
    env = gym.make(env_name)
    score = env.get_normalized_score(raw_return) * 100
    env.close()
    return score


def read_final_return(root, env, trial_name):
    pattern = os.path.join(root, '**', trial_name, '**', env, '**', 'progress.csv')
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    df = pd.read_csv(files[0])
    if EVAL_COL not in df.columns:
        return None
    return df[EVAL_COL].dropna().iloc[-1]


rows = []
for svdd_epochs in SVDD_EPOCHS:
    raw_returns = []
    for seed in range(N_SEEDS):
        trial = f'e{svdd_epochs}_s{seed}'
        r = read_final_return(SAVE_ROOT, ENV, trial)
        if r is not None:
            raw_returns.append(r)
        else:
            print(f"[WARN] No result: epochs={svdd_epochs}, seed={seed}")

    norm_scores = [get_normalized_score(ENV, r) for r in raw_returns] if raw_returns else []
    rows.append({
        'svdd_train_epochs':        svdd_epochs,
        'n_seeds':                  len(norm_scores),
        'mean_normalized_return':   np.mean(norm_scores) if norm_scores else float('nan'),
        'std_normalized_return':    np.std(norm_scores)  if norm_scores else float('nan'),
    })

os.makedirs('results', exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv('results/exp2_epoch_sensitivity.csv', index=False)
print(df.to_string(index=False))

# Line plot
fig, ax = plt.subplots(figsize=(7, 4))
means = df['mean_normalized_return'].values
stds  = df['std_normalized_return'].values
epochs = df['svdd_train_epochs'].values

ax.plot(epochs, means, marker='o', linewidth=2, color='steelblue', label='Mean (3 seeds)')
ax.fill_between(epochs, means - stds, means + stds, alpha=0.25, color='steelblue', label='±1 std')
ax.set_xlabel('SVDD Training Epochs', fontsize=12)
ax.set_ylabel('Normalized Return (%)', fontsize=12)
ax.set_title(f'SVDD Epoch Sensitivity — {ENV}', fontsize=13)
ax.set_xticks(epochs)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('results/exp2_epoch_sensitivity.png', dpi=150)
plt.close()
print("Plot saved to results/exp2_epoch_sensitivity.png")
