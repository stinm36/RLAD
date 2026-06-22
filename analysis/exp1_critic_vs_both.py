"""Exp 1: Critic-only vs Both weighting ablation.

Reads progress.csv from:
  saves_ablation_critic_only_identity/  (weight_actor=False)
  saves_ablation_both_identity/         (weight_actor=True)

Outputs: results/exp1_critic_vs_both.csv
  columns: variant, env, mean_normalized_return, std_normalized_return
"""
import os
import glob
import numpy as np
import pandas as pd
import gym
import d4rl  # noqa: F401  (registers envs)

ENVS = ['hopper-medium-v2', 'walker2d-medium-v2', 'halfcheetah-medium-v2']
VARIANTS = {
    'critic_only': 'saves_ablation_critic_only_identity',
    'both':        'saves_ablation_both_identity',
}
N_SEEDS = 5
EVAL_COL = 'evaluation/Average Returns'   # rlkit default column name


def get_normalized_score(env_name, raw_return):
    env = gym.make(env_name)
    score = env.get_normalized_score(raw_return) * 100
    env.close()
    return score


def read_final_return(root, env, seed):
    """Return the last logged evaluation/Average Returns for a given seed."""
    pattern = os.path.join(root, '**', f'seed_{seed}', '**', env, '**', 'progress.csv')
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    df = pd.read_csv(files[0])
    if EVAL_COL not in df.columns:
        print(f"[WARN] Column '{EVAL_COL}' not found in {files[0]}")
        return None
    return df[EVAL_COL].dropna().iloc[-1]


rows = []
for variant, save_root in VARIANTS.items():
    for env in ENVS:
        raw_returns = []
        for seed in range(N_SEEDS):
            r = read_final_return(save_root, env, seed)
            if r is not None:
                raw_returns.append(r)
            else:
                print(f"[WARN] No result found: variant={variant}, env={env}, seed={seed}")

        if raw_returns:
            norm_scores = [get_normalized_score(env, r) for r in raw_returns]
            rows.append({
                'variant':                  variant,
                'env':                      env,
                'n_seeds':                  len(norm_scores),
                'mean_normalized_return':   np.mean(norm_scores),
                'std_normalized_return':    np.std(norm_scores),
            })
        else:
            rows.append({'variant': variant, 'env': env, 'n_seeds': 0,
                         'mean_normalized_return': float('nan'),
                         'std_normalized_return':  float('nan')})

os.makedirs('results', exist_ok=True)
df = pd.DataFrame(rows)
out_path = 'results/exp1_critic_vs_both.csv'
df.to_csv(out_path, index=False)
print(df.to_string(index=False))
print(f"\nSaved to {out_path}")
