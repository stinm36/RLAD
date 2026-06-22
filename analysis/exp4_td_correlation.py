"""Exp 4: TD error vs anomaly score correlation.

Requires:
  - A trained Deep SVDD for the target env
  - A saved RL snapshot (itr_*.pkl) from any RLAD run

Loads the snapshot's qf1/qf2/policy, iterates over the offline dataset,
computes per-sample TD error and anomaly score, then reports Spearman
correlation and saves a scatter plot.

Usage:
  cd /path/to/2026rlocc
  python analysis/exp4_td_correlation.py \
      --snapshot path/to/itr_3000.pkl \
      --env hopper-medium-v2
"""
import sys
import os
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
import gym
import d4rl  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from svdd.svdd import network
from svdd.dataset import D4RLDataset
import rlkit.torch.pytorch_util as ptu


def load_svdd(env_name, hidden_dim, latent_dim, svdd_tag, device):
    tag = ('_' + svdd_tag) if svdd_tag else ''
    svdd_path   = f'weights/SVDD_{env_name}{tag}.pth'
    pretrain_path = f'weights/pretrained_parameters_{env_name}.pth'

    env = gym.make(env_name)
    obs_dim    = env.observation_space.low.size
    action_dim = env.action_space.low.size
    env.close()

    svdd = network(obs_dim, action_dim, hidden_dim, latent_dim).to(device)
    svdd.load_state_dict(torch.load(svdd_path, map_location=device))
    svdd.eval()

    state_dict = torch.load(pretrain_path, map_location=device)
    c = torch.tensor(state_dict['center'], dtype=torch.float32).to(device)
    return svdd, c


def load_snapshot(path, device):
    with open(path, 'rb') as f:
        snapshot = pickle.load(f)
    policy = snapshot['policy'].to(device).eval()
    qf1    = snapshot['qf1'].to(device).eval()
    qf2    = snapshot['qf2'].to(device).eval()
    return policy, qf1, qf2


@torch.no_grad()
def compute_td_and_anomaly(env_name, policy, qf1, qf2, svdd, c, device,
                           discount=0.99, batch_size=512):
    env = gym.make(env_name)
    data = env.get_dataset()
    env.close()

    obs     = torch.tensor(data['observations'],      dtype=torch.float32)
    actions = torch.tensor(data['actions'],           dtype=torch.float32)
    rewards = torch.tensor(data['rewards'],           dtype=torch.float32).unsqueeze(1)
    next_obs = torch.tensor(data['next_observations'], dtype=torch.float32)
    terminals = torch.tensor(data['terminals'],       dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(obs, actions, rewards, next_obs, terminals)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    td_errors   = []
    anom_scores = []

    for o, a, r, no, t in loader:
        o, a, r, no, t = [x.to(device) for x in (o, a, r, no, t)]

        # TD error: |Q(s,a) - (r + γ * min(Q1,Q2)(s', π(s')))|
        next_action, *_ = policy(no, reparameterize=False, return_log_prob=False)
        target_q = torch.min(qf1(no, next_action), qf2(no, next_action))
        td_target = r + (1.0 - t) * discount * target_q
        td_err = (qf1(o, a) - td_target.detach()).abs().squeeze(1)

        # Anomaly score at (s, a)
        sa = torch.cat([o, a], dim=-1)
        z  = svdd(sa)
        score = torch.sqrt(torch.sum((z - c) ** 2, dim=1))

        td_errors.append(td_err.cpu().numpy())
        anom_scores.append(score.cpu().numpy())

    return np.concatenate(td_errors), np.concatenate(anom_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot',   required=True, help='Path to itr_*.pkl snapshot')
    parser.add_argument('--env',        default='hopper-medium-v2')
    parser.add_argument('--svdd_tag',   default='')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--discount',   default=0.99, type=float)
    parser.add_argument('--gpu',        default='0')
    parser.add_argument('--subsample',  default=50000, type=int,
                        help='Max samples for scatter plot (0 = all)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    ptu.set_gpu_mode(True)

    print(f'Loading SVDD from weights/SVDD_{args.env}...')
    svdd, c = load_svdd(args.env, args.hidden_dim, args.latent_dim, args.svdd_tag, device)

    print(f'Loading snapshot from {args.snapshot}...')
    policy, qf1, qf2 = load_snapshot(args.snapshot, device)

    print('Computing TD errors and anomaly scores...')
    td_errors, anom_scores = compute_td_and_anomaly(
        args.env, policy, qf1, qf2, svdd, c, device, discount=args.discount
    )

    rho, pval = spearmanr(anom_scores, td_errors)
    print(f'\nSpearman correlation: ρ = {rho:.4f}  (p = {pval:.2e})')
    print(f'N samples: {len(td_errors)}')

    os.makedirs('results', exist_ok=True)

    # Scatter plot (subsample for readability)
    n = len(td_errors)
    if args.subsample > 0 and n > args.subsample:
        idx = np.random.choice(n, args.subsample, replace=False)
    else:
        idx = np.arange(n)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(anom_scores[idx], td_errors[idx],
               alpha=0.15, s=2, color='steelblue', rasterized=True)
    ax.set_xlabel('Anomaly Score (SVDD distance)', fontsize=12)
    ax.set_ylabel('TD Error', fontsize=12)
    ax.set_title(
        f'TD Error vs Anomaly Score — {args.env}\nSpearman ρ = {rho:.3f}  (p = {pval:.2e})',
        fontsize=12
    )
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    out_path = f'results/exp4_td_correlation_{args.env}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Scatter plot saved to {out_path}')

    # Also save raw data
    np.savez(f'results/exp4_td_correlation_{args.env}.npz',
             td_errors=td_errors, anom_scores=anom_scores)


if __name__ == '__main__':
    main()
