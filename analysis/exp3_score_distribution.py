"""Exp 3: Anomaly score distribution visualization.

Loads a trained Deep SVDD (default: weights/SVDD_hopper-medium-v2.pth),
computes scores for hopper-medium-v2 and hopper-medium-replay-v2 samples,
and plots an overlapping histogram.

Usage:
  cd /path/to/2026rlocc
  python analysis/exp3_score_distribution.py [--env hopper-medium-v2] [--svdd_tag '']
"""
import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import gym
import d4rl  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from svdd.svdd import network
from svdd.dataset import D4RLDataset


def load_svdd(env_name, hidden_dim, latent_dim, svdd_tag, device):
    tag = ('_' + svdd_tag) if svdd_tag else ''
    weight_path = f'weights/SVDD_{env_name}{tag}.pth'
    pretrain_path = f'weights/pretrained_parameters_{env_name}.pth'

    env = gym.make(env_name)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    env.close()

    svdd = network(obs_dim, action_dim, hidden_dim, latent_dim).to(device)
    svdd.load_state_dict(torch.load(weight_path, map_location=device))
    svdd.eval()

    state_dict = torch.load(pretrain_path, map_location=device)
    c = torch.tensor(state_dict['center'], dtype=torch.float32).to(device)
    return svdd, c


@torch.no_grad()
def compute_scores(dataset_name, svdd, c, device, batch_size=512):
    dataset = D4RLDataset(dataset_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    scores = []
    for x in loader:
        x = x.float().to(device)
        z = svdd(x)
        s = torch.sqrt(torch.sum((z - c) ** 2, dim=1))
        scores.append(s.cpu().numpy())
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',        default='hopper-medium-v2')
    parser.add_argument('--replay_env', default='',
                        help='OOD dataset name (default: env with -replay suffix)')
    parser.add_argument('--svdd_tag',   default='')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--gpu',        default='0')
    parser.add_argument('--bins',       default=100, type=int)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Derive OOD dataset name: hopper-medium-v2 → hopper-medium-replay-v2
    if not args.replay_env:
        base = args.env.replace('-v2', '')
        task = base.split('-')[0]
        args.replay_env = f'{task}-medium-replay-v2'

    print(f'In-distribution  : {args.env}')
    print(f'OOD dataset      : {args.replay_env}')

    svdd, c = load_svdd(args.env, args.hidden_dim, args.latent_dim, args.svdd_tag, device)

    print('Computing scores for in-distribution data...')
    scores_id  = compute_scores(args.env,        svdd, c, device)
    print('Computing scores for OOD data...')
    scores_ood = compute_scores(args.replay_env, svdd, c, device)

    os.makedirs('results', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    # Clip extreme outliers for readability
    p99 = np.percentile(np.concatenate([scores_id, scores_ood]), 99)
    scores_id_clipped  = scores_id[scores_id   <= p99]
    scores_ood_clipped = scores_ood[scores_ood <= p99]

    ax.hist(scores_id_clipped,  bins=args.bins, density=True, alpha=0.6,
            color='steelblue', label=args.env)
    ax.hist(scores_ood_clipped, bins=args.bins, density=True, alpha=0.6,
            color='tomato',    label=args.replay_env)

    ax.set_xlabel('Anomaly Score (SVDD distance)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('SVDD Score Distribution: In-distribution vs OOD', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    out_path = f'results/exp3_score_dist_{args.env}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Histogram saved to {out_path}')

    print(f'\nIn-dist  mean={scores_id.mean():.4f}  std={scores_id.std():.4f}')
    print(f'OOD      mean={scores_ood.mean():.4f}  std={scores_ood.std():.4f}')


if __name__ == '__main__':
    main()
