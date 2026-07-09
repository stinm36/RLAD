"""Load trained Deep SVDD / DSEBM detectors and emit per-sample scores.

Shared by the Goal-B (spatial/contour) and Goal-A (distribution) visualisations.
All scoring runs on CPU (the pinned sh_rlad env), in mini-batches so a full
1M-row dataset can be scored without blowing up memory.
"""

import os
from types import SimpleNamespace

import numpy as np
import torch

from svdd.svdd import _SVDDNetwork
from DSEBM.model import DSEBM
from DSEBM.trainer import Trainer as DSEBMTrainer

WEIGHTS = "weights_ad"
HIDDEN_DIM = 256
LATENT_DIM = 128


# ─────────────────────────── Deep SVDD ────────────────────────────
def load_svdd(env, state_dim, action_dim, weights=WEIGHTS, device="cpu"):
    net = _SVDDNetwork(state_dim, action_dim, HIDDEN_DIM, LATENT_DIM).to(device)
    net.load_state_dict(torch.load(os.path.join(weights, "svdd", env, "model.pth"),
                                   map_location=device))
    net.eval()
    ck = torch.load(os.path.join(weights, "svdd", env, "pretrained_ae.pth"),
                    map_location=device)
    c = torch.tensor(ck["center"], device=device)
    return net, c


@torch.no_grad()
def svdd_embed(net, sa, batch=65536, device="cpu"):
    """Project (s,a) into the SVDD R^128 latent. Returns np.ndarray (N, 128)."""
    out = []
    for i in range(0, len(sa), batch):
        out.append(net(sa[i:i + batch].to(device)).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def svdd_score(net, c, sa, batch=65536, device="cpu"):
    """Radial distance to hypersphere centre. Returns np.ndarray (N,)."""
    out = []
    for i in range(0, len(sa), batch):
        z = net(sa[i:i + batch].to(device))
        out.append(torch.sqrt(((z - c) ** 2).sum(-1)).cpu().numpy())
    return np.concatenate(out, 0)


# ─────────────────────────── DSEBM ────────────────────────────────
def load_dsebm(env, state_dim, action_dim, weights=WEIGHTS, device="cpu"):
    args = SimpleNamespace(
        lr=1e-4, batch_size_ad=8192, dim=state_dim + action_dim, env=env,
        ad_save_path=os.path.join(weights, "dsebm"),
        epochs_ad=1, lr_milestones=[1],
    )
    model = DSEBM(state_dim, action_dim, hidden_dim=HIDDEN_DIM).to(device)
    t = DSEBMTrainer(model, device, [], args)
    t.load_ckpt()                     # restores model weights + b_prime
    model.eval()
    return t


@torch.no_grad()
def dsebm_energy(trainer, sa, batch=65536):
    """Per-sample DSEBM energy. Returns np.ndarray (N,)."""
    out = []
    for i in range(0, len(sa), batch):
        out.append(trainer.energy_per_sample(sa[i:i + batch]).cpu().numpy())
    return np.concatenate(out, 0)


# ─────────────────────── convenience bundle ───────────────────────
def score_dataset(env, sa, state_dim, action_dim, weights=WEIGHTS, device="cpu"):
    """Return dict with svdd score, svdd 128-D embedding, dsebm energy."""
    net, c = load_svdd(env, state_dim, action_dim, weights, device)
    dse = load_dsebm(env, state_dim, action_dim, weights, device)
    return {
        "svdd_score": svdd_score(net, c, sa, device=device),
        "svdd_embed": svdd_embed(net, sa, device=device),
        "dsebm_energy": dsebm_energy(dse, sa),
    }
