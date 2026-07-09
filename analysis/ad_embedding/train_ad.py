"""Train Deep SVDD and DSEBM anomaly detectors on a D4RL dataset.

Reuses the repo's own trainers (`svdd.svdd.TrainerDeepSVDD`, `DSEBM.trainer.Trainer`)
but feeds them data through the lightweight h5py loader (`d4rl_data`) so no
gym / mujoco-py / d4rl install is needed.

AD nets are tiny MLPs, so training is dispatch-bound — a LARGE batch (default 8192)
slashes the per-epoch minibatch count and makes CPU training fast.

Weights land where the repo expects them:
    weights_ad/svdd/{env}/pretrained_ae.pth , model.pth
    weights_ad/dsebm/{env}/checkpoint.pth

Run from the repo root:
    python analysis/ad_embedding/train_ad.py --env hopper-medium-v2
"""

import argparse
import os
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

# repo root on path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from analysis.ad_embedding import d4rl_data
from svdd.svdd import TrainerDeepSVDD
from DSEBM.model import DSEBM
from DSEBM.trainer import Trainer as DSEBMTrainer


def make_args(env, epochs, batch, dim):
    """Minimal args namespace covering both trainers' attribute needs."""
    return SimpleNamespace(
        env=env,
        ad_save_path="weights_ad",          # SVDD adds its own 'svdd/{env}' subdir
        hidden_dim=256,
        latent_dim=128,
        epochs_ad=epochs,
        lr_ad=1e-4,                          # SVDD (matches examples/RLAD.py)
        lr=1e-4,                             # DSEBM
        weight_decay_ae=0.5e-3,
        weight_decay_svdd=0.5e-6,
        lr_milestones=[int(epochs * 0.25), int(epochs * 0.75)],
        batch_size_ad=batch,
        dim=dim,                             # obs_dim + action_dim, for DSEBM b_prime
        svdd_tag="",
        svdd_train_epochs=0,                 # 0 -> falls back to epochs_ad
    )


def get_loader(sa, batch, dim, name):
    # drop_last=True is REQUIRED for DSEBM: its b_prime is fixed (batch, dim),
    # so a short final batch would break the (x - b_prime) broadcast. Harmless
    # for SVDD (also avoids a size-1 final batch breaking BatchNorm).
    ds = TensorDataset(sa)
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                      num_workers=0)


class _Unwrap:
    """TensorDataset yields (tensor,) tuples; the trainers expect bare tensors."""
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        for (x,) in self.loader:
            yield x
    def __len__(self):
        return len(self.loader)


def train_svdd(args, sa, device):
    print(f"\n===== Deep SVDD : {args.env} =====")
    loader = _Unwrap(get_loader(sa, args.batch_size_ad, args.dim, "svdd"))
    t = TrainerDeepSVDD(args, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                        data=loader, device=device)
    t.pretrain()
    t.train()
    print(f"[svdd] done -> weights_ad/svdd/{args.env}/")


def train_dsebm(args, sa, device):
    print(f"\n===== DSEBM : {args.env} =====")
    dargs = SimpleNamespace(**vars(args))
    dargs.ad_save_path = os.path.join("weights_ad", "dsebm")  # -> weights_ad/dsebm/{env}/checkpoint.pth
    loader = _Unwrap(get_loader(sa, args.batch_size_ad, args.dim, "dsebm"))
    model = DSEBM(STATE_DIM, ACTION_DIM, hidden_dim=args.hidden_dim).to(device)
    t = DSEBMTrainer(model, device, loader, dargs)
    t.train()
    print(f"[dsebm] done -> weights_ad/dsebm/{args.env}/checkpoint.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="hopper-medium-v2")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--modules", nargs="+", default=["svdd", "dsebm"],
                    choices=["svdd", "dsebm"])
    ap.add_argument("--threads", type=int, default=8,
                    help="CPU threads; small MLPs don't benefit from all cores and "
                         "grabbing them all is antisocial on a shared box")
    a = ap.parse_args()

    if not torch.cuda.is_available() and a.threads > 0:
        torch.set_num_threads(a.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_ad] env={a.env} device={device} batch={a.batch} epochs={a.epochs}")

    global STATE_DIM, ACTION_DIM
    STATE_DIM, ACTION_DIM = d4rl_data.dims(a.env)
    dim = STATE_DIM + ACTION_DIM
    sa = d4rl_data.load_state_action(a.env)          # (N, dim) float32 tensor
    print(f"[train_ad] loaded (s,a): {tuple(sa.shape)}  obs={STATE_DIM} act={ACTION_DIM}")

    args = make_args(a.env, a.epochs, a.batch, dim)
    if "svdd" in a.modules:
        train_svdd(args, sa, device)
    if "dsebm" in a.modules:
        train_dsebm(args, sa, device)
    print("\n[train_ad] ALL DONE")


if __name__ == "__main__":
    main()
