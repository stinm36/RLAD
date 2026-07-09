"""Goal B — spatial / contour view of AD scores in embedding space.

Renders Deep SVDD score and DSEBM energy as smooth fields over two 2D canvases:

  • Shared canvas (PRIMARY): PCA of the raw (state, action) vectors — neutral,
    privileges neither detector. DSEBM energy is *defined* on this input space,
    so its contour is meaningful here; SVDD score is overlaid on the same coords.
  • SVDD-native canvas (SECONDARY): PCA of SVDD's R^128 latent. SVDD score is
    (near-)radial here by construction; overlaying DSEBM energy shows where the
    two detectors agree / disagree.

Output: a 2x2 grid (rows = canvas, cols = detector) saved as PNG.

    python analysis/ad_embedding/viz_goalB.py --env hopper-medium-v2
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from analysis.ad_embedding import d4rl_data, score as ad_score


def _field(ax, xy, vals, title, cmap, grid=220, pct=(2, 98)):
    """Filled contour of `vals` over 2D coords `xy`, with a faint scatter."""
    lo, hi = np.percentile(vals, pct)
    v = np.clip(vals, lo, hi)
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, grid),
                         np.linspace(ymin, ymax, grid))
    gz = griddata(xy, v, (gx, gy), method="linear")
    cf = ax.contourf(gx, gy, gz, levels=14, cmap=cmap, alpha=0.9)
    ax.scatter(xy[:, 0], xy[:, 1], s=1, c="k", alpha=0.04, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="hopper-medium-v2")
    ap.add_argument("--n", type=int, default=30000, help="points to plot (subsample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sdim, adim = d4rl_data.dims(a.env)
    sa = d4rl_data.load_state_action(a.env)
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(sa), size=min(a.n, len(sa)), replace=False)
    sa_sub = sa[idx]

    sc = ad_score.score_dataset(a.env, sa_sub, sdim, adim)
    svdd, dsebm, emb = sc["svdd_score"], sc["dsebm_energy"], sc["svdd_embed"]

    # canvases
    raw_xy = PCA(n_components=2, random_state=a.seed).fit_transform(sa_sub.numpy())
    nat_xy = PCA(n_components=2, random_state=a.seed).fit_transform(emb)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    _field(axes[0, 0], raw_xy, svdd, "SVDD score  ·  raw (s,a) PCA [shared]", "viridis")
    _field(axes[0, 1], raw_xy, dsebm, "DSEBM energy  ·  raw (s,a) PCA [shared]", "magma")
    _field(axes[1, 0], nat_xy, svdd, "SVDD score  ·  SVDD-latent PCA [native]", "viridis")
    _field(axes[1, 1], nat_xy, dsebm, "DSEBM energy  ·  SVDD-latent PCA [native]", "magma")
    fig.suptitle(f"AD score fields — {a.env}  (n={len(idx)})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = a.out or os.path.join(os.path.dirname(__file__), "figs",
                                f"goalB_{a.env}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[viz_goalB] saved {out}")


if __name__ == "__main__":
    main()
