"""Goal A — score distribution & detector-agreement view (no shared embedding).

Compares Deep SVDD score vs DSEBM energy directly, across one or more D4RL
dataset variants (e.g. medium / medium-expert / medium-replay). This sidesteps
the incompatible-embedding problem: we compare the *scores themselves*.

Two panels:
  • Left  : per-point scatter of z-scored SVDD score (x) vs DSEBM energy (y),
            coloured by dataset. Spearman ρ per dataset quantifies agreement.
  • Right : marginal distributions (KDE) of each detector's score per dataset,
            showing distribution shift across dataset quality.

    python analysis/ad_embedding/viz_goalA.py \
        --envs hopper-medium-v2 hopper-medium-expert-v2 hopper-medium-replay-v2
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from analysis.ad_embedding import d4rl_data, score as ad_score


def _have_weights(env, weights="weights_ad"):
    return (os.path.exists(os.path.join(weights, "svdd", env, "model.pth"))
            and os.path.exists(os.path.join(weights, "dsebm", env, "checkpoint.pth")))


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def _kde_curve(vals, n=200):
    lo, hi = np.percentile(vals, [1, 99])
    xs = np.linspace(lo, hi, n)
    return xs, gaussian_kde(vals)(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["hopper-medium-v2"])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    data = {}
    for env in a.envs:
        if not _have_weights(env):
            print(f"[viz_goalA] skip {env} (no trained weights yet)")
            continue
        sdim, adim = d4rl_data.dims(env)
        sa = d4rl_data.load_state_action(env)
        idx = rng.choice(len(sa), size=min(a.n, len(sa)), replace=False)
        sc = ad_score.score_dataset(env, sa[idx], sdim, adim)
        data[env] = (sc["svdd_score"], sc["dsebm_energy"])

    if not data:
        print("[viz_goalA] nothing to plot — no envs have trained weights.")
        return

    fig, (axL, axR_top) = plt.subplots(1, 2, figsize=(14, 6))
    # right column split into two stacked marginal axes
    axR_top.remove()
    axR1 = fig.add_subplot(2, 2, 2)
    axR2 = fig.add_subplot(2, 2, 4)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(data), 3)))

    for (env, (svdd, dsebm)), col in zip(data.items(), colors):
        rho, _ = spearmanr(svdd, dsebm)
        label = f"{env.replace('-v2','')}  (ρ={rho:.2f})"
        axL.scatter(_z(svdd), _z(dsebm), s=3, alpha=0.25, color=col, label=label,
                    linewidths=0)
        xs, ys = _kde_curve(svdd); axR1.plot(xs, ys, color=col, label=env.replace('-v2',''))
        xs, ys = _kde_curve(dsebm); axR2.plot(xs, ys, color=col)

    axL.set_xlabel("SVDD score (z)"); axL.set_ylabel("DSEBM energy (z)")
    axL.set_title("Detector agreement — SVDD vs DSEBM")
    axL.legend(fontsize=8, markerscale=3, loc="best")
    axR1.set_title("SVDD score distribution"); axR1.set_xlabel("score"); axR1.set_ylabel("density")
    axR1.legend(fontsize=8)
    axR2.set_title("DSEBM energy distribution"); axR2.set_xlabel("energy"); axR2.set_ylabel("density")
    fig.suptitle("AD score distribution & agreement", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    tag = a.envs[0].split("-")[0] if len(a.envs) > 1 else a.envs[0].replace("-v2", "")
    out = a.out or os.path.join(os.path.dirname(__file__), "figs", f"goalA_{tag}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[viz_goalA] saved {out}  ({len(data)} dataset(s))")


if __name__ == "__main__":
    main()
