"""Measure raw AD-score scale for every trained env × module combination.

svdd / dsebm  : loaded from trained weights (weights_ad/)
mahal / gmm   : fit live on each env's data (cheap: mean/cov, EM)
maf/dagmm/fanogan : neural, not trained yet — skipped (noted in the doc)

Outputs combined JSON + an env×module heatmap of hill(median) (= what the
hardcoded T=5.0 weight function maps each module's typical score to).
"""
import argparse, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from analysis.ad_embedding import d4rl_data, score as ad_score
from mahal.mahal import MahalanobisAD
from gmm.gmm import GPUGMM

ENVS = [
    "hopper-medium-v2", "hopper-medium-replay-v2", "hopper-medium-expert-v2",
    "walker2d-medium-v2", "walker2d-medium-replay-v2", "walker2d-medium-expert-v2",
    "halfcheetah-medium-v2", "halfcheetah-medium-replay-v2", "halfcheetah-medium-expert-v2",
]
MODULES = ["svdd", "dsebm", "mahal", "gmm"]


def _loader(sa, batch=8192):
    class _U:
        def __iter__(self):
            for (x,) in DataLoader(TensorDataset(sa), batch_size=batch, shuffle=False):
                yield x
        def __len__(self): return (len(sa) + batch - 1) // batch
    return _U()


def _stat(env, module, v):
    v = np.asarray(v, dtype=np.float64)
    med = float(np.percentile(v, 50))
    return dict(env=env, module=module,
               sign=("nonneg" if v.min() >= 0 else "has_neg"),
               median=med, mean=float(v.mean()),
               p99=float(np.percentile(v, 99)), std=float(v.std()),
               hill_med=(med / (med + 5.0)) if (med + 5.0) != 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_score", type=int, default=50000)
    ap.add_argument("--n_fit", type=int, default=200000)
    a = ap.parse_args()
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)

    rows = []
    for env in ENVS:
        if not os.path.exists(os.path.join("weights_ad", "svdd", env, "model.pth")):
            print(f"skip {env}: no svdd weights"); continue
        sdim, adim = d4rl_data.dims(env)
        sa = d4rl_data.load_state_action(env)
        si = rng.choice(len(sa), size=min(a.n_score, len(sa)), replace=False)
        fi = rng.choice(len(sa), size=min(a.n_fit, len(sa)), replace=False)
        sa_s, sa_f = sa[si], sa[fi]

        sc = ad_score.score_dataset(env, sa_s, sdim, adim)
        rows.append(_stat(env, "svdd", sc["svdd_score"]))
        rows.append(_stat(env, "dsebm", sc["dsebm_energy"]))

        m = MahalanobisAD("cpu"); m.fit(_loader(sa_f))
        with torch.no_grad():
            rows.append(_stat(env, "mahal", m.score(sa_s).cpu().numpy()))

        g = GPUGMM(n_components=2, device="cpu", n_iter=100); g.fit(_loader(sa_f))
        with torch.no_grad():
            rows.append(_stat(env, "gmm", g.score(sa_s).cpu().numpy()))
        print(f"done {env}")

    outdir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "scales_all.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"saved {outdir}/scales_all.json  ({len(rows)} rows)")

    # heatmap: env (rows) × module (cols), cell = hill(median)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    envs = [e for e in ENVS if any(r["env"] == e for r in rows)]
    grid = np.full((len(envs), len(MODULES)), np.nan)
    for r in rows:
        grid[envs.index(r["env"]), MODULES.index(r["module"])] = r["hill_med"]
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    # diverging around the "useful" midpoint 0.5; far from 0.5 = mis-scaled
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    im = ax.imshow(grid, cmap="RdYlGn_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(MODULES))); ax.set_xticklabels(MODULES)
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([e.replace("-v2", "") for e in envs])
    for i in range(len(envs)):
        for j in range(len(MODULES)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color="black")
    ax.set_title("hill(median) per env × module under hardcoded T=5.0\n"
                 "(0.5 = usable; →0 penalty vanishes; →1 saturates/breaks)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight at median score")
    fig.tight_layout()
    fp = os.path.join(outdir, "scales_heatmap.png")
    fig.savefig(fp, dpi=130)
    print(f"saved {fp}")


if __name__ == "__main__":
    main()
