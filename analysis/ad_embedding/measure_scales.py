"""Measure the raw anomaly-score scale of each AD module on a D4RL dataset.

Demonstrates that, under the CURRENT code, each module emits scores on a wildly
different scale/sign — all of which are then squashed by the same hardcoded
T=5.0 in make_weight_function. Produces a stats table (JSON + stdout) and a
distribution figure for the meeting deck.

Empirically measured here: svdd, dsebm (trained weights), mahal, gmm (fit live).
maf / dagmm / fanogan are neural and slow to train — characterised analytically
in the meeting doc instead.
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


def _loader(sa, batch=8192):
    class _U:
        def __iter__(self):
            for (x,) in DataLoader(TensorDataset(sa), batch_size=batch, shuffle=False):
                yield x
        def __len__(self): return (len(sa) + batch - 1) // batch
    return _U()


def stats(name, v):
    v = np.asarray(v, dtype=np.float64)
    q = np.percentile(v, [0, 1, 50, 99, 100])
    return dict(module=name, sign=("≥0" if v.min() >= 0 else "has negatives"),
                min=q[0], p1=q[1], median=q[2], mean=float(v.mean()),
                p99=q[3], max=q[4], std=float(v.std()),
                # what hill(T=5) maps the MEDIAN score to:
                hill_at_median=float(q[2] / (q[2] + 5.0)) if (q[2] + 5.0) != 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="hopper-medium-v2")
    ap.add_argument("--n_score", type=int, default=50000)
    ap.add_argument("--n_fit", type=int, default=200000)
    a = ap.parse_args()
    torch.set_num_threads(4)

    sdim, adim = d4rl_data.dims(a.env)
    sa = d4rl_data.load_state_action(a.env)
    rng = np.random.default_rng(0)
    si = rng.choice(len(sa), size=min(a.n_score, len(sa)), replace=False)
    fi = rng.choice(len(sa), size=min(a.n_fit, len(sa)), replace=False)
    sa_s, sa_f = sa[si], sa[fi]

    rows, raw = [], {}

    # svdd + dsebm (trained)
    sc = ad_score.score_dataset(a.env, sa_s, sdim, adim)
    raw["svdd"] = sc["svdd_score"];   rows.append(stats("svdd (distance)", raw["svdd"]))
    raw["dsebm"] = sc["dsebm_energy"]; rows.append(stats("dsebm (energy)", raw["dsebm"]))

    # mahal (fit live — mean/cov)
    m = MahalanobisAD("cpu"); m.fit(_loader(sa_f))
    with torch.no_grad():
        raw["mahal"] = m.score(sa_s).cpu().numpy()
    rows.append(stats("mahal (sq-dist)", raw["mahal"]))

    # gmm (fit live — EM)
    g = GPUGMM(n_components=2, device="cpu", n_iter=100); g.fit(_loader(sa_f))
    with torch.no_grad():
        raw["gmm"] = g.score(sa_s).cpu().numpy()
    rows.append(stats("gmm (neg-log-lik)", raw["gmm"]))

    # report
    print(f"\n=== Raw AD-score scale on {a.env} (n={len(si)}) ===")
    hdr = f"{'module':22} {'sign':14} {'median':>12} {'mean':>12} {'p99':>12} {'std':>12} {'hill(med)':>10}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['module']:22} {r['sign']:14} {r['median']:12.4g} {r['mean']:12.4g} "
              f"{r['p99']:12.4g} {r['std']:12.4g} {r['hill_at_median']:10.4f}")

    outdir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"scales_{a.env}.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # figure: per-module distribution, each on its own x-axis (scales incomparable)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in zip(axes.flat, ["svdd", "dsebm", "mahal", "gmm"]):
        v = raw[key]
        lo, hi = np.percentile(v, [0.5, 99.5])
        ax.hist(np.clip(v, lo, hi), bins=80, color="steelblue", alpha=0.85)
        ax.set_title(f"{key}   [{v.min():.3g}, {v.max():.3g}]", fontsize=11)
        ax.axvline(5.0, color="crimson", ls="--", lw=1)   # the hardcoded T
        ax.set_yticks([])
    fig.suptitle(f"Raw AD-score distributions — {a.env}\n"
                 f"(red dashed = hardcoded T=5.0; note each panel's own x-range)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fp = os.path.join(outdir, f"scales_{a.env}.png")
    fig.savefig(fp, dpi=130)
    print(f"\nsaved {fp}")
    print(f"saved {os.path.join(outdir, f'scales_{a.env}.json')}")


if __name__ == "__main__":
    main()
