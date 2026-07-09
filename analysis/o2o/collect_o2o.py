"""Aggregate O2O run curves and compute warm-start metrics.

Each run's rlkit progress.csv contains BOTH phases appended, separated by a
repeated header row (Phase 1 offline, then Phase 2 online). We split on that
boundary, then compute:
  offline_final : eval return at end of offline pretrain (init quality proxy)
  online_start  : first online eval return
  dip           : min online return over the first --dip_k epochs (transition drop)
  online_final  : eval return at end of online
  online_auc    : mean online return (sample efficiency proxy)

Produces a comparison figure (offline dashed → online solid, per arm) and a
metrics table.

    python analysis/o2o/collect_o2o.py --root /data2/sohyung/runs_o2o
"""
import argparse, csv, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RET = "evaluation/Average Returns"


def read_run(run_dir):
    """Return (offline_returns, online_returns) as lists of float."""
    p = os.path.join(run_dir, "progress.csv")
    if not os.path.isfile(p):
        return [], []
    with open(p) as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = header.index(RET)
        blocks, cur = [], []
        for row in reader:
            if not row:
                continue
            if row[idx] == RET:          # repeated header = phase boundary
                blocks.append(cur); cur = []
                continue
            try:
                cur.append(float(row[idx]))
            except ValueError:
                continue
        blocks.append(cur)
    # first block = offline, remainder concatenated = online (scratch: only online)
    if len(blocks) == 1:
        return [], blocks[0]
    return blocks[0], [v for b in blocks[1:] for v in b]


def metrics(off, on, dip_k):
    m = {}
    m["offline_final"] = off[-1] if off else None
    m["online_start"] = on[0] if on else None
    m["dip"] = float(np.min(on[:dip_k])) if on else None
    m["online_final"] = on[-1] if on else None
    m["online_auc"] = float(np.mean(on)) if on else None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data2/sohyung/runs_o2o")
    ap.add_argument("--dip_k", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = sorted(glob.glob(os.path.join(a.root, "o2o_*")))
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    table = []
    for i, rd in enumerate(runs):
        name = os.path.basename(rd)
        off, on = read_run(rd)
        if not off and not on:
            continue
        c = colors[i % 10]
        x_off = list(range(-len(off), 0))
        x_on = list(range(0, len(on)))
        if off:
            ax.plot(x_off, off, ls="--", color=c, alpha=0.7)
        if on:
            ax.plot(x_on, on, ls="-", color=c, label=name.replace("o2o_", ""))
        m = metrics(off, on, a.dip_k); m["run"] = name
        table.append(m)

    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.text(0.2, ax.get_ylim()[1], "offline→online", fontsize=8, va="top")
    ax.set_xlabel("epoch (offline < 0 dashed | online ≥ 0 solid)")
    ax.set_ylabel("eval return")
    ax.set_title("O2O warm-start comparison")
    ax.legend(fontsize=7, ncol=2)
    out = a.out or os.path.join(a.root, "o2o_curves.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)

    print(f"\nsaved {out}\n")
    hdr = f"{'run':42}{'off_final':>11}{'on_start':>10}{'dip':>9}{'on_final':>10}{'on_auc':>9}"
    print(hdr); print("-" * len(hdr))
    for m in table:
        def f(x): return f"{x:.1f}" if isinstance(x, (int, float)) else "-"
        print(f"{m['run'][:42]:42}{f(m['offline_final']):>11}{f(m['online_start']):>10}"
              f"{f(m['dip']):>9}{f(m['online_final']):>10}{f(m['online_auc']):>9}")
    json.dump(table, open(os.path.join(a.root, "o2o_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
