#!/usr/bin/env python3
"""
Plot Experiment 3 — Pareto scatter (runtime vs infidelity), adaptive only.

Layout: 1 row x 2 columns (TFIM, XXZ).
Each panel shows all adaptive runs as scatter points, with the Pareto frontier
traced for each method.

Usage:
    python benchmarks/plot_exp03_pareto.py [--T 1.0]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "exp03_pareto")
FIGURES_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

MODELS = ["tfim", "xxz"]
MODEL_TITLES = {"tfim": "TFIM", "xxz": "XXZ"}
MACHINE_EPS = 1e-16

STYLE = {
    "2TDVP":         dict(color="#2166ac", marker="o", label="2-TDVP", zorder=4),
    "BUG2_adaptive": dict(color="#d6604d", marker="s", label="BUG2",   zorder=4),
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_adaptive(model: str, T: float) -> pd.DataFrame:
    base_dir = os.path.join(RESULTS_ROOT, model, f"T_{T}")
    path = os.path.join(base_dir, "run_manifest.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["method"].isin(["2TDVP", "BUG2_adaptive"])]
    df["err_infidelity_T"] = df["err_infidelity_T"].abs().clip(lower=MACHINE_EPS)
    # Drop useless runs (infidelity > 0.1)
    df = df[df["err_infidelity_T"] < 0.1]
    return df

# ---------------------------------------------------------------------------
# Pareto frontier (lower runtime AND lower error = better)
# ---------------------------------------------------------------------------

def pareto_frontier(df, rt_col="runtime_seconds", err_col="err_infidelity_T"):
    """Return rows on the Pareto frontier (non-dominated in runtime & error)."""
    pts = df[[rt_col, err_col]].values
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (pts[j, 0] <= pts[i, 0] and pts[j, 1] <= pts[i, 1] and
                    (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1])):
                is_pareto[i] = False
                break
    return df[is_pareto].sort_values(rt_col)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1.0)
    args = ap.parse_args()
    T = args.T
    os.makedirs(FIGURES_DIR, exist_ok=True)

    data = {}
    for m in MODELS:
        data[m] = load_adaptive(m, T)
        print(f"  {m} T={T}: {len(data[m])} adaptive rows")
    if all(d.empty for d in data.values()):
        sys.exit("No data.")

    # Publication style (matching exp01/exp02)
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.8))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.18,
                        wspace=0.38)
    tags = ["(a)", "(b)"]

    for ci, model in enumerate(MODELS):
        ax = axes[ci]
        df = data[model]

        if df.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="0.6")
            continue

        # Scatter all points + Pareto frontier per method
        for meth, sty in STYLE.items():
            sub = df[df["method"] == meth]
            if sub.empty:
                continue

            # Scatter (faded)
            ax.scatter(sub["runtime_seconds"], sub["err_infidelity_T"],
                       c=sty["color"], marker=sty["marker"], s=20,
                       alpha=0.35, edgecolors="none", zorder=3)

            # Pareto frontier (solid line + markers)
            pf = pareto_frontier(sub)
            ax.plot(pf["runtime_seconds"], pf["err_infidelity_T"],
                    color=sty["color"], marker=sty["marker"], ls="-",
                    lw=1.6, ms=4.5, label=sty["label"], zorder=5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wall-clock time (s)")
        ax.set_ylabel("Infidelity")
        ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)

        # Panel tag
        ax.text(0.04, 0.96, tags[ci], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

        # Model label
        ax.text(0.96, 0.96, MODEL_TITLES[model],
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="0.7", alpha=0.85))

        # Legend
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9,
                  edgecolor="0.7", handlelength=1.8)

    # Column titles
    axes[0].set_title("TFIM — Pareto frontier", fontsize=10, pad=8)
    axes[1].set_title("XXZ — Pareto frontier", fontsize=10, pad=8)

    # Save
    stem = f"exp03_pareto_T{T}"
    for ext in ("pdf", "png"):
        out = os.path.join(FIGURES_DIR, f"{stem}.{ext}")
        fig.savefig(out)
        print(f"  Saved {out}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
