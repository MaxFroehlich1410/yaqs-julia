#!/usr/bin/env python3
"""
Plot Experiment 2 — Truncation sensitivity (XXZ only).

Layout: 1 row x 2 columns:
  (a) Infidelity vs SVD threshold  (adaptive: 2-TDVP vs BUG2)
  (b) Infidelity vs chi            (fixed:    1-TDVP vs BUG2_fixed)

Usage:
    python benchmarks/plot_exp02_trunc.py [--T 1.0]
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

BENCH_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = os.path.join(BENCH_DIR, "results")
FIGURES_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

MACHINE_EPS = 1e-16

STYLE_ADAPTIVE = {
    "2TDVP":         dict(color="#2166ac", marker="o", ls="-",  lw=1.6,
                          ms=3.5, label="2-TDVP", zorder=4),
    "BUG2_adaptive": dict(color="#d6604d", marker="s", ls="--", lw=1.6,
                          ms=3.5, label="BUG2",   zorder=4),
}

STYLE_FIXED = {
    "1TDVP":      dict(color="#2166ac", marker="o", ls="-",  lw=1.6,
                       ms=3.5, label="1-TDVP", zorder=4),
    "BUG2_fixed": dict(color="#d6604d", marker="s", ls="--", lw=1.6,
                       ms=3.5, label="BUG2 fixed", zorder=4),
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_exp_root(exp_prefix: str, n_sites: int | None, run_label: str | None) -> str | None:
    """Find the experiment root inside the latest (or specified) timestamped run dir."""
    if not os.path.isdir(RESULTS_BASE):
        return None

    if run_label:
        run_dirs = [os.path.join(RESULTS_BASE, run_label)]
    else:
        run_dirs = []
        for d in os.listdir(RESULTS_BASE):
            p = os.path.join(RESULTS_BASE, d)
            if os.path.isdir(p):
                run_dirs.append(p)
        run_dirs.sort(reverse=True)  # timestamped folder names sort chronologically

    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        exp_dirs = []
        for d in os.listdir(run_dir):
            p = os.path.join(run_dir, d)
            if os.path.isdir(p) and d.startswith(exp_prefix):
                exp_dirs.append(d)
        if not exp_dirs:
            continue

        if n_sites is not None:
            wanted = f"{exp_prefix}_N{n_sites}"
            if wanted in exp_dirs:
                return os.path.join(run_dir, wanted)

        # Fallback for older layout (without _N suffix) or any matching run.
        exp_dirs.sort(reverse=True)
        return os.path.join(run_dir, exp_dirs[0])

    return None


def load_manifest(model: str, T: float, n_sites: int | None, run_label: str | None) -> pd.DataFrame:
    exp_root = _resolve_exp_root("exp02_trunc", n_sites, run_label)
    if exp_root is None:
        return pd.DataFrame()

    base_dir = os.path.join(exp_root, model, f"T_{T}")
    path = os.path.join(base_dir, "run_manifest.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["err_infidelity_T"] = df["err_infidelity_T"].abs().clip(lower=MACHINE_EPS)
    df["err_energy_T"]     = df["err_energy_T"].abs().clip(lower=MACHINE_EPS)
    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--N", type=int, default=None,
                    help="Optional system size N to select exp02_trunc_N<N>.")
    ap.add_argument("--run", type=str, default=None,
                    help="Optional timestamped run folder under benchmarks/results/.")
    args = ap.parse_args()
    T = args.T
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df_all = load_manifest("xxz", T, args.N, args.run)
    if df_all.empty:
        sys.exit("No XXZ data found.")
    print(f"  xxz T={T}: {len(df_all)} rows total")

    df_adapt = df_all[df_all["method"].isin(["2TDVP", "BUG2_adaptive"])].copy()
    df_fixed = df_all[df_all["method"].isin(["1TDVP", "BUG2_fixed"])].copy()

    # Deduplicate (keep last in case of reruns)
    df_adapt = df_adapt.drop_duplicates(subset=["method", "svd_threshold"], keep="last")
    df_fixed = df_fixed.drop_duplicates(subset=["method", "chi_fixed"], keep="last")

    print(f"    adaptive rows: {len(df_adapt)}")
    print(f"    fixed rows:    {len(df_fixed)}")

    # Publication style (matching exp01 plots)
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.8, 2.8))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.18,
                        wspace=0.38)

    # ── Panel (a): Infidelity vs SVD threshold (adaptive) ────────────
    for meth, sty in STYLE_ADAPTIVE.items():
        sub = df_adapt[df_adapt["method"] == meth].sort_values("svd_threshold")
        if sub.empty:
            continue
        ax_a.plot(sub["svd_threshold"], sub["err_infidelity_T"], **sty)

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("SVD threshold")
    ax_a.set_ylabel("Infidelity")
    ax_a.invert_xaxis()
    ax_a.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)
    ax_a.set_title("Adaptive truncation", fontsize=10, pad=8)

    ax_a.text(0.04, 0.96, "(a)", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold", va="top", ha="left")

    # Legend for panel (a) — lower left where data has plateaued
    ax_a.legend(loc="lower left", fontsize=7.5, framealpha=0.9,
                edgecolor="0.7", handlelength=1.8)

    # ── Panel (b): Infidelity vs chi (fixed) ─────────────────────────
    for meth, sty in STYLE_FIXED.items():
        sub = df_fixed[df_fixed["method"] == meth].sort_values("chi_fixed")
        if sub.empty:
            continue
        ax_b.plot(sub["chi_fixed"], sub["err_infidelity_T"], **sty)

    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"Bond dimension $\chi$")
    ax_b.set_ylabel("Infidelity")
    ax_b.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)
    ax_b.set_title("Fixed truncation", fontsize=10, pad=8)

    # Set x-ticks to actual chi values
    chi_vals = sorted(df_fixed["chi_fixed"].unique())
    ax_b.set_xticks(chi_vals)
    ax_b.set_xticklabels([str(int(c)) for c in chi_vals])

    ax_b.text(0.04, 0.96, "(b)", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold", va="top", ha="left")

    # Legend for panel (b)
    ax_b.legend(loc="center left", fontsize=7.5, framealpha=0.9,
                edgecolor="0.7", handlelength=1.8,
                bbox_to_anchor=(0.22, 0.5))

    # Machine epsilon floor annotation on panel (b)
    ax_b.axhline(MACHINE_EPS, color="0.6", ls=":", lw=0.6, zorder=1)
    ax_b.text(chi_vals[0], MACHINE_EPS * 5, r"machine $\varepsilon$",
              fontsize=6.5, color="0.5", ha="left", va="bottom")

    # Save
    stem = f"exp02_trunc_xxz_T{T}"
    for ext in ("pdf", "png"):
        out = os.path.join(FIGURES_DIR, f"{stem}.{ext}")
        fig.savefig(out)
        print(f"  Saved {out}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()