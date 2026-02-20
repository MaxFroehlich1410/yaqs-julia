#!/usr/bin/env python3
"""
Plot Experiment 1 (Order Verification) results for paper.

Creates a 2x3 figure:
  Row 1: TFIM       Row 2: XXZ
  Col 1: Infidelity vs dt    (log-log convergence)
  Col 2: Energy error vs dt  (log-log convergence)
  Col 3: Infidelity vs wall-clock runtime (log-log efficiency)

Usage:
    python benchmarks/plot_exp01.py [--T 5.0] [--outdir benchmarks/figures]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "exp01_order")

METHODS = {
    "1TDVP":          dict(color="#1f4e79", marker="o", ls="-",  label="1-TDVP (fixed $\\chi$)"),
    "BUG2_fixed":     dict(color="#5b9bd5", marker="s", ls="--", label="BUG2 (fixed $\\chi$)"),
    "2TDVP":          dict(color="#c0392b", marker="^", ls="-",  label="2-TDVP (adaptive)"),
    "BUG2_adaptive":  dict(color="#e67e22", marker="D", ls="--", label="BUG2 (adaptive)"),
}

MODELS = ["tfim", "xxz"]
MODEL_TITLES = {"tfim": "TFIM", "xxz": "XXZ"}

MACHINE_EPS = 1e-16

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest(model: str, T: float) -> pd.DataFrame:
    """Load run_manifest.csv, overlay rerun data if available, deduplicate."""
    base_dir = os.path.join(RESULTS_ROOT, model, f"T_{T}")
    path = os.path.join(base_dir, "run_manifest.csv")
    if not os.path.isfile(path):
        print(f"  [WARN] Not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["_source"] = "original"

    # If a rerun manifest exists, append it so its rows take priority
    rerun_path = os.path.join(base_dir, "run_manifest_rerun.csv")
    if os.path.isfile(rerun_path):
        df_rerun = pd.read_csv(rerun_path)
        df_rerun["_source"] = "rerun"
        n_rerun = len(df_rerun)
        df = pd.concat([df, df_rerun], ignore_index=True)
        print(f"    merged {n_rerun} rerun rows from {rerun_path}")

    # Deduplicate: keep LAST occurrence per (method, dt).
    # Since rerun rows are appended after originals, "last" prefers reruns.
    df = df.drop_duplicates(subset=["method", "dt"], keep="last")

    df["err_infidelity_T"] = df["err_infidelity_T"].abs().clip(lower=MACHINE_EPS)
    df["err_energy_T"]     = df["err_energy_T"].abs().clip(lower=MACHINE_EPS)
    return df

# ---------------------------------------------------------------------------
# Reference-slope triangle (log-log)
# ---------------------------------------------------------------------------

def slope_triangle(ax, x_center, y_center, slope=2, width_decades=0.30,
                   color="0.45", label=None):
    """
    Draw a right triangle whose hypotenuse has the given slope in log-log
    space, centred (in log-space) on (x_center, y_center).
    """
    half = width_decades / 2.0
    x0 = x_center * 10**(-half)
    x1 = x_center * 10**(+half)
    y0 = y_center * 10**(-slope * half)
    y1 = y_center * 10**(+slope * half)

    ax.fill([x0, x1, x1, x0], [y0, y0, y1, y0],
            fc=color, alpha=0.12, ec=color, lw=0.8, zorder=1)
    if label:
        ax.text(x1 * 1.15, np.sqrt(y0 * y1), label,
                fontsize=6.5, color=color, va="center", ha="left")

# ---------------------------------------------------------------------------
# Panel plotters
# ---------------------------------------------------------------------------

def plot_convergence(ax, df, ycol, ylabel):
    """Error vs dt (log-log) for all four methods."""
    for method, sty in METHODS.items():
        sub = df[df["method"] == method].sort_values("dt")
        if sub.empty:
            continue
        ax.plot(sub["dt"], sub[ycol],
                color=sty["color"], marker=sty["marker"], ls=sty["ls"],
                markersize=4.5, lw=1.3, label=sty["label"], zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(ylabel)

    dt_vals = sorted(df["dt"].unique())
    ax.set_xticks(dt_vals)
    ax.set_xticklabels([f"{v:g}" for v in dt_vals], fontsize=6)
    ax.minorticks_off()
    ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)


def plot_efficiency(ax, df):
    """Infidelity vs wall-clock runtime (log-log)."""
    for method, sty in METHODS.items():
        sub = df[df["method"] == method].sort_values("runtime_seconds")
        if sub.empty:
            continue
        ax.plot(sub["runtime_seconds"], sub["err_infidelity_T"],
                color=sty["color"], marker=sty["marker"], ls=sty["ls"],
                markersize=4.5, lw=1.3, label=sty["label"], zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Infidelity")
    ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)

# ---------------------------------------------------------------------------
# Slope triangle placement helper
# ---------------------------------------------------------------------------

def place_slope_triangle(ax, df, method, ycol, slope=2):
    """Place an O(dt^slope) triangle near the midpoint of *method*'s data."""
    sub = df[df["method"] == method].sort_values("dt")
    if len(sub) < 2:
        return
    dts = sub["dt"].values
    errs = sub[ycol].values
    x_c = np.exp(np.mean(np.log(dts)))
    y_c = np.exp(np.mean(np.log(errs))) * 0.08
    slope_triangle(ax, x_c, y_c, slope=slope, width_decades=0.25,
                   label=r"$O(\Delta t^{%d})$" % slope)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=5.0)
    ap.add_argument("--outdir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "figures"))
    args = ap.parse_args()
    T = args.T
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    data = {}
    for m in MODELS:
        data[m] = load_manifest(m, T)
        nrows = len(data[m])
        print(f"  Loaded {m} T={T}: {nrows} rows")

    if all(d.empty for d in data.values()):
        sys.exit("ERROR: no data found.")

    # Publication rcParams
    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        8,
        "axes.labelsize":   9,
        "axes.titlesize":   9,
        "legend.fontsize":  7,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.05,
    })

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.96, bottom=0.14,
                        hspace=0.42, wspace=0.38)
    panel_tags = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]

    for ri, model in enumerate(MODELS):
        df = data[model]
        ax_inf, ax_eng, ax_eff = axes[ri]

        if df.empty:
            for ax in (ax_inf, ax_eng, ax_eff):
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="0.5")
            continue

        # ── Col 1: Infidelity vs dt ──────────────────────────────────
        plot_convergence(ax_inf, df, "err_infidelity_T", "Infidelity")
        place_slope_triangle(ax_inf, df, "BUG2_fixed", "err_infidelity_T")

        # Annotate machine-precision floor for 1TDVP if visible
        tdvp1 = df[df["method"] == "1TDVP"]
        if not tdvp1.empty and tdvp1["err_infidelity_T"].max() < 1e-13:
            y_eps = tdvp1["err_infidelity_T"].median()
            ax_inf.axhline(y_eps, color="#1f4e79", ls=":", lw=0.6, alpha=0.4)
            ax_inf.text(
                0.50, 0.03, r"machine $\varepsilon$",
                transform=ax_inf.transAxes, fontsize=5.5, color="#1f4e79",
                alpha=0.7, va="bottom", ha="center")

        # ── Col 2: Energy error vs dt ────────────────────────────────
        plot_convergence(ax_eng, df, "err_energy_T", r"Energy error $|E-E_{\rm ex}|$")
        place_slope_triangle(ax_eng, df, "BUG2_fixed", "err_energy_T")

        tdvp1_e = df[df["method"] == "1TDVP"]
        if not tdvp1_e.empty and tdvp1_e["err_energy_T"].max() < 1e-10:
            y_eps_e = tdvp1_e["err_energy_T"].median()
            ax_eng.axhline(y_eps_e, color="#1f4e79", ls=":", lw=0.6, alpha=0.4)
            ax_eng.text(
                0.97, 0.03, r"machine $\varepsilon$",
                transform=ax_eng.transAxes, fontsize=5.5, color="#1f4e79",
                alpha=0.7, va="bottom", ha="right")

        # ── Col 3: Infidelity vs Runtime ─────────────────────────────
        plot_efficiency(ax_eff, df)

        # ── Panel labels & row titles ────────────────────────────────
        for ci, ax in enumerate((ax_inf, ax_eng, ax_eff)):
            ax.text(0.04, 0.96, panel_tags[ri][ci],
                    transform=ax.transAxes, fontsize=9,
                    fontweight="bold", va="top", ha="left")

    # Row titles — place as a text box inside the last panel of each row
    for ri, model in enumerate(MODELS):
        axes[ri, 2].text(
            0.96, 0.96, MODEL_TITLES[model],
            transform=axes[ri, 2].transAxes, fontsize=10, fontweight="bold",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7",
                      alpha=0.85))

    # Column titles
    col_titles = ["Convergence (infidelity)", "Convergence (energy)",
                  "Efficiency"]
    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=8.5, pad=6)

    # Shared legend below figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   frameon=True, fancybox=False, edgecolor="0.7",
                   fontsize=7.5, bbox_to_anchor=(0.53, 0.005))

    # Save
    stem = f"exp01_order_T{T}"
    for ext in ("pdf", "png"):
        out = os.path.join(args.outdir, f"{stem}.{ext}")
        fig.savefig(out)
        print(f"  Saved {out}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
