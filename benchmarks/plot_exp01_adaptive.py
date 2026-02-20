#!/usr/bin/env python3
"""
Plot Experiment 1 — Adaptive pairing only (2-TDVP vs BUG2 adaptive).

Focused figure showing BUG's speed advantage at moderate accuracy cost.

Layout: 2 rows (TFIM, XXZ) x 2 columns:
  Col 1: Infidelity vs dt       (convergence)
  Col 2: Infidelity vs Runtime   (efficiency, with speedup annotations)

Usage:
    python benchmarks/plot_exp01_adaptive.py [--T 5.0]
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
                            "results", "exp01_order")
FIGURES_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

MODELS = ["tfim", "xxz"]
MODEL_TITLES = {"tfim": "TFIM", "xxz": "XXZ"}
MACHINE_EPS = 1e-16

STYLE = {
    "2TDVP":         dict(color="#2166ac", marker="o", ls="-",  lw=1.6,
                          ms=5.5, label="2-TDVP", zorder=4),
    "BUG2_adaptive": dict(color="#d6604d", marker="s", ls="--", lw=1.6,
                          ms=5.5, label="BUG2",   zorder=4),
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
    df["_source"] = "original"

    rerun_path = os.path.join(base_dir, "run_manifest_rerun.csv")
    if os.path.isfile(rerun_path):
        df_rr = pd.read_csv(rerun_path)
        df_rr["_source"] = "rerun"
        df = pd.concat([df, df_rr], ignore_index=True)

    df = df[df["method"].isin(["2TDVP", "BUG2_adaptive"])]
    df = df.drop_duplicates(subset=["method", "dt"], keep="last")
    df["err_infidelity_T"] = df["err_infidelity_T"].abs().clip(lower=MACHINE_EPS)
    return df

# ---------------------------------------------------------------------------
# Slope triangle
# ---------------------------------------------------------------------------

def slope_triangle(ax, x_c, y_c, slope=2, w=0.25, color="0.45", label=None):
    h = w / 2.0
    x0, x1 = x_c * 10**(-h), x_c * 10**(h)
    y0, y1 = y_c * 10**(-slope * h), y_c * 10**(slope * h)
    ax.fill([x0, x1, x1, x0], [y0, y0, y1, y0],
            fc=color, alpha=0.12, ec=color, lw=0.8, zorder=1)
    if label:
        ax.text(x1 * 1.15, np.sqrt(y0 * y1), label,
                fontsize=6.5, color=color, va="center", ha="left")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=5.0)
    args = ap.parse_args()
    T = args.T
    os.makedirs(FIGURES_DIR, exist_ok=True)

    data = {}
    for m in MODELS:
        data[m] = load_adaptive(m, T)
        print(f"  {m} T={T}: {len(data[m])} adaptive rows")
    if all(d.empty for d in data.values()):
        sys.exit("No data.")

    # Publication style
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    fig, axes = plt.subplots(2, 2, figsize=(5.8, 5.0))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.12,
                        hspace=0.42, wspace=0.38)
    tags = [["(a)", "(b)"], ["(c)", "(d)"]]

    for ri, model in enumerate(MODELS):
        df = data[model]
        ax_rt, ax_eff = axes[ri]

        if df.empty:
            for ax in (ax_rt, ax_eff):
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="0.6")
            continue

        # ── Panel left: Relative runtime bars (BUG = 100%) ────────────
        dt_vals = sorted(df["dt"].unique())
        x_pos = np.arange(len(dt_vals))
        bar_w = 0.35

        tdvp_pct, bug_pct = [], []
        for dt_val in dt_vals:
            t_row = df[(df["method"] == "2TDVP") & (df["dt"] == dt_val)]
            b_row = df[(df["method"] == "BUG2_adaptive") & (df["dt"] == dt_val)]
            tt = t_row["runtime_seconds"].values[0] if not t_row.empty else 0
            bt = b_row["runtime_seconds"].values[0] if not b_row.empty else 0
            bug_pct.append(100.0)
            tdvp_pct.append(tt / bt * 100.0 if bt > 0 else 0)

        bars_t = ax_rt.bar(x_pos - bar_w/2, tdvp_pct, bar_w,
                           color=STYLE["2TDVP"]["color"], alpha=0.85,
                           label="2-TDVP", zorder=3)
        bars_b = ax_rt.bar(x_pos + bar_w/2, bug_pct, bar_w,
                           color=STYLE["BUG2_adaptive"]["color"], alpha=0.85,
                           label="BUG2", zorder=3)

        # Label TDVP bars with their percentage
        for i, pct in enumerate(tdvp_pct):
            txt = ax_rt.text(x_pos[i] - bar_w/2, pct + 2, f"{pct:.0f}%",
                             fontsize=7, fontweight="bold", color="0.20",
                             ha="center", va="bottom")
            txt.set_path_effects([
                pe.withStroke(linewidth=2, foreground="white")])

        # 100% reference line
        ax_rt.axhline(100, color="0.5", ls="--", lw=0.7, zorder=1)

        ax_rt.set_xticks(x_pos)
        ax_rt.set_xticklabels([f"{v:g}" for v in dt_vals], fontsize=6.5)
        ax_rt.set_xlabel(r"$\Delta t$")
        ax_rt.set_ylabel("Relative runtime (%)")
        ax_rt.set_ylim(0, max(tdvp_pct) * 1.20)
        ax_rt.grid(True, which="major", axis="y", ls=":", lw=0.4, alpha=0.5)

        # ── Panel right: Infidelity vs Runtime (Pareto arrows) ───────
        for meth, sty in STYLE.items():
            sub = df[df["method"] == meth].sort_values("runtime_seconds")
            if sub.empty:
                continue
            ax_eff.plot(sub["runtime_seconds"], sub["err_infidelity_T"], **sty)

        ax_eff.set_xscale("log"); ax_eff.set_yscale("log")
        ax_eff.set_xlabel("Wall-clock time (s)"); ax_eff.set_ylabel("Infidelity")
        ax_eff.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)

        tdvp_rows = df[df["method"] == "2TDVP"].sort_values("dt")
        bug_rows  = df[df["method"] == "BUG2_adaptive"].sort_values("dt")
        for _, tr in tdvp_rows.iterrows():
            br = bug_rows[bug_rows["dt"] == tr["dt"]]
            if br.empty:
                continue
            br = br.iloc[0]
            rt_t, rt_b = tr["runtime_seconds"], br["runtime_seconds"]
            er_t, er_b = tr["err_infidelity_T"], br["err_infidelity_T"]
            speedup = rt_t / rt_b

            ax_eff.annotate(
                "", xy=(rt_b, er_b), xytext=(rt_t, er_t),
                arrowprops=dict(arrowstyle="->", color="0.55",
                                lw=0.8, shrinkA=4, shrinkB=4),
                zorder=2)
            mx = np.sqrt(rt_t * rt_b)
            my = np.sqrt(er_t * er_b)
            txt = ax_eff.text(
                mx, my, f"{speedup:.1f}x",
                fontsize=7, fontweight="bold", color="0.25",
                ha="center", va="center", zorder=5)
            txt.set_path_effects([
                pe.withStroke(linewidth=2.5, foreground="white")])

        # ── Panel labels & model title ───────────────────────────────
        for ci, ax in enumerate((ax_rt, ax_eff)):
            ax.text(0.04, 0.96, tags[ri][ci], transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="top", ha="left")
        ax_eff.text(0.96, 0.04, MODEL_TITLES[model],
                    transform=ax_eff.transAxes, fontsize=11, fontweight="bold",
                    va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="0.7", alpha=0.85))

        ax_eff.text(0.50, 0.96, r"$N\!\times$ = speedup",
                    transform=ax_eff.transAxes, fontsize=6.5,
                    color="0.40", va="top", ha="center", fontstyle="italic")

    # Column titles
    axes[0, 0].set_title("Runtime comparison (BUG2 = 100%)", fontsize=10, pad=8)
    axes[0, 1].set_title("Accuracy vs. cost", fontsize=10, pad=8)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   frameon=True, fancybox=False, edgecolor="0.7",
                   fontsize=9, bbox_to_anchor=(0.5, 0.005))

    # Save
    stem = f"exp01_adaptive_T{T}"
    for ext in ("pdf", "png"):
        out = os.path.join(FIGURES_DIR, f"{stem}.{ext}")
        fig.savefig(out)
        print(f"  Saved {out}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
