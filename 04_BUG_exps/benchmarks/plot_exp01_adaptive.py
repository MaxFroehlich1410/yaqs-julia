#!/usr/bin/env python3
"""
Plot Experiment 1 adaptive summary (TFIM + XXZ).

Layout: 2 rows x 2 columns:
  Row 1 (TFIM): (a) relative runtime bars, (b) accuracy vs cost
  Row 2 (XXZ):  (c) relative runtime bars, (d) accuracy vs cost

Usage:
    python benchmarks/plot_exp01_adaptive.py [--T 1.0] [--N 5] [--run <timestamp_dir>]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = os.path.join(BENCH_DIR, "results")
FIGURES_DIR = os.path.join(BENCH_DIR, "figures")
MACHINE_EPS = 1e-16

STYLE = {
    "2TDVP":         dict(color="#2166ac", marker="o", ls="-",  lw=1.6, ms=6.0, label="2-TDVP"),
    "BUG2_adaptive": dict(color="#d6604d", marker="s", ls="--", lw=1.6, ms=6.0, label="BUG2"),
}


def _resolve_exp_root(exp_prefix: str, n_sites: int | None, run_label: str | None) -> str | None:
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
        run_dirs.sort(reverse=True)

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
        exp_dirs.sort(reverse=True)
        return os.path.join(run_dir, exp_dirs[0])
    return None


def load_manifest(model: str, T: float, n_sites: int | None, run_label: str | None) -> pd.DataFrame:
    exp_root = _resolve_exp_root("exp01_order", n_sites, run_label)
    if exp_root is None:
        return pd.DataFrame()

    base_dir = os.path.join(exp_root, model, f"T_{T}")
    path = os.path.join(base_dir, "run_manifest.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df[df["method"].isin(["2TDVP", "BUG2_adaptive"])].copy()
    df["err_infidelity_T"] = df["err_infidelity_T"].abs().clip(lower=MACHINE_EPS)
    df["dt"] = pd.to_numeric(df["dt"], errors="coerce")
    df["runtime_seconds"] = pd.to_numeric(df["runtime_seconds"], errors="coerce")
    df = df.dropna(subset=["dt", "runtime_seconds", "err_infidelity_T"])
    df = df.drop_duplicates(subset=["method", "dt"], keep="last")
    return df


def _shared_points(df: pd.DataFrame) -> pd.DataFrame:
    tdvp = df[df["method"] == "2TDVP"][["dt", "runtime_seconds", "err_infidelity_T"]].copy()
    bug2 = df[df["method"] == "BUG2_adaptive"][["dt", "runtime_seconds", "err_infidelity_T"]].copy()
    tdvp = tdvp.rename(columns={"runtime_seconds": "rt_tdvp", "err_infidelity_T": "err_tdvp"})
    bug2 = bug2.rename(columns={"runtime_seconds": "rt_bug2", "err_infidelity_T": "err_bug2"})
    merged = pd.merge(tdvp, bug2, on="dt", how="inner")
    return merged.sort_values("dt")


def _plot_runtime_bars(ax, df: pd.DataFrame, model_label: str, tag: str):
    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="0.5")
        return

    shared = _shared_points(df)
    if shared.empty:
        ax.text(0.5, 0.5, "No matching dt pairs", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="0.5")
        return

    dt_vals = shared["dt"].to_numpy()
    rel_tdvp = 100.0 * shared["rt_tdvp"].to_numpy() / np.maximum(shared["rt_bug2"].to_numpy(), 1e-12)
    rel_bug2 = np.full_like(rel_tdvp, 100.0)

    x = np.arange(len(dt_vals))
    w = 0.34
    ax.bar(x - w / 2, rel_tdvp, width=w, color=STYLE["2TDVP"]["color"], alpha=0.85, label="2-TDVP")
    ax.bar(x + w / 2, rel_bug2, width=w, color=STYLE["BUG2_adaptive"]["color"], alpha=0.85, label="BUG2")
    ax.axhline(100.0, color="0.45", ls="--", lw=0.9, alpha=0.7)

    for i, y in enumerate(rel_tdvp):
        ax.text(x[i] - w / 2, y + 4.0, f"{int(round(y))}%",
                ha="center", va="bottom", fontsize=8.5, color="0.2", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{dt:g}" for dt in dt_vals])
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("Relative runtime (%)")
    ax.grid(True, axis="y", which="major", ls=":", lw=0.4, alpha=0.5)
    ax.set_ylim(0, max(180, rel_tdvp.max() * 1.20))
    ax.text(0.04, 0.96, tag, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")
    ax.text(0.98, 0.03, model_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", alpha=0.9))


def _plot_accuracy_cost(ax, df: pd.DataFrame, model_label: str, tag: str):
    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="0.5")
        return

    for method, sty in STYLE.items():
        sub = df[df["method"] == method].sort_values("runtime_seconds")
        if sub.empty:
            continue
        ax.plot(sub["runtime_seconds"], sub["err_infidelity_T"], **sty)

    shared = _shared_points(df)
    for _, row in shared.iterrows():
        speedup = row["rt_tdvp"] / max(row["rt_bug2"], 1e-12)
        y_txt = (row["err_tdvp"] * row["err_bug2"]) ** 0.5
        ax.annotate(
            f"{speedup:.1f}x",
            xy=(row["rt_bug2"], row["err_bug2"]),
            xytext=(row["rt_tdvp"], y_txt),
            fontsize=10,
            color="0.25",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9, alpha=0.8),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Infidelity")
    ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)
    ax.text(0.52, 0.97, r"$N\times$ = speedup", transform=ax.transAxes,
            fontsize=10, color="0.4", va="top", ha="center", style="italic")
    ax.text(0.04, 0.96, tag, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")
    ax.text(0.98, 0.03, model_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", alpha=0.9))


def _plot_row(ax_bar, ax_scatter, df: pd.DataFrame, model_label: str, tags: tuple[str, str]):
    if df.empty:
        for ax in (ax_bar, ax_scatter):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="0.5")
        return

    _plot_runtime_bars(ax_bar, df, model_label, tags[0])
    _plot_accuracy_cost(ax_scatter, df, model_label, tags[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--run", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    df_tfim = load_manifest("tfim", args.T, args.N, args.run)
    df_xxz = load_manifest("xxz", args.T, args.N, args.run)
    print(f"  tfim T={args.T}: {len(df_tfim)} adaptive rows")
    print(f"  xxz T={args.T}: {len(df_xxz)} adaptive rows")
    if df_tfim.empty and df_xxz.empty:
        sys.exit("No TFIM/XXZ adaptive data found.")

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.55))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.10,
                        hspace=0.43, wspace=0.35)
    _plot_row(axes[0, 0], axes[0, 1], df_tfim, "TFIM", ("(a)", "(b)"))
    _plot_row(axes[1, 0], axes[1, 1], df_xxz, "XXZ", ("(c)", "(d)"))
    axes[0, 0].set_title("Runtime comparison (BUG2 = 100%)", fontsize=14, pad=10)
    axes[0, 1].set_title("Accuracy vs. cost", fontsize=14, pad=10)

    handles = [
        plt.Line2D([0], [0], color=STYLE["2TDVP"]["color"], lw=8, alpha=0.85),
        plt.Line2D([0], [0], color=STYLE["BUG2_adaptive"]["color"], lw=8, alpha=0.85),
    ]
    fig.legend(handles, ["2-TDVP", "BUG2"], loc="lower center", ncol=2,
               framealpha=0.95, edgecolor="0.75", bbox_to_anchor=(0.5, -0.01), fontsize=12)

    stem = f"exp01_adaptive_T{args.T}"
    for ext in ("pdf", "png"):
        out = os.path.join(FIGURES_DIR, f"{stem}.{ext}")
        fig.savefig(out)
        print(f"  Saved {out}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
