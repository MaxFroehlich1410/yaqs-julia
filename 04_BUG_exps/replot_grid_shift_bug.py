#!/usr/bin/env python3

"""
Replot all `grid_results/**/timeseries.csv` similarly to `04_BUG_exps/exp_runs.jl`,
but shift BUG expectation values left by one timestep:

- keep the initial value at t=0
- then drop BUG's first evolved sample (t=dt) and align BUG[t=2*dt] at t=dt, etc.
- last time point is dropped for BUG curves (no data after shifting)

Outputs one additional PNG per folder:
  `runs_comparison_bug_shift_left.png`

This script uses system python + matplotlib (no PythonCall / CondaPkg).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BUG_METHODS = {
    "FIXED",
    "ADAPTIVE",
    "DOUBLEFIXED",
    "DOUBLEADAPTIVE",
    "HYBRID",
}


def parse_meta(meta_path: Path) -> Tuple[Dict[str, str], Dict[str, float]]:
    kv: Dict[str, str] = {}
    runtimes: Dict[str, float] = {}
    in_rt = False
    for raw in meta_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "[runtimes_s]":
            in_rt = True
            continue
        if in_rt:
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    runtimes[k] = float(v)
                except ValueError:
                    pass
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k] = v
    return kv, runtimes


def read_timeseries(csv_path: Path) -> Tuple[List[str], np.ndarray]:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader]
    data = np.asarray(rows, dtype=float)
    return header, data


def shift_bug_left_one_step(t: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (t_shift, z_shift) where:
      z_shift[0] = z[0]
      z_shift[k] = z[k+1] for k>=1   (i.e. drop z[1])
    and the last time point is dropped accordingly.
    """
    if len(z) < 3:
        # Nothing sensible to do; fall back to original.
        return t, z
    z_shift = np.concatenate([z[:1], z[2:]])  # length N-1
    t_shift = t[: len(z_shift)]
    return t_shift, z_shift


def label_with_time(method: str, runtimes: Dict[str, float]) -> str:
    if method in runtimes:
        return f"{method} {runtimes[method]:.2f} s"
    return method


def plot_folder(folder: Path) -> None:
    csv_path = folder / "timeseries.csv"
    meta_path = folder / "meta.txt"
    if not csv_path.exists() or not meta_path.exists():
        return

    meta, runtimes = parse_meta(meta_path)
    header, data = read_timeseries(csv_path)
    col = {name: i for i, name in enumerate(header)}

    if "t" not in col:
        return

    t = data[:, col["t"]]
    has_ref = "z_ref_qutip" in col
    z_ref = data[:, col["z_ref_qutip"]] if has_ref else None

    # Titles: mirror exp_runs.jl as closely as possible.
    model = meta.get("model", "")
    tag = meta.get("tag", "")
    L = meta.get("L", "")
    dt = meta.get("dt", "")
    steps = meta.get("steps", "")
    site = meta.get("site", "")
    fixed_pad = meta.get("fixed_max_bond_dim", meta.get("max_bond_dim", ""))
    adaptive_pad = meta.get("adaptive_pad", "")
    chi_max = meta.get("max_bond_dim", "")
    thr = meta.get("threshold", "")

    tagstr = f"  tag={tag}" if tag else ""
    qutip_label = (
        f"exact (qutip) {runtimes.get('qutip', float('nan')):.2f} s" if has_ref and "qutip" in runtimes else "exact (qutip)"
    )

    # Fixed / adaptive method groups (same as exp_runs.jl defaults)
    fixed_methods = ["SINGLE_SITE_TDVP", "DOUBLEFIXED"]
    adaptive_methods = ["TWO_SITE_TDVP", "DOUBLEADAPTIVE"]

    nrows = 4 if has_ref else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 14 if nrows == 4 else 11))

    # 1) <Z> fixed family
    ax = axes[0]
    if has_ref:
        ax.plot(t, z_ref, "-", color="black", linewidth=2.0, alpha=0.85, label=qutip_label)
    for m in fixed_methods:
        if m not in col:
            continue
        z = data[:, col[m]]
        if m in BUG_METHODS:
            tt, zz = shift_bug_left_one_step(t, z)
            ax.plot(tt, zz, linewidth=1.5, alpha=0.8, label=label_with_time(m, runtimes) + " (shift-left 1)")
        else:
            ax.plot(t, z, linewidth=1.5, alpha=0.8, label=label_with_time(m, runtimes))
    ax.set_xlabel("Time")
    ax.set_ylabel("⟨Z⟩")
    ax.set_title(
        f"⟨Z⟩ vs time (fixed family: 1TDVP + BUG2nd fixed)  (site={site})  model={model}{tagstr}  L={L}, dt={dt}, steps={steps}, χpad={fixed_pad}"
    )
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")

    # 2) <Z> adaptive family
    ax2 = axes[1]
    if has_ref:
        ax2.plot(t, z_ref, "-", color="black", linewidth=2.0, alpha=0.85, label=qutip_label)
    for m in adaptive_methods:
        if m not in col:
            continue
        z = data[:, col[m]]
        if m in BUG_METHODS:
            tt, zz = shift_bug_left_one_step(t, z)
            ax2.plot(tt, zz, linewidth=1.5, alpha=0.8, label=label_with_time(m, runtimes) + " (shift-left 1)")
        else:
            ax2.plot(t, z, linewidth=1.5, alpha=0.8, label=label_with_time(m, runtimes))
    ax2.set_xlabel("Time")
    ax2.set_ylabel("⟨Z⟩")
    ax2.set_title(
        f"⟨Z⟩ vs time (adaptive family: 2TDVP + BUG2nd adaptive)  (site={site})  model={model}{tagstr}  L={L}, dt={dt}, steps={steps}, χpad={adaptive_pad}"
    )
    ax2.grid(True, alpha=0.3)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(loc="best")

    # 3) Bond dimension growth subplot cannot be reconstructed from timeseries.csv alone.
    ax_bd = axes[2]
    ax_bd.axis("off")
    ax_bd.text(
        0.5,
        0.5,
        "Bond dimension growth not available:\n(timeseries.csv does not store χ(t))",
        ha="center",
        va="center",
        fontsize=12,
    )

    if has_ref:
        # 4) Squared error vs exact (qutip)
        ax3 = axes[3]
        for name in header:
            if name in ("t", "z_ref_qutip"):
                continue
            z = data[:, col[name]]
            if name in BUG_METHODS:
                tt, zz = shift_bug_left_one_step(t, z)
                err_sq = (zz - z_ref[: len(zz)]) ** 2
                ax3.plot(tt, err_sq, linewidth=1.5, alpha=0.8, label=label_with_time(name, runtimes) + " (shift-left 1)")
            else:
                err_sq = (z - z_ref) ** 2
                ax3.plot(t, err_sq, linewidth=1.5, alpha=0.8, label=label_with_time(name, runtimes))
        ax3.set_xlabel("Time")
        ax3.set_ylabel("|Δ⟨Z⟩|²")
        ax3.set_title("Squared error vs exact (qutip): all methods  (BUG shifted left by 1 step)")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.3)
        if ax3.get_legend_handles_labels()[0]:
            ax3.legend(loc="best")

    fig.tight_layout()
    outpng = folder / "runs_comparison_bug_shift_left.png"
    fig.savefig(outpng, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parent / "grid_results"
    if not base.exists():
        raise SystemExit(f"grid_results not found at {base}")

    folders = sorted({p.parent for p in base.rglob("timeseries.csv")})
    print(f"[replot] found {len(folders)} folders")
    for folder in folders:
        plot_folder(folder)
    print("[replot] done")


if __name__ == "__main__":
    main()

