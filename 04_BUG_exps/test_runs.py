#!/usr/bin/env python3
"""Test and compare BUG algorithm with qutip time evolution.

This script demonstrates the Basis-Update Galerkin (BUG) method for time evolution
of quantum states on a small spin chain and compares the results with exact qutip
time evolution.
"""

from __future__ import annotations

from copy import deepcopy
import logging
import re
import time
import argparse
from typing import Self, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from h5py import File

from .ising_sim_util import (BUGKind, QUTIP_AVAILABLE,
                            qutip_time_evolution,
                            run_bug_time_evolution,
                            QUTIP_RESULTS_FILE_NAME,
                            TIMES_KEY,
                            ISING_J_DEFAULT,
                            ISING_G_DEFAULT)
from .results_class import SimResults as TestResults

# -----------------------------------------------------------------------------
# Quick in-code toggles (used when you DON'T pass --methods on the command line)
# -----------------------------------------------------------------------------
RUN_QUTIP = True

# Toggle methods here:
METHOD_ENABLED: dict[str, bool] = {
    "ADAPTIVE": True,
    "FIXED": False,
    "DOUBLEADAPTIVE": True,
    "DOUBLEFIXED": False,
    "HYBRID": True,
    "SINGLE_SITE_TDVP": False,
    "TWO_SITE_TDVP": True,
}

def plot_comparison(
    results: TestResults,
    num_sites: int,
    params: dict[str, Any],
    runtimes_s: dict[str, float] | None = None,
) -> None:
    """Plot comparison between BUG methods and qutip results.

    Args:
        results: TestResults object containing all results.
        num_sites: Number of sites.
        params: Dictionary of simulation parameters for title.
    """
    # Select a few representative sites to plot (avoid overcrowding)
    # sites_to_plot = [0, num_sites // 2, num_sites -
    #                  1] if num_sites > 3 else list(range(num_sites))
    sites_to_plot = [num_sites // 2]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    # Define colors for each method
    color_exact = 'black'

    # Plot all sites
    ax = axes[0]

    times = results.times

    for idx, site in enumerate(sites_to_plot):
        # Determine if this is the first site (for legend)
        show_label = (idx == 0)
        if not show_label:
            ignored_keys = {"label"}
        else:
            ignored_keys = set()

        # Qutip exact
        if results.has_qutip_results():
            qutip_results = results.qutip_results[site]
            ax.plot(times, qutip_results, '-',
                    color=color_exact,
                    label='Exact (qutip)' if show_label else '',
                    linewidth=2.0, alpha=0.8)

        for bug_kind in BUGKind:
            if results.has_bug_results(bug_kind):
                bug_results = results.obtain_res_by_kind(bug_kind)[site]
                kwargs = bug_kind.plot_kwargs(ignore=ignored_keys)
                ax.plot(times, bug_results, **kwargs,
                        markersize=3, alpha=0.7, linewidth=1.5)

    # Add text annotation for which sites are shown
    ax.text(0.02, 0.98, f'Sites: {sites_to_plot}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('⟨Z⟩', fontsize=12)
    title = f'Time Evolution: Ising Model ('''
    for param, value in params.items():
        title += f'{param}={value}, '
    title += f'{num_sites} sites)'
    ax.set_title(title,
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot difference if qutip available
    if results.has_qutip_results():
        ax = axes[1]
        errors = results.make_error_results(absolute=True)

        for idx, site in enumerate(sites_to_plot):
            show_label = (idx == 0)
            if not show_label:
                ignored_keys = {"label"}
            else:
                ignored_keys = set()
            
            for bug_kind in BUGKind:
                if errors.has_bug_results(bug_kind):
                    error_results = errors.obtain_res_by_kind(bug_kind)[site]
                    kwargs = bug_kind.plot_kwargs(ignore=ignored_keys)
                    ax.plot(times, error_results, **kwargs,
                            markersize=3, alpha=0.7, linewidth=1.5)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Δ⟨Z⟩ (BUG - exact)', fontsize=12)
        ax.set_title('Error: BUG vs exact', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'qutip not available for comparison',
                     ha='center', va='center', fontsize=14,
                     transform=axes[1].transAxes)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    # Runtime comparison (relative to ADAPTIVE)
    ax = axes[2]
    if runtimes_s is None or "ADAPTIVE" not in runtimes_s:
        ax.text(0.5, 0.5, 'runtime data not available',
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ref = runtimes_s["ADAPTIVE"]
        # Keep stable order: methods only (qutip is for accuracy reference, not part of runtime subplot)
        labels = [k.name for k in BUGKind if k.name in runtimes_s]
        rel = [runtimes_s[l] / ref for l in labels]
        bars = ax.bar(labels, rel, color="gray", alpha=0.8)
        ax.axhline(1.0, color="tab:blue", linestyle="--", linewidth=1.5, label="ADAPTIVE = 1.0")
        ax.set_ylabel("Runtime / ADAPTIVE", fontsize=12)
        ax.set_title("Runtime comparison (relative to ADAPTIVE)", fontsize=14, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=35)
        ax.legend(loc="best", fontsize=10)
        ax.set_ylim(0, max(rel) * 1.15 if rel else 1.0)

        # Annotate bars with factor vs ADAPTIVE (inside bars)
        for bar, factor in zip(bars, rel, strict=False):
            x = bar.get_x() + bar.get_width() / 2
            h = bar.get_height()
            y = h / 2.0
            # White text reads better on medium/tall gray bars
            color = "white" if h >= 0.6 else "black"
            ax.text(x, y, f"{factor:.2f}×", ha="center", va="center", fontsize=9, color=color)

    plt.tight_layout()
    plt.show()


def print_summary(
    results: TestResults,
    num_sites: int
) -> None:
    """Print summary statistics.

    Args:
        results: TestResults object containing all results.
        num_sites: Number of sites.
    """
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nFinal time: {results.final_time():.4f}")
    print(f"Number of time steps: {results.num_time_steps()}")

    # Show only a few representative sites
    sites_to_show = [0, num_sites // 2, num_sites -
                     1] if num_sites > 5 else list(range(num_sites))

    print(f"\nFinal ⟨Z⟩ values (showing sites {sites_to_show}):")
    print("-" * 70)
    for site in sites_to_show:
        line = f"Site {site:2d}: "
        for bug_kind in BUGKind:
            if results.has_bug_results(bug_kind):
                bug_val = results.obtain_res_by_kind(bug_kind)[site][-1]
                line += f"{bug_kind.name} = {bug_val:+.6f}, "
        if results.has_qutip_results():
            qutip_val = results.qutip_results[site][-1]
            line += f"Exact = {qutip_val:+.6f}"
        line = line.rstrip(', ')
        line += "\n"
        print(line)

    print("\n")
    if results.has_qutip_results():
        print(f"\nError statistics vs exact (sites {sites_to_show}):")
        print("-" * 70)
        #print(f"{'Site':>6} │ {'BUG1 Max |Δ|':>14} │ {'BUG1 RMS':>12} │ {'BUG2 Max |Δ|':>14} │ {'BUG2 RMS':>12}")
        print("-" * 70)

        # Compute errors
        max_errors = {}
        rms_errors = {}
        for bug_kind in BUGKind:
            max_errors[bug_kind] = {}
            rms_errors[bug_kind] = {}
            if results.has_bug_results(bug_kind):
                error = results.compute_error(bug_kind, absolute=True)
                for site in range(num_sites):
                    max_errors[bug_kind][site] = np.max(error)
                    rms_errors[bug_kind][site] = np.sqrt(np.mean(np.pow(error, 2)))

        # Print max RMS errors for each BUG kind
        head_line_max_delta = f"{'Site':>6} | "
        for bug_kind in BUGKind:
            if results.has_bug_results(bug_kind):
                head_line_max_delta += f"{bug_kind.name + ' Max |Δ|':>14} | "
        head_line_max_delta = head_line_max_delta.rstrip(' |')
        print(head_line_max_delta)
        for site in sites_to_show:
            line = f"{site:6d} | "
            for bug_lind in BUGKind:
                if results.has_bug_results(bug_kind):
                    max_abs_err = max_errors[bug_kind][site]
                    line += f"{max_abs_err:14.2e} | "
            line = line.rstrip(' |')
            print(line)

        print("-" * 70)
        # Print RMS errors for each BUG kind
        head_line_rms = f"{'Site':>6} | "
        for bug_kind in BUGKind:
            if results.has_bug_results(bug_kind):
                head_line_rms += f"{bug_kind.name + ' RMS':>12} | "
        head_line_rms = head_line_rms.rstrip(' |')
        print(head_line_rms)
        print("-" * 70)
        for site in sites_to_show:
            line = f"{site:6d} | "
            for bug_kind in BUGKind:
                if results.has_bug_results(bug_kind):
                    rms_err = rms_errors[bug_kind][site]
                    line += f"{rms_err:12.2e} | "
            line = line.rstrip(' |')
            print(line)

        print("-" * 70)

        # Compute overall statistics
        full_rms = results.full_rms()

        print("-" * 70)
        line = f"{'Mean':>6}"
        for bug_kind in BUGKind:
            if bug_kind in full_rms:
                mean_rms = full_rms[bug_kind]
                line += f" │ {mean_rms:12.2e}"
        line += "\n"
        print(line)

    print("="*70 + "\n")

def main():
    """Main function to run the comparison."""
    print("="*70)
    print("BUG Algorithm Comparison: Test")
    print("="*70)
    print()

    cli = argparse.ArgumentParser(description="Compare BUG + TDVP methods against qutip for Ising time evolution.")
    cli.add_argument(
        "--methods",
        nargs="*",
        default=[],
        choices=[k.name for k in BUGKind],
        help="Subset of methods to run (default: all). Example: --methods ADAPTIVE HYBRID TWO_SITE_TDVP",
    )
    cli.add_argument("--no-qutip", action="store_true", help="Skip qutip exact evolution/comparison.")
    cli.add_argument("--num-sites", type=int, default=10)
    cli.add_argument("--J", type=float, default=ISING_J_DEFAULT)
    cli.add_argument("--g", type=float, default=ISING_G_DEFAULT)
    cli.add_argument("--elapsed-time", type=float, default=40.0)
    cli.add_argument("--dt", type=float, default=0.1)
    cli.add_argument("--initial-state", type=str, default="zeros")
    cli.add_argument("--max-bond-dim", type=int, default=128)
    cli.add_argument("--threshold", type=float, default=1e-10)

    args = cli.parse_args()

    # Parameters
    num_sites = args.num_sites
    J = args.J
    g = args.g
    elapsed_time = args.elapsed_time
    dt = args.dt
    initial_state = args.initial_state
    max_bond_dim = args.max_bond_dim
    threshold = args.threshold

    print(f"System parameters:")
    print(f"  Number of sites: {num_sites}")
    print(f"  Coupling J: {J}")
    print(f"  Transverse field g: {g}")
    print(f"  Initial state: |{initial_state}⟩")
    print(f"  Total time: {elapsed_time}")
    print(f"  Time step dt: {dt}")
    print(f"  Max bond dimension: {max_bond_dim}")
    print(f"  Truncation threshold: {threshold}")
    print()

    results = TestResults()
    runtimes_s: dict[str, float] = {}

    # Choose methods
    if args.methods:
        methods = [BUGKind[m] for m in args.methods]
    else:
        # Use the in-code toggles
        methods = [k for k in BUGKind if METHOD_ENABLED.get(k.name, False)]
        if not methods:
            raise ValueError("No methods enabled. Set at least one entry in METHOD_ENABLED to True.")

    # Run evolutions
    for bug_kind in methods:
        print(f"Running {bug_kind.name} BUG time evolution...")
        print("-" * 70)
        t0 = time.perf_counter()
        times_bug, expect_bug = run_bug_time_evolution(
            num_sites=num_sites,
            J=J,
            g=g,
            elapsed_time=elapsed_time,
            dt=dt,
            initial_state=initial_state,
            max_bond_dim=max_bond_dim,
            threshold=threshold,
            bug_kind=bug_kind,
            verbose=False
        )
        runtimes_s[bug_kind.name] = time.perf_counter() - t0
        if results.times.size == 0:
            results.times = times_bug
        else:
            if not np.allclose(results.times, times_bug):
                errstr = "Time arrays do not match between different BUG runs!"
                raise ValueError(errstr)
        results.set_res_by_kind(bug_kind, expect_bug)
        print(f"✓ {bug_kind.name} BUG evolution complete!  (runtime: {runtimes_s[bug_kind.name]:.3f}s)")

    # Run qutip evolution if available (unless disabled)
    times_qutip = None
    expect_qutip = None
    want_qutip = (not args.no_qutip) and RUN_QUTIP
    if want_qutip and QUTIP_AVAILABLE:
        print("Running qutip exact evolution...")
        print("-" * 70)
        t0 = time.perf_counter()
        times_qutip, expect_qutip = qutip_time_evolution(
            num_sites=num_sites,
            J=J,
            g=g,
            times=results.times,  # Use same time points as BUG
            initial_state=initial_state
        )
        runtimes_s["qutip"] = time.perf_counter() - t0
        results.qutip_results = expect_qutip
        if not np.allclose(results.times, times_qutip):
            errstr = "Time arrays do not match between BUG and qutip results!"
            raise ValueError(errstr)
        print(f"✓ qutip exact evolution complete!  (runtime: {runtimes_s['qutip']:.3f}s)")
        print()
    else:
        if args.no_qutip or (not RUN_QUTIP):
            print("⚠ Skipping qutip comparison (--no-qutip or RUN_QUTIP=False)")
            print()
        else:
            print("⚠ Skipping qutip comparison (not installed)")
            print("  To install: pip install qutip")
            print()

    # Print summary
    # print_summary(results, num_sites)

    # Plot results
    print("Generating plots...")
    params = {
        "J": J,
        "g": g
    }
    # Runtime printout (relative to ADAPTIVE)
    if "ADAPTIVE" in runtimes_s:
        ref = runtimes_s["ADAPTIVE"]
        print("=" * 70)
        print("RUNTIME COMPARISON (relative to ADAPTIVE)")
        print("=" * 70)
        print(f"{'Method':<22} {'Runtime [s]':>12} {'/ ADAPTIVE':>12}")
        for k in [bk.name for bk in BUGKind]:
            if k in runtimes_s:
                print(f"{k:<22} {runtimes_s[k]:>12.3f} {runtimes_s[k] / ref:>12.3f}")
        if "qutip" in runtimes_s:
            print(f"{'qutip':<22} {runtimes_s['qutip']:>12.3f} {runtimes_s['qutip'] / ref:>12.3f}")
        print("=" * 70)

    plot_comparison(results, num_sites, params, runtimes_s=runtimes_s)
    print("✓ Done!")


if __name__ == "__main__":
    main()
