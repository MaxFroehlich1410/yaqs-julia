"""
Utility functions for running and evaluating BUG simulations.
"""
from __future__ import annotations
from typing import Any
import os
import argparse
from copy import copy

import numpy as np
import matplotlib.pyplot as plt

from .ising_sim_util import (ISING_G_DEFAULT,
                            ISING_J_DEFAULT,
                            run_bug_time_evolution,
                            BUGKind,
                            qutip_time_evolution,
                            QUTIP_RESULTS_FILE_NAME,
                            QUTIP_AVAILABLE)
from .results_class import SimResults

DEFAULT_PAR_DICT = {
    "max_bond_dim": 200,
    "threshold": 1e-15,
    "num_sites": 20,
    "J": ISING_J_DEFAULT,
    "g": ISING_G_DEFAULT,
    "elapsed_time": 20.0,
    "dt": 0.1,
    "initial_state": "zeros"
}

def bug_filename_creation(par_name: str,
                          par_value: Any,
                          data_folder: str) -> str:
    """
    Create a filename for storing results based on the maximum bond dimension.
    """
    filename = f"ising_{par_name}_{par_value}_results.h5"
    return os.path.join(data_folder, filename)

def run_bug_simulations(par_pair: tuple[str, Any],
                        data_folder: str,
                        bug_kinds: list[BUGKind]):
    """
    Runs BUG simulations for different BUG kinds with the specified maximum bond dimension.
    """
    # Parameters
    par_name, par_value = par_pair
    par_dict = copy(DEFAULT_PAR_DICT)
    par_dict[par_name] = par_value

    results = SimResults()
    for key, value in par_dict.items():
        results.metadata[key] = str(value)

    # Run BUG evolutions
    total_methods = len(bug_kinds)
    for method_idx, bug_kind in enumerate(bug_kinds, start=1):
        print(f"[{par_name}={par_value}] ({method_idx}/{total_methods}) Running {bug_kind.name} ...", flush=True)
        times_bug, expect_bug = run_bug_time_evolution(
            bug_kind=bug_kind,
            verbose=False,
            **par_dict
        )
        if results.times.size == 0:
            results.times = times_bug
        else:
            if not np.allclose(results.times, times_bug):
                errstr = "Time arrays do not match between different BUG runs!"
                raise ValueError(errstr)
        results.set_res_by_kind(bug_kind, expect_bug)

    # Save results
    filename = bug_filename_creation(par_pair[0], par_pair[1], data_folder)
    results.save_to_file(filename)
    print(f"[{par_name}={par_value}] Saved results to {filename}", flush=True)


def run_and_save_qutip_simulation(data_folder: str):
    """
    Runs and saves the qutip simulation results for comparison.
    """
    if not QUTIP_AVAILABLE:
        print("⚠ Skipping qutip simulation (not installed)")
        print("  To install: pip install qutip")
        return

    # Parameters
    par_dict = copy(DEFAULT_PAR_DICT)
    par_dict.pop("max_bond_dim")  # Not needed for qutip
    par_dict.pop("threshold")     # Not needed for qutip
    results = SimResults()
    for key, value in par_dict.items():
        results.metadata[key] = str(value)

    # Run qutip simulation
    print("[qutip] Running exact time evolution for reference ...", flush=True)
    times_qutip, expect_qutip = qutip_time_evolution(
        num_sites=par_dict["num_sites"],
        J=par_dict["J"],
        g=par_dict["g"],
        times=np.arange(0, par_dict["elapsed_time"] + par_dict["dt"], par_dict["dt"]),
        initial_state=par_dict["initial_state"]
    )

    # Save results
    results.times = times_qutip
    results.qutip_results = expect_qutip
    filename = os.path.join(data_folder, QUTIP_RESULTS_FILE_NAME)
    results.save_to_file(filename)
    print(f"[qutip] Saved results to {filename}", flush=True)

def plot_error(data_folder: str,
                 par_name: str,
                 par_values: list[Any],
                 bug_kinds: list[BUGKind],
                 plot_save: str = "",
                 plot_kwargs: dict[str, Any] | None = None,):
    """
    Plot the results from the simulations.
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    print(f"[plot] Loading results from {data_folder} ...", flush=True)
    results: dict[int, SimResults] = {}
    qt_filename = os.path.join(data_folder, QUTIP_RESULTS_FILE_NAME)
    for par_value in par_values:
        filename = bug_filename_creation(par_name, par_value, data_folder)
        print(f"[plot] Loading {filename}", flush=True)
        sim_res = SimResults()
        sim_res.load_from_file(filename)
        sim_res.load_qutip_results(qt_filename)
        results[par_value] = sim_res
    errors: dict[BUGKind, dict[int, float]] = {bug_kind: {} for bug_kind in bug_kinds}
    for par_value, sim_res in results.items():
        err = sim_res.full_rms()
        for bug_kind, error in err.items():
            errors[bug_kind][par_value] = error
    min_errs: dict[BUGKind, dict[int, float]] = {bug_kind: {} for bug_kind in bug_kinds}
    for par_value, sim_res in results.items():
        err = sim_res.minimum_error(exclude_first=2)
        for bug_kind, min_err in err.items():
            min_errs[bug_kind][par_value] = min_err
    max_errs: dict[BUGKind, dict[int, float]] = {bug_kind: {} for bug_kind in bug_kinds}
    for par_value, sim_res in results.items():
        err = sim_res.maximum_error()
        for bug_kind, max_err in err.items():
            max_errs[bug_kind][par_value] = max_err
    # Plotting
    plt.figure(figsize=(10, 6))
    for bug_kind in bug_kinds:
        bd_vals = sorted(errors[bug_kind].keys())
        err_vals = [errors[bug_kind][bd] for bd in bd_vals]
        err_min_vals = [min_errs[bug_kind][bd] for bd in bd_vals]
        err_max_vals = [max_errs[bug_kind][bd] for bd in bd_vals]
        plt.errorbar(bd_vals, err_vals,
                     yerr=[err_min_vals, err_max_vals],
                     capsize=5,
                     **bug_kind.plot_kwargs())
    plt.yscale(plot_kwargs.get("yscale", "log"))
    plt.xscale(plot_kwargs.get("xscale", "linear"))
    plt.ylabel('RMS Error vs Qutip')
    plt.xlabel(plot_kwargs.get("xlabel", "Parameter"))
    plt.legend()
    if plot_save:
        plt.savefig(plot_save)
    else:
        plt.show()
    plt.close()


def main(param_name: str = "max_bond_dim",
         param_dtype: type = int,
         plot_kwargs: dict[str, Any] | None = None,
         bug_kinds: list[BUGKind] | None = None):
    """
    Main function to run simulations and plot results.

    Args:
        param_name: The name of the parameter to vary (default is "max_bond_dim").
    """
    cli = argparse.ArgumentParser(
        description="Run max bond dimension scaling simulations for the transverse field Ising model using BUG."
    )
    cli.add_argument("data_folder",
                     type=str,
                     help="Folder to save simulation results.")
    cli.add_argument(f"--{param_name}s",
                     type=param_dtype,
                     nargs="*",
                     default=[],
                     help=f"List of {param_name}s to simulate.")
    cli.add_argument("--plot_save",
                     type=str,
                     default="",
                     help="Filename to save the plot. If empty, the plot is not saved.")
    cli.add_argument("-run_qutip",
                     action="store_true",
                     help="Whether to run and save the qutip simulation for comparison.")
    cli.add_argument("-plot",
                     action="store_true",
                     help="Whether to plot the results after running simulations.")
    cli.add_argument("-run_bug",
                     action="store_true",
                     help="Whether to run the BUG simulations.")

    args = cli.parse_args()
    data_folder = args.data_folder
    par_values = getattr(args, f"{param_name}s")
    run_qutip = args.run_qutip
    run_bug = args.run_bug
    plot_save = args.plot_save

    if bug_kinds is None:
        bug_kinds =list(BUGKind)

    print("=" * 70)
    print(f"Scaling run: param={param_name} values={par_values}")
    print(f"Data folder: {data_folder}")
    print(f"Methods: {[bk.name for bk in bug_kinds]}")
    print("=" * 70, flush=True)

    # Run and save qutip results
    if run_qutip:
        run_and_save_qutip_simulation(data_folder)

    # Run BUG simulations for each max bond dimension
    if run_bug:
        total_vals = len(par_values)
        for idx, par_val in enumerate(par_values, start=1):
            print("-" * 70)
            print(f"({idx}/{total_vals}) Running simulations for {param_name}={par_val}", flush=True)
            run_bug_simulations((param_name, par_val), data_folder, bug_kinds)

    # Plot results
    if args.plot:
        plot_error(data_folder, param_name, par_values, bug_kinds,
                   plot_save=plot_save, plot_kwargs=plot_kwargs)

if __name__ == "__main__":
    main()
