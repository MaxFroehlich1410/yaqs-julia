#!/usr/bin/env python3
"""
Theoretical variance comparison for 2-qubit bit-flip SPLM.

Simulates a two-qubit system with sparse Pauli-Lindblad noise on {X⊗I, I⊗X, X⊗X}
with equal rates γ, comparing simulation results with theoretical variance formulas
for different unraveling methods.
"""

import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
import pickle
from datetime import datetime

# Import YAQS components
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams, Observable
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs import simulator

from mqt.yaqs.codex_experiments.worker_functions.yaqs_simulator import run_yaqs, build_noise_models
from mqt.yaqs.codex_experiments.worker_functions.qiskit_simulators import run_qiskit_exact, run_qiskit_mps

# Import Qiskit noise components
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel, PauliLindbladError


def build_simple_circuit() -> QuantumCircuit:
    """Build a simple 2-qubit circuit with identity gates that noise can act on."""
    qc = QuantumCircuit(2)
    qc.rxx(0.001*np.pi/2, 0, 1)
    return qc

def theoretical_variance_formulas(
    t_layers: np.ndarray,
    gamma: float,
    *,
    scheme_params: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Exact two-qubit IX/XI/XX identity-layer variance curves for <Z_i>.

    For each qubit i, the total anticommuting rate is r_i = 2γ (from X_i and X⊗X).

    Formulas (see Appendix):
    - Mean (all unravelings): E[<Z_i>_t] = exp(-2*r_i*t) = exp(-4γt)
    - Standard unraveling: Var_std = 1 - exp(-4*r_i*t) = 1 - exp(-8γt)
    - Projector unraveling: Var_proj = exp(-2*r_i*t) * (1 - exp(-2*r_i*t)) = exp(-4γt) * (1 - exp(-4γt))
    - Analog 2-point: Var_2pt = 1/4 + 1/2 e^{-8γ(1-s)t} + 1/4 e^{-16γ(1-s)t} - e^{-8γ t}
    - Analog Gaussian: Var_gauss = 1/4 + 1/2 e^{-2 b t} + 1/4 e^{-4 b t} - e^{-8γ t},
        with b = (γ/s) * (1 - (1 - 2s)^4)/2

    The parameter s is the angle-law second moment s = E_w[sin^2 θ]. If not provided,
    we default to s = 1/3 as a typical analog discretization choice.
    """
    t = np.asarray(t_layers, dtype=float)

    # Parameters for analog laws
    if scheme_params is None:
        s = 1.0 / 3.0
    else:
        s = float(scheme_params.get("s", 1.0 / 3.0))

    # Total anticommuting rate for each qubit: r_i = 2γ
    r_i = 2.0 * gamma

    # Precompute common term for subtraction of mean^2
    mean_sq = np.exp(-8.0 * gamma * t)  # (E[Z])^2 = e^{-8γ t}

    # Standard unraveling: telegraph process under unitary X-jumps
    var_std = 1.0 - np.exp(-4.0 * r_i * t)  # = 1 - exp(-8γt)

    # Projector unraveling: absorbing with rate 2*r_i = 4γ
    var_proj = np.exp(-2.0 * r_i * t) * (1.0 - np.exp(-2.0 * r_i * t))  # = exp(-4γt) * (1 - exp(-4γt))

    # Analog two-point: Var = 1/4 + 1/2 e^{-8γ(1-s)t} + 1/4 e^{-16γ(1-s)t} - e^{-8γ t}
    var_2pt = 0.25 + 0.5 * np.exp(-8.0 * gamma * (1.0 - s) * t) + 0.25 * np.exp(-16.0 * gamma * (1.0 - s) * t) - mean_sq

    # Analog Gaussian: b = (γ/s) * (1 - (1 - 2s)^4)/2
    one_minus_2s = 1.0 - 2.0 * s
    b = (gamma / max(s, 1e-16)) * (1.0 - one_minus_2s ** 4) / 2.0
    var_gauss = 0.25 + 0.5 * np.exp(-2.0 * b * t) + 0.25 * np.exp(-4.0 * b * t) - mean_sq

    return {
        "standard": var_std,
        "projector": var_proj,
        "unitary_2pt": var_2pt,
        "unitary_gauss": var_gauss,
    }


def plot_comparison(
    t_layers: np.ndarray,
    theoretical_vars: Dict[str, np.ndarray],
    simulation_vars: Dict[str, np.ndarray],
    simulation_expectations: Dict[str, np.ndarray],
    gamma: float,
) -> None:
    """Plot theoretical vs simulation variances for 2-qubit system."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Variance comparison (Qubit 0)
    ax1.set_title(f"Variance Comparison - Qubit 0 (γ={gamma})")
    ax1.set_xlabel("Physical Time (t)")
    ax1.set_ylabel("Variance")
    ax1.grid(True, alpha=0.3)
    
    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    
    for i, (method, theoretical_var) in enumerate(theoretical_vars.items()):
        color = colors[i % len(colors)]
        
        # Theoretical
        ax1.plot(t_layers, theoretical_var, "--", color=color, 
                label=f"{method} (theoretical)", linewidth=2)
        
        # Simulation
        if method in simulation_vars:
            sim_var_data = simulation_vars[method]
            # Handle 2D arrays (qubits × time) - show qubit 0
            if hasattr(sim_var_data, 'shape') and len(sim_var_data.shape) == 2:
                sim_var_1d = sim_var_data[0, :]  # Qubit 0
            else:
                sim_var_1d = sim_var_data
            
            # Adjust t_layers to match the length of sim_var_1d
            if len(sim_var_1d) == len(t_layers) - 1:
                t_plot = t_layers[1:]
            else:
                t_plot = t_layers
                
            ax1.plot(t_plot, sim_var_1d, "-", color=color,
                    label=f"{method} (simulation)", linewidth=1.5, alpha=0.8)
    
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Variance comparison (Qubit 1)
    ax2.set_title(f"Variance Comparison - Qubit 1 (γ={gamma})")
    ax2.set_xlabel("Physical Time (t)")
    ax2.set_ylabel("Variance")
    ax2.grid(True, alpha=0.3)
    
    for i, (method, theoretical_var) in enumerate(theoretical_vars.items()):
        color = colors[i % len(colors)]
        
        # Theoretical
        ax2.plot(t_layers, theoretical_var, "--", color=color, 
                label=f"{method} (theoretical)", linewidth=2)
        
        # Simulation
        if method in simulation_vars:
            sim_var_data = simulation_vars[method]
            # Handle 2D arrays (qubits × time) - show qubit 1
            if hasattr(sim_var_data, 'shape') and len(sim_var_data.shape) == 2:
                sim_var_1d = sim_var_data[1, :]  # Qubit 1
            else:
                sim_var_1d = sim_var_data
            
            # Adjust t_layers to match the length of sim_var_1d
            if len(sim_var_1d) == len(t_layers) - 1:
                t_plot = t_layers[1:]
            else:
                t_plot = t_layers
                
            ax2.plot(t_plot, sim_var_1d, "-", color=color,
                    label=f"{method} (simulation)", linewidth=1.5, alpha=0.8)
    
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    # Plot 3: Expectation value comparison (Qubit 0)
    ax3.set_title(f"Expectation Value Comparison - Qubit 0 (γ={gamma})")
    ax3.set_xlabel("Physical Time (t)")
    ax3.set_ylabel("⟨Z⟩")
    ax3.grid(True, alpha=0.3)
    
    # Use exact simulation as theoretical reference
    if "exact" in simulation_expectations:
        exact_exp = simulation_expectations["exact"]
        if hasattr(exact_exp, 'shape') and len(exact_exp.shape) == 2:
            exact_exp_0 = exact_exp[0, :]  # Qubit 0
        else:
            exact_exp_0 = exact_exp
        
        # Add initial state (t=0) with expectation value 1.0 for |0⟩ state
        if len(exact_exp_0) == len(t_layers) - 1:
            exact_exp_with_init = np.concatenate([[1.0], exact_exp_0])
        else:
            exact_exp_with_init = exact_exp_0
            
        ax3.plot(t_layers, exact_exp_with_init, "--", color="black", 
                label="exact (reference)", linewidth=2)
    
    # Simulation expectations
    for i, (method, exp_vals) in enumerate(simulation_expectations.items()):
        if method == "exact":  # Skip exact as it's already plotted as reference
            continue
        color = colors[i % len(colors)]
        # Handle 2D arrays (qubits × time) - show qubit 0
        if hasattr(exp_vals, 'shape') and len(exp_vals.shape) == 2:
            exp_vals_1d = exp_vals[0, :]  # Qubit 0
        else:
            exp_vals_1d = exp_vals
        
        # Adjust t_layers to match the length of exp_vals_1d
        if len(exp_vals_1d) == len(t_layers) - 1:
            # Add initial state (t=0) with expectation value 1.0 for |0⟩ state
            exp_vals_with_init = np.concatenate([[1.0], exp_vals_1d])
            t_plot = t_layers
        else:
            exp_vals_with_init = exp_vals_1d
            t_plot = t_layers
            
        ax3.plot(t_plot, exp_vals_with_init, "-", color=color,
                label=f"{method}", linewidth=1.5, alpha=0.8)
    
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    
    # Plot 4: Expectation value comparison (Qubit 1)
    ax4.set_title(f"Expectation Value Comparison - Qubit 1 (γ={gamma})")
    ax4.set_xlabel("Physical Time (t)")
    ax4.set_ylabel("⟨Z⟩")
    ax4.grid(True, alpha=0.3)
    
    # Use exact simulation as theoretical reference
    if "exact" in simulation_expectations:
        exact_exp = simulation_expectations["exact"]
        if hasattr(exact_exp, 'shape') and len(exact_exp.shape) == 2:
            exact_exp_1 = exact_exp[1, :]  # Qubit 1
        else:
            exact_exp_1 = exact_exp
        
        # Add initial state (t=0) with expectation value 1.0 for |0⟩ state
        if len(exact_exp_1) == len(t_layers) - 1:
            exact_exp_with_init = np.concatenate([[1.0], exact_exp_1])
        else:
            exact_exp_with_init = exact_exp_1
            
        ax4.plot(t_layers, exact_exp_with_init, "--", color="black", 
                label="exact (reference)", linewidth=2)
    
    # Simulation expectations
    for i, (method, exp_vals) in enumerate(simulation_expectations.items()):
        if method == "exact":  # Skip exact as it's already plotted as reference
            continue
        color = colors[i % len(colors)]
        # Handle 2D arrays (qubits × time) - show qubit 1
        if hasattr(exp_vals, 'shape') and len(exp_vals.shape) == 2:
            exp_vals_1d = exp_vals[1, :]  # Qubit 1
        else:
            exp_vals_1d = exp_vals
        
        # Adjust t_layers to match the length of exp_vals_1d
        if len(exp_vals_1d) == len(t_layers) - 1:
            # Add initial state (t=0) with expectation value 1.0 for |0⟩ state
            exp_vals_with_init = np.concatenate([[1.0], exp_vals_1d])
            t_plot = t_layers
        else:
            exp_vals_with_init = exp_vals_1d
            t_plot = t_layers
            
        ax4.plot(t_plot, exp_vals_with_init, "-", color=color,
                label=f"{method}", linewidth=1.5, alpha=0.8)
    
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()


def plot_multi_gamma(
    t_layers: np.ndarray,
    gammas: list[float],
    theoretical_vars_list: list[Dict[str, np.ndarray]],
    simulation_vars_list: list[Dict[str, np.ndarray]],
    simulation_expectations_list: list[Dict[str, np.ndarray]],
) -> None:
    """Plot only qubit 0 in a 2x3 grid: row 1 variances, row 2 expectations; columns are γ values."""
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    colors = {
        "standard": "tab:blue",
        "projector": "tab:red",
        "unitary_2pt": "tab:green",
        "unitary_gauss": "tab:orange",
        "qiskit": "tab:purple",
        "exact": "black",
    }

    for j, gamma in enumerate(gammas):
        theo = theoretical_vars_list[j]
        sim_vars = simulation_vars_list[j]
        sim_exp = simulation_expectations_list[j]

        ax_var = axs[0, j]
        ax_exp = axs[1, j]

        # Variance panel (qubit 0)
        ax_var.set_title(f"γ={gamma}")
        ax_var.set_ylabel("Variance" if j == 0 else "")
        ax_var.grid(True, alpha=0.3)
        for method in ["standard", "projector", "unitary_2pt", "unitary_gauss"]:
            if method in theo:
                ax_var.plot(t_layers, theo[method], "--", color=colors.get(method, None), linewidth=2, label=f"{method} (theory)" if j == 0 else None)
            if method in sim_vars:
                sim_var_data = sim_vars[method]
                if hasattr(sim_var_data, 'shape') and len(sim_var_data.shape) == 2:
                    sim_1d = sim_var_data[0, :]  # qubit 0
                else:
                    sim_1d = sim_var_data
                t_plot = t_layers[1:] if len(sim_1d) == len(t_layers) - 1 else t_layers
                ax_var.plot(t_plot, sim_1d, '-', color=colors.get(method, None), linewidth=1.5, alpha=0.85, label=f"{method} (sim)" if j == 0 else None)

        ax_var.set_ylim(0, 1.1)

        # Expectation panel (qubit 0)
        ax_exp.set_xlabel("Physical Time (t)")
        ax_exp.set_ylabel("⟨Z⟩" if j == 0 else "")
        ax_exp.grid(True, alpha=0.3)

        # Exact as reference (if available)
        if "exact" in sim_exp:
            exact_exp = sim_exp["exact"]
            if hasattr(exact_exp, 'shape') and len(exact_exp.shape) == 2:
                exact0 = exact_exp[0, :]
            else:
                exact0 = exact_exp
            exact_with_init = np.concatenate([[1.0], exact0]) if len(exact0) == len(t_layers) - 1 else exact0
            ax_exp.plot(t_layers, exact_with_init, '--', color=colors["exact"], linewidth=2, label="exact" if j == 0 else None)

        for method, exp_vals in sim_exp.items():
            if method == "exact":
                continue
            if hasattr(exp_vals, 'shape') and len(exp_vals.shape) == 2:
                e0 = exp_vals[0, :]
            else:
                e0 = exp_vals
            exp_with_init = np.concatenate([[1.0], e0]) if len(e0) == len(t_layers) - 1 else e0
            ax_exp.plot(t_layers, exp_with_init, '-', color=colors.get(method, None), linewidth=1.5, alpha=0.85, label=method if j == 0 else None)

        ax_exp.set_ylim(0, 1.1)

    # Single legend for the whole figure on the right
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if not handles:
        # build from expectation row if variance had none
        handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()


def print_comparison_table(
    t_layers: np.ndarray,
    theoretical_vars: Dict[str, np.ndarray],
    simulation_vars: Dict[str, np.ndarray],
    gamma: float,
) -> None:
    """Print a comparison table of theoretical vs simulation variances for 2-qubit system."""
    print(f"\nVariance Comparison (γ={gamma}) - 2-Qubit System")
    print("=" * 100)
    print(f"{'Time':<8} {'Qubit':<6} {'Method':<12} {'Theoretical':<12} {'Simulation':<12} {'Error':<12}")
    print("-" * 100)
    
    # Sample a few time points
    time_indices = np.linspace(0, len(t_layers)-1, min(5, len(t_layers)), dtype=int)
    
    for t_idx in time_indices:
        t_val = t_layers[t_idx]
        print(f"{t_val:<8.2f}")
        
        for qubit in [0, 1]:
            for method in theoretical_vars.keys():
                theo_var = float(theoretical_vars[method][t_idx])
                sim_var_array = simulation_vars.get(method, np.zeros_like(t_layers))
                
                # Handle 2D arrays (qubits × time) - show specific qubit
                if hasattr(sim_var_array, 'shape') and len(sim_var_array.shape) == 2:
                    sim_var = float(sim_var_array[qubit, t_idx])
                elif hasattr(sim_var_array, 'shape') and len(sim_var_array.shape) == 1:
                    sim_var = float(sim_var_array[t_idx])
                else:
                    sim_var = float(sim_var_array)
                
                error = abs(theo_var - sim_var)
                
                print(f"{'':8} {qubit:<6} {method:<12} {theo_var:<12.6f} {sim_var:<12.6f} {error:<12.6f}")
        print()


if __name__ == "__main__":
    # Parameters (shared)
    L = 2
    num_layers = 150
    num_traj = 2000
    dt = 1.0  # Time step per layer

    # Choose three noise strengths (columns)
    gammas = [0.1, 0.01, 0.001]

    # Physical times at each layer
    t_layers = np.arange(num_layers + 1) * dt

    print(f"Running theoretical variance comparison (multi-γ)...")
    print(f"num_layers={num_layers}, num_traj={num_traj}, t_layers={t_layers}")

    # Build circuits once
    basis_circuit = QuantumCircuit(L)
    basis_circuit.rxx(0.0, 0, 1)
    init_circuit = QuantumCircuit(L)

    theoretical_vars_list: list[Dict[str, np.ndarray]] = []
    simulation_vars_list: list[Dict[str, np.ndarray]] = []
    simulation_expectations_list: list[Dict[str, np.ndarray]] = []

    for gamma in gammas:
        print(f"\n=== γ = {gamma} ===")

        # Compute theoretical variances for this γ
        theoretical_vars = theoretical_variance_formulas(t_layers, gamma)

        # YAQS noise models for this γ
        processes = (
            [{"name": "pauli_x", "sites": [0], "strength": gamma}]
            + [{"name": "crosstalk_xx", "sites": [0, 1], "strength": gamma}]
            + [{"name": "pauli_x", "sites": [1], "strength": gamma}]
        )
        yaqs_noise_models = build_noise_models(processes)

        # Qiskit noise model for this γ
        noise_model = QiskitNoiseModel()
        x_error = PauliLindbladError([Pauli("XX"), Pauli("IX"), Pauli("XI")], [gamma, gamma, gamma])
        noise_model.add_all_qubit_quantum_error(x_error, ["rxx"]) 

        # Run simulations for this γ
        simulation_vars: Dict[str, np.ndarray] = {}
        simulation_expectations: Dict[str, np.ndarray] = {}

        print("Running YAQS simulations...")
        method_names = ["standard", "projector", "unitary_2pt", "unitary_gauss"]
        for i, yaqs_noise_model in enumerate(yaqs_noise_models):
            method = method_names[i]
            print(f"  {method}...")
            exp_vals, _ , vars = run_yaqs(init_circuit, basis_circuit, L, num_layers, yaqs_noise_model, num_traj=num_traj)
            simulation_vars[method] = vars
            simulation_expectations[method] = exp_vals

        print("Running Qiskit simulations...")
        exp_vals_exact, _ = run_qiskit_exact(L, num_layers, init_circuit, basis_circuit, noise_model)
        simulation_expectations["exact"] = exp_vals_exact

        # Optional: MPS variant (kept for parity with earlier runs)
        exp_vals, _ , vars = run_qiskit_mps(L, num_layers, init_circuit, basis_circuit, noise_model, num_traj=num_traj)
        simulation_vars["qiskit"] = vars
        simulation_expectations["qiskit_mps"] = exp_vals

        # Store results for this γ
        theoretical_vars_list.append(theoretical_vars)
        simulation_vars_list.append(simulation_vars)
        simulation_expectations_list.append(simulation_expectations)

        # Optionally print a brief table for γ
        # print_comparison_table(t_layers, theoretical_vars, simulation_vars, gamma)

    # Persist all data needed to recreate the plots
    def fmt_float(x: float) -> str:
        s = ("%g" % x)
        return s.replace(".", "p")

    gamma_tag = "-".join(fmt_float(g) for g in gammas)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"variance_comparison_ntraj{num_traj}_L{L}_layers{num_layers}_dt{fmt_float(dt)}_gammas_{gamma_tag}_{ts}.pkl"
    out_path = os.path.join(_HERE, filename)

    payload = {
        "meta": {
            "L": L,
            "num_layers": num_layers,
            "num_traj": num_traj,
            "dt": dt,
            "gammas": gammas,
            "hazard_gain_analog": 3.0,   # used in yaqs_simulator.build_noise_models
            "gauss_M": 11,
            "note": "2x3 figure; only qubit 0 plotted"
        },
        "t_layers": t_layers,
        "theoretical_vars_list": theoretical_vars_list,
        "simulation_vars_list": simulation_vars_list,
        "simulation_expectations_list": simulation_expectations_list,
    }

    try:
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved results to: {out_path}")
    except Exception as e:
        print(f"WARNING: Failed to save pickle to {out_path}: {e}")

    # Plot combined 2x3 (variance top, expectation bottom), only qubit 0
    plot_multi_gamma(t_layers, gammas, theoretical_vars_list, simulation_vars_list, simulation_expectations_list)

    print("Comparison complete!")


