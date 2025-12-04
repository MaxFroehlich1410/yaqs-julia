#!/usr/bin/env python3
"""
Micro-experiment: 2-qubit circuit with a no-op trotter step and X-type noise.

Runs four YAQS analog_auto configurations sweeping a "hazard" parameter and
compares against an exact Qiskit density-matrix reference.
"""

import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.codex_experiments.worker_functions.yaqs_simulator import run_yaqs
from mqt.yaqs.codex_experiments.worker_functions.qiskit_simulators import run_qiskit_exact
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel, PauliLindbladError

import matplotlib.pyplot as plt


def build_noop_trotter_step(num_qubits: int) -> QuantumCircuit:
    """Build a trotter step that has no effect on |00...0> (no-op for our purposes)."""
    qc = QuantumCircuit(num_qubits)
    qc.rxx(0.0, 0, 1)
    return qc


if __name__ == "__main__":
    # Parameters
    L = 2
    gamma = 0.001  # Noise rate
    num_layers = 15
    num_traj = 1000
    dt = 1.0  # Time step per layer
    
    # Physical times at each layer
    t_layers = np.arange(num_layers + 1) * dt
    
    print("Running theoretical variance comparison...")
    print(f"Parameters: γ={gamma}, num_layers={num_layers}, num_traj={num_traj}")
    print(f"Physical times: {t_layers}")
    
    # Build circuit and noise models
    init_circuit = QuantumCircuit(L)
    trotter_step = build_noop_trotter_step(L)

    hazards = [0.01, 0.3, 0.7, 1.0, 1.5]
    processes = [
        {"name": "pauli_x", "sites": [0], "strength": gamma, "unraveling": "analog_auto"},
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": gamma, "unraveling": "analog_auto"},
        {"name": "pauli_x", "sites": [1], "strength": gamma, "unraveling": "analog_auto"},
    ]
    yaqs_noise_models = [
        NoiseModel(processes, num_qubits=L, hazard_gain=1.0, hazard_cap=0.0, gauss_M=11, gauss_k=4.0)
        for h in hazards
    ]
    noise_model = QiskitNoiseModel()
    # Pauli-X error with rate gamma
    x_error = PauliLindbladError([Pauli("XX"), Pauli("IX"), Pauli("XI")], [gamma, gamma, gamma])
    noise_model.add_all_qubit_quantum_error(x_error, ["rxx"]) 

    # Run simulations
    simulation_vars = {}
    simulation_expectations = {}

    print("\nRunning YAQS simulations...")
    for h, yaqs_noise_model in zip(hazards, yaqs_noise_models):
        label = f"hazard_{h}"
        print(f"  {label}...")
        exp_vals, _bonds, yaqs_vars = run_yaqs(init_circuit, trotter_step, L, num_layers, yaqs_noise_model, num_traj=num_traj)
        simulation_vars[label] = yaqs_vars
        simulation_expectations[label] = exp_vals

    print("Running Qiskit simulations...")
    # Run exact density matrix simulation as theoretical reference
    exp_vals_exact = run_qiskit_exact(L, num_layers, init_circuit, trotter_step, noise_model)
    simulation_expectations["exact"] = exp_vals_exact

    # Print concise summary
    print("\nSummary (mean |⟨Z⟩_YAQS - ⟨Z⟩_exact| over qubits and layers):")
    exact = exp_vals_exact
    for label, expvals in simulation_expectations.items():
        if label == "exact":
            continue
        mae = float(np.mean(np.abs(expvals - exact)))
        print(f"  {label:>12}: {mae:.6e}")

    # Plot all simulations (two subplots for qubits 0 and 1)
    layers = np.arange(0, num_layers + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)
    colors = {
        "hazard_0.3": "tab:blue",
        "hazard_0.7": "tab:orange",
        "hazard_1.0": "tab:green",
        "hazard_1.5": "tab:red",
        "exact": "black",
    }
    # Build augmented series with initial t=0 value == 1.0 for all qubits
    augmented = {}
    for label, series in simulation_expectations.items():
        if series.ndim == 2:
            ones = np.ones((series.shape[0], 1), dtype=float)
            augmented[label] = np.concatenate([ones, series], axis=1)
        else:
            augmented[label] = np.concatenate([[1.0], series])
    for q in [0, 1]:
        ax = axs[q]
        for label, series in augmented.items():
            y = series[q]
            style = "--" if label == "exact" else "-"
            alpha = 0.9 if label == "exact" else 0.8
            ax.plot(layers, y, style, label=label, color=colors.get(label, None), alpha=alpha)
        ax.set_title(f"qubit {q}")
        ax.set_xlabel("Layer")
        if q == 0:
            ax.set_ylabel("⟨Z⟩")
        ax.grid(True, linestyle="--", alpha=0.5)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    plt.show()


