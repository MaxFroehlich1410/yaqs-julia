#!/usr/bin/env python3
"""
Practical proof: analog_gauss variance decreases with lower s values.

Uses the same circuit and setup as theoretical_variance_comparison.py but
simulates with 3 different s values for analog_gauss unraveling to demonstrate
variance reduction with decreasing s.
"""

import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

# Import YAQS components
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.codex_experiments.worker_functions.yaqs_simulator import run_yaqs
from mqt.yaqs.codex_experiments.worker_functions.qiskit_simulators import run_qiskit_exact

# Import Qiskit noise components
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel, PauliLindbladError


def build_simple_circuit() -> QuantumCircuit:
    """Build a simple 2-qubit circuit with identity gates that noise can act on."""
    qc = QuantumCircuit(2)
    qc.rxx(0.0, 0, 1)
    return qc


def create_analog_gauss_models(process_list, s_vals, num_qubits=2):
    """Create NoiseModel instances with analog_gauss unraveling for different s values."""
    models = []
    for s in s_vals:
        # For analog_gauss, we need to set sigma such that s = (1 - exp(-2*sigma^2))/2
        # Solving: sigma = sqrt(-0.5 * log(1 - 2*s))
        if s >= 0.5:
            s = 0.5 - 1e-6  # Clamp to avoid numerical issues
        sigma = np.sqrt(-0.5 * np.log(1.0 - 2.0 * s))
        
        # Create processes with analog_gauss unraveling
        gauss_processes = []
        for proc in process_list:
            gauss_proc = proc.copy()
            gauss_proc["unraveling"] = "unitary_gauss"
            gauss_proc["sigma"] = sigma
            gauss_processes.append(gauss_proc)
        
        model = NoiseModel(gauss_processes, num_qubits=num_qubits, gauss_M=11, gauss_k=4.0)
        models.append(model)
    
    return models


def plot_s_variance_comparison(
    time_layers: np.ndarray,
    sim_vars: Dict[str, np.ndarray],
    sim_expectations: Dict[str, np.ndarray],
    s_vals: list[float],
    gamma_val: float,
) -> None:
    """Plot variance and expectation value comparison for different s values."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Variance comparison (Qubit 0)
    ax1.set_title(f"Variance vs s - Qubit 0 (γ={gamma_val})")
    ax1.set_xlabel("Physical Time (t)")
    ax1.set_ylabel("Variance")
    ax1.grid(True, alpha=0.3)
    
    colors = ["blue", "red", "green", "orange", "purple"]
    
    for i, s in enumerate(s_vals):
        label = f"analog_gauss s={s:.3f}"
        color = colors[i % len(colors)]
        
        if label in sim_vars:
            sim_var_data = sim_vars[label]
            # Handle 2D arrays (qubits × time) - show qubit 0
            if hasattr(sim_var_data, 'shape') and len(sim_var_data.shape) == 2:
                sim_var_1d = sim_var_data[0, :]  # Qubit 0
            else:
                sim_var_1d = sim_var_data
            
            # Adjust time_layers to match the length of sim_var_1d
            if len(sim_var_1d) == len(time_layers) - 1:
                t_plot = time_layers[1:]
            else:
                t_plot = time_layers
                
            ax1.plot(t_plot, sim_var_1d, "-", color=color,
                    label=label, linewidth=2, alpha=0.8)
    
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Variance comparison (Qubit 1)
    ax2.set_title(f"Variance vs s - Qubit 1 (γ={gamma_val})")
    ax2.set_xlabel("Physical Time (t)")
    ax2.set_ylabel("Variance")
    ax2.grid(True, alpha=0.3)
    
    for j, s_val in enumerate(s_vals):
        label = f"analog_gauss s={s_val:.3f}"
        color = colors[j % len(colors)]
        
        if label in sim_vars:
            sim_var_data = sim_vars[label]
            # Handle 2D arrays (qubits × time) - show qubit 1
            if hasattr(sim_var_data, 'shape') and len(sim_var_data.shape) == 2:
                sim_var_1d = sim_var_data[1, :]  # Qubit 1
            else:
                sim_var_1d = sim_var_data
            
            # Adjust time_layers to match the length of sim_var_1d
            if len(sim_var_1d) == len(time_layers) - 1:
                t_plot = time_layers[1:]
            else:
                t_plot = time_layers
                
            ax2.plot(t_plot, sim_var_1d, "-", color=color,
                    label=label, linewidth=2, alpha=0.8)
    
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    # Plot 3: Expectation value comparison (Qubit 0)
    ax3.set_title(f"Expectation Value vs s - Qubit 0 (γ={gamma})")
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
    for i, s in enumerate(s_values):
        label = f"analog_gauss s={s:.3f}"
        color = colors[i % len(colors)]
        if label in simulation_expectations:
            exp_vals = simulation_expectations[label]
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
                    label=label, linewidth=2, alpha=0.8)
    
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    
    # Plot 4: Expectation value comparison (Qubit 1)
    ax4.set_title(f"Expectation Value vs s - Qubit 1 (γ={gamma})")
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
    for i, s in enumerate(s_values):
        label = f"analog_gauss s={s:.3f}"
        color = colors[i % len(colors)]
        if label in simulation_expectations:
            exp_vals = simulation_expectations[label]
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
                    label=label, linewidth=2, alpha=0.8)
    
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()


def print_s_comparison_table(
    t_layers: np.ndarray,
    simulation_vars: Dict[str, np.ndarray],
    s_values: list[float],
    gamma: float,
) -> None:
    """Print a comparison table of variances for different s values."""
    print(f"\nVariance Comparison for analog_gauss (γ={gamma})")
    print("=" * 100)
    print(f"{'Time':<8} {'Qubit':<6} {'s value':<12} {'Variance':<12}")
    print("-" * 100)
    
    # Sample a few time points
    time_indices = np.linspace(0, len(t_layers)-1, min(5, len(t_layers)), dtype=int)
    
    for t_idx in time_indices:
        t_val = t_layers[t_idx]
        print(f"{t_val:<8.2f}")
        
        for qubit in [0, 1]:
            for s in s_values:
                label = f"analog_gauss s={s:.3f}"
                sim_var_array = simulation_vars.get(label, np.zeros_like(t_layers))
                
                # Handle 2D arrays (qubits × time) - show specific qubit
                if hasattr(sim_var_array, 'shape') and len(sim_var_array.shape) == 2:
                    sim_var = float(sim_var_array[qubit, t_idx])
                elif hasattr(sim_var_array, 'shape') and len(sim_var_array.shape) == 1:
                    sim_var = float(sim_var_array[t_idx])
                else:
                    sim_var = float(sim_var_array)
                
                print(f"{'':8} {qubit:<6} {s:<12.3f} {sim_var:<12.6f}")
        print()


if __name__ == "__main__":
    # Parameters - same as theoretical_variance_comparison.py
    L = 2
    gamma = 0.01  # Noise rate
    num_layers = 20
    num_traj = 1000
    dt = 1.0  # Time step per layer
    
    # Physical times at each layer
    t_layers = np.arange(num_layers + 1) * dt
    
    print(f"Running analog_gauss s-variance comparison...")
    print(f"Parameters: γ={gamma}, num_layers={num_layers}, num_traj={num_traj}")
    print(f"Physical times: {t_layers}")
    
    # Build circuit - same as theoretical_variance_comparison.py
    init_circuit = QuantumCircuit(L)
    trotter_step = build_simple_circuit()
    
    # Define s values to test (decreasing order to show variance reduction)
    s_values = [0.4, 0.2, 0.1]  # Higher s should have higher variance
    
    # Build processes - same as theoretical_variance_comparison.py
    processes = [
        {"name": "pauli_x", "sites": [0], "strength": gamma},
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": gamma},
        {"name": "pauli_x", "sites": [1], "strength": gamma}
    ]
    
    # Create analog_gauss models with different s values
    yaqs_noise_models = create_analog_gauss_models(processes, s_values, L)
    
    # Qiskit noise model - same as theoretical_variance_comparison.py
    noise_model = QiskitNoiseModel()
    x_error = PauliLindbladError([Pauli("XX"), Pauli("IX"), Pauli("XI")], [gamma, gamma, gamma])
    noise_model.add_all_qubit_quantum_error(x_error, ["rxx"]) 
    
    # Run simulations
    simulation_vars = {}
    simulation_expectations = {}
    
    print("\nRunning YAQS analog_gauss simulations...")
    for i, (s, yaqs_noise_model) in enumerate(zip(s_values, yaqs_noise_models)):
        label = f"analog_gauss s={s:.3f}"
        print(f"  {label}...")
        exp_vals, _, yaqs_vars = run_yaqs(init_circuit, trotter_step, L, num_layers, yaqs_noise_model, num_traj=num_traj)
        simulation_vars[label] = yaqs_vars
        simulation_expectations[label] = exp_vals
    
    print("Running Qiskit exact simulation...")
    # Run exact density matrix simulation as theoretical reference
    exp_vals_exact = run_qiskit_exact(L, num_layers, init_circuit, trotter_step, noise_model)
    simulation_expectations["exact"] = exp_vals_exact
    
    # Print comparison table
    print_s_comparison_table(t_layers, simulation_vars, s_values, gamma)
    
    # Plot results
    plot_s_variance_comparison(t_layers, simulation_vars, simulation_expectations, s_values, gamma)
    
    print("Comparison complete!")
    print(f"\nExpected result: Lower s values should show lower variance")
    print(f"Tested s values: {s_values} (in decreasing order)")
