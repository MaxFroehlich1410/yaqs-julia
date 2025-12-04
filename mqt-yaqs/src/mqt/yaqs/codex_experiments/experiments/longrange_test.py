import numpy as np
import matplotlib.pyplot as plt

from mqt.yaqs.codex_experiments.worker_functions.qiskit_simulators import run_qiskit_exact, run_qiskit_mps
from mqt.yaqs.codex_experiments.worker_functions.yaqs_simulator import run_yaqs
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
import copy


from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer.noise.errors import PauliLindbladError
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit

def staggered_magnetization(z, num_qubits):
    return np.sum([(-1)**i * z[i] for i in range(num_qubits)]) / num_qubits


def build_noise_models(processes, num_qubits):
    # Always deep-copy; each NoiseModel gets its own process list.
    procs_std  = copy.deepcopy(processes)
    procs_proj = copy.deepcopy(processes)
    procs_2pt  = copy.deepcopy(processes)
    procs_gaus = copy.deepcopy(processes)

    # (1) standard (whatever your default is)
    noise_model_normal = NoiseModel(procs_std, num_qubits=num_qubits)

    # (2) projector unraveling: same Lindblad rate γ per process
    for p in procs_proj:
        p["unraveling"] = "projector"
    for p in procs_2pt:
        p["unraveling"] = "unitary_2pt"
    for p in procs_gaus:
        p["unraveling"] = "unitary_gauss"
        # strength unchanged
    noise_model_projector = NoiseModel(procs_proj, num_qubits=num_qubits)
    noise_model_unitary_2pt = NoiseModel(procs_2pt, num_qubits=num_qubits)
    noise_model_unitary_gauss = NoiseModel(procs_gaus, num_qubits=num_qubits, gauss_M=11)

    return (noise_model_normal,
            noise_model_projector,
            noise_model_unitary_2pt,
            noise_model_unitary_gauss)



def xy_trotter_layer(N, tau, order="YX") -> QuantumCircuit:
    """Create one Trotter step for the XY Hamiltonian."""
    qc = QuantumCircuit(N)
    even = [(i, i+1) for i in range(0, N-1, 2)]
    odd  = [(i, i+1) for i in range(1, N-1, 2)]

    def apply_pairwise(gate_name):
        for a, b in even: 
            getattr(qc, gate_name)(2*tau, a, b)
        for a, b in odd:  
            getattr(qc, gate_name)(2*tau, a, b)

    if order == "YX":
        apply_pairwise("ryy")
        apply_pairwise("rxx")
    else:
        apply_pairwise("rxx")
        apply_pairwise("ryy")
    
    return qc


def compute_mse(pred, exact):
    """Compute Mean Squared Error between prediction and exact solution."""
    return np.mean([(pred[i] - exact[i])**2 for i in range(len(pred))])


def find_required_trajectories(
    method_name,
    simulator_func,
    exact_z_expvals,
    threshold_mse,
    init_circuit,
    trotter_step,
    num_qubits,
    num_layers,
    noise_model,
    qiskit_noise_model,
    z_initial,
    max_traj=1000,
    fixed_traj=None
):
    """
    Find minimum number of trajectories needed to achieve target MSE.
    
    Runs trajectories incrementally, accumulating results one at a time
    until threshold is met. This is much more efficient than re-running
    all trajectories from scratch each time.
    
    Args:
        method_name: Name of the method for logging
        simulator_func: Function to run simulation (run_yaqs or run_qiskit_mps) (None if not running)
        exact_z_expvals: Exact Z expectation values reference (None if unavailable) - shape (num_qubits, num_layers+1)
        threshold_mse: Target MSE threshold (None if no exact reference)
        fixed_traj: If set, run exactly this many trajectories (for large systems)
        max_traj: Maximum trajectories to try
    
    Returns:
        (num_trajectories_needed, final_mse, z_expvals, bond_dims)
    """
    # Determine how many trajectories to run
    use_mse_threshold = (exact_z_expvals is not None and threshold_mse is not None)
    target_traj = fixed_traj if fixed_traj is not None else max_traj
    
    # If we're running a fixed number of trajectories (large system mode),
    # run them all in parallel for efficiency
    if fixed_traj is not None and not use_mse_threshold:
        print(f"  {method_name}: Running {fixed_traj} trajectories in parallel...")
        
        # Run all trajectories at once
        if method_name.startswith("YAQS"):
            results_all, bond_dims, _ = simulator_func(
                init_circuit, trotter_step, num_qubits, num_layers, 
                noise_model, num_traj=fixed_traj, parallel=True
            )
        else:  # Qiskit MPS
            results_all, bond_dims, _ = simulator_func(
                num_qubits, num_layers, init_circuit, trotter_step,
                qiskit_noise_model, num_traj=fixed_traj
            )
        
        # Store Z expectation values: shape (num_qubits, num_layers+1)
        z_expvals = np.column_stack([z_initial, results_all])
        
        print(f"  ✓ {method_name}: Completed {fixed_traj} trajectory(ies) in parallel")
        return fixed_traj, None, z_expvals, bond_dims
    
    # Otherwise, run trajectories incrementally (small system mode with MSE checking)
    cumulative_results = None  # Will store sum of z-expectation values
    bond_dims_list = []  # Collect bond dims from each trajectory
    
    for num_traj in range(1, target_traj + 1):
        # Run a single trajectory
        if method_name.startswith("YAQS"):
            single_result, single_bonds, _ = simulator_func(
                init_circuit, trotter_step, num_qubits, num_layers, 
                noise_model, num_traj=1, parallel=False
            )
        else:  # Qiskit MPS
            single_result, single_bonds, _ = simulator_func(
                num_qubits, num_layers, init_circuit, trotter_step,
                qiskit_noise_model, num_traj=1
            )
        
        # Accumulate results
        if cumulative_results is None:
            cumulative_results = single_result.copy()
        else:
            cumulative_results += single_result
        
        # Store bond dimensions
        if single_bonds is not None:
            if isinstance(single_bonds, dict):
                # Qiskit format: extract per-shot data
                if "per_shot_per_layer_max_bond_dim" in single_bonds:
                    bond_dims_list.append(single_bonds["per_shot_per_layer_max_bond_dim"][0])
            elif isinstance(single_bonds, np.ndarray):
                # YAQS format: single trajectory result
                if single_bonds.ndim == 2:
                    bond_dims_list.append(single_bonds[0])
                else:
                    bond_dims_list.append(single_bonds)
        
        # Compute average over all trajectories so far
        avg_results = cumulative_results / num_traj
        
        # Store Z expectation values: shape (num_qubits, num_layers+1)
        z_expvals = np.column_stack([z_initial, avg_results])
        
        # Compute MSE if exact reference is available
        if use_mse_threshold:
            mse = compute_mse(z_expvals.flatten(), exact_z_expvals.flatten())
            print(f"  {method_name}: Trajectory {num_traj}: MSE = {mse:.6e} (threshold = {threshold_mse:.6e})")
            
            # Check if threshold is met
            if mse < threshold_mse:
                print(f"  ✓ {method_name}: Target reached with {num_traj} trajectory(ies)!")
                
                # Format bond dimensions for return
                if len(bond_dims_list) > 0:
                    if method_name.startswith("YAQS"):
                        bond_dims = np.array(bond_dims_list)
                    else:
                        bond_array = np.array(bond_dims_list)
                        bond_dims = {
                            "per_shot_per_layer_max_bond_dim": bond_array,
                            "per_layer_mean_across_shots": np.mean(bond_array, axis=0)
                        }
                else:
                    bond_dims = None
                
                return num_traj, mse, z_expvals, bond_dims
        else:
            # No exact reference - just report progress
            print(f"  {method_name}: Trajectory {num_traj}/{target_traj} completed")
            mse = None
    
    # Finished all trajectories
    if use_mse_threshold:
        print(f"  ✗ {method_name}: Failed to reach threshold with {target_traj} trajectories")
    else:
        print(f"  ✓ {method_name}: Completed {target_traj} trajectory(ies)")
    
    # Format final bond dimensions
    if len(bond_dims_list) > 0:
        if method_name.startswith("YAQS"):
            bond_dims = np.array(bond_dims_list)
        else:
            bond_array = np.array(bond_dims_list)
            bond_dims = {
                "per_shot_per_layer_max_bond_dim": bond_array,
                "per_layer_mean_across_shots": np.mean(bond_array, axis=0)
            }
    else:
        bond_dims = None
    
    return target_traj, mse, z_expvals, bond_dims


if __name__ == "__main__":
    # Simulation parameters
    num_qubits = 6
    num_layers = 10
    tau = 0.1
    noise_strength = 0.01
    
    # ========== MODE SELECTION ==========
    # For small systems: Set run_density_matrix=True and specify threshold_mse
    # For large systems: Set run_density_matrix=False and specify fixed_trajectories
    run_density_matrix = True  # Set to False for large systems (>12 qubits)
    enable_qiskit_mps = True
    enable_yaqs_standard = True
    enable_yaqs_projector = True
    enable_yaqs_unitary_2pt = True
    enable_yaqs_unitary_gauss = True
    threshold_mse = 5e-4  # Target MSE threshold (only used if run_density_matrix=True)
    fixed_trajectories = 100  # Number of trajectories for large systems (only used if run_density_matrix=False)
    # ====================================
    
    print("="*70)
    print("Trajectory Efficiency Comparison for Unraveling Methods")
    print("="*70)
    print(f"System: {num_qubits} qubits, {num_layers} layers")
    print(f"Noise strength: {noise_strength}")
    if run_density_matrix:
        print(f"Mode: With exact reference (density matrix)")
        print(f"Target MSE threshold: {threshold_mse:.2e}")
    else:
        print(f"Mode: No exact reference (large system)")
        print(f"Fixed trajectories per method: {fixed_trajectories}")
    print("="*70)

    # Prepare initial state circuit
    init_circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        if i % 4 == 3:
            init_circuit.x(i)
    
    # One Trotter step
    trotter_step = create_ising_circuit(num_qubits, 1.0, 0.5, tau, 1, periodic=True)
    # trotter_step.draw(output="mpl")
    # plt.show()

    # Initialize noise models (YAQS)
    processes = [
        {"name": "pauli_x", "sites": [i], "strength": noise_strength}
        for i in range(num_qubits)
     ] + [
        {"name": "crosstalk_xx", "sites": [i, i+1], "strength": noise_strength}
        for i in range(num_qubits - 1)
    ] + [
        {"name": "crosstalk_xx", "sites": [0, num_qubits - 1], "strength": noise_strength}
    ]
    noise_model_normal, noise_model_projector, noise_model_unitary_2pt, noise_model_unitary_gauss = build_noise_models(processes, num_qubits)

    # Initialize Qiskit noise model
    qiskit_noise_model = QiskitNoiseModel()
    TwoQubit_XX_error = PauliLindbladError(
        [Pauli("IX"), Pauli("XI"), Pauli("XX")],
        [noise_strength, noise_strength, noise_strength]
    )
    for qubit in range(num_qubits):
        next_qubit = (qubit + 1) % num_qubits
        qiskit_noise_model.add_quantum_error(
            TwoQubit_XX_error,
            ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"],
            [qubit, next_qubit]
        )
    qiskit_noise_model.add_quantum_error(
        TwoQubit_XX_error,
        ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"],
        [0, num_qubits - 1]
    )

    # Compute initial Z expectation values
    z_initial = np.array([1.0 if i % 4 != 3 else -1.0 for i in range(num_qubits)])
    
    # Run exact density matrix simulation (reference) if enabled
    if run_density_matrix:
        print("\nRunning exact density matrix simulation (reference)...")
        z_expvals_exact_results = run_qiskit_exact(
            num_qubits, num_layers, init_circuit, trotter_step, 
            qiskit_noise_model, method="density_matrix"
        )
        # Shape: (num_qubits, num_layers+1) - prepend initial values
        exact_z_expvals = np.column_stack([z_initial, z_expvals_exact_results])
        print("Exact reference computed.\n")
    else:
        print("\nSkipping exact density matrix simulation (large system mode).\n")
        exact_z_expvals = None

    # Test each method
    results = {}
    
    if run_density_matrix:
        print("Finding minimum trajectories for each method...")
    else:
        print(f"Running {fixed_trajectories} trajectories for each method...")
    print("-"*70)
    
    # Qiskit MPS
    if enable_qiskit_mps:
        print("\n1. Qiskit MPS (Standard Unraveling)")
        num_traj_mps, mse_mps, z_expvals_mps, bonds_mps = find_required_trajectories(
            "Qiskit MPS",
            run_qiskit_mps,
            exact_z_expvals,
            threshold_mse,
            init_circuit,
            trotter_step,
            num_qubits,
            num_layers,
            None,
            qiskit_noise_model,
            z_initial,
            max_traj=1000,
            fixed_traj=None if run_density_matrix else fixed_trajectories
        )
        results["Qiskit MPS"] = {"trajectories": num_traj_mps, "mse": mse_mps, "z_expvals": z_expvals_mps, "bonds": bonds_mps}
    
    # YAQS Standard
    if enable_yaqs_standard:
        print("\n2. YAQS Standard Unraveling")
        num_traj_std, mse_std, z_expvals_std, bonds_std = find_required_trajectories(
            "YAQS Standard",
            run_yaqs,
            exact_z_expvals,
            threshold_mse if run_density_matrix else None,
            init_circuit,
            trotter_step,
            num_qubits,
            num_layers,
            noise_model_normal,
            None,
            z_initial,
            max_traj=1000,
            fixed_traj=None if run_density_matrix else fixed_trajectories
        )
        results["YAQS Standard"] = {"trajectories": num_traj_std, "mse": mse_std, "z_expvals": z_expvals_std, "bonds": bonds_std}
        
    # YAQS Projector
    if enable_yaqs_projector:
        print("\n3. YAQS Projector Unraveling")
        num_traj_proj, mse_proj, z_expvals_proj, bonds_proj = find_required_trajectories(
            "YAQS Projector",
            run_yaqs,
            exact_z_expvals,
            threshold_mse if run_density_matrix else None,
            init_circuit,
            trotter_step,
            num_qubits,
            num_layers,
            noise_model_projector,
            None,
            z_initial,
            max_traj=1000,
            fixed_traj=None if run_density_matrix else fixed_trajectories
        )
        results["YAQS Projector"] = {"trajectories": num_traj_proj, "mse": mse_proj, "z_expvals": z_expvals_proj, "bonds": bonds_proj}
        
    # YAQS Unitary 2pt
    if enable_yaqs_unitary_2pt:
        print("\n4. YAQS Unitary 2pt Unraveling")
        num_traj_2pt, mse_2pt, z_expvals_2pt, bonds_2pt = find_required_trajectories(
            "YAQS Unitary 2pt",
            run_yaqs,
            exact_z_expvals,
            threshold_mse if run_density_matrix else None,
            init_circuit,
            trotter_step,
            num_qubits,
            num_layers,
            noise_model_unitary_2pt,
            None,
            z_initial,
            max_traj=1000,
            fixed_traj=None if run_density_matrix else fixed_trajectories
        )
        results["YAQS Unitary 2pt"] = {"trajectories": num_traj_2pt, "mse": mse_2pt, "z_expvals": z_expvals_2pt, "bonds": bonds_2pt}
        
    # YAQS Unitary Gauss
    if enable_yaqs_unitary_gauss:
        print("\n5. YAQS Unitary Gauss Unraveling")
        num_traj_gauss, mse_gauss, z_expvals_gauss, bonds_gauss = find_required_trajectories(
            "YAQS Unitary Gauss",
            run_yaqs,
            exact_z_expvals,
            threshold_mse if run_density_matrix else None,
            init_circuit,
            trotter_step,
            num_qubits,
            num_layers,
            noise_model_unitary_gauss,
            None,
            z_initial,
            max_traj=1000,
            fixed_traj=None if run_density_matrix else fixed_trajectories
        )
        results["YAQS Unitary Gauss"] = {"trajectories": num_traj_gauss, "mse": mse_gauss, "z_expvals": z_expvals_gauss, "bonds": bonds_gauss}

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Define qubit indices to track
    first_qubit = 0
    middle_qubit = num_qubits // 2
    last_qubit = num_qubits - 1
    
    if run_density_matrix:
        # With exact reference: show MSE and speedup
        print(f"{'Method':<25} {'Trajectories':<15} {'Final MSE':<15} {'Speedup':<10}")
        print("-"*70)
        baseline_traj = results["YAQS Standard"]["trajectories"] if "YAQS Standard" in results else list(results.values())[0]["trajectories"]
        for method, data in results.items():
            speedup = baseline_traj / data["trajectories"]
            mse_str = f"{data['mse']:.2e}" if data['mse'] is not None else "N/A"
            print(f"{method:<25} {data['trajectories']:<15} {mse_str:<15} {speedup:<10.2f}x")
    else:
        # Without exact reference: just show trajectories and final values
        print(f"{'Method':<25} {'Trajectories':<15} {'Z_0 (final)':<15} {'Z_mid (final)':<15} {'Z_N (final)':<15}")
        print("-"*70)
        for method, data in results.items():
            z_exp = data["z_expvals"]  # shape: (num_qubits, num_layers+1)
            z0_final = f"{z_exp[first_qubit, -1]:.6f}" if z_exp is not None else "N/A"
            zmid_final = f"{z_exp[middle_qubit, -1]:.6f}" if z_exp is not None else "N/A"
            zN_final = f"{z_exp[last_qubit, -1]:.6f}" if z_exp is not None else "N/A"
            print(f"{method:<25} {data['trajectories']:<15} {z0_final:<15} {zmid_final:<15} {zN_final:<15}")
    
    print("="*70)

    # Create visualization with 2 rows of 3 subplots each
    # Top row: Z expectation values for first, middle, last qubit
    # Bottom row: Errors compared to exact for first, middle, last qubit
    fig = plt.figure(figsize=(20, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    method_names = list(results.keys())
    times = np.arange(num_layers + 1) * tau
    
    # Top row: Z expectation values
    ax1 = plt.subplot(2, 3, 1)  # First qubit
    ax2 = plt.subplot(2, 3, 2)  # Middle qubit
    ax3 = plt.subplot(2, 3, 3)  # Last qubit
    
    # Bottom row: Errors
    ax4 = plt.subplot(2, 3, 4)  # First qubit error
    ax5 = plt.subplot(2, 3, 5)  # Middle qubit error
    ax6 = plt.subplot(2, 3, 6)  # Last qubit error
    
    qubit_indices = [first_qubit, middle_qubit, last_qubit]
    qubit_labels = [f"Qubit 0", f"Qubit {middle_qubit} (middle)", f"Qubit {last_qubit} (last)"]
    expectval_axes = [ax1, ax2, ax3]
    error_axes = [ax4, ax5, ax6]
    
    # Plot Z expectation values for each qubit
    for qubit_idx, qubit_label, ax_exp, ax_err in zip(qubit_indices, qubit_labels, expectval_axes, error_axes):
        # Plot exact solution if available
        if run_density_matrix and exact_z_expvals is not None:
            ax_exp.plot(times, exact_z_expvals[qubit_idx, :], '-', 
                       label="Exact (Density Matrix)", 
                       alpha=1.0, linewidth=3, color='red', zorder=10)
        
        # Plot each method
        for i, (method, data) in enumerate(results.items()):
            z_exp = data["z_expvals"][qubit_idx, :]  # Time series for this qubit
            ax_exp.plot(times, z_exp, '-o', 
                       label=f"{method} ({data['trajectories']} traj)", 
                       alpha=0.7, markersize=3, color=colors[i])
            
            # Compute and plot errors if exact reference is available
            if run_density_matrix and exact_z_expvals is not None:
                error = np.abs(z_exp - exact_z_expvals[qubit_idx, :])
                ax_err.plot(times, error, '-o', 
                           label=f"{method}", 
                           alpha=0.7, markersize=3, color=colors[i])
        
        # Configure expectation value subplot
        ax_exp.set_xlabel("Time", fontsize=11)
        ax_exp.set_ylabel(r"$\langle Z \rangle$", fontsize=11)
        ax_exp.set_title(f"{qubit_label} - Z Expectation Value", fontsize=12)
        ax_exp.legend(fontsize=7, loc='best')
        ax_exp.grid(True, linestyle="--", alpha=0.5)
        
        # Configure error subplot
        if run_density_matrix:
            ax_err.set_xlabel("Time", fontsize=11)
            ax_err.set_ylabel(r"$|\Delta\langle Z \rangle|$", fontsize=11)
            ax_err.set_title(f"{qubit_label} - Absolute Error vs Exact", fontsize=12)
            ax_err.legend(fontsize=7, loc='best')
            ax_err.grid(True, linestyle="--", alpha=0.5)
            ax_err.set_yscale('log')
        else:
            # Hide error plots if no exact reference
            ax_err.text(0.5, 0.5, 'No exact reference\n(large system mode)', 
                       ha='center', va='center', transform=ax_err.transAxes, fontsize=12)
            ax_err.set_xticks([])
            ax_err.set_yticks([])
    
    plt.tight_layout()
    plt.savefig("local_z_expectation_values_comparison.png", dpi=300)
    plt.show()
    
    print("\nPlot saved as 'local_z_expectation_values_comparison.png'")

