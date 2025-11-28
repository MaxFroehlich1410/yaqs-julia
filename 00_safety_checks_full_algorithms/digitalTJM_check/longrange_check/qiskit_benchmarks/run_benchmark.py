print("import starts...")

import os
import sys
import pickle
import math
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# Ensure local modules can be imported
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

def _format_float_short(value: float) -> str:
    """Format floats compactly for filenames (e.g., 0.1 -> 0p1)."""
    return f"{value:.4g}".replace('.', 'p')

def _build_experiment_name(
    num_qubits: int,
    num_layers: int,
    tau: float,
    noise_strength: float,
    run_density_matrix: bool,
    threshold_mse: float,
    fixed_trajectories: int,
    basis_label: str,
    observable_basis: str = "Z",
    error_suffix: Optional[str] = None,
) -> str:
    tokens = [
        "unraveling_eff",
        f"N{num_qubits}",
        f"L{num_layers}",
        f"tau{_format_float_short(tau)}",
        f"noise{_format_float_short(noise_strength)}",
        f"basis{basis_label}",
        f"obs{observable_basis}",
        "modeDM" if run_density_matrix else "modeLarge",
    ]
    if run_density_matrix:
        # In simple mode, we just append trajectories if we are running parallel
        tokens.append(f"traj{fixed_trajectories}")
    else:
        tokens.append(f"traj{fixed_trajectories}")
    
    if error_suffix and error_suffix != "None":
        tokens.append(f"err{error_suffix}")
        
    return "_".join(tokens)

def staggered_magnetization(expvals, num_qubits):
    """Compute staggered magnetization from expectation values (works for Z, X, or Y basis)."""
    return np.sum([(-1)**i * expvals[i] for i in range(num_qubits)]) / num_qubits 

def compute_mse(pred, exact):
    """Compute Mean Squared Error between prediction and exact solution (for staggered magnetization)."""
    return np.mean([(pred[i] - exact[i])**2 for i in range(len(pred))])


def compute_mse_local_expvals(pred_expvals, exact_expvals):
    """Compute MSE over all local expectation values at all time steps.
    
    Args:
        pred_expvals: List of expectation value arrays [expvals_t0, expvals_t1, ...], each shape (num_qubits,)
        exact_expvals: List of exact expectation value arrays
    
    Returns:
        float: Mean squared error averaged over all qubits and all time steps
    """
    mse_total = 0.0
    count = 0
    for pred_t, exact_t in zip(pred_expvals, exact_expvals):
        mse_total += np.sum((pred_t - exact_t)**2) 
        count += len(pred_t)
    return mse_total / count if count > 0 else 0.0


def find_required_trajectories(
    method_name,
    simulator_func,
    exact_stag,
    threshold_mse,
    init_circuit,
    trotter_step,
    num_qubits,
    num_layers,
    qiskit_noise_model,
    stag_initial,
    expvals_initial,
    observable_basis="Z",
    max_traj=1000,
    fixed_traj=None,
    mse_metric="staggered",
    exact_local_expvals=None,
):
    """
    Find minimum number of trajectories needed to achieve target MSE while tracking
    both staggered magnetization and local expectation values.
    """
    # Determine how many trajectories to run
    metric = mse_metric.lower()
    if metric not in {"staggered", "local"}:
        raise ValueError(f"Unsupported mse_metric '{mse_metric}'. Expected 'staggered' or 'local'.")

    use_mse_threshold = False
    if threshold_mse is not None:
        if metric == "staggered" and exact_stag is not None:
            use_mse_threshold = True
        elif metric == "local" and exact_local_expvals is not None:
            use_mse_threshold = True
    target_traj = fixed_traj if fixed_traj is not None else max_traj

    expvals_initial = np.asarray(expvals_initial, dtype=float)
    
    # If we're running a fixed number of trajectories (large system mode),
    # run them all in parallel for efficiency
    if fixed_traj is not None and not use_mse_threshold:
        print(f"  {method_name}: Running {fixed_traj} trajectories in parallel...")

        # Single parallel run collects mean expvals and per-qubit per-timestep variance across trajectories
        expvals, bond_dims, var_array = simulator_func(
            num_qubits, num_layers, init_circuit, trotter_step,
            qiskit_noise_model, num_traj=fixed_traj, observable_basis=observable_basis
        )

        # Compute observable means
        stag_series = [stag_initial] + [
            staggered_magnetization(expvals[:, t], num_qubits) for t in range(num_layers)
        ]
        local_expvals_series = [expvals_initial.copy()] + [
            expvals[:, t].copy() for t in range(num_layers)
        ]

        # Approximate variances
        variance_dict = {}
        if var_array is not None:
            variance_dict["staggered"] = (np.sum(var_array, axis=0) / (num_qubits ** 2)).astype(float)
            middle_qubit = num_qubits // 2
            variance_dict["local_middle"] = var_array[middle_qubit, :].astype(float)
        else:
            variance_dict = None

        print(f"  ✓ {method_name}: Completed {fixed_traj} trajectory(ies) in parallel")
        return fixed_traj, None, stag_series, local_expvals_series, bond_dims, variance_dict
    
    # Otherwise, run trajectories incrementally (small system mode with MSE checking)
    cumulative_results = None  # Will store sum of z-expectation values
    bond_dims_list = []  # Collect bond dims from each trajectory
    
    for num_traj in range(1, target_traj + 1):
        # Run a single trajectory
        single_result, single_bonds, _ = simulator_func(
            num_qubits, num_layers, init_circuit, trotter_step,
            qiskit_noise_model, num_traj=1, observable_basis=observable_basis
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
                # Fallback format
                if single_bonds.ndim == 2:
                    bond_dims_list.append(single_bonds[0])
                else:
                    bond_dims_list.append(single_bonds)
        
        # Compute average over all trajectories so far
        avg_results = cumulative_results / num_traj
        
        # Compute observables (staggered magnetization and local expectation values)
        stag_series = [stag_initial] + [
            staggered_magnetization(avg_results[:, t], num_qubits) for t in range(num_layers)
        ]
        local_expvals_series = [expvals_initial.copy()] + [
            avg_results[:, t].copy() for t in range(num_layers)
        ]
        
        # Compute MSE if exact reference is available
        if use_mse_threshold:
            if metric == "staggered":
                mse = compute_mse(stag_series, exact_stag)
            else:
                mse = compute_mse_local_expvals(local_expvals_series, exact_local_expvals)
            print(f"  {method_name}: Trajectory {num_traj}: MSE = {mse:.6e} (threshold = {threshold_mse:.6e})")
            
            # Check if threshold is met
            if mse < threshold_mse:
                print(f"  ✓ {method_name}: Target reached with {num_traj} trajectory(ies)!")
                
                # Format bond dimensions for return
                if len(bond_dims_list) > 0:
                    bond_array = np.array(bond_dims_list)
                    bond_dims = {
                        "per_shot_per_layer_max_bond_dim": bond_array,
                        "per_layer_mean_across_shots": np.mean(bond_array, axis=0)
                    }
                else:
                    bond_dims = None
                
                # No variance returned in threshold mode
                return num_traj, mse, stag_series, local_expvals_series, bond_dims, None
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
        bond_array = np.array(bond_dims_list)
        bond_dims = {
            "per_shot_per_layer_max_bond_dim": bond_array,
            "per_layer_mean_across_shots": np.mean(bond_array, axis=0)
        }
    else:
        bond_dims = None
    
    return target_traj, mse, stag_series, local_expvals_series, bond_dims, None


if __name__ == "__main__":

    # Parameters
    num_qubits = 8
    num_layers = 20
    tau = 0.1
    noise_strengths = [0.1]
    observable_basis = "Z"
    
    # Toggle Methods
    enable_qiskit_mps = True
    run_density_matrix = True

    # Mode-specific settings
    # 1. FIXED Mode
    fixed_trajectories = 500
     # If True, computes density matrix ref even in FIXED mode

    # Internal logic setup
    # specific args for finding trajectories
    arg_fixed_traj = fixed_trajectories
    arg_max_traj = fixed_trajectories
    print(f"Mode: FIXED ({fixed_trajectories} trajectories, Parallel)")

    staggedered_magn = False
    local_expectation = True
    
    # Noise channel configuration
    enable_qiskit_x_error = True
    enable_qiskit_y_error = False
    enable_qiskit_z_error = False
    # ====================================


    print("import of qiskit functions starts...")
    from qiskit_simulators import run_qiskit_exact, run_qiskit_mps
    
    # Import circuit builders
    from circuit_library import (
        qaoa_ising_layer, 
        create_ising_circuit, 
        create_heisenberg_circuit, 
        xy_trotter_layer, 
        xy_trotter_layer_longrange, 
        create_clifford_cz_frame_circuit, 
        create_echoed_xx_pi_over_2, 
        create_sy_cz_parity_frame,
        create_2d_fermi_hubbard_circuit, 
        create_1d_fermi_hubbard_circuit,
        create_2d_heisenberg_circuit, 
        create_2d_ising_circuit, 
        create_cz_brickwork_circuit, 
        create_rzz_pi_over_2_brickwork
    )

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli
    from qiskit_aer.noise.errors import PauliLindbladError
    from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
    print("imports done")

    # Define circuit configurations
    def make_ising_step(periodic: bool):
        return create_ising_circuit(L=num_qubits, J=1.0, g=0.5, dt=tau, timesteps=1, periodic=periodic)

    def make_heisenberg_step(periodic: bool):
        return create_heisenberg_circuit(L=num_qubits, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0, dt=tau, timesteps=1, periodic=periodic)

    def make_1d_fermi_hubbard_step():
        # Fermi-Hubbard uses 2 qubits per site (spin up and spin down)
        L = num_qubits // 2
        return create_1d_fermi_hubbard_circuit(L=L, u=1.0, t=0.5, mu=0.0, num_trotter_steps=1, dt=tau, timesteps=1)

    def make_2d_fermi_hubbard_step():
        # For 2D Fermi-Hubbard, use a 2D grid layout
        # Fermi-Hubbard uses 2 qubits per site (spin up and spin down)
        num_sites = num_qubits // 2
        num_rows = int(math.sqrt(num_sites))
        while num_sites % num_rows != 0 and num_rows > 1:
            num_rows -= 1
        num_cols = num_sites // num_rows
        return create_2d_fermi_hubbard_circuit(Lx=num_cols, Ly=num_rows, u=1.0, t=0.5, mu=0.0, num_trotter_steps=1, dt=tau, timesteps=1)

    def make_2d_heisenberg_step():
        # For 2D Heisenberg, use a 2D grid layout
        num_rows = int(math.sqrt(num_qubits))
        while num_qubits % num_rows != 0 and num_rows > 1:
            num_rows -= 1
        num_cols = num_qubits // num_rows
        return create_2d_heisenberg_circuit(num_rows=num_rows, num_cols=num_cols, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0, dt=tau, timesteps=1)

    def make_2d_ising_step():
        # For 2D Ising, use a 2D grid layout
        num_rows = int(math.sqrt(num_qubits))
        while num_qubits % num_rows != 0 and num_rows > 1:
            num_rows -= 1
        num_cols = num_qubits // num_rows
        return create_2d_ising_circuit(num_rows=num_rows, num_cols=num_cols, J=1.0, g=0.5, dt=tau, timesteps=1)


    circuit_configs = [{"label": "XY_longrange", "builder": lambda: xy_trotter_layer_longrange(num_qubits, tau, order="YX")}]


    # Prepare base output directory for large-system runs
    base_dir = os.path.dirname(__file__)
    # Save to ../results relative to this script
    parent_dir = os.path.join(base_dir, "../results")
    os.makedirs(parent_dir, exist_ok=True)

    for noise_strength in noise_strengths:
        print("="*70)
        print("Trajectory Efficiency Comparison for Unraveling Methods (Qiskit Standalone)")
        print("="*70)
        print(f"System: {num_qubits} qubits, {num_layers} layers")
        print(f"Noise strength: {noise_strength}")
        print(f"Observable basis: {observable_basis}")
        if run_density_matrix:
            print("Mode: With exact reference (density matrix)")
        else:
            print("Mode: No exact reference (large system)")
            print(f"Fixed trajectories per method: {fixed_trajectories}")
        print("="*70)

        for cfg in circuit_configs:
            basis_label = cfg["label"]
            print("-"*70)
            print(f"Running circuit: {basis_label} @ noise {noise_strength}")

            error_channels = set()
            if enable_qiskit_x_error:
                error_channels.add("X")
            if enable_qiskit_y_error:
                error_channels.add("Y")
            if enable_qiskit_z_error:
                error_channels.add("Z")
            error_suffix = "".join(sorted(error_channels)) if error_channels else "None"

            # Skip if this experiment has already completed (all outputs exist)
            experiment_name = _build_experiment_name(
                num_qubits,
                num_layers,
                tau,
                noise_strength,
                run_density_matrix,
                None, # threshold_mse not used
                fixed_trajectories,
                basis_label,
                observable_basis,
                error_suffix=error_suffix,
            )


            # Prepare initial state circuit
            init_circuit = QuantumCircuit(num_qubits)
            for i in range(num_qubits):
                if i % 4 == 3:
                    init_circuit.x(i)

            # Build one Trotter step for this circuit
            trotter_step = cfg["builder"]()

            # Initialize Qiskit noise model
            qiskit_noise_model = QiskitNoiseModel()
            TwoQubit_XX_error = PauliLindbladError(
                [Pauli("IX"), Pauli("XI"), Pauli("XX")],
                [noise_strength, noise_strength, noise_strength]
            )
            TwoQubit_YY_error = PauliLindbladError(
                [Pauli("IY"), Pauli("YI"), Pauli("YY")],
                [noise_strength, noise_strength, noise_strength]
            )
            TwoQubit_ZZ_error = PauliLindbladError(
                [Pauli("IZ"), Pauli("ZI"), Pauli("ZZ")],
                [noise_strength, noise_strength, noise_strength]
            )
            if enable_qiskit_x_error:
                for qubit in range(num_qubits):
                    next_qubit = (qubit + 1) % num_qubits
                    qiskit_noise_model.add_quantum_error(
                        TwoQubit_XX_error,
                        ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx", "cp"],
                        [qubit, next_qubit]
                    )
            if enable_qiskit_y_error:
                for qubit in range(num_qubits):
                    next_qubit = (qubit + 1) % num_qubits
                    qiskit_noise_model.add_quantum_error(
                        TwoQubit_YY_error,
                        ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx", "cp"],
                        [qubit, next_qubit]
                    )
            if enable_qiskit_z_error:
                for qubit in range(num_qubits):
                    next_qubit = (qubit + 1) % num_qubits
                    qiskit_noise_model.add_quantum_error(
                        TwoQubit_ZZ_error,
                        ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx", "cp"],
                        [qubit, next_qubit]
                    )

            # Compute initial expectation values and staggered magnetization
            if observable_basis == "Z":
                expvals_initial = np.array([1.0 if i % 4 != 3 else -1.0 for i in range(num_qubits)])
            elif observable_basis == "X":
                expvals_initial = np.zeros(num_qubits)
            else:  # Y basis
                expvals_initial = np.zeros(num_qubits)
            stag_initial = staggered_magnetization(expvals_initial, num_qubits)

            # Run exact density matrix simulation (reference) if enabled
            if run_density_matrix:
                print(f"\nRunning exact density matrix simulation (reference) with {observable_basis} basis...")
                expvals_exact = run_qiskit_exact(
                    num_qubits, num_layers, init_circuit, trotter_step,
                    qiskit_noise_model, method="density_matrix", observable_basis=observable_basis
                )
                
                # Prepare reference data for both observables
                exact_stag = [
                    stag_initial
                ] + [
                    staggered_magnetization(expvals_exact[:, t], num_qubits)
                    for t in range(num_layers)
                ]
                exact_local_expvals = [expvals_initial] + [expvals_exact[:, t] for t in range(num_layers)]
                
                print("Exact reference computed.\n")
            else:
                print("\nSkipping exact density matrix simulation (large system mode).\n")
                exact_stag = None
                exact_local_expvals = None

    # Test each method
            results = {}
            
            if exact_stag is not None:
                results["Exact"] = {
                    "staggered_magnetization": exact_stag,
                    "local_expvals": exact_local_expvals
                }

            print(f"Running {fixed_trajectories} trajectories for each method...")
            print("-" * 70)

            # Qiskit MPS
            if enable_qiskit_mps:
                print("\n1. Qiskit MPS (Standard Unraveling)")
                num_traj_mps, mse_mps, stag_mps, local_mps, bonds_mps, var_mps = find_required_trajectories(
                    "Qiskit MPS",
                    run_qiskit_mps,
                    exact_stag,
                    None, # threshold_mse not used
                    init_circuit,
                    trotter_step,
                    num_qubits,
                    num_layers,
                    qiskit_noise_model,
                    stag_initial,
                    expvals_initial,
                    observable_basis,
                    max_traj=fixed_trajectories,
                    fixed_traj=fixed_trajectories,
                    mse_metric="staggered" if staggedered_magn else "local",
                    exact_local_expvals=exact_local_expvals,
                )
                results["Qiskit MPS"] = {
                    "trajectories": num_traj_mps,
                    "mse": mse_mps,
                    "staggered_magnetization": stag_mps,
                    "local_expvals": local_mps,
                    "bonds": bonds_mps,
                    "variance": var_mps,
                }
            
            # Save results
            filename = f"{experiment_name}_results.pkl"
            filepath = os.path.join(parent_dir, filename)
            with open(filepath, "wb") as f:
                pickle.dump(results, f)
            print(f"Results saved to {filepath}")

