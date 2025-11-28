print("import starts...")

import os
import sys
import pickle
import math
from typing import Optional
import numpy as np
# Delayed imports for heavy modules (qiskit, matplotlib, yaqs) occur in __main__

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
        tokens.append(f"mse{_format_float_short(threshold_mse)}")
    else:
        tokens.append(f"traj{fixed_trajectories}")
    if error_suffix:
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
    noise_model,
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

    Runs trajectories incrementally, accumulating results one at a time
    until threshold is met. This is much more efficient than re-running
    all trajectories from scratch each time.

    Args:
        method_name: Name of the method for logging
        simulator_func: Function to run simulation (run_yaqs or run_qiskit_mps)
        exact_stag: Exact staggered magnetization reference (None if unavailable)
        threshold_mse: Target MSE threshold (None if no exact reference)
        init_circuit: Initial state preparation circuit
        trotter_step: Single trotter step circuit
        num_qubits: Number of qubits
        num_layers: Number of trotter layers
        noise_model: YAQS noise model (None for Qiskit simulators)
        qiskit_noise_model: Qiskit noise model (None for YAQS simulators)
        stag_initial: Initial staggered magnetization value
        expvals_initial: Initial local expectation values (1d array of length num_qubits)
        observable_basis: Basis label ("Z", "X", or "Y")
        max_traj: Maximum trajectories to try (when searching for threshold)
        fixed_traj: If set, run exactly this many trajectories (for large systems)
        mse_metric: Which observable to compare against the threshold ("staggered" or "local")
        exact_local_expvals: Exact local expectation values (list of arrays), optional

    Returns:
        (num_trajectories_needed, final_mse, stag_series, local_expvals_series, bond_dims, variance_dict)
            - stag_series: list[float] of staggered magnetization over time (length num_layers + 1)
            - local_expvals_series: list[np.ndarray] of per-qubit expectation values over time
            - variance_dict: dict with entries such as "staggered" and "local_middle" containing variance series
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
        if method_name.startswith("YAQS"):
            expvals, bond_dims, var_array = simulator_func(
                init_circuit, trotter_step, num_qubits, num_layers,
                noise_model, num_traj=fixed_traj, parallel=True, observable_basis=observable_basis
            )
        else:
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
        if method_name.startswith("YAQS"):
            single_result, single_bonds, _ = simulator_func(
                init_circuit, trotter_step, num_qubits, num_layers, 
                noise_model, num_traj=1, parallel=True, observable_basis=observable_basis
            )
        else:  # Qiskit MPS
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
                # YAQS format: single trajectory result
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
    
    return target_traj, mse, stag_series, local_expvals_series, bond_dims, None


if __name__ == "__main__":
    # Simulation parameters
    num_qubits = 12
    num_layers = 40
    tau = 0.1

    staggedered_magn = False
    local_expectation = True

    # ========== MODE SELECTION ==========
    # For small systems: Set run_density_matrix=True and specify threshold_mse
    # For large systems: Set run_density_matrix=False and specify fixed_trajectories
    run_density_matrix = False  # Always run fixed trajectories (no density matrix reference)
    enable_qiskit_mps = True
    threshold_mse = 5e-3  # Target MSE threshold (only used if run_density_matrix=True)
    fixed_trajectories = 1000  # Number of trajectories for large systems (only used if run_density_matrix=False)
    noise_strengths = [0.1, 0.01, 0.001]
    observable_basis = "Z"  # Choose observable basis: "Z", "X", or "Y"

    # Noise channel configuration
    enable_tjm_x_error = False
    enable_tjm_y_error = True
    enable_tjm_z_error = False


    enable_qiskit_x_error = False
    enable_qiskit_y_error = True
    enable_qiskit_z_error = False
    # ====================================


    print("import of yaqs functions starts...")
    from codex_experiments.worker_functions.qiskit_simulators import run_qiskit_exact, run_qiskit_mps
    from codex_experiments.worker_functions.yaqs_simulator import run_yaqs, build_noise_models
    # Import circuit builders
    from core.libraries.circuit_library import qaoa_ising_layer, create_ising_circuit, create_heisenberg_circuit, xy_trotter_layer, xy_trotter_layer_longrange, create_clifford_cz_frame_circuit, create_echoed_xx_pi_over_2, create_sy_cz_parity_frame

    from core.libraries.circuit_library import create_2d_fermi_hubbard_circuit, create_1d_fermi_hubbard_circuit
    from core.libraries.circuit_library import create_2d_heisenberg_circuit, create_2d_ising_circuit, create_cz_brickwork_circuit, create_rzz_pi_over_2_brickwork
    print("import of qiskit functions starts...")
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
        # Try to make grid as square as possible das wichtigste ist eigentlich folgendes wo sind die Flex
        num_rows = int(math.sqrt(num_sites))
        while num_sites % num_rows != 0 and num_rows > 1:
            num_rows -= 1
        num_cols = num_sites // num_rows
        return create_2d_fermi_hubbard_circuit(Lx=num_cols, Ly=num_rows, u=1.0, t=0.5, mu=0.0, num_trotter_steps=1, dt=tau, timesteps=1)

    def make_2d_heisenberg_step():
        # For 2D Heisenberg, use a 2D grid layout
        # Try to make grid as square as possible
        num_rows = int(math.sqrt(num_qubits))
        while num_qubits % num_rows != 0 and num_rows > 1:
            num_rows -= 1
        num_cols = num_qubits // num_rows
        return create_2d_heisenberg_circuit(num_rows=num_rows, num_cols=num_cols, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0, dt=tau, timesteps=1)

    def make_2d_ising_step():
        # For 2D Ising, use a 2D grid layout
        # Try to make grid as square as possible
        num_rows = int(math.sqrt(num_qubits))
        while num_qubits % num_rows != 0 and num_rows > 1:
            num_rows -= 1
        num_cols = num_qubits // num_rows
        return create_2d_ising_circuit(num_rows=num_rows, num_cols=num_cols, J=1.0, g=0.5, dt=tau, timesteps=1)


    circuit_configs = [{"label": "XY_longrange", "builder": lambda: xy_trotter_layer_longrange(num_qubits, tau, order="YX")}]


    # Prepare base output directory for large-system runs
    base_dir = os.path.dirname(__file__)
    parent_dir = os.path.join(base_dir, "CTJM_interesting")
    os.makedirs(parent_dir, exist_ok=True)

    for noise_strength in noise_strengths:
        print("="*70)
        print("Trajectory Efficiency Comparison for Unraveling Methods")
        print("="*70)
        print(f"System: {num_qubits} qubits, {num_layers} layers")
        print(f"Noise strength: {noise_strength}")
        print(f"Observable basis: {observable_basis}")
        if run_density_matrix:
            print("Mode: With exact reference (density matrix)")
            print(f"Target MSE threshold: {threshold_mse:.2e}")
        else:
            print("Mode: No exact reference (large system)")
            print(f"Fixed trajectories per method: {fixed_trajectories}")
        print("="*70)

        for cfg in circuit_configs:
            basis_label = cfg["label"]
            print("-"*70)
            print(f"Running circuit: {basis_label} @ noise {noise_strength}")

            # Configure noise processes (YAQS). Change noise models here.
            processes = []
            if enable_tjm_x_error:
                processes.extend(
                    {"name": "pauli_x", "sites": [i], "strength": noise_strength}
                    for i in range(num_qubits)
                )
            if enable_tjm_y_error:
                processes.extend(
                    {"name": "pauli_y", "sites": [i], "strength": noise_strength}
                    for i in range(num_qubits)
                )
            if enable_tjm_z_error:
                processes.extend(
                    {"name": "pauli_z", "sites": [i], "strength": noise_strength}
                    for i in range(num_qubits)
                )
            if enable_tjm_x_error:
                processes.extend(
                    {"name": "crosstalk_xx", "sites": [i, i + 1], "strength": noise_strength}
                    for i in range(num_qubits - 1)
                )
            if enable_tjm_y_error:
                processes.extend(
                    {"name": "crosstalk_yy", "sites": [i, i + 1], "strength": noise_strength}
                    for i in range(num_qubits - 1)
                )
            if enable_tjm_z_error:
                processes.extend(
                    {"name": "crosstalk_zz", "sites": [i, i + 1], "strength": noise_strength}
                    for i in range(num_qubits - 1)
                )

            error_channels = set()
            if enable_tjm_x_error or enable_qiskit_x_error:
                error_channels.add("X")
            if enable_tjm_y_error or enable_qiskit_y_error:
                error_channels.add("Y")
            if enable_tjm_z_error or enable_qiskit_z_error:
                error_channels.add("Z")
            error_suffix = "".join(sorted(error_channels)) if error_channels else "None"

            # Skip if this experiment has already completed (all outputs exist)
            experiment_name = _build_experiment_name(
                num_qubits,
                num_layers,
                tau,
                noise_strength,
                run_density_matrix,
                threshold_mse,
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

            noise_model_normal = build_noise_models(processes)

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
            # For Z basis: initial state has X gates on qubits where i % 4 == 3, giving Z = -1
            # For X basis: same initial state preparation gives X expectation = 0 (need H gates for definite X eigenstates)
            # For simplicity, we compute the initial expectation values from the initial state
            # The initial state preparation uses X gates, so for Z basis: |1⟩ → Z = -1, |0⟩ → Z = +1
            if observable_basis == "Z":
                expvals_initial = np.array([1.0 if i % 4 != 3 else -1.0 for i in range(num_qubits)])
            elif observable_basis == "X":
                # X gates on |0⟩ create |1⟩, which has X expectation = 0
                # To get definite X eigenstates, we'd need H gates, but for now use 0
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
            if run_density_matrix:
                print("Finding minimum trajectories for each method...")
            else:
                print(f"Running {fixed_trajectories} trajectories for each method...")
            print("-" * 70)

            # Qiskit MPS
            if enable_qiskit_mps:
                print("\n1. Qiskit MPS (Standard Unraveling)")
                num_traj_mps, mse_mps, stag_mps, local_mps, bonds_mps, var_mps = find_required_trajectories(
                    "Qiskit MPS",
                    run_qiskit_mps,
                    exact_stag,
                    threshold_mse if run_density_matrix else None,
                    init_circuit,
                    trotter_step,
                    num_qubits,
                    num_layers,
                    None,
                    qiskit_noise_model,
                    stag_initial,
                    expvals_initial,
                    observable_basis,
                    max_traj=1000,
                    fixed_traj=None if run_density_matrix else fixed_trajectories,
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

