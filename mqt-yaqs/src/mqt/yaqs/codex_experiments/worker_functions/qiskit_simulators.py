
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit_aer.noise.errors import PauliLindbladError


@dataclass
class BenchmarkConfig:
    num_qubits: int
    num_layers: int
    shots: int
    seed: Optional[int] = None

def collect_expectations_and_mps_bond_dims(
    init_circuit: QuantumCircuit,
    trotter_step: QuantumCircuit,
    num_qubits: int,
    num_layers: int,
    noise_model: QiskitNoiseModel,
    *,
    shots: int = 1024,
    seed: Optional[int] = None,
    method: str = "matrix_product_state",
    include_initial: bool = True,
    mps_options: Optional[Dict[str, Any]] = None,
    observable_basis: str = "Z",
    track_bond_dims: bool = True,
    track_variance: bool = True,
) -> Dict[str, Any]:
    """
    Collect expectation values and optionally MPS bond-dimension stats / variances.

    Args:
        track_bond_dims: If True, saves MPS state per shot per layer (Heavy memory usage!).
        track_variance: If True, saves expectation values per shot (Moderate memory usage).
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be >= 1")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be >= 1")
    if observable_basis not in ["Z", "X", "Y"]:
        raise ValueError(f"observable_basis must be one of ['Z', 'X', 'Y'], got {observable_basis}")

    from qiskit.quantum_info import Pauli
    from qiskit_aer.library import SetMatrixProductState

    # Prepare storage
    T = num_layers + 1 if include_initial else num_layers
    expvals_mean = np.zeros((num_qubits, T), dtype=float)
    expvals_var = np.zeros((num_qubits, T), dtype=float)
    
    # Store accumulated sums for mean and variance
    # We will compute online variance: M2 = sum((x - mean)^2) ?? 
    # Or just store sum(x) and sum(x^2).
    # sum_x[q, t]
    # sum_sq_x[q, t]
    sum_x = np.zeros((num_qubits, T), dtype=float)
    sum_sq_x = np.zeros((num_qubits, T), dtype=float)
    
    # Bond dims: store sum and max
    # dims[layer]
    bond_dim_sum = np.zeros(num_layers, dtype=float)
    bond_dim_max = np.zeros(num_layers, dtype=int)
    # Store simplified dims matrix to satisfy return type: (shots, layers)
    dims_matrix = np.ones((shots, num_layers), dtype=int)
    
    # overall max
    overall_max_per_shot = np.zeros(shots, dtype=int)
    
    # If shots > 1, we must run sequentially to save memory for large systems
    # For small systems, we could batch, but let's prioritize robustness for now.
    
    sim = AerSimulator(method=method, noise_model=noise_model)
    if mps_options is None:
        mps_options = {}
    # Ensure reasonable truncation to prevent explosion
    if "matrix_product_state_truncation_threshold" not in mps_options:
        mps_options["matrix_product_state_truncation_threshold"] = 1e-16
    sim.set_options(**mps_options)

    basis_lower = observable_basis.lower()
    pauli_op = Pauli(observable_basis)

    for shot_idx in range(shots):
        # 1. Run Initialization
        qc_init = init_circuit.copy()
        
        # Save t=0 stats
        if include_initial:
            for q in range(num_qubits):
                qc_init.save_expectation_value(pauli_op, qubits=[q], label=f"obs_q{q}", pershot=False)  # type: ignore[attr-defined]
        
        # If we need to continue, we must save the state
        qc_init.save_matrix_product_state(label="mps_state", pershot=False)  # type: ignore[attr-defined]
        
        res_init = sim.run(qc_init, shots=1).result()
        
        if include_initial:
            for q in range(num_qubits):
                val = float(res_init.data(0)[f"obs_q{q}"])
                sum_x[q, 0] += val
                sum_sq_x[q, 0] += val**2
        
        # Get MPS state to feed into next layer
        current_mps_state = res_init.data(0)["mps_state"]
        
        # 2. Loop Layers
        shot_max_bond = 0
        
        for t in range(num_layers):
            qc_step = QuantumCircuit(num_qubits)
            qc_step.append(SetMatrixProductState(current_mps_state), range(num_qubits))  # type: ignore[arg-type]
            qc_step.compose(trotter_step, inplace=True)
            
            # Measurements
            for q in range(num_qubits):
                qc_step.save_expectation_value(pauli_op, qubits=[q], label=f"obs_q{q}", pershot=False)  # type: ignore[attr-defined]
            
            # Save state for next step and bond dim check
            qc_step.save_matrix_product_state(label="mps_state", pershot=False)  # type: ignore[attr-defined]
            
            res_step = sim.run(qc_step, shots=1).result()
            
            # Record Observables
            time_idx = (t + 1) if include_initial else t
            for q in range(num_qubits):
                val = float(res_step.data(0)[f"obs_q{q}"])
                sum_x[q, time_idx] += val
                sum_sq_x[q, time_idx] += val**2
                
            # Record Bond Dim
            current_mps_state = res_step.data(0)["mps_state"]
            
            # Default to 1 if not tracking, but we still need state for next step
            # If track_bond_dims is False, we just take 1
            b_dim = 1
            if track_bond_dims:
                b_dim = _extract_max_bond_dim_from_saved_mps_state(current_mps_state)
            
            dims_matrix[shot_idx, t] = b_dim
            bond_dim_sum[t] += b_dim
            bond_dim_max[t] = max(bond_dim_max[t], b_dim)
            shot_max_bond = max(shot_max_bond, b_dim)
        
        overall_max_per_shot[shot_idx] = shot_max_bond

    # Compute Statistics
    expvals_mean = sum_x / shots
    # Var[x] = E[x^2] - (E[x])^2.
    # But wait, sum_sq_x is sum of squares of EXPECTATION values (which are means of 1 shot).
    # Since shots=1 per run, the "expectation value" returned is just the measurement outcome (eigenvalue) 
    # or the exact expectation w.r.t the trajectory state?
    # With Aer noisy simulation + save_expectation_value on 1 shot, it returns <psi|P|psi>.
    # So yes, it's the quantum expectation of that trajectory.
    # The variance across trajectories is correct.
    expvals_var = (sum_sq_x / shots) - (expvals_mean ** 2)
    
    # Bond stats
    per_layer_mean = bond_dim_sum / shots
    
    bonds = {
        "per_shot_per_layer_max_bond_dim": dims_matrix,
        "per_layer_mean_across_shots": per_layer_mean,
        "per_layer_max_across_shots": bond_dim_max,
        "overall_max_per_shot": overall_max_per_shot,
        "labels": [_save_label(t) for t in range(num_layers)],
        "shots": shots,
        "num_qubits": num_qubits,
        "num_layers": num_layers,
    }
    
    return {"expvals": expvals_mean, "expvals_var": expvals_var, "bonds": bonds, "observable_basis": observable_basis}




def run_qiskit_exact(num_qubits: int, num_layers: int, init_circuit, trotter_step, noise_model: Optional[QiskitNoiseModel], method = "density_matrix", observable_basis: str = "Z") -> np.ndarray:
    from .qiskit_noisy_sim import qiskit_noisy_simulator

    baseline = [[] for _ in range(num_qubits)]
    # Start from i=1 to apply at least 1 Trotter step (initial state is handled separately)
    for i in range(1, num_layers + 1):
        qc = init_circuit.copy()
        for _ in range(i):
            qc = qc.compose(trotter_step)
        vals = np.real(np.asarray(qiskit_noisy_simulator(qc, noise_model, num_qubits, 1, method=method, observable_basis=observable_basis))).flatten()
        for q in range(num_qubits):
            baseline[q].append(float(vals[q]))
    arr = np.stack([np.asarray(b) for b in baseline])
    return arr


def run_qiskit_mps(
    num_qubits: int,
    num_layers: int,
    init_circuit: QuantumCircuit,
    trotter_step: QuantumCircuit,
    noise_model: QiskitNoiseModel,
    *,
    num_traj: int = 1024,
    seed: Optional[int] = None,
    method: str = "matrix_product_state",
    mps_options: Optional[Dict[str, Any]] = None,
    observable_basis: str = "Z",
    track_bond_dims: Optional[bool] = None,
    track_variance: Optional[bool] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Run Qiskit MPS on a layered circuit and return expectation values and bond stats.
    
    If track_bond_dims is None, it defaults to False if num_qubits > 30 to avoid memory errors.
    If track_variance is None, it defaults to True.
    """
    
    # Auto-disable expensive tracking for large systems to avoid 2TB memory error
    if track_bond_dims is None:
        track_bond_dims = (num_qubits <= 30)
        
    if track_variance is None:
        track_variance = True

    combined = collect_expectations_and_mps_bond_dims(
        init_circuit,
        trotter_step,
        num_qubits,
        num_layers,
        noise_model,
        shots=num_traj,
        seed=seed,
        method=method,
        include_initial=True,
        mps_options=mps_options,
        observable_basis=observable_basis,
        track_bond_dims=track_bond_dims,
        track_variance=track_variance,
    )
    expvals = combined["expvals"][:, 1:]
    # include initial variance at t=0 for the variance plot
    expvals_var = combined.get("expvals_var", np.zeros_like(combined["expvals"]))
    bonds = combined["bonds"]
    return expvals, bonds, expvals_var




def _save_label(layer_index: int) -> str:
    return f"mps_t{layer_index+1}"




def _extract_max_bond_dim_from_saved_mps_state(state_shot: Any) -> int:
    """
    Given one "shot" of a saved MPS state (tuple of (site_tensors, lambdas)),
    return the maximum bond dimension across all bonds.
    """
    try:
        _, lambdas = state_shot
    except Exception:
        # Unexpected structure → treat as product state
        return 1
    if not isinstance(lambdas, (list, tuple)) or len(lambdas) == 0:
        return 1
    try:
        return max((len(vec) for vec in lambdas), default=1)
    except Exception:
        # Fallback if elements are not sized arrays
        vals = []
        for vec in lambdas:
            try:
                vals.append(len(vec))
            except Exception:
                vals.append(1)
        return max(vals) if vals else 1
