
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
    """Store benchmark configuration for Qiskit simulations.

    This holds basic circuit sizing and sampling parameters used across Qiskit-based benchmarks.
    It can be passed between helper functions to keep simulation settings consistent.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_layers (int): Number of circuit layers.
        shots (int): Number of shots to simulate.
        seed (Optional[int]): Optional RNG seed for reproducibility.

    Returns:
        BenchmarkConfig: Configuration container with the provided values.
    """
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
) -> Dict[str, Any]:
    """Collect expectation values and MPS bond-dimension stats in one Aer run.

    This builds a layered circuit, inserts per-shot expectation saves and per-shot MPS saves after
    each layer, then runs a single Aer simulation and parses both outputs.

    Args:
        init_circuit (QuantumCircuit): Circuit that prepares the initial state.
        trotter_step (QuantumCircuit): Circuit for one Trotter step.
        num_qubits (int): Number of qubits in the circuit.
        num_layers (int): Number of layers to simulate.
        noise_model (QiskitNoiseModel): Qiskit noise model to apply.
        shots (int): Number of shots to simulate.
        seed (Optional[int]): RNG seed for reproducibility.
        method (str): Aer simulation method.
        include_initial (bool): Whether to include the initial state as t=0.
        mps_options (Optional[Dict[str, Any]]): Aer MPS simulator options.
        observable_basis (str): Pauli basis to measure ("Z", "X", or "Y").

    Returns:
        Dict[str, Any]: Dictionary with expectation values, variances, and bond statistics.

    Raises:
        ValueError: If `num_layers`, `num_qubits`, or `observable_basis` are invalid.
        RuntimeError: If per-shot MPS data is missing or malformed.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be >= 1")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be >= 1")
    if observable_basis not in ["Z", "X", "Y"]:
        raise ValueError(f"observable_basis must be one of ['Z', 'X', 'Y'], got {observable_basis}")

    from qiskit.quantum_info import Pauli

    # Build a single circuit: init once, then repeat Trotter steps
    qc = init_circuit.copy()
    pauli_op = Pauli(observable_basis)
    if include_initial:
        for q in range(num_qubits):
            # Save initial expval per-shot for stochastic variance
            qc.save_expectation_value(pauli_op, qubits=[q], label=f"{observable_basis.lower()}_q{q}_t0", pershot=True)  # type: ignore[attr-defined]
    for t in range(num_layers):
        composed = qc.compose(trotter_step, qubits=range(num_qubits))
        assert composed is not None
        qc = composed
        # Save Pauli observable on all qubits
        for q in range(num_qubits):
            qc.save_expectation_value(pauli_op, qubits=[q], label=f"{observable_basis.lower()}_q{q}_t{t+1}", pershot=True)  # type: ignore[attr-defined]
        # Save per-shot MPS to extract bond dims
        qc.save_matrix_product_state(pershot=True, label=_save_label(t))  # type: ignore[attr-defined]

    sim = AerSimulator(method=method, noise_model=noise_model)
    if mps_options:
        sim.set_options(**mps_options)
    result = sim.run(qc, shots=shots, seed_simulator=seed).result()

    # Parse expectations (pershot lists) and compute means/variances
    T = num_layers + 1 if include_initial else num_layers
    expvals_mean = np.zeros((num_qubits, T), dtype=float)
    expvals_var = np.zeros((num_qubits, T), dtype=float)
    col = 0
    basis_lower = observable_basis.lower()
    if include_initial:
        for q in range(num_qubits):
            vals = np.asarray(result.data(0)[f"{basis_lower}_q{q}_t0"], dtype=float)
            expvals_mean[q, 0] = float(np.mean(vals))
            expvals_var[q, 0] = float(np.var(vals))
        col = 1
    for t in range(num_layers):
        for q in range(num_qubits):
            vals = np.asarray(result.data(0)[f"{basis_lower}_q{q}_t{t+1}"], dtype=float)
            expvals_mean[q, col + t] = float(np.mean(vals))
            expvals_var[q, col + t] = float(np.var(vals))

    # Parse MPS snapshots → per-shot bond dims per layer
    labels: List[str] = [_save_label(t) for t in range(num_layers)]
    dims_per_shot_per_layer: List[List[int]] = [[1 for _ in range(num_layers)] for _ in range(shots)]
    for layer_idx, label in enumerate(labels):
        mps_list = result.data(0)[label]
        if not isinstance(mps_list, (list, tuple)) or len(mps_list) != shots:
            raise RuntimeError(f"Expected per-shot MPS list for {label} of length {shots}")
        for shot_idx, state_shot in enumerate(mps_list):
            dims_per_shot_per_layer[shot_idx][layer_idx] = _extract_max_bond_dim_from_saved_mps_state(state_shot)

    dims_array = np.asarray(dims_per_shot_per_layer, dtype=int)
    per_layer_mean = dims_array.mean(axis=0)
    per_layer_max = dims_array.max(axis=0)
    overall_max_per_shot = dims_array.max(axis=1)

    bonds = {
        "per_shot_per_layer_max_bond_dim": dims_array,
        "per_layer_mean_across_shots": per_layer_mean,
        "per_layer_max_across_shots": per_layer_max,
        "overall_max_per_shot": overall_max_per_shot,
        "labels": labels,
        "shots": shots,
        "num_qubits": num_qubits,
        "num_layers": num_layers,
    }
    return {"expvals": expvals_mean, "expvals_var": expvals_var, "bonds": bonds, "observable_basis": observable_basis}




def run_qiskit_exact(num_qubits: int, num_layers: int, init_circuit, trotter_step, noise_model: Optional[QiskitNoiseModel], method = "density_matrix", observable_basis: str = "Z") -> np.ndarray:
    """Run an exact Qiskit simulation for layered circuits.

    This composes the initial circuit with 1..N Trotter steps and evaluates the observable basis
    at each layer using the exact simulator backend.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_layers (int): Number of layers to simulate.
        init_circuit (QuantumCircuit): Circuit that prepares the initial state.
        trotter_step (QuantumCircuit): Circuit for one Trotter step.
        noise_model (Optional[QiskitNoiseModel]): Noise model to apply.
        method (str): Aer simulation method (e.g., "density_matrix").
        observable_basis (str): Pauli basis to measure ("Z", "X", or "Y").

    Returns:
        np.ndarray: Expectation values of shape (num_qubits, num_layers).
    """
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
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Run Qiskit MPS simulation and return expectations and bond stats.

    This delegates to the combined collection routine and returns per-layer expectation values,
    bond statistics, and per-layer variances.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_layers (int): Number of layers to simulate.
        init_circuit (QuantumCircuit): Circuit that prepares the initial state.
        trotter_step (QuantumCircuit): Circuit for one Trotter step.
        noise_model (QiskitNoiseModel): Noise model to apply.
        num_traj (int): Number of trajectories/shots to simulate.
        seed (Optional[int]): RNG seed for reproducibility.
        method (str): Aer simulation method.
        mps_options (Optional[Dict[str, Any]]): Aer MPS simulator options.
        observable_basis (str): Pauli basis to measure ("Z", "X", or "Y").

    Returns:
        Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]: Expectation values, bond stats, and
        variances.
    """
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
    )
    expvals = combined["expvals"][:, 1:]
    # include initial variance at t=0 for the variance plot
    expvals_var = combined.get("expvals_var", np.zeros_like(combined["expvals"]))
    bonds = combined["bonds"]
    return expvals, bonds, expvals_var




def _save_label(layer_index: int) -> str:
    """Build the Aer save label for a given layer index.

    This generates a stable label name used to store per-layer MPS snapshots in Aer results.

    Args:
        layer_index (int): Zero-based layer index.

    Returns:
        str: Save label for the layer.
    """
    return f"mps_t{layer_index+1}"




def _extract_max_bond_dim_from_saved_mps_state(state_shot: Any) -> int:
    """Extract the maximum bond dimension from a saved MPS snapshot.

    This parses a single per-shot MPS state tuple `(site_tensors, lambdas)` and returns the maximum
    bond dimension, falling back to 1 for unexpected structures.

    Args:
        state_shot (Any): Per-shot MPS state object from Aer results.

    Returns:
        int: Maximum bond dimension for the shot.
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
