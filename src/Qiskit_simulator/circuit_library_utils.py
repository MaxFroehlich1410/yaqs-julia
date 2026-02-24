import numpy as np
from qiskit.circuit import QuantumCircuit

def add_random_single_qubit_rotation(qc: QuantumCircuit, qubit: int, rng: np.random.Generator) -> None:
    """Apply a random single-qubit U rotation to a circuit.

    This samples angles from uniform ranges and applies a U gate on the specified qubit using the
    provided RNG for reproducibility.

    Args:
        qc (QuantumCircuit): Circuit to modify.
        qubit (int): Target qubit index.
        rng (np.random.Generator): Random number generator for angles.

    Returns:
        None: The circuit is modified in-place.
    """
    theta = rng.uniform(0, np.pi)
    phi = rng.uniform(0, 2 * np.pi)
    lam = rng.uniform(0, 2 * np.pi)
    qc.u(theta, phi, lam, qubit)

