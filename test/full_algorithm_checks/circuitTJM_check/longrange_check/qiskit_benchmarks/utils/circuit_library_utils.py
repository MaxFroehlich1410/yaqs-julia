import numpy as np
from qiskit.circuit import QuantumCircuit

def add_random_single_qubit_rotation(qc: QuantumCircuit, qubit: int, rng: np.random.Generator) -> None:
    """Apply a random single qubit rotation."""
    theta = rng.uniform(0, np.pi)
    phi = rng.uniform(0, 2 * np.pi)
    lam = rng.uniform(0, 2 * np.pi)
    qc.u(theta, phi, lam, qubit)

