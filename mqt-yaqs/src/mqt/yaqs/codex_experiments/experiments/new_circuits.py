# pip install qiskit numpy
from qiskit import QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt

def qaoa_ising_layer(n_qubits: int) -> QuantumCircuit:
    """
    One QAOA layer for a 1D Ising cost:
        - Apply RX(β) on all qubits
        - Apply RZZ(γ) on nearest neighbors with an even-odd then odd-even sweep

    Angle conventions:
    You defined RX(β) = exp(-i β X) and RZZ(γ) = exp(-i γ Z⊗Z).
    Qiskit uses RX(θ) = exp(-i θ/2 X) and RZZ(θ) = exp(-i θ/2 Z⊗Z),
    so we pass θ_rx = 2β and θ_rzz = 2γ to QuantumCircuit.rx / .rzz.

    Args:
        n_qubits: number of qubits (≥1)

    Returns:
        QuantumCircuit with one QAOA layer.
    """
    rng = np.random.default_rng()
    beta = rng.uniform(0.0, 2.0*np.pi)
    gamma = rng.uniform(0.0, 2.0*np.pi)

    qc = QuantumCircuit(n_qubits, name="QAOA_layer")

    # RX(β) on all qubits
    for q in range(n_qubits):
        qc.rx(2.0 * beta, q)  # Qiskit θ = 2β

    # Cost unitary with brickwork execution:
    # even edges: (0,1), (2,3), ...
    for i in range(0, n_qubits - 1, 2):
        qc.rzz(2.0 * gamma, i, i + 1)  # Qiskit θ = 2γ

    # odd edges: (1,2), (3,4), ...
    for i in range(1, n_qubits - 1, 2):
        qc.rzz(2.0 * gamma, i, i + 1)

    qc.barrier()
    return qc


def hea_layer(n_qubits: int) -> QuantumCircuit:
    """
    One hardware-efficient ansatz (HEA) layer:
        - Arbitrary single-qubit rotations via U3 decomposition Rz(φ) Ry(θ) Rz(λ)
        - Brickwork CZ entanglers on neighbors, using either even-odd or odd-even pattern
          (chosen randomly for a single layer)

    Sampling:
        φ, λ ~ Uniform[0, 2π), θ ~ Uniform[0, π] (covers the Bloch sphere without redundancy).

    Args:
        n_qubits: number of qubits (≥1)

    Returns:
        QuantumCircuit with one HEA layer.
    """
    rng = np.random.default_rng()
    qc = QuantumCircuit(n_qubits, name="HEA_layer")

    # Single-qubit U3 = Rz(φ) Ry(θ) Rz(λ)
    for q in range(n_qubits):
        phi   = rng.uniform(0.0, 2.0*np.pi)
        theta = rng.uniform(0.0, np.pi)
        lam   = rng.uniform(0.0, 2.0*np.pi)
        qc.rz(phi, q)
        qc.ry(theta, q)
        qc.rz(lam, q)

    # Brickwork CZ pattern: pick parity at random for this layer
    start = int(rng.integers(0, 2))  # 0 => even edges; 1 => odd edges
    for i in range(start, n_qubits - 1, 2):
        qc.cz(i, i + 1)

    qc.barrier()
    return qc


# --- tiny demo (optional) ---
if __name__ == "__main__":
    n = 6
    print(qaoa_ising_layer(n).draw("text"))
    print(hea_layer(n).draw("text"))
    plt.show()