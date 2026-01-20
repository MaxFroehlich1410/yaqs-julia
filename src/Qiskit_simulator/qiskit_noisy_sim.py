from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.backends.aer_simulator import AerSimulator
from qiskit_aer import Aer
import numpy as np

def qiskit_noisy_simulator_stepwise(circuit, noise_model, num_qubits, method="automatic", observable_basis="Z"):
    """Compute per-qubit Pauli expectations using Aer with noise.

    This builds Pauli observables for each qubit, runs the noisy Aer estimator on the provided
    circuit, and returns the expectation values in reverse order.

    Args:
        circuit (QuantumCircuit): Circuit to simulate.
        noise_model (NoiseModel): Qiskit noise model to apply.
        num_qubits (int): Number of qubits in the circuit.
        method (str): Aer simulator method.
        observable_basis (str): Pauli basis to measure ("Z", "X", or "Y").

    Returns:
        np.ndarray: Expectation values for each qubit.

    Raises:
        ValueError: If `observable_basis` is not one of "Z", "X", or "Y".
    """
    if observable_basis not in ["Z", "X", "Y"]:
        raise ValueError(f"observable_basis must be one of ['Z', 'X', 'Y'], got {observable_basis}")
    observables = []
    for i in range(num_qubits):
        pauli_str = "I"*i + observable_basis + "I"*(num_qubits - i - 1)
        observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
        observables.append(observable)

    z_expectations = []
    qc_copy = circuit.copy()

    # print('circut:')
    # print(qc_copy.draw())
    # exact_estimator = Estimator()
    noisy_estimator = Estimator(options=dict(backend_options=dict(noise_model=noise_model, method=method)))
    pub = (qc_copy, observables)
    job = noisy_estimator.run([pub])
    result = job.result()
    pub_result = result[0] 

    # .data is a DataBin
    data = pub_result.data

    # The Z expectation values
    evs = np.array(data.evs).squeeze()  # type: ignore[attr-defined]  # This is a numpy array of shape (num_qubits,)
    evs = evs.reshape(-1)

    return evs[::-1]

def qiskit_noisy_simulator(circuit, noise_model, num_qubits, num_layers, method="automatic", observable_basis="Z"):
    """Compute layered Pauli expectations with a noisy Aer simulation.

    This composes the circuit with itself for each layer count and evaluates per-qubit expectation
    values using the stepwise noisy simulator.

    Args:
        circuit (QuantumCircuit): Base circuit to repeat.
        noise_model (NoiseModel): Qiskit noise model to apply.
        num_qubits (int): Number of qubits in the circuit.
        num_layers (int): Number of layers to simulate.
        method (str): Aer simulator method.
        observable_basis (str): Pauli basis to measure ("Z", "X", or "Y").

    Returns:
        np.ndarray: Array of expectation values per layer.
    """
    expvals_list = []
    for layer in range(num_layers):
        qc_copy = circuit.copy()
        for j in range(layer):
            qc_copy = qc_copy.compose(circuit)
        expvals_list.append(qiskit_noisy_simulator_stepwise(qc_copy, noise_model, num_qubits, method=method, observable_basis=observable_basis))
    return np.array(expvals_list)


