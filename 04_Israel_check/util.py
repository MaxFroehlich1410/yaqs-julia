from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit import qpy

def save_circuit_qpy(circuit: QuantumCircuit, filename: str) -> None:
    """
    Save a Qiskit QuantumCircuit to a QPY file.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to save.
    filename : str
        Output file path (e.g. "circuit.qpy").
    """
    with open(filename, "wb") as fd:
        qpy.dump(circuit, fd)


def load_circuit_qpy(filename: str) -> QuantumCircuit:
    """
    Load a QuantumCircuit from a QPY file.

    Parameters
    ----------
    filename : str
        Path to the QPY file.

    Returns
    -------
    QuantumCircuit
        The reconstructed circuit.
    """
    with open(filename, "rb") as fd:
        circuits = qpy.load(fd)

    return circuits[0]


def save_npz(
    filename: str,
    bitstrings: np.ndarray,
    a_ref: np.ndarray,
    a_circ: np.ndarray | None = None,
    fidelity: float | None = None,
) -> None:
    """
    Save bitstrings, reference amplitudes, and circuit amplitudes to a compressed NPZ file.

    Parameters
    ----------
    filename : str
        Output file path, e.g. "state_compare.npz".
    bitstrings : np.ndarray
        Array of bitstrings.
    a_ref : np.ndarray
        Reference amplitudes.
    a_circ : np.ndarray | None
        Circuit amplitudes.
    fidelity : float | None
        Optional fidelity value to store as metadata.
    """
    if a_circ is None:
        if len(bitstrings) != len(a_ref):
            raise ValueError("bitstrings and a_ref must have the same length.")

        payload = {
            "bitstrings": np.asarray(bitstrings),
            "a_ref": np.asarray(a_ref, dtype=np.complex128),
        }
    else:
        if len(bitstrings) != len(a_ref) or len(a_ref) != len(a_circ):
            raise ValueError("bitstrings, a_ref, and a_circ must have the same length.")

        payload = {
            "bitstrings": np.asarray(bitstrings),
            "a_ref": np.asarray(a_ref, dtype=np.complex128),
            "a_circ": np.asarray(a_circ, dtype=np.complex128),
        }

    if fidelity is not None:
        payload["fidelity"] = np.asarray(fidelity, dtype=np.float64)

    np.savez_compressed(filename, **payload)


def load_npz(filename: str):
    """
    Load bitstrings, reference amplitudes, and circuit amplitudes from an NPZ file.

    Returns
    -------
    dict
        Keys:
        - 'bitstrings'
        - 'a_ref'
        - optionally 'a_circ'
        - optionally 'fidelity'
    """
    with np.load(filename, allow_pickle=False) as data:
        if "a_circ" in data:
            result = {
                "bitstrings": data["bitstrings"],
                "a_ref": data["a_ref"],
                "a_circ": data["a_circ"],
            }
        else:
            result = {
                "bitstrings": data["bitstrings"],
                "a_ref": data["a_ref"],
            }
        if "fidelity" in data:
            result["fidelity"] = float(data["fidelity"])
    return result


def write_qasm_file(
    circuit: QuantumCircuit,
    filepath: str | Path,
    version: int = 3,
    encoding: str = "utf-8",
) -> Path:
    """
    Write a Qiskit QuantumCircuit to a .qasm file (OpenQASM 2 or 3).

    Parameters
    ----------
    circuit
        The Qiskit QuantumCircuit to export.
    filepath
        Output path (e.g., "my_circuit.qasm").
    version
        3 for OpenQASM 3 (recommended if your Qiskit supports it),
        2 for OpenQASM 2.
    encoding
        File encoding.

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if version == 3:
        # Qiskit: circuit.qasm() may default to OpenQASM 3 in newer versions.
        # Prefer the explicit exporter when available.
        try:
            from qiskit.qasm3 import dumps as qasm3_dumps  # type: ignore
            qasm_str = qasm3_dumps(circuit)
        except Exception:
            # Fallback: try circuit.qasm() (may be QASM2 or QASM3 depending on Qiskit)
            qasm_str = circuit.qasm()
    elif version == 2:
        # OpenQASM 2 exporter is commonly available via qiskit.qasm2
        try:
            from qiskit.qasm2 import dumps as qasm2_dumps  # type: ignore
            qasm_str = qasm2_dumps(circuit)
        except Exception:
            # Fallback: many older versions provide QASM2 via circuit.qasm()
            qasm_str = circuit.qasm()
    else:
        raise ValueError("version must be 2 or 3")

    path.write_text(qasm_str, encoding=encoding)
    return path
