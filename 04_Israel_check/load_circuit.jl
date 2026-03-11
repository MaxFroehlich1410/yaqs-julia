"""
    Circuit and reference data loading for the Israel fidelity-check experiment.

    Uses PythonCall to:
      - Load QPY circuits via Qiskit
      - Load NPZ reference data via NumPy
"""

using PythonCall

"""
    load_qpy_circuit(filepath::String) -> Py

Load a Qiskit QuantumCircuit from a QPY file and return the Python object.
"""
function load_qpy_circuit(filepath::String)
    qpy = pyimport("qiskit.qpy")
    builtins = pyimport("builtins")

    fd = builtins.open(filepath, "rb")
    circuits = qpy.load(fd)
    fd.close()

    qc = circuits[0]
    num_qubits = pyconvert(Int, qc.num_qubits)
    num_gates = pyconvert(Int, pybuiltins.len(qc.data))
    println("  Loaded QPY: $num_qubits qubits, $num_gates instructions")
    return qc
end

"""
    load_npz_reference(filepath::String) -> (bitstrings, a_ref)

Load reference data from an NPZ file.
Returns:
  - `bitstrings::Vector{String}` : bitstring labels
  - `a_ref::Vector{ComplexF64}` : reference amplitudes
"""
function load_npz_reference(filepath::String)
    np = pyimport("numpy")
    data = np.load(filepath, allow_pickle=false)

    bs_raw = data["bitstrings"]
    a_ref_raw = data["a_ref"]

    n_bs = pyconvert(Int, pybuiltins.len(bs_raw))

    bitstrings = Vector{String}(undef, n_bs)
    for i in 0:(n_bs - 1)
        bitstrings[i + 1] = pyconvert(String, pybuiltins.str(bs_raw[i]))
    end

    a_ref = Vector{ComplexF64}(undef, n_bs)
    for i in 0:(n_bs - 1)
        a_ref[i + 1] = pyconvert(ComplexF64, pybuiltins.complex(a_ref_raw[i]))
    end

    data.close()

    println("  Loaded NPZ: $n_bs bitstrings")
    println("  Sample bitstring: \"$(bitstrings[1])\" (length=$(length(bitstrings[1])))")
    println("  Sample a_ref[1]: $(a_ref[1])")
    println("  a_ref norm: $(sqrt(sum(abs2, a_ref)))")

    return bitstrings, a_ref
end
