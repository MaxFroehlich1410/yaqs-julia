"""
    Israel Fidelity Check — Main Script

    Usage:
        julia --project=. 04_Israel_check/run_check.jl

    This script:
      1. Loads collaborator QPY circuits (4q, 6q)
      2. Simulates each with TDVP and TEBD via run_digital_tjm
      3. Extracts amplitudes for the bitstrings in the .npz reference files
      4. Compares against a_ref and computes fidelity / diagnostics
      5. Saves results to 04_Israel_check/results/
"""

using LinearAlgebra
using PythonCall
using Printf
using Dates

# ---------- Load Yaqs ----------
include(joinpath(@__DIR__, "..", "src", "Yaqs.jl"))
using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.GateLibrary
using .Yaqs.SimulationConfigs
using .Yaqs.DigitalTJM: DigitalCircuit, DigitalGate, TJMOptions, run_digital_tjm
using .Yaqs.CircuitIngestion: ingest_qiskit_circuit

# ---------- Local helpers ----------
include(joinpath(@__DIR__, "utils_local.jl"))
include(joinpath(@__DIR__, "load_circuit.jl"))
include(joinpath(@__DIR__, "compare_amplitudes.jl"))

# ==============================================================================
# Configuration
# ==============================================================================

const DATA_DIR = joinpath(@__DIR__, "max")
const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

const CIRCUIT_LABELS = ["4q", "6q"]

const METHODS = [
    ("TEBD", TJMOptions(local_method=:TEBD, long_range_method=:TEBD)),
    ("TDVP", TJMOptions(local_method=:TDVP, long_range_method=:TDVP)),
]

# Conservative simulation parameters for correctness verification.
# max_bond_dim=512 is generous for circuits with 4q/6q unitary decompositions.
# truncation_threshold=1e-12 means almost no truncation.
const MAX_BOND_DIM = 512
const TRUNCATION_THRESHOLD = 1e-12

const EXPECTED_FIDELITY_LOSS = Dict(
    "4q"  => 0.14101255424513792,
    "6q"  => 0.13516477073819633,
    "8q"  => 0.12057961798663441,
    "10q" => 0.11341694844835948,
    "12q" => 0.11103924492721984,
)

# ==============================================================================
# Main
# ==============================================================================

function run_single(circuit_label::String, method_name::String, alg_options::TJMOptions)
    println("\n", "=" ^ 72)
    println("Running: circuit=$circuit_label  method=$method_name")
    println("=" ^ 72)

    # --- Load circuit ---
    qpy_path = joinpath(DATA_DIR, "circuit $(circuit_label).qpy")
    println("\n[1/5] Loading QPY circuit from: $qpy_path")
    qc = load_qpy_circuit(qpy_path)
    num_qubits = pyconvert(Int, qc.num_qubits)
    println("  Circuit has $num_qubits qubits")

    # --- Ingest into Julia DigitalCircuit ---
    println("\n[2/5] Ingesting Qiskit circuit into DigitalCircuit...")
    digital_circuit = ingest_qiskit_circuit(qc)
    num_layers = length(digital_circuit.layers)
    num_gates = length(digital_circuit.gates)
    println("  DigitalCircuit: $num_qubits qubits, $num_gates gates, $num_layers layers")

    # --- Load reference data ---
    npz_path = joinpath(DATA_DIR, "circuit $(circuit_label).npz")
    println("\n[3/5] Loading NPZ reference data from: $npz_path")
    bitstrings, a_ref = load_npz_reference(npz_path)
    @assert all(bs -> length(bs) == num_qubits, bitstrings) "Bitstring length mismatch: expected $num_qubits, got $(length(bitstrings[1]))"

    # --- Simulate ---
    println("\n[4/5] Simulating with $method_name (max_bond=$MAX_BOND_DIM, trunc=$(TRUNCATION_THRESHOLD))...")
    initial_state = MPS(num_qubits; state="zeros")

    sim_config = TimeEvolutionConfig(
        Observable[], 1.0;
        dt=1.0, num_traj=1, sample_timesteps=false,
        max_bond_dim=MAX_BOND_DIM,
        truncation_threshold=TRUNCATION_THRESHOLD,
    )

    t_start = time()
    final_state, _, bond_dims = run_digital_tjm(
        initial_state, digital_circuit, nothing, sim_config;
        alg_options=alg_options,
    )
    runtime_s = time() - t_start

    max_bond = MPSModule.write_max_bond_dim(final_state)
    state_norm = sqrt(real(MPSModule.scalar_product(final_state, final_state)))
    println("  Simulation complete in $(round(runtime_s; digits=2)) s")
    println("  Max bond dimension reached: $max_bond")
    println("  Final state norm: $state_norm")

    # --- Extract amplitudes and compare ---
    println("\n[5/5] Extracting amplitudes and comparing...")

    convention, reverse_bits, a_circ, res = determine_bitstring_convention(
        final_state, bitstrings, a_ref
    )

    expected_fl = get(EXPECTED_FIDELITY_LOSS, circuit_label, NaN)
    result = build_result(
        circuit_label, method_name, length(bitstrings),
        res, expected_fl, convention, runtime_s, MAX_BOND_DIM, TRUNCATION_THRESHOLD,
    )
    print_result(result)
    return result
end

function main()
    println("Israel Fidelity Check — $(Dates.now())")
    println("Circuits: $(CIRCUIT_LABELS)")
    println("Methods:  $(first.(METHODS))")
    println("Max bond dim:         $MAX_BOND_DIM")
    println("Truncation threshold: $TRUNCATION_THRESHOLD")

    all_results = ComparisonResult[]

    for label in CIRCUIT_LABELS
        for (method_name, alg_opts) in METHODS
            result = run_single(label, method_name, alg_opts)
            push!(all_results, result)
        end
    end

    # --- Save ---
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    json_path = joinpath(RESULTS_DIR, "results_$(timestamp).json")
    txt_path  = joinpath(RESULTS_DIR, "summary_$(timestamp).txt")

    save_results_json(all_results, json_path)
    save_results_summary(all_results, txt_path)

    println("\n", "=" ^ 72)
    println("ALL DONE — $(length(all_results)) experiments completed.")
    println("=" ^ 72)
end

main()
