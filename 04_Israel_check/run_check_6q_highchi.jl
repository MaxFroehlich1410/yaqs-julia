"""
    Israel Fidelity Check — 6q TEBD with higher max_bond_dim

    Usage:
        julia --project=. 04_Israel_check/run_check_6q_highchi.jl

    Re-runs 6q TEBD with max_bond_dim=1024 since 512 was insufficient.
"""

using LinearAlgebra
using PythonCall
using Printf
using Dates

include(joinpath(@__DIR__, "..", "src", "Yaqs.jl"))
using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.GateLibrary
using .Yaqs.SimulationConfigs
using .Yaqs.DigitalTJM: DigitalCircuit, DigitalGate, TJMOptions, run_digital_tjm
using .Yaqs.CircuitIngestion: ingest_qiskit_circuit

include(joinpath(@__DIR__, "utils_local.jl"))
include(joinpath(@__DIR__, "load_circuit.jl"))
include(joinpath(@__DIR__, "compare_amplitudes.jl"))

const DATA_DIR = joinpath(@__DIR__, "max")
const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

const MAX_BOND_DIM = 1024
const TRUNCATION_THRESHOLD = 1e-12

const EXPECTED_FIDELITY_LOSS = Dict(
    "6q" => 0.13516477073819633,
)

function main()
    label = "6q"
    method_name = "TEBD"
    alg_options = TJMOptions(local_method=:TEBD, long_range_method=:TEBD)

    println("Israel Fidelity Check — 6q high-chi run — $(Dates.now())")
    println("Max bond dim: $MAX_BOND_DIM")

    qpy_path = joinpath(DATA_DIR, "circuit $(label).qpy")
    println("\n[1/5] Loading QPY circuit...")
    qc = load_qpy_circuit(qpy_path)
    num_qubits = pyconvert(Int, qc.num_qubits)

    println("\n[2/5] Ingesting circuit...")
    digital_circuit = ingest_qiskit_circuit(qc)
    num_layers = length(digital_circuit.layers)
    println("  $num_qubits qubits, $(length(digital_circuit.gates)) gates, $num_layers layers")

    npz_path = joinpath(DATA_DIR, "circuit $(label).npz")
    println("\n[3/5] Loading NPZ reference...")
    bitstrings, a_ref = load_npz_reference(npz_path)

    println("\n[4/5] Simulating with $method_name (max_bond=$MAX_BOND_DIM)...")
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
    println("  Done in $(round(runtime_s; digits=1)) s, max bond = $max_bond")

    println("\n[5/5] Extracting amplitudes and comparing...")
    convention, reverse_bits, a_circ, res = determine_bitstring_convention(
        final_state, bitstrings, a_ref
    )

    expected_fl = EXPECTED_FIDELITY_LOSS[label]
    result = build_result(
        label, method_name, length(bitstrings),
        res, expected_fl, convention, runtime_s, MAX_BOND_DIM, TRUNCATION_THRESHOLD,
    )
    print_result(result)

    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    save_results_json([result], joinpath(RESULTS_DIR, "results_6q_chi$(MAX_BOND_DIM)_$(timestamp).json"))
    save_results_summary([result], joinpath(RESULTS_DIR, "summary_6q_chi$(MAX_BOND_DIM)_$(timestamp).txt"))
end

main()
