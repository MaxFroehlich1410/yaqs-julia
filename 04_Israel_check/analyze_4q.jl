"""
    Israel Fidelity Check — Detailed 4q Method Comparison

    Usage:
        julia --project=. 04_Israel_check/analyze_4q.jl

    Loads the 4q collaborator circuit and runs it with selectable methods,
    comparing amplitudes against the .npz reference.  Edit the CONFIGURATION
    block below to choose which methods to run and their parameters.
"""

using LinearAlgebra
using PythonCall
using Printf
using Dates

# ── Load Yaqs ────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "Yaqs.jl"))
using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.GateLibrary
using .Yaqs.SimulationConfigs
using .Yaqs.DigitalTJM
using .Yaqs.CircuitIngestion: ingest_qiskit_circuit

# ── Local helpers ────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "utils_local.jl"))
include(joinpath(@__DIR__, "load_circuit.jl"))
include(joinpath(@__DIR__, "compare_amplitudes.jl"))

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================

const CIRCUIT_LABEL = "6q"          # which circuit to load
const DRAW_CIRCUIT  = true         # set true to save a Qiskit circuit diagram

# Max bond dimension and truncation for ALL methods (use the same for fair comparison)
const MAX_BOND_DIM          = 512
const TRUNCATION_THRESHOLD  = 1e-12   # near-exact

# ── Method selection ─────────────────────────────────────────────────────────
# Toggle each method on/off.  For TDVP you can also set the number of
# back-and-forth sweeps per gate (tdvp_gate_sweeps) and truncation timing.
#
# Available local/long-range modes: :TEBD, :TDVP, :BUG, :ZIPUP, :SRC

const RUN_TEBD       = true
const RUN_TDVP       = false
const RUN_TDVP_MULTI = false         # TDVP with multiple sweeps per gate
const RUN_BUG        = true
const RUN_ZIPUP      = true
const RUN_SRC        = false
const RUN_VARIATIONAL = false        # Julia apply_variational! (TenPy-style)
                                    # NOTE: variational MPO application grows bond
                                    # dim slowly from a product-state start. It
                                    # needs many sweeps per gate or an enriched
                                    # initial guess to converge.

# TDVP settings
const TDVP_GATE_SWEEPS         = 1              # single-sweep TDVP
const TDVP_TRUNCATION_TIMING   = :after_window  # :during | :after_window
const TDVP_MULTI_GATE_SWEEPS   = 6             # multi-sweep TDVP: how many sweeps per gate
const TDVP_MULTI_TRUNCATION    = :after_window
const TDVP_PAD_BOND_DIM        = 4              # pad initial MPS bond dim before TDVP runs
                                                 # (set to 0 or 1 to disable padding)

# BUG settings
const BUG_TRUNCATION_GRANULARITY = :after_sweep  # :after_sweep | :after_site

# Variational settings (TenPy-style MPO application)
const VAR_MIN_SWEEPS = 10
const VAR_MAX_SWEEPS = 50

# Expected fidelity loss (from collaborator)
const EXPECTED_FIDELITY_LOSS = Dict(
    "4q"  => 0.14101255424513792,
    "6q"  => 0.13516477073819633,
)

# =============================================================================
# PATHS
# =============================================================================

const DATA_DIR    = joinpath(@__DIR__, "max")
const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

# =============================================================================
# Circuit drawing (optional)
# =============================================================================

function draw_circuit(qc, label)
    try
        plt = pyimport("matplotlib.pyplot")
        fig = qc.draw("mpl"; fold=-1)
        path = joinpath(RESULTS_DIR, "circuit_$(label).png")
        fig.savefig(path; dpi=150, bbox_inches="tight")
        plt.close(fig)
        println("  Circuit diagram saved to $path")
    catch e
        @warn "Circuit drawing failed (missing pylatexenc?), skipping." exception=e
    end
end

# =============================================================================
# Variational runner (local, experiment-only)
#
# This mirrors run_digital_tjm but uses apply_variational! for multi-qubit gates.
# =============================================================================

function run_variational(initial_state::MPS, circuit::DigitalCircuit,
                         sim_params::AbstractSimConfig;
                         min_sweeps::Int=2, max_sweeps::Int=10)
    state = deepcopy(initial_state)
    layers, _ = DigitalTJM.process_circuit(circuit)
    num_layers = length(layers)

    for (l_idx, layer) in enumerate(layers)
        for gate in layer
            if length(gate.sites) == 1
                DigitalTJM.apply_single_qubit_gate!(state, gate)
            end
        end

        for gate in layer
            if length(gate.sites) >= 2
                sites = sort(gate.sites)
                s1, s2 = sites[1], sites[end]
                is_local = (length(sites) == 2 && s2 == s1 + 1)

                if is_local
                    DigitalTJM.apply_local_gate_exact!(state, gate.op, s1, s2, sim_params)
                    MPSModule.normalize!(state)
                else
                    op_mat = Matrix{ComplexF64}(GateLibrary.matrix(gate.op))
                    L = state.length
                    d = state.phys_dims[s1]

                    if length(sites) == 2
                        mpo_gate = DigitalTJM._mpo_from_two_qubit_gate_matrix(op_mat, s1, s2, L; d=d)
                    else
                        mpo_gate = DigitalTJM._mpo_from_contiguous_k_qubit_gate_matrix(op_mat, sites, L; d=d)
                    end

                    MPOModule.apply_variational!(state, mpo_gate;
                        chi_max=sim_params.max_bond_dim,
                        trunc=max(sim_params.truncation_threshold, 0.0),
                        min_sweeps=min_sweeps,
                        max_sweeps=max_sweeps)
                    MPSModule.normalize!(state)
                end
            end
        end

        bond = MPSModule.write_max_bond_dim(state)
        print("\r\tLayer $l_idx/$num_layers | Max Bond: $bond")
        flush(stdout)
    end
    println()
    return state
end

# =============================================================================
# Single experiment
# =============================================================================

function run_method(label, method_name, digital_circuit, bitstrings, a_ref, num_qubits;
                    alg_options=nothing, custom_runner=nothing,
                    var_min_sweeps=2, var_max_sweeps=10,
                    pad_bond_dim::Int=0)

    println("\n", "-" ^ 60)
    println("  Method: $method_name")
    println("-" ^ 60)

    initial_state = MPS(num_qubits; state="zeros")
    if pad_bond_dim >= 2
        pad_bond_dimension!(initial_state, pad_bond_dim; noise_scale=1e-8)
        println("  Padded initial MPS to bond dim $pad_bond_dim")
    end

    sim_config = TimeEvolutionConfig(
        Observable[], 1.0;
        dt=1.0, num_traj=1, sample_timesteps=false,
        max_bond_dim=MAX_BOND_DIM,
        truncation_threshold=TRUNCATION_THRESHOLD,
    )

    t_start = time()
    if custom_runner === :variational
        final_state = run_variational(initial_state, digital_circuit, sim_config;
                                      min_sweeps=var_min_sweeps, max_sweeps=var_max_sweeps)
    else
        final_state, _, _ = run_digital_tjm(
            initial_state, digital_circuit, nothing, sim_config;
            alg_options=alg_options,
        )
    end
    runtime_s = time() - t_start

    max_bond = MPSModule.write_max_bond_dim(final_state)
    state_norm = sqrt(real(MPSModule.scalar_product(final_state, final_state)))
    println("  Runtime: $(round(runtime_s; digits=2)) s")
    println("  Max bond dim reached: $max_bond")
    println("  State norm: $state_norm")

    a_circ = amplitudes_for_bitstrings(final_state, bitstrings; reverse_bits=true)
    res = compare_amplitudes(a_ref, a_circ)

    expected_fl = get(EXPECTED_FIDELITY_LOSS, label, NaN)
    result = build_result(
        label, method_name, length(bitstrings),
        res, expected_fl, "reversed", runtime_s, MAX_BOND_DIM, TRUNCATION_THRESHOLD,
    )
    print_result(result)
    return result
end

# =============================================================================
# Main
# =============================================================================

function main()
    label = CIRCUIT_LABEL
    println("=" ^ 72)
    println("Israel Fidelity Check — Detailed $label Analysis")
    println("$(Dates.now())")
    println("=" ^ 72)

    # ── Load circuit ─────────────────────────────────────────────────────
    qpy_path = joinpath(DATA_DIR, "circuit $(label).qpy")
    println("\n[1] Loading QPY circuit...")
    qc = load_qpy_circuit(qpy_path)
    num_qubits = pyconvert(Int, qc.num_qubits)

    if DRAW_CIRCUIT
        println("  Drawing circuit...")
        draw_circuit(qc, label)
    end

    println("\n[2] Ingesting circuit...")
    digital_circuit = ingest_qiskit_circuit(qc)
    n_gates = length(digital_circuit.gates)
    n_layers = length(digital_circuit.layers)
    println("  $num_qubits qubits, $n_gates gates, $n_layers layers")

    # ── Load reference data ──────────────────────────────────────────────
    npz_path = joinpath(DATA_DIR, "circuit $(label).npz")
    println("\n[3] Loading NPZ reference...")
    bitstrings, a_ref = load_npz_reference(npz_path)

    # ── Gate statistics ──────────────────────────────────────────────────
    n_1q = count(g -> length(g.sites) == 1, digital_circuit.gates)
    n_2q = count(g -> length(g.sites) == 2, digital_circuit.gates)
    lr_gates = filter(g -> length(g.sites) == 2 && abs(g.sites[1] - g.sites[2]) > 1, digital_circuit.gates)
    n_lr = length(lr_gates)
    max_dist = n_lr > 0 ? maximum(abs(g.sites[1] - g.sites[2]) for g in lr_gates) : 0
    println("\n  Gate breakdown:")
    println("    1-qubit gates:          $n_1q")
    println("    2-qubit gates:          $n_2q ($(n_lr) long-range, max distance $max_dist)")

    # ── Run methods ──────────────────────────────────────────────────────
    println("\n[4] Running methods...")
    println("  Max bond dim: $MAX_BOND_DIM")
    println("  Truncation:   $TRUNCATION_THRESHOLD")

    all_results = ComparisonResult[]

    if RUN_TEBD
        opts = TJMOptions(local_method=:TEBD, long_range_method=:TEBD)
        r = run_method(label, "TEBD", digital_circuit, bitstrings, a_ref, num_qubits;
                       alg_options=opts)
        push!(all_results, r)
    end

    if RUN_TDVP
        opts = TJMOptions(local_method=:TEBD, long_range_method=:TDVP,
                          tdvp_truncation_timing=TDVP_TRUNCATION_TIMING,
                          tdvp_gate_sweeps=TDVP_GATE_SWEEPS)
        r = run_method(label, "TEBD+TDVP_LR (sweeps=$(TDVP_GATE_SWEEPS), trunc=$(TDVP_TRUNCATION_TIMING), pad=$(TDVP_PAD_BOND_DIM))",
                       digital_circuit, bitstrings, a_ref, num_qubits;
                       alg_options=opts, pad_bond_dim=TDVP_PAD_BOND_DIM)
        push!(all_results, r)
    end

    if RUN_TDVP_MULTI
        opts = TJMOptions(local_method=:TEBD, long_range_method=:TDVP,
                          tdvp_truncation_timing=TDVP_MULTI_TRUNCATION,
                          tdvp_gate_sweeps=TDVP_MULTI_GATE_SWEEPS)
        r = run_method(label, "TEBD+TDVP_LR (sweeps=$(TDVP_MULTI_GATE_SWEEPS), trunc=$(TDVP_MULTI_TRUNCATION), pad=$(TDVP_PAD_BOND_DIM))",
                       digital_circuit, bitstrings, a_ref, num_qubits;
                       alg_options=opts, pad_bond_dim=TDVP_PAD_BOND_DIM)
        push!(all_results, r)
    end

    if RUN_BUG
        opts = TJMOptions(local_method=:TEBD, long_range_method=:BUG,
                          bug_truncation_granularity=BUG_TRUNCATION_GRANULARITY)
        r = run_method(label, "TEBD+BUG_LR", digital_circuit, bitstrings, a_ref, num_qubits;
                       alg_options=opts)
        push!(all_results, r)
    end

    if RUN_ZIPUP
        opts = TJMOptions(local_method=:TEBD, long_range_method=:ZIPUP)
        r = run_method(label, "TEBD+ZIPUP_LR", digital_circuit, bitstrings, a_ref, num_qubits;
                       alg_options=opts)
        push!(all_results, r)
    end

    if RUN_SRC
        opts = TJMOptions(local_method=:TEBD, long_range_method=:SRC)
        r = run_method(label, "TEBD+SRC_LR", digital_circuit, bitstrings, a_ref, num_qubits;
                       alg_options=opts)
        push!(all_results, r)
    end

    if RUN_VARIATIONAL
        r = run_method(label, "TEBD+Variational_LR (sweeps=$VAR_MIN_SWEEPS-$VAR_MAX_SWEEPS)", digital_circuit, bitstrings, a_ref, num_qubits;
                       custom_runner=:variational,
                       var_min_sweeps=VAR_MIN_SWEEPS, var_max_sweeps=VAR_MAX_SWEEPS)
        push!(all_results, r)
    end

    # ── Save ─────────────────────────────────────────────────────────────
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    json_path = joinpath(RESULTS_DIR, "analyze_$(label)_$(timestamp).json")
    txt_path  = joinpath(RESULTS_DIR, "analyze_$(label)_$(timestamp).txt")

    save_results_json(all_results, json_path)
    save_results_summary(all_results, txt_path)

    # ── Console summary table ────────────────────────────────────────────
    println("\n", "=" ^ 100)
    @printf("%-35s  %12s  %14s  %14s  %10s  %8s\n",
            "Method", "Fidelity", "Fidelity Loss", "Expected FL", "Deviation", "Time (s)")
    println("-" ^ 100)
    for r in all_results
        @printf("%-35s  %12.10f  %14.12f  %14.12f  %10.2e  %8.2f\n",
                r.method, r.fidelity, r.fidelity_loss, r.expected_fidelity_loss, r.abs_deviation, r.runtime_s)
    end
    println("=" ^ 100)
end

main()
