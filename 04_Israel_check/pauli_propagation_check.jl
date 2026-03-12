"""
Israel observable consistency cross-check using PauliPropagation.

Scope:
- local-only files in `04_Israel_check`
- observable consistency checks (not amplitude/fidelity reproduction)
"""

using LinearAlgebra
using Dates
using Printf
using PythonCall

include(joinpath(@__DIR__, "load_circuit.jl"))
include(joinpath(@__DIR__, "local_pauli_utils.jl"))

const DATA_DIR = joinpath(@__DIR__, "max")
const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

const CIRCUITS = ["4q", "6q"]
const PP_MIN_ABS_COEFF = 1e-12
const RUN_MPS_COMPARISON = false

function observable_rows_from_states(observables, states, amp_dict, pp_circuit, pp_params, nqubits)
    rows = NamedTuple[]
    for obs in observables
        ref_val = expectation_from_sparse_states(obs.symbols, obs.qinds, states, amp_dict)
        pp_val = expectation_pp(
            pp_circuit, pp_params, nqubits, obs.symbols, obs.qinds;
            min_abs_coeff=PP_MIN_ABS_COEFF,
        )
        push!(rows, (
            name = obs.name,
            family = obs.family,
            ref = ref_val,
            pp = pp_val,
            abs_err = abs(ref_val - pp_val),
        ))
    end
    return rows
end

function marginal_rows_from_states(marg_specs, states, amp_dict, pp_circuit, pp_params, nqubits)
    rows = NamedTuple[]
    for ms in marg_specs
        pref = marginal_from_sparse_states(ms.qinds, states, amp_dict)
        ppp = marginal_from_pp(
            pp_circuit, pp_params, nqubits, ms.qinds;
            min_abs_coeff=PP_MIN_ABS_COEFF,
        )
        push!(rows, (name = ms.name, qinds = ms.qinds, pref = pref, pp = ppp))
    end
    return rows
end

function aggregate_marginal_metrics(marg_rows)
    errs = Float64[]
    for r in marg_rows
        append!(errs, abs.(r.pref .- r.pp))
    end
    return metrics(errs)
end

function score_convention(bitstrings, a_ref, observables, marg_specs, pp_circuit, pp_params, nqubits; convention::Symbol)
    _, states, amp_dict = parse_states(bitstrings, a_ref, convention)
    obs_rows = observable_rows_from_states(observables, states, amp_dict, pp_circuit, pp_params, nqubits)
    marg_rows = marginal_rows_from_states(marg_specs, states, amp_dict, pp_circuit, pp_params, nqubits)
    fam_metrics = summarize_by_family(obs_rows)
    marg_metrics = aggregate_marginal_metrics(marg_rows)
    obs_errs = [r.abs_err for r in obs_rows]
    marg_errs = reduce(vcat, [abs.(r.pref .- r.pp) for r in marg_rows]; init=Float64[])
    total_mae = mean(vcat(obs_errs, marg_errs))
    return (
        convention = convention,
        obs_rows = obs_rows,
        marg_rows = marg_rows,
        fam_metrics = fam_metrics,
        marg_metrics = marg_metrics,
        total_mae = total_mae,
    )
end

function write_summary_txt(path::String, label::String, nqubits::Int, op_names, selected_qs, selected_pairs, chosen, convention_scores, best_rows, worst_rows, run_mps::Bool, mps_lines::Vector{String})
    open(path, "w") do io
        println(io, "PauliPropagation observable consistency check")
        println(io, "Generated: $(Dates.now())")
        println(io, "Circuit label: $label")
        println(io, "Qubits: $nqubits")
        println(io, "Initial state: |0...0>")
        println(io, "Heisenberg propagation: true (PauliPropagation reverses Schr-order internally)")
        println(io, "Gate map: u(theta,phi,lambda)->Rz(phi)Ry(theta)Rz(lambda), cx->CNOT")
        println(io, "Observed qiskit op histogram: $op_names")
        println(io, "Chosen bitstring convention for reference amplitudes: $(chosen.convention)")
        println(io, "Convention MAE scores: $convention_scores")
        println(io, "Selected qubits: $selected_qs")
        println(io, "Selected 2q pairs: $selected_pairs")
        println(io)
        println(io, "Error metrics (Reference vs PauliPropagation)")
        for fam in ("1q", "2q")
            if haskey(chosen.fam_metrics, fam)
                m = chosen.fam_metrics[fam]
                @printf(io, "  %s: MAE=%.3e  MAX=%.3e  RMSE=%.3e\n", fam, m.mae, m.maxae, m.rmse)
            end
        end
        mm = chosen.marg_metrics
        @printf(io, "  marginals: MAE=%.3e  MAX=%.3e  RMSE=%.3e\n", mm.mae, mm.maxae, mm.rmse)
        println(io)
        println(io, "Best-matching observables:")
        for r in best_rows
            @printf(io, "  %s (%s): ref=(%.6e%+.6ei) pp=(%.6e%+.6ei) |err|=%.3e\n",
                    r.name, r.family, real(r.ref), imag(r.ref), real(r.pp), imag(r.pp), r.abs_err)
        end
        println(io)
        println(io, "Worst-matching observables:")
        for r in worst_rows
            @printf(io, "  %s (%s): ref=(%.6e%+.6ei) pp=(%.6e%+.6ei) |err|=%.3e\n",
                    r.name, r.family, real(r.ref), imag(r.ref), real(r.pp), imag(r.pp), r.abs_err)
        end
        println(io)
        if run_mps
            println(io, "Optional MPS comparison")
            for ln in mps_lines
                println(io, ln)
            end
        else
            println(io, "Optional MPS comparison: skipped")
        end
    end
end

function run_single(label::String)
    println("\n", "="^88)
    println("PauliPropagation observable check for $label")
    println("="^88)

    qpy_path = joinpath(DATA_DIR, "circuit $(label).qpy")
    npz_path = joinpath(DATA_DIR, "circuit $(label).npz")
    qc = load_qpy_circuit(qpy_path)
    bitstrings, a_ref = load_npz_reference(npz_path)
    nqubits = pyconvert(Int, qc.num_qubits)

    pp_circuit, pp_params, op_names, cx_edges = qiskit_to_pauli_propagation(qc)
    println("Mapped Qiskit -> PauliPropagation gates: $(length(pp_circuit))")
    println("Qiskit op histogram: $op_names")

    selected_qs, selected_pairs = choose_probe_qubits_and_pairs(qc; max_qubits=12, max_pairs=10)
    observables = vcat(
        make_single_qubit_observables(selected_qs),
        make_two_qubit_observables(selected_pairs),
    )
    marg_specs = choose_marginal_subsets(selected_qs, selected_pairs)

    convs = (:native, :reversed, :native_inverted, :reversed_inverted)
    scored = [
        score_convention(bitstrings, a_ref, observables, marg_specs, pp_circuit, pp_params, nqubits; convention=c)
        for c in convs
    ]
    maes = [s.total_mae for s in scored]
    best_idx = argmin(maes)
    chosen = scored[best_idx]
    convention_scores = Dict(convs[i] => maes[i] for i in eachindex(convs))
    println("Convention MAE scores: $convention_scores -> using $(chosen.convention)")

    best_rows, worst_rows = best_and_worst(chosen.obs_rows; nshow=5)

    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    base = "pauli_check_$(label)_$(ts)"
    obs_csv = joinpath(RESULTS_DIR, "$(base)_observables.csv")
    marg_csv = joinpath(RESULTS_DIR, "$(base)_marginals.csv")
    txt_path = joinpath(RESULTS_DIR, "$(base)_summary.txt")

    write_observable_csv(obs_csv, chosen.obs_rows)
    write_marginal_csv(marg_csv, chosen.marg_rows)

    mps_lines = String[]

    write_summary_txt(
        txt_path, label, nqubits, op_names, selected_qs, selected_pairs, chosen, convention_scores, best_rows, worst_rows,
        RUN_MPS_COMPARISON, mps_lines
    )

    println("Saved:")
    println("  $obs_csv")
    println("  $marg_csv")
    println("  $txt_path")
end

function main()
    println("Israel PauliPropagation observable cross-check — $(Dates.now())")
    for label in CIRCUITS
        run_single(label)
    end
    println("\nDone.")
end

main()
