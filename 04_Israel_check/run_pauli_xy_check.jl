"""
    run_pauli_xy_check.jl

Exact PauliPropagation cross-check for all single-qubit X/Y/Z observables on a
single Israel circuit, defaulting to 4q.

This is intended to answer whether off-diagonal observables (X and Y) can be
compared reliably against the sparse reference amplitudes in `.npz`.
"""

using Dates
using Printf
using PythonCall
using Statistics

include(joinpath(@__DIR__, "load_circuit.jl"))
include(joinpath(@__DIR__, "local_pauli_utils.jl"))

const DATA_DIR = joinpath(@__DIR__, "max")
const RESULTS_DIR = joinpath(@__DIR__, "results_pauli_xy")
mkpath(RESULTS_DIR)

const CONVENTIONS = (:native, :reversed, :native_inverted, :reversed_inverted)

function all_single_qubit_observables(nqubits::Int)
    obs = ObservableSpec[]
    for q in 1:nqubits
        push!(obs, ObservableSpec("X_$q", [:X], [q], "X"))
        push!(obs, ObservableSpec("Y_$q", [:Y], [q], "Y"))
        push!(obs, ObservableSpec("Z_$q", [:Z], [q], "Z"))
    end
    return obs
end

function observable_rows(observables, states, amp_dict, pp_circuit, pp_params, nqubits)
    rows = NamedTuple[]
    for obs in observables
        ref_val = expectation_from_sparse_states(obs.symbols, obs.qinds, states, amp_dict)
        pp_val = expectation_pp(
            pp_circuit, pp_params, nqubits, obs.symbols, obs.qinds;
            min_abs_coeff=0.0,
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

function family_metrics(rows)
    fams = Dict{String,NamedTuple}()
    for fam in ("X", "Y", "Z")
        sub = [r for r in rows if r.family == fam]
        errs = [r.abs_err for r in sub]
        ref_abs = [abs(r.ref) for r in sub]
        pp_abs = [abs(r.pp) for r in sub]
        fams[fam] = (
            mae = mean(errs),
            maxae = maximum(errs),
            rmse = sqrt(mean(abs2, errs)),
            max_ref_abs = maximum(ref_abs),
            max_pp_abs = maximum(pp_abs),
            n_ref_gt_1e6 = count(>(1e-6), ref_abs),
            n_pp_gt_1e6 = count(>(1e-6), pp_abs),
        )
    end
    return fams
end

function score_convention(bitstrings, a_ref, observables, pp_circuit, pp_params, nqubits, convention)
    _, states, amp_dict = parse_states(bitstrings, a_ref, convention)
    rows = observable_rows(observables, states, amp_dict, pp_circuit, pp_params, nqubits)
    fams = family_metrics(rows)
    total_mae = mean([r.abs_err for r in rows])
    return (convention=convention, rows=rows, fams=fams, total_mae=total_mae)
end

function write_rows_csv(path, rows)
    open(path, "w") do io
        println(io, "name,family,real_ref,imag_ref,real_pp,imag_pp,abs_err")
        for r in rows
            @printf(io, "%s,%s,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                    r.name, r.family,
                    real(r.ref), imag(r.ref),
                    real(r.pp), imag(r.pp),
                    r.abs_err)
        end
    end
end

function write_summary(path, label, op_names, scores, best)
    sorted_rows = sort(best.rows; by = r -> r.abs_err)
    best_rows = first(sorted_rows, min(8, length(sorted_rows)))
    worst_rows = reverse(last(sorted_rows, min(8, length(sorted_rows))))

    open(path, "w") do io
        println(io, "Single-qubit X/Y/Z PauliPropagation cross-check")
        println(io, "Generated: $(Dates.now())")
        println(io, "Circuit label: $label")
        println(io, "Observed qiskit op histogram: $op_names")
        println(io, "Truncation: NONE (min_abs_coeff=0.0)")
        println(io, "Chosen amplitude convention: $(best.convention)")
        println(io, "Convention total-MAE scores: $(Dict(s.convention => s.total_mae for s in scores))")
        println(io)
        println(io, "Per-family metrics:")
        for fam in ("X", "Y", "Z")
            m = best.fams[fam]
            @printf(io, "  %s: MAE=%.6e  MAX=%.6e  RMSE=%.6e  max|ref|=%.6e  max|pp|=%.6e  count(|ref|>1e-6)=%d  count(|pp|>1e-6)=%d\n",
                    fam, m.mae, m.maxae, m.rmse, m.max_ref_abs, m.max_pp_abs, m.n_ref_gt_1e6, m.n_pp_gt_1e6)
        end
        println(io)
        println(io, "Best-matching observables:")
        for r in best_rows
            @printf(io, "  %s: ref=(%.6e%+.6ei)  pp=(%.6e%+.6ei)  |err|=%.3e\n",
                    r.name, real(r.ref), imag(r.ref), real(r.pp), imag(r.pp), r.abs_err)
        end
        println(io)
        println(io, "Worst-matching observables:")
        for r in worst_rows
            @printf(io, "  %s: ref=(%.6e%+.6ei)  pp=(%.6e%+.6ei)  |err|=%.3e\n",
                    r.name, real(r.ref), imag(r.ref), real(r.pp), imag(r.pp), r.abs_err)
        end
    end
end

function main()
    label = isempty(ARGS) ? "4q" : ARGS[1]
    println("Single-qubit X/Y/Z check for $label — $(Dates.now())")

    qc = load_qpy_circuit(joinpath(DATA_DIR, "circuit $(label).qpy"))
    bitstrings, a_ref = load_npz_reference(joinpath(DATA_DIR, "circuit $(label).npz"))
    nqubits = pyconvert(Int, qc.num_qubits)

    pp_circuit, pp_params, op_names, _ = qiskit_to_pauli_propagation(qc)
    observables = all_single_qubit_observables(nqubits)

    scores = [
        score_convention(bitstrings, a_ref, observables, pp_circuit, pp_params, nqubits, convention)
        for convention in CONVENTIONS
    ]
    best = scores[argmin([s.total_mae for s in scores])]

    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    csv_path = joinpath(RESULTS_DIR, "single_xyz_$(label)_$(ts).csv")
    txt_path = joinpath(RESULTS_DIR, "single_xyz_$(label)_$(ts).txt")

    write_rows_csv(csv_path, best.rows)
    write_summary(txt_path, label, op_names, scores, best)

    println("Chosen convention: $(best.convention)")
    for fam in ("X", "Y", "Z")
        m = best.fams[fam]
        @printf("  %s: MAE=%.6e  MAX=%.6e  max|ref|=%.6e  count(|ref|>1e-6)=%d\n",
                fam, m.mae, m.maxae, m.max_ref_abs, m.n_ref_gt_1e6)
    end
    println("Saved:")
    println("  $csv_path")
    println("  $txt_path")
end

main()
