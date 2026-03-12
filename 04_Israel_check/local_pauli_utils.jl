"""
    Local helpers for PauliPropagation observable cross-checks on Israel circuits.

This file is intentionally self-contained and local to `04_Israel_check`.
"""

using Printf
using Statistics
using PythonCall

include(joinpath(@__DIR__, "load_circuit.jl"))

using PauliPropagation

struct ObservableSpec
    name::String
    symbols::Vector{Symbol}
    qinds::Vector{Int}
    family::String
end

struct MarginalSpec
    name::String
    qinds::Vector{Int}
end

"""
    qiskit_to_pauli_propagation(qc::Py)

Convert a Qiskit circuit (`u` and `cx`) into a PauliPropagation circuit.
Mapping is local and explicit:
- `u(θ, φ, λ)` -> `Rz(φ) * Ry(θ) * Rz(λ)`
- `cx(c, t)`   -> `CliffordGate(:CNOT, [c+1, t+1])`
"""
function qiskit_to_pauli_propagation(qc::Py)
    circ = Any[]
    params = Float64[]
    op_names = Dict{String,Int}()
    twoq_edges = Tuple{Int,Int}[]

    for instr in qc.data
        op = instr.operation
        name = pyconvert(String, op.name)
        op_names[name] = get(op_names, name, 0) + 1

        qargs = instr.qubits
        if name == "u"
            @assert length(qargs) == 1
            q = pyconvert(Int, qargs[0]._index) + 1
            θ = pyconvert(Float64, op.params[0])
            φ = pyconvert(Float64, op.params[1])
            λ = pyconvert(Float64, op.params[2])

            # Qiskit U(θ,φ,λ) = Rz(φ) Ry(θ) Rz(λ) up to global phase.
            push!(circ, PauliRotation(:Z, q)); push!(params, φ)
            push!(circ, PauliRotation(:Y, q)); push!(params, θ)
            push!(circ, PauliRotation(:Z, q)); push!(params, λ)
        elseif name == "cx"
            @assert length(qargs) == 2
            c = pyconvert(Int, qargs[0]._index) + 1
            t = pyconvert(Int, qargs[1]._index) + 1
            push!(circ, CliffordGate(:CNOT, [c, t]))
            push!(twoq_edges, (c, t))
        else
            error("Unsupported gate for local PauliPropagation conversion: $name")
        end
    end

    return circ, params, op_names, twoq_edges
end

"""
    parse_states(bitstrings, a_ref, convention)

Convert bitstrings to integer basis states for fast expectation calculations.
`convention` must be `:native` or `:reversed`.
"""
function parse_states(bitstrings::Vector{String}, amps::Vector{ComplexF64}, convention::Symbol)
    @assert convention in (:native, :reversed, :native_inverted, :reversed_inverted)
    reverse_bits = convention in (:reversed, :reversed_inverted)
    invert_bits = convention in (:native_inverted, :reversed_inverted)
    n = length(bitstrings[1])
    states = Vector{UInt128}(undef, length(bitstrings))
    amp_dict = Dict{UInt128,ComplexF64}()

    for (k, bs) in enumerate(bitstrings)
        @assert length(bs) == n
        x = UInt128(0)
        for q in 1:n
            idx = reverse_bits ? (n - q + 1) : q
            b = bs[idx] == '1'
            if invert_bits
                b = !b
            end
            if b
                x |= (UInt128(1) << (q - 1))
            end
        end
        states[k] = x
        amp_dict[x] = amps[k]
    end
    return n, states, amp_dict
end

@inline function _observable_phase_and_flip(x::UInt128, symbols::Vector{Symbol}, qinds::Vector{Int})
    phase = 1.0 + 0.0im
    flipmask = UInt128(0)
    @inbounds for (sym, q) in zip(symbols, qinds)
        bit = ((x >> (q - 1)) & UInt128(1)) == UInt128(1)
        if sym == :Z
            if bit
                phase = -phase
            end
        elseif sym == :X
            flipmask |= (UInt128(1) << (q - 1))
        elseif sym == :Y
            flipmask |= (UInt128(1) << (q - 1))
            phase *= bit ? -im : im
        elseif sym == :I
            nothing
        else
            error("Unsupported Pauli symbol: $sym")
        end
    end
    return phase, flipmask
end

function expectation_from_sparse_states(
    symbols::Vector{Symbol},
    qinds::Vector{Int},
    states::Vector{UInt128},
    amp_dict::Dict{UInt128,ComplexF64},
)
    acc = 0.0 + 0.0im
    @inbounds for x in states
        ax = get(amp_dict, x, 0.0 + 0.0im)
        phase, flipmask = _observable_phase_and_flip(x, symbols, qinds)
        y = x ⊻ flipmask
        ay = get(amp_dict, y, 0.0 + 0.0im)
        acc += conj(ax) * phase * ay
    end
    return acc
end

function expectation_pp(
    pp_circuit,
    pp_params::Vector{Float64},
    nqubits::Int,
    symbols::Vector{Symbol},
    qinds::Vector{Int};
    min_abs_coeff::Float64=1e-12,
)
    obs = PauliString(nqubits, symbols, qinds, 1.0)
    propagated = propagate(pp_circuit, obs, pp_params; min_abs_coeff=min_abs_coeff, heisenberg=true)
    return overlapwithzero(propagated)
end

function make_single_qubit_observables(qubits::Vector{Int})
    out = ObservableSpec[]
    for q in qubits
        push!(out, ObservableSpec("X_$q", [:X], [q], "1q"))
        push!(out, ObservableSpec("Y_$q", [:Y], [q], "1q"))
        push!(out, ObservableSpec("Z_$q", [:Z], [q], "1q"))
    end
    return out
end

function make_two_qubit_observables(pairs::Vector{Tuple{Int,Int}})
    out = ObservableSpec[]
    for (i, j) in pairs
        push!(out, ObservableSpec("XX_$(i)_$(j)", [:X, :X], [i, j], "2q"))
        push!(out, ObservableSpec("YY_$(i)_$(j)", [:Y, :Y], [i, j], "2q"))
        push!(out, ObservableSpec("ZZ_$(i)_$(j)", [:Z, :Z], [i, j], "2q"))
        push!(out, ObservableSpec("XZ_$(i)_$(j)", [:X, :Z], [i, j], "2q"))
    end
    return out
end

function choose_probe_qubits_and_pairs(qc::Py; max_qubits::Int=12, max_pairs::Int=10)
    touched = Set{Int}()
    pairs = Tuple{Int,Int}[]
    degree = Dict{Int,Int}()

    for instr in qc.data
        name = pyconvert(String, instr.operation.name)
        if name == "u"
            q = pyconvert(Int, instr.qubits[0]._index) + 1
            push!(touched, q)
        elseif name == "cx"
            a = pyconvert(Int, instr.qubits[0]._index) + 1
            b = pyconvert(Int, instr.qubits[1]._index) + 1
            push!(touched, a); push!(touched, b)
            push!(pairs, (a, b))
            degree[a] = get(degree, a, 0) + 1
            degree[b] = get(degree, b, 0) + 1
        end
    end

    touched_vec = sort!(collect(touched))
    deg_sorted = sort!(collect(keys(degree)); by=q -> (-degree[q], q))

    qsel = Int[]
    for q in deg_sorted
        push!(qsel, q)
        length(qsel) >= min(max_qubits, 8) && break
    end
    for q in touched_vec
        if q ∉ qsel
            push!(qsel, q)
            length(qsel) >= max_qubits && break
        end
    end
    qsel = sort!(unique(qsel))

    pair_unique = unique((min(a, b), max(a, b)) for (a, b) in pairs)
    short_pairs = sort(pair_unique; by=p -> (abs(p[2] - p[1]), p[1], p[2]))
    long_pairs = reverse(sort(pair_unique; by=p -> (abs(p[2] - p[1]), p[1], p[2])))

    psel = Tuple{Int,Int}[]
    for p in short_pairs
        push!(psel, p)
        length(psel) >= max_pairs ÷ 2 && break
    end
    for p in long_pairs
        if p ∉ psel
            push!(psel, p)
        end
        length(psel) >= max_pairs && break
    end

    return qsel, psel
end

function choose_marginal_subsets(qsel::Vector{Int}, psel::Vector{Tuple{Int,Int}})
    margs = MarginalSpec[]

    for (k, (i, j)) in enumerate(psel)
        push!(margs, MarginalSpec("marg2_$(k)_$(i)_$(j)", [i, j]))
        k >= 4 && break
    end

    if length(qsel) >= 3
        push!(margs, MarginalSpec("marg3_a", [qsel[1], qsel[2], qsel[end]]))
        mid = qsel[clamp(length(qsel) ÷ 2, 1, length(qsel))]
        push!(margs, MarginalSpec("marg3_b", [qsel[1], mid, qsel[end]]))
    end

    if length(qsel) >= 4
        q2 = qsel[2]
        qmid = qsel[clamp(length(qsel) ÷ 2, 1, length(qsel))]
        push!(margs, MarginalSpec("marg4_a", [qsel[1], q2, qmid, qsel[end]]))
    end

    return margs
end

@inline function _bit_is_set(x::UInt128, q::Int)
    return ((x >> (q - 1)) & UInt128(1)) == UInt128(1)
end

function marginal_from_sparse_states(
    qinds::Vector{Int},
    states::Vector{UInt128},
    amp_dict::Dict{UInt128,ComplexF64},
)
    k = length(qinds)
    probs = zeros(Float64, 1 << k)
    @inbounds for x in states
        idx = 0
        for (j, q) in enumerate(qinds)
            if _bit_is_set(x, q)
                idx |= (1 << (j - 1))
            end
        end
        probs[idx + 1] += abs2(get(amp_dict, x, 0.0 + 0.0im))
    end
    return probs
end

function marginal_from_pp(
    pp_circuit,
    pp_params::Vector{Float64},
    nqubits::Int,
    qinds::Vector{Int};
    min_abs_coeff::Float64=1e-12,
)
    k = length(qinds)
    z_moments = zeros(ComplexF64, 1 << k)
    for mask in 0:((1 << k) - 1)
        syms = Symbol[]
        qs = Int[]
        for j in 1:k
            if ((mask >> (j - 1)) & 1) == 1
                push!(syms, :Z)
                push!(qs, qinds[j])
            end
        end
        if isempty(qs)
            z_moments[mask + 1] = 1.0 + 0im
        else
            z_moments[mask + 1] = expectation_pp(
                pp_circuit, pp_params, nqubits, syms, qs; min_abs_coeff=min_abs_coeff
            )
        end
    end

    probs = zeros(Float64, 1 << k)
    for b in 0:((1 << k) - 1)
        acc = 0.0 + 0.0im
        for z in 0:((1 << k) - 1)
            phase = isodd(count_ones(UInt(z & b))) ? -1.0 : 1.0
            acc += phase * z_moments[z + 1]
        end
        probs[b + 1] = real(acc) / (2.0^k)
    end
    return probs
end

function metrics(errors::Vector{Float64})
    return (
        mae = mean(errors),
        maxae = maximum(errors),
        rmse = sqrt(mean(abs2, errors)),
    )
end

function summarize_by_family(obs_rows::Vector{NamedTuple})
    families = unique(row.family for row in obs_rows)
    out = Dict{String,Any}()
    for fam in families
        errs = [row.abs_err for row in obs_rows if row.family == fam]
        out[fam] = metrics(errs)
    end
    return out
end

function best_and_worst(obs_rows::Vector{NamedTuple}; nshow::Int=5)
    sorted_rows = sort(obs_rows; by=row -> row.abs_err)
    best = first(sorted_rows, min(nshow, length(sorted_rows)))
    worst = last(sorted_rows, min(nshow, length(sorted_rows)))
    return best, reverse(worst)
end

function write_observable_csv(path::String, obs_rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, "name,family,real_ref,imag_ref,real_pp,imag_pp,abs_err")
        for r in obs_rows
            @printf(
                io,
                "%s,%s,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                r.name, r.family,
                real(r.ref), imag(r.ref),
                real(r.pp), imag(r.pp),
                r.abs_err,
            )
        end
    end
end

function write_marginal_csv(path::String, marg_rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, "name,state_index,p_ref,p_pp,abs_err")
        for r in marg_rows
            for idx in eachindex(r.pref)
                @printf(io, "%s,%d,%.16e,%.16e,%.16e\n", r.name, idx - 1, r.pref[idx], r.pp[idx], abs(r.pref[idx] - r.pp[idx]))
            end
        end
    end
end
