module DigitalTJM

using LinearAlgebra
using Base.Threads
using TensorOperations
using Random
using ..MPSModule
using ..MPOModule
using ..GateLibrary
using ..NoiseModule
using ..DissipationModule
using ..SimulationConfigs
using ..Algorithms
using ..BUGModule
using ..StochasticProcessModule
import ..Timing
using ..Timing: @t

export DigitalGate, DigitalCircuit, RepeatedDigitalCircuit, add_gate!, process_circuit, run_digital_tjm, TJMOptions,
       enable_timing!, set_timing_print_each_call!, reset_timing!, print_timing_summary!

# SRC: keep Python default cutoff for the randomized contraction itself.
# Any user-provided `truncation_threshold` is enforced *afterwards* via standard
# SVD-based `MPSModule.truncate!` so the bond Schmidt spectra satisfy the same
# truncation condition as TEBD/TDVP/BUG.
const SRC_DEFAULT_CUTOFF = 1e-6

"""
    enable_timing!(flag::Bool=true)

Enable timing collection for DigitalTJM execution (also enables deep timings in
TDVP / dissipation / stochastic submodules).
"""
enable_timing!(flag::Bool=true) = Timing.enable_timing!(flag)

set_timing_print_each_call!(flag::Bool=true) = Timing.set_timing_print_each_call!(flag)
reset_timing!() = Timing.reset_timing!()
print_timing_summary!(; header::AbstractString="DigitalTJM timing summary", top::Int=20) =
    Timing.print_timing_summary!(; header=header, top=top)

# --- Data Structures ---

struct DigitalGate
    op::AbstractOperator
    sites::Vector{Int}
    generator::Union{Vector{<:AbstractMatrix{ComplexF64}}, Nothing}
end

mutable struct DigitalCircuit
    num_qubits::Int
    gates::Vector{DigitalGate}
    layers::Vector{Vector{DigitalGate}} # Optional: processed layers
end

"""
    RepeatedDigitalCircuit(step::DigitalCircuit, repeats::Int)

Represents a circuit obtained by repeating the same `step` circuit `repeats` times.

This avoids materializing `repeats` copies of an identical gate list, which is
important for very large systems (e.g. IBM127-style circuits).
"""
struct RepeatedDigitalCircuit
    step::DigitalCircuit
    repeats::Int
end

function DigitalCircuit(n::Int)
    return DigitalCircuit(n, DigitalGate[], Vector{Vector{DigitalGate}}())
end

function add_gate!(circ::DigitalCircuit, op::AbstractOperator, sites::Vector{Int}; generator=nothing)
    push!(circ.gates, DigitalGate(op, sites, generator))
end

# --- Options ---

"""
    TJMOptions(; local_method=:TDVP,
                long_range_method=:TDVP,
                tdvp_truncation_timing=:during,
                bug_truncation_granularity=:after_sweep)

Algorithm options for `run_digital_tjm` (circuit simulation).

## Methods

- **`local_method` / `long_range_method`**:
  - `:TDVP`: window evolution via `Algorithms.two_site_tdvp!` (circuit-directed sweep)
  - `:BUG`: window evolution via `BUGModule.bug_second_order!`
  - `:TEBD`: apply 2-qubit gate directly (plus SWAP network for long-range gates)
  - `:ZIPUP`: apply gate as MPO×MPS via `MPOModule.apply_zipup!` (2-qubit gates + long-range 2-qubit gates via MPO;
    contiguous multi-qubit gates via MPO)
  - `:SRC`: apply gate as MPO×MPS via randomized contraction + post-SVD compression

## Truncation controls

- **`tdvp_truncation_timing`** (`:during` | `:after_window`):
  - Affects both `:TDVP` and `:BUG` window evolutions.
  - `:during`: truncate within the evolution (default; cheaper, lower peak χ).
  - `:after_window`: defer threshold-based truncation until after the whole window step
    (can reduce truncation artifacts but may increase peak χ).

- **`bug_truncation_granularity`** (`:after_sweep` | `:after_site`) (BUG-only):
  - Only relevant when `tdvp_truncation_timing == :during`.
  - `:after_sweep`: truncate once per BUG half-step sweep (default; cheapest).
  - `:after_site`: truncate after every local BUG site update (tight bond control; more SVDs).
"""
struct TJMOptions
    local_method::Symbol # :TEBD | :TDVP | :BUG | :ZIPUP | :SRC
    long_range_method::Symbol # :TEBD | :TDVP | :BUG | :ZIPUP | :SRC
    tdvp_truncation_timing::Symbol # :during (default) | :after_window
    bug_truncation_granularity::Symbol # :after_sweep (default) | :after_site
end

# Default constructor
TJMOptions(; local_method::Symbol=:TDVP,
             long_range_method::Symbol=:TDVP,
             tdvp_truncation_timing::Symbol=:during,
             bug_truncation_granularity::Symbol=:after_sweep) =
    TJMOptions(local_method, long_range_method, tdvp_truncation_timing, bug_truncation_granularity)

# ==============================================================================
# SRC helpers: construct MPO for (possibly long-range) multi-qubit gates.
# We apply these MPOs using `Algorithms.random_contraction`.
# ==============================================================================

@inline function _id_site_tensor(::Type{T}, d::Int) where {T}
    W = zeros(T, 1, d, d, 1)
    @inbounds for p in 1:d
        W[1, p, p, 1] = one(T)
    end
    return W
end

"""
    _mpo_from_two_qubit_gate_matrix(U, i, j, L; d=2)

Build an MPO (length `L`) for a 2-qubit operator `U` acting on sites `i<j`
with identity elsewhere, using an operator-Schmidt (SVD) decomposition.

This avoids constructing a dense 2^(j-i+1) × 2^(j-i+1) operator for long-range gates.
"""
function _mpo_from_two_qubit_gate_matrix(U::AbstractMatrix{ComplexF64},
                                         i::Int,
                                         j::Int,
                                         L::Int;
                                         d::Int=2)
    @assert 1 ≤ i < j ≤ L
    @assert size(U, 1) == d^2 && size(U, 2) == d^2

    # Tensor convention consistent with existing TEBD code:
    # U_tensor[p1_out, p2_out, p1_in, p2_in]
    Uten = reshape(Matrix(U), d, d, d, d)
    # Form matrix M[(p1_out,p1_in), (p2_out,p2_in)]
    X = permutedims(Uten, (1, 3, 2, 4))                 # (p1_out,p1_in,p2_out,p2_in)
    M = reshape(X, d*d, d*d)

    F = svd(M)
    r = length(F.S)

    tensors = Vector{Array{ComplexF64,4}}(undef, L)
    # Identities outside the support
    for s in 1:(i-1)
        tensors[s] = _id_site_tensor(ComplexF64, d)
    end
    for s in (j+1):L
        tensors[s] = _id_site_tensor(ComplexF64, d)
    end

    # Site i: (1, d, d, r) with Aα
    Wi = zeros(ComplexF64, 1, d, d, r)
    @inbounds for α in 1:r
        Avec = F.U[:, α] * F.S[α]          # absorb σ into left factor
        Aα = reshape(Avec, d, d)           # (pout, pin)
        Wi[1, :, :, α] .= Aα
    end
    tensors[i] = Wi

    # Middle identities: (r, d, d, r) diag in bond
    if j > i + 1
        Wmid = zeros(ComplexF64, r, d, d, r)
        @inbounds for α in 1:r
            for p in 1:d
                Wmid[α, p, p, α] = 1.0
            end
        end
        for s in (i+1):(j-1)
            tensors[s] = Wmid
        end
    end

    # Site j: (r, d, d, 1) with Bα
    Wj = zeros(ComplexF64, r, d, d, 1)
    @inbounds for α in 1:r
        Bvec = F.Vt[α, :]                 # row vector
        Bα = reshape(Bvec, d, d)
        Wj[α, :, :, 1] .= Bα
    end
    tensors[j] = Wj

    return MPO(L, tensors, fill(d, L), 0)
end

"""
    _mpo_from_contiguous_k_qubit_gate_matrix(U, sites, L; d=2)

Build an MPO (length `L`) for a k-qubit operator `U` acting on a *contiguous*
block `sites = [s, s+1, ..., s+k-1]`, with identity elsewhere.

Uses a TT-SVD (successive SVD) on the operator viewed as an MPO with per-site
physical dimension `d` on both input/output legs.
"""
function _mpo_from_contiguous_k_qubit_gate_matrix(U::AbstractMatrix{ComplexF64},
                                                  sites::Vector{Int},
                                                  L::Int;
                                                  d::Int=2)
    @assert !isempty(sites)
    ss = sort(unique(sites))
    k = length(ss)
    @assert k ≥ 3
    @assert ss[end] - ss[1] + 1 == k "k-qubit MPO builder currently requires contiguous sites"
    @assert 1 ≤ ss[1] && ss[end] ≤ L
    @assert size(U, 1) == d^k && size(U, 2) == d^k

    # Operator tensor: (pout1..poutk, pin1..pink)
    Uten0 = reshape(Matrix(U), ntuple(_->d, k)..., ntuple(_->d, k)...)
    # Interleave as (pout1,pin1,pout2,pin2,...)
    perm = Vector{Int}(undef, 2k)
    @inbounds for s in 1:k
        perm[2s-1] = s
        perm[2s] = k + s
    end
    T = permutedims(Uten0, perm)

    # TT-SVD over site pairs (pout_s,pin_s) of dimension d^2.
    Ws_block = Vector{Array{ComplexF64,4}}(undef, k)
    rank = 1
    X = T
    for s in 1:(k-1)
        left_dim = rank * d * d
        M = reshape(X, left_dim, :)
        F = svd(M)
        rnew = length(F.S)
        Umat = F.U[:, 1:rnew]
        Svec = F.S[1:rnew]
        Vt = F.Vt[1:rnew, :]
        Ws_block[s] = reshape(Umat, rank, d, d, rnew)
        X = reshape(Diagonal(Svec) * Vt, rnew, ntuple(_->d, 2*(k-s))...)
        rank = rnew
    end
    Ws_block[k] = reshape(X, rank, d, d, 1)

    tensors = Vector{Array{ComplexF64,4}}(undef, L)
    for s in 1:(ss[1]-1)
        tensors[s] = _id_site_tensor(ComplexF64, d)
    end
    for (tidx, site) in enumerate(ss)
        tensors[site] = Ws_block[tidx]
    end
    for s in (ss[end]+1):L
        tensors[s] = _id_site_tensor(ComplexF64, d)
    end

    return MPO(L, tensors, fill(d, L), 0)
end

# --- Circuit Processing ---

"""
    process_circuit(circuit::DigitalCircuit)

Process the circuit gates into layers of commuting operations.
Returns `(layers, barrier_map)`.
"""
function process_circuit(circuit::DigitalCircuit)
    # Simple greedy layering
    layers = Vector{Vector{DigitalGate}}()
    barrier_map = Dict{Int, Vector{String}}()
    
    current_layer = DigitalGate[]
    busy_qubits = Set{Int}()
    
    layer_idx = 1
    
    for gate in circuit.gates
        # Check for Barrier
        if gate.op isa GateLibrary.Barrier
            # If barrier, finish current layer
            if !isempty(current_layer)
                push!(layers, current_layer)
                current_layer = DigitalGate[]
                busy_qubits = Set{Int}()
                layer_idx += 1
            end
            
            # Record barrier
            idx = !isempty(layers) ? length(layers) : 0
            if !haskey(barrier_map, idx)
                barrier_map[idx] = String[]
            end
            push!(barrier_map[idx], gate.op.label)
            continue
        end
        
        # Check commutativity / qubit overlap
        overlap = false
        for q in gate.sites
            if q in busy_qubits
                overlap = true
                break
            end
        end
        
        if overlap
            # Push current layer
            push!(layers, current_layer)
            current_layer = DigitalGate[]
            busy_qubits = Set{Int}()
            layer_idx += 1
        end
        
        # Add to current
        push!(current_layer, gate)
        for q in gate.sites
            push!(busy_qubits, q)
        end
    end
    
    if !isempty(current_layer)
        push!(layers, current_layer)
    end
    
    return layers, barrier_map
end

@inline function _has_sample_barrier(barrier_map::Dict{Int, Vector{String}}, idx::Int)
    if !haskey(barrier_map, idx)
        return false
    end
    for label in barrier_map[idx]
        if uppercase(label) == "SAMPLE_OBSERVABLES"
            return true
        end
    end
    return false
end

@inline function _sample_plan(barrier_map::Dict{Int, Vector{String}}, num_layers::Int)
    sample_at_start = _has_sample_barrier(barrier_map, 0)
    sample_after = Int[]
    for l in 1:num_layers
        if _has_sample_barrier(barrier_map, l)
            push!(sample_after, l)
        end
    end
    return sample_at_start, sample_after
end

# --- Noise Helpers ---

function create_local_noise_model(noise_model::NoiseModel{T}, site1::Int, site2::Int) where T
    affected_sites = Set([site1, site2])
    local_procs = Vector{AbstractNoiseProcess{T}}()
    for proc in noise_model.processes
        p_sites = proc.sites
        if length(p_sites) == 1
            if p_sites[1] == site1 || p_sites[1] == site2
                push!(local_procs, proc)
            end
        elseif length(p_sites) == 2
            if (p_sites[1] == site1 && p_sites[2] == site2) || (p_sites[1] == site2 && p_sites[2] == site1)
                 push!(local_procs, proc)
            end
        end
    end
    return NoiseModel(local_procs)
end

# --- Helpers ---

function construct_window_mpo(gate::DigitalGate, window_start::Int, window_end::Int)
    L_window = window_end - window_start + 1
    tensors = Vector{Array{ComplexF64, 4}}(undef, L_window)
    phys_dims = fill(2, L_window)
    
    s1, s2 = sort(gate.sites)
    rel_s1 = s1 - window_start + 1
    rel_s2 = s2 - window_start + 1
    
    # Get generator from gate, or from operator if not set
    gen = gate.generator
    if isnothing(gen)
        gen = GateLibrary.generator(gate.op)
    end
    coeff = GateLibrary.hamiltonian_coeff(gate.op)
    
    @assert rel_s1 >= 1 && rel_s2 <= L_window "Gate sites outside window"

    # Determine which generator element goes to which site
    # gate.sites[1] corresponds to gen[1]
    # gate.sites[2] corresponds to gen[2]
    # s1 is min(sites), s2 is max(sites)
    
    g_s1 = gen[1]
    g_s2 = gen[2]
    
    if gate.sites[1] != s1
        # If sites were sorted/swapped (e.g. [2, 1] -> s1=1, s2=2), 
        # then s1 corresponds to sites[2] -> gen[2]
        g_s1 = gen[2]
        g_s2 = gen[1]
    end
    
    for i in 1:L_window
        T = zeros(ComplexF64, 1, 2, 2, 1)
        T[1, :, :, 1] = Matrix{ComplexF64}(I, 2, 2)
        if i == rel_s1
            T[1, :, :, 1] = coeff * g_s1
        elseif i == rel_s2
            T[1, :, :, 1] = g_s2
        end
        tensors[i] = T
    end
    return MPO(L_window, tensors, phys_dims, 0)
end

function apply_single_qubit_gate!(mps::MPS, gate::DigitalGate)
    site = gate.sites[1]
    @t :matrix_1q begin
        op_mat = matrix(gate.op)
        A = mps.tensors[site]
        L, d, R = size(A)
        @t :permutedims_1q A_perm = reshape(permutedims(A, (2, 1, 3)), d, L*R)
        @t :mul_1q New_A_mat = op_mat * A_perm
        @t :permutedims_1q New_A = permutedims(reshape(New_A_mat, d, L, R), (2, 1, 3))
        mps.tensors[site] = New_A
    end
end

function apply_local_gate_exact!(mps::MPS, op::AbstractOperator, s1::Int, s2::Int, config::AbstractSimConfig)
    # Standard TEBD update for nearest neighbor gate
    # Moves orthogonality center to s1 (assumes mixed/right canonical to right of s1)
    @t :shift_orth_center MPSModule.shift_orthogonality_center!(mps, s1)
    
    A1 = mps.tensors[s1]
    A2 = mps.tensors[s2]
    
    # Contract A1 * A2
    # A1: (l1, p1, b)
    # A2: (b, p2, r2)
    @t :contract_theta @tensor Theta[l1, p1, p2, r2] := A1[l1, p1, k] * A2[k, p2, r2]
    
    # Contract with Gate
    # op_mat: 4x4 (acting on p1, p2)
    # Reshape op_mat to (p1_out, p2_out, p1_in, p2_in)
    @t :matrix_2q op_mat = matrix(op)
    op_tensor = reshape(op_mat, 2, 2, 2, 2)
    
    @t :contract_gate @tensor Theta_prime[l1, p1_out, p2_out, r2] := op_tensor[p1_out, p2_out, p1_in, p2_in] * Theta[l1, p1_in, p2_in, r2]
    
    # SVD and Truncate
    l1_dim, p1_dim, p2_dim, r2_dim = size(Theta_prime)
    @t :reshape_theta Theta_matrix = reshape(Theta_prime, l1_dim * p1_dim, p2_dim * r2_dim)
    
    @t :svd F = svd(Theta_matrix)
    
    # Truncation Logic
    threshold = config.truncation_threshold
    max_bond = config.max_bond_dim
    
    # Truncation mode (consistent with TDVP/BUG/MPS.truncate!):
    # - if `threshold >= 0`: relative discarded weight  sum(discarded S^2)/sum(all S^2) <= threshold
    # - if `threshold < 0`: absolute discarded weight  sum(discarded S^2) <= -threshold
    total_sq = sum(abs2, F.S)
    discarded_sq = 0.0
    keep_rank = length(F.S)
    min_keep = 2
    @t :truncation_loop for k in length(F.S):-1:1
        discarded_sq += F.S[k]^2
        if threshold < 0
            if discarded_sq >= -threshold
                keep_rank = max(k, min_keep)
                break
            end
        else
            frac = (total_sq == 0.0) ? 0.0 : (discarded_sq / total_sq)
            if frac >= threshold
                keep_rank = max(k, min_keep)
                break
            end
        end
    end
    keep = clamp(keep_rank, 1, max_bond)
    
    U = F.U[:, 1:keep]
    S = F.S[1:keep]
    Vt = F.Vt[1:keep, :]
    
    # Update MPS
    # A1 becomes U (Left Canonical)
    @t :update_mps mps.tensors[s1] = reshape(U, l1_dim, p1_dim, keep)
    
    # A2 becomes S * V (Right Canonical? No, Center)
    @t :update_mps mps.tensors[s2] = reshape(Diagonal(S) * Vt, keep, p2_dim, r2_dim)
    mps.orth_center = s2
end

function apply_window!(state::MPS, gate::DigitalGate, sim_params::AbstractSimConfig, alg_options::TJMOptions; rng::AbstractRNG=Random.default_rng())
    sites = sort(gate.sites)
    k = length(sites)
    @assert k ≥ 2 "apply_window! expects a multi-qubit gate (got sites=$(gate.sites))."

    if k == 2
        s1 = sites[1]
        s2 = sites[2]
        is_long_range = (s2 > s1 + 1)
        method = is_long_range ? alg_options.long_range_method : alg_options.local_method

        if method == :TEBD
            if is_long_range
                # Swap Network + Local TEBD
                #
                # Swap Path: (s2-1, s2), (s2-2, s2-1), ..., (s1+1, s1+2)
                # s2 moves LEFT to s1+1
                for kk in (s2-1):-1:(s1+1)
                    @t :apply_swap apply_local_gate_exact!(state, SWAPGate(), kk, kk+1, sim_params)
                end

                # Apply gate on (s1, s1+1)
                @t :apply_local_gate_exact apply_local_gate_exact!(state, gate.op, s1, s1+1, sim_params)

                # Unwind Swaps: (s1+1, s1+2), ..., (s2-1, s2)
                # s2 moves RIGHT back to s2
                for kk in (s1+1):(s2-1)
                    @t :apply_swap apply_local_gate_exact!(state, SWAPGate(), kk, kk+1, sim_params)
                end
            else
                # Nearest Neighbor: Direct
                @t :apply_local_gate_exact apply_local_gate_exact!(state, gate.op, s1, s2, sim_params)
            end
        elseif method == :TDVP
            padding = 0
            win_start = max(1, s1 - padding)
            win_end = min(state.length, s2 + padding)
            @t :shift_orth_center MPSModule.shift_orthogonality_center!(state, win_start)
            win_len = win_end - win_start + 1
            @t :window_slice begin
                win_tensors = state.tensors[win_start:win_end]
                win_phys = state.phys_dims[win_start:win_end]
                short_state = MPS(win_len, win_tensors, win_phys, 1)
            end
            @t :construct_window_mpo short_mpo = construct_window_mpo(gate, win_start, win_end)

            # Use StrongMeasurementConfig to trigger the efficient Directed Circuit Sweep
            # instead of the Symmetric Hamiltonian Sweep triggered by TimeEvolutionConfig.
            trunc_timing = alg_options.tdvp_truncation_timing
            @assert trunc_timing === :during || trunc_timing === :after_window

            # Optional: defer threshold-based truncation until after the whole window evolution.
            # We still cap intermediate bond dimensions by `max_bond_dim` to avoid blow-ups.
            #
            # Implementation detail: `Algorithms.split_mps_tensor_svd` interprets
            #   threshold >= 0 as relative discarded weight and truncates when frac >= threshold.
            # Setting threshold = ±Inf disables that truncation (but keeps the hard chi cap).
            th = sim_params.truncation_threshold
            th_tdvp = if trunc_timing === :after_window
                (th >= 0) ? Inf : -Inf
            else
                th
            end

            gate_config = StrongMeasurementConfig(Observable[];
                                                 max_bond_dim=sim_params.max_bond_dim,
                                                 truncation_threshold=th_tdvp)

            @t :two_site_tdvp two_site_tdvp!(short_state, short_mpo, gate_config)

            if trunc_timing === :after_window
                # Single post-pass compression of the *short* MPS using the same threshold semantics
                # as the usual in-sweep truncation (relative if th>=0, absolute if th<0).
                @t :post_truncate MPSModule.truncate!(short_state; threshold=th, max_bond_dim=sim_params.max_bond_dim)
            end

            # Ensure a well-defined canonical gauge for writeback: make the window left-canonical
            # with the orthogonality center at the right boundary of the window.
            @t :window_canonicalize MPSModule.shift_orthogonality_center!(short_state, win_len)

            @t :window_writeback state.tensors[win_start:win_end] .= short_state.tensors
            state.orth_center = win_end
        elseif method == :BUG
            padding = 0
            win_start = max(1, s1 - padding)
            win_end = min(state.length, s2 + padding)
            @t :shift_orth_center MPSModule.shift_orthogonality_center!(state, win_start)
            win_len = win_end - win_start + 1
            @t :window_slice begin
                win_tensors = state.tensors[win_start:win_end]
                win_phys = state.phys_dims[win_start:win_end]
                short_state = MPS(win_len, win_tensors, win_phys, 1)
            end
            @t :construct_window_mpo short_mpo = construct_window_mpo(gate, win_start, win_end)

            trunc_timing = alg_options.tdvp_truncation_timing
            @assert trunc_timing === :during || trunc_timing === :after_window

            # BUG uses a time step dt=1.0 to apply exp(-i * 1 * H_gate).
            # For 2nd order BUG, we internally use dt/2 per sweep direction.
            bug_cfg = TimeEvolutionConfig(Observable[], 1.0;
                                          dt=1.0,
                                          num_traj=1,
                                          sample_timesteps=false,
                                          max_bond_dim=sim_params.max_bond_dim,
                                          truncation_threshold=sim_params.truncation_threshold,
                                          order=2)

            @t :bug_second_order BUGModule.bug_second_order!(short_state, short_mpo, bug_cfg;
                                                            numiter_lanczos=25,
                                                            truncation_timing=trunc_timing,
                                                            truncation_granularity=alg_options.bug_truncation_granularity)

            # Canonicalize for writeback: make window left-canonical with center at right boundary.
            @t :window_canonicalize MPSModule.shift_orthogonality_center!(short_state, win_len)
            @t :window_writeback state.tensors[win_start:win_end] .= short_state.tensors
            state.orth_center = win_end
        elseif method == :ZIPUP
            op_mat = Matrix{ComplexF64}(matrix(gate.op))
            L = state.length
            d = state.phys_dims[s1]
            @assert state.phys_dims[s2] == d "ZIPUP expects equal physical dimensions on all involved sites."
            mpo_gate = _mpo_from_two_qubit_gate_matrix(op_mat, s1, s2, L; d=d)

            # Zip-up has only an absolute `svd_min` control internally; we enforce the user’s
            # threshold semantics via a standard post-pass SVD compression.
            @t :zipup_apply MPOModule.apply_zipup!(state, mpo_gate; chi_max=sim_params.max_bond_dim, svd_min=0.0)
            @t :zipup_post_truncate MPSModule.truncate!(state; threshold=sim_params.truncation_threshold,
                                                       max_bond_dim=sim_params.max_bond_dim)
        elseif method == :SRC
            # Apply a (possibly long-range) 2-qubit gate as an MPO×MPS product, compressed via SRC.
            op_mat = Matrix{ComplexF64}(matrix(gate.op))
            L = state.length
            mpo_gate = _mpo_from_two_qubit_gate_matrix(op_mat, s1, s2, L; d=state.phys_dims[1])

            # Keep the SRC contraction cutoff at the Python default and enforce the
            # user-provided truncation threshold via a standard post-pass SVD compression.
            stop = Algorithms.Cutoff(SRC_DEFAULT_CUTOFF;
                                     mindim=sim_params.min_bond_dim,
                                     maxdim=sim_params.max_bond_dim)
            new_state = Algorithms.random_contraction(mpo_gate, state; stop=stop, rng=rng)
            @t :src_post_truncate MPSModule.truncate!(new_state; threshold=sim_params.truncation_threshold,
                                                     max_bond_dim=sim_params.max_bond_dim)
            state.tensors = new_state.tensors
            state.orth_center = new_state.orth_center
        else
            error("Unknown method $(method). Expected :TEBD, :TDVP, :BUG, :ZIPUP, or :SRC.")
        end

        return nothing
    end

    # --------------------------------------------------------------------------
    # Multi-qubit gates (k ≥ 3): only supported for contiguous site sets via MPO.
    # --------------------------------------------------------------------------
    @assert all(@view(sites[2:end]) .== @view(sites[1:end-1]) .+ 1) "Multi-qubit gates are only supported on contiguous sites (got sites=$(gate.sites))."

    method = alg_options.long_range_method
    if method == :ZIPUP || method == :SRC
        op_mat = Matrix{ComplexF64}(matrix(gate.op))
        L = state.length
        d = state.phys_dims[sites[1]]
        @inbounds for s in sites
            @assert state.phys_dims[s] == d "Multi-qubit MPO application currently expects uniform physical dimension across the gate window."
        end

        mpo_gate = _mpo_from_contiguous_k_qubit_gate_matrix(op_mat, sites, L; d=d)

        if method == :ZIPUP
            @t :zipup_apply MPOModule.apply_zipup!(state, mpo_gate; chi_max=sim_params.max_bond_dim, svd_min=0.0)
            @t :zipup_post_truncate MPSModule.truncate!(state; threshold=sim_params.truncation_threshold,
                                                       max_bond_dim=sim_params.max_bond_dim)
        else
            stop = Algorithms.Cutoff(SRC_DEFAULT_CUTOFF;
                                     mindim=sim_params.min_bond_dim,
                                     maxdim=sim_params.max_bond_dim)
            new_state = Algorithms.random_contraction(mpo_gate, state; stop=stop, rng=rng)
            @t :src_post_truncate MPSModule.truncate!(new_state; threshold=sim_params.truncation_threshold,
                                                     max_bond_dim=sim_params.max_bond_dim)
            state.tensors = new_state.tensors
            state.orth_center = new_state.orth_center
        end
    else
        error("Multi-qubit gates (>2 sites) are only supported for long_range_method = :ZIPUP or :SRC (got $(method)).")
    end

    return nothing
end

# --- Main Runner ---

function run_digital_tjm(initial_state::MPS, circuit::DigitalCircuit, 
                            noise_model::Union{NoiseModel, Nothing}, 
                            sim_params::AbstractSimConfig;
                            alg_options::TJMOptions = TJMOptions(local_method=:TDVP, long_range_method=:TDVP),
                            rng::AbstractRNG=Random.default_rng())

    local ts = Timing.begin_scope!()
    try
        state = @t :deepcopy_state deepcopy(initial_state)
        layers, barrier_map = @t :process_circuit process_circuit(circuit)
        num_layers = length(layers)
        num_obs = length(sim_params.observables)

        sample_at_start, sample_after = _sample_plan(barrier_map, num_layers)
        num_steps = (sample_at_start ? 1 : 0) + length(sample_after)
        if num_steps == 0
            num_steps = 1
        end
    
    results = zeros(ComplexF64, num_obs, num_steps)
    bond_dims = zeros(Int, num_steps)
    current_meas_idx = 1
    
    function measure!(idx)
        @t :measure begin
            for (i, obs) in enumerate(sim_params.observables)
                results[i, idx] = @t :expect SimulationConfigs.expect(state, obs)
            end
            bond_dims[idx] = @t :write_max_bond_dim MPSModule.write_max_bond_dim(state)
        end
    end
    
    if sample_at_start
        measure!(current_meas_idx)
        current_meas_idx += 1
    end
    
    for (l_idx, layer) in enumerate(layers)
        for gate in layer
            if length(gate.sites) == 1
                @t :apply_single_qubit_gate apply_single_qubit_gate!(state, gate)
            end
        end
        for gate in layer
            if length(gate.sites) ≥ 2
                @t :apply_window apply_window!(state, gate, sim_params, alg_options; rng=rng)
                if !isnothing(noise_model) && !isempty(noise_model.processes)
                    sites = sort(gate.sites)
                    s1 = sites[1]
                    s2 = sites[end]
                    local_noise = @t :create_local_noise_model create_local_noise_model(noise_model, s1, s2)
                    if !isempty(local_noise.processes)
                        @t :apply_dissipation apply_dissipation(state, local_noise, 1.0, sim_params)
                        @t :stochastic_process stochastic_process!(state, local_noise, 1.0, sim_params)
                    else
                         @t :normalize MPSModule.normalize!(state)
                    end
                else
                     @t :normalize MPSModule.normalize!(state)
                end
            end
        end

        # Progress logging (overwrite line with \r)
        current_bond = @t :write_max_bond_dim MPSModule.write_max_bond_dim(state)
        print("\r\tLayer $l_idx/$num_layers | Max Bond: $current_bond")
        flush(stdout)

        if l_idx in sample_after
            measure!(current_meas_idx)
            current_meas_idx += 1
        end
    end
    print("\n") # Newline after finishing all layers of this trajectory
    
    # If the circuit did not request sampling (no SAMPLE_OBSERVABLES barriers),
    # still produce at least one measurement for compatibility.
    if !sample_at_start && isempty(sample_after) && num_obs > 0
        measure!(1)
    end
        # If the circuit did not request sampling (no SAMPLE_OBSERVABLES barriers),
        # still produce at least one measurement for compatibility.
        if !sample_at_start && isempty(sample_after) && num_obs > 0
            measure!(1)
        end
    
    return state, results, bond_dims
    finally
        Timing.end_scope!(ts; header="DigitalTJM per-trajectory timing")
    end
end

function run_digital_tjm(initial_state::MPS, circuit::RepeatedDigitalCircuit,
                         noise_model::Union{NoiseModel, Nothing},
                         sim_params::AbstractSimConfig;
                         alg_options::TJMOptions = TJMOptions(local_method=:TDVP, long_range_method=:TDVP),
                         rng::AbstractRNG=Random.default_rng())
    local ts = Timing.begin_scope!()
    try
        state = @t :deepcopy_state deepcopy(initial_state)
        layers, barrier_map = @t :process_circuit process_circuit(circuit.step)
        step_layers = length(layers)
        repeats = circuit.repeats

        num_obs = length(sim_params.observables)
        sample_at_start, sample_after = _sample_plan(barrier_map, step_layers)

        num_steps = (sample_at_start ? 1 : 0) + repeats * length(sample_after)
        if num_steps == 0
            num_steps = 1
        end

        results = zeros(ComplexF64, num_obs, num_steps)
        bond_dims = zeros(Int, num_steps)
        current_meas_idx = 1

        function measure!(idx)
            @t :measure begin
                for (i, obs) in enumerate(sim_params.observables)
                    results[i, idx] = @t :expect SimulationConfigs.expect(state, obs)
                end
                bond_dims[idx] = @t :write_max_bond_dim MPSModule.write_max_bond_dim(state)
            end
        end

        if sample_at_start
            measure!(current_meas_idx)
            current_meas_idx += 1
        end

        total_layers = repeats * step_layers
        for rep in 1:repeats
            for (l_idx, layer) in enumerate(layers)
                for gate in layer
                    if length(gate.sites) == 1
                        @t :apply_single_qubit_gate apply_single_qubit_gate!(state, gate)
                    end
                end
                for gate in layer
                    if length(gate.sites) ≥ 2
                        @t :apply_window apply_window!(state, gate, sim_params, alg_options; rng=rng)
                        if !isnothing(noise_model) && !isempty(noise_model.processes)
                            sites = sort(gate.sites)
                            s1 = sites[1]
                            s2 = sites[end]
                            local_noise = @t :create_local_noise_model create_local_noise_model(noise_model, s1, s2)
                            if !isempty(local_noise.processes)
                                @t :apply_dissipation apply_dissipation(state, local_noise, 1.0, sim_params)
                                @t :stochastic_process stochastic_process!(state, local_noise, 1.0, sim_params)
                            else
                                @t :normalize MPSModule.normalize!(state)
                            end
                        else
                            @t :normalize MPSModule.normalize!(state)
                        end
                    end
                end

                global_layer = (rep - 1) * step_layers + l_idx
                current_bond = @t :write_max_bond_dim MPSModule.write_max_bond_dim(state)
                print("\r\tLayer $global_layer/$total_layers | Max Bond: $current_bond")
                flush(stdout)

                if l_idx in sample_after
                    measure!(current_meas_idx)
                    current_meas_idx += 1
                end
            end
        end
        print("\n")

        if num_steps == 1 && num_obs > 0 && !sample_at_start && isempty(sample_after)
            measure!(1)
        end

        return state, results, bond_dims
    finally
        Timing.end_scope!(ts; header="DigitalTJM per-trajectory timing")
    end
end

end # module
