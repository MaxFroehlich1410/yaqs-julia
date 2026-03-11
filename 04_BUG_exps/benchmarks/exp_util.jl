module BUGExpUtil

using LinearAlgebra
using Statistics
using Printf

using Yaqs

const BUG = Yaqs.BUGModule
const Algo = Yaqs.Algorithms
const MPSMod = Yaqs.MPSModule
const MPOMod = Yaqs.MPOModule
const Cfg = Yaqs.SimulationConfigs
const GL = Yaqs.GateLibrary
const Timing = Yaqs.Timing
using Yaqs.Timing: @t

export parse_kv_args,
       mid_site,
       dense_expect_z_site,
       run_method_expect_z_site,
       rms_error,
       min_abs_error,
       max_abs_error

@inline mid_site(L::Int) = (L + 1) ÷ 2

function _print_runtime_table(ts::Timing.TimingStats; header::AbstractString="Timing summary", top::Int=50)
    total_ns = UInt64(0)
    @inbounds for v in values(ts.times_ns)
        total_ns += v
    end

    pairs = collect(ts.times_ns)
    sort!(pairs; by = p -> p[2], rev = true)

    @printf "\n\t%s (total %.3f ms)\n" header (total_ns / 1e6)
    nshow = min(top, length(pairs))
    for i in 1:nshow
        key, tns = pairs[i]
        c = get(ts.counts, key, 0)
        ms_per = c > 0 ? (tns / 1e6) / c : 0.0
        frac = total_ns > 0 ? 100.0 * (tns / float(total_ns)) : 0.0
        @printf "\t  %-36s %10.3f ms  (%5.1f%%)  %8d calls  %9.3f ms/call\n" String(key) (tns / 1e6) frac c ms_per
    end
    return nothing
end

function parse_kv_args(args::Vector{String})
    # Minimal `--key=value` parser (also supports `--flag` => "true").
    # Also accepts `key=value` / `flag` (useful when calling `main([...])`).
    out = Dict{String, String}()
    for a in args
        s = if startswith(a, "--")
            a[3:end]
        elseif startswith(a, "-")
            continue
        else
            a
        end

        if occursin("=", s)
            k, v = split(s, "=", limit=2)
            out[k] = v
        else
            out[s] = "true"
        end
    end
    return out
end

function _dense_initial_state(L::Int, state::AbstractString)
    if state == "zeros"
        v1 = ComplexF64[1.0, 0.0]
    elseif state == "ones"
        v1 = ComplexF64[0.0, 1.0]
    elseif state == "x+"
        v1 = ComplexF64[1.0, 1.0] ./ sqrt(2)
    else
        error("Unsupported initial state: $state (supported: zeros, ones, x+)")
    end
    v = v1
    for _ in 2:L
        v = kron(v, v1)
    end
    return v
end

function _dense_op_on_site(L::Int, op::AbstractMatrix{<:Complex}, site::Int)
    @assert 1 ≤ site ≤ L
    I2 = Matrix{ComplexF64}(I, 2, 2)
    acc = (site == 1) ? ComplexF64.(op) : I2
    for s in 2:L
        acc = kron(acc, (s == site) ? ComplexF64.(op) : I2)
    end
    return acc
end

function _dense_ising_hamiltonian(L::Int; J::Real=1.0, g::Real=0.5)
    X = ComplexF64.(Matrix(GL.matrix(GL.XGate())))
    Z = ComplexF64.(Matrix(GL.matrix(GL.ZGate())))

    H = zeros(ComplexF64, 2^L, 2^L)
    for i in 1:(L - 1)
        Zi = _dense_op_on_site(L, Z, i)
        Zj = _dense_op_on_site(L, Z, i + 1)
        H .+= (-J) .* (Zi * Zj)
    end
    for i in 1:L
        Xi = _dense_op_on_site(L, X, i)
        H .+= (-g) .* Xi
    end
    return H
end

@inline function _get_param(p, i::Int)
    return isa(p, Vector) ? (i <= Base.length(p) ? p[i] : 0.0) : p
end

function _mpo_hamiltonian(model::AbstractString, L::Int;
                          # TFIM params
                          J::Real=1.0, g::Real=0.5,
                          # General params (used by `general`)
                          Jxx::Real=0.0, Jyy::Real=0.0, Jzz::Real=0.0,
                          hx::Real=0.0, hy::Real=0.0, hz::Real=0.0,
                          # Convenience params for common models
                          Delta::Real=1.0, gamma::Real=0.0)
    m = lowercase(strip(model))
    if m in ("tfim", "ising")
        return MPOMod.init_ising(L, float(J), float(g))
    elseif m in ("heisenberg", "xxx")
        return MPOMod.init_general_hamiltonian(L, float(J), float(J), float(J), float(hx), float(hy), float(hz))
    elseif m in ("xx",)
        return MPOMod.init_general_hamiltonian(L, float(J), float(J), 0.0, float(hx), float(hy), float(hz))
    elseif m in ("xy",)
        Jx = float(J) * (1 + float(gamma))
        Jy = float(J) * (1 - float(gamma))
        return MPOMod.init_general_hamiltonian(L, Jx, Jy, 0.0, float(hx), float(hy), float(hz))
    elseif m in ("xxz",)
        return MPOMod.init_general_hamiltonian(L, float(J), float(J), float(Delta) * float(J), float(hx), float(hy), float(hz))
    elseif m in ("general",)
        return MPOMod.init_general_hamiltonian(L, float(Jxx), float(Jyy), float(Jzz), float(hx), float(hy), float(hz))
    else
        error("Unsupported model=$model (supported: tfim/ising, xx, xy, xxz, heisenberg, general)")
    end
end

"""
    dense_expect_z_site(L; J, g, dt, steps, initial_state, site) -> (times, z)

Compute exact real-time evolution for the transverse-field Ising Hamiltonian
and return ⟨Z_site⟩ at each time point using dense matrices (feasible for small L).
"""
function dense_expect_z_site(L::Int;
                             J::Real,
                             g::Real,
                             dt::Real,
                             steps::Int,
                             initial_state::AbstractString="x+",
                             site::Int=mid_site(L))
    @assert steps ≥ 0
    times = collect(0:dt:(steps * dt))

    ψ = _dense_initial_state(L, initial_state)
    H = _dense_ising_hamiltonian(L; J=J, g=g)
    U = exp((-1im * dt) .* H)

    Z = ComplexF64.(Matrix(GL.matrix(GL.ZGate())))
    Zsite = _dense_op_on_site(L, Z, site)

    out = Vector{Float64}(undef, steps + 1)
    out[1] = real(dot(conj(ψ), Zsite * ψ))
    for k in 1:steps
        ψ = U * ψ
        out[k + 1] = real(dot(conj(ψ), Zsite * ψ))
    end
    return times, out
end

function _method_from_string(s::AbstractString)
    if s == "ADAPTIVE" || s == "bug"
        return :bug
    elseif s == "FIXED" || s == "fixed_bug"
        return :fixed_bug
    elseif s == "DOUBLEADAPTIVE" || s == "bug_second_order"
        return :bug_second_order
    elseif s == "DOUBLEFIXED" || s == "fixed_bug_second_order"
        return :fixed_bug_second_order
    elseif s == "HYBRID" || s == "hybrid_bug_second_order"
        return :hybrid_bug_second_order
    elseif s == "SINGLE_SITE_TDVP" || s == "single_site_tdvp"
        return :single_site_tdvp
    elseif s == "TWO_SITE_TDVP" || s == "two_site_tdvp"
        return :two_site_tdvp
    else
        error("Unknown method: $s")
    end
end

"""
    run_method_expect_z_site(method; ...) -> (times, z, wall_s)

Run one method (BUG variants or TDVP) for a small Ising chain and return ⟨Z_site⟩ vs time.
"""
function run_method_expect_z_site(method::Union{Symbol, AbstractString};
                                  L::Int,
                                  model::AbstractString="tfim",
                                  # TFIM params
                                  J::Real=1.0,
                                  g::Real=0.5,
                                  # General params (used by `general`)
                                  Jxx::Real=0.0,
                                  Jyy::Real=0.0,
                                  Jzz::Real=0.0,
                                  hx::Real=0.0,
                                  hy::Real=0.0,
                                  hz::Real=0.0,
                                  # Convenience params for common models
                                  Delta::Real=1.0,
                                  gamma::Real=0.0,
                                  dt::Real,
                                  steps::Int,
                                  initial_state::AbstractString="x+",
                                  site::Int=mid_site(L),
                                  max_bond_dim::Int=128,
                                  adaptive_pad::Int=4,
                                  truncation_mode::Symbol=:during,
                                  threshold::Real=1e-12,
                                  numiter_lanczos::Int=25,
                                  track_bond_dims::Bool=false,
                                  measure_runtime::Bool=false)
    method_sym = method isa AbstractString ? _method_from_string(method) : method
    @assert truncation_mode === :during || truncation_mode === :after_sweep

    times = collect(0:dt:(steps * dt))
    Z = ComplexF64.(Matrix(GL.matrix(GL.ZGate())))

    ψ = MPSMod.MPS(L; state=initial_state)
    H = _mpo_hamiltonian(model, L;
                         J=J, g=g,
                         Jxx=Jxx, Jyy=Jyy, Jzz=Jzz,
                         hx=hx, hy=hy, hz=hz,
                         Delta=Delta, gamma=gamma)
    cfg = Cfg.TimeEvolutionConfig(Cfg.Observable[], float(dt); dt=float(dt),
                                  max_bond_dim=max_bond_dim,
                                  truncation_threshold=float(threshold),
                                  sample_timesteps=false)

    # TDVP helper: to defer threshold-based truncation until after the step, disable the
    # in-sweep truncation rule by setting threshold to ±Inf (keeps the hard chi cap).
    tdvp_thr = if truncation_mode === :after_sweep
        (cfg.truncation_threshold >= 0) ? Inf : -Inf
    else
        cfg.truncation_threshold
    end
    cfg_tdvp = Cfg.TimeEvolutionConfig(Cfg.Observable[], float(dt); dt=float(dt),
                                       max_bond_dim=max_bond_dim,
                                       truncation_threshold=float(tdvp_thr),
                                       sample_timesteps=false)

    # Mirror the Python harness: pad selected methods.
    #
    # Important: Python pads with *zeros* (then normalizes). In Julia we previously seeded
    # padding with small noise to help 1-site TDVP explore an enlarged manifold, but that
    # noise contaminates FIXED BUG variants unless you explicitly truncate afterwards.
    #
    # Therefore:
    # - FIXED BUG variants: zero padding (noise_scale=0.0) to match Python behavior.
    # - 1-site TDVP       : keep tiny noise to allow entanglement growth.
    if method_sym in (:fixed_bug, :fixed_bug_second_order)
        MPSMod.pad_bond_dimension!(ψ, max_bond_dim; noise_scale=0.0)
    elseif method_sym === :single_site_tdvp
        MPSMod.pad_bond_dimension!(ψ, max_bond_dim; noise_scale=1e-10)
    end

    # Optional: ensure adaptive runs do not start at χ=1 (can reduce early-time artifacts).
    # We cap the pad by max_bond_dim to avoid exceeding the method's truncation ceiling.
    if method_sym in (:bug, :bug_second_order, :two_site_tdvp)
        padχ = min(adaptive_pad, max_bond_dim)
        if padχ > 1
            MPSMod.pad_bond_dimension!(ψ, padχ; noise_scale=1e-10)
        end
    end

    out = Vector{Float64}(undef, steps + 1)
    out[1] = real(MPSMod.local_expect(ψ, Z, site))

    bond_dims = track_bond_dims ? Vector{Int}(undef, steps + 1) : Int[]
    if track_bond_dims
        bond_dims[1] = MPSMod.write_max_bond_dim(ψ)
    end

    if measure_runtime
        Timing.enable_timing!(true)
        Timing.set_timing_print_each_call!(false)
        Timing.reset_timing!()
    end
    ts = measure_runtime ? Timing.begin_scope!() : nothing

    t0 = time()
    for k in 1:steps
        if method_sym === :bug
            @t :bug_step BUG.bug!(ψ, H, cfg; numiter_lanczos=numiter_lanczos)
        elseif method_sym === :fixed_bug
            @t :fixed_bug_step BUG.fixed_bug!(ψ, H, cfg; numiter_lanczos=numiter_lanczos)
            # @t :fixed_bug_truncate MPSMod.truncate!(ψ; threshold=cfg.truncation_threshold, max_bond_dim=cfg.max_bond_dim)
        elseif method_sym === :bug_second_order
            # Second-order adaptive BUG truncation timing controlled by `truncation_mode`.
            bug_trunc_timing = (truncation_mode === :after_sweep) ? :after_window : :during
            @t :bug_second_order_step BUG.bug_second_order!(ψ, H, cfg;
                                                          numiter_lanczos=numiter_lanczos,
                                                          truncation_timing=bug_trunc_timing)
        elseif method_sym === :fixed_bug_second_order
            @t :fixed_bug_second_order_step BUG.fixed_bug_second_order!(ψ, H, cfg; numiter_lanczos=numiter_lanczos)
            @t :fixed_bug_second_order_truncate MPSMod.truncate!(ψ; threshold=cfg.truncation_threshold, max_bond_dim=cfg.max_bond_dim)
        elseif method_sym === :hybrid_bug_second_order
            @t :hybrid_bug_second_order_step BUG.hybrid_bug_second_order!(ψ, H, cfg; numiter_lanczos=numiter_lanczos)
        elseif method_sym === :single_site_tdvp
            @t :single_site_tdvp_step Algo.single_site_tdvp!(ψ, H, cfg_tdvp; numiter_lanczos=numiter_lanczos)
            # @t :single_site_tdvp_truncate MPSMod.truncate!(ψ; threshold=cfg.truncation_threshold, max_bond_dim=cfg.max_bond_dim)
        elseif method_sym === :two_site_tdvp
            @t :two_site_tdvp_step Algo.two_site_tdvp!(ψ, H, cfg_tdvp; numiter_lanczos=numiter_lanczos)
            if truncation_mode === :after_sweep
                # Single post-pass compression using the user threshold semantics.
                @t :two_site_tdvp_truncate MPSMod.truncate!(ψ; threshold=cfg.truncation_threshold, max_bond_dim=cfg.max_bond_dim)
            end
        else
            error("Unknown method_sym: $method_sym")
        end
        out[k + 1] = real(MPSMod.local_expect(ψ, Z, site))
        if track_bond_dims
            bond_dims[k + 1] = MPSMod.write_max_bond_dim(ψ)
        end
    end
    wall = time() - t0

    if measure_runtime
        Timing.end_scope!(ts; header="Timing scope: $(method_sym)")
        _print_runtime_table(ts; header="Timing summary: $(method_sym)", top=50)
        Timing.enable_timing!(false)
    end

    return track_bond_dims ? (times, out, bond_dims, wall) : (times, out, wall)
end

@inline function rms_error(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; skip::Int=0)
    @assert length(x) == length(y)
    i0 = 1 + skip
    return sqrt(mean((x[i0:end] .- y[i0:end]) .^ 2))
end

@inline function min_abs_error(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; skip::Int=0)
    @assert length(x) == length(y)
    i0 = 1 + skip
    return minimum(abs.(x[i0:end] .- y[i0:end]))
end

@inline function max_abs_error(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; skip::Int=0)
    @assert length(x) == length(y)
    i0 = 1 + skip
    return maximum(abs.(x[i0:end] .- y[i0:end]))
end

end # module BUGExpUtil