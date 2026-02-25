"""
    BenchmarkRunner

Core simulation driver for the benchmark suite.

Adapts the pattern from `04_BUG_exps/exp_util.jl` (run_method_expect_z_site)
to collect full observable time-series for all sites.
"""
module BenchmarkRunner

using LinearAlgebra
using Printf

using Yaqs
const BUG = Yaqs.BUGModule
const Algo = Yaqs.Algorithms
const MPSMod = Yaqs.MPSModule
const MPOMod = Yaqs.MPOModule
const Cfg = Yaqs.SimulationConfigs
const GL = Yaqs.GateLibrary

export RunResult, run_single_benchmark

struct RunResult
    t_grid::Vector{Float64}
    z_expect::Matrix{Float64}     # (N, n_obs)
    energy::Vector{Float64}       # (n_obs,)
    norm_vals::Vector{Float64}    # (n_obs,)
    chi_max::Vector{Int}          # (n_obs,)
    wall_seconds::Float64
    psi_final::MPSMod.MPS{ComplexF64}
end

"""
    method_symbol(s::AbstractString) -> Symbol

Parse method string into the internal symbol.
"""
function method_symbol(s::AbstractString)
    m = lowercase(strip(s))
    if m in ("single_site_tdvp", "1tdvp")
        return :single_site_tdvp
    elseif m in ("two_site_tdvp", "2tdvp")
        return :two_site_tdvp
    elseif m in ("fixed_bug_second_order", "doublefixed", "double_fixed_bug")
        return :fixed_bug_second_order
    elseif m in ("bug_second_order", "doubleadaptive", "double_adaptive_bug")
        return :bug_second_order
    else
        error("Unknown method: $s")
    end
end

"""
    method_label(sym::Symbol) -> String

Human-readable label for manifest.
"""
function method_label(sym::Symbol)
    if sym === :single_site_tdvp
        return "1TDVP"
    elseif sym === :two_site_tdvp
        return "2TDVP"
    elseif sym === :fixed_bug_second_order
        return "BUG2_fixed"
    elseif sym === :bug_second_order
        return "BUG2_adaptive"
    else
        return string(sym)
    end
end

"""
    is_fixed_method(sym::Symbol) -> Bool
"""
function is_fixed_method(sym::Symbol)
    return sym in (:single_site_tdvp, :fixed_bug_second_order)
end

"""
    run_single_benchmark(;
        method, H, N, dt, T, t_obs_grid,
        initial_state_str, model_name,
        max_bond_dim, svd_threshold,
        adaptive_pad, numiter_lanczos,
        truncation_mode
    ) -> RunResult

Run a single benchmark simulation and collect observables at each point in `t_obs_grid`.

The simulation advances in steps of `dt`. At each observation time in `t_obs_grid`,
we record:
- ⟨Z_i⟩ for all sites
- ⟨H⟩ (energy)
- ‖ψ‖ (norm)
- max bond dimension
"""
function run_single_benchmark(;
        method::AbstractString,
        H::MPOMod.MPO{ComplexF64},
        N::Int,
        dt::Float64,
        T::Float64,
        t_obs_grid::Vector{Float64},
        initial_state_str::AbstractString,
        model_name::AbstractString,
        max_bond_dim::Int,
        svd_threshold::Float64,
        adaptive_pad::Int=4,
        numiter_lanczos::Int=25,
        truncation_mode::Symbol=:during)

    msym = method_symbol(method)
    n_obs = length(t_obs_grid)

    # Total steps
    total_steps = round(Int, T / dt)
    @assert abs(total_steps * dt - T) < 1e-12 * T "T=$T not evenly divisible by dt=$dt"

    # Build observation schedule: at which step index do we observe?
    obs_at_step = Int[]
    for t_obs in t_obs_grid
        step_idx = round(Int, t_obs / dt)
        push!(obs_at_step, step_idx)
    end

    # Z operator for local expectations
    Z = ComplexF64.(Matrix(GL.matrix(GL.ZGate())))
    Z_ops = [Z for _ in 1:N]

    # Initialize MPS
    ψ = MPSMod.MPS(N; state=initial_state_str)

    # Config for BUG / TDVP
    cfg = Cfg.TimeEvolutionConfig(Cfg.Observable[], dt;
                                  dt=dt,
                                  max_bond_dim=max_bond_dim,
                                  truncation_threshold=svd_threshold,
                                  sample_timesteps=false)

    # For TDVP: in-sweep truncation handling
    tdvp_thr = if truncation_mode === :after_sweep
        (cfg.truncation_threshold >= 0) ? Inf : -Inf
    else
        cfg.truncation_threshold
    end
    cfg_tdvp = Cfg.TimeEvolutionConfig(Cfg.Observable[], dt;
                                       dt=dt,
                                       max_bond_dim=max_bond_dim,
                                       truncation_threshold=tdvp_thr,
                                       sample_timesteps=false)

    # Padding (mirroring exp_util.jl logic)
    if msym in (:fixed_bug_second_order,)
        MPSMod.pad_bond_dimension!(ψ, max_bond_dim; noise_scale=0.0)
    elseif msym === :single_site_tdvp
        MPSMod.pad_bond_dimension!(ψ, max_bond_dim; noise_scale=1e-10)
    end
    if msym in (:bug_second_order, :two_site_tdvp)
        padχ = min(adaptive_pad, max_bond_dim)
        if padχ > 1
            MPSMod.pad_bond_dimension!(ψ, padχ; noise_scale=1e-10)
        end
    end

    # Output arrays
    z_expect = Matrix{Float64}(undef, N, n_obs)
    energy_out = Vector{Float64}(undef, n_obs)
    norm_out = Vector{Float64}(undef, n_obs)
    chi_max_out = Vector{Int}(undef, n_obs)

    obs_idx = 1

    # Record initial observables if t=0 is in the grid
    function _record_obs!(idx::Int)
        z_vals = real.(MPSMod.evaluate_all_local_expectations(ψ, Z_ops))
        z_expect[:, idx] .= z_vals
        energy_out[idx] = real(MPOMod.expect_mpo(H, ψ))
        norm_out[idx] = real(MPSMod.scalar_product(ψ, ψ))
        chi_max_out[idx] = MPSMod.write_max_bond_dim(ψ)
    end

    # Check if step 0 is an observation point
    if obs_idx <= n_obs && obs_at_step[obs_idx] == 0
        _record_obs!(obs_idx)
        obs_idx += 1
    end

    # Time the simulation loop
    t_start = time_ns()

    for step in 1:total_steps
        # Advance one time step
        if msym === :bug_second_order
            bug_trunc_timing = (truncation_mode === :after_sweep) ? :after_window : :during
            BUG.bug_second_order!(ψ, H, cfg;
                                  numiter_lanczos=numiter_lanczos,
                                  truncation_timing=bug_trunc_timing)
        elseif msym === :fixed_bug_second_order
            BUG.fixed_bug_second_order!(ψ, H, cfg; numiter_lanczos=numiter_lanczos)
            MPSMod.truncate!(ψ; threshold=cfg.truncation_threshold, max_bond_dim=cfg.max_bond_dim)
        elseif msym === :single_site_tdvp
            Algo.single_site_tdvp!(ψ, H, cfg_tdvp; numiter_lanczos=numiter_lanczos)
        elseif msym === :two_site_tdvp
            Algo.two_site_tdvp!(ψ, H, cfg_tdvp; numiter_lanczos=numiter_lanczos)
            if truncation_mode === :after_sweep
                MPSMod.truncate!(ψ; threshold=cfg.truncation_threshold, max_bond_dim=cfg.max_bond_dim)
            end
        else
            error("Unknown method_sym: $msym")
        end

        # Record if this step is an observation point
        if obs_idx <= n_obs && obs_at_step[obs_idx] == step
            _record_obs!(obs_idx)
            obs_idx += 1
        end
    end

    wall_ns = time_ns() - t_start
    wall_seconds = wall_ns / 1e9

    @assert obs_idx == n_obs + 1 "Not all observation points were recorded (got $(obs_idx-1) of $n_obs)"

    return RunResult(collect(t_obs_grid), z_expect, energy_out, norm_out, chi_max_out, wall_seconds, ψ)
end

end # module BenchmarkRunner