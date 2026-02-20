#!/usr/bin/env julia
"""
    run_all_benchmarks.jl

Main entrypoint for the reproducible benchmark suite.

Usage:
    julia --project=. benchmarks/run_all_benchmarks.jl
    julia --project=. benchmarks/run_all_benchmarks.jl --exp=exp01_order
    julia --project=. benchmarks/run_all_benchmarks.jl --exp=exp01_order --model=tfim --T=5.0

CLI flags (all optional):
    --exp=EXP_NAME       Run only this experiment (exp01_order, exp02_trunc, exp03_pareto)
    --model=MODEL        Run only this model (tfim, xxz)
    --T=VALUE            Run only this final time
    --pairing=PAIRING    Run only this pairing (fixed, adaptive)
    --N=VALUE            Override system size N (default: from each config file).
                         Results go into exp0X_YYYYYY_N<N>/ subdirectories so that
                         different N values never share a results folder.
                         Feasibility note (exact reference uses full diagonalization):
                           N=12 → dim=4096,   trivial
                           N=14 → dim=16384,  ~10 GB RAM, ~5–30 min first run
                           N≥16 → requires >64 GB RAM; not recommended on a Mac
"""

# Point PythonCall to system Python to avoid CondaPkg/pixi hangs.
# The benchmark suite itself uses pure-Julia exact references and does NOT need Python.
ENV["JULIA_CONDAPKG_BACKEND"] = "System"
if !haskey(ENV, "JULIA_PYTHONCALL_EXE")
    # Try to find system python3
    try
        py = strip(read(`which python3`, String))
        if !isempty(py) && isfile(py)
            ENV["JULIA_PYTHONCALL_EXE"] = py
        end
    catch; end
end

using Printf
using Dates
using LinearAlgebra
using TOML
using Serialization

using Yaqs
const MPSMod = Yaqs.MPSModule
const MPOMod = Yaqs.MPOModule

include("utils_models.jl")
include("utils_reference.jl")
include("utils_metrics.jl")
include("utils_io.jl")
include("utils_runner.jl")

using .BenchmarkModels
using .BenchmarkReference
using .BenchmarkMetrics
using .BenchmarkIO: ManifestRow, write_manifest_header!, write_manifest_row!,
                    save_timeseries, save_metadata, TimeseriesData, machine_info
using .BenchmarkRunner

# ─────────────────────────────────────────────────────────────────────
# CLI parsing
# ─────────────────────────────────────────────────────────────────────

function parse_cli(args::Vector{String})
    out = Dict{String,String}()
    for a in args
        s = startswith(a, "--") ? a[3:end] : a
        if occursin("=", s)
            k, v = split(s, "=", limit=2)
            out[k] = v
        else
            out[s] = "true"
        end
    end
    return out
end

function git_commit_short()
    try
        return String(strip(read(`git rev-parse --short HEAD`, String)))
    catch
        return "unknown"
    end
end

# ─────────────────────────────────────────────────────────────────────
# Model configs
# ─────────────────────────────────────────────────────────────────────

struct ModelConfig
    name::String            # "tfim" or "general"
    label::String           # "tfim" or "xxz"
    initial_state::String
    H_params::NamedTuple
end

function load_model_configs(cfg::Dict)
    models = ModelConfig[]
    for (key, mcfg) in cfg["models"]
        name = mcfg["name"]
        init_state = mcfg["initial_state"]
        if name in ("tfim", "ising")
            hp = (J=Float64(mcfg["J"]), g=Float64(mcfg["g"]))
        else
            hp = (Jxx=Float64(mcfg["Jxx"]), Jyy=Float64(mcfg["Jyy"]),
                  Jzz=Float64(mcfg["Jzz"]),
                  hx=Float64(mcfg["hx"]), hy=Float64(mcfg["hy"]),
                  hz=Float64(mcfg["hz"]))
        end
        push!(models, ModelConfig(name, model_label(name), init_state, hp))
    end
    return models
end

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

function build_H(mc::ModelConfig, N::Int)
    if mc.name in ("tfim", "ising")
        return build_hamiltonian(mc.name, N; J=mc.H_params.J, g=mc.H_params.g)
    else
        return build_hamiltonian(mc.name, N;
                                Jxx=mc.H_params.Jxx, Jyy=mc.H_params.Jyy,
                                Jzz=mc.H_params.Jzz,
                                hx=mc.H_params.hx, hy=mc.H_params.hy,
                                hz=mc.H_params.hz)
    end
end

function make_obs_grid(T::Float64, dt::Float64, n_points::Int)
    # Build obs grid that is a subset of the dt grid
    raw_grid = collect(range(0.0, T; length=n_points))
    # Snap each point to nearest dt multiple
    snapped = [round(t / dt) * dt for t in raw_grid]
    # Remove duplicates and sort
    snapped = sort(unique(snapped))
    # Clamp to [0, T]
    filter!(t -> 0.0 <= t <= T + dt/2, snapped)
    # Ensure T is the last point
    if abs(snapped[end] - T) > dt/2
        push!(snapped, T)
    end
    return snapped
end

function compute_errors_at_T(result::BenchmarkRunner.RunResult,
                             ref::BenchmarkReference.ReferenceData,
                             H::MPOMod.MPO{ComplexF64},
                             N::Int)
    # Infidelity
    psi_mps_vec = mps_to_statevector(result.psi_final)
    # Normalize (MPS may have slight norm drift)
    psi_mps_vec ./= norm(psi_mps_vec)
    err_infid = infidelity(psi_mps_vec, ref.psi_T)

    # Max Z error at final time
    z_mps_T = result.z_expect[:, end]
    z_exact_T = ref.z_expect[:, end]
    err_maxZ = maxZ_error(z_mps_T, z_exact_T)

    # Energy error at final time
    e_mps_T = result.energy[end]
    e_exact_T = ref.energy[end]
    err_E = energy_error(e_mps_T, e_exact_T)

    # Norm drift
    norm_drift = abs(result.norm_vals[end] - 1.0)

    # Max chi over time
    chi_max_all = maximum(result.chi_max)

    return err_infid, err_maxZ, err_E, norm_drift, chi_max_all
end

function run_and_record!(;
        exp_id::String,
        mc::ModelConfig,
        N::Int,
        T::Float64,
        dt::Float64,
        method::String,
        pairing::String,
        trunc_mode::String,
        chi_fixed::Int,
        svd_threshold::Float64,
        max_bond_dim::Int,
        obs_grid::Vector{Float64},
        H::MPOMod.MPO{ComplexF64},
        ref::BenchmarkReference.ReferenceData,
        outdir::String,
        manifest_path::String,
        adaptive_pad::Int,
        numiter_lanczos::Int,
        truncation_mode::Symbol,
        git_sha::String,
        run_counter::Ref{Int})

    run_counter[] += 1
    run_id = run_counter[]

    msym = BenchmarkRunner.method_symbol(method)
    mlabel = BenchmarkRunner.method_label(msym)

    @printf("[run %04d] %s | %s | T=%.1f | dt=%.5f | χ=%d | thr=%.1e\n",
            run_id, mlabel, mc.label, T, dt, chi_fixed, svd_threshold)
    flush(stdout)

    result = run_single_benchmark(;
        method=method,
        H=H,
        N=N,
        dt=dt,
        T=T,
        t_obs_grid=obs_grid,
        initial_state_str=mc.initial_state,
        model_name=mc.name,
        max_bond_dim=max_bond_dim,
        svd_threshold=svd_threshold,
        adaptive_pad=adaptive_pad,
        numiter_lanczos=numiter_lanczos,
        truncation_mode=truncation_mode)

    err_infid, err_maxZ, err_E, norm_drift, chi_max_all =
        compute_errors_at_T(result, ref, H, N)

    @printf("  -> wall=%.2fs  infid=%.3e  maxZ=%.3e  E_err=%.3e  χ_max=%d\n",
            result.wall_seconds, err_infid, err_maxZ, err_E, chi_max_all)
    flush(stdout)

    ts_now = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")

    row = ManifestRow(
        exp_id, mc.label, N, T, dt, mlabel, pairing, trunc_mode,
        chi_fixed, svd_threshold,
        result.wall_seconds,
        err_infid, err_maxZ, err_E, norm_drift,
        chi_max_all,
        -1,  # is_pareto: set later for exp03
        0,   # seed
        ts_now, git_sha)

    write_manifest_row!(manifest_path, row)

    # Save timeseries
    run_dir = joinpath(outdir, "run_$(lpad(run_id, 4, '0'))_$(mlabel)_dt$(dt)_chi$(chi_fixed)_thr$(svd_threshold)")
    mkpath(run_dir)
    ts_data = TimeseriesData(result.t_grid, result.z_expect, result.energy,
                             result.norm_vals, result.chi_max)
    save_timeseries(joinpath(run_dir, "timeseries.jls"), ts_data)

    # Save metadata
    meta = Dict{String,Any}(
        "exp_id" => exp_id,
        "model" => mc.label,
        "model_name" => mc.name,
        "N" => N,
        "T" => T,
        "dt" => dt,
        "method" => mlabel,
        "method_raw" => method,
        "pairing" => pairing,
        "trunc_mode" => trunc_mode,
        "chi_fixed" => chi_fixed,
        "svd_threshold" => svd_threshold,
        "max_bond_dim" => max_bond_dim,
        "adaptive_pad" => adaptive_pad,
        "numiter_lanczos" => numiter_lanczos,
        "truncation_mode" => string(truncation_mode),
        "initial_state" => mc.initial_state,
        "runtime_seconds" => result.wall_seconds,
        "err_infidelity_T" => err_infid,
        "err_maxZ_T" => err_maxZ,
        "err_energy_T" => err_E,
        "norm_drift_T" => norm_drift,
        "chi_max_over_time" => chi_max_all,
        "git_commit" => git_sha,
        "timestamp" => ts_now,
    )
    merge!(meta, machine_info())
    if mc.name in ("tfim", "ising")
        meta["J"] = mc.H_params.J
        meta["g"] = mc.H_params.g
    else
        for k in keys(mc.H_params)
            meta[string(k)] = mc.H_params[k]
        end
    end
    save_metadata(joinpath(run_dir, "metadata.json"), meta)

    return row
end

# ─────────────────────────────────────────────────────────────────────
# Chi-list helper
# ─────────────────────────────────────────────────────────────────────

"""
    _extended_chi_list(base_chi, N) -> Vector{Int}

Return `base_chi` extended (using its own step size) up to the maximum
meaningful bond dimension `2^(N÷2)` for an N-site chain, so that fixed-
truncation sweeps cover the full entanglement range at larger N.

For N=12 (max chi=64) the base list is returned unchanged.
For N=14 (max chi=128) new values are appended in steps matching the
existing spacing (e.g. 8 for exp02, 16 for exp03).
"""
function _extended_chi_list(base_chi::Vector{Int}, N::Int)::Vector{Int}
    chi_max = 2^(N ÷ 2)
    chi_max <= base_chi[end] && return base_chi
    step  = base_chi[end] - base_chi[end-1]
    extra = collect((base_chi[end] + step):step:chi_max)
    return vcat(base_chi, extra)
end

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Order verification
# ─────────────────────────────────────────────────────────────────────

function run_exp01_order(cli::Dict, git_sha::String)
    cfg = TOML.parsefile(joinpath(@__DIR__, "configs", "exp01_order.toml"))
    glob = cfg["global"]
    N = haskey(cli, "N") ? parse(Int, cli["N"]) : Int(glob["N"])
    dt_list = Float64.(glob["dt_list"])
    T_list = Float64.(glob["T_list"])
    n_pts = glob["obs_grid_points"]
    numiter = glob["numiter_lanczos"]
    apad = glob["adaptive_pad"]
    tmode = Symbol(glob["truncation_mode"])

    models = load_model_configs(cfg)

    # Filter by CLI
    if haskey(cli, "model")
        filter!(m -> m.label == cli["model"], models)
    end
    if haskey(cli, "T")
        T_list = [parse(Float64, cli["T"])]
    end

    # Pre-count total runs for progress
    n_fixed_methods = length(cfg["pairing_fixed"]["methods"])
    n_adaptive_methods = length(cfg["pairing_adaptive"]["methods"])
    n_dts = length(dt_list)
    runs_per_T = 0
    if !haskey(cli, "pairing") || cli["pairing"] == "fixed"
        runs_per_T += n_dts * n_fixed_methods
    end
    if !haskey(cli, "pairing") || cli["pairing"] == "adaptive"
        runs_per_T += n_dts * n_adaptive_methods
    end
    total_runs = length(models) * length(T_list) * runs_per_T
    global_run = Ref(0)
    t_exp = time()

    @printf("  exp01: %d models x %d T values x %d runs/T = %d total runs\n\n",
            length(models), length(T_list), runs_per_T, total_runs)
    flush(stdout)

    for mc in models
        H = build_H(mc, N)
        for T in T_list
            exp_dir = joinpath(@__DIR__, "results", "exp01_order_N$(N)", mc.label, "T_$(T)")
            mkpath(exp_dir)
            manifest_path = joinpath(exp_dir, "run_manifest.csv")
            run_ctr = Ref(0)

            @printf("── exp01 | model=%s | T=%.1f ──\n", mc.label, T)
            flush(stdout)

            # Compute reference once per (model, T) on a minimal 2-point grid
            ref = compute_or_load_reference(mc.name, N, T, [0.0, T], mc.H_params;
                    cache_dir=joinpath(@__DIR__, "results", "reference_cache"))

            # Pairing A: fixed (1TDVP vs double fixed BUG)
            if !haskey(cli, "pairing") || cli["pairing"] == "fixed"
                chi_fixed = cfg["pairing_fixed"]["chi_fixed"]
                @printf("  pairing=fixed  chi=%d  dts=%s\n", chi_fixed, dt_list)
                flush(stdout)
                for dt in dt_list
                    obs_grid = make_obs_grid(T, dt, n_pts)
                    for meth in cfg["pairing_fixed"]["methods"]
                        global_run[] += 1
                        @printf("  [%d/%d] ", global_run[], total_runs)
                        flush(stdout)
                        run_and_record!(;
                            exp_id="exp01_order",
                            mc=mc, N=N, T=T, dt=dt,
                            method=meth,
                            pairing="fixed",
                            trunc_mode="fixed",
                            chi_fixed=chi_fixed,
                            svd_threshold=0.0,
                            max_bond_dim=chi_fixed,
                            obs_grid=obs_grid,
                            H=H, ref=ref,
                            outdir=exp_dir,
                            manifest_path=manifest_path,
                            adaptive_pad=apad,
                            numiter_lanczos=numiter,
                            truncation_mode=tmode,
                            git_sha=git_sha,
                            run_counter=run_ctr)
                    end
                end
            end

            # Pairing B: adaptive (2TDVP vs double adaptive BUG)
            if !haskey(cli, "pairing") || cli["pairing"] == "adaptive"
                svd_thr = Float64(cfg["pairing_adaptive"]["svd_threshold"])
                max_bd = cfg["pairing_adaptive"]["max_bond_dim_cap"]
                @printf("  pairing=adaptive  thr=%.1e  dts=%s\n", svd_thr, dt_list)
                flush(stdout)
                for dt in dt_list
                    obs_grid = make_obs_grid(T, dt, n_pts)
                    for meth in cfg["pairing_adaptive"]["methods"]
                        global_run[] += 1
                        @printf("  [%d/%d] ", global_run[], total_runs)
                        flush(stdout)
                        run_and_record!(;
                            exp_id="exp01_order",
                            mc=mc, N=N, T=T, dt=dt,
                            method=meth,
                            pairing="adaptive",
                            trunc_mode="adaptive",
                            chi_fixed=0,
                            svd_threshold=svd_thr,
                            max_bond_dim=max_bd,
                            obs_grid=obs_grid,
                            H=H, ref=ref,
                            outdir=exp_dir,
                            manifest_path=manifest_path,
                            adaptive_pad=apad,
                            numiter_lanczos=numiter,
                            truncation_mode=tmode,
                            git_sha=git_sha,
                            run_counter=run_ctr)
                    end
                end
            end
        end
    end
    elapsed = time() - t_exp
    @printf("\n  exp01 finished: %d runs in %.1f s (%.1f min)\n", global_run[], elapsed, elapsed/60)
    flush(stdout)
end

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Truncation sensitivity
# ─────────────────────────────────────────────────────────────────────

function run_exp02_trunc(cli::Dict, git_sha::String)
    cfg = TOML.parsefile(joinpath(@__DIR__, "configs", "exp02_trunc.toml"))
    glob = cfg["global"]
    N = haskey(cli, "N") ? parse(Int, cli["N"]) : Int(glob["N"])
    dt_small = Float64(glob["dt_small"])
    T_list = Float64.(glob["T_list"])
    n_pts = glob["obs_grid_points"]
    numiter = glob["numiter_lanczos"]
    apad = glob["adaptive_pad"]
    tmode = Symbol(glob["truncation_mode"])

    models = load_model_configs(cfg)
    if haskey(cli, "model")
        filter!(m -> m.label == cli["model"], models)
    end
    if haskey(cli, "T")
        T_list = [parse(Float64, cli["T"])]
    end

    # Pre-count total runs.
    # Compute the extended chi_list upfront so the count reflects N correctly.
    n_svd     = length(cfg["pairing_adaptive"]["svd_threshold_list"])
    n_am      = length(cfg["pairing_adaptive"]["methods"])
    n_fm      = length(cfg["pairing_fixed"]["methods"])
    chi_list_full = _extended_chi_list(Int.(cfg["pairing_fixed"]["chi_list"]), N)
    runs_per_T = 0
    if !haskey(cli, "pairing") || cli["pairing"] == "adaptive"
        runs_per_T += n_svd * n_am
    end
    if !haskey(cli, "pairing") || cli["pairing"] == "fixed"
        runs_per_T += length(chi_list_full) * n_fm
    end
    total_runs = length(models) * length(T_list) * runs_per_T
    global_run = Ref(0)
    t_exp = time()

    @printf("  exp02: %d models x %d T values x %d runs/T = %d total runs  (dt=%.5f)\n\n",
            length(models), length(T_list), runs_per_T, total_runs, dt_small)
    flush(stdout)

    for mc in models
        H = build_H(mc, N)
        for T in T_list
            exp_dir = joinpath(@__DIR__, "results", "exp02_trunc_N$(N)", mc.label, "T_$(T)")
            mkpath(exp_dir)
            manifest_path = joinpath(exp_dir, "run_manifest.csv")
            run_ctr = Ref(0)

            @printf("── exp02 | model=%s | T=%.1f ──\n", mc.label, T)
            flush(stdout)

            obs_grid = make_obs_grid(T, dt_small, n_pts)
            ref = compute_or_load_reference(mc.name, N, T, [0.0, T], mc.H_params;
                    cache_dir=joinpath(@__DIR__, "results", "reference_cache"))

            # Pairing A: adaptive — sweep SVD thresholds
            if !haskey(cli, "pairing") || cli["pairing"] == "adaptive"
                svd_list = Float64.(cfg["pairing_adaptive"]["svd_threshold_list"])
                max_bd = cfg["pairing_adaptive"]["max_bond_dim_cap"]
                @printf("  pairing=adaptive  svd_thresholds=%s\n", svd_list)
                flush(stdout)
                for svd_thr in svd_list
                    for meth in cfg["pairing_adaptive"]["methods"]
                        global_run[] += 1
                        @printf("  [%d/%d] ", global_run[], total_runs)
                        flush(stdout)
                        run_and_record!(;
                            exp_id="exp02_trunc",
                            mc=mc, N=N, T=T, dt=dt_small,
                            method=meth,
                            pairing="adaptive",
                            trunc_mode="adaptive",
                            chi_fixed=0,
                            svd_threshold=svd_thr,
                            max_bond_dim=max_bd,
                            obs_grid=obs_grid,
                            H=H, ref=ref,
                            outdir=exp_dir,
                            manifest_path=manifest_path,
                            adaptive_pad=apad,
                            numiter_lanczos=numiter,
                            truncation_mode=tmode,
                            git_sha=git_sha,
                            run_counter=run_ctr)
                    end
                end
            end

            # Pairing B: fixed — sweep chi
            if !haskey(cli, "pairing") || cli["pairing"] == "fixed"
                @printf("  pairing=fixed  chi_list=%s\n", chi_list_full)
                flush(stdout)
                for chi in chi_list_full
                    for meth in cfg["pairing_fixed"]["methods"]
                        global_run[] += 1
                        @printf("  [%d/%d] ", global_run[], total_runs)
                        flush(stdout)
                        run_and_record!(;
                            exp_id="exp02_trunc",
                            mc=mc, N=N, T=T, dt=dt_small,
                            method=meth,
                            pairing="fixed",
                            trunc_mode="fixed",
                            chi_fixed=chi,
                            svd_threshold=0.0,
                            max_bond_dim=chi,
                            obs_grid=obs_grid,
                            H=H, ref=ref,
                            outdir=exp_dir,
                            manifest_path=manifest_path,
                            adaptive_pad=apad,
                            numiter_lanczos=numiter,
                            truncation_mode=tmode,
                            git_sha=git_sha,
                            run_counter=run_ctr)
                    end
                end
            end
        end
    end
    elapsed = time() - t_exp
    @printf("\n  exp02 finished: %d runs in %.1f s (%.1f min)\n", global_run[], elapsed, elapsed/60)
    flush(stdout)
end

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: Pareto (runtime vs error)
# ─────────────────────────────────────────────────────────────────────

function compute_pareto_flags(rows::Vector{ManifestRow})
    # Within each (model, pairing) group, mark Pareto-optimal runs.
    # A run is Pareto-optimal if no other run dominates it on both
    # (runtime_seconds, err_infidelity_T).
    n = length(rows)
    is_pareto = fill(true, n)
    for i in 1:n
        for j in 1:n
            i == j && continue
            if rows[j].runtime_seconds <= rows[i].runtime_seconds &&
               rows[j].err_infidelity_T <= rows[i].err_infidelity_T &&
               (rows[j].runtime_seconds < rows[i].runtime_seconds ||
                rows[j].err_infidelity_T < rows[i].err_infidelity_T)
                is_pareto[i] = false
                break
            end
        end
    end
    return is_pareto
end

function run_exp03_pareto(cli::Dict, git_sha::String)
    cfg = TOML.parsefile(joinpath(@__DIR__, "configs", "exp03_pareto.toml"))
    glob = cfg["global"]
    N = haskey(cli, "N") ? parse(Int, cli["N"]) : Int(glob["N"])
    T_list = Float64.(glob["T_list"])
    n_pts = glob["obs_grid_points"]
    numiter = glob["numiter_lanczos"]
    apad = glob["adaptive_pad"]
    tmode = Symbol(glob["truncation_mode"])

    models = load_model_configs(cfg)
    if haskey(cli, "model")
        filter!(m -> m.label == cli["model"], models)
    end
    if haskey(cli, "T")
        T_list = [parse(Float64, cli["T"])]
    end

    # Pre-count total runs.
    # Use the extended chi_list for accurate count when N > default.
    n_dt_a = length(cfg["pairing_adaptive"]["dt_list"])
    n_svd  = length(cfg["pairing_adaptive"]["svd_threshold_list"])
    n_am   = length(cfg["pairing_adaptive"]["methods"])
    n_dt_f = length(cfg["pairing_fixed"]["dt_list"])
    n_fm   = length(cfg["pairing_fixed"]["methods"])
    chi_list_full = _extended_chi_list(Int.(cfg["pairing_fixed"]["chi_list"]), N)
    runs_per_T = 0
    if !haskey(cli, "pairing") || cli["pairing"] == "adaptive"
        runs_per_T += n_dt_a * n_svd * n_am
    end
    if !haskey(cli, "pairing") || cli["pairing"] == "fixed"
        runs_per_T += n_dt_f * length(chi_list_full) * n_fm
    end
    total_runs = length(models) * length(T_list) * runs_per_T
    global_run = Ref(0)
    t_exp = time()

    @printf("  exp03: %d models x %d T values x %d runs/T = %d total runs\n\n",
            length(models), length(T_list), runs_per_T, total_runs)
    flush(stdout)

    for mc in models
        H = build_H(mc, N)
        for T in T_list
            exp_dir = joinpath(@__DIR__, "results", "exp03_pareto_N$(N)", mc.label, "T_$(T)")
            mkpath(exp_dir)
            manifest_path = joinpath(exp_dir, "run_manifest.csv")
            run_ctr = Ref(0)

            all_rows = ManifestRow[]

            @printf("── exp03 | model=%s | T=%.1f ──\n", mc.label, T)
            flush(stdout)

            # Single reference per (model, T) — only need final-time values
            ref = compute_or_load_reference(mc.name, N, T, [0.0, T], mc.H_params;
                    cache_dir=joinpath(@__DIR__, "results", "reference_cache"))

            # Pairing A: adaptive
            if !haskey(cli, "pairing") || cli["pairing"] == "adaptive"
                dt_list_a = Float64.(cfg["pairing_adaptive"]["dt_list"])
                svd_list = Float64.(cfg["pairing_adaptive"]["svd_threshold_list"])
                max_bd = cfg["pairing_adaptive"]["max_bond_dim_cap"]
                @printf("  pairing=adaptive  dts=%s  svd_thresholds=%s\n", dt_list_a, svd_list)
                flush(stdout)
                for dt in dt_list_a
                    obs_grid = make_obs_grid(T, dt, n_pts)
                    for svd_thr in svd_list
                        for meth in cfg["pairing_adaptive"]["methods"]
                            global_run[] += 1
                            @printf("  [%d/%d] ", global_run[], total_runs)
                            flush(stdout)
                            row = run_and_record!(;
                                exp_id="exp03_pareto",
                                mc=mc, N=N, T=T, dt=dt,
                                method=meth,
                                pairing="adaptive",
                                trunc_mode="adaptive",
                                chi_fixed=0,
                                svd_threshold=svd_thr,
                                max_bond_dim=max_bd,
                                obs_grid=obs_grid,
                                H=H, ref=ref,
                                outdir=exp_dir,
                                manifest_path=manifest_path,
                                adaptive_pad=apad,
                                numiter_lanczos=numiter,
                                truncation_mode=tmode,
                                git_sha=git_sha,
                                run_counter=run_ctr)
                            push!(all_rows, row)
                        end
                    end
                end
            end

            # Pairing B: fixed
            if !haskey(cli, "pairing") || cli["pairing"] == "fixed"
                dt_list_f = Float64.(cfg["pairing_fixed"]["dt_list"])
                @printf("  pairing=fixed  dts=%s  chi_list=%s\n", dt_list_f, chi_list_full)
                flush(stdout)
                for dt in dt_list_f
                    obs_grid = make_obs_grid(T, dt, n_pts)
                    for chi in chi_list_full
                        for meth in cfg["pairing_fixed"]["methods"]
                            global_run[] += 1
                            @printf("  [%d/%d] ", global_run[], total_runs)
                            flush(stdout)
                            row = run_and_record!(;
                                exp_id="exp03_pareto",
                                mc=mc, N=N, T=T, dt=dt,
                                method=meth,
                                pairing="fixed",
                                trunc_mode="fixed",
                                chi_fixed=chi,
                                svd_threshold=0.0,
                                max_bond_dim=chi,
                                obs_grid=obs_grid,
                                H=H, ref=ref,
                                outdir=exp_dir,
                                manifest_path=manifest_path,
                                adaptive_pad=apad,
                                numiter_lanczos=numiter,
                                truncation_mode=tmode,
                                git_sha=git_sha,
                                run_counter=run_ctr)
                            push!(all_rows, row)
                        end
                    end
                end
            end

            # Compute Pareto flags per (model, pairing) group
            if !isempty(all_rows)
                pareto_manifest = joinpath(exp_dir, "run_manifest_pareto.csv")
                open(pareto_manifest, "w") do io
                    write_manifest_header!(io)

                    for pairing_key in unique(r.pairing for r in all_rows)
                        group = filter(r -> r.pairing == pairing_key, all_rows)
                        flags = compute_pareto_flags(group)
                        for (i, r) in enumerate(group)
                            updated = ManifestRow(
                                r.exp_id, r.model, r.N, r.T, r.dt, r.method,
                                r.pairing, r.trunc_mode,
                                r.chi_fixed, r.svd_threshold,
                                r.runtime_seconds,
                                r.err_infidelity_T, r.err_maxZ_T, r.err_energy_T,
                                r.norm_drift_T, r.chi_max_over_time,
                                flags[i] ? 1 : 0,
                                r.seed, r.timestamp, r.git_commit)
                            write_manifest_row!(io, updated)
                        end
                    end
                end
                @printf("[exp03] Pareto manifest written: %s\n", pareto_manifest)
                flush(stdout)
            end
        end
    end
    elapsed = time() - t_exp
    @printf("\n  exp03 finished: %d runs in %.1f s (%.1f min)\n", global_run[], elapsed, elapsed/60)
    flush(stdout)
end

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

function main(args=ARGS)
    cli = parse_cli(args)
    git_sha = git_commit_short()

    exp_filter = get(cli, "exp", "all")

    @printf("═══════════════════════════════════════════════════\n")
    @printf("  YAQS-Julia Benchmark Suite\n")
    @printf("  %s  git:%s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), git_sha)
    @printf("  Filter: exp=%s model=%s T=%s pairing=%s N=%s\n",
            exp_filter,
            get(cli, "model", "all"),
            get(cli, "T", "all"),
            get(cli, "pairing", "all"),
            get(cli, "N", "from_config"))
    @printf("═══════════════════════════════════════════════════\n\n")
    flush(stdout)

    t_total = time()

    if exp_filter in ("all", "exp01_order")
        @printf("\n▶ EXPERIMENT 1: Order verification\n")
        flush(stdout)
        run_exp01_order(cli, git_sha)
    end

    if exp_filter in ("all", "exp02_trunc")
        @printf("\n▶ EXPERIMENT 2: Truncation sensitivity\n")
        flush(stdout)
        run_exp02_trunc(cli, git_sha)
    end

    if exp_filter in ("all", "exp03_pareto")
        @printf("\n▶ EXPERIMENT 3: Pareto (runtime vs error)\n")
        flush(stdout)
        run_exp03_pareto(cli, git_sha)
    end

    elapsed = time() - t_total
    @printf("\n═══════════════════════════════════════════════════\n")
    @printf("  All done. Total wall time: %.1f s (%.1f min)\n", elapsed, elapsed/60)
    @printf("═══════════════════════════════════════════════════\n")
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
