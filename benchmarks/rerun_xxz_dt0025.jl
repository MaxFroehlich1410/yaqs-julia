#!/usr/bin/env julia
"""
    rerun_xxz_dt0025.jl

Re-run ONLY the XXZ T=5.0 dt=0.0025 2TDVP (adaptive) benchmark.
The original had 0.40 s/step vs expected ~0.22 (likely concurrent load).

Usage (make sure nothing heavy is running!):
    julia --project=. benchmarks/rerun_xxz_dt0025.jl
"""

ENV["JULIA_CONDAPKG_BACKEND"] = "System"
if !haskey(ENV, "JULIA_PYTHONCALL_EXE")
    try
        py = strip(read(`which python3`, String))
        if !isempty(py) && isfile(py); ENV["JULIA_PYTHONCALL_EXE"] = py; end
    catch; end
end

using Printf, Dates, LinearAlgebra, Serialization
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

const N            = 12
const T_FINAL      = 5.0
const DT           = 0.0025
const SVD_THR      = 1e-12
const MAX_BD       = 512
const NUMITER      = 25
const APAD         = 4
const TMODE        = :during
const N_PTS        = 101

function git_commit_short()
    try; return String(strip(read(`git rev-parse --short HEAD`, String)))
    catch; return "unknown"; end
end

function make_obs_grid(T, dt, n_points)
    raw = collect(range(0.0, T; length=n_points))
    snapped = sort(unique([round(t / dt) * dt for t in raw]))
    filter!(t -> 0.0 <= t <= T + dt/2, snapped)
    if abs(snapped[end] - T) > dt/2; push!(snapped, T); end
    return snapped
end

function main()
    git_sha = git_commit_short()

    @printf("═══════════════════════════════════════════════════\n")
    @printf("  XXZ T=%.1f dt=%.4f  2TDVP (adaptive) Re-run\n", T_FINAL, DT)
    @printf("  %s  git:%s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), git_sha)
    @printf("  IMPORTANT: make sure no heavy processes are running!\n")
    @printf("═══════════════════════════════════════════════════\n\n")
    flush(stdout)

    # Build XXZ Hamiltonian
    H = build_hamiltonian("general", N; Jxx=1.0, Jyy=1.0, Jzz=1.0, hx=0.0, hy=0.0, hz=0.3)

    # Load / compute reference
    ref = compute_or_load_reference("general", N, T_FINAL, [0.0, T_FINAL],
            (Jxx=1.0, Jyy=1.0, Jzz=1.0, hx=0.0, hy=0.0, hz=0.3);
            cache_dir=joinpath(@__DIR__, "results", "reference_cache"))

    obs_grid = make_obs_grid(T_FINAL, DT, N_PTS)

    exp_dir = joinpath(@__DIR__, "results", "exp01_order", "xxz", "T_$(T_FINAL)")
    rerun_manifest = joinpath(exp_dir, "run_manifest_rerun.csv")

    @printf("[1/1] 2TDVP (adaptive) | T=%.1f | dt=%.4f | χ_max=%d thr=%.1e\n",
            T_FINAL, DT, MAX_BD, SVD_THR)
    flush(stdout)

    t0 = time()
    result = run_single_benchmark(;
        method       = "two_site_tdvp",
        H            = H,
        N            = N,
        dt           = DT,
        T            = T_FINAL,
        t_obs_grid   = obs_grid,
        initial_state_str = "Neel",
        model_name   = "general",
        max_bond_dim = MAX_BD,
        svd_threshold = SVD_THR,
        adaptive_pad  = APAD,
        numiter_lanczos = NUMITER,
        truncation_mode = TMODE)

    psi_vec = mps_to_statevector(result.psi_final)
    psi_vec ./= norm(psi_vec)
    err_infid = infidelity(psi_vec, ref.psi_T)
    err_maxZ  = maxZ_error(result.z_expect[:, end], ref.z_expect[:, end])
    err_E     = energy_error(result.energy[end], ref.energy[end])
    norm_dr   = abs(result.norm_vals[end] - 1.0)
    chi_max   = maximum(result.chi_max)

    @printf("  -> wall=%.2fs  infid=%.3e  E_err=%.3e  χ_max=%d  s/step=%.4f\n",
            result.wall_seconds, err_infid, err_E, chi_max,
            result.wall_seconds / (T_FINAL / DT))
    flush(stdout)

    ts_now = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
    row = ManifestRow(
        "exp01_order", "xxz", N, T_FINAL, DT, "2TDVP", "adaptive",
        "adaptive", 0, SVD_THR, result.wall_seconds,
        err_infid, err_maxZ, err_E, norm_dr, chi_max,
        -1, 0, ts_now, git_sha)
    write_manifest_row!(rerun_manifest, row; append=true)

    run_dir = joinpath(exp_dir, "rerun_xxz_2TDVP_dt$(DT)")
    mkpath(run_dir)
    ts_data = TimeseriesData(result.t_grid, result.z_expect, result.energy,
                             result.norm_vals, result.chi_max)
    save_timeseries(joinpath(run_dir, "timeseries.jls"), ts_data)
    meta = Dict{String,Any}(
        "exp_id"=>"exp01_order", "model"=>"xxz", "N"=>N, "T"=>T_FINAL, "dt"=>DT,
        "method"=>"2TDVP", "pairing"=>"adaptive", "runtime_seconds"=>result.wall_seconds,
        "err_infidelity_T"=>err_infid, "err_energy_T"=>err_E,
        "chi_max_over_time"=>chi_max, "git_commit"=>git_sha, "rerun"=>true,
        "rerun_reason"=>"elevated timing 0.40 s/step vs expected ~0.22")
    merge!(meta, machine_info())
    save_metadata(joinpath(run_dir, "metadata.json"), meta)

    @printf("\nDone in %.1fs. Result written to:\n  %s\n", time() - t0, rerun_manifest)
end

main()
