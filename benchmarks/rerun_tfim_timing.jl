#!/usr/bin/env julia
"""
    rerun_tfim_timing.jl

Re-runs ONLY the TFIM benchmark configurations whose wall-clock timing was
contaminated by concurrent processes.  Results are written to a separate
`run_manifest_rerun.csv` in each output directory so the original data is
preserved.

Contaminated runs (10 total):
  TFIM T=1.0 fixed:    BUG2_fixed  dt=0.00125        (1 run)
  TFIM T=5.0 fixed:    1TDVP       dt=0.0025,0.00125 (2 runs)
                        BUG2_fixed  dt=0.0025,0.00125 (2 runs)
  TFIM T=5.0 adaptive: 2TDVP       dt=0.01,0.005,0.0025  (3 runs)
                        BUG2_adapt  dt=0.01,0.005         (2 runs)

Usage  (make sure nothing heavy is running!):
    julia --project=. benchmarks/rerun_tfim_timing.jl
"""

ENV["JULIA_CONDAPKG_BACKEND"] = "System"
if !haskey(ENV, "JULIA_PYTHONCALL_EXE")
    try
        py = strip(read(`which python3`, String))
        if !isempty(py) && isfile(py)
            ENV["JULIA_PYTHONCALL_EXE"] = py
        end
    catch; end
end

using Printf, Dates, LinearAlgebra, TOML, Serialization
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

# ── Config (from exp01_order.toml) ──────────────────────────────────
const N            = 12
const NUMITER      = 25
const APAD         = 4
const TMODE        = :during
const N_PTS        = 101
const CHI_FIXED    = 64
const SVD_THR      = 1e-12
const MAX_BD_CAP   = 512

# TFIM model parameters
const TFIM_NAME    = "tfim"
const TFIM_INIT    = "x+"
const TFIM_HP      = (J = 1.0, g = 1.05)

# ── Runs to re-do ───────────────────────────────────────────────────
struct RerunSpec
    T::Float64
    dt::Float64
    method::String          # raw method name for BenchmarkRunner
    pairing::String         # "fixed" or "adaptive"
    chi_fixed::Int
    svd_threshold::Float64
    max_bond_dim::Int
end

const RERUNS = RerunSpec[
    # --- T=1.0, fixed: 1 run ---
    RerunSpec(1.0, 0.00125, "fixed_bug_second_order", "fixed", CHI_FIXED, 0.0, CHI_FIXED),
    # --- T=5.0, fixed: 4 runs ---
    RerunSpec(5.0, 0.0025,  "single_site_tdvp",       "fixed", CHI_FIXED, 0.0, CHI_FIXED),
    RerunSpec(5.0, 0.0025,  "fixed_bug_second_order",  "fixed", CHI_FIXED, 0.0, CHI_FIXED),
    RerunSpec(5.0, 0.00125, "single_site_tdvp",       "fixed", CHI_FIXED, 0.0, CHI_FIXED),
    RerunSpec(5.0, 0.00125, "fixed_bug_second_order",  "fixed", CHI_FIXED, 0.0, CHI_FIXED),
    # --- T=5.0, adaptive: 5 runs ---
    RerunSpec(5.0, 0.01,   "two_site_tdvp",    "adaptive", 0, SVD_THR, MAX_BD_CAP),
    RerunSpec(5.0, 0.01,   "bug_second_order",  "adaptive", 0, SVD_THR, MAX_BD_CAP),
    RerunSpec(5.0, 0.005,  "two_site_tdvp",    "adaptive", 0, SVD_THR, MAX_BD_CAP),
    RerunSpec(5.0, 0.005,  "bug_second_order",  "adaptive", 0, SVD_THR, MAX_BD_CAP),
    RerunSpec(5.0, 0.0025, "two_site_tdvp",    "adaptive", 0, SVD_THR, MAX_BD_CAP),
]

# ── Helpers (duplicated from run_all_benchmarks.jl) ─────────────────

function git_commit_short()
    try; return String(strip(read(`git rev-parse --short HEAD`, String)))
    catch; return "unknown"; end
end

function make_obs_grid(T::Float64, dt::Float64, n_points::Int)
    raw = collect(range(0.0, T; length=n_points))
    snapped = sort(unique([round(t / dt) * dt for t in raw]))
    filter!(t -> 0.0 <= t <= T + dt/2, snapped)
    if abs(snapped[end] - T) > dt/2
        push!(snapped, T)
    end
    return snapped
end

function compute_errors_at_T(result, ref, H, N_sites)
    psi_vec = mps_to_statevector(result.psi_final)
    psi_vec ./= norm(psi_vec)
    err_infid = infidelity(psi_vec, ref.psi_T)
    err_maxZ  = maxZ_error(result.z_expect[:, end], ref.z_expect[:, end])
    err_E     = energy_error(result.energy[end], ref.energy[end])
    norm_dr   = abs(result.norm_vals[end] - 1.0)
    chi_max   = maximum(result.chi_max)
    return err_infid, err_maxZ, err_E, norm_dr, chi_max
end

# ── Main ────────────────────────────────────────────────────────────

function main()
    git_sha = git_commit_short()
    total = length(RERUNS)

    @printf("═══════════════════════════════════════════════════\n")
    @printf("  TFIM Timing Re-run  (%d runs)\n", total)
    @printf("  %s  git:%s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), git_sha)
    @printf("  IMPORTANT: make sure no heavy processes are running!\n")
    @printf("═══════════════════════════════════════════════════\n\n")
    flush(stdout)

    H = build_hamiltonian(TFIM_NAME, N; J=TFIM_HP.J, g=TFIM_HP.g)

    ref_cache = Dict{Float64, BenchmarkReference.ReferenceData}()

    t_total = time()

    for (i, spec) in enumerate(RERUNS)
        T = spec.T

        # Cache reference per T
        if !haskey(ref_cache, T)
            ref_cache[T] = compute_or_load_reference(
                TFIM_NAME, N, T, [0.0, T], TFIM_HP;
                cache_dir=joinpath(@__DIR__, "results", "reference_cache"))
        end
        ref = ref_cache[T]

        obs_grid = make_obs_grid(T, spec.dt, N_PTS)

        msym   = BenchmarkRunner.method_symbol(spec.method)
        mlabel = BenchmarkRunner.method_label(msym)
        trunc_mode_str = spec.pairing == "fixed" ? "fixed" : "adaptive"

        @printf("[%2d/%2d] %s | T=%.1f | dt=%.5f | %s | χ=%d thr=%.1e\n",
                i, total, mlabel, T, spec.dt, spec.pairing, spec.chi_fixed, spec.svd_threshold)
        flush(stdout)

        result = run_single_benchmark(;
            method       = spec.method,
            H            = H,
            N            = N,
            dt           = spec.dt,
            T            = T,
            t_obs_grid   = obs_grid,
            initial_state_str = TFIM_INIT,
            model_name   = TFIM_NAME,
            max_bond_dim = spec.max_bond_dim,
            svd_threshold = spec.svd_threshold,
            adaptive_pad  = APAD,
            numiter_lanczos = NUMITER,
            truncation_mode = TMODE)

        err_infid, err_maxZ, err_E, norm_drift, chi_max_all =
            compute_errors_at_T(result, ref, H, N)

        @printf("  -> wall=%.2fs  infid=%.3e  maxZ=%.3e  E_err=%.3e  χ_max=%d\n",
                result.wall_seconds, err_infid, err_maxZ, err_E, chi_max_all)
        flush(stdout)

        ts_now = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")

        row = ManifestRow(
            "exp01_order", "tfim", N, T, spec.dt, mlabel, spec.pairing,
            trunc_mode_str, spec.chi_fixed, spec.svd_threshold,
            result.wall_seconds,
            err_infid, err_maxZ, err_E, norm_drift,
            chi_max_all, -1, 0, ts_now, git_sha)

        # Write to *_rerun.csv (separate from original manifest)
        exp_dir = joinpath(@__DIR__, "results", "exp01_order", "tfim", "T_$(T)")
        mkpath(exp_dir)
        rerun_manifest = joinpath(exp_dir, "run_manifest_rerun.csv")
        write_manifest_row!(rerun_manifest, row; append=true)

        # Save timeseries & metadata in a rerun subfolder
        run_dir = joinpath(exp_dir,
            "rerun_$(lpad(i, 4, '0'))_$(mlabel)_dt$(spec.dt)_chi$(spec.chi_fixed)_thr$(spec.svd_threshold)")
        mkpath(run_dir)
        ts_data = TimeseriesData(result.t_grid, result.z_expect, result.energy,
                                 result.norm_vals, result.chi_max)
        save_timeseries(joinpath(run_dir, "timeseries.jls"), ts_data)

        meta = Dict{String,Any}(
            "exp_id" => "exp01_order", "model" => "tfim", "N" => N,
            "T" => T, "dt" => spec.dt, "method" => mlabel,
            "pairing" => spec.pairing, "trunc_mode" => trunc_mode_str,
            "chi_fixed" => spec.chi_fixed, "svd_threshold" => spec.svd_threshold,
            "runtime_seconds" => result.wall_seconds,
            "err_infidelity_T" => err_infid, "err_maxZ_T" => err_maxZ,
            "err_energy_T" => err_E, "norm_drift_T" => norm_drift,
            "chi_max_over_time" => chi_max_all,
            "git_commit" => git_sha, "timestamp" => ts_now,
            "J" => TFIM_HP.J, "g" => TFIM_HP.g,
            "rerun" => true, "rerun_reason" => "timing contamination from concurrent processes",
        )
        merge!(meta, machine_info())
        save_metadata(joinpath(run_dir, "metadata.json"), meta)
    end

    elapsed = time() - t_total
    @printf("\n═══════════════════════════════════════════════════\n")
    @printf("  Re-run complete: %d runs in %.1f s (%.1f min)\n", total, elapsed, elapsed/60)
    @printf("  Results in: benchmarks/results/exp01_order/tfim/T_*/run_manifest_rerun.csv\n")
    @printf("═══════════════════════════════════════════════════\n")
    flush(stdout)
end

main()
