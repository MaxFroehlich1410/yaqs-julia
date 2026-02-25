#!/usr/bin/env julia
"""
    run_hs_benchmark.jl

End-to-end benchmark for the Haldane–Shastry (HS) spin-1/2 chain.
Runs three simulations and compares them:

  1. Exact reference via full eigendecomposition of the dense Hamiltonian.
  2. MPS time evolution with two-site TDVP (2TDVP).
  3. MPS time evolution with second-order adaptive BUG.

Produces:
  • Error-summary table in stdout.
  • Figure 1: heatmaps of C_zz(t, x) for exact / 2TDVP / BUG.
  • Figure 2: pointwise-error heatmaps (2TDVP and BUG) + line cuts at
              selected times comparing all three methods.

Usage
─────
  julia --project=. benchmarks/run_hs_benchmark.jl
  julia --project=. benchmarks/run_hs_benchmark.jl --N=8 --T=3.0 --chi=32

CLI flags (all optional):
  --N=<int>      System size            (default: 10)
  --J=<float>    Coupling               (default: 1.0)
  --T=<float>    Total time             (default: 5.0)
  --dt=<float>   Time step              (default: 0.05)
  --chi=<int>    Max MPS bond dimension (default: 64)
  --thr=<float>  SVD threshold          (default: 1e-9)
  --j_ref=<int>  Reference site for C_zz (1-indexed, default: N÷2)
  --init=<str>   Initial state: neel | wall | spin_flip  (default: neel)
                   neel      = alternating |↑↓↑↓…⟩  (Sz=0, grows high entanglement)
                   wall      = domain wall |↑…↑↓…↓⟩  (Sz=0, linear entanglement growth)
                   spin_flip = single down-spin on |↑…↑⟩ background  (1-magnon, χ stays ~2)
  --no_validate  Skip MPO dense validation step
  --no_plot      Skip matplotlib figures

Operator convention (consistent across MPO, MPS, and exact reference):
  S^z = (1/2) σ^z     eigenvalues ± 1/2
  C_zz(j+x, j; t) = ⟨S^z_{j+x} S^z_j⟩ − ⟨S^z_{j+x}⟩⟨S^z_j⟩  (connected)
"""

# ── Environment ────────────────────────────────────────────────────────────────
ENV["JULIA_CONDAPKG_BACKEND"] = "System"
if !haskey(ENV, "JULIA_PYTHONCALL_EXE")
    try
        py = strip(read(`which python3`, String))
        if !isempty(py) && isfile(py)
            ENV["JULIA_PYTHONCALL_EXE"] = py
        end
    catch; end
end

using Printf
using LinearAlgebra
using Serialization

using Yaqs
const MPSMod  = Yaqs.MPSModule
const MPOMod  = Yaqs.MPOModule
const Algo    = Yaqs.Algorithms
const BUGMod  = Yaqs.BUGModule
const Cfg     = Yaqs.SimulationConfigs
const GL      = Yaqs.GateLibrary

# ── CLI parsing ────────────────────────────────────────────────────────────────

function _parse_cli(args)
    kv = Dict{String,String}()
    for a in args
        s = startswith(a, "--") ? a[3:end] : a
        if occursin("=", s)
            k, v = split(s, "="; limit=2)
            kv[k] = v
        else
            kv[s] = "true"
        end
    end
    return kv
end

cli = _parse_cli(ARGS)

const N          = parse(Int,     get(cli, "N",   "10"))
const J          = parse(Float64, get(cli, "J",   "1.0"))
const T_total    = parse(Float64, get(cli, "T",   "5.0"))
const dt         = parse(Float64, get(cli, "dt",  "0.05"))
const chi_max    = parse(Int,     get(cli, "chi", "64"))
const svd_thr    = parse(Float64, get(cli, "thr", "1e-9"))
const j_ref_raw  = get(cli, "j_ref", "")
const init_state = get(cli, "init", "neel")
const do_validate= !haskey(cli, "no_validate")
const do_plot    = !haskey(cli, "no_plot")

const j_ref = isempty(j_ref_raw) ? max(1, N ÷ 2) : parse(Int, j_ref_raw)
@assert 1 <= j_ref <= N "j_ref=$j_ref out of range [1,$N]"
@assert init_state in ("neel", "wall", "spin_flip") "Unknown --init=$init_state"

# ── Derived quantities ─────────────────────────────────────────────────────────
const n_steps = round(Int, T_total / dt)
@assert abs(n_steps * dt - T_total) < 1e-12 * T_total "T=$T_total not divisible by dt=$dt"
const t_grid = collect(0.0:dt:T_total)   # length n_steps+1

# ── Output directories ─────────────────────────────────────────────────────────
const results_dir = joinpath(@__DIR__, "results")
const figures_dir = joinpath(@__DIR__, "figures")
mkpath(results_dir)
mkpath(figures_dir)

@printf "\n%s\n" ("─"^65)
@printf "  Haldane–Shastry benchmark  (2TDVP vs BUG)\n"
@printf "  N=%d  J=%.4g  T=%.4g  dt=%.4g  χ_max=%d  thr=%.2g\n" N J T_total dt chi_max svd_thr
@printf "  j_ref=%d (1-indexed)  init=%s\n" j_ref init_state
@printf "%s\n\n" ("─"^65)
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Build HS MPO
# ═══════════════════════════════════════════════════════════════════════════════

@printf "[1/5] Building HS MPO (N=%d, J=%.4g, PBC=true)… " N J
flush(stdout)
t0 = time_ns()
H_mpo = MPOMod.init_haldane_shastry(N; J=J, pbc=true)
elapsed_ms = (time_ns() - t0) / 1e6
@printf "done (%.1f ms)\n" elapsed_ms

max_bd = maximum(s -> max(size(s, 1), size(s, 4)), H_mpo.tensors)
@printf "    MPO max bond dim = %d\n" max_bd
flush(stdout)

# ── Optional MPO validation ────────────────────────────────────────────────────
if do_validate
    N_val = min(N, 8)
    @printf "[1/5] Validating HS MPO (N=%d)… " N_val
    flush(stdout)
    t0 = time_ns()
    ok = MPOMod.validate_hs_mpo(N_val; J=J, pbc=true)
    elapsed_ms = (time_ns() - t0) / 1e6
    if ok
        @printf "PASSED  (%.1f ms)\n\n" elapsed_ms
    else
        @printf "FAILED  (%.1f ms)\n\n" elapsed_ms
        @warn "MPO validation failed; results may be incorrect."
    end
    flush(stdout)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Exact dense reference (eigendecomposition)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Dense H uses LSB convention: state index = Σ_k s_k·2^(k−1) with site 1 = bit 0.
# This matches the MPS `to_vec` convention.

@printf "[2/5] Building dense reference (N=%d, dim=%d)… " N 2^N
flush(stdout)
t0 = time_ns()

const _I2 = ComplexF64[1 0; 0 1]
const _Sx = ComplexF64[0 0.5; 0.5 0]
const _Sy = ComplexF64[0 -0.5im; 0.5im 0]
const _Sz = ComplexF64[0.5 0; 0 -0.5]

function _kron_lsb(ops::NTuple{K,Matrix{ComplexF64}}) where K
    # kron from site N down to site 1 gives site 1 = bit 0 (LSB)
    acc = ones(ComplexF64, 1, 1)
    for k in K:-1:1
        acc = kron(acc, ops[k])
    end
    return acc
end

function _build_dense_hs(N::Int; J::Real=1.0, pbc::Bool=true)
    dim = 2^N
    H   = zeros(ComplexF64, dim, dim)
    S_ops_vec = (_Sx, _Sy, _Sz)
    for i in 1:N, j in (i + 1):N
        Jij = pbc ? Float64(J) * (π / N)^2 / sin(π * (j - i) / N)^2 :
                    Float64(J) / (j - i)^2
        iszero(Jij) && continue
        for S in S_ops_vec
            ops = ntuple(k -> (k == i || k == j) ? S : _I2, N)
            H .+= Jij .* _kron_lsb(ops)
        end
    end
    return Hermitian(H)
end

H_dense = _build_dense_hs(N; J=J, pbc=true)
elapsed_ms = (time_ns() - t0) / 1e6
@printf "done (%.1f ms)\n" elapsed_ms
flush(stdout)

@printf "    Diagonalizing %dx%d Hermitian matrix… " 2^N 2^N
flush(stdout)
t0 = time_ns()
F_eig   = eigen(H_dense)
E_eig   = real.(F_eig.values)
V_eig   = F_eig.vectors
elapsed_ms = (time_ns() - t0) / 1e6
@printf "done (%.1f ms)   E_gs = %.6f\n\n" elapsed_ms E_eig[1]
flush(stdout)

# Initial state (same convention used for both dense reference and MPS)
# LSB: site k = bit (k-1);  '0'=↑ (spin up), '1'=↓ (spin down).
function _make_init_state(init::String, N_::Int, j_ref_::Int)
    bits = if init == "neel"
        # |↑↓↑↓…⟩  — Sz=0 sector, fastest entanglement growth
        [isodd(k) ? '0' : '1' for k in 1:N_]
    elseif init == "wall"
        # |↑…↑↓…↓⟩  domain wall at centre — Sz=0, linear entanglement growth
        [k <= N_ ÷ 2 ? '0' : '1' for k in 1:N_]
    else  # spin_flip
        b = fill('0', N_)
        b[j_ref_] = '1'
        b
    end
    str = String(bits)
    # Dense vector index (1-based): Σ_k bit(k) * 2^(k-1)  +  1
    idx = sum(parse(Int, bits[k]) * 2^(k - 1) for k in 1:N_) + 1
    return str, idx
end

_init_str_dense, _init_idx_dense = _make_init_state(init_state, N, j_ref)
psi0_dense = zeros(ComplexF64, 2^N)
psi0_dense[_init_idx_dense] = 1.0
c0 = V_eig' * psi0_dense
@printf "    Initial state: %s  (basis string: %s)\n" init_state _init_str_dense

# Precompute diagonal Sz operators (LSB convention, site k = bit k-1)
function _sz_diag_lsb(site::Int, N::Int)
    d   = 2^N
    out = zeros(Float64, d)
    @inbounds for x in 0:(d - 1)
        bit       = (x >> (site - 1)) & 1
        out[x + 1] = 0.5 * (1 - 2 * bit)   # +0.5 if ↑, −0.5 if ↓
    end
    return out
end

sz_diags = [_sz_diag_lsb(k, N) for k in 1:N]

# ── Exact time evolution & C_zz collection ────────────────────────────────────
n_t   = length(t_grid)
xs    = collect(0:(N - 1))
xs_vec = xs   # alias for plotting

C_zz_exact   = zeros(Float64, n_t, N)
sz_exact_all = zeros(Float64, n_t, N)

@printf "[2/5] Exact time evolution (%d snapshots)… " n_t
flush(stdout)
t0 = time_ns()

for (ti, t) in enumerate(t_grid)
    phases = exp.((-im) .* E_eig .* t)
    psi_t  = V_eig * (c0 .* phases)
    prob   = abs2.(psi_t)

    for k in 1:N
        sz_exact_all[ti, k] = dot(sz_diags[k], prob)
    end

    sz_j = sz_exact_all[ti, j_ref]
    for (xi, x) in enumerate(xs)
        i_wrap = mod1(j_ref + x, N)
        if i_wrap == j_ref
            sz2                = dot(abs2.(sz_diags[j_ref]), prob)
            C_zz_exact[ti, xi] = sz2 - sz_j^2
        else
            szsz               = dot(sz_diags[i_wrap] .* sz_diags[j_ref], prob)
            C_zz_exact[ti, xi] = szsz - sz_exact_all[ti, i_wrap] * sz_j
        end
    end
end

elapsed_ms = (time_ns() - t0) / 1e6
@printf "done (%.1f ms)\n\n" elapsed_ms
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: run one MPS simulation and return C_zz(t, x)
# ═══════════════════════════════════════════════════════════════════════════════

const Sz_mat  = ComplexF64.(0.5 .* Matrix(GL.matrix(GL.ZGate())))  # S^z = (1/2)σ^z
const Z_ops   = [Sz_mat for _ in 1:N]

# Reuse the same initial-state string computed for the dense reference
const _init_str = _init_str_dense

# Observation schedule
const obs_at_step = round.(Int, t_grid ./ dt)   # length n_t

function _run_mps_method(
        method_name::String,
        H_mpo,
        n_steps::Int, dt_val::Float64, chi::Int, thr::Float64,
        init_str::String, j_ref_::Int, xs_vec_::Vector{Int},
        t_grid_::Vector{Float64}, sz_mat::Matrix{ComplexF64},
        N_::Int, obs_at_step_::Vector{Int})

    # Fresh MPS
    psi = MPSMod.MPS(N_; state="basis", basis_string=init_str)

    # Config
    cfg = Cfg.TimeEvolutionConfig(
        Cfg.Observable[], dt_val;
        dt=dt_val,
        max_bond_dim=chi,
        truncation_threshold=thr,
        sample_timesteps=false,
    )

    # Pad initial bond dimension
    MPSMod.pad_bond_dimension!(psi, min(4, chi); noise_scale=1e-10)

    n_t_  = length(t_grid_)
    N_obs = length(xs_vec_)
    C_zz  = zeros(ComplexF64, n_t_, N_)
    z_ops = [sz_mat for _ in 1:N_]

    function measure!(idx)
        czz = MPSMod.connected_czz(psi, sz_mat, j_ref_, xs_vec_; periodic=true)
        C_zz[idx, :] .= czz
    end

    obs_idx = Ref(1)
    if obs_at_step_[obs_idx[]] == 0
        measure!(obs_idx[])
        obs_idx[] += 1
    end

    t_start = time_ns()

    for step in 1:n_steps
        if method_name == "two_site_tdvp"
            Algo.two_site_tdvp!(psi, H_mpo, cfg)
        elseif method_name == "bug_second_order"
            BUGMod.bug_second_order!(psi, H_mpo, cfg)
        else
            error("Unknown method: $method_name")
        end

        if obs_idx[] <= n_t_ && obs_at_step_[obs_idx[]] == step
            measure!(obs_idx[])
            obs_idx[] += 1
        end
    end

    wall_s    = (time_ns() - t_start) / 1e9
    chi_final = MPSMod.write_max_bond_dim(psi)

    return real.(C_zz), wall_s, chi_final
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MPS time evolution — two-site TDVP
# ═══════════════════════════════════════════════════════════════════════════════

@printf "[3/5] MPS time evolution — two_site_tdvp (χ_max=%d)…\n" chi_max
flush(stdout)

C_zz_tdvp, wall_tdvp, chi_tdvp = _run_mps_method(
    "two_site_tdvp", H_mpo,
    n_steps, dt, chi_max, svd_thr,
    _init_str, j_ref, xs_vec, t_grid, Sz_mat,
    N, obs_at_step,
)

@printf "    2TDVP done: %.2f s,  χ_final_max = %d\n\n" wall_tdvp chi_tdvp
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MPS time evolution — second-order adaptive BUG
# ═══════════════════════════════════════════════════════════════════════════════

@printf "[4/5] MPS time evolution — bug_second_order (χ_max=%d)…\n" chi_max
flush(stdout)

C_zz_bug, wall_bug, chi_bug = _run_mps_method(
    "bug_second_order", H_mpo,
    n_steps, dt, chi_max, svd_thr,
    _init_str, j_ref, xs_vec, t_grid, Sz_mat,
    N, obs_at_step,
)

@printf "    BUG done:   %.2f s,  χ_final_max = %d\n\n" wall_bug chi_bug
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Compare both methods vs exact & report errors
# ═══════════════════════════════════════════════════════════════════════════════

@printf "[5/5] Error comparison vs exact…\n\n"

function _error_stats(C_approx, C_ref)
    diff     = C_approx .- C_ref
    abs_diff = abs.(diff)
    max_err  = maximum(abs_diff)
    mean_err = sum(abs_diff) / length(abs_diff)
    rel_err  = norm(diff) / max(1e-14, norm(C_ref))
    err_t    = vec(maximum(abs_diff, dims=2))   # max over x at each t
    err_x    = vec(maximum(abs_diff, dims=1))   # max over t at each x
    return (; abs_diff, max_err, mean_err, rel_err, err_t, err_x)
end

stats_tdvp = _error_stats(C_zz_tdvp, C_zz_exact)
stats_bug  = _error_stats(C_zz_bug,  C_zz_exact)

@printf "  %-46s  %-14s  %-14s\n" "Metric" "2TDVP" "BUG (2nd order)"
@printf "  %s\n" ("─"^78)
@printf "  %-46s  %-14.4e  %-14.4e\n" "max  |C_zz_approx − C_zz_exact|" stats_tdvp.max_err  stats_bug.max_err
@printf "  %-46s  %-14.4e  %-14.4e\n" "mean |C_zz_approx − C_zz_exact|" stats_tdvp.mean_err stats_bug.mean_err
@printf "  %-46s  %-14.4e  %-14.4e\n" "relative Frobenius norm error"    stats_tdvp.rel_err  stats_bug.rel_err
@printf "  %-46s  %-14.4e  %-14.4e\n" "max error in t" maximum(stats_tdvp.err_t) maximum(stats_bug.err_t)
@printf "  %-46s  %-14.4e  %-14.4e\n" "max error in x" maximum(stats_tdvp.err_x) maximum(stats_bug.err_x)
@printf "  %-46s  %-14.2f  %-14.2f\n" "wall time (s)" wall_tdvp wall_bug
@printf "  %-46s  %-14d  %-14d\n"     "final max χ"    chi_tdvp   chi_bug
@printf "\n"
flush(stdout)

# ── Save CSVs ─────────────────────────────────────────────────────────────────
function _save_czz_csv(path, t_grid_, C_zz, xs_vec_)
    open(path, "w") do f
        println(f, "t," * join(["Czz_x$(x)" for x in xs_vec_], ","))
        for ti in axes(C_zz, 1)
            println(f, "$(t_grid_[ti])," * join(["$(C_zz[ti,xi])" for xi in axes(C_zz, 2)], ","))
        end
    end
end

exact_csv = joinpath(results_dir, "exact_hs_czz_N$(N).csv")
tdvp_csv  = joinpath(results_dir, "tdvp_hs_czz_N$(N)_chi$(chi_max).csv")
bug_csv   = joinpath(results_dir, "bug_hs_czz_N$(N)_chi$(chi_max).csv")

_save_czz_csv(exact_csv, t_grid, C_zz_exact, xs_vec)
_save_czz_csv(tdvp_csv,  t_grid, C_zz_tdvp,  xs_vec)
_save_czz_csv(bug_csv,   t_grid, C_zz_bug,   xs_vec)

@printf "  Exact C_zz CSV  → %s\n"  exact_csv
@printf "  2TDVP C_zz CSV  → %s\n"  tdvp_csv
@printf "  BUG   C_zz CSV  → %s\n\n" bug_csv
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Figures (optional – requires PythonCall / matplotlib)
# ═══════════════════════════════════════════════════════════════════════════════

if do_plot
    try
        using PythonCall
        mpl = pyimport("matplotlib")
        mpl.use("Agg")
        plt = pyimport("matplotlib.pyplot")
        np  = pyimport("numpy")

        _jl2py(A) = np.array(A)

        # ── Figure 1: C_zz heatmaps (exact | 2TDVP | BUG) ────────────────────
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

        ext   = _jl2py([0.0, T_total, -0.5, Float64(N) - 0.5])
        vmax  = max(
            Float64(maximum(abs.(C_zz_exact))),
            Float64(maximum(abs.(C_zz_tdvp))),
            Float64(maximum(abs.(C_zz_bug))),
        ) * 0.85 + 1e-14

        im_e = axes1[0].imshow(_jl2py(C_zz_exact'),
            aspect="auto", origin="lower", extent=ext,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig1.colorbar(im_e, ax=axes1[0]).set_label(raw"$C_{zz}$")
        axes1[0].set_xlabel("t"); axes1[0].set_ylabel("x")
        axes1[0].set_title("Exact")

        im_t = axes1[1].imshow(_jl2py(C_zz_tdvp'),
            aspect="auto", origin="lower", extent=ext,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig1.colorbar(im_t, ax=axes1[1]).set_label(raw"$C_{zz}$")
        axes1[1].set_xlabel("t"); axes1[1].set_ylabel("x")
        axes1[1].set_title("2TDVP  (χ_max=$chi_max,  χ_final=$chi_tdvp)")

        im_b = axes1[2].imshow(_jl2py(C_zz_bug'),
            aspect="auto", origin="lower", extent=ext,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig1.colorbar(im_b, ax=axes1[2]).set_label(raw"$C_{zz}$")
        axes1[2].set_xlabel("t"); axes1[2].set_ylabel("x")
        axes1[2].set_title("BUG 2nd order  (χ_max=$chi_max,  χ_final=$chi_bug)")

        fig1.suptitle(
            "Haldane–Shastry  N=$N  J=$J  PBC  init=$init_state  ($(_init_str))",
            fontsize=11,
        )
        plt.tight_layout()
        p1 = joinpath(figures_dir, "hs_czz_heatmap_N$(N)_chi$(chi_max).png")
        fig1.savefig(p1, dpi=150)
        plt.close(fig1)
        @printf "  Heatmap figure         → %s\n" p1

        # ── Figure 2: error heatmaps + line cuts ──────────────────────────────
        fig2, axes2 = plt.subplots(1, 3, figsize=(20, 5))

        vmax_err = max(
            Float64(stats_tdvp.max_err),
            Float64(stats_bug.max_err),
        ) + 1e-14

        im_et = axes2[0].imshow(_jl2py(stats_tdvp.abs_diff'),
            aspect="auto", origin="lower", extent=ext,
            cmap="hot_r", vmin=0.0, vmax=vmax_err)
        fig2.colorbar(im_et, ax=axes2[0]).set_label(raw"|$\Delta C_{zz}$|")
        axes2[0].set_xlabel("t"); axes2[0].set_ylabel("x")
        axes2[0].set_title("2TDVP error  (max=$(round(stats_tdvp.max_err; sigdigits=2)))")

        im_eb = axes2[1].imshow(_jl2py(stats_bug.abs_diff'),
            aspect="auto", origin="lower", extent=ext,
            cmap="hot_r", vmin=0.0, vmax=vmax_err)
        fig2.colorbar(im_eb, ax=axes2[1]).set_label(raw"|$\Delta C_{zz}$|")
        axes2[1].set_xlabel("t"); axes2[1].set_ylabel("x")
        axes2[1].set_title("BUG error  (max=$(round(stats_bug.max_err; sigdigits=2)))")

        # Line cuts at evenly spaced time points
        n_cuts  = min(5, n_t)
        cut_idx = round.(Int, LinRange(1, n_t, n_cuts))
        colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        for (ci, ti) in enumerate(cut_idx)
            t_val  = t_grid[ti]
            lbl    = "t=$(round(t_val, digits=2))"
            col    = colors[ci]
            axes2[2].plot(xs_vec, C_zz_exact[ti, :],  color=col, lw=1.8, ls="-",  label="exact $lbl")
            axes2[2].plot(xs_vec, C_zz_tdvp[ti, :],   color=col, lw=1.2, ls="--", label="2TDVP $lbl")
            axes2[2].plot(xs_vec, C_zz_bug[ti, :],    color=col, lw=1.2, ls=":",  label="BUG $lbl")
        end
        axes2[2].axhline(0, color="gray", lw=0.7, ls=":")
        axes2[2].set_xlabel("x")
        axes2[2].set_ylabel(raw"$C_{zz}(t,x)$")
        axes2[2].set_title("Line cuts: exact (─), 2TDVP (--), BUG (⋯)")
        axes2[2].legend(fontsize=5, ncol=2)

        fig2.suptitle(
            "HS N=$N  χ_max=$chi_max  — 2TDVP rel_err=$(round(stats_tdvp.rel_err; sigdigits=2))  BUG rel_err=$(round(stats_bug.rel_err; sigdigits=2))",
            fontsize=10,
        )
        plt.tight_layout()
        p2 = joinpath(figures_dir, "hs_czz_error_N$(N)_chi$(chi_max).png")
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)
        @printf "  Error/line-cut figure  → %s\n\n" p2

    catch e
        @warn "Matplotlib plotting failed: $e"
        @printf "  (Skipping plots; CSVs were saved above.)\n\n"
    end
end

@printf "%s\n" ("─"^65)
@printf "  HS benchmark complete.\n"
@printf "%s\n\n" ("─"^65)