using LinearAlgebra
using Printf
using Dates

include("../src/Yaqs.jl")
using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.Algorithms
using .Yaqs.BUGModule
using .Yaqs.SimulationConfigs

include("nonhermitian_tfim.jl")
include("exact_reference.jl")
include("longrange_nonhermitian_tfim.jl")
include("longrange_exact_reference.jl")
include("bug_nonhermitian.jl")

function parse_args_bug(args::Vector{String})
    params = Dict{String, String}()
    i = 1
    while i <= length(args)
        key = args[i]
        if startswith(key, "--")
            if i == length(args) || startswith(args[i + 1], "--")
                params[key] = "true"
                i += 1
            else
                params[key] = args[i + 1]
                i += 2
            end
        else
            i += 1
        end
    end
    return params
end

getp(d::Dict{String, String}, k::String, v::Int) = haskey(d, k) ? parse(Int, d[k]) : v
getp(d::Dict{String, String}, k::String, v::Float64) = haskey(d, k) ? parse(Float64, d[k]) : v
getp(d::Dict{String, String}, k::String, v::String) = haskey(d, k) ? lowercase(d[k]) : v

function parse_D_list_bug(params::Dict{String, String})
    if !haskey(params, "--D")
        return [4, 8, 16, 32]
    end
    vals = Int[]
    for token in split(params["--D"], ",")
        s = strip(token)
        isempty(s) && continue
        push!(vals, parse(Int, s))
    end
    isempty(vals) && error("Empty --D list.")
    return vals
end

function parse_lambda_list_bug(params::Dict{String, String}, default_lambda::Float64)
    if !haskey(params, "--lambda-list")
        return [default_lambda]
    end
    vals = Float64[]
    for token in split(params["--lambda-list"], ",")
        s = strip(token)
        isempty(s) && continue
        push!(vals, parse(Float64, s))
    end
    isempty(vals) && error("Empty --lambda-list.")
    return vals
end

function ensure_outdir_bug()
    outdir = joinpath(@__DIR__, "results_bug2_compare")
    isdir(outdir) || mkpath(outdir)
    return outdir
end

@inline function _normed_state(v::Vector{ComplexF64})
    n = norm(v)
    if n > 0
        return v ./ n
    end
    return similar(v, length(v)) .= ComplexF64(NaN)
end

function write_timeseries_bug(path::String, rows::Vector{NamedTuple}, N::Int)
    open(path, "w") do io
        header = [
            "t", "norm_ref", "norm_tdvp", "norm_bug",
            "err_state_tdvp", "err_state_bug",
            "err_state_normed_tdvp", "err_state_normed_bug",
            "max_abs_err_z_tdvp", "max_abs_err_z_bug",
            "max_bond_dim_tdvp", "max_bond_dim_bug",
        ]
        for j in 1:N
            push!(header, "z_ref_$j")
        end
        for j in 1:N
            push!(header, "z_tdvp_$j")
        end
        for j in 1:N
            push!(header, "z_bug_$j")
        end
        write(io, join(header, ",") * "\n")

        for row in rows
            fields = String[
                @sprintf("%.10f", row.t),
                @sprintf("%.16e", row.norm_ref),
                @sprintf("%.16e", row.norm_tdvp),
                @sprintf("%.16e", row.norm_bug),
                @sprintf("%.16e", row.err_tdvp),
                @sprintf("%.16e", row.err_bug),
                @sprintf("%.16e", row.errn_tdvp),
                @sprintf("%.16e", row.errn_bug),
                @sprintf("%.16e", row.maxz_tdvp),
                @sprintf("%.16e", row.maxz_bug),
                string(row.maxbond_tdvp),
                string(row.maxbond_bug),
            ]
            for j in 1:N
                push!(fields, @sprintf("%.16e", row.z_ref[j]))
            end
            for j in 1:N
                push!(fields, @sprintf("%.16e", row.z_tdvp[j]))
            end
            for j in 1:N
                push!(fields, @sprintf("%.16e", row.z_bug[j]))
            end
            write(io, join(fields, ",") * "\n")
        end
    end
end

function run_compare_for_model(model_name::String,
                               K_mpo::MPO,
                               K_dense::Matrix{ComplexF64},
                               psi0_dense::Vector{ComplexF64},
                               D::Int,
                               N::Int;
                               dt::Float64,
                               tmax::Float64,
                               bug_numiter::Int=25)
    times, ref_states, zdiag, exact_runtime = run_longrange_exact_reference(K_dense, psi0_dense; dt=dt, tmax=tmax)

    K_tdvp = (1im) * K_mpo
    K_bug = K_tdvp
    z_ops = [pauli_z_lr() for _ in 1:N]

    psi_tdvp = initial_plus_mps_lr(N)
    psi_bug = initial_plus_mps_lr(N)
    psi0_mps = to_vec(initial_plus_mps_lr(N))
    init_err = norm(psi0_mps - psi0_dense)

    cfg_tdvp = TimeEvolutionConfig(Observable[], dt; dt=dt, max_bond_dim=D, truncation_threshold=1e-12)
    cfg_bug = TimeEvolutionConfig(Observable[], dt; dt=dt, max_bond_dim=D, truncation_threshold=1e-12)

    rows = Vector{NamedTuple}(undef, length(times))
    t_tdvp = 0.0
    t_bug = 0.0
    bug_warn_norm = true
    nan_warn = 0

    for n in eachindex(times)
        if n > 1
            t0 = time()
            two_site_tdvp!(psi_tdvp, K_tdvp, cfg_tdvp)
            t_tdvp += time() - t0

            t0 = time()
            bug_second_order_nonhermitian!(psi_bug, K_bug, cfg_bug;
                                           numiter_lanczos=bug_numiter,
                                           truncation_granularity=:after_site)
            t_bug += time() - t0
        end

        v_ref = ref_states[n]
        v_tdvp = to_vec(psi_tdvp)
        v_bug = to_vec(psi_bug)

        # BUG performs internal truncation via `truncate!`, which normalizes.
        if bug_warn_norm && abs(norm(v_bug) - 1.0) < 1e-6 && abs(norm(v_ref) - 1.0) > 1e-3
            @warn "BUG trajectory appears normalized while reference is not; use normalized-state error for fair direction comparison."
            bug_warn_norm = false
        end

        err_tdvp = norm(v_tdvp - v_ref)
        err_bug = norm(v_bug - v_ref)
        if !(isfinite(err_tdvp) && isfinite(err_bug))
            nan_warn += 1
            @warn "Non-finite state error at t=$(times[n]) in model=$model_name, D=$D"
        end

        vn_ref = _normed_state(v_ref)
        vn_tdvp = _normed_state(v_tdvp)
        vn_bug = _normed_state(v_bug)
        errn_tdvp = norm(vn_tdvp - vn_ref)
        errn_bug = norm(vn_bug - vn_ref)

        z_ref_u, _ = z_expectations_from_state(v_ref, zdiag)
        z_tdvp_u = ComplexF64.(evaluate_all_local_expectations(psi_tdvp, z_ops))
        z_bug_u = ComplexF64.(evaluate_all_local_expectations(psi_bug, z_ops))
        z_ref = real.(z_ref_u)
        z_tdvp = real.(z_tdvp_u)
        z_bug = real.(z_bug_u)

        rows[n] = (
            t = times[n],
            norm_ref = norm(v_ref),
            norm_tdvp = norm(v_tdvp),
            norm_bug = norm(v_bug),
            err_tdvp = err_tdvp,
            err_bug = err_bug,
            errn_tdvp = errn_tdvp,
            errn_bug = errn_bug,
            maxz_tdvp = maximum(abs.(z_tdvp .- z_ref)),
            maxz_bug = maximum(abs.(z_bug .- z_ref)),
            maxbond_tdvp = write_max_bond_dim(psi_tdvp),
            maxbond_bug = write_max_bond_dim(psi_bug),
            z_ref = z_ref,
            z_tdvp = z_tdvp,
            z_bug = z_bug,
        )
    end

    return rows, exact_runtime, t_tdvp, t_bug, init_err, nan_warn
end

function main(args::Vector{String}=ARGS)
    p = parse_args_bug(args)

    N = getp(p, "--N", 10)
    J = getp(p, "--J", 1.0)
    J0 = getp(p, "--J0", 1.0)
    lambda0 = getp(p, "--lambda", 0.5)
    g = getp(p, "--g", 0.7)
    gamma = getp(p, "--gamma", 0.2)
    dt = getp(p, "--dt", 0.05)
    tmax = getp(p, "--tmax", 1.0)
    Ds = parse_D_list_bug(p)
    lambdas = parse_lambda_list_bug(p, lambda0)
    krylov_mode = Symbol(getp(p, "--krylov", "arnoldi"))
    @assert krylov_mode == :arnoldi || krylov_mode == :lanczos || krylov_mode == :auto

    outdir = ensure_outdir_bug()
    set_krylov_ishermitian_mode!(krylov_mode)
    println("=== BUG2 vs TDVP Non-Hermitian Comparison ===")
    println("krylov mode: ", krylov_mode)
    println("D list: ", Ds)
    println("long-range lambdas: ", lambdas)

    summaries = String[]
    push!(summaries, "BUG2 vs TDVP non-Hermitian comparison")
    push!(summaries, "timestamp=$(Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")
    push!(summaries, "N=$N J=$J J0=$J0 g=$g gamma=$gamma dt=$dt tmax=$tmax krylov=$krylov_mode")

    # Models: nearest-neighbor + long-range sweep
    models = Vector{Tuple{String, MPO, Matrix{ComplexF64}, Vector{ComplexF64}}}()
    begin
        K_mpo = build_nonhermitian_k_mpo(N; J=J, g=g, gamma=gamma)
        K_dense = build_k_dense(N; J=J, g=g, gamma=gamma)
        psi0 = initial_plus_dense(N)
        push!(models, ("nn", K_mpo, K_dense, psi0))
    end
    for lam in lambdas
        @assert 0 < lam < 1
        K_mpo = build_longrange_nonhermitian_k_mpo(N; J0=J0, lambda=lam, g=g, gamma=gamma)
        K_dense = build_longrange_k_dense(N; J0=J0, lambda=lam, g=g, gamma=gamma)
        psi0 = initial_plus_dense_lr(N)
        push!(models, ("lr_lambda$(lam)", K_mpo, K_dense, psi0))
    end

    for (model_name, K_mpo, K_dense, psi0) in models
        println("\n--- model: ", model_name, " ---")
        # MPO-vs-dense sanity
        psi0_mps = initial_plus_mps_lr(N)
        act_err = norm(to_vec(contract_mpo_mps(K_mpo, psi0_mps)) - (K_dense * psi0))
        @printf("Sanity: ||K_mpo|psi0>-K_dense|psi0>|| = %.3e\n", act_err)
        push!(summaries, @sprintf("model=%s mpo_dense_action_error=%.6e", model_name, act_err))

        for D in Ds
            rows, texact, ttdvp, tbug, init_err, nan_warn = run_compare_for_model(
                model_name, K_mpo, K_dense, psi0, D, N; dt=dt, tmax=tmax, bug_numiter=25
            )
            csv_path = joinpath(outdir, "timeseries_$(replace(model_name, '.' => 'p'))_D$(D).csv")
            write_timeseries_bug(csv_path, rows, N)

            final_row = rows[end]
            line = @sprintf(
                "model=%s D=%d exact_runtime=%.6f tdvp_runtime=%.6f bug2_runtime=%.6f final_err_tdvp=%.6e final_err_bug2=%.6e final_errn_tdvp=%.6e final_errn_bug2=%.6e final_maxz_tdvp=%.6e final_maxz_bug2=%.6e init_err=%.6e nan_warnings=%d",
                model_name, D, texact, ttdvp, tbug,
                final_row.err_tdvp, final_row.err_bug,
                final_row.errn_tdvp, final_row.errn_bug,
                final_row.maxz_tdvp, final_row.maxz_bug,
                init_err, nan_warn
            )
            push!(summaries, line)
            @printf("D=%d | final err TDVP=%.3e BUG2=%.3e | normalized err TDVP=%.3e BUG2=%.3e\n",
                    D, final_row.err_tdvp, final_row.err_bug, final_row.errn_tdvp, final_row.errn_bug)
        end
    end

    summary_path = joinpath(outdir, "summary.txt")
    open(summary_path, "w") do io
        for s in summaries
            write(io, s * "\n")
        end
    end
    println("\nWrote comparison results to ", outdir)
    println("Summary: ", summary_path)
end

main()
