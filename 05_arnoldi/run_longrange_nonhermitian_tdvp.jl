using LinearAlgebra
using Printf
using Dates

include("../src/Yaqs.jl")
using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.Algorithms
using .Yaqs.SimulationConfigs

include("longrange_nonhermitian_tfim.jl")
include("longrange_exact_reference.jl")

function parse_args_lr(args::Vector{String})
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

function get_param_lr(params::Dict{String, String}, key::String, default::Int)
    return haskey(params, key) ? parse(Int, params[key]) : default
end
function get_param_lr(params::Dict{String, String}, key::String, default::Float64)
    return haskey(params, key) ? parse(Float64, params[key]) : default
end
function get_param_lr(params::Dict{String, String}, key::String, default::String)
    return haskey(params, key) ? lowercase(params[key]) : default
end

function parse_list_f64(csv_like::String)
    vals = Float64[]
    for token in split(csv_like, ",")
        s = strip(token)
        isempty(s) && continue
        push!(vals, parse(Float64, s))
    end
    isempty(vals) && error("Expected non-empty comma-separated list.")
    return vals
end

function parse_bond_dims_lr(params::Dict{String, String})
    if !haskey(params, "--D")
        return [4, 8, 16]
    end
    vals = Int[]
    for token in split(params["--D"], ",")
        s = strip(token)
        isempty(s) && continue
        push!(vals, parse(Int, s))
    end
    isempty(vals) && error("Parsed empty bond-dimension list from --D.")
    return vals
end

function parse_lambdas_lr(params::Dict{String, String}, default_lambda::Float64)
    if haskey(params, "--lambda-list")
        return parse_list_f64(params["--lambda-list"])
    end
    return [default_lambda]
end

function lambda_tag(lam::Float64)
    s = @sprintf("%.6f", lam)
    s = rstrip(s, '0')
    s = rstrip(s, '.')
    isempty(s) && (s = "0")
    return s
end

function ensure_outdir_lr()
    outdir = joinpath(@__DIR__, "results_longrange")
    isdir(outdir) || mkpath(outdir)
    return outdir
end

function write_timeseries_lr(path::String, rows::Vector{NamedTuple}, N::Int)
    open(path, "w") do io
        header = ["t", "norm_ref", "norm_tdvp", "err_state", "max_abs_err_z", "max_bond_dim"]
        for j in 1:N
            push!(header, "z_ref_$j")
        end
        for j in 1:N
            push!(header, "z_tdvp_$j")
        end
        for j in 1:N
            push!(header, "znorm_ref_$j")
        end
        for j in 1:N
            push!(header, "znorm_tdvp_$j")
        end
        write(io, join(header, ",") * "\n")

        for row in rows
            fields = String[
                @sprintf("%.10f", row.t),
                @sprintf("%.16e", row.norm_ref),
                @sprintf("%.16e", row.norm_tdvp),
                @sprintf("%.16e", row.err_state),
                @sprintf("%.16e", row.max_abs_err_z),
                string(row.max_bond_dim),
            ]
            for j in 1:N
                push!(fields, @sprintf("%.16e", row.z_ref[j]))
            end
            for j in 1:N
                push!(fields, @sprintf("%.16e", row.z_tdvp[j]))
            end
            for j in 1:N
                push!(fields, @sprintf("%.16e", row.znorm_ref[j]))
            end
            for j in 1:N
                push!(fields, @sprintf("%.16e", row.znorm_tdvp[j]))
            end
            write(io, join(fields, ",") * "\n")
        end
    end
end

function run_tdvp_for_D_lr(times::Vector{Float64},
                           ref_states::Vector{Vector{ComplexF64}},
                           zdiag::Matrix{Float64},
                           K_tdvp::MPO,
                           N::Int,
                           D::Int;
                           dt::Float64=0.05)
    psi = initial_plus_mps_lr(N)
    psi0_mps = to_vec(psi)
    init_err = norm(psi0_mps - ref_states[1])

    cfg = TimeEvolutionConfig(Observable[], dt; dt=dt, max_bond_dim=D, truncation_threshold=1e-12)
    z_ops = [pauli_z_lr() for _ in 1:N]
    rows = Vector{NamedTuple}(undef, length(times))
    warnings_nan = 0

    t0 = time()
    for n in eachindex(times)
        if n > 1
            two_site_tdvp!(psi, K_tdvp, cfg)
        end

        psi_tdvp = to_vec(psi)
        psi_ref = ref_states[n]
        err_state = norm(psi_tdvp - psi_ref)
        if !isfinite(err_state)
            warnings_nan += 1
            @warn "Non-finite state error at t=$(times[n]) for D=$D"
        end

        z_ref_u, z_ref_n = z_expectations_from_state(psi_ref, zdiag)
        z_tdvp_u_c = evaluate_all_local_expectations(psi, z_ops)
        z_tdvp_u = ComplexF64.(z_tdvp_u_c)
        norm_sq_tdvp = real(dot(psi_tdvp, psi_tdvp))
        z_tdvp_n = norm_sq_tdvp > 0 ? (z_tdvp_u ./ norm_sq_tdvp) : fill(ComplexF64(NaN), N)

        z_ref = real.(z_ref_u)
        z_tdvp = real.(z_tdvp_u)
        znorm_ref = real.(z_ref_n)
        znorm_tdvp = real.(z_tdvp_n)
        max_abs_err_z = maximum(abs.(z_tdvp .- z_ref))

        rows[n] = (
            t = times[n],
            norm_ref = norm(psi_ref),
            norm_tdvp = norm(psi_tdvp),
            err_state = err_state,
            max_abs_err_z = max_abs_err_z,
            max_bond_dim = write_max_bond_dim(psi),
            z_ref = z_ref,
            z_tdvp = z_tdvp,
            znorm_ref = znorm_ref,
            znorm_tdvp = znorm_tdvp,
        )
    end
    runtime_seconds = time() - t0
    return rows, runtime_seconds, init_err, warnings_nan
end

function main(args::Vector{String}=ARGS)
    p = parse_args_lr(args)

    N = get_param_lr(p, "--N", 10)
    J0 = get_param_lr(p, "--J0", 1.0)
    lambda0 = get_param_lr(p, "--lambda", 0.5)
    g = get_param_lr(p, "--g", 0.7)
    gamma = get_param_lr(p, "--gamma", 0.2)
    dt = get_param_lr(p, "--dt", 0.05)
    tmax = get_param_lr(p, "--tmax", 1.0)
    D_list = parse_bond_dims_lr(p)
    lambdas = parse_lambdas_lr(p, lambda0)
    krylov_s = get_param_lr(p, "--krylov", "arnoldi")
    krylov_mode = Symbol(krylov_s)
    @assert krylov_mode == :arnoldi || krylov_mode == :lanczos "krylov must be arnoldi|lanczos"

    nsteps = Int(round(tmax / dt))
    @assert isapprox(nsteps * dt, tmax; atol=1e-12) "tmax must be an integer multiple of dt."

    outdir = ensure_outdir_lr()
    println("=== Long-range Non-Hermitian TDVP Benchmark ===")
    @printf("N=%d, J0=%.4f, g=%.4f, gamma=%.4f, dt=%.4f, tmax=%.4f, steps=%d\n",
            N, J0, g, gamma, dt, tmax, nsteps)
    println("lambda list: ", lambdas)
    println("bond dimensions: ", D_list)
    println("krylov mode: ", krylov_mode)

    # Arnoldi/Lanczos switch (existing API)
    set_krylov_ishermitian_mode!(krylov_mode)
    if krylov_mode != :arnoldi
        @warn "Non-Hermitian default is Arnoldi; using $(krylov_mode) as requested."
    end
    println("Normalization note: TDVP path used here does not call normalize! each step.")

    summaries = String[]
    push!(summaries, "Long-range non-Hermitian TDVP benchmark summary")
    push!(summaries, "timestamp=$(Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")
    push!(summaries, "N=$N J0=$J0 g=$g gamma=$gamma dt=$dt tmax=$tmax steps=$nsteps krylov=$krylov_mode")
    push!(summaries, "lambdas=$(join(string.(lambdas), ",")) D_list=$(join(string.(D_list), ","))")

    for lam in lambdas
        @assert 0 < lam < 1 "Each lambda must satisfy 0 < lambda < 1."
        tag = lambda_tag(lam)
        println("\n--- lambda = ", lam, " ---")

        K_mpo = build_longrange_nonhermitian_k_mpo(N; J0=J0, lambda=lam, g=g, gamma=gamma)
        verify_longrange_mpo_dims(K_mpo, N)
        K_tdvp = tdvp_generator_from_k_lr(K_mpo) # A = i K_lr

        psi0_dense = initial_plus_dense_lr(N)
        psi0_mps = to_vec(initial_plus_mps_lr(N))
        psi0_err = norm(psi0_dense - psi0_mps)
        @printf("Sanity: ||psi0_dense - psi0_mps|| = %.3e\n", psi0_err)

        K_dense = build_longrange_k_dense(N; J0=J0, lambda=lam, g=g, gamma=gamma)
        times_ref, ref_states, zdiag, exact_runtime = run_longrange_exact_reference(K_dense, psi0_dense; dt=dt, tmax=tmax)
        @printf("Exact reference runtime: %.3f s\n", exact_runtime)

        # Indirect MPO-vs-dense check at t=0 using action on |+>^N
        psi0_mps_obj = initial_plus_mps_lr(N)
        kpsi_mps = to_vec(contract_mpo_mps(K_mpo, psi0_mps_obj))
        kpsi_dense = K_dense * psi0_dense
        action_err = norm(kpsi_mps - kpsi_dense)
        @printf("Sanity: ||K_mpo|psi0> - K_dense|psi0>|| = %.3e\n", action_err)

        push!(summaries, @sprintf("lambda=%s exact_runtime_seconds=%.6f psi0_mps_dense_error=%.6e mpo_dense_action_error=%.6e",
                                   tag, exact_runtime, psi0_err, action_err))

        for D in D_list
            rows, tdvp_runtime, init_err, nan_warn = run_tdvp_for_D_lr(
                times_ref, ref_states, zdiag, K_tdvp, N, D; dt=dt
            )
            csv_path = joinpath(outdir, "timeseries_lambda$(tag)_D$(D).csv")
            write_timeseries_lr(csv_path, rows, N)

            final_row = rows[end]
            line = @sprintf(
                "lambda=%s D=%d tdvp_runtime_seconds=%.6f final_err_state=%.6e final_max_abs_err_z=%.6e final_norm_ref=%.6e final_norm_tdvp=%.6e init_err=%.6e nan_warnings=%d",
                tag, D, tdvp_runtime, final_row.err_state, final_row.max_abs_err_z,
                final_row.norm_ref, final_row.norm_tdvp, init_err, nan_warn
            )
            push!(summaries, line)
            @printf("D=%d done: runtime=%.3fs, final err_state=%.3e, final max|ΔZ|=%.3e\n",
                    D, tdvp_runtime, final_row.err_state, final_row.max_abs_err_z)
        end
    end

    summary_path = joinpath(outdir, "summary.txt")
    open(summary_path, "w") do io
        for line in summaries
            write(io, line * "\n")
        end
    end
    println("\nResults written to: ", outdir)
    println("Summary file: ", summary_path)
end

main()
