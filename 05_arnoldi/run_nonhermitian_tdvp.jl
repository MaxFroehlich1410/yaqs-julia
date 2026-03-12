using LinearAlgebra
using Printf
using Dates

include("../src/Yaqs.jl")
using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.Algorithms
using .Yaqs.SimulationConfigs

include("nonhermitian_tfim.jl")
include("exact_reference.jl")

function parse_args(args::Vector{String})
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

function get_param(params::Dict{String, String}, key::String, default::Int)
    return haskey(params, key) ? parse(Int, params[key]) : default
end

function get_param(params::Dict{String, String}, key::String, default::Float64)
    return haskey(params, key) ? parse(Float64, params[key]) : default
end

function get_param(params::Dict{String, String}, key::String, default::String)
    return haskey(params, key) ? lowercase(params[key]) : default
end

function parse_bond_dims(params::Dict{String, String})
    if !haskey(params, "--D")
        return [4, 8, 16]
    end
    raw = split(params["--D"], ",")
    vals = Int[]
    for token in raw
        stripped = strip(token)
        isempty(stripped) && continue
        push!(vals, parse(Int, stripped))
    end
    isempty(vals) && error("Parsed empty bond-dimension list from --D.")
    return vals
end

function ensure_results_dir()
    outdir = joinpath(@__DIR__, "results")
    isdir(outdir) || mkpath(outdir)
    return outdir
end

function verify_mpo_consistency(K_mpo::MPO, N::Int)
    @assert K_mpo.length == N
    for i in 1:N
        T = K_mpo.tensors[i]
        @assert ndims(T) == 4
        @assert size(T, 2) == 2
        @assert size(T, 3) == 2
    end
    return true
end

function write_timeseries_csv(path::String, rows::Vector{NamedTuple}, N::Int)
    open(path, "w") do io
        header = ["t", "norm_ref", "norm_tdvp", "err_state", "max_abs_err_z", "max_bond_dim"]
        for j in 1:N
            push!(header, "z_ref_$j")
        end
        for j in 1:N
            push!(header, "z_tdvp_$j")
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
            write(io, join(fields, ",") * "\n")
        end
    end
end

function run_one_tdvp_bond_dim(times::Vector{Float64},
                               ref_states::Vector{Vector{ComplexF64}},
                               zdiag::Matrix{Float64},
                               K_tdvp::MPO,
                               N::Int,
                               D::Int;
                               dt::Float64=0.05,
                               krylov::Symbol=:arnoldi)
    psi = initial_plus_mps(N)

    # Initial-state consistency check (MPS <-> dense)
    psi0_mps = to_vec(psi)
    psi0_ref = ref_states[1]
    init_err = norm(psi0_mps - psi0_ref)
    @printf("D=%d: initial state mismatch ||psi_mps(0)-psi_ref(0)|| = %.3e\n", D, init_err)

    cfg = TimeEvolutionConfig(Observable[], dt; dt=dt, max_bond_dim=D, truncation_threshold=1e-12)

    z_ops = [pauli_z() for _ in 1:N]
    rows = Vector{NamedTuple}(undef, length(times))
    warnings_nan = 0

    t_start = time()
    for n in eachindex(times)
        if n > 1
            two_site_tdvp!(psi, K_tdvp, cfg)
        end

        psi_tdvp = to_vec(psi)
        psi_ref = ref_states[n]

        z_ref_complex = exact_z_expectations(psi_ref, zdiag)
        z_tdvp_complex = evaluate_all_local_expectations(psi, z_ops)

        z_ref = real.(z_ref_complex)
        z_tdvp = real.(z_tdvp_complex)

        err_state = norm(psi_tdvp - psi_ref)
        max_abs_err_z = maximum(abs.(z_tdvp .- z_ref))
        if !isfinite(err_state)
            warnings_nan += 1
            @warn "Non-finite state error at t=$(times[n]) for D=$D."
        end

        rows[n] = (
            t = times[n],
            norm_ref = norm(psi_ref),
            norm_tdvp = norm(psi_tdvp),
            err_state = err_state,
            max_abs_err_z = max_abs_err_z,
            max_bond_dim = write_max_bond_dim(psi),
            z_ref = z_ref,
            z_tdvp = z_tdvp,
        )
    end
    runtime_seconds = time() - t_start
    return rows, runtime_seconds, init_err, warnings_nan
end

function main(args::Vector{String}=ARGS)
    p = parse_args(args)

    N = get_param(p, "--N", 10)
    J = get_param(p, "--J", 1.0)
    g = get_param(p, "--g", 0.7)
    gamma = get_param(p, "--gamma", 0.2)
    dt = get_param(p, "--dt", 0.05)
    tmax = get_param(p, "--tmax", 1.0)
    D_list = parse_bond_dims(p)
    krylov_s = get_param(p, "--krylov", "arnoldi")
    krylov_mode = Symbol(krylov_s)
    @assert krylov_mode == :arnoldi || krylov_mode == :lanczos "krylov must be arnoldi|lanczos"

    nsteps = Int(round(tmax / dt))
    @assert isapprox(nsteps * dt, tmax; atol=1e-12) "tmax must be an integer multiple of dt."

    outdir = ensure_results_dir()

    println("=== Non-Hermitian TDVP Benchmark (Arnoldi/Lanczos) ===")
    @printf("N=%d, J=%.4f, g=%.4f, gamma=%.4f, dt=%.4f, tmax=%.4f, steps=%d\n",
            N, J, g, gamma, dt, tmax, nsteps)
    println("Bond dimensions: ", D_list)
    println("Krylov mode: ", krylov_mode)

    # Build model objects
    K_mpo = build_nonhermitian_k_mpo(N; J=J, g=g, gamma=gamma)
    verify_mpo_consistency(K_mpo, N)
    K_tdvp = tdvp_generator_from_k(K_mpo) # A = iK so TDVP internal exp(-im dt A) gives exp(dt K)

    psi0_dense = initial_plus_dense(N)
    psi0_mps = to_vec(initial_plus_mps(N))
    psi0_check = norm(psi0_dense - psi0_mps)
    @printf("Sanity: ||psi0_dense - psi0_mps|| = %.3e\n", psi0_check)

    K_dense = build_k_dense(N; J=J, g=g, gamma=gamma)
    @assert size(K_dense, 1) == (1 << N)
    @assert size(K_dense, 2) == (1 << N)

    times_ref, ref_states, zdiag, exact_runtime = run_exact_reference(K_dense, psi0_dense; dt=dt, tmax=tmax)
    @printf("Exact reference runtime: %.3f s\n", exact_runtime)

    # Configure Krylov mode (Arnoldi is recommended for non-Hermitian dynamics)
    set_krylov_ishermitian_mode!(krylov_mode)
    if krylov_mode == :lanczos
        @warn "Lanczos requested on a non-Hermitian problem; Arnoldi is generally the safe choice."
    end

    # Normalization caveat from code inspection.
    println("Normalization note: TDVP core path does not call normalize! each step (unnormalized dynamics preserved).")

    summaries = String[]
    push!(summaries, "Non-Hermitian TDVP benchmark summary")
    push!(summaries, "timestamp=$(Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")
    push!(summaries, "N=$N J=$J g=$g gamma=$gamma dt=$dt tmax=$tmax steps=$nsteps krylov=$krylov_mode")
    push!(summaries, "exact_runtime_seconds=$(exact_runtime)")
    push!(summaries, "psi0_mps_dense_error=$(psi0_check)")

    for D in D_list
        rows, tdvp_runtime, init_err, nan_warnings = run_one_tdvp_bond_dim(
            times_ref, ref_states, zdiag, K_tdvp, N, D; dt=dt, krylov=krylov_mode
        )
        csv_path = joinpath(outdir, "timeseries_D$(D).csv")
        write_timeseries_csv(csv_path, rows, N)

        final_row = rows[end]
        line = @sprintf(
            "D=%d tdvp_runtime_seconds=%.6f final_err_state=%.6e final_max_abs_err_z=%.6e final_norm_ref=%.6e final_norm_tdvp=%.6e init_err=%.6e nan_warnings=%d",
            D, tdvp_runtime, final_row.err_state, final_row.max_abs_err_z,
            final_row.norm_ref, final_row.norm_tdvp, init_err, nan_warnings
        )
        push!(summaries, line)

        @printf("D=%d done: runtime=%.3fs, final err_state=%.3e, final max|ΔZ|=%.3e\n",
                D, tdvp_runtime, final_row.err_state, final_row.max_abs_err_z)
    end

    summary_path = joinpath(outdir, "summary.txt")
    open(summary_path, "w") do io
        for line in summaries
            write(io, line * "\n")
        end
    end

    println("Results written to: ", outdir)
    println("Summary file: ", summary_path)
end

main()
