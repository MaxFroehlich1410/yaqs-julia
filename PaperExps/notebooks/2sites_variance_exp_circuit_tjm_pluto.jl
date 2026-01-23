### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ f4d4c42f-94c4-49f6-9c8f-7c3e3a55b21b
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

    # Toggle to avoid slow instantiation during interactive work
    const DO_INSTANTIATE = false
    if DO_INSTANTIATE
        Pkg.instantiate()
    end

    using Random
    using Statistics
    using Printf
    using Plots

    include("../src/Yaqs.jl")
    using .Yaqs
    using .Yaqs.GateLibrary
    using .Yaqs.MPSModule
    using .Yaqs.NoiseModule
    using .Yaqs.SimulationConfigs
    using .Yaqs.CircuitTJM
    using .Yaqs.Simulator
end

# ╔═╡ 2b1d5f14-9f18-4c5a-8b2b-2a1d74c0c3c5
md"""
# How to run this Pluto notebook

1. Install Pluto (once):  
   `using Pkg; Pkg.add("Pluto")`
2. Start Pluto:  
   `using Pluto; Pluto.run()`
3. Open this file in the Pluto UI.

If you see an error about `Plots` being missing, run:
`using Pkg; Pkg.add("Plots")`
"""

# ╔═╡ 5f3a4a2f-3d0f-4bb3-97f1-9b0b7d6e56e8
md"""
# 2-qubit variance experiment (CircuitTJM)

This notebook mirrors `01_PaperExps/2sites_variance_exp_digitaltjm.jl` and compares
simulated trajectory variance of ⟨Z₁⟩ and ⟨Z₂⟩ against closed-form theory for
different unravelings:

- standard
- projector
- unitary_2pt (two-point law)
- unitary_gauss (Gaussian law)
"""

# ╔═╡ 1b8b7778-0fa4-4f1e-b9de-979e2a1cb3da
md"""
## Setup

This notebook uses the local `Yaqs` source and the repo's `Project.toml`.
"""

# ╔═╡ 4f6738c5-39d5-47d2-bf9a-7a2b40a27f7f
md"""
## Configuration

Adjust these settings to trade accuracy vs. runtime.
"""

# ╔═╡ 7bdbd9cc-79b0-4565-8c61-f3a8cc3d7e36
begin
    L = 2
    NUM_LAYERS = 100
    NUM_TRAJ = 1000
    DT = 1.0
    SUBSTEPS = 1
    GAMMAS = [0.1, 0.01, 0.001]
end

# ╔═╡ 3a9d1c2f-9f8e-4c41-a434-8b8a97f8f2db
md"""
## Theory
"""

# ╔═╡ 2d8c7f32-2a4b-4ae1-9c77-e7c0f2a4b145
begin
    const S_TARGET = 1.0 / 3.0
    const THETA0_2PT = asin(sqrt(S_TARGET))

    function theoretical_variances(t::AbstractVector{<:Real}, γ::Real; s::Float64 = 1 / 3)
        tt = Float64.(t)
        gamma = Float64(γ)

        mean_sq = @. exp(-8.0 * gamma * tt)
        var_std = @. 1.0 - exp(-8.0 * gamma * tt)
        var_proj = @. exp(-4.0 * gamma * tt) * (1.0 - exp(-4.0 * gamma * tt))
        var_2pt = @. 0.25 + 0.5 * exp(-8.0 * gamma * (1.0 - s) * tt) +
            0.25 * exp(-16.0 * gamma * (1.0 - s) * tt) - mean_sq

        one_minus_2s = 1.0 - 2.0 * s
        b = (gamma / max(s, 1e-16)) * (1.0 - one_minus_2s^4) / 2.0
        var_gauss = @. 0.25 + 0.5 * exp(-2.0 * b * tt) + 0.25 * exp(-4.0 * b * tt) - mean_sq

        return Dict(
            "standard" => var_std,
            "projector" => var_proj,
            "unitary_2pt" => var_2pt,
            "unitary_gauss" => var_gauss,
        )
    end

    mean_Z(t::AbstractVector{<:Real}, γ::Real) = @. exp(-4.0 * Float64(γ) * Float64.(t))

    @inline function _gauss_s_weight(σ::Float64; M::Int = 11, k::Float64 = 4.0)
        theta_max = k * σ
        npos = (M + 1) ÷ 2
        thetas_pos = range(0.0, theta_max; length = npos)
        thetas = vcat(-reverse(collect(thetas_pos[2:end])), collect(thetas_pos))

        w = exp.(-0.5 .* (thetas ./ σ) .^ 2)
        w ./= sum(w)
        w = 0.5 .* (w .+ reverse(w))

        return sum(w .* (sin.(thetas) .^ 2))
    end

    function sigma_for_s_target(; s_target::Float64 = S_TARGET, M::Int = 11, k::Float64 = 4.0)
        σ_lo = 1e-4
        σ_hi = 0.1
        s_hi = _gauss_s_weight(σ_hi; M=M, k=k)
        while s_hi < s_target
            σ_hi *= 2
            s_hi = _gauss_s_weight(σ_hi; M=M, k=k)
            if σ_hi > 50
                error("Failed to bracket sigma for s_target=$(s_target).")
            end
        end
        for _ in 1:80
            σ_mid = 0.5 * (σ_lo + σ_hi)
            s_mid = _gauss_s_weight(σ_mid; M=M, k=k)
            if s_mid < s_target
                σ_lo = σ_mid
            else
                σ_hi = σ_mid
            end
            if abs(s_mid - s_target) < 1e-6
                return σ_mid
            end
        end
        return 0.5 * (σ_lo + σ_hi)
    end
end

# ╔═╡ 621c6ad3-8ee7-4ef2-b5df-8b4b3f8dc7e6
md"""
## Circuit + noise helpers
"""

# ╔═╡ 0a21518f-5e93-41e6-9c61-f2fa4da5fd4f
begin
    function build_identity_step_circuit(L::Int, substeps::Int)
        @assert substeps >= 1
        step = DigitalCircuit(L)
        add_gate!(step, Barrier("SAMPLE_OBSERVABLES"), Int[])
        for _ in 1:substeps
            add_gate!(step, RxxGate(0.0), [1, 2])
        end
        add_gate!(step, Barrier("SAMPLE_OBSERVABLES"), Int[])
        return step
    end

    function base_processes(gamma::Float64, substeps::Int)
        γeff = gamma / substeps
        return [
            Dict{String, Any}("name" => "pauli_x", "sites" => [1], "strength" => γeff),
            Dict{String, Any}("name" => "crosstalk_xx", "sites" => [1, 2], "strength" => γeff),
            Dict{String, Any}("name" => "pauli_x", "sites" => [2], "strength" => γeff),
        ]
    end

    function processes_with_unraveling(
        gamma::Float64,
        substeps::Int,
        unraveling::String;
        theta0::Float64 = THETA0_2PT,
        sigma::Float64 = 1.0,
        M::Int = 11,
        k::Float64 = 4.0,
    )
        procs = base_processes(gamma, substeps)
        for p in procs
            p["unraveling"] = unraveling
            if unraveling == "unitary_2pt"
                p["theta0"] = theta0
            elseif unraveling == "unitary_gauss"
                p["sigma"] = sigma
                p["M"] = M
                p["gauss_k"] = k
            end
        end
        return procs
    end
end

# ╔═╡ 76241b3a-b83b-4be1-9322-7f39b2e6b74b
md"""
## Simulation
"""

# ╔═╡ 1f8e4b15-3bc0-4e07-92ce-0bd7a672ad5a
begin
    function run_one_method(;
        L::Int,
        num_layers::Int,
        num_traj::Int,
        dt::Float64,
        gamma::Float64,
        method::String,
        theta0_2pt::Float64,
        sigma_gauss::Float64,
        gauss_M::Int,
        gauss_k::Float64,
        seed::Int,
        substeps::Int,
    )
        Random.seed!(seed)
        ψ0 = MPS(L; state="zeros")

        obs = Observable[
            Observable("Z1", ZGate(), 1),
            Observable("Z2", ZGate(), 2),
        ]

        total_time = num_layers * dt
        sim = TimeEvolutionConfig(
            obs,
            total_time;
            dt=dt,
            num_traj=num_traj,
            sample_timesteps=true,
            max_bond_dim=64,
            truncation_threshold=1e-12,
        )

        step = build_identity_step_circuit(L, substeps)
        circ = RepeatedDigitalCircuit(step, num_layers)

        if method == "standard"
            procs = base_processes(gamma, substeps)
            noise = NoiseModel(procs, L; dt=dt)
        elseif method == "projector"
            procs = processes_with_unraveling(gamma, substeps, "projector")
            noise = NoiseModel(procs, L; dt=dt)
        elseif method == "unitary_2pt"
            procs = processes_with_unraveling(gamma, substeps, "unitary_2pt"; theta0=theta0_2pt)
            noise = NoiseModel(procs, L; theta0=theta0_2pt, dt=dt)
        elseif method == "unitary_gauss"
            procs = processes_with_unraveling(gamma, substeps, "unitary_gauss"; sigma=sigma_gauss, M=gauss_M, k=gauss_k)
            noise = NoiseModel(procs, L; sigma=sigma_gauss, gauss_M=gauss_M, gauss_k=gauss_k, dt=dt)
        else
            error("Unknown method: $method")
        end

        Simulator.run(ψ0, circ, sim, noise; parallel=true)

        Z1_traj = real.(obs[1].trajectories)
        Z2_traj = real.(obs[2].trajectories)
        varZ1 = vec(var(Z1_traj; dims=1, corrected=false))
        varZ2 = vec(var(Z2_traj; dims=1, corrected=false))

        meanZ1 = obs[1].results
        meanZ2 = obs[2].results

        return (meanZ1=meanZ1, meanZ2=meanZ2, varZ1=varZ1, varZ2=varZ2)
    end
end

# ╔═╡ 9b84c2fb-8d0b-4ef6-9d1d-9d19b5b5f05d
md"""
## Run experiment and plot

This cell runs the trajectories, compares to theory, and saves a PNG. The plot also
renders directly in Pluto.
"""

# ╔═╡ 4f8e5c46-c0e3-4c0b-9e8b-dce55cb1b9ef
begin
    t_layers = collect(0.0:DT:(NUM_LAYERS * DT))
    theta0 = THETA0_2PT
    gauss_M = 11
    gauss_k = 4.0
    sigma_gauss = sigma_for_s_target(; s_target=S_TARGET, M=gauss_M, k=gauss_k)

    methods = ["standard", "projector", "unitary_2pt", "unitary_gauss"]
    colors = Dict(
        "standard" => :blue,
        "projector" => :red,
        "unitary_2pt" => :green,
        "unitary_gauss" => :orange,
        "theory_mean" => :black,
    )

    theo_list = Vector{Dict{String, Vector{Float64}}}(undef, length(GAMMAS))
    sim_list = Vector{Dict{String, NamedTuple}}(undef, length(GAMMAS))

    for (idx, γ) in enumerate(GAMMAS)
        theo_list[idx] = theoretical_variances(t_layers, γ; s=S_TARGET)
        sim_for_gamma = Dict{String, NamedTuple}()
        for (mi, method) in enumerate(methods)
            seed = 12345 + 1000 * idx + 10 * mi
            sim_for_gamma[method] = run_one_method(
                ; L=L,
                num_layers=NUM_LAYERS,
                num_traj=NUM_TRAJ,
                dt=DT,
                gamma=γ,
                method=method,
                theta0_2pt=theta0,
                sigma_gauss=sigma_gauss,
                gauss_M=gauss_M,
                gauss_k=gauss_k,
                seed=seed,
                substeps=SUBSTEPS,
            )
        end
        sim_list[idx] = sim_for_gamma
    end

    plot_panels = Any[]
    for (j, γ) in enumerate(GAMMAS)
        theo = theo_list[j]
        sim = sim_list[j]

        p_var = plot(
            title=@sprintf("γ=%.3g (variance)", γ),
            xlabel="Physical Time (t)",
            ylabel="Variance",
            ylim=(0, 1.1),
            grid=true,
            legend=(j == 1 ? :topright : :none),
        )
        for m in methods
            plot!(p_var, t_layers, theo[m]; linestyle=:dash, color=colors[m], linewidth=2, label="$(m) (theory)")
            plot!(p_var, t_layers, sim[m][:varZ1]; linestyle=:solid, color=colors[m], linewidth=1.5, alpha=0.85, label="$(m) (sim)")
        end
        push!(plot_panels, p_var)

        p_exp = plot(
            title=@sprintf("γ=%.3g (mean)", γ),
            xlabel="Physical Time (t)",
            ylabel="⟨Z₁⟩",
            ylim=(0, 1.1),
            grid=true,
            legend=(j == 1 ? :topright : :none),
        )
        plot!(p_exp, t_layers, mean_Z(t_layers, γ); linestyle=:dash, color=colors["theory_mean"], linewidth=2, label="theory")
        for m in methods
            plot!(p_exp, t_layers, sim[m][:meanZ1]; linestyle=:solid, color=colors[m], linewidth=1.5, alpha=0.85, label=m)
        end
        push!(plot_panels, p_exp)
    end

    final_plot = plot(plot_panels...; layout=(2, length(GAMMAS)), size=(1200, 600))

    png_path = joinpath(@__DIR__, "variance_comparison_digitaltjm_pluto.png")
    savefig(final_plot, png_path)
    println("Saved plot to: $png_path")

    display(final_plot)
    final_plot
end

# ╔═╡ Cell order:
# ╟─2b1d5f14-9f18-4c5a-8b2b-2a1d74c0c3c5
# ╟─5f3a4a2f-3d0f-4bb3-97f1-9b0b7d6e56e8
# ╟─1b8b7778-0fa4-4f1e-b9de-979e2a1cb3da
# ╠═f4d4c42f-94c4-49f6-9c8f-7c3e3a55b21b
# ╟─4f6738c5-39d5-47d2-bf9a-7a2b40a27f7f
# ╠═7bdbd9cc-79b0-4565-8c61-f3a8cc3d7e36
# ╟─3a9d1c2f-9f8e-4c41-a434-8b8a97f8f2db
# ╠═2d8c7f32-2a4b-4ae1-9c77-e7c0f2a4b145
# ╟─621c6ad3-8ee7-4ef2-b5df-8b4b3f8dc7e6
# ╠═0a21518f-5e93-41e6-9c61-f2fa4da5fd4f
# ╟─76241b3a-b83b-4be1-9322-7f39b2e6b74b
# ╠═1f8e4b15-3bc0-4e07-92ce-0bd7a672ad5a
# ╟─9b84c2fb-8d0b-4ef6-9d1d-9d19b5b5f05d
# ╠═4f8e5c46-c0e3-4c0b-9e8b-dce55cb1b9ef
