### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 8b8c987b-9bfa-4fa1-97f5-7fdcfa26f1c9
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

    # Toggle to avoid slow instantiation during interactive work
    const DO_INSTANTIATE = false
    if DO_INSTANTIATE
        Pkg.instantiate()
    end

    using LinearAlgebra
    using Random
    using Statistics
    using Printf
    using Plots

    # Include Yaqs source
    include("../src/Yaqs.jl")
    using .Yaqs
    using .Yaqs.MPSModule
    using .Yaqs.MPOModule
    using .Yaqs.GateLibrary
    using .Yaqs.NoiseModule
    using .Yaqs.SimulationConfigs
    using .Yaqs.CircuitLibrary
    using .Yaqs.DigitalTJM: DigitalCircuit, add_gate!, DigitalGate
end

# ╔═╡ 0f7b2b6b-6c3c-4dc3-bfc7-1a6a7c4d2e42
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

# ╔═╡ 6a98c6d0-2f3b-4e23-8f2f-1a7f49b09c6a
md"""
# 25-site XY quench experiment (walkthrough)

This notebook walks through the 25-site, 20-layer XY quench experiment from `01_PaperExps/25q_exp.jl` and shows how to customize the circuit, qubit count, noise axes, and other parameters. By the end, you will have generated a plot of local observables and bond dimensions.

**What you will do**
- Configure the XY quench (or switch to Ising/Heisenberg/QAOA/HEA).
- Choose X/Y/Z noise (single-qubit + nearest-neighbor crosstalk + long-range crosstalk).
- Run trajectories with the Julia simulator.
- Plot local observables, variance, and bond growth.
"""

# ╔═╡ 0d40a0b1-4a20-4ef5-9f1f-81847a0c32d0
md"""
## Setup

This notebook uses the local `Yaqs` source and the repo's `Project.toml`.
"""

# ╔═╡ 52ef43ed-1e7c-45bd-8a62-85857e12c843
md"""
## Configuration

These defaults reproduce the 25-site, 20-layer XY quench with selectable X/Y/Z noise. You can change the circuit family, qubit count, layers, noise strength, or trajectories here.
"""

# ╔═╡ 0c3e46b7-7194-4f3d-8d6a-91bc3c2e204f
begin
    # Circuit selection
    # Options: "Ising", "Ising_periodic", "Heisenberg", "Heisenberg_periodic",
    #          "XY", "XY_longrange", "QAOA", "HEA", "longrange_test"
    CIRCUIT_NAME = "XY"

    # System size and time step
    NUM_QUBITS = 25
    NUM_LAYERS = 20
    TAU = 0.1

    dt = TAU  # alias used in some circuit constructors

    # Noise selection (choose any subset of X/Y/Z)
    NOISE_STRENGTH = 0.01
    ENABLE_X_ERROR = true
    ENABLE_Y_ERROR = false
    ENABLE_Z_ERROR = false

    # Trajectories and truncation
    NUM_TRAJECTORIES = 200
    MAX_BOND_DIM = 32

    # Time evolution methods
    longrange_mode = "TDVP" # "TEBD" or "TDVP"
    local_mode = "TDVP"     # "TEBD" or "TDVP"

    # Plotting
    OBSERVABLE_BASIS = "Z"
    SITES_TO_PLOT = [1, 2, 3, 4, 5, 6]

    # Model parameters
    J = 1.0
    g = 1.0
    Jx, Jy, Jz = 1.0, 1.0, 1.0
    h_field = 0.0

    beta_qaoa = 0.3
    gamma_qaoa = 0.5

    phi_hea = 0.2
    theta_hea = 0.4
    lam_hea = 0.6
    start_parity_hea = 0

    longrange_theta = π / 4
end

# ╔═╡ b68b2b0d-064e-4d6c-b861-0a3f91f0c5f0
md"""
## Circuit construction

The circuit is built layer-by-layer. You can replace the `custom_circuit_builder` with your own function that returns a `DigitalCircuit` if you want complete control of gates.
"""

# ╔═╡ 8f8793b1-ec96-4b8a-8f37-75bfa3f4b6c6
begin
    function resolve_model(circuit_name::String)
        periodic = false
        long_range_gates = false
        base_model = ""

        if startswith(circuit_name, "Ising")
            base_model = "Ising"
            if occursin("periodic", circuit_name)
                periodic = true
                long_range_gates = true
            end
        elseif startswith(circuit_name, "Heisenberg")
            base_model = "Heisenberg"
            if occursin("periodic", circuit_name)
                periodic = true
                long_range_gates = true
            end
        elseif startswith(circuit_name, "XY")
            base_model = "XY"
            if occursin("longrange", circuit_name) || occursin("periodic", circuit_name)
                long_range_gates = true
            end
        elseif startswith(circuit_name, "QAOA")
            base_model = "QAOA"
        elseif startswith(circuit_name, "HEA")
            base_model = "HEA"
        elseif circuit_name == "longrange_test"
            base_model = "longrange_test"
            long_range_gates = true
        else
            error("Unknown CIRCUIT_NAME: $circuit_name")
        end

        return base_model, periodic, long_range_gates
    end

    function build_circuit(; custom_circuit_builder=nothing)
        if custom_circuit_builder !== nothing
            return custom_circuit_builder()
        end

        base_model, periodic, long_range_gates = resolve_model(CIRCUIT_NAME)
        circ = DigitalCircuit(NUM_QUBITS)
        add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])

        if base_model == "Ising"
            circ = create_ising_circuit(NUM_QUBITS, J, g, dt, NUM_LAYERS, periodic=periodic)
        elseif base_model == "XY"
            for _ in 1:NUM_LAYERS
                layer = long_range_gates ? xy_trotter_layer_longrange(NUM_QUBITS, TAU) :
                    xy_trotter_layer(NUM_QUBITS, TAU)
                for gate in layer.gates
                    add_gate!(circ, gate.op, gate.sites)
                end
                add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])
            end
        elseif base_model == "Heisenberg"
            circ = create_heisenberg_circuit(NUM_QUBITS, Jx, Jy, Jz, h_field, dt, NUM_LAYERS, periodic=periodic)
        elseif base_model == "QAOA"
            for _ in 1:NUM_LAYERS
                layer = qaoa_ising_layer(NUM_QUBITS; beta=beta_qaoa, gamma=gamma_qaoa)
                for gate in layer.gates
                    add_gate!(circ, gate.op, gate.sites)
                end
                add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])
            end
        elseif base_model == "HEA"
            phis = fill(phi_hea, NUM_QUBITS)
            thetas = fill(theta_hea, NUM_QUBITS)
            lams = fill(lam_hea, NUM_QUBITS)
            for _ in 1:NUM_LAYERS
                layer = hea_layer(NUM_QUBITS; phis=phis, thetas=thetas, lams=lams, start_parity=start_parity_hea)
                for gate in layer.gates
                    add_gate!(circ, gate.op, gate.sites)
                end
                add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])
            end
        elseif base_model == "longrange_test"
            for _ in 1:NUM_LAYERS
                for q in 1:NUM_QUBITS
                    add_gate!(circ, HGate(), [q])
                end
                add_gate!(circ, RzzGate(longrange_theta), [NUM_QUBITS, 1])
                add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])
            end
        end

        return circ
    end

    circ_jl = build_circuit()

    println("Circuit: $CIRCUIT_NAME")
    println("Qubits:  $NUM_QUBITS")
    println("Layers:  $NUM_LAYERS")
    println("Gates:   $(length(circ_jl.gates))")
end

# ╔═╡ 8b12d5c7-4699-4d7a-9bb5-8a54d1b8f9f1
md"""
## Noise model

The noise model matches `25q_exp.jl`: single-qubit Pauli noise plus nearest-neighbor crosstalk for each enabled axis. Toggle X/Y/Z above.
"""

# ╔═╡ 63b18bd2-6af5-42c4-9b5d-6d0879b66d93
begin
    function build_noise_model()
        _, _, long_range_gates = resolve_model(CIRCUIT_NAME)
        processes = Vector{Dict{String, Any}}()

        function add_single_and_crosstalk(axis_name::String, crosstalk_name::String)
            for i in 1:NUM_QUBITS
                d = Dict{String, Any}("name" => axis_name, "sites" => [i], "strength" => NOISE_STRENGTH)
                push!(processes, d)
            end
            for i in 1:(NUM_QUBITS - 1)
                d = Dict{String, Any}("name" => crosstalk_name, "sites" => [i, i + 1], "strength" => NOISE_STRENGTH)
                push!(processes, d)
            end
            if long_range_gates && NUM_QUBITS > 1
                d = Dict{String, Any}("name" => crosstalk_name, "sites" => [NUM_QUBITS, 1], "strength" => NOISE_STRENGTH)
                push!(processes, d)
            end
        end

        if ENABLE_X_ERROR
            add_single_and_crosstalk("pauli_x", "crosstalk_xx")
        end
        if ENABLE_Y_ERROR
            add_single_and_crosstalk("pauli_y", "crosstalk_yy")
        end
        if ENABLE_Z_ERROR
            add_single_and_crosstalk("pauli_z", "crosstalk_zz")
        end

        return NoiseModel(processes, NUM_QUBITS)
    end

    noise_model = build_noise_model()
    println("Noise processes: $(length(noise_model.processes))")
end

# ╔═╡ 51cd5a52-1d86-42cd-a2f6-3d1f0b5a10ad
md"""
## Simulation runner

We run multiple stochastic trajectories and average the local observables. This mirrors the `runner_julia` logic in the script, but only uses the Julia simulator for clarity.
"""

# ╔═╡ 6d8a69e4-6f76-43cb-8e5c-7c2945b52b78
begin
    function staggered_magnetization(expvals::Vector{Float64}, L::Int)
        sum_val = 0.0
        for i in 1:L
            sum_val += (-1)^(i - 1) * expvals[i]
        end
        return sum_val / L
    end

    function init_state!(psi)
        for i in 1:NUM_QUBITS
            if (i - 1) % 4 == 3
                Yaqs.DigitalTJM.apply_single_qubit_gate!(psi, DigitalGate(XGate(), [i], nothing))
            end
        end
    end

    function run_single_trajectory(circ::DigitalCircuit, noise_model::NoiseModel)
        psi = MPS(NUM_QUBITS; state="zeros")
        init_state!(psi)

        obs = [Observable("Z_$i", ZGate(), i) for i in 1:NUM_QUBITS]
        evolution_options = Yaqs.DigitalTJM.TJMOptions(
            local_method=Symbol(local_mode),
            long_range_method=Symbol(longrange_mode)
        )
        sim_params = TimeEvolutionConfig(
            obs,
            Float64(NUM_LAYERS);
            dt=1.0,
            num_traj=1,
            max_bond_dim=MAX_BOND_DIM,
            truncation_threshold=1e-6
        )

        bond_dims_traj = Simulator.run(
            psi,
            circ,
            sim_params,
            noise_model;
            parallel=false,
            alg_options=evolution_options
        )

        results = zeros(ComplexF64, length(obs), length(sim_params.times))
        for (i, o) in enumerate(obs)
            results[i, :] = o.trajectories[1, :]
        end

        bond_dims = isnothing(bond_dims_traj) ? nothing : bond_dims_traj[1]
        return real.(results), bond_dims
    end

    function run_trajectories(circ::DigitalCircuit, noise_model::NoiseModel)
        cumulative = nothing
        cumulative_sq = nothing
        cumulative_bonds = nothing

        t_start = time()
        for n in 1:NUM_TRAJECTORIES
            res_mat, bond_dims = run_single_trajectory(circ, noise_model)
            if isnothing(cumulative)
                cumulative = copy(res_mat)
                cumulative_sq = res_mat .^ 2
                if !isnothing(bond_dims)
                    cumulative_bonds = Float64.(bond_dims)
                end
            else
                cumulative .+= res_mat
                cumulative_sq .+= res_mat .^ 2
                if !isnothing(bond_dims) && !isnothing(cumulative_bonds)
                    cumulative_bonds .+= Float64.(bond_dims)
                end
            end
            if n % 10 == 0 || n == NUM_TRAJECTORIES
                println("  Traj $n / $NUM_TRAJECTORIES")
            end
        end
        t_total = time() - t_start

        avg_res = cumulative ./ NUM_TRAJECTORIES
        avg_sq = cumulative_sq ./ NUM_TRAJECTORIES
        var_res = avg_sq .- (avg_res .^ 2)

        avg_bonds = isnothing(cumulative_bonds) ? nothing : (cumulative_bonds ./ NUM_TRAJECTORIES)
        return avg_res, var_res, avg_bonds, t_total
    end
end

# ╔═╡ 3b4d70a4-8f7f-4f1e-b6c4-1c76f2c4b4e2
md"""
## Run experiment and plot

This cell runs the trajectories and plots local observables, variance, and bond dimension growth. If you want a faster test run, reduce `NUM_TRAJECTORIES`, `NUM_QUBITS`, or `NUM_LAYERS` above.
"""

# ╔═╡ 4d0e7f1b-df55-4f18-b0b3-9f3350ebf848
begin
    println("Running: $CIRCUIT_NAME (N=$NUM_QUBITS, layers=$NUM_LAYERS, noise=$NOISE_STRENGTH)")

    avg_res, var_res, avg_bonds, t_total = run_trajectories(circ_jl, noise_model)
    println(@sprintf("Done in %.2f s (%.3f s/traj)", t_total, t_total / NUM_TRAJECTORIES))

    T_steps = size(avg_res, 2)
    stag_series = [staggered_magnetization(avg_res[:, t], NUM_QUBITS) for t in 1:T_steps]

    num_site_plots = length(SITES_TO_PLOT)
    total_plots = num_site_plots + 2
    min_len = size(avg_res, 2)
    x = 0:(min_len - 1)

    plots = Any[]

    for site in SITES_TO_PLOT
        if site <= size(avg_res, 1)
            p = plot(
                x, avg_res[site, 1:min_len];
                label="Julia",
                linewidth=2,
                title="Site $site Evolution",
                ylabel="<Z>",
                legend=:topright,
                grid=true
            )
            push!(plots, p)
        end
    end

    avg_var = isnothing(var_res) ? nothing : vec(mean(var_res, dims=1))
    p_var = plot(
        x, avg_var[1:min_len];
        label="Variance",
        linewidth=2,
        title="Average Local Variance",
        ylabel="Var(<Z>)",
        grid=true
    )
    push!(plots, p_var)

    p_bond = isnothing(avg_bonds) ? plot(title="Average Max Bond Dimension") : plot(
        x, avg_bonds[1:min_len];
        label="Bond dim",
        linewidth=2,
        title="Average Max Bond Dimension",
        ylabel="χ",
        xlabel="Layer",
        grid=true
    )
    push!(plots, p_bond)

    plot(
        plots...;
        layout=(total_plots, 1),
        size=(1000, 300 * total_plots),
        title="$CIRCUIT_NAME (N=$NUM_QUBITS, noise=$NOISE_STRENGTH, layers=$NUM_LAYERS)"
    )
end

# ╔═╡ 8d0a7f20-4f2b-43b2-9a49-f9713aa1713a
md"""
## Custom circuit example

To define your own circuit, set a custom builder that returns a `DigitalCircuit`. For example, this creates a circuit with alternating single-qubit X rotations and nearest-neighbor RZZ gates for a few layers:

```julia
function my_circuit_builder()
    circ = DigitalCircuit(NUM_QUBITS)
    add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])
    for _ in 1:NUM_LAYERS
        for q in 1:NUM_QUBITS
            add_gate!(circ, RXGate(0.2), [q])
        end
        for q in 1:(NUM_QUBITS - 1)
            add_gate!(circ, RzzGate(0.3), [q, q + 1])
        end
        add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end
    return circ
end

circ_jl = build_circuit(custom_circuit_builder=my_circuit_builder)
```
"""

# ╔═╡ Cell order:
# ╟─0f7b2b6b-6c3c-4dc3-bfc7-1a6a7c4d2e42
# ╟─6a98c6d0-2f3b-4e23-8f2f-1a7f49b09c6a
# ╟─0d40a0b1-4a20-4ef5-9f1f-81847a0c32d0
# ╠═8b8c987b-9bfa-4fa1-97f5-7fdcfa26f1c9
# ╟─52ef43ed-1e7c-45bd-8a62-85857e12c843
# ╠═0c3e46b7-7194-4f3d-8d6a-91bc3c2e204f
# ╟─b68b2b0d-064e-4d6c-b861-0a3f91f0c5f0
# ╠═8f8793b1-ec96-4b8a-8f37-75bfa3f4b6c6
# ╟─8b12d5c7-4699-4d7a-9bb5-8a54d1b8f9f1
# ╠═63b18bd2-6af5-42c4-9b5d-6d0879b66d93
# ╟─51cd5a52-1d86-42cd-a2f6-3d1f0b5a10ad
# ╠═6d8a69e4-6f76-43cb-8e5c-7c2945b52b78
# ╟─3b4d70a4-8f7f-4f1e-b6c4-1c76f2c4b4e2
# ╠═4d0e7f1b-df55-4f18-b0b3-9f3350ebf848
# ╟─8d0a7f20-4f2b-43b2-9a49-f9713aa1713a
