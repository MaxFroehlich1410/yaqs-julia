### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 3e70a0f1-6172-4a0b-a93f-3e56e0c61b83
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
    using .Yaqs.CircuitTJM: DigitalCircuit, RepeatedDigitalCircuit, add_gate!, DigitalGate
end

# ╔═╡ 51a7ef20-5f80-4b84-93b1-bd6bd0786056
md"""
# How to run this Pluto notebook

1. Install Pluto (once):  
   `using Pkg; Pkg.add("Pluto")`
2. Start Pluto:  
   `using Pluto; Pluto.run()`
3. Open this file in the Pluto UI.

This notebook mirrors `01_PaperExps/ibm127_exp.jl` but focuses only on the IBM 127-qubit kicked-Ising case.
"""

# ╔═╡ 1dcaf8f6-83d0-4f1d-9c16-d2da2e5daab4
md"""
# IBM 127-qubit kicked-Ising experiment (walkthrough)

This notebook explains the IBM 127-qubit kicked-Ising experiment step by step. We fix the system size to 127 qubits and only expose:

- **Hamiltonian parameters** (`IBM_THETA_H`, `IBM_THETA_J`)
- **Trotter step parameters** (`NUM_LAYERS`, `TAU`)
- **Noise model** (Pauli axes + strength)
- **Unraveling method** (`jump`, `unitary_2pt`, `unitary_gauss`, `projector`)

Everything else is kept at the script defaults so the notebook stays close to `ibm127_exp.jl`.
"""

# ╔═╡ 0fb97132-6f22-4d39-9ee5-f26f68e0d0c1
md"""
## Setup

We use the repo environment and local `Yaqs` source. PythonCall is needed to read the IBM Kyiv backend connectivity from a local Qiskit wheel.
"""

# ╔═╡ b49df78f-1e49-4f03-855f-a4f1ee0ba937
md"""
## Configuration (the only knobs)

Change the parameters below to explore the kicked-Ising dynamics. The qubit count is fixed to 127.
"""

# ╔═╡ 4e3949ad-0c79-40b8-9f0e-5081741b3c06
begin
    # Fixed system size
    NUM_QUBITS = 127

    # Hamiltonian parameters (kicked Ising)
    IBM_THETA_H = π / 4   # RX kick (clifford for any k * π / 2)
    IBM_THETA_J = -π / 2  # RZZ coupling

    # Trotter step parameters
    NUM_LAYERS = 5
    TAU = 0.1

    # Noise model
    NOISE_STRENGTH = 0.01
    ENABLE_X_ERROR = true
    ENABLE_Y_ERROR = true
    ENABLE_Z_ERROR = true

    # Unraveling method (choose one)
    # Options: "jump", "unitary_2pt", "unitary_gauss", "projector"
    UNRAVELING = "jump"
    UNRAVELING_THETA0 = π / 3
    UNRAVELING_SIGMA = 1.0

    # Plotting (local Z observables)
    SITES_TO_PLOT = [1, 32, 64, 96, 127]
end

# ╔═╡ 3e4f57e2-6c62-4391-b99b-258b2309bc6a
begin
    using PythonCall
    pybuiltins = pyimport("builtins")

    const KYIV_WHEEL_PATH = joinpath(@__DIR__, "qiskit_ibm_runtime-0.43.1-py3-none-any.whl")
    const KYIV_CONF_PATH_IN_WHEEL = "qiskit_ibm_runtime/fake_provider/backends/kyiv/conf_kyiv.json"

    function _kyiv_load_conf_from_wheel(; wheel_path::AbstractString=KYIV_WHEEL_PATH,
                                        conf_path_in_wheel::AbstractString=KYIV_CONF_PATH_IN_WHEEL)
        if !isfile(wheel_path)
            error("Missing local wheel file: $wheel_path\n" *
                  "Expected it next to this notebook (`01_PaperExps/`).")
        end
        zipfile = pyimport("zipfile")
        json = pyimport("json")
        z = zipfile.ZipFile(wheel_path)
        raw = z.read(conf_path_in_wheel)
        return json.loads(raw)
    end

    """
    Load IBM Kyiv connectivity and edge-color it into disjoint layers.
    Returns (n_qubits, layers0) where indices are 0-based.
    """
    function kyiv_edge_colored_layers_from_wheel(; wheel_path::AbstractString=KYIV_WHEEL_PATH)
        conf = _kyiv_load_conf_from_wheel(; wheel_path=wheel_path)
        n = pyconvert(Int, conf["n_qubits"])

        rx = pyimport("rustworkx")
        g = rx.PyGraph()
        g.add_nodes_from(pybuiltins.list(pybuiltins.range(n)))
        coupling = conf["coupling_map"]
        for e in coupling
            i0 = pyconvert(Int, e[0])
            j0 = pyconvert(Int, e[1])
            g.add_edge(i0, j0, nothing)
        end

        edge_coloring = rx.graph_bipartite_edge_color(g)
        layer_map = Dict{Int, Vector{Tuple{Int, Int}}}()
        for item in edge_coloring.items()
            edge_idx = pyconvert(Int, item[0])
            color = pyconvert(Int, item[1])
            endpoints = g.get_edge_endpoints_by_index(edge_idx)
            u = pyconvert(Int, endpoints[0])
            v = pyconvert(Int, endpoints[1])
            a, b = min(u, v), max(u, v)
            push!(get!(layer_map, color, Tuple{Int, Int}[]), (a, b))
        end

        colors = sort!(collect(keys(layer_map)))
        layers0 = Vector{Vector{Tuple{Int, Int}}}(undef, length(colors))
        for (k, c) in enumerate(colors)
            layer = layer_map[c]
            sort!(layer)
            layers0[k] = layer
        end

        return n, layers0
    end

    n_backend, ibm_layers0 = kyiv_edge_colored_layers_from_wheel()
    if n_backend != NUM_QUBITS
        error("Kyiv conf n_qubits ($n_backend) != NUM_QUBITS ($NUM_QUBITS).")
    end

    ibm_edges0 = Tuple{Int, Int}[]
    for layer in ibm_layers0
        append!(ibm_edges0, layer)
    end
    ibm_edges1 = [(i0 + 1, j0 + 1) for (i0, j0) in ibm_edges0]

    println("Kyiv layers: $(length(ibm_layers0))")
    println("Kyiv edges:  $(length(ibm_edges0))")
end

# ╔═╡ 0d435dc8-91f4-4131-a0e4-6fb8f6475e6c
md"""
## Step 1: Load IBM Kyiv connectivity

The kicked-Ising experiment uses the IBM Kyiv heavy-hex topology. We load the backend configuration from
`qiskit_ibm_runtime-0.43.1-py3-none-any.whl` and build edge-disjoint layers for the RZZ gates.
"""

# ╔═╡ b85f4f6d-05cc-49d2-a28e-5adf5d6d0d6c
md"""
## Step 2: Build the kicked-Ising trotter step

Each trotter step applies:
1. `RX(IBM_THETA_H)` on all 127 qubits
2. `RZZ(IBM_THETA_J)` on each heavy-hex edge, in 3 disjoint layers
3. A barrier to mark sampling boundaries

We build **one** step and repeat it `NUM_LAYERS` times using `RepeatedDigitalCircuit`.
"""

# ╔═╡ 0460c54f-a89f-4a8c-8f91-2b2ad9a27b9d
begin
    IBM_ADD_BARRIERS = true

    function build_kicked_ising_step(layers0)
        step = DigitalCircuit(NUM_QUBITS)
        add_gate!(step, Barrier("SAMPLE_OBSERVABLES"), Int[])
        for q in 1:NUM_QUBITS
            add_gate!(step, RxGate(IBM_THETA_H), [q])
        end
        for layer in layers0
            for (i0, j0) in layer
                add_gate!(step, RzzGate(IBM_THETA_J), [i0 + 1, j0 + 1])
            end
        end
        if IBM_ADD_BARRIERS
            add_gate!(step, Barrier("SAMPLE_OBSERVABLES"), Int[])
        end
        return step
    end

    step = build_kicked_ising_step(ibm_layers0)
    circ_jl = RepeatedDigitalCircuit(step, NUM_LAYERS)

    println("Kicked-Ising circuit built.")
end

# ╔═╡ eebeecea-63f0-4d7d-9fd5-f4db1ba70fe2
md"""
## Step 3: Build the noise model

The noise model mirrors `ibm127_exp.jl`:
- single-qubit Pauli noise on each qubit
- two-qubit crosstalk on each heavy-hex edge

You can toggle X/Y/Z and choose the unraveling method.
"""

# ╔═╡ 0d6a4596-209f-4cbd-bbf8-225ed417f4cd
begin
    function build_noise_processes()
        processes = Vector{Dict{String, Any}}()

        function add_axis(axis_name::String, crosstalk_name::String)
            for i in 1:NUM_QUBITS
                push!(processes, Dict("name" => axis_name, "sites" => [i], "strength" => NOISE_STRENGTH))
            end
            for (i, j) in ibm_edges1
                push!(processes, Dict("name" => crosstalk_name, "sites" => [i, j], "strength" => NOISE_STRENGTH))
            end
        end

        if ENABLE_X_ERROR
            add_axis("pauli_x", "crosstalk_xx")
        end
        if ENABLE_Y_ERROR
            add_axis("pauli_y", "crosstalk_yy")
        end
        if ENABLE_Z_ERROR
            add_axis("pauli_z", "crosstalk_zz")
        end

        return processes
    end

    function build_noise_model(processes::Vector{Dict{String, Any}})
        procs = [copy(d) for d in processes]
        if UNRAVELING == "jump"
            return NoiseModel(procs, NUM_QUBITS)
        elseif UNRAVELING == "unitary_2pt"
            for d in procs
                d["unraveling"] = "unitary_2pt"
                d["theta0"] = UNRAVELING_THETA0
            end
            return NoiseModel(procs, NUM_QUBITS)
        elseif UNRAVELING == "unitary_gauss"
            for d in procs
                d["unraveling"] = "unitary_gauss"
            end
            return NoiseModel(procs, NUM_QUBITS; sigma=UNRAVELING_SIGMA)
        elseif UNRAVELING == "projector"
            for d in procs
                d["unraveling"] = "projector"
            end
            return NoiseModel(procs, NUM_QUBITS)
        else
            error("Unknown UNRAVELING: $UNRAVELING")
        end
    end

    noise_processes = build_noise_processes()
    noise_model = build_noise_model(noise_processes)

    println("Noise processes: $(length(noise_model.processes))")
end

# ╔═╡ 83af4f8f-9b1c-410b-8d15-067a6a1ce9a7
md"""
## Step 4: Run a Julia trajectory

This mirrors the `runner_julia` logic in `ibm127_exp.jl` but keeps it minimal for the walkthrough.
"""

# ╔═╡ b11c2231-7df6-40f1-8081-4f8696cd4daa
begin
    MAX_BOND_DIM = 128
    SVD_TRUNCATION_THRESHOLD = 1e-16
    NUM_TRAJECTORIES = 1
    local_mode = "TDVP"
    longrange_mode = "TDVP"

    function staggered_magnetization(expvals::Vector{Float64}, L::Int)
        sum_val = 0.0
        for i in 1:L
            sum_val += (-1)^(i - 1) * expvals[i]
        end
        return sum_val / L
    end

    function run_single_trajectory(circ::RepeatedDigitalCircuit, noise_model::NoiseModel)
        psi = MPS(NUM_QUBITS; state="zeros")
        obs = [Observable("Z_$i", ZGate(), i) for i in 1:NUM_QUBITS]
        evolution_options = Yaqs.CircuitTJM.TJMOptions(
            local_method=Symbol(local_mode),
            long_range_method=Symbol(longrange_mode)
        )
        sim_params = TimeEvolutionConfig(
            obs,
            Float64(NUM_LAYERS);
            dt=1.0,
            num_traj=1,
            max_bond_dim=MAX_BOND_DIM,
            truncation_threshold=SVD_TRUNCATION_THRESHOLD
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

    function run_trajectories(circ::RepeatedDigitalCircuit, noise_model::NoiseModel)
        cumulative = nothing
        cumulative_bonds = nothing
        t_start = time()
        for n in 1:NUM_TRAJECTORIES
            res_mat, bond_dims = run_single_trajectory(circ, noise_model)
            if isnothing(cumulative)
                cumulative = copy(res_mat)
                if !isnothing(bond_dims)
                    cumulative_bonds = Float64.(bond_dims)
                end
            else
                cumulative .+= res_mat
                if !isnothing(bond_dims) && !isnothing(cumulative_bonds)
                    cumulative_bonds .+= Float64.(bond_dims)
                end
            end
            println("  Traj $n / $NUM_TRAJECTORIES")
        end
        t_total = time() - t_start

        avg_res = cumulative ./ NUM_TRAJECTORIES
        avg_bonds = isnothing(cumulative_bonds) ? nothing : (cumulative_bonds ./ NUM_TRAJECTORIES)
        return avg_res, avg_bonds, t_total
    end
end

# ╔═╡ 563c8f28-5381-4e32-bf99-2b2e89c7dddf
md"""
## Step 5: Execute and inspect

This cell runs the chosen configuration. For a first test, keep `NUM_LAYERS` small.
"""

# ╔═╡ c50d2a7c-55d2-4d54-93f4-70a3e2a14970
begin
    println("Running IBM127 kicked-Ising")
    println("  Layers:  $NUM_LAYERS")
    println("  TAU:     $TAU")
    println("  Noise:   $NOISE_STRENGTH")
    println("  Unravel: $UNRAVELING")

    avg_res, avg_bonds, t_total = run_trajectories(circ_jl, noise_model)
    println(@sprintf("Done in %.2f s", t_total))

    T_steps = size(avg_res, 2)
    times = (0:(T_steps - 1)) .* TAU
    stag_series = [staggered_magnetization(avg_res[:, t], NUM_QUBITS) for t in 1:T_steps]

    (avg_res=avg_res, avg_bonds=avg_bonds, times=times, staggered=stag_series)
end

# ╔═╡ 9e90dd9d-5ef1-4a35-9c11-6ea131c555bb
md"""
## Step 6: Plot local ⟨Z⟩ observables

This plots the local Z expectation values for the qubits listed in `SITES_TO_PLOT`.
"""


# ╔═╡ 6d3fe71a-5d7e-4b32-bc19-5a3fe60e08e4
begin
    local_sites = [s for s in SITES_TO_PLOT if 1 <= s <= NUM_QUBITS]
    if isempty(local_sites)
        error("SITES_TO_PLOT has no valid sites for NUM_QUBITS=$(NUM_QUBITS).")
    end

    T_steps_plot = size(avg_res, 2)
    times_plot = (0:(T_steps_plot - 1)) .* TAU

    p = plot(
        title="IBM127 kicked-Ising: local ⟨Z⟩",
        xlabel="Time (t)",
        ylabel="⟨Z⟩",
        grid=true,
        legend=:topright,
    )
    for s in local_sites
        plot!(p, times_plot, avg_res[s, 1:T_steps_plot]; label="site $s", linewidth=2)
    end

    p
end

# ╔═╡ Cell order:
# ╟─51a7ef20-5f80-4b84-93b1-bd6bd0786056
# ╟─1dcaf8f6-83d0-4f1d-9c16-d2da2e5daab4
# ╟─0fb97132-6f22-4d39-9ee5-f26f68e0d0c1
# ╠═3e70a0f1-6172-4a0b-a93f-3e56e0c61b83
# ╟─b49df78f-1e49-4f03-855f-a4f1ee0ba937
# ╠═4e3949ad-0c79-40b8-9f0e-5081741b3c06
# ╟─0d435dc8-91f4-4131-a0e4-6fb8f6475e6c
# ╠═3e4f57e2-6c62-4391-b99b-258b2309bc6a
# ╠═b85f4f6d-05cc-49d2-a28e-5adf5d6d0d6c
# ╠═0460c54f-a89f-4a8c-8f91-2b2ad9a27b9d
# ╠═eebeecea-63f0-4d7d-9fd5-f4db1ba70fe2
# ╠═0d6a4596-209f-4cbd-bbf8-225ed417f4cd
# ╠═83af4f8f-9b1c-410b-8d15-067a6a1ce9a7
# ╠═b11c2231-7df6-40f1-8081-4f8696cd4daa
# ╠═563c8f28-5381-4e32-bf99-2b2e89c7dddf
# ╠═c50d2a7c-55d2-4d54-93f4-70a3e2a14970
# ╠═9e90dd9d-5ef1-4a35-9c11-6ea131c555bb
# ╠═6d3fe71a-5d7e-4b32-bc19-5a3fe60e08e4
