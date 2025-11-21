# experiments/benchmark_julia.jl

if !isdefined(Main, :Yaqs)
    include("../src/Yaqs.jl")
end

using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.NoiseModule
using .Yaqs.SimulationConfigs
using .Yaqs.Simulator
using .Yaqs.GateLibrary
using LinearAlgebra
using Base.Threads
using CSV
using DataFrames

function run_benchmark()
    # CRITICAL: Disable BLAS threading
    LinearAlgebra.BLAS.set_num_threads(1)

    # 1. Parameters
    L = 12
    J = 1.0
    h = 0.5
    dt = 0.05
    t_total = 1.0
    num_traj = 500
    strength = 0.01

    println("Starting Julia Analog TJM Benchmark (L=6)...")
    println("L=$L, J=$J, h=$h, dt=$dt, T=$t_total, Traj=$num_traj, Noise=Lowering($strength)")
    println("Threads: $(Threads.nthreads()) (BLAS threads set to 1)")

    # 2. Initialize Objects
    initial_state = MPS(L, state="zeros")
    H = init_ising(L, J, h)
    processes = [Dict("name" => "lowering", "sites" => [i], "strength" => strength) for i in 1:L]
    noise_model = NoiseModel(processes, L)
    
    # Observables: First, Middle, Last (1-based indexing)
    # L=6 -> 1, 3, 6
    mid = L ÷ 2
    sites_to_measure = [1, mid, L] 
    labels = ["Z_First", "Z_Middle", "Z_Last"]
    observables = [Observable(labels[i], ZGate(), site) for (i, site) in enumerate(sites_to_measure)]

    # 3. Warmup
    println("\n[Warmup] Running 1 trajectory...")
    warmup_params = TimeEvolutionConfig(observables, dt; dt=dt, num_traj=1, max_bond_dim=32, truncation_threshold=1e-9, order=2, sample_timesteps=true)
    Simulator.run(initial_state, H, warmup_params, noise_model; parallel=false)
    println("[Warmup] Done.")

    # 4. Actual Benchmark
    sim_params = TimeEvolutionConfig(observables, t_total;
                                     dt=dt,
                                     num_traj=num_traj,
                                     max_bond_dim=32,
                                     truncation_threshold=1e-9,
                                     order=2,
                                     sample_timesteps=true)
    
    println("\n[Benchmark] Running full simulation...")
    
    elapsed = @elapsed Simulator.run(initial_state, H, sim_params, noise_model; parallel=true)
    
    println("\nJulia Simulation Finished.")
    println("Elapsed Time: $(round(elapsed, digits=4)) seconds")

    # 5. Save Results
    times = sim_params.times
    results_dict = Dict{String, Vector{Float64}}()
    results_dict["Time"] = times

    for obs in sim_params.observables
        results_dict[obs.name] = obs.results
    end

    df = DataFrame(results_dict)
    output_file = "experiments/julia_results_L6.csv"
    CSV.write(output_file, df)
    println("Results saved to $output_file")
end

run_benchmark()
