using DelimitedFiles
using LinearAlgebra
using Statistics

# Ensure we use the local Yaqs package
include("../src/Yaqs.jl")
using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.NoiseModule
using .Yaqs.SimulationConfigs
using .Yaqs.AnalogTJM
using .Yaqs.GateLibrary

function run_benchmark()
    # Parameters
    L = 6
    J = 1.0
    h = 0.5
    strength = 0.1
    max_bond = 8
    num_traj = 1000
    dt = 0.05
    T_total = 2.0
    
    println("Starting TJM Benchmark...")
    println("L=$L, J=$J, h=$h, gamma=$strength")
    println("Trajectories: $num_traj")
    
    # 1. Initialize State (All Zeros)
    state = MPS(L, state="zeros")
    
    # 2. Initialize Hamiltonian
    # H = -J sum Z Z - h sum X
    H = MPOModule.init_ising(L, J, h)
    
    # 3. Initialize Noise
    # Raising operators on every site
    processes = [Dict("name" => "raising", "sites" => [i], "strength" => strength) for i in 1:L]
    noise_model = NoiseModel(processes, L)
    
    # 4. Observables: Z on site 1, 3, 6
    # Note: ZGate returns sigma_z.
    # We track indices 1, 3, 6.
    obs_sites = [1, 3, 6]
    observables = [Observable("Z_$i", ZGate(), i) for i in obs_sites]
    
    # 5. Simulation Config
    sim_params = TimeEvolutionConfig(
        observables, 
        T_total; 
        dt=dt, 
        num_traj=num_traj, 
        max_bond_dim=max_bond, 
        sample_timesteps=true
    )
    
    # 6. Run Simulation
    # analog_tjm_2 returns matrix (num_obs, num_steps)
    # args: (traj_idx, initial_state, noise_model, sim_params, hamiltonian)
    
    # We need to run multiple trajectories and average them.
    # analog_tjm_2 runs ONE trajectory.
    # We need a loop or parallel execution.
    # The `SimulationConfigs.aggregate_trajectories!` logic usually handles this if we store in `obs.trajectories`.
    # But `analog_tjm_2` returns a simple matrix.
    
    # Let's implement the parallel loop here.
    
    times = sim_params.times
    num_steps = length(times)
    num_obs = length(observables)
    
    # Accumulator for mean
    results_sum = zeros(Float64, num_obs, num_steps)
    
    println("Running $num_traj trajectories...")
    
    # Threaded loop
    Threads.@threads for i in 1:num_traj
        # Run single trajectory
        traj_res = analog_tjm_2((i, state, noise_model, sim_params, H))
        
        # Add to sum (atomic not needed if we reduce later, but for simple script lock or partition)
        # Simplest: Use atomic add or per-thread storage.
        # For simplicity in script: use a lock for adding to sum.
        
        # Actually, let's just allocate a big array?
        # 1000 * 3 * 40 floats is small.
        # But better:
    end
    
    # Re-do with channel or simple reduction
    # Since we are in a script, let's do it properly.
    
    all_results = zeros(Float64, num_obs, num_steps, num_traj)
    
    Threads.@threads for i in 1:num_traj
        traj_res = analog_tjm_2((i, state, noise_model, sim_params, H))
        all_results[:, :, i] = traj_res
    end
    
    # Average
    mean_results = dropdims(mean(all_results, dims=3), dims=3)
    
    # 7. Save Results
    # Format: Time, Z1, Z3, Z6
    data_to_save = hcat(times, mean_results')
    
    filename = "experiments/tjm_results.csv"
    writedlm(filename, data_to_save, ',')
    println("Saved results to $filename")
end

run_benchmark()

