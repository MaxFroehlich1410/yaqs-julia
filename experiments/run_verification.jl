# experiments/run_verification.jl

# Include the local package
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
using DelimitedFiles

# 1. Parameters
L = 6
J = 1.0
h = 0.5
dt = 0.05
t_total = 2.0
num_traj = 500
strength = 0.1

println("Starting Analog TJM Simulation in Julia...")
println("L=$L, J=$J, h=$h, dt=$dt, T=$t_total, Traj=$num_traj, Noise=Raising($strength)")

# 2. Initialize
# Initial State: |000000> (Z basis)
initial_state = MPS(L, state="zeros")

# Hamiltonian: Ising MPO
H = init_ising(L, J, h)

# Noise Model: Raising on all sites
processes = [Dict("name" => "raising", "sites" => [i], "strength" => strength) for i in 1:L]
noise_model = NoiseModel(processes, L)

# Observables: Z on site 1, site 3, site 6 (1-based indexing)
# Sites: 1, 3, 6
sites_to_measure = [1, 3, 6] 
observables = [Observable("Z_$i", ZGate(), i) for i in sites_to_measure]

# Config
sim_params = TimeEvolutionConfig(observables, t_total;
                                 dt=dt,
                                 num_traj=num_traj,
                                 max_bond_dim=32,
                                 truncation_threshold=1e-9,
                                 order=2,
                                 sample_timesteps=true)

# 3. Run Simulation
# We use the run function from Simulator which handles parallel execution and aggregation
Simulator.run(initial_state, H, sim_params, noise_model; parallel=true)

# 4. Save Results
println("\nSimulation finished. Saving results...")

# Format Data for CSV
# Header: Time, Z_1, Z_3, Z_6
header = ["Time" "Z_1" "Z_3" "Z_6"]
times = sim_params.times
data = Matrix{Float64}(undef, length(times), 4)
data[:, 1] = times

for (i, obs) in enumerate(sim_params.observables)
    # obs.results is Vector{Float64} of means
    data[:, i+1] = obs.results
end

# Write to CSV
output_file = "experiments/julia_analog_results.csv"
open(output_file, "w") do io
    writedlm(io, header, ',')
    writedlm(io, data, ',')
end

println("Results saved to $output_file")

