using Test
using LinearAlgebra
using Yaqs
using Yaqs.CircuitLibrary
using Yaqs.GateLibrary
using Yaqs.MPSModule
using Yaqs.SimulationConfigs
using Yaqs.Simulator
using Yaqs.NoiseModule

@testset "Ising Digital TJM 200 Trajectories" begin
    L = 6
    J = 1.0
    g = 1.0
    dt_circuit = 0.1
    timesteps = 10
    
    # 1. Create Circuit
    circ = ising_circuit(L, J, g, dt_circuit, timesteps)
    
    # 2. Define Noise Model
    gamma = 0.05
    noise_procs = Vector{Dict{String, Any}}()
    for i in 1:L
        push!(noise_procs, Dict("name"=>"bitflip_$i", "sites"=>[i], "strength"=>gamma, "matrix"=>matrix(XGate())))
    end
    nm = NoiseModel(noise_procs, L)
    
    # 3. Define Observables
    obs_list = Observable[]
    for i in 1:L
        push!(obs_list, Observable("Z_$i", ZGate(), i))
    end
    
    # 4. Simulation Config
    # We need enough storage for all layers. 10 timesteps * ~3 layers/step = 30 layers.
    # We set total_time=50.0 with dt=1.0 to allocate buffer of size 51.
    sim_params = TimeEvolutionConfig(obs_list, 50.0; dt=1.0, num_traj=200, sample_timesteps=true)
    
    # 5. Initial State
    psi = MPS(L; state="zeros")
    
    # 6. Run Simulation
    println("Running 200 trajectories for 6-qubit Ising model...")
    Simulator.run(psi, circ, sim_params, nm)
    
    # 7. Verify
    results_z1 = obs_list[1].results
    
    @test !isempty(results_z1)
    println("Number of sampled points: $(length(results_z1))")
    
    @test length(results_z1) > timesteps
    
    # Check values are within bounds [-1, 1]
    @test all(x -> -1.0 - 1e-9 <= x <= 1.0 + 1e-9, results_z1)
    
    # Check decay
    @test isapprox(results_z1[1], 1.0; atol=1e-1)
    
    println("Final Z1: $(results_z1[end])")
end
