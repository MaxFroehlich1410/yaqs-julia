using Test
using LinearAlgebra
using Yaqs
using Yaqs.GateLibrary
using Yaqs.MPSModule
using Yaqs.DigitalTJM
using Yaqs.SimulationConfigs
using Yaqs.CircuitLibrary
using Yaqs.NoiseModule

@testset "Digital TJM Simulation - 6 Qubit Ising with Noise" begin

    # 1. Parameters
    L = 6
    J = 1.0
    g = 0.5
    dt = 0.1
    timesteps = 10
    num_traj = 200
    
    # 2. Circuit
    circ = ising_circuit(L, J, g, dt, timesteps; periodic=false)
    
    # 3. Noise Model: Bitflip on every 2-qubit gate
    gamma = 0.01
    
    processes_all = Vector{Dict{String, Any}}()
    for i in 1:L
        push!(processes_all, Dict("name"=>"pauli_x", "sites"=>[i], "strength"=>gamma))
    end
    
    # Explicitly construct NoiseModel with correct types if inference failed before
    nm = NoiseModel(processes_all, L)
    
    # 4. Simulation Params (Strong Simulation with Trajectories)
    # Observables: <Z> on each site
    observables = [Observable("Z$i", ZGate(), i) for i in 1:L]
    
    sim_params = StrongMeasurementConfig(observables; 
                                         num_traj=num_traj, 
                                         sample_layers=true)
                                         
    # 5. Run Simulation (Manual Trajectory Loop)
    
    psi_init = MPS(L; state="zeros") # |000000>
    
    # Shared variables must be mutable or references
    # We use a Ref for scalar, but arrays are mutated in place.
    # However, we need to initialize results_sum safely.
    
    results_sum = zeros(ComplexF64, L, 1) # Initial placeholder
    results_sq_sum = zeros(ComplexF64, L, 1)
    num_steps_found = Ref(0)
    initialized = Ref(false)
    
    # Lock for accumulation
    lck = ReentrantLock()
    
    Threads.@threads for traj in 1:num_traj
        # Run one trajectory
        # We need a fresh copy of psi_init for each trajectory inside run_digital_tjm (it does deepcopy)
        # So passing psi_init is fine.
        
        _, traj_res = run_digital_tjm(psi_init, circ, nm, sim_params)
        
        lock(lck)
        try
            if !initialized[]
                # Resize/Reallocate accumulators based on actual steps
                # traj_res size: (num_obs, num_steps)
                results_sum = zeros(ComplexF64, size(traj_res))
                results_sq_sum = zeros(ComplexF64, size(traj_res))
                num_steps_found[] = size(traj_res, 2)
                initialized[] = true
            end
            
            if size(traj_res, 2) == num_steps_found[]
                results_sum .+= traj_res
                results_sq_sum .+= (traj_res .^ 2) 
            end
        finally
            unlock(lck)
        end
    end
    
    # Average
    results_avg = real.(results_sum) ./ num_traj
    
    # 6. Check Results
    @test num_steps_found[] > 0
    @test size(results_avg, 1) == L
    
    # Check evolution
    last_step_Z = results_avg[:, end]
    @test all(abs.(last_step_Z) .<= 1.0 + 1e-9)
    
end
