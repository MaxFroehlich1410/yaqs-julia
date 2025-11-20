using LinearAlgebra
using Printf

# Helper to ensure modules are loaded
if !isdefined(Main, :GateLibrary)
    include("../src/GateLibrary.jl")
end
if !isdefined(Main, :Decompositions)
    include("../src/Decompositions.jl")
end
if !isdefined(Main, :MPSModule)
    include("../src/MPS.jl")
end
if !isdefined(Main, :MPOModule)
    include("../src/MPO.jl")
end
if !isdefined(Main, :SimulationConfigs)
    include("../src/SimulationConfigs.jl")
end
if !isdefined(Main, :Algorithms)
    include("../src/Algorithms.jl")
end

using .GateLibrary
using .MPSModule
using .MPOModule
using .SimulationConfigs
using .Algorithms

function run_large_simulation_timed()
    L = 30
    J = 1.0
    g = 0.5
    dt = 0.05
    t_max = 2.0
    steps = Int(t_max / dt)
    max_bond = 32
    
    println("Initializing Large Ising Simulation (L=$L, max_bond=$max_bond)...")
    println("Hamiltonian: Ising Ferromagnetic (J=$J, g=$g)")
    
    H = init_ising(L, J, g)
    psi = MPS(L; state="zeros")
    
    # Config
    config = TimeEvolutionConfig(Observable[], t_max; dt=dt, max_bond_dim=max_bond)
    
    sites = [1, 16, 30] 
    op_Z = matrix(ZGate())
    
    # Pre-allocate results storage (Time, Site, Value)
    # Rows: (Initial + steps) * num_sites
    num_measurements = (steps + 1) * length(sites)
    results = Vector{Tuple{Float64, Int, Float64}}(undef, num_measurements)
    
    println("Starting Simulation Loop (Timing started)...")
    
    # Start Timer
    t_start = time()
    
    # Initial Measurement
    idx = 1
    for s in sites
        val = real(local_expect(psi, op_Z, s))
        results[idx] = (0.0, s, val)
        idx += 1
    end
    
    for step in 1:steps
        two_site_tdvp!(psi, H, config)
        current_time = step * dt
        
        # Measure (in-memory)
        for s in sites
            val = real(local_expect(psi, op_Z, s))
            results[idx] = (current_time, s, val)
            idx += 1
        end
    end
    
    t_end = time()
    elapsed = t_end - t_start
    
    println("Simulation complete.")
    @printf("Total Execution Time: %.4f seconds\n", elapsed)
    @printf("Average Time per Step: %.4f seconds\n", elapsed / steps)
    
    # Write results to file (outside timed region)
    filename = "large_ising_results.csv"
    open(filename, "w") do io
        write(io, "Time,Site,ExpVal\n")
        for (t, s, val) in results
            write(io, "$t,$s,$val\n")
        end
    end
    println("Data saved to $filename")
end

run_large_simulation_timed()
