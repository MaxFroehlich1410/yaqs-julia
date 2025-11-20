using LinearAlgebra
using Printf
using Test

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

function debug_ising()
    println("--- Debugging Ising Model TDVP ---")
    L = 6
    J = 1.0
    g = 0.5
    
    println("Initializing Ising MPO (J=$J, g=$g, L=$L)...")
    H = init_ising(L, J, g)
    
    println("Initializing MPS (zeros)...")
    psi = MPS(L; state="zeros")
    
    # Check Initial Energy
    E0 = expect_mpo(H, psi)
    println("Initial Energy <psi|H|psi>: $E0")
    
    # Expected Energy:
    # |000000>
    # Z Z term: <00|Z Z|00> = 1.0. Coeff -J = -1.0.
    # 5 bonds -> -5.0.
    # X term: <0|X|0> = 0.
    # Total = -5.0.
    if !isapprox(real(E0), -5.0; atol=1e-8)
        println("WARNING: Initial Energy incorrect! Expected -5.0, got $E0")
    else
        println("Initial Energy correct (-5.0).")
    end
    
    # Evolve 1 step
    dt = 0.05
    config = TimeEvolutionConfig(Observable[], dt; dt=dt)
    
    println("Running 1-Site TDVP Step (dt=$dt)...")
    single_site_tdvp!(psi, H, config)
    
    E1 = expect_mpo(H, psi)
    println("Energy after 1 step: $E1")
    
    if !isapprox(real(E1), -5.0; atol=1e-4)
        println("WARNING: Energy NOT conserved!")
    else
        println("Energy conserved.")
    end
    
    # Measure Z_1
    z1 = real(local_expect(psi, matrix(ZGate()), 1))
    println("Expectation <Z_1>: $z1")
    
    # Measure Z_2
    z2 = real(local_expect(psi, matrix(ZGate()), 2))
    println("Expectation <Z_2>: $z2")
    
    println("Running 20 steps...")
    for step in 1:20
        single_site_tdvp!(psi, H, config)
        E_curr = real(expect_mpo(H, psi))
        z1 = real(local_expect(psi, matrix(ZGate()), 1))
        @printf("T=%.2f, E=%.4f, <Z1>=%.4f\n", config.dt * step, E_curr, z1)
    end
    
    println("--- Done ---")
end

debug_ising()

