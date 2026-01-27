using Test
using LinearAlgebra

if !isdefined(Main, :Yaqs)
    include("../src/Yaqs.jl")
    using .Yaqs
end
using .Yaqs.GateLibrary
using .Yaqs.Decompositions
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.SimulationConfigs
using .Yaqs.Algorithms

@testset "Algorithms Tests" begin

    @testset "Krylov Expm" begin
        # Test exp(-im * dt * A) * v
        # Case 1: A = Pauli Z, v = |0> -> |0> * exp(-im * dt * 1)
        # Case 2: A = Pauli X, v = |0> -> cos(dt)|0> - im*sin(dt)|1>
        
        Z = [1.0 0.0; 0.0 -1.0]
        X = [0.0 1.0; 1.0 0.0]
        v0 = ComplexF64[1.0, 0.0]
        dt = 0.5
        
        func_Z(x) = Z * x
        v_out = Algorithms.expm_krylov(func_Z, v0, dt, 5)
        @test isapprox(v_out, v0 * exp(-1im * dt), atol=1e-10)
        
        func_X(x) = X * x
        v_out_X = Algorithms.expm_krylov(func_X, v0, dt, 5)
        expected = cos(dt)*v0 - 1im*sin(dt)*ComplexF64[0.0, 1.0]
        @test isapprox(v_out_X, expected, atol=1e-10)
    end

    @testset "Single Site TDVP - Ising Eigenstate" begin
        L = 4
        J = 1.0
        g = 0.0
        mpo = init_ising(L, J, g)
        # Ground state |0000> or |1111>. Energy = -3.0
        mps = MPS(L; state="zeros") 
        
        # Evolve
        dt = 0.1
        t_total = 0.5
        config = TimeEvolutionConfig(Observable[], t_total; dt=dt)
        
        # Run steps
        steps = Int(t_total / dt)
        for _ in 1:steps
            single_site_tdvp!(mps, mpo, config)
        end
        
        # Check overlap with initial state
        mps0 = MPS(L; state="zeros")
        ov = scalar_product(mps, mps0)
        
        # Expected phase: exp(-i * E * t) = exp(-i * -3.0 * 0.5) = exp(1.5i)
        expected = exp(1.5im)
        
        @test isapprox(abs(ov), 1.0, atol=1e-5) # Norm preserved
        # Phase check might be tricky due to global phase ambiguity in MPS?
        # But TDVP should preserve phase if exact.
        # Let's check real/imag parts.
        
        @test isapprox(ov, expected, atol=1e-2) # Allow some error due to Trotter/Krylov
    end
    
    @testset "Two Site TDVP - Ising Eigenstate" begin
        L = 4
        J = 1.0
        g = 0.0
        mpo = init_ising(L, J, g)
        mps = MPS(L; state="zeros") 
        
        dt = 0.1
        t_total = 0.5
        config = TimeEvolutionConfig(Observable[], t_total; dt=dt)
        
        steps = Int(t_total / dt)
        for _ in 1:steps
            two_site_tdvp!(mps, mpo, config)
        end
        
        mps0 = MPS(L; state="zeros")
        ov = scalar_product(mps, mps0)
        expected = exp(1.5im)
        
        @test isapprox(abs(ov), 1.0, atol=1e-5)
        @test isapprox(ov, expected, atol=1e-2)
    end

    @testset "SRC MPO×MPS (random_contraction)" begin
        using Random

        # Small deterministic problem: compare SRC result to exact MPO×MPS contraction.
        rng = MersenneTwister(1234)
        L = 4
        d = 2

        # Random MPS with modest bond dimensions.
        # Bonds: 1 - χ2 - χ3 - 1
        χ2, χ3 = 3, 2
        A = Vector{Array{ComplexF64,3}}(undef, L)
        A[1] = randn(rng, ComplexF64, 1, d, χ2)
        A[2] = randn(rng, ComplexF64, χ2, d, χ3)
        A[3] = randn(rng, ComplexF64, χ3, d, 1)
        # Add a trivial site 4 to make L=4 with final bond 1
        A[4] = randn(rng, ComplexF64, 1, d, 1)
        psi = MPS(L, A, fill(d, L), 1)
        MPSModule.normalize!(psi)

        # Random MPO with modest bond dimensions (square local operator)
        # Bonds: 1 - μ2 - μ3 - 1
        μ2, μ3 = 2, 3
        W = Vector{Array{ComplexF64,4}}(undef, L)
        W[1] = randn(rng, ComplexF64, 1, d, d, μ2)
        W[2] = randn(rng, ComplexF64, μ2, d, d, μ3)
        W[3] = randn(rng, ComplexF64, μ3, d, d, 1)
        W[4] = randn(rng, ComplexF64, 1, d, d, 1)
        H = MPO(L, W, fill(d, L), 0)

        # Exact (uncompressed) application
        exact = contract_mpo_mps(H, psi)
        v_exact = to_vec(exact)

        # SRC (adaptive). Use a small cutoff and seeded RNG for determinism.
        psi_src = random_contraction(H, psi; stop=Cutoff(1e-10), sketchdim=1, sketchincrement=1, rng=rng)
        @test check_if_valid_mps(psi_src)

        v_src = to_vec(psi_src)
        relerr = norm(v_src - v_exact) / max(norm(v_exact), 1e-30)
        @test relerr ≤ 1e-6
    end

end

