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

    @testset "Krylov Mode Controls & Stats" begin
        # Exercise exported control surface
        reset_krylov_ishermitian_stats!()
        reset_krylov_ishermitian_cache!()

        set_krylov_ishermitian_mode!(:lanczos)
        Z = [1.0 0.0; 0.0 -1.0]
        v0 = ComplexF64[1.0, 0.0]
        func_Z(x) = Z * x
        _ = Algorithms.expm_krylov(func_Z, v0, 0.1, 5)

        set_krylov_ishermitian_mode!(:arnoldi)
        X = [0.0 1.0; 1.0 0.0]
        func_X(x) = X * x
        _ = Algorithms.expm_krylov(func_X, v0, 0.1, 5)

        # Print function should run and produce output
        io = IOBuffer()
        redirect_stdout(io) do
            print_krylov_ishermitian_stats(header="stats")
        end
        out = String(take!(io))
        @test occursin("stats", out)
        @test occursin("expm_krylov calls", out)

        # Also check Bool overload
        set_krylov_ishermitian_mode!(true)
        set_krylov_ishermitian_mode!(false)
        set_krylov_ishermitian_mode!(:auto)
    end

    @testset "Internal helpers" begin
        # _ishermitian_check should accept a linear map and prototype vector
        Z = [1.0 0.0; 0.0 -1.0]
        func_Z(x) = Z * x
        @test Algorithms._ishermitian_check(func_Z, ComplexF64[1.0, 0.0]) == true

        X = [0.0 1.0; 1.0 0.0]
        func_X(x) = X * x
        @test Algorithms._ishermitian_check(func_X, ComplexF64[1.0, 0.0]) == true

        # _ensure_size! grows and preserves element type
        A = Array{ComplexF64,3}(undef, 2, 2, 2)
        Algorithms._ensure_size!(A, (3, 1, 4))
        @test size(A) == (3, 1, 4)
        @test eltype(A) == ComplexF64
    end

end

