using Test
using LinearAlgebra
using Yaqs
using Yaqs.MPOModule
using Yaqs.MPSModule
using Yaqs.GateLibrary

@testset "MPO Tests" begin

    @testset "Initialization" begin
        L = 4
        mpo = MPO(L; identity=true)
        @test mpo.length == L
        
        # Check Identity
        T1 = mpo.tensors[1] # (1, 2, 2, 1)
        @test T1[1, 1, 1, 1] ≈ 1.0
        @test T1[1, 2, 2, 1] ≈ 1.0
        @test T1[1, 1, 2, 1] ≈ 0.0
        
        mpo_zero = MPO(L; identity=false)
        T1 = mpo_zero.tensors[1]
        @test sum(abs.(T1)) == 0.0
    end
    
    @testset "Ising Model" begin
        L = 3
        J = 1.0
        g = 0.5
        H = init_ising(L, J, g)
        
        # Check Dimensions
        @test size(H.tensors[1]) == (1, 2, 2, 3)
        @test size(H.tensors[2]) == (3, 2, 2, 3)
        @test size(H.tensors[3]) == (3, 2, 2, 1)
        
        # Check Expectation <000|H|000>
        # Ising H = -J ZZ - g X
        # |000> -> Z=+1. ZZ=+1.
        # <Z> = 1. <X> = 0.
        # Energy = -J(1*1 + 1*1) - g(0) = -2J = -2.0
        
        psi = MPS(L; state="zeros")
        val = expect_mpo(H, psi)
        @test isapprox(val, -2.0; atol=1e-10)
        
        # <+++|H|+++>
        # |+> is X eigenstate +1. <X>=1. <Z>=0.
        # Energy = -J(0) - g(1+1+1) = -3g = -1.5
        psi_x = MPS(L; state="x+")
        val_x = expect_mpo(H, psi_x)
        @test isapprox(val_x, -1.5; atol=1e-10)
    end

    @testset "Orthogonalize & Truncate" begin
        L = 4
        # Create a random MPO with high bond dimension
        tensors = Vector{Array{ComplexF64, 4}}(undef, L)
        chi = 4
        tensors[1] = randn(ComplexF64, 1, 2, 2, chi)
        for i in 2:L-1
            tensors[i] = randn(ComplexF64, chi, 2, 2, chi)
        end
        tensors[L] = randn(ComplexF64, chi, 2, 2, 1)
        
        mpo = MPO(L, tensors, fill(2, L))
        
        # Contract with random MPS to get baseline expectation
        psi = MPS(L; state="random")
        val_orig = expect_mpo(mpo, psi)
        
        # Orthogonalize to center 1
        orthogonalize!(mpo, 1)
        @test mpo.orth_center == 1
        
        val_orth = expect_mpo(mpo, psi)
        @test isapprox(val_orig, val_orth; atol=1e-10)
        
        # Truncate
        err = MPOModule.truncate!(mpo; max_bond_dim=2)
        @test mpo.orth_center == L
        
        # Check bond dims
        max_b = 0
        for t in mpo.tensors
            max_b = max(max_b, size(t, 1), size(t, 4))
        end
        @test max_b <= 2
        
        # Expectation should be close-ish (random MPO might not compress well, but code shouldn't crash)
        val_trunc = expect_mpo(mpo, psi)
        # Just check it runs and gives a number
        @test !isnan(val_trunc)
    end

    @testset "MPO Addition" begin
        L = 3
        H1 = init_ising(L, 1.0, 0.0) # -ZZ
        H2 = init_ising(L, 0.0, 1.0) # -X
        
        H_sum = H1 + H2
        
        # Should be equiv to J=1, g=1
        H_ref = init_ising(L, 1.0, 1.0)
        
        psi = MPS(L; state="random")
        v1 = expect_mpo(H_sum, psi)
        v2 = expect_mpo(H_ref, psi)
        
        @test isapprox(v1, v2; atol=1e-10)
        
        # Check bond dim growth
        # H1 bond 3, H2 bond 3. Sum should be max 6 (actually 3+3=6 at bulk)
        # Ising is optimized, so naive sum is larger than optimal construction.
        # H_sum bulk bond dim: 3+3 = 6.
        # H_ref bulk bond dim: 3.
        
        orthogonalize!(H_sum, 1)
        MPOModule.truncate!(H_sum; threshold=1e-10)
        
        # After compression, should be small (around 3 or 4)
        bond_dim = size(H_sum.tensors[2], 4)
        @test bond_dim <= 5 # Should compress down
    end

    @testset "MPO Multiplication" begin
        L = 2
        # X * X = I
        X = MPO(L) # Identity
        # Make it X on all sites
        X_op = Matrix(matrix(XGate()))
        for i in 1:L
            X.tensors[i] = reshape(X_op, 1, 2, 2, 1)
        end
        
        XX = contract_mpo_mpo(X, X)
        
        psi = MPS(L; state="zeros") # |00>
        # <00| X*X |00> = <00|I|00> = 1
        
        val = expect_mpo(XX, psi)
        @test isapprox(val, 1.0; atol=1e-10)
        
        # Bond dim of X is 1. XX is 1.
        @test size(XX.tensors[1], 4) == 1
end

end
