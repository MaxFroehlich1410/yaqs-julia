using Test
using LinearAlgebra
using TensorOperations

if !isdefined(Main, :Yaqs)
    include("../src/Yaqs.jl")
    using .Yaqs
end
using .Yaqs.Decompositions

@testset "Decompositions" begin
    
    # Helper to create random complex tensor
    function randc(dims...)
        return randn(ComplexF64, dims...) + 1im * randn(ComplexF64, dims...)
    end

    @testset "Right QR" begin
        # Input: (Left, Phys, Right)
        l, p, r = 4, 2, 5
        T = randc(l, p, r)
        
        Q, R = right_qr(T)
        
        # Check dimensions
        # Q: (Left, Phys, NewBond)
        # R: (NewBond, Right)
        nb = size(Q, 3)
        @test size(Q) == (l, p, nb)
        @test size(R) == (nb, r)
        
        # Check Reconstruction
        # T[l,p,r] = Q[l,p,k] * R[k,r]
        @tensor T_rec[l,p,r] := Q[l,p,k] * R[k,r]
        @test T_rec ≈ T
        
        # Check Q is Left Canonical (Wait, Right QR makes Q left canonical?)
        # right_qr returns Q, R. 
        # Q is usually Isometry.
        # reshape Q to (l*p, nb). Q'Q = I.
        Q_mat = reshape(Q, l*p, nb)
        @test Q_mat' * Q_mat ≈ I
    end

    @testset "Left QR" begin
        # Input: (Left, Phys, Right)
        l, p, r = 5, 2, 4
        T = randc(l, p, r)
        
        L, Q = left_qr(T)
        
        # Check dimensions
        # L: (Left, NewBond)
        # Q: (NewBond, Phys, Right)
        nb = size(Q, 1)
        @test size(L) == (l, nb)
        @test size(Q) == (nb, p, r)
        
        # Check Reconstruction
        # T[l,p,r] = L[l,k] * Q[k,p,r]
        @tensor T_rec[l,p,r] := L[l,k] * Q[k,p,r]
        @test T_rec ≈ T
        
        # Check Q is Right Canonical (Q Q' = I on right indices?)
        # Q matrix (nb, p*r). Q Q' = I.
        Q_mat = reshape(Q, nb, p*r)
        @test Q_mat * Q_mat' ≈ I
    end
    
    @testset "Two Site SVD" begin
        # A: (Left_A, Phys_A, Bond)
        # B: (Bond, Phys_B, Right_B)
        bond = 4
        A = randc(3, 2, bond)
        B = randc(bond, 2, 5)
        
        # Full reconstruction (no truncation)
        A_new, B_new = two_site_svd(A, B, 1e-15)
        
        # Contract original
        @tensor T_orig[l, pa, pb, r] := A[l, pa, k] * B[k, pb, r]
        
        # Contract new
        @tensor T_new[l, pa, pb, r] := A_new[l, pa, k] * B_new[k, pb, r]
        
        @test T_orig ≈ T_new
        
        # Test Truncation
        # Create rank-deficient state
        # T = U * S * V. Set S to have only 2 non-zeros.
        U, _, V = svd(randc(4,4))
        S_trunc = [1.0, 0.5, 1e-8, 1e-9]
        M = U * Diagonal(S_trunc) * V' # Rank 2 effectively
        
        # Reshape M into our 4-leg tensor structure (simplification)
        # Just test that singular values are truncated.
        
        # Let's rely on numerical truncation of random full rank matrix
        # If we set threshold high, bond dim should drop.
        
        # Case 1: High threshold, no max_bond constraint
        # Random matrix (6x10) has rank 6. 
        # With threshold 0.1, we expect SOME truncation from max rank 6.
        A_trunc, B_trunc = two_site_svd(A, B, 2.0) # Very High threshold to force drop
        @test size(A_trunc, 3) < 6 
        
        # Case 2: Explicit max_bond_dim constraint
        A_lim, B_lim = two_site_svd(A, B, 1e-15; max_bond_dim=2)
        @test size(A_lim, 3) == 2
        @test size(B_lim, 1) == 2
    end

end

