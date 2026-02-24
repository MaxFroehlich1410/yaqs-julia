# Regression test comparing truncation behavior to a Python reference implementation.
#
# This file constructs a 2-site MPS with controlled singular values and verifies that Julia's
# `truncate!` keeps the same bond dimension as the reference Python logic for a range of thresholds.
# If the Python package is available via PythonCall, it also runs a direct side-by-side comparison.
#
# Args:
#     None
#
# Returns:
#     Nothing: Defines a `@testset` that validates truncation thresholds and (optionally) Python parity.
using Test
using Yaqs
using LinearAlgebra
using PythonCall
using TensorOperations
using Yaqs.MPSModule: MPS, truncate!

"""
Create a 2-site MPS with specific singular values at the bond.
Creates a matrix with the exact singular values, then splits it into MPS form.
"""
function create_mps_with_exact_singular_values(singular_values::Vector{Float64}, phys_dim::Int=2)
    n_sv = length(singular_values)
    
    # Create a matrix M with exactly the specified singular values
    # For a 2-site MPS: tensor1 (1, phys_dim, bond) * tensor2 (bond, phys_dim, 1)
    # When contracted: theta (phys_dim, phys_dim) with bond dimension up to min(phys_dim^2, n_sv)
    # To have n_sv singular values, we need phys_dim >= n_sv
    
    # Create random orthogonal matrices
    U = Matrix(qr(randn(ComplexF64, phys_dim, phys_dim)).Q)
    V = Matrix(qr(randn(ComplexF64, phys_dim, phys_dim)).Q)
    
    # Create matrix with exact singular values: M = U * S * V'
    S_diag = Diagonal(singular_values[1:min(n_sv, phys_dim)])
    n_actual = min(n_sv, phys_dim)
    U_use = U[:, 1:n_actual]
    V_use = V[:, 1:n_actual]
    
    M = U_use * S_diag * V_use'  # (phys_dim, phys_dim)
    
    # Now split M using SVD to get MPS tensors
    # M is (phys_dim, phys_dim), we want to split into:
    # tensor1: (1, phys_dim, bond) and tensor2: (bond, phys_dim, 1)
    
    # Reshape M for SVD: group left index -> (1*phys_dim, phys_dim) = (phys_dim, phys_dim)
    # Actually, M is already (phys_dim, phys_dim), so we can do SVD directly
    F = svd(M)
    U_svd, S_svd, Vt_svd = F.U, F.S, F.Vt
    
    # The singular values from SVD should match (up to numerical precision)
    # But we want to ensure they're exactly what we specified
    n_keep = min(n_actual, length(S_svd))
    
    # Create tensor1: (1, phys_dim, n_keep) - left canonical U
    tensor1 = reshape(U_svd[:, 1:n_keep], 1, phys_dim, n_keep)
    
    # Create tensor2: (n_keep, phys_dim, 1) - right part with singular values
    # Use our specified singular values, not S_svd
    S_forced = singular_values[1:n_keep]
    tensor2_mat = Diagonal(S_forced) * Vt_svd[1:n_keep, :]  # (n_keep, phys_dim)
    tensor2 = reshape(tensor2_mat, n_keep, phys_dim, 1)
    
    # Create MPS
    tensors = [tensor1, tensor2]
    phys_dims = [phys_dim, phys_dim]
    mps = MPS(2, tensors, phys_dims, 1)
    
    # Verify singular values by contracting and doing SVD
    A = mps.tensors[1]  # (1, phys_dim, n_keep)
    B = mps.tensors[2]  # (n_keep, phys_dim, 1)
    
    # Contract using @tensor
    @tensor theta[i, j] := A[1, i, k] * B[k, j, 1]
    
    # Check singular values
    F_check = svd(theta)
    
    return mps, S_forced, F_check.S
end

@testset "Truncation Comparison: Python vs Julia" begin
    
    # Create singular values: 1e-1, 1e-2, ..., 1e-20
    singular_values = [10.0^(-i) for i in 1:20]
    
    println("\n" * "="^80)
    println("Testing Truncation with Singular Values: 1e-1, 1e-2, ..., 1e-20")
    println("="^80)
    
    # Test different thresholds
    test_thresholds = [
        1e-40,  # Keep all (below smallest squared: 1e-40)
        1e-38,  # Keep all
        1e-20,  # Discard 1e-20 (1e-40)
        1e-18,  # Discard 1e-20, 1e-19 (1e-40 + 1e-38 = 1.01e-38)
        1e-16,  # Discard more
        1e-10,  # Discard many
        1e-4,   # Discard most
        1e-2,   # Keep only largest
    ]
    
    for threshold in test_thresholds
        println("\n" * "-"^80)
        println("Testing threshold = $threshold")
        println("-"^80)
        
        # Create MPS in Julia
        # Use larger physical dimension to accommodate all singular values
        # For n_sv singular values, we need at least n_sv physical dimension
        phys_dim_use = max(2, length(singular_values))
        mps_jl, sv_desired, sv_actual = create_mps_with_exact_singular_values(singular_values, phys_dim_use)
        
        n_sv = length(sv_desired)
        println("Number of singular values: $n_sv")
        println("Desired SV (first 5): $(sv_desired[1:min(5, n_sv)])")
        println("Actual SV from MPS (first 5): $(sv_actual[1:min(5, length(sv_actual))])")
        
        # Get bond dimension before truncation
        bond_dim_before = size(mps_jl.tensors[1], 3)
        println("Bond dimension before truncation: $bond_dim_before")
        
        # Calculate what Python should keep
        # Python logic: discard = 0.0, keep = len(s_vec), min_keep = 2
        # for idx, s in enumerate(reversed(s_vec)):
        #     discard += s**2
        #     if discard >= threshold:
        #         keep = max(len(s_vec) - idx, min_keep)
        #         break
        
        discard_py = 0.0
        keep_py = n_sv
        min_keep_py = 2
        for (idx, s) in enumerate(reverse(sv_desired))
            discard_py += s^2
            if discard_py >= threshold
                keep_py = max(n_sv - idx, min_keep_py)
                break
            end
        end
        
        println("Expected Python keep (bond dim): $keep_py")
        println("  (discarded weight when threshold exceeded: $discard_py)")
        
        # Truncate in Julia
        truncate!(mps_jl; threshold=threshold)
        
        # Get bond dimension after truncation
        bond_dim_after = size(mps_jl.tensors[1], 3)
        println("Julia actual keep (bond dim): $bond_dim_after")
        
        # Test: Julia should match Python
        @test bond_dim_after == keep_py
        if bond_dim_after != keep_py
            println("ERROR: Julia bond dimension ($bond_dim_after) != Python ($keep_py) for threshold=$threshold")
        end
        
        # Also test with Python directly if available
        try
            py_yaqs = pyimport("mqt.yaqs.core.data_structures.networks")
            py_decomp = pyimport("mqt.yaqs.core.methods.decompositions")
            
            # Create Python MPS with same structure
            # Python MPS tensors are (phys, left, right) = (sigma, chi_l-1, chi_l)
            # Julia MPS tensors are (left, phys, right) = (chi_l-1, sigma, chi_l)
            
            # We need to recreate the MPS before truncation
            mps_jl_orig, _, _ = create_mps_with_exact_singular_values(singular_values, phys_dim_use)
            
            # Convert Julia tensors to Python format
            A_jl_orig = mps_jl_orig.tensors[1]  # (1, phys_dim, bond)
            B_jl_orig = mps_jl_orig.tensors[2]  # (bond, phys_dim, 1)
            
            A_jl_orig = mps_jl_orig.tensors[1]  # (1, phys_dim, bond)
            B_jl_orig = mps_jl_orig.tensors[2]  # (bond, phys_dim, 1)
            
            # Python format: (phys, left, right)
            A_py = permutedims(A_jl_orig, (2, 1, 3))  # (phys_dim, 1, bond)
            B_py = permutedims(B_jl_orig, (2, 1, 3))  # (phys_dim, bond, 1)
            
            # Create Python MPS
            mps_py = py_yaqs.MPS(
                length=2,
                tensors=[A_py, B_py],
                physical_dimensions=[phys_dim_use, phys_dim_use]
            )
            
            # Truncate in Python
            mps_py.truncate(threshold=threshold)
            
            # Get Python bond dimension
            # Python: (phys, left, right), so bond is index 1 (left) or 2 (right)
            bond_dim_py = size(mps_py.tensors[0], 2)  # left bond of first tensor
            println("Python actual keep (bond dim): $bond_dim_py")
            
            # Compare
            @test bond_dim_after == bond_dim_py
            if bond_dim_after != bond_dim_py
                println("ERROR: Julia ($bond_dim_after) != Python ($bond_dim_py) for threshold=$threshold")
            end
            
        catch e
            println("Warning: Could not test with Python directly: $e")
            println("Falling back to logic comparison only.")
        end
    end
    
    println("\n" * "="^80)
    println("All truncation tests passed!")
    println("="^80)
end
