using Test
using LinearAlgebra
using Random

# Include source files (assuming running from project root or test dir)
# Adjust paths as necessary
if !isdefined(Main, :GateLibrary)
    include("../src/GateLibrary.jl")
end
if !isdefined(Main, :Decompositions)
    include("../src/Decompositions.jl")
end
if !isdefined(Main, :MPSModule)
    include("../src/MPS.jl")
end

using .GateLibrary
using .Decompositions
using .MPSModule

# Helper functions for testing
function crandn(shape)
    return (randn(ComplexF64, shape) .+ 1im .* randn(ComplexF64, shape)) ./ sqrt(2)
end

function random_mps(length, phys_dim, bond_dim; normalize_state=true)
    # Helper to create a random MPS with specific bond dimensions
    # Simplified version of the Python one
    shapes = []
    # Left boundary
    push!(shapes, (1, phys_dim, min(bond_dim, phys_dim)))
    
    current_bond = min(bond_dim, phys_dim)
    for i in 2:length-1
        next_bond = min(bond_dim, current_bond * phys_dim) # Heuristic
        # Actually let's just use fixed bond dim for inner
        push!(shapes, (bond_dim, phys_dim, bond_dim))
    end
    # Right boundary
    push!(shapes, (bond_dim, phys_dim, 1))
    
    # Override for small lengths or specific shapes if needed
    # For now, just generate random tensors fitting the (Left, Phys, Right) layout
    tensors = [crandn((1, phys_dim, bond_dim))]
    for _ in 2:length-1
        push!(tensors, crandn((bond_dim, phys_dim, bond_dim)))
    end
    push!(tensors, crandn((bond_dim, phys_dim, 1)))
    
    # Fix dimensions for boundaries if length is small
    if length == 1
        tensors = [crandn((1, phys_dim, 1))]
    elseif length == 2
        tensors = [crandn((1, phys_dim, bond_dim)), crandn((bond_dim, phys_dim, 1))]
    end
    
    mps = MPS(length, tensors, [phys_dim for _ in 1:length])
    if normalize_state
        MPSModule.normalize!(mps)
    end
    return mps
end

@testset "MPS Tests" begin

    @testset "Initialization" begin
        length = 4
        pdim = 2
        
        # Test "zeros"
        mps = MPS(length; physical_dimensions=pdim, state="zeros")
        @test mps.length == length
        @test Base.length(mps.tensors) == length
        @test all(d == pdim for d in mps.phys_dims)
        for i in 1:length
            # Check |0> state: [1, 0]
            # Tensor shape (1, 2, 1)
            vec = mps.tensors[i][1, :, 1]
            @test isapprox(vec, [1.0, 0.0])
        end

        # Test "ones"
        mps = MPS(length; physical_dimensions=pdim, state="ones")
        for i in 1:length
            vec = mps.tensors[i][1, :, 1]
            @test isapprox(vec, [0.0, 1.0])
        end
        
        # Test "x+"
        mps = MPS(length; physical_dimensions=pdim, state="x+")
        expected = [1.0, 1.0] ./ sqrt(2)
        for i in 1:length
            vec = mps.tensors[i][1, :, 1]
            @test isapprox(vec, expected)
        end
        
        # Test "basis"
        basis_str = "1001"
        mps = MPS(length; physical_dimensions=pdim, state="basis", basis_string=basis_str)
        for (i, char) in enumerate(basis_str)
            bit = parse(Int, char)
            vec = mps.tensors[i][1, :, 1]
            expected_vec = zeros(ComplexF64, 2)
            expected_vec[bit+1] = 1.0
            @test isapprox(vec, expected_vec)
        end
    end

    @testset "Custom Tensors" begin
        length = 3
        pdim = 2
        t1 = crandn((1, pdim, 2))
        t2 = crandn((2, pdim, 2))
        t3 = crandn((2, pdim, 1))
        tensors = [t1, t2, t3]
        
        mps = MPS(length; tensors=tensors, physical_dimensions=[pdim, pdim, pdim])
        @test mps.length == length
        for i in 1:length
            @test mps.tensors[i] ≈ tensors[i]
        end
    end

    @testset "Scalar Product" begin
        # <psi|psi> = 1
        mps = MPS(3; state="random")
        val = scalar_product(mps, mps)
        @test isapprox(val, 1.0 + 0.0im; atol=1e-12)
        
        # Orthogonal
        mps0 = MPS(3; state="zeros")
        mps1 = MPS(3; state="ones")
        val = scalar_product(mps0, mps1)
        @test isapprox(val, 0.0 + 0.0im; atol=1e-12)
    end
    
    @testset "Expectation Values" begin
        # Z on |0> -> 1
        mps = MPS(2; state="zeros")
        Z_gate = [1.0 0.0; 0.0 -1.0]
        
        # Julia 1-based indexing: site 1
        val = local_expect(mps, Z_gate, 1)
        @test isapprox(val, 1.0 + 0.0im; atol=1e-12)
        
        val = local_expect(mps, Z_gate, 2)
        @test isapprox(val, 1.0 + 0.0im; atol=1e-12)
        
        # X on |+> -> 1
        mps_x = MPS(2; state="x+")
        X_gate = [0.0 1.0; 1.0 0.0]
        val = local_expect(mps_x, X_gate, 1)
        @test isapprox(val, 1.0 + 0.0im; atol=1e-12)
    end
    
    @testset "Efficient All-Site Expectations" begin
        mps = MPS(4; state="random")
        Z_gate = [1.0 0.0; 0.0 -1.0]
        ops = [Z_gate for _ in 1:4]
        
        # Calculate individually
        vals_individual = [local_expect(mps, Z_gate, i) for i in 1:4]
        
        # Calculate all at once (O(L))
        vals_all = evaluate_all_local_expectations(mps, ops)
        
        @test isapprox(vals_individual, vals_all; atol=1e-12)
    end

    @testset "Measurements" begin
        # Single shot on |000> -> 0 (int 0)
        mps = MPS(3; state="zeros")
        # Our measure implementation returns an int bitstring
        res = MPSModule.single_shot_measure(mps)
        # |000> -> 0
        @test res == 0
        
        # Single shot on |111> -> 7
        mps = MPS(3; state="ones")
        res = MPSModule.single_shot_measure(mps)
        @test res == 7 # 1*1 + 1*2 + 1*4 = 7 (Little Endian?)
        # Python logic: sum(c << i for i, c in enumerate(bitstring)) -> Site 0 is LSB.
        # Julia logic implemented: bitstring += (outcome - 1) * (1 << (i - 1))
        # i=1 (Site 1) -> shift 0.
        # So Site 1 is LSB.
        
        # Measure shots
        shots = 100
        results = measure_shots(mps, shots)
        @test haskey(results, 7)
        @test results[7] == shots

        # New measurement checks
        mps_x = MPS(4; state="x+") # |++++>
        counts = measure_shots(mps_x, 100)
        @test sum(values(counts)) == 100
    end

    @testset "Initialization States" begin
        L = 4
        
        # x+ state: (1/sqrt(2), 1/sqrt(2))
        mps = MPS(L; state="x+")
        # Overlap with zeros should be (1/sqrt(2))^L
        mps0 = MPS(L; state="zeros")
        ov = scalar_product(mps, mps0)
        expected = (1.0/sqrt(2))^L
        @test isapprox(abs(ov), expected; atol=1e-10)
        
        # y+ state: (1/sqrt(2), i/sqrt(2))
        mps_y = MPS(L; state="y+")
        # Norm should be 1
        @test isapprox(scalar_product(mps_y, mps_y), 1.0; atol=1e-10)
        
        # Basis state
        basis_str = "0101"
        mps_basis = MPS(L; state="basis", basis_string=basis_str)
        # Check tensors directly
        # Site 1 (0): [1, 0]
        # Site 2 (1): [0, 1]
        @test mps_basis.tensors[1][1, 1, 1] ≈ 1.0
        @test mps_basis.tensors[1][1, 2, 1] ≈ 0.0
        @test mps_basis.tensors[2][1, 1, 1] ≈ 0.0
        @test mps_basis.tensors[2][1, 2, 1] ≈ 1.0
    end
    
    @testset "Canonical Forms & Shifting" begin
        mps = random_mps(4, 2, 4)
        
        # Shift to right
        shift_orthogonality_center_right!(mps, 1) # Shift 1 -> 2
        # We can't easily check canonical form without exposing internal details or implementing `check_canonical_form` fully.
        # The function `check_canonical_form` was implemented.
        
        # Check canonical form
        centers = check_canonical_form(mps)
        # It should be non-empty if valid
        @test !isempty(centers)
        
        # Normalize
        MPSModule.normalize!(mps)
        @test isapprox(scalar_product(mps, mps), 1.0; atol=1e-12)
    end
    
    @testset "Truncation" begin
        # Create MPS with large bond dim
        L = 6
        pdim = 2
        large_bond = 8
        # Manually build random tensors
        tensors = [crandn((1, pdim, large_bond))]
        for _ in 2:L-1
            push!(tensors, crandn((large_bond, pdim, large_bond)))
        end
        push!(tensors, crandn((large_bond, pdim, 1)))
        
        mps = MPS(L, tensors, [pdim for _ in 1:L])
        MPSModule.normalize!(mps)
        
        # Truncate to max_bond=2
        truncate!(mps; max_bond_dim=2)
        
        # Check bond dims
        max_b = 0
        for t in mps.tensors
            max_b = max(max_b, size(t, 1), size(t, 3))
        end
        @test max_b <= 2
        
        # Check valid state
        @test isapprox(scalar_product(mps, mps), 1.0; atol=1e-12) # Should likely remain normalized or close?
        # Truncation usually keeps norm if we handle singular values right. 
        # Our implementation normalizes? No, it just truncates U, S, V. 
        # The norm might decrease. But typically we re-normalize after truncation if desired.
        # The `truncate!` function updates tensors with U*S*V approx.
    end

end

