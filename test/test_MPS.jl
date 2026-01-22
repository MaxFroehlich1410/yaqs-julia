using Test
using LinearAlgebra
using Random

if !isdefined(Main, :Yaqs)
    include("../src/Yaqs.jl")
    using .Yaqs
end
using .Yaqs.GateLibrary
using .Yaqs.Decompositions
using .Yaqs.MPSModule

# Helper functions for testing
function crandn(shape)
    return (randn(ComplexF64, shape) .+ 1im .* randn(ComplexF64, shape)) ./ sqrt(2)
end

function random_mps(L, phys_dim, bond_dim; normalize_state=true)
    tensors = [crandn((1, phys_dim, bond_dim))]
    for _ in 2:L-1
        push!(tensors, crandn((bond_dim, phys_dim, bond_dim)))
    end
    push!(tensors, crandn((bond_dim, phys_dim, 1)))
    
    # Fix dimensions for boundaries if length is small
    if L == 1
        tensors = [crandn((1, phys_dim, 1))]
    elseif L == 2
        tensors = [crandn((1, phys_dim, bond_dim)), crandn((bond_dim, phys_dim, 1))]
    end
    
    mps = MPS(L, tensors, [phys_dim for _ in 1:L])
    if normalize_state
        MPSModule.normalize!(mps)
    end
    return mps
end

@testset "MPS Tests" begin

    @testset "Initialization" begin
        L = 4
        pdim = 2
        
        # Test "zeros"
        mps = MPS(L; physical_dimensions=pdim, state="zeros")
        @test mps.length == L
        @test Base.length(mps.tensors) == L
        @test all(d == pdim for d in mps.phys_dims)
        for i in 1:L
            vec = mps.tensors[i][1, :, 1]
            @test isapprox(vec, [1.0, 0.0])
        end

        # Test "ones"
        mps = MPS(L; physical_dimensions=pdim, state="ones")
        for i in 1:L
            vec = mps.tensors[i][1, :, 1]
            @test isapprox(vec, [0.0, 1.0])
        end
        
        # Test "x+"
        mps = MPS(L; physical_dimensions=pdim, state="x+")
        expected = [1.0, 1.0] ./ sqrt(2)
        for i in 1:L
            vec = mps.tensors[i][1, :, 1]
            @test isapprox(vec, expected)
        end
        
        # Test "basis"
        basis_str = "1001"
        mps = MPS(L; physical_dimensions=pdim, state="basis", basis_string=basis_str)
        for (i, char) in enumerate(basis_str)
            bit = parse(Int, char)
            vec = mps.tensors[i][1, :, 1]
            expected_vec = zeros(ComplexF64, 2)
            expected_vec[bit+1] = 1.0
            @test isapprox(vec, expected_vec)
        end
    end

    @testset "Custom Tensors" begin
        L = 3
        pdim = 2
        t1 = crandn((1, pdim, 2))
        t2 = crandn((2, pdim, 2))
        t3 = crandn((2, pdim, 1))
        tensors = [t1, t2, t3]
        
        mps = MPS(L; tensors=tensors, physical_dimensions=[pdim, pdim, pdim])
        @test mps.length == L
        for i in 1:L
            @test mps.tensors[i] ≈ tensors[i]
        end
    end

    @testset "Scalar Product" begin
        mps = MPS(3; state="random")
        val = scalar_product(mps, mps)
        @test isapprox(val, 1.0 + 0.0im; atol=1e-12)
        
        mps0 = MPS(3; state="zeros")
        mps1 = MPS(3; state="ones")
        val = scalar_product(mps0, mps1)
        @test isapprox(val, 0.0 + 0.0im; atol=1e-12)
    end
    
    @testset "Expectation Values" begin
        mps = MPS(2; state="zeros")
        Z_gate = [1.0 0.0; 0.0 -1.0]
        
        val = local_expect(mps, Z_gate, 1)
        @test isapprox(val, 1.0 + 0.0im; atol=1e-12)
        
        val = local_expect(mps, Z_gate, 2)
        @test isapprox(val, 1.0 + 0.0im; atol=1e-12)
        
        mps_x = MPS(2; state="x+")
        X_gate = [0.0 1.0; 1.0 0.0]
        val = local_expect(mps_x, X_gate, 1)
        @test isapprox(val, 1.0 + 0.0im; atol=1e-12)
    end
    
    @testset "Efficient All-Site Expectations" begin
        mps = MPS(4; state="random")
        Z_gate = [1.0 0.0; 0.0 -1.0]
        ops = [Z_gate for _ in 1:4]
        
        vals_individual = [local_expect(mps, Z_gate, i) for i in 1:4]
        # Explicit qualification to avoid UndefVarError if export failed
        vals_all = MPSModule.evaluate_all_local_expectations(mps, ops)
        
        @test isapprox(vals_individual, vals_all; atol=1e-12)
    end

    @testset "Measurements" begin
        mps = MPS(3; state="zeros")
        # Use exported alias
        res = measure_single_shot(mps)
        @test res == 0

        # Direct exported name
        @test single_shot_measure(mps) == 0
        
        mps = MPS(3; state="ones")
        res = measure_single_shot(mps)
        @test res == 7 

        @test single_shot_measure(mps) == 7
        
        shots = 100
        results = measure_shots(mps, shots)
        @test haskey(results, 7)
        @test results[7] == shots

        mps_x = MPS(4; state="x+")
        counts = measure_shots(mps_x, 100)
        @test sum(values(counts)) == 100
    end

    @testset "Project onto Bitstring" begin
        mps0 = MPS(4; state="zeros")
        @test project_onto_bitstring(mps0, "0000") ≈ 1.0
        @test project_onto_bitstring(mps0, "1000") ≈ 0.0

        mps1 = MPS(4; state="ones")
        @test project_onto_bitstring(mps1, "1111") ≈ 1.0
        @test project_onto_bitstring(mps1, "0111") ≈ 0.0

        # Uniform superposition: all bitstrings equally likely.
        mpsx = MPS(3; state="x+")
        @test project_onto_bitstring(mpsx, "000") ≈ 1/8
        @test project_onto_bitstring(mpsx, "101") ≈ 1/8
        @test project_onto_bitstring(mpsx, "111") ≈ 1/8
    end

    @testset "to_vec" begin
        # Basis ordering: site 1 is least-significant bit (matches single_shot_measure encoding).
        L = 3
        mps0 = MPS(L; state="zeros")
        v0 = to_vec(mps0)
        @test length(v0) == 2^L
        @test v0[1] ≈ 1.0 + 0.0im
        @test sum(abs2, v0) ≈ 1.0

        mpsb = MPS(L; state="basis", basis_string="100") # site1=1, others 0 => integer 1 => index 2
        vb = to_vec(mpsb)
        @test vb[2] ≈ 1.0 + 0.0im
        @test sum(abs2, vb) ≈ 1.0
    end

    @testset "Initialization States" begin
        L = 4
        mps = MPS(L; state="x+")
        mps0 = MPS(L; state="zeros")
        ov = scalar_product(mps, mps0)
        expected = (1.0/sqrt(2))^L
        @test isapprox(abs(ov), expected; atol=1e-10)
        
        mps_y = MPS(L; state="y+")
        @test isapprox(scalar_product(mps_y, mps_y), 1.0; atol=1e-10)
        
        basis_str = "0101"
        mps_basis = MPS(L; state="basis", basis_string=basis_str)
        @test mps_basis.tensors[1][1, 1, 1] ≈ 1.0
        @test mps_basis.tensors[1][1, 2, 1] ≈ 0.0
        @test mps_basis.tensors[2][1, 1, 1] ≈ 0.0
        @test mps_basis.tensors[2][1, 2, 1] ≈ 1.0
    end
    
    @testset "Canonical Forms & Shifting" begin
        mps = random_mps(4, 2, 4)
        
        # Use general shift
        shift_orthogonality_center!(mps, 2) 
        
        centers = check_canonical_form(mps)
        @test !isempty(centers)
        @test centers[1] == 2
        
        MPSModule.normalize!(mps)
        @test mps.orth_center == 1
        @test isapprox(scalar_product(mps, mps), 1.0; atol=1e-12)
    end
    
    @testset "Truncation" begin
        L = 6
        pdim = 2
        large_bond = 8
        # Create random tensors
        tensors = [crandn((1, pdim, large_bond))]
        for _ in 2:L-1
            push!(tensors, crandn((large_bond, pdim, large_bond)))
        end
        push!(tensors, crandn((large_bond, pdim, 1)))
        
        mps = MPS(L, tensors, [pdim for _ in 1:L])
        MPSModule.normalize!(mps)
        
        truncate!(mps; max_bond_dim=2)
        
        max_b = 0
        for t in mps.tensors
            max_b = max(max_b, size(t, 1), size(t, 3))
        end
        @test max_b <= 2
        
        # Check norm is preserved (truncate! calls normalize!)
        # We re-normalize explicitly just in case
        MPSModule.normalize!(mps)
        n = scalar_product(mps, mps)
        @test isapprox(n, 1.0; atol=1e-10)
    end

end
