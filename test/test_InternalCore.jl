using Test
using LinearAlgebra
using PythonCall
using Yaqs

@testset "Internal core helpers" begin
    @testset "Type hierarchy / method overrides" begin
        @test Yaqs.MPOModule.AbstractTensorNetwork isa DataType
        @test Yaqs.MPOModule.MPO <: Yaqs.MPOModule.AbstractTensorNetwork

        mps = Yaqs.MPSModule.MPS(2; state="zeros")
        @test LinearAlgebra.norm(mps) ≈ 1.0
    end

    @testset "Timing internals" begin
        ts = Yaqs.Timing.TimingStats()
        Yaqs.Timing._timing_add!(ts, :k, UInt64(1))
        io = IOBuffer()
        redirect_stdout(io) do
            Yaqs.Timing._print_timing_summary(ts; header="hdr", top=5)
        end
        out = String(take!(io))
        @test occursin("hdr", out)
    end

    @testset "DigitalTJM sampling plan helpers" begin
        bm = Dict{Int, Vector{String}}(0 => ["SAMPLE_OBSERVABLES"], 2 => ["SAMPLE_OBSERVABLES"])
        @test Yaqs.DigitalTJM._has_sample_barrier(bm, 0) == true
        @test Yaqs.DigitalTJM._has_sample_barrier(bm, 1) == false
        sample_at_start, sample_after = Yaqs.DigitalTJM._sample_plan(bm, 3)
        @test sample_at_start == true
        @test sample_after == [2]
    end

    @testset "TDVP projector workspace helpers" begin
        using Yaqs.Algorithms

        # Toggle + query
        Algorithms.set_tdvp_projector_workspaces!(true)
        @test Algorithms.get_tdvp_projector_workspaces() == true
        Algorithms.set_tdvp_projector_workspaces!(false)
        @test Algorithms.get_tdvp_projector_workspaces() == false
        Algorithms.set_tdvp_projector_workspaces!(true)

        # Workspaces are thread-local and lazily created
        ws_site = Algorithms._get_project_site_ws(ComplexF64)
        ws_bond = Algorithms._get_project_bond_ws(ComplexF64)
        @test ws_site isa Algorithms._ProjectSiteWS{ComplexF64}
        @test ws_bond isa Algorithms._ProjectBondWS{ComplexF64}

        # Compare allocating vs workspace-backed projector application on tiny tensors
        L = ones(ComplexF64, 1, 1, 1)
        R = ones(ComplexF64, 1, 1, 1)
        W = reshape(Matrix(Yaqs.GateLibrary.matrix(Yaqs.GateLibrary.IdGate())), 1, 2, 2, 1)
        A = reshape(ComplexF64[1, 0], 1, 2, 1)

        alloc = Algorithms._site_op(Val(false), L, R, W)
        wsop = Algorithms._site_op(Val(true), L, R, W)
        @test wsop isa Algorithms._ProjectSiteOpC64
        @test alloc(A) ≈ wsop(A)

        C = ones(ComplexF64, 1, 1)
        allocb = Algorithms._bond_op(Val(false), L, R)
        wsopb = Algorithms._bond_op(Val(true), L, R)
        @test wsopb isa Algorithms._ProjectBondOpC64
        @test allocb(C) ≈ wsopb(C)
    end

    @testset "DigitalTJM internal gate application helpers" begin
        psi = Yaqs.MPSModule.MPS(2; state="zeros")
        gate = Yaqs.DigitalTJM.DigitalGate(Yaqs.GateLibrary.XGate(), [1], Yaqs.GateLibrary.generator(Yaqs.GateLibrary.XGate()))
        Yaqs.DigitalTJM.apply_single_qubit_gate!(psi, gate)
        z1 = real(Yaqs.MPSModule.local_expect(psi, Yaqs.GateLibrary.matrix(Yaqs.GateLibrary.ZGate()), 1))
        @test isapprox(z1, -1.0; atol=1e-10)

        psi2 = Yaqs.MPSModule.MPS(2; state="zeros")
        cfg = Yaqs.SimulationConfigs.TimeEvolutionConfig(Yaqs.SimulationConfigs.Observable[], 1.0; dt=1.0)
        Yaqs.DigitalTJM.apply_local_gate_exact!(psi2, Yaqs.GateLibrary.RzzGate(0.3), 1, 2, cfg)
        @test Yaqs.MPSModule.check_if_valid_mps(psi2)
    end

    @testset "CircuitIngestion internal convert_node_to_gate (python stubs)" begin
        py"""
class _Op:
    def __init__(self, name, params):
        self.name = name
        self.params = params

class _Qubit:
    def __init__(self, idx):
        self._index = idx

class _Node:
    def __init__(self, op, qargs):
        self.op = op
        self.qargs = qargs
"""
        Op = py"_Op"
        Qubit = py"_Qubit"
        Node = py"_Node"
        node = Node(Op("ry", [0.25]), [Qubit(0)])
        dg = Yaqs.CircuitIngestion.convert_node_to_gate(node)
        @test dg.op isa Yaqs.GateLibrary.RyGate
        @test dg.sites == [1]
    end
end

