using Test
using LinearAlgebra

using Yaqs
using Yaqs.GateLibrary
using Yaqs.MPSModule
using Yaqs.MPOModule

@testset "Single MPO×MPS: ZipUp vs variational" begin
    # This is a minimal debug test: apply ONE MPO to ONE MPS and compare
    # - ZipUp (reference)
    # - Variational MPO application
    #
    # We use a *long-range* 2-qubit gate MPO (sites 1 and L) since that is where
    # variational application tends to be more fragile.

    L = 8
    θ = 0.231
    s1, s2 = 1, L

    psi0 = MPS(L; state="Neel")
    MPSModule.normalize!(psi0)

    # Build an MPO for a 2-qubit unitary on (s1, s2), identity elsewhere.
    # (Bond rank is <= 4, but the MPO is long-range due to identities in-between.)
    U = Matrix{ComplexF64}(matrix(RxxGate(θ))) # 4x4
    W = mpo_from_two_qubit_gate_matrix(U, s1, s2, L; d=2)

    # Exact reference: contract without compression (should be feasible at these sizes).
    psi_exact = contract_mpo_mps(W, psi0)
    MPSModule.normalize!(psi_exact)

    # ZipUp reference: no truncation if chi_relax is sufficiently large and svd_min=0.
    psi_zip = deepcopy(psi0)
    apply_zipup!(psi_zip, W; chi_max=256, svd_min=0.0, m_temp=4, trunc_weight=1.0)
    MPSModule.normalize!(psi_zip)

    # Variational MPO apply: try to converge with "no truncation" (still chi-capped).
    psi_var = deepcopy(psi0)
    apply_variational!(psi_var, W;
                       chi_max=256,
                       trunc=0.0,
                       svd_min=eps(Float64),
                       min_sweeps=2,
                       max_sweeps=50,
                       tol_theta_diff=1e-14)
    MPSModule.normalize!(psi_var)

    # Compare via fidelity (phase-invariant).
    function fidelity(a::MPSModule.MPS, b::MPSModule.MPS)
        va = MPSModule.to_vec(a); vb = MPSModule.to_vec(b)
        va ./= norm(va); vb ./= norm(vb)
        return abs(dot(conj(va), vb))
    end

    f_zip_exact = fidelity(psi_zip, psi_exact)
    f_var_exact = fidelity(psi_var, psi_exact)
    f_var_zip   = fidelity(psi_var, psi_zip)

    @info "fidelity(zipup, exact)" f_zip_exact
    @info "fidelity(variational, exact)" f_var_exact
    @info "fidelity(variational, zipup)" f_var_zip

    # ZipUp should essentially match exact here.
    @test f_zip_exact > 1 - 1e-12

    # Variational should (in principle) converge close to ZipUp for a single unitary MPO.
    # If this fails, it's a focused repro to debug the variational algorithm.
    @test f_var_zip > 0.99

    # Also compare a simple diagnostic: local ⟨Z⟩ on a few sites.
    Zop = Matrix(matrix(ZGate()))
    function zsites(psi::MPSModule.MPS)
        tmp = deepcopy(psi)
        z = real.(MPSModule.evaluate_all_local_expectations(tmp, [Zop for _ in 1:L]))
        return z[[1, 4, 8]]
    end
    z_zip = zsites(psi_zip)
    z_var = zsites(psi_var)
    @info "Z sites (zipup)" z_zip
    @info "Z sites (variational)" z_var
    @test isapprox(z_var, z_zip; atol=1e-6, rtol=1e-6)
end

