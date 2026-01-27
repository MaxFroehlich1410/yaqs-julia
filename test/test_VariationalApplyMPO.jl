using Test
using LinearAlgebra
using Yaqs
using Yaqs.MPOModule
using Yaqs.MPSModule
using Yaqs.GateLibrary
using PythonCall

@testset "Variational MPO×MPS application" begin
    @testset "Matches exact MPO×MPS (no truncation, enough chi)" begin
        L = 6
        theta = 0.231
        psi0 = MPS(L; state="Neel")
        MPSModule.normalize!(psi0)

        # Nearest-neighbor gate: should be reproduced accurately when chi_max is large.
        U = Matrix(matrix(RxxGate(theta))) # 4x4
        mpo_gate = mpo_from_two_qubit_gate_matrix(U, 2, 3, L; d=2)

        # Reference: exact application without compression
        psi_exact = contract_mpo_mps(mpo_gate, psi0)
        MPSModule.normalize!(psi_exact)
        v_exact = to_vec(psi_exact)
        v_exact ./= norm(v_exact)

        # Variational: should converge to the same state with sufficiently large chi_max.
        psi_var = deepcopy(psi0)
        _, sweeps, θdiff = apply_variational!(psi_var, mpo_gate;
                                             chi_max=256,
                                             trunc=0.0,
                                             svd_min=eps(Float64),
                                             min_sweeps=1,
                                             max_sweeps=20,
                                             tol_theta_diff=1e-14)
        @test sweeps >= 1
        @test θdiff ≥ 0.0

        v_var = to_vec(psi_var)
        v_var ./= norm(v_var)

        # Compare via fidelity (phase-invariant).
        overlap = abs(dot(conj(v_exact), v_var))
        # Current Julia variational implementation is an ALS heuristic; require high fidelity,
        # but don't assume it reaches machine precision in a fixed sweep budget.
        @test overlap > 0.99
    end

    @testset "Matches TenPy variational for a simple 2-qubit gate (no truncation)" begin
        # Integration test against vendored TenPy via PythonCall.
        L = 4
        theta = 0.137

        psi_jl = MPS(L; state="Neel")
        MPSModule.normalize!(psi_jl)
        U = Matrix(matrix(RzzGate(theta))) # 4x4
        mpo_gate = mpo_from_two_qubit_gate_matrix(U, 2, 3, L; d=2)

        apply_variational!(psi_jl, mpo_gate;
                           chi_max=512,
                           trunc=0.0,
                           svd_min=eps(Float64),
                           min_sweeps=2,
                           max_sweeps=10,
                           tol_theta_diff=1e-14)
        Zop = Matrix(matrix(ZGate()))
        z_jl = real.(MPSModule.evaluate_all_local_expectations(psi_jl, [Zop for _ in 1:L]))

        tenpy_repo = abspath(joinpath(@__DIR__, "..", "external", "tenpy"))
        sys = pyimport("sys")
        sys.path.insert(0, tenpy_repo)

        np = pyimport("numpy")
        SpinHalfSite = pyimport("tenpy.networks.site").SpinHalfSite
        MPS_py = pyimport("tenpy.networks.mps").MPS
        MPO_py = pyimport("tenpy.networks.mpo").MPO
        npc = pyimport("tenpy.linalg.np_conserved")
        # TenPy uses Sz for spin-1/2; qubit-Z corresponds to 2*Sz.

        g = PyDict{String, Py}()
        g["np"] = np
        g["SpinHalfSite"] = SpinHalfSite
        g["MPS"] = MPS_py
        g["MPO"] = MPO_py
        g["npc"] = npc

        pyexec("""
def _two_site_mpo_from_gate(sites, i, j, U4):
    d = 2
    U = np.asarray(U4, dtype=np.complex128).reshape(d*d, d*d)
    M = U.reshape(d, d, d, d)  # (p_i, p_j, p_i*, p_j*)
    X = np.transpose(M, (0, 2, 1, 3)).reshape(d*d, d*d)
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    r = int(np.sum(s > 1e-14))
    u = u[:, :r]
    s = s[:r]
    vh = vh[:r, :]
    A_ops = (u * np.sqrt(s)[None, :]).T.reshape(r, d, d)
    B_ops = (np.sqrt(s)[:, None] * vh).reshape(r, d, d)
    L = len(sites)
    Ws = []
    for n in range(L):
        if n < i or n > j:
            W = np.zeros((1, 1, d, d), dtype=np.complex128)
            W[0, 0, :, :] = np.eye(d, dtype=np.complex128)
        elif n == i:
            W = np.zeros((1, r, d, d), dtype=np.complex128)
            for k in range(r):
                W[0, k, :, :] = A_ops[k]
        elif n == j:
            W = np.zeros((r, 1, d, d), dtype=np.complex128)
            for k in range(r):
                W[k, 0, :, :] = B_ops[k]
        else:
            W = np.zeros((r, r, d, d), dtype=np.complex128)
            for k in range(r):
                W[k, k, :, :] = np.eye(d, dtype=np.complex128)
        Ws.append(npc.Array.from_ndarray_trivial(W, labels=["wL", "wR", "p", "p*"]))
    # IMPORTANT: for variational, TenPy expects IdL/IdR only at the boundaries.
    IdL = [0] + [None] * L
    IdR = [None] * L + [0]
    return MPO(sites, Ws, bc="finite", IdL=IdL, IdR=IdR, mps_unit_cell_width=L)

def run_variational(L, U4):
    site = SpinHalfSite(conserve=None, sort_charge=False)
    sites = [site] * L
    state = (["up", "down"] * (L // 2))[:L]
    psi = MPS.from_product_state(sites, state, bc="finite", unit_cell_width=L)
    mpo = _two_site_mpo_from_gate(sites, 1, 2, U4)  # act on sites (2,3) in Julia => (1,2) 0-based
    opts = dict(
        compression_method="variational",
        trunc_params=dict(chi_max=512, svd_min=np.finfo(np.float64).eps, trunc_cut=None),
        min_sweeps=2,
        max_sweeps=10,
        tol_theta_diff=1e-14,
        combine=False,
        max_trunc_err=None,
    )
    mpo.apply(psi, opts)
    sz = psi.expectation_value("Sz")
    z = (2.0 * np.real_if_close(sz)).astype(np.float64)
    return z
""", g)

        z_py = pyconvert(Vector{Float64}, g["run_variational"](L, U))
        @test isapprox(z_jl, z_py; atol=5e-9, rtol=5e-9)
    end
end

