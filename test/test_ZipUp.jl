using Test
using LinearAlgebra
using Yaqs
using Yaqs.MPOModule
using Yaqs.MPSModule
using Yaqs.GateLibrary
using PythonCall

@testset "Zip-up MPO×MPS application" begin
    @testset "Identity MPO leaves state invariant" begin
        L = 4
        psi0 = MPS(L; state="random")
        MPSModule.normalize!(psi0)
        v0 = to_vec(psi0)

        W = MPO(L; identity=true)
        psi = deepcopy(psi0)
        err = apply_zipup!(psi, W; chi_max=256, svd_min=0.0, m_temp=2, trunc_weight=1.0)
        @test err ≈ 0.0 atol=1e-10

        v1 = to_vec(psi)
        # Global phase differences should not occur for identity, but allow tiny roundoff.
        @test isapprox(v0, v1; atol=1e-10, rtol=1e-10)
    end

    @testset "Matches TenPy zip_up for a simple 2-qubit gate (no truncation)" begin
        # This is an integration test using vendored TenPy via PythonCall.
        # We choose parameters such that there is effectively no truncation.
        L = 4
        theta = 0.137

        # Julia side: build MPO for a single nearest-neighbor gate and apply zipup.
        psi_jl = MPS(L; state="Neel")
        MPSModule.normalize!(psi_jl)
        U = Matrix(matrix(RzzGate(theta))) # 4x4
        mpo_gate = Yaqs.DigitalTJM._mpo_from_two_qubit_gate_matrix(U, 2, 3, L; d=2)
        apply_zipup!(psi_jl, mpo_gate; chi_max=512, svd_min=1e-14, m_temp=2, trunc_weight=1.0)
        Zop = Matrix(matrix(ZGate()))
        z_jl = real.(MPSModule.evaluate_all_local_expectations(psi_jl, [Zop for _ in 1:L]))

        # Python/TenPy side: apply the same operator using TenPy's zip_up compression.
        # NOTE: TenPy depends on scipy; this test assumes the repo's CondaPkg env provides it.
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
    return MPO(sites, Ws, bc="finite", IdL=[0]*(L+1), IdR=[0]*(L+1), mps_unit_cell_width=L)

def run_zipup(L, U4):
    site = SpinHalfSite(conserve=None, sort_charge=False)
    sites = [site] * L
    state = (["up", "down"] * (L // 2))[:L]
    psi = MPS.from_product_state(sites, state, bc="finite", unit_cell_width=L)
    mpo = _two_site_mpo_from_gate(sites, 1, 2, U4)  # act on sites (2,3) in Julia => (1,2) 0-based
    opts = dict(
        compression_method="zip_up",
        trunc_params=dict(chi_max=512, svd_min=1e-14),
        m_temp=2,
        trunc_weight=1.0,
    )
    mpo.apply(psi, opts)
    sz = psi.expectation_value("Sz")
    z = (2.0 * np.real_if_close(sz)).astype(np.float64)
    return z
""", g)

        v_py = g["run_zipup"](L, U)
        z_py = pyconvert(Vector{Float64}, v_py)
        @test isapprox(z_jl, z_py; atol=5e-9, rtol=5e-9)
    end
end

