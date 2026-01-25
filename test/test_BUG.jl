using Test
using Yaqs

const BUG = Yaqs.BUGModule
const MPSMod = Yaqs.MPSModule
const MPOMod = Yaqs.MPOModule
const Algo = Yaqs.Algorithms
const Cfg = Yaqs.SimulationConfigs

@testset "BUG basic API" begin
    L = 4
    psi = MPSMod.MPS(L; state="zeros")
    H = MPOMod.init_ising(L, 1.0, 0.7)
    cfg = Cfg.TimeEvolutionConfig(Cfg.Observable[], 0.1; dt=0.05, max_bond_dim=128, truncation_threshold=1e-12)

    # Smoke tests: construction and single call should not error.
    psi1 = deepcopy(psi)
    BUG.bug!(psi1, H, cfg; numiter_lanczos=10)
    @test MPSMod.check_if_valid_mps(psi1)
    @test abs(MPSMod.norm(psi1) - 1.0) < 1e-8

    psi2 = deepcopy(psi)
    BUG.fixed_bug!(psi2, H, cfg; numiter_lanczos=10)
    @test MPSMod.check_if_valid_mps(psi2)
    @test abs(MPSMod.norm(psi2) - 1.0) < 1e-8

    psi3 = deepcopy(psi)
    BUG.bug_second_order!(psi3, H, cfg; numiter_lanczos=10)
    @test MPSMod.check_if_valid_mps(psi3)
    @test abs(MPSMod.norm(psi3) - 1.0) < 1e-8

    psi4 = deepcopy(psi)
    BUG.fixed_bug_second_order!(psi4, H, cfg; numiter_lanczos=10)
    @test MPSMod.check_if_valid_mps(psi4)
    @test isfinite(MPSMod.norm(psi4)) && (MPSMod.norm(psi4) > 0)

    psi5 = deepcopy(psi)
    BUG.hybrid_bug_second_order!(psi5, H, cfg; numiter_lanczos=10)
    @test MPSMod.check_if_valid_mps(psi5)
    @test isfinite(MPSMod.norm(psi5)) && (MPSMod.norm(psi5) > 0)
end

@testset "BUG vs TDVP accuracy (small system, 1 step)" begin
    L = 5
    psi0 = MPSMod.MPS(L; state="x+")
    H = MPOMod.init_ising(L, 1.0, 0.3)

    # Keep truncation loose enough to not dominate, but bounded.
    dt = 1e-3
    cfg = Cfg.TimeEvolutionConfig(Cfg.Observable[], dt; dt=dt, max_bond_dim=256, truncation_threshold=1e-14)

    # TDVP (Julia implementation is already 2nd order symmetric for TimeEvolutionConfig).
    psi_tdvp = deepcopy(psi0)
    Algo.single_site_tdvp!(psi_tdvp, H, cfg)
    MPSMod.truncate!(psi_tdvp; threshold=cfg.truncation_threshold, max_bond_dim=cfg.max_bond_dim)

    # BUG 2nd order
    psi_bug = deepcopy(psi0)
    BUG.bug_second_order!(psi_bug, H, cfg; numiter_lanczos=25)

    # Compare full state vectors (small L only).
    v_tdvp = MPSMod.to_vec(psi_tdvp)
    v_bug = MPSMod.to_vec(psi_bug)

    rel = norm(v_bug - v_tdvp) / max(norm(v_tdvp), 1e-15)
    @test rel < 5e-3

    # Also compare a simple observable: <Z_3>
    z = Yaqs.GateLibrary.matrix(Yaqs.GateLibrary.ZGate())
    ez_tdvp = real(MPSMod.local_expect(deepcopy(psi_tdvp), z, 3))
    ez_bug = real(MPSMod.local_expect(deepcopy(psi_bug), z, 3))
    @test abs(ez_bug - ez_tdvp) < 5e-3
end

