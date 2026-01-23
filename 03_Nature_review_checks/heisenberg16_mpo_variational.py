#!/usr/bin/env python3
"""
Noise-free (default: 8-qubit) periodic Heisenberg "circuit" simulated with TenPy:
apply the same Trotter gate circuit as `create_heisenberg_circuit` (Julia), but by
representing each gate as an MPO and applying it with the variational MPO ansatz.

This uses TenPy's MPO application with:
  compression_method = "variational"

TenPy is expected at: <repo_root>/external/tenpy (cloned repo).
"""

# pylint: disable=import-error

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def _add_tenpy_to_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tenpy_repo = repo_root / "external" / "tenpy"
    sys.path.insert(0, str(tenpy_repo))


def _rxx(theta: float):
    import numpy as np

    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    return np.array(
        [
            [c, 0, 0, s],
            [0, c, s, 0],
            [0, s, c, 0],
            [s, 0, 0, c],
        ],
        dtype=np.complex128,
    )


def _ryy(theta: float):
    import numpy as np

    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    return np.array(
        [
            [c, 0, 0, -s],
            [0, c, s, 0],
            [0, s, c, 0],
            [-s, 0, 0, c],
        ],
        dtype=np.complex128,
    )


def _rzz(theta: float):
    import numpy as np

    e_m = np.exp(-1j * theta / 2.0)
    e_p = np.exp(1j * theta / 2.0)
    return np.diag([e_m, e_p, e_p, e_m]).astype(np.complex128)


def _rz(theta: float):
    import numpy as np

    return np.diag([np.exp(-1j * theta / 2.0), np.exp(1j * theta / 2.0)]).astype(np.complex128)


def _two_site_mpo_from_gate(sites, i: int, j: int, U4):
    import numpy as np
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mpo import MPO

    if i == j:
        raise ValueError("two-site gate requires i != j")
    if i > j:
        i, j = j, i

    d = 2
    U = np.asarray(U4, dtype=np.complex128).reshape(d * d, d * d)

    M = U.reshape(d, d, d, d)  # (p_i, p_j, p_i*, p_j*)
    X = np.transpose(M, (0, 2, 1, 3)).reshape(d * d, d * d)  # (p_i,p_i*) x (p_j,p_j*)
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

    L = len(sites)
    return MPO(sites, Ws, bc="finite", IdL=[0] * (L + 1), IdR=[0] * (L + 1), mps_unit_cell_width=L)


def _one_site_mpo_from_gate(sites, i: int, U2):
    import numpy as np
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mpo import MPO

    d = 2
    L = len(sites)
    Ws = []
    for n in range(L):
        W = np.zeros((1, 1, d, d), dtype=np.complex128)
        if n == i:
            W[0, 0, :, :] = np.asarray(U2, dtype=np.complex128).reshape(d, d)
        else:
            W[0, 0, :, :] = np.eye(d, dtype=np.complex128)
        Ws.append(npc.Array.from_ndarray_trivial(W, labels=["wL", "wR", "p", "p*"]))
    L = len(sites)
    return MPO(sites, Ws, bc="finite", IdL=[0] * (L + 1), IdR=[0] * (L + 1), mps_unit_cell_width=L)


def main() -> None:
    _add_tenpy_to_syspath()

    import numpy as np
    from tenpy.networks.site import SpinHalfSite
    from tenpy.networks.mps import MPS

    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.05, help="Trotter step size (matches Julia circuit).")
    p.add_argument("--steps", type=int, default=20, help="Number of Trotter steps (matches Julia circuit).")
    p.add_argument("--chi-max", type=int, default=256, help="Max MPS bond dimension.")
    p.add_argument("--svd-min", type=float, default=1.0e-10, help="Discard singular values below this.")
    p.add_argument("--min-sweeps", type=int, default=1)
    p.add_argument("--max-sweeps", type=int, default=2)
    p.add_argument("--tol-theta-diff", type=float, default=1.0e-8)
    p.add_argument(
        "--max-trunc-err",
        type=str,
        default="none",
        help="TenPy safety threshold for variational MPO application. Use 'none' to warn instead of error.",
    )
    p.add_argument("--state", type=str, default="neel", choices=["up", "neel"])
    p.add_argument("--periodic", type=str, default="true")
    p.add_argument(
        "--sites",
        type=str,
        default="1,4,8",
        help="Comma-separated 1-based sites for <Z> expectation values (e.g. '1,4,8').",
    )
    p.add_argument("--outdir", type=str, default="03_Nature_review_checks/results")
    p.add_argument("--tag", type=str, default="tenpy_variational")
    args = p.parse_args()

    L = args.L
    dt = float(args.dt)
    site = SpinHalfSite(conserve=None, sort_charge=False)
    sites = [site] * L
    if args.state == "up":
        state = ["up"] * L
    else:
        state = (["up", "down"] * (L // 2))[:L]
    psi = MPS.from_product_state(sites, state, bc="finite", unit_cell_width=L)

    print(f"[variational] L={L} Heisenberg trotter-circuit, init state={args.state}, dt={dt}, steps={args.steps}")
    print(f"[variational] sweeps={args.min_sweeps}..{args.max_sweeps}")

    max_trunc_err = args.max_trunc_err.strip().lower()
    max_trunc_err_val = None if max_trunc_err in ("none", "null", "") else float(max_trunc_err)

    apply_opts = dict(
        compression_method="variational",
        trunc_params=dict(chi_max=int(args.chi_max), svd_min=float(args.svd_min)),
        min_sweeps=int(args.min_sweeps),
        max_sweeps=int(args.max_sweeps),
        tol_theta_diff=float(args.tol_theta_diff),
        combine=False,
        max_trunc_err=max_trunc_err_val,
    )

    site_list = [int(s.strip()) for s in args.sites.split(",") if s.strip()]
    site_list = [s for s in site_list if 1 <= s <= L]
    if not site_list:
        raise ValueError("No valid --sites given.")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    obs_path = outdir / f"{args.tag}_obs.csv"
    chi_path = outdir / f"{args.tag}_chi.csv"
    timing_path = outdir / f"{args.tag}_timing.csv"

    def measure_Z() -> list[float]:
        # TenPy SpinModel uses Sz; map to qubit-Z via Z = 2*Sz for spin-1/2
        sz = psi.expectation_value("Sz")  # length L, float/complex
        z = 2.0 * np.real_if_close(sz)
        return [float(z[s - 1]) for s in site_list]

    def measure_chi_row(step: int) -> list[int]:
        chi = np.asarray(psi.chi, dtype=int)  # length L+1, with boundaries
        chi_internal = chi[1:-1]  # length L-1
        chi_max_now = int(np.max(chi))
        return [step, chi_max_now, *[int(x) for x in chi_internal.tolist()]]

    # Buffer results; write CSVs after timing to avoid I/O in the timed section.
    obs_rows: list[list[float]] = [[0.0, *measure_Z()]]
    chi_rows: list[list[float]] = [[float(x) for x in measure_chi_row(0)]]

    txx = -2.0 * dt * 1.0
    tyy = -2.0 * dt * 1.0
    tzz = -2.0 * dt * 1.0
    tz = -2.0 * dt * 0.0
    periodic = str(args.periodic).lower() in ("1", "true", "yes", "y")

    t0 = time.time()
    trunc_total = 0.0
    for n in range(args.steps):
        if tz != 0.0:
            U2 = _rz(tz)
            for q in range(L):
                mpo1 = _one_site_mpo_from_gate(sites, q, U2)
                trunc_err = mpo1.apply(psi, apply_opts)
                trunc_total += float(getattr(trunc_err, "eps", 0.0))

        # RZZ
        U4 = _rzz(tzz)
        for i in range(0, L - 1, 2):
            mpo2 = _two_site_mpo_from_gate(sites, i, i + 1, U4)
            trunc_err = mpo2.apply(psi, apply_opts)
            trunc_total += float(getattr(trunc_err, "eps", 0.0))
        for i in range(1, L - 1, 2):
            mpo2 = _two_site_mpo_from_gate(sites, i, i + 1, U4)
            trunc_err = mpo2.apply(psi, apply_opts)
            trunc_total += float(getattr(trunc_err, "eps", 0.0))
        if periodic and L > 1:
            mpo2 = _two_site_mpo_from_gate(sites, 0, L - 1, U4)
            trunc_err = mpo2.apply(psi, apply_opts)
            trunc_total += float(getattr(trunc_err, "eps", 0.0))

        # RXX
        U4 = _rxx(txx)
        for i in range(0, L - 1, 2):
            mpo2 = _two_site_mpo_from_gate(sites, i, i + 1, U4)
            trunc_err = mpo2.apply(psi, apply_opts)
            trunc_total += float(getattr(trunc_err, "eps", 0.0))
        for i in range(1, L - 1, 2):
            mpo2 = _two_site_mpo_from_gate(sites, i, i + 1, U4)
            trunc_err = mpo2.apply(psi, apply_opts)
            trunc_total += float(getattr(trunc_err, "eps", 0.0))
        if periodic and L > 1:
            mpo2 = _two_site_mpo_from_gate(sites, 0, L - 1, U4)
            trunc_err = mpo2.apply(psi, apply_opts)
            trunc_total += float(getattr(trunc_err, "eps", 0.0))

        # RYY
        U4 = _ryy(tyy)
        for i in range(0, L - 1, 2):
            mpo2 = _two_site_mpo_from_gate(sites, i, i + 1, U4)
            trunc_err = mpo2.apply(psi, apply_opts)
            trunc_total += float(getattr(trunc_err, "eps", 0.0))
        for i in range(1, L - 1, 2):
            mpo2 = _two_site_mpo_from_gate(sites, i, i + 1, U4)
            trunc_err = mpo2.apply(psi, apply_opts)
            trunc_total += float(getattr(trunc_err, "eps", 0.0))
        if periodic and L > 1:
            mpo2 = _two_site_mpo_from_gate(sites, 0, L - 1, U4)
            trunc_err = mpo2.apply(psi, apply_opts)
            trunc_total += float(getattr(trunc_err, "eps", 0.0))

        step = n + 1
        obs_rows.append([float(step), *measure_Z()])
        chi_rows.append([float(x) for x in measure_chi_row(step)])
        if (n + 1) % max(1, args.steps // 5) == 0 or (n + 1) == args.steps:
            chi_max_now = int(np.max(psi.chi))
            print(f"[variational] step {n+1:4d}/{args.steps}: chi_max={chi_max_now:4d}")
    wall = time.time() - t0

    # Write outputs (not timed)
    with obs_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", *[f"Z_site{s}" for s in site_list]])
        w.writerows(obs_rows)
    with chi_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "chi_max", *[f"chi_bond{i}" for i in range(1, L)]])
        w.writerows(chi_rows)
    with timing_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall_s_sim"])
        w.writerow([wall])

    chi_max_final = int(np.max(psi.chi))
    print(f"[variational] done: chi_max={chi_max_final}  wall_s_sim={wall:.3f}")
    print(f"[variational] trunc_err_total(eps)≈{trunc_total:.6g}")
    print(f"[variational] wrote: {obs_path}")
    print(f"[variational] wrote: {chi_path}")
    print(f"[variational] wrote: {timing_path}")


if __name__ == "__main__":
    main()

