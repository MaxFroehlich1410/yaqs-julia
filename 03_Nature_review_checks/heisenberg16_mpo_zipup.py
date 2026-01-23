#!/usr/bin/env python3
"""
Noise-free 16-qubit periodic Heisenberg "circuit" simulated with TenPy:
apply an MPO approximation of U = exp(-i dt H) repeatedly, compressing with the MPO zip-up method.

This uses TenPy's MPO application with:
  compression_method = "zip_up"

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


def main() -> None:
    _add_tenpy_to_syspath()

    import numpy as np
    from tenpy.models.spins import SpinModel
    from tenpy.networks.mps import MPS

    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--dt", type=float, default=0.05, help="Real time step Δt (U = exp(-i Δt H)).")
    p.add_argument("--steps", type=int, default=20, help="Number of circuit layers (applications of U).")
    p.add_argument("--chi-max", type=int, default=256, help="Max MPS bond dimension.")
    p.add_argument("--svd-min", type=float, default=1.0e-10, help="Discard singular values below this.")
    p.add_argument("--m-temp", type=int, default=2, help="Zip-up temporary factor: trunc to m_temp*chi_max.")
    p.add_argument(
        "--trunc-weight",
        type=float,
        default=1.0,
        help="Zip-up: relax svd_min to trunc_weight*svd_min during intermediate SVDs.",
    )
    p.add_argument("--state", type=str, default="neel", choices=["up", "neel"])
    p.add_argument(
        "--sites",
        type=str,
        default="1,8,16",
        help="Comma-separated 1-based sites for <Z> expectation values (e.g. '1,8,16').",
    )
    p.add_argument("--outdir", type=str, default="03_Nature_review_checks/results")
    p.add_argument("--tag", type=str, default="tenpy_zipup")
    args = p.parse_args()

    L = args.L
    dt_real = float(args.dt)
    dt = -1j * dt_real  # TenPy expects imaginary dt for real-time evolution

    model_params = dict(
        lattice="Chain",
        L=L,
        S=0.5,
        Jx=1.0,
        Jy=1.0,
        Jz=1.0,
        hx=0.0,
        hy=0.0,
        hz=0.0,
        bc_MPS="finite",
        bc_x="periodic",  # periodic couplings (finite MPS => long-range edge coupling)
        conserve="Sz",
        sort_charge=True,
    )
    M = SpinModel(model_params)

    sites = M.lat.mps_sites()
    if args.state == "up":
        state = ["up"] * L
    else:
        state = (["up", "down"] * (L // 2))[:L]
    psi = MPS.from_product_state(sites, state, bc="finite", unit_cell_width=M.lat.mps_unit_cell_width)

    E0 = float(np.real_if_close(M.H_MPO.expectation_value(psi)))
    print(f"[zip_up] L={L} periodic Heisenberg, init state={args.state}, E0={E0:.12g}")

    # Build the MPO for one layer of the "circuit".
    U_MPO = M.H_MPO.make_U_II(dt=dt)

    apply_opts = dict(
        compression_method="zip_up",
        trunc_params=dict(chi_max=int(args.chi_max), svd_min=float(args.svd_min)),
        m_temp=int(args.m_temp),
        trunc_weight=float(args.trunc_weight),
    )

    site_list = [int(s.strip()) for s in args.sites.split(",") if s.strip()]
    site_list = [s for s in site_list if 1 <= s <= L]
    if not site_list:
        raise ValueError("No valid --sites given.")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    obs_path = outdir / f"{args.tag}_obs.csv"
    chi_path = outdir / f"{args.tag}_chi.csv"

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

    # write CSV headers + initial measurement (step 0)
    with obs_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", *[f"Z_site{s}" for s in site_list]])
        w.writerow([0, *measure_Z()])
    with chi_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "chi_max", *[f"chi_bond{i}" for i in range(1, L)]])
        w.writerow(measure_chi_row(0))

    t0 = time.time()
    trunc_total = 0.0
    for n in range(args.steps):
        trunc_err = U_MPO.apply(psi, apply_opts)
        trunc_total += float(getattr(trunc_err, "eps", 0.0))
        step = n + 1
        with obs_path.open("a", newline="") as f:
            csv.writer(f).writerow([step, *measure_Z()])
        with chi_path.open("a", newline="") as f:
            csv.writer(f).writerow(measure_chi_row(step))
        if (n + 1) % max(1, args.steps // 5) == 0 or (n + 1) == args.steps:
            chi_max_now = int(np.max(psi.chi))
            E = float(np.real_if_close(M.H_MPO.expectation_value(psi)))
            print(f"[zip_up] step {n+1:4d}/{args.steps}: chi_max={chi_max_now:4d}  E={E:.12g}")
    wall = time.time() - t0

    E1 = float(np.real_if_close(M.H_MPO.expectation_value(psi)))
    chi_max_final = int(np.max(psi.chi))
    norm = float(np.real_if_close(psi.norm))
    print(f"[zip_up] done: E={E1:.12g}  norm={norm:.12g}  chi_max={chi_max_final}  wall_s={wall:.3f}")
    print(f"[zip_up] trunc_err_total(eps)≈{trunc_total:.6g}")
    print(f"[zip_up] wrote: {obs_path}")
    print(f"[zip_up] wrote: {chi_path}")


if __name__ == "__main__":
    main()

