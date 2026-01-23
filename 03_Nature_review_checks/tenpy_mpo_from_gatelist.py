#!/usr/bin/env python3
"""
Run TenPy MPS simulation driven by a gate-list exported from Julia `CircuitLibrary.jl`.

Workflow:
1) Load gate-list CSV
2) Build a Qiskit QuantumCircuit from it (canonical instruction stream)
3) Iterate instructions:
   - on SAMPLE_OBSERVABLES barrier: measure <Z> and chi
   - otherwise: build MPO for the 1q/2q unitary and apply to MPS

Supports:
- compression_method="zip_up"
- compression_method="variational"
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import warnings
from pathlib import Path

from qiskit_gatelist import build_qiskit_circuit_from_gatelist, is_sample_barrier, read_gatelist_csv


def _add_tenpy_to_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tenpy_repo = repo_root / "external" / "tenpy"
    sys.path.insert(0, str(tenpy_repo))
    # TenPy will warn if optional Cython extensions are not built.
    # This is benign (just slower) and otherwise clutters output.
    warnings.filterwarnings("ignore", message=r"Couldn't load compiled cython code.*", category=UserWarning)


def _two_site_mpo_from_gate(sites, i: int, j: int, U4):
    """
    Build an MPO for a two-site operator U acting on sites i and j (0-based), identity elsewhere.
    Works also for long-range (i < j) by propagating bond index with identities.
    """
    import numpy as np
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mpo import MPO

    if i == j:
        raise ValueError("two-site gate requires i != j")
    if i > j:
        i, j = j, i

    d = 2
    U = np.asarray(U4, dtype=np.complex128).reshape(d * d, d * d)

    # Operator Schmidt decomposition: U = sum_k A_k ⊗ B_k, with k<=d^2=4
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

    return MPO(sites, Ws, bc="finite", IdL=[0] * (L + 1), IdR=[0] * (L + 1), mps_unit_cell_width=L)


def _apply_one_site_unitary_mps(psi, i: int, U2) -> None:
    """
    Apply a 1-site unitary directly to the MPS at site i (0-based), without constructing an MPO.
    """
    import numpy as np
    from tenpy.linalg import np_conserved as npc

    U = np.asarray(U2, dtype=np.complex128).reshape(2, 2)
    op = npc.Array.from_ndarray_trivial(U, labels=["p", "p*"])
    psi.apply_local_op(i, op, unitary=True, renormalize=False)


def _op_matrix(op):
    # Prefer `to_matrix` for gate-like operations
    if hasattr(op, "to_matrix"):
        try:
            return op.to_matrix()
        except Exception:
            pass
    # Fallback: Operator(op).data
    from qiskit.quantum_info import Operator

    return Operator(op).data


def main() -> None:
    _add_tenpy_to_syspath()

    import numpy as np
    from tenpy.networks.site import SpinHalfSite
    from tenpy.networks.mps import MPS

    p = argparse.ArgumentParser()
    p.add_argument("--gatelist", type=str, required=True, help="Gate-list CSV exported by Julia.")
    p.add_argument("--method", type=str, required=True, choices=["zip_up", "variational"])
    p.add_argument("--chi-max", type=int, default=256, help="Max MPS bond dimension.")
    # Truncation threshold as *relative discarded weight*: discard until sum(S_discarded^2)/sum(S^2) <= trunc.
    # This matches Julia's (normalized) convention in this comparison pipeline.
    # Set to 0.0 for "no truncation" (still limited by chi_max).
    p.add_argument("--trunc", type=float, default=1.0e-12, help="Relative discarded weight tolerance.")
    # zip-up specific
    p.add_argument("--m-temp", type=int, default=2, help="Zip-up temporary factor: trunc to m_temp*chi_max.")
    p.add_argument("--trunc-weight", type=float, default=1.0, help="Zip-up: relax svd_min during intermediate SVDs.")
    # variational specific
    # Note: with too few sweeps, variational MPO application may stop before convergence even without truncation.
    p.add_argument("--min-sweeps", type=int, default=2)
    p.add_argument("--max-sweeps", type=int, default=10)
    p.add_argument("--tol-theta-diff", type=float, default=1.0e-12)
    p.add_argument("--max-trunc-err", type=str, default="none")
    # common
    p.add_argument("--state", type=str, default="neel", choices=["up", "neel"])
    p.add_argument("--sites", type=str, default="1", help="Comma-separated 1-based sites for <Z> (e.g. '1,4,8').")
    p.add_argument("--outdir", type=str, default="03_Nature_review_checks/results")
    p.add_argument("--tag", type=str, default="tenpy")
    args = p.parse_args()

    gatelist = read_gatelist_csv(args.gatelist)
    qc = build_qiskit_circuit_from_gatelist(gatelist)
    L = int(qc.num_qubits)

    # TenPy sites + initial state
    site = SpinHalfSite(conserve=None, sort_charge=False)
    sites = [site] * L
    if args.state == "up":
        state = ["up"] * L
    else:
        state = (["up", "down"] * (L // 2))[:L]
    psi = MPS.from_product_state(sites, state, bc="finite", unit_cell_width=L)

    site_list = [int(s.strip()) for s in args.sites.split(",") if s.strip()]
    site_list = [s for s in site_list if 1 <= s <= L]
    if not site_list:
        raise ValueError("No valid --sites given.")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    obs_path = outdir / f"{args.tag}_obs.csv"
    chi_path = outdir / f"{args.tag}_chi.csv"
    timing_path = outdir / f"{args.tag}_timing.csv"

    max_trunc_err = args.max_trunc_err.strip().lower()
    max_trunc_err_val = None if max_trunc_err in ("none", "null", "") else float(max_trunc_err)

    trunc = float(args.trunc)
    if trunc < 0.0:
        raise ValueError("--trunc must be >= 0.")
    trunc_cut = None if trunc == 0.0 else math.sqrt(trunc)
    # zip_up internally relaxes `svd_min *= trunc_weight`, so it must be numeric.
    # Use an extremely small positive value to effectively disable the constraint without log(0).
    trunc_params = dict(chi_max=int(args.chi_max), svd_min=1.0e-300, trunc_cut=trunc_cut)

    if args.method == "zip_up":
        apply_opts = dict(
            compression_method="zip_up",
            trunc_params=trunc_params,
            m_temp=int(args.m_temp),
            trunc_weight=float(args.trunc_weight),
        )
    else:
        apply_opts = dict(
            compression_method="variational",
            trunc_params=trunc_params,
            min_sweeps=int(args.min_sweeps),
            max_sweeps=int(args.max_sweeps),
            tol_theta_diff=float(args.tol_theta_diff),
            combine=False,
            max_trunc_err=max_trunc_err_val,
        )

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

    # Buffer results; only write CSVs after timing
    obs_rows: list[list[float]] = []
    chi_rows: list[list[float]] = []
    step = -1

    print(f"[tenpy_{args.method}] L={L} gates={len(qc.data)} init={args.state} chi_max={args.chi_max}")

    t0 = time.time()
    trunc_total = 0.0
    for inst in qc.data:
        op = inst.operation
        if is_sample_barrier(op):
            step += 1
            obs_rows.append([float(step), *measure_Z()])
            chi_rows.append([float(x) for x in measure_chi_row(step)])
            continue

        # Only support 1q/2q unitary ops
        qidx = [qc.find_bit(q).index for q in inst.qubits]
        if len(qidx) == 1:
            U2 = _op_matrix(op)
            _apply_one_site_unitary_mps(psi, int(qidx[0]), U2)
        elif len(qidx) == 2:
            U4 = _op_matrix(op)
            mpo2 = _two_site_mpo_from_gate(sites, int(qidx[0]), int(qidx[1]), U4)
            trunc_err = mpo2.apply(psi, apply_opts)
            trunc_total += float(getattr(trunc_err, "eps", 0.0))
        else:
            raise ValueError(f"Unsupported gate arity {len(qidx)} for op {op.name}")

    wall = time.time() - t0

    # If no SAMPLE_OBSERVABLES barriers existed, still emit one measurement at "step 0".
    if step < 0:
        step = 0
        obs_rows.append([0.0, *measure_Z()])
        chi_rows.append([float(x) for x in measure_chi_row(0)])

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

    chi_max_final = int(np.max(np.asarray(psi.chi, dtype=int)))
    print(f"[tenpy_{args.method}] done: meas_points={step+1} chi_max={chi_max_final} wall_s_sim={wall:.3f}")
    print(f"[tenpy_{args.method}] trunc_err_total(eps)≈{trunc_total:.6g}")
    print(f"[tenpy_{args.method}] wrote: {obs_path}")
    print(f"[tenpy_{args.method}] wrote: {chi_path}")
    print(f"[tenpy_{args.method}] wrote: {timing_path}")


if __name__ == "__main__":
    main()

