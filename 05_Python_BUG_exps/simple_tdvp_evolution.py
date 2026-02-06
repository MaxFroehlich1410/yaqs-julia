#!/usr/bin/env python3
"""
Minimal MPS Hamiltonian evolution using mqt-yaqs 1TDVP and 2TDVP.

Standalone script: no dependency on exp_util.py or exp_runs.py.
Only needs: mqt-yaqs, numpy, matplotlib; qutip optional for exact reference.

Usage (from repo root or from 05_Python_BUG_exps):
  python3 05_Python_BUG_exps/simple_tdvp_evolution.py

Or: pip install -e ./mqt-yaqs  (or set PYTHONPATH to mqt-yaqs/src)
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Allow importing mqt.yaqs from repo
_repo = Path(__file__).resolve().parent.parent
_mqt_src = _repo / "mqt-yaqs" / "src"
if _mqt_src.exists() and str(_mqt_src) not in sys.path:
    sys.path.insert(0, str(_mqt_src))

import copy
import numpy as np
import matplotlib.pyplot as plt
import time

from mqt.yaqs.core.data_structures.networks import MPS, MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.methods.bug import bug_second_order, fixed_bug_second_order
from mqt.yaqs.core.methods.tdvp import single_site_tdvp, two_site_tdvp


def canonical_local_expect(psi: MPS, obs: Observable, site: int) -> float:
    """Compute <obs> with the orthogonality center at `site`.

    mqt-yaqs `local_expect` contracts only the local tensor overlap; it is
    only correct when the MPS is in mixed-canonical form with the center at
    the measured site. We enforce that here before measuring.
    """
    tmp = copy.deepcopy(psi)
    tmp.set_canonical_form(site)
    norm = tmp.norm()
    if norm == 0.0:
        return 0.0
    return float(np.real(tmp.local_expect(obs, site)) / norm)


class MPOWithFlip(MPO):
    """MPO subclass that adds flip_network() required by mqt-yaqs BUG second-order methods."""

    def __init__(self) -> None:
        super().__init__()
        self.flipped = False

    def flip_network(self) -> None:
        # MPO tensors: (phys_out, phys_in, left, right) -> swap left/right and reverse list
        new_tensors = [np.transpose(t, (0, 1, 3, 2)) for t in self.tensors]
        new_tensors.reverse()
        self.tensors = new_tensors
        self.flipped = not self.flipped


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TDVP/BUG evolution with optional QuTiP reference.")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--model", default="ising", choices=["ising", "tfim", "heisenberg"])
    parser.add_argument("--reference", default="qutip", choices=["qutip", "none"])
    parser.add_argument("--plot", default="true")
    parser.add_argument("--methods", default="DOUBLEFIXED,DOUBLEADAPTIVE,SINGLE_SITE_TDVP,TWO_SITE_TDVP")
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--site", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--initial_state", default="Neel")
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=0.5)
    parser.add_argument("--hx", type=float, default=0.0)
    parser.add_argument("--hy", type=float, default=0.0)
    parser.add_argument("--hz", type=float, default=0.0)
    parser.add_argument("--max_bond_dim", type=int, default=32)
    parser.add_argument("--fixed_max_bond_dim", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=1e-16)
    parser.add_argument("--numiter_lanczos", type=int, default=25)
    return parser.parse_args(argv)


def _next_experiment_outdir(basedir: Path) -> tuple[Path, int]:
    max_n = 0
    for p in basedir.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("experiment"):
            try:
                n = int(p.name.replace("experiment", ""))
            except ValueError:
                continue
            max_n = max(max_n, n)
    n_new = max_n + 1
    outdir = basedir / f"experiment{n_new}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir, n_new


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    L = args.L
    J = args.J
    g = args.g
    dt = args.dt
    steps = args.steps
    elapsed_time = steps * dt
    numiter_lanczos = args.numiter_lanczos
    max_bond_dim = args.max_bond_dim
    fixed_max_bond_dim = args.fixed_max_bond_dim or max_bond_dim
    threshold = args.threshold
    site = args.site if args.site is not None else L // 2
    initial_state = args.initial_state
    model = args.model
    hx = args.hx
    hy = args.hy
    hz = args.hz
    reference = args.reference
    plot = str(args.plot).lower() in ("true", "1", "yes", "y")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir, n_new = _next_experiment_outdir(Path(__file__).resolve().parent)
        print(f"[simple_tdvp] using auto outdir {outdir} (experiment {n_new})")

    H = MPOWithFlip()
    if model in ("ising", "tfim"):
        H.init_ising(L, J, g)
    elif model == "heisenberg":
        # init_heisenberg builds H = -J(XX+YY+ZZ) - hx X - hy Y - hz Z
        H.init_heisenberg(L, J, J, J, hz, hx=hx, hy=hy, hz=hz)
    else:
        raise ValueError(f"Unsupported model={model}")

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=elapsed_time,
        dt=dt,
        max_bond_dim=max_bond_dim,
        threshold=threshold,
        sample_timesteps=False,
    )
    z_obs = Observable(Z(), site)
    times = np.arange(0, steps * dt + 1e-12, dt)

    fixed_methods = ["SINGLE_SITE_TDVP", "DOUBLEFIXED"]
    adaptive_methods = ["TWO_SITE_TDVP", "DOUBLEADAPTIVE"]

    results: dict[str, np.ndarray] = {}
    bond_dims: dict[str, list[int]] = {}
    runtimes: dict[str, float] = {}

    def run_method(method: str, chi: int, track_bd: bool) -> tuple[np.ndarray, list[int] | None, float]:
        psi = MPS(L, state=initial_state, pad=chi if method in fixed_methods else None)
        zs = [canonical_local_expect(psi, z_obs, site)]
        bd = [psi.write_max_bond_dim()] if track_bd else None
        t0 = time.perf_counter()
        for _ in range(steps):
            if method == "SINGLE_SITE_TDVP":
                single_site_tdvp(psi, H, sim_params, numiter_lanczos=numiter_lanczos)
                psi.truncate(threshold, chi)
            elif method == "TWO_SITE_TDVP":
                two_site_tdvp(psi, H, sim_params, numiter_lanczos=numiter_lanczos, dynamic=False)
            elif method == "DOUBLEFIXED":
                fixed_bug_second_order(psi, H, sim_params, numiter_lanczos=numiter_lanczos)
                psi.truncate(threshold, chi)
            elif method == "DOUBLEADAPTIVE":
                bug_second_order(psi, H, sim_params, numiter_lanczos=numiter_lanczos)
            else:
                raise ValueError(f"Unknown method {method}")
            zs.append(canonical_local_expect(psi, z_obs, site))
            if track_bd and bd is not None:
                bd.append(psi.write_max_bond_dim())
        wall = time.perf_counter() - t0
        return np.array(zs), bd, wall

    for m in methods:
        chi = fixed_max_bond_dim if m in fixed_methods else max_bond_dim
        track_bd = m in adaptive_methods
        z_vals, bd, wall = run_method(m, chi, track_bd)
        results[m] = z_vals
        runtimes[m] = wall
        if track_bd and bd is not None:
            bond_dims[m] = bd

    expvals_exact = None
    if reference == "qutip":
        try:
            import qutip
            id2 = qutip.qeye(2)
            sx = qutip.sigmax()
            sz = qutip.sigmaz()
            sy = qutip.sigmay()
            if model in ("ising", "tfim"):
                H_dense = 0 * qutip.tensor([sz] * L)
                for i in range(L - 1):
                    op_list = [id2] * L
                    op_list[i] = sz
                    op_list[i + 1] = sz
                    H_dense = H_dense + (-J) * qutip.tensor(op_list)
                for i in range(L):
                    op_list = [id2] * L
                    op_list[i] = sx
                    H_dense = H_dense + (-g) * qutip.tensor(op_list)
            else:
                # Match MPO convention: H = -J(XX+YY+ZZ) - hx X - hy Y - hz Z
                H_dense = 0 * qutip.tensor([sz] * L)
                for i in range(L - 1):
                    if J != 0:
                        op_list = [id2] * L
                        op_list[i], op_list[i + 1] = sx, sx
                        H_dense = H_dense + (-J) * qutip.tensor(op_list)
                        op_list = [id2] * L
                        op_list[i], op_list[i + 1] = sy, sy
                        H_dense = H_dense + (-J) * qutip.tensor(op_list)
                        op_list = [id2] * L
                        op_list[i], op_list[i + 1] = sz, sz
                        H_dense = H_dense + (-J) * qutip.tensor(op_list)
                for i in range(L):
                    if hx != 0:
                        op_list = [id2] * L
                        op_list[i] = sx
                        H_dense = H_dense + (-hx) * qutip.tensor(op_list)
                    if hy != 0:
                        op_list = [id2] * L
                        op_list[i] = sy
                        H_dense = H_dense + (-hy) * qutip.tensor(op_list)
                    if hz != 0:
                        op_list = [id2] * L
                        op_list[i] = sz
                        H_dense = H_dense + (-hz) * qutip.tensor(op_list)

            if initial_state == "zeros":
                psi0 = qutip.tensor([qutip.basis(2, 0) for _ in range(L)])
            elif initial_state == "ones":
                psi0 = qutip.tensor([qutip.basis(2, 1) for _ in range(L)])
            elif initial_state == "x+":
                plus = (qutip.basis(2, 0) + qutip.basis(2, 1)) / np.sqrt(2)
                psi0 = qutip.tensor([plus for _ in range(L)])
            elif initial_state == "Neel":
                psi0 = qutip.tensor([
                    qutip.basis(2, 1) if i % 2 == 0 else qutip.basis(2, 0)
                    for i in range(L)
                ])
            else:
                raise ValueError(f"Unsupported initial_state for qutip: {initial_state}")

            z_site_op = qutip.tensor([sz if i == site else id2 for i in range(L)])
            result = qutip.sesolve(
                H_dense, psi0, times.tolist(), [z_site_op],
                options=qutip.Options(store_states=False),
            )
            expvals_exact = np.real(np.array(result.expect[0]))
        except ImportError:
            expvals_exact = None
    elif reference != "none":
        raise ValueError(f"Unsupported reference={reference}")

    if plot:
        nrows = 4 if expvals_exact is not None else 3
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 14 if nrows == 4 else 11))
        ax = axes[0]
        if expvals_exact is not None:
            ax.plot(times, expvals_exact, "k-", linewidth=2, alpha=0.85, label="exact (QuTiP)")
        for m in fixed_methods:
            if m in results:
                ax.plot(times, results[m], label=f"{m} ({runtimes[m]:.2f} s)")
        ax.set_xlabel("Time")
        ax.set_ylabel("⟨Z⟩")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax2 = axes[1]
        if expvals_exact is not None:
            ax2.plot(times, expvals_exact, "k-", linewidth=2, alpha=0.85, label="exact (QuTiP)")
        for m in adaptive_methods:
            if m in results:
                ax2.plot(times, results[m], label=f"{m} ({runtimes[m]:.2f} s)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("⟨Z⟩")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax_bd = axes[2]
        for m in adaptive_methods:
            if m in bond_dims:
                ax_bd.plot(times, bond_dims[m], label=f"{m} ({runtimes[m]:.2f} s)")
        ax_bd.set_xlabel("Time")
        ax_bd.set_ylabel("max χ")
        ax_bd.grid(True, alpha=0.3)
        ax_bd.legend()

        if expvals_exact is not None:
            ax_err = axes[3]
            for m in methods:
                if m in results:
                    err = (results[m] - expvals_exact) ** 2
                    ax_err.plot(times, err, label=f"{m} ({runtimes[m]:.2f} s)")
            ax_err.set_xlabel("Time")
            ax_err.set_ylabel("|Δ⟨Z⟩|²")
            ax_err.set_yscale("log")
            ax_err.grid(True, alpha=0.3)
            ax_err.legend()

        fig.tight_layout()
        out = outdir / "simple_tdvp_evolution.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"[simple_tdvp] wrote {out}")

    csvfile = outdir / "timeseries.csv"
    keys_sorted = sorted(results.keys())
    with open(csvfile, "w", encoding="utf-8") as f:
        header = ["t"]
        if expvals_exact is not None:
            header.append("z_ref_qutip")
        header.extend(keys_sorted)
        f.write(",".join(header) + "\n")
        for i, t in enumerate(times):
            row = [f"{t:.16g}"]
            if expvals_exact is not None:
                row.append(f"{expvals_exact[i]:.16g}")
            for k in keys_sorted:
                row.append(f"{results[k][i]:.16g}")
            f.write(",".join(row) + "\n")

    metafile = outdir / "meta.txt"
    with open(metafile, "w", encoding="utf-8") as f:
        f.write(f"tag={args.tag}\n")
        f.write(f"model={model}\n")
        f.write(f"reference={reference}\n")
        f.write(f"L={L}\n")
        f.write(f"site={site}\n")
        f.write(f"dt={dt}\n")
        f.write(f"steps={steps}\n")
        f.write(f"initial_state={initial_state}\n")
        f.write(f"max_bond_dim={max_bond_dim}\n")
        f.write(f"fixed_max_bond_dim={fixed_max_bond_dim}\n")
        f.write(f"threshold={threshold}\n")
        f.write(f"numiter_lanczos={numiter_lanczos}\n")
        f.write(f"J={J}\n")
        f.write(f"g={g}\n")
        f.write(f"hx={hx}\n")
        f.write(f"hy={hy}\n")
        f.write(f"hz={hz}\n\n")
        f.write("[runtimes_s]\n")
        for k in sorted(runtimes.keys()):
            f.write(f"{k}={runtimes[k]:.6f}\n")


if __name__ == "__main__":
    main()
