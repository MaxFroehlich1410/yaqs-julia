"""
Exact Schrödinger simulation of the Haldane–Shastry (HS) spin-1/2 chain
using full eigendecomposition for fast, exact time evolution.

Hamiltonian (periodic boundary conditions, ring of N sites):

    H = J · Σ_{i<j} J_{ij} (S_i · S_j)

where   S_α = σ_α / 2   (spin-1/2 operators)

and     J_{ij} = (π/N)² / sin²(π(j−i)/N)   (HS chord-distance coupling)

Strategy:
    1. Build H as a dense NumPy matrix using QuTiP tensor products.
    2. Diagonalize once:  H = V diag(E) V†   (O(dim³))
    3. For each snapshot t:  ψ(t) = V · (e^{-iEt} ⊙ (V† ψ₀))  (O(dim²))

Observable – connected spin correlator:

    C_zz(x; t) = ⟨ψ(t)| S_{j+x}^z S_j^z |ψ(t)⟩
               − ⟨ψ(t)| S_{j+x}^z |ψ(t)⟩  ⟨ψ(t)| S_j^z |ψ(t)⟩

    (translation-invariant for PBC → independent of j; we fix j = N//2)

Operator convention:
    S_i^z = (1/2) σ_i^z   (diagonal; eigenvalues ±1/2)

Initial state: |ψ₀⟩ = |↑↑…↑⟩ = |0…0⟩  (all spins up, σ_z = +1)

Output:
    - CSV  benchmarks/results/exact_hs_czz.csv
    - PNG  benchmarks/figures/exact_hs_czz.png
"""

import os
import time
import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import qutip as qt
    _HAS_QUTIP = True
except ImportError:
    _HAS_QUTIP = False

# ── Parameters ────────────────────────────────────────────────────────────────
N  = 10          # system size (keep ≤ 12 for fast exact simulation)
J  = 1.0         # overall coupling scale
T  = 5.0         # total evolution time
dt = 0.05        # time step
j_ref = N // 2   # reference site for C_zz  (0-indexed in Python/QuTiP)
PBC   = True     # True = canonical HS; False = open 1/r² chain

# ── Spin operators ─────────────────────────────────────────────────────────────

def op_at(op_single, site, N):
    """Embed a single-site operator into the N-site Hilbert space."""
    ops = [qt.qeye(2)] * N
    ops[site] = op_single
    return qt.tensor(ops)


sx_half = qt.sigmax() / 2
sy_half = qt.sigmay() / 2
sz_half = qt.sigmaz() / 2

# ── HS coupling ────────────────────────────────────────────────────────────────

def hs_coupling(i, j, N, J=1.0, pbc=True):
    """
    J_{ij} for the HS model.
      PBC:  (π/N)² / sin²(π|i-j|/N)
      OBC:  1 / |i-j|²
    Sites i, j are 0-indexed.
    """
    dx = abs(i - j)
    if pbc:
        return J * (np.pi / N) ** 2 / np.sin(np.pi * dx / N) ** 2
    else:
        return J / dx ** 2

# ── Build Hamiltonian ──────────────────────────────────────────────────────────

print(f"Building HS Hamiltonian  N={N}  J={J}  PBC={PBC} …", end=" ", flush=True)
t0 = time.perf_counter()

dim = 2 ** N

if _HAS_QUTIP:
    H_qt = qt.Qobj(np.zeros((dim, dim), dtype=complex), dims=[[2] * N, [2] * N])
    for i in range(N):
        for j in range(i + 1, N):
            Jij = hs_coupling(i, j, N, J=J, pbc=PBC)
            H_qt += Jij * (
                op_at(sx_half, i, N) * op_at(sx_half, j, N)
                + op_at(sy_half, i, N) * op_at(sy_half, j, N)
                + op_at(sz_half, i, N) * op_at(sz_half, j, N)
            )
    H_np = H_qt.full()
else:
    # Pure NumPy fallback (no QuTiP)
    I2  = np.eye(2, dtype=complex)
    SX  = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    SY  = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    SZ  = np.diag([0.5, -0.5]).astype(complex)
    S_ops = [SX, SY, SZ]

    def _full_op(op, site, N):
        acc = np.array([[1.0 + 0j]])
        for k in range(N):
            acc = np.kron(acc, op if k == site else I2)
        return acc

    H_np = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        for j in range(i + 1, N):
            Jij = hs_coupling(i, j, N, J=J, pbc=PBC)
            for S in S_ops:
                term = np.array([[1.0 + 0j]])
                for k in range(N):
                    term = np.kron(term, S if k == i or k == j else I2)
                H_np += Jij * term

t_build = time.perf_counter() - t0
print(f"done ({t_build:.2f} s)   dim = {dim}")

# ── Eigendecomposition ──────────────────────────────────────────────────────────

print("Diagonalizing …", end=" ", flush=True)
t0 = time.perf_counter()
E_eig, V = scipy.linalg.eigh(H_np)   # H = V diag(E) V†, E real
Vdag = V.conj().T
t_diag = time.perf_counter() - t0
print(f"done ({t_diag:.2f} s)   E_gs = {E_eig[0]:.6f}")

# ── Initial state: single spin-flip at j_ref ──────────────────────────────────
# |ψ₀⟩ = |↑…↑ ↓_{j_ref} ↑…↑⟩  (NOT an eigenstate → non-trivial dynamics)
# Convention: site 0 = MSB (bit N-1), site N-1 = LSB (bit 0).
# Index of basis state with bit (N-1-j_ref) set: 2^(N-1-j_ref)
psi0 = np.zeros(dim, dtype=complex)
psi0[2 ** (N - 1 - j_ref)] = 1.0
print(f"Initial state: single spin-flip at site {j_ref} (0-indexed), index {2**(N-1-j_ref)}")

c0 = Vdag @ psi0   # projection onto eigenbasis

# ── Precompute diagonal S^z operators ──────────────────────────────────────────
# S^z_i = (1/2) σ^z_i is diagonal with eigenvalues ±1/2.
# For QuTiP/kron with site 0 = MSB:  bit (N-1-site) of basis index x.

def build_sz_diag(site, N):
    """Diagonal of S^z_site in the 2^N computational basis (float64)."""
    d = np.zeros(2 ** N)
    for x in range(2 ** N):
        bit = (x >> (N - 1 - site)) & 1   # 0 → ↑ → +1/2,  1 → ↓ → −1/2
        d[x] = 0.5 * (1 - 2 * bit)
    return d

sz_diags = [build_sz_diag(i, N) for i in range(N)]   # list of length N

# ── Time evolution & observable collection ────────────────────────────────────

tlist   = np.arange(0.0, T + dt * 0.5, dt)
n_times = len(tlist)
xs_plot = list(range(N))   # displacements x = 0..N-1

# C_zz[t_idx, x_idx]:  C_zz(j_ref + x, j_ref; t)
C_zz    = np.zeros((n_times, N))
sz_all  = np.zeros((n_times, N))      # ⟨S^z_i⟩(t)
energy  = np.zeros(n_times)

c0_sq   = np.abs(c0) ** 2
E_mean  = float(E_eig @ c0_sq)        # ⟨H⟩ = const (time-independent)

print(f"Evolving {n_times} snapshots (N={N}) …", end=" ", flush=True)
t0 = time.perf_counter()

for ti, t in enumerate(tlist):
    phase  = np.exp(-1j * E_eig * t)
    psi_t  = V @ (phase * c0)
    prob   = np.abs(psi_t) ** 2       # |⟨x|ψ(t)⟩|²

    energy[ti] = E_mean

    # Single-site expectations ⟨S^z_i⟩(t)
    for i in range(N):
        sz_all[ti, i] = float(sz_diags[i] @ prob)

    # Connected correlator C_zz(j_ref + x, j_ref; t)
    sz_j = sz_all[ti, j_ref]
    for xi, x in enumerate(xs_plot):
        i = (j_ref + x) % N if PBC else j_ref + x
        if 0 <= i < N:
            if i == j_ref:
                sz2      = float((sz_diags[j_ref] ** 2) @ prob)   # = 1/4 always
                C_zz[ti, xi] = sz2 - sz_j ** 2
            else:
                szsz         = float((sz_diags[i] * sz_diags[j_ref]) @ prob)
                C_zz[ti, xi] = szsz - sz_all[ti, i] * sz_j

t_evolve = time.perf_counter() - t0
print(f"done ({t_evolve:.3f} s)")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  HS chain  N={N}  J={J}  PBC={PBC}")
print(f"  dim = {dim},   E_gs = {E_eig[0]:.6f}")
print(f"  T = {T},  dt = {dt},  snapshots = {n_times}")
print(f"  Reference site j_ref = {j_ref} (0-indexed)")
print(f"  Build time  : {t_build:.3f} s")
print(f"  Diag time   : {t_diag:.3f} s")
print(f"  Evolve time : {t_evolve:.3f} s")
print(f"  ⟨H⟩ (const) : {E_mean:.6f}")
print(f"  max|C_zz|   : {np.max(np.abs(C_zz)):.4f}")
print(f"{'─'*60}\n")

# ── Save CSV ───────────────────────────────────────────────────────────────────
os.makedirs("benchmarks/results", exist_ok=True)
csv_path = "benchmarks/results/exact_hs_czz.csv"

with open(csv_path, "w") as f:
    # Header: t, x=0, x=1, ..., x=N-1
    header = "t," + ",".join(f"Czz_x{x}" for x in xs_plot)
    f.write(header + "\n")
    for ti, t in enumerate(tlist):
        row = f"{t:.6f}," + ",".join(f"{C_zz[ti, xi]:.8f}" for xi in range(N))
        f.write(row + "\n")

print(f"CSV saved → {csv_path}")

# ── Plots ──────────────────────────────────────────────────────────────────────
os.makedirs("benchmarks/figures", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (A) Heatmap  C_zz(t, x)
im = axes[0].imshow(
    C_zz.T,
    aspect="auto",
    origin="lower",
    extent=[tlist[0], tlist[-1], -0.5, N - 0.5],
    cmap="RdBu_r",
    vmin=-np.max(np.abs(C_zz)) * 0.8,
    vmax= np.max(np.abs(C_zz)) * 0.8,
)
cbar = plt.colorbar(im, ax=axes[0])
cbar.set_label(r"$C_{zz}(t,x)$")
axes[0].set_xlabel("Time $t$")
axes[0].set_ylabel("Displacement $x$")
axes[0].set_title(
    r"Exact HS $C_{zz}(t,x)$"
    f"\nN={N},  J={J},  PBC={PBC},  $j_{{ref}}$={j_ref}"
    r"   [$S^z = \sigma^z/2$, connected]"
)

# (B) Line cuts at selected times
n_cuts = min(6, n_times)
cut_indices = np.linspace(0, n_times - 1, n_cuts, dtype=int)
for ti_idx in cut_indices:
    axes[1].plot(xs_plot, C_zz[ti_idx, :], lw=1.5, label=f"t={tlist[ti_idx]:.2f}")
axes[1].axhline(0, color="gray", lw=0.7, ls="--")
axes[1].set_xlabel("Displacement $x$")
axes[1].set_ylabel(r"$C_{zz}(t,x)$")
axes[1].set_title("Line cuts at selected times")
axes[1].legend(fontsize=8)

fig.suptitle(
    f"Haldane–Shastry model  (exact eigendecomp)\n"
    f"N={N},  J={J},  PBC={PBC},  initial state |↑↑…↑⟩",
    fontsize=11,
)
plt.tight_layout()

fig_path = "benchmarks/figures/exact_hs_czz.png"
fig.savefig(fig_path, dpi=150)
print(f"Figure saved → {fig_path}")
plt.close(fig)
