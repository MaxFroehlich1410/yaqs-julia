"""
Exact Schrödinger equation simulation of the 1D Transverse-Field Ising Model (TFIM)
using full eigendecomposition (scipy.linalg.eigh) for fast exact time evolution.

Strategy:
    1. Build H as a sparse operator with QuTiP, then extract the dense numpy matrix.
    2. Diagonalize once: H = V diag(E) V†  (O(d³), d = 2^N)
    3. For each snapshot time t:  ψ(t) = V · (e^{-iEt} ⊙ (V† ψ₀))  (O(d²))
    This avoids the ODE solver entirely — it's exact to floating-point precision.

Hamiltonian:
    H = -J Σ_i Z_i Z_{i+1}  -  g Σ_i X_i    (open boundary conditions)

Parameters:
    N  = 12 qubits
    J  = 1.0, g = 1.0
    T  = 2.0,  dt = 0.01  →  201 snapshots

Initial state: |↑↑…↑⟩ = |0⟩^⊗N
Observables:   ⟨Z_{N/2}⟩, ⟨H⟩, Loschmidt echo |⟨ψ(0)|ψ(t)⟩|²
"""

import time
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import qutip as qt

# ── Parameters ───────────────────────────────────────────────────────────────
N  = 12
J  = 1.0
g  = 1.0
T  = 2.0
dt = 0.01
PBC = False

# ── Build single-site operators embedded in N-qubit space ────────────────────
def op_at(op_single, site, N):
    ops = [qt.qeye(2)] * N
    ops[site] = op_single
    return qt.tensor(ops)

sx = qt.sigmax()
sz = qt.sigmaz()

# ── Build Hamiltonian (QuTiP) ─────────────────────────────────────────────────
print(f"Building TFIM Hamiltonian  N={N}  J={J}  g={g}  PBC={PBC} … ",
      end="", flush=True)
t0 = time.perf_counter()

dim = 2**N
H_qt = qt.Qobj(np.zeros((dim, dim)), dims=[[2]*N, [2]*N])

bonds = list(range(N - 1))
if PBC:
    bonds.append(N - 1)

for i in bonds:
    j = (i + 1) % N
    H_qt += -J * op_at(sz, i, N) * op_at(sz, j, N)

for i in range(N):
    H_qt += -g * op_at(sx, i, N)

t_build = time.perf_counter() - t0
print(f"done ({t_build:.2f} s)   dim = {dim}")

# ── Extract dense numpy matrix & diagonalize ─────────────────────────────────
print("Diagonalizing … ", end="", flush=True)
t0 = time.perf_counter()

H_np = H_qt.full()                          # complex128, shape (dim, dim)
E, V = scipy.linalg.eigh(H_np)              # H = V diag(E) V†, real E
Vdag = V.conj().T                           # precompute V†

t_diag = time.perf_counter() - t0
print(f"done ({t_diag:.2f} s)   E_gs = {E[0]:.6f}")

# ── Initial state vector ──────────────────────────────────────────────────────
psi0 = np.zeros(dim, dtype=complex)
psi0[0] = 1.0          # |0…0⟩ = |↑↑…↑⟩  (QuTiP's |0⟩ = spin-up for σz)

c0 = Vdag @ psi0       # projection onto eigenbasis  (dim,)

# ── Single-site Z operator for the middle site ────────────────────────────────
mid = N // 2
Z_mid_np = op_at(sz, mid, N).full()         # (dim, dim), diagonal

# For fast expectation: diag(Z_mid) is ±1, precompute it
z_diag = np.real(np.diag(Z_mid_np))         # (dim,) with values ±1

# ── Time-evolve snapshot by snapshot ─────────────────────────────────────────
tlist = np.arange(0.0, T + dt * 0.5, dt)
n_steps = len(tlist)

z_mid  = np.empty(n_steps)
energy = np.empty(n_steps)
echo   = np.empty(n_steps)

print(f"Evolving {n_steps} snapshots … ", end="", flush=True)
t0 = time.perf_counter()

for k, t in enumerate(tlist):
    # ψ(t) in eigenbasis: amplitudes ct_k = e^{-i E_k t} c0_k
    phase = np.exp(-1j * E * t)
    ct = phase * c0                          # (dim,) eigenbasis amplitudes

    # ψ(t) in computational basis
    psi_t = V @ ct                           # (dim,)

    prob = np.abs(psi_t)**2                  # |ψ_i|²  (real, sums to 1)

    # ⟨Z_mid⟩ = Σ_i z_diag[i] |ψ_i|²
    z_mid[k]  = z_diag @ prob

    # ⟨H⟩ = Σ_k E_k |ct_k|²  (diagonal in eigenbasis)
    energy[k] = E @ np.abs(ct)**2

    # Loschmidt echo: |⟨ψ(0)|ψ(t)⟩|² = |Σ_k |c0_k|² e^{-i E_k t}|²
    echo[k] = abs(np.dot(np.abs(c0)**2, phase))**2

t_evolve = time.perf_counter() - t0
print(f"done ({t_evolve:.3f} s)")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*58}")
print(f"  N={N}  J={J}  g={g}  T={T}  dt={dt}  PBC={PBC}")
print(f"  Hilbert-space dim       : {dim}")
print(f"  Hamiltonian build time  : {t_build:.3f} s")
print(f"  Diagonalization time    : {t_diag:.3f} s")
print(f"  Time evolution ({n_steps:3d} snaps): {t_evolve:.3f} s")
print(f"  Total wall time         : {t_build+t_diag+t_evolve:.3f} s")
print(f"  ──")
print(f"  ⟨Z_mid⟩ at t=0         : {z_mid[0]:.6f}  (expect +1.0)")
print(f"  ⟨Z_mid⟩ at t=T         : {z_mid[-1]:.6f}")
print(f"  ⟨H⟩     at t=0         : {energy[0]:.6f}")
print(f"  ⟨H⟩     at t=T         : {energy[-1]:.6f}  (should be const)")
print(f"  energy drift |δE|       : {abs(energy[-1]-energy[0]):.2e}  (machine eps)")
print(f"  Loschmidt echo at t=T   : {echo[-1]:.6f}")
print(f"{'─'*58}\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

axes[0].plot(tlist, z_mid, lw=1.5, color="steelblue")
axes[0].set_ylabel(r"$\langle Z_{\mathrm{mid}} \rangle$")
axes[0].set_title(
    f"TFIM exact Schrödinger (eigendecomp) — "
    f"N={N}, J={J}, g={g}, dt={dt}, T={T}"
)
axes[0].axhline(0, color="gray", lw=0.7, ls="--")

axes[1].plot(tlist, energy, lw=1.5, color="firebrick")
axes[1].set_ylabel(r"$\langle H \rangle$")

axes[2].plot(tlist, echo, lw=1.5, color="seagreen")
axes[2].set_ylabel(r"Loschmidt echo $|\langle\psi(0)|\psi(t)\rangle|^2$")
axes[2].set_xlabel("Time $t$")
axes[2].set_ylim(0, 1.05)

plt.tight_layout()
import os; os.makedirs("figures", exist_ok=True)
out = "figures/exact_schrodinger_ising.png"
fig.savefig(out, dpi=150)
print(f"Figure saved → {out}")
plt.show()
