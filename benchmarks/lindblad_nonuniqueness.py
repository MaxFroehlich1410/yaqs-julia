# """
# Demonstrate non-uniqueness of Pauli-jump Lindblad rates from ⟨Z(t)⟩ alone.

# For H = -g σx and collapse operators √γk σk (k ∈ {X,Y,Z}), the Bloch
# equations decouple: ⟨Z(t)⟩ depends only on the combinations
#     A = γX + γY   (transverse decay of Z-component)
#     B = γX + γZ   (enters via coherent mixing with X)
# so any two parameter triples sharing the same A and B yield identical ⟨Z(t)⟩.
# Measuring only Z therefore cannot uniquely determine (γX, γY, γZ).
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from qutip import basis, sigmax, sigmay, sigmaz, mesolve

# # --- Hamiltonian ---
# g = 1.0
# H = -g * sigmax()

# # --- Two parameter sets with identical A = γX+γY and B = γX+γZ ---
# params = {
#     "set 1": {"gX": 0.02, "gY": 0.08, "gZ": 0.05},  # A=0.10, B=0.07
#     "set 2": {"gX": 0.04, "gY": 0.06, "gZ": 0.03},  # A=0.10, B=0.07
# }

# # --- Initial state and time grid ---
# rho0 = basis(2, 0) * basis(2, 0).dag()
# tlist = np.linspace(0, 10, 400)

# # --- Solve for each parameter set ---
# results = {}
# for label, p in params.items():
#     c_ops = [
#         np.sqrt(p["gX"]) * sigmax(),
#         np.sqrt(p["gY"]) * sigmay(),
#         np.sqrt(p["gZ"]) * sigmaz(),
#     ]
#     res = mesolve(H, rho0, tlist, c_ops, e_ops=[sigmaz()])
#     results[label] = res.expect[0]

# # --- Diagnostics ---
# z1, z2 = results["set 1"], results["set 2"]
# diff = np.abs(z1 - z2)
# print(f"max |ΔZ|  = {diff.max():.2e}")
# print(f"L2  norm  = {np.sqrt(np.trapezoid(diff**2, tlist)):.2e}")

# # --- Plot ---
# fig, ax = plt.subplots(figsize=(7, 4))
# ax.plot(tlist, z1, label="set 1  (γX=0.02, γY=0.08, γZ=0.05)", linewidth=2)
# ax.plot(tlist, z2, "--", label="set 2  (γX=0.04, γY=0.06, γZ=0.03)", linewidth=2)
# ax.set_xlabel("t")
# ax.set_ylabel("⟨Z⟩(t)")
# ax.set_title("Same A = γX+γY, same B = γX+γZ  →  identical ⟨Z(t)⟩")
# ax.legend(fontsize=9)
# ax.grid(alpha=0.3)
# fig.tight_layout()
# plt.savefig("benchmarks/figures/lindblad_nonuniqueness.png", dpi=150)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor, qeye, sigmax, sigmay, sigmaz, basis, mesolve

def op_on_site(op, site, N):
    ops = [qeye(2)] * N
    ops[site] = op
    return tensor(ops)

def build_tfim_hamiltonian(N, J, g):
    sx, sz = sigmax(), sigmaz()
    H = 0
    # -J sum Z_i Z_{i+1}
    for i in range(N - 1):
        H += -J * (op_on_site(sz, i, N) * op_on_site(sz, i + 1, N))
    # -g sum X_i
    for i in range(N):
        H += -g * op_on_site(sx, i, N)
    return H

def build_collapse_ops(N, gX, gY, gZ):
    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    c_ops = []
    if gX > 0:
        for i in range(N):
            c_ops.append(np.sqrt(gX) * op_on_site(sx, i, N))
    if gY > 0:
        for i in range(N):
            c_ops.append(np.sqrt(gY) * op_on_site(sy, i, N))
    if gZ > 0:
        for i in range(N):
            c_ops.append(np.sqrt(gZ) * op_on_site(sz, i, N))
    return c_ops

# --- TFIM parameters ---
N = 4          # small, QuTiP scales as 2^N
J = 1.0
g = 1.0
H = build_tfim_hamiltonian(N, J, g)

# initial state |0...0>
psi0 = tensor([basis(2, 0)] * N)
rho0 = psi0 * psi0.dag()

tlist = np.linspace(0, 6, 301)

# Measure only Z expectations: either each site or average magnetization
Z_ops = [op_on_site(sigmaz(), i, N) for i in range(N)]
Zavg = sum(Z_ops) / N

# --- Two different parameter sets ---
# Keep SAME sums A=gX+gY and B=gX+gZ (as in the 1-qubit case)
p1 = dict(gX=0.02, gY=0.08, gZ=0.05)  # A=0.10, B=0.07
p2 = dict(gX=0.04, gY=0.06, gZ=0.03)  # A=0.10, B=0.07

def run(params):
    c_ops = build_collapse_ops(N, **params)
    res = mesolve(H, rho0, tlist, c_ops, e_ops=[Zavg])
    return np.array(res.expect[0])

z1 = run(p1)
z2 = run(p2)

diff = np.abs(z1 - z2)
print("Params 1:", p1)
print("Params 2:", p2)
print(f"max |Δ⟨Zavg⟩| = {diff.max():.3e}")
print(f"L2 difference = {np.sqrt(np.trapezoid(diff**2, tlist)):.3e}")

plt.figure(figsize=(7,4))
plt.plot(tlist, z1, label="set 1")
plt.plot(tlist, z2, "--", label="set 2")
plt.xlabel("t")
plt.ylabel("⟨Z⟩ average")
plt.title("TFIM: compare ⟨Z(t)⟩ for different (γX,γY,γZ)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
