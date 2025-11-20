import qutip as qt
import numpy as np

L = 6
J = 1.0
g = 0.5

# H = sum(-J Z Z - g X)
H = 0
for i in range(L-1):
    op = [qt.qeye(2)]*L
    op[i] = qt.sigmaz()
    op[i+1] = qt.sigmaz()
    H += -J * qt.tensor(op)
for i in range(L):
    op = [qt.qeye(2)]*L
    op[i] = qt.sigmax()
    H += -g * qt.tensor(op)

psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

times = np.linspace(0, 1.0, 21) # 0, 0.05, ... 1.0
res = qt.sesolve(H, psi0, times, [])

print("Qutip Results:")
for i, t in enumerate(times):
    st = res.states[i]
    # Energy
    E = qt.expect(H, st)
    # Z1
    op_z1 = [qt.qeye(2)]*L
    op_z1[0] = qt.sigmaz()
    z1 = qt.expect(qt.tensor(op_z1), st)
    print(f"T={t:.2f}, E={E:.4f}, <Z1>={z1:.4f}")

