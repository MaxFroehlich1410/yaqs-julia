import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import pandas as pd
import os

# 1. Parameters
L = 6
J = 1.0
h = 0.5
gamma = 0.1
dt = 0.05
t_total = 2.0
times = np.arange(0, t_total + 1e-9, dt) # Match Julia's collect(0:dt:total_time)

print("Running QuTiP exact simulation...")

# Operators
si_z = qt.sigmaz()
si_x = qt.sigmax()
si_plus = qt.sigmap() # Raising operator (creation op for spin)

sx_list = [qt.tensor([si_x if j == i else qt.qeye(2) for j in range(L)]) for i in range(L)]
sz_list = [qt.tensor([si_z if j == i else qt.qeye(2) for j in range(L)]) for i in range(L)]
sp_list = [qt.tensor([si_plus if j == i else qt.qeye(2) for j in range(L)]) for i in range(L)]

# Hamiltonian
# H = \sum -J Z_i Z_{i+1} - h X_i
# Note: YAQS Ising definition is usually -J ZZ - h X.
H = 0
for i in range(L - 1):
    H += -J * sz_list[i] * sz_list[i+1]
for i in range(L):
    H += -h * sx_list[i]

# Collapse Operators (Jump Operators)
# L_k = sqrt(gamma) * sigma_plus_k
c_ops = []
for i in range(L):
    c_ops.append(np.sqrt(gamma) * sp_list[i])

# Initial State: All Zeros (|0...0>)
# In QuTiP, basis(2,0) is |0> (eigenvalue +1 of Z usually, or excited state? QuTiP convention: |0> is [1,0], |1> is [0,1])
# Sigma Z = [[1, 0], [0, -1]]. So |0> is +1.
psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

# Evolve
result = qt.mesolve(H, psi0, times, c_ops, e_ops=sz_list)

# Extract exact data
sites_to_plot = [0, 2, 5] # 0-based indices corresponding to 1, 3, 6
labels = ["Site 1", "Site 3", "Site 6"]

# Load Julia Data
julia_csv_path = "experiments/julia_analog_results.csv"
if not os.path.exists(julia_csv_path):
    print(f"Error: {julia_csv_path} not found. Run the Julia script first.")
    exit(1)

julia_df = pd.read_csv(julia_csv_path)
julia_times = julia_df["Time"].values

# 3. Plot
plt.figure(figsize=(12, 8))

colors = ['b', 'g', 'r']

for idx, site in enumerate(sites_to_plot):
    label_site = labels[idx]
    color = colors[idx]
    
    # QuTiP
    plt.plot(times, result.expect[site], linestyle='--', color=color, label=f"QuTiP Exact {label_site}", linewidth=2)
    
    # Julia
    col_name = f"Z_{site+1}"
    if col_name in julia_df.columns:
        plt.plot(julia_times, julia_df[col_name], linestyle='-', color=color, label=f"Julia TJM {label_site}", linewidth=1.5, alpha=0.7)

plt.xlabel("Time")
plt.ylabel("Expectation Value <Z>")
plt.title(f"Analog TJM (Julia) vs QuTiP (Exact)\nIsing Chain L={L}, J={J}, h={h}, Raising Noise $\gamma$={gamma}, 500 Trajectories")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiments/comparison_plot.png")
print("Plot saved to experiments/comparison_plot.png")
