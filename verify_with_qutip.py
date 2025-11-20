import qutip as qt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load Julia Results
try:
    df = pd.read_csv("tdvp_benchmark_results.csv")
except FileNotFoundError:
    print("Error: tdvp_benchmark_results.csv not found.")
    sys.exit(1)

L = 6
times = np.linspace(0, 2.0, 41)
sites = [1, 3, 6] # 1-based indices from Julia (1, L/2=3, L=6)

def run_qutip(model_name):
    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)]) # |00...0>
    
    if model_name == "Ising_Ferro":
        J = 1.0
        g = 0.5
        H = 0
        for i in range(L-1):
            op = [qt.qeye(2)] * L
            op[i] = qt.sigmaz()
            op[i+1] = qt.sigmaz()
            H += -J * qt.tensor(op)
        for i in range(L):
            op = [qt.qeye(2)] * L
            op[i] = qt.sigmax()
            H += -g * qt.tensor(op)
            
    elif model_name == "Ising_Disordered":
        J_list = [0.6, 1.2, 0.8, 1.4, 0.9]
        g_list = [0.7, 1.3, 0.5, 1.1, 0.8, 1.2]
        H = 0
        for i in range(L-1):
            op = [qt.qeye(2)] * L
            op[i] = qt.sigmaz()
            op[i+1] = qt.sigmaz()
            H += -J_list[i] * qt.tensor(op)
        for i in range(L):
            op = [qt.qeye(2)] * L
            op[i] = qt.sigmax()
            H += -g_list[i] * qt.tensor(op)
            
    elif model_name == "Heisenberg":
        J = 1.0
        h = 0.5
        H = 0
        for i in range(L-1):
            opx = [qt.qeye(2)] * L; opx[i]=qt.sigmax(); opx[i+1]=qt.sigmax()
            opy = [qt.qeye(2)] * L; opy[i]=qt.sigmay(); opy[i+1]=qt.sigmay()
            opz = [qt.qeye(2)] * L; opz[i]=qt.sigmaz(); opz[i+1]=qt.sigmaz()
            H += -J * (qt.tensor(opx) + qt.tensor(opy) + qt.tensor(opz))
        for i in range(L):
            op = [qt.qeye(2)] * L
            op[i] = qt.sigmaz()
            H += -h * qt.tensor(op)
            
    else:
        return None

    result = qt.sesolve(H, psi0, times, [])
    
    # Measure <Z> at sites
    data = {s: [] for s in sites}
    for state in result.states:
        for s in sites:
            idx = s - 1 # 0-based
            op = [qt.qeye(2)] * L
            op[idx] = qt.sigmaz()
            val = qt.expect(qt.tensor(op), state)
            data[s].append(val)
            
    return data

# Plotting
models = ["Ising_Ferro", "Ising_Disordered", "Heisenberg"]

for model in models:
    print(f"Running Qutip for {model}...")
    qutip_data = run_qutip(model)
    
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"TDVP Verification: {model} (Julia vs Qutip)", fontsize=16)
    
    df_model = df[df["Model"] == model]
    
    for i, site in enumerate(sites):
        plt.subplot(1, 3, i+1)
        plt.title(f"Site {site} <Z>")
        
        # Qutip Result
        plt.plot(times, qutip_data[site], 'k-', label="Qutip Exact", linewidth=3.0, alpha=0.3)
        
        # Julia Results
        for method, color in [("1TDVP", "r"), ("2TDVP", "b")]:
            data = df_model[(df_model["Site"] == site) & (df_model["Method"] == method)]
            plt.plot(data["Time"], data["ExpVal"], f'{color}--', label=f"Julia {method}", linewidth=1.5, alpha=1.0)
        
        plt.xlabel("Time")
        plt.ylabel("<Z>")
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()
            
    plt.tight_layout()
    plt.savefig(f"verify_{model}.png")
    print(f"Saved verify_{model}.png")

print("Done.")

