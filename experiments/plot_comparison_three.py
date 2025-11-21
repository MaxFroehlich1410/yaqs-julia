import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import pandas as pd
import os

def run_qutip_and_plot():
    # --- Simulation Parameters (Matching Julia/Python L=6) ---
    L = 6
    J = 1.0
    h = 0.5
    gamma = 0.1
    dt = 0.05
    total_time = 2.0
    
    print(f"Running QuTiP Exact Simulation (L={L})...")

    # --- QuTiP Setup ---
    si = qt.qeye(2)
    sx = qt.sigmax()
    sz = qt.sigmaz()
    
    # YAQS 'raising' is [[0,0],[1,0]] (0->1 excitation).
    # In QuTiP basis (0=Up, 1=Down), this corresponds to sigmam (0->1).
    # sigmam = [[0,0],[1,0]].
    sp = qt.sigmam() 

    # Hamiltonian (Ising)
    # H = sum(-J Z_i Z_{i+1} - h X_i)
    H_qutip = 0
    for i in range(L):
        # -h X_i term
        op_list_x = [si] * L
        op_list_x[i] = sx
        H_qutip += -h * qt.tensor(op_list_x)

        # -J Z_i Z_{i+1} term
        if i < L - 1:
            op_list_zz = [si] * L
            op_list_zz[i] = sz
            op_list_zz[i+1] = sz
            H_qutip += -J * qt.tensor(op_list_zz)

    # Noise Operators (Raising noise on all sites)
    c_ops = []
    for i in range(L):
        op_list_noise = [si] * L
        op_list_noise[i] = sp 
        c_ops.append(np.sqrt(gamma) * qt.tensor(op_list_noise))

    # Initial State (|00...0>)
    initial_state_qutip = qt.tensor([qt.basis(2, 0)] * L)

    # Time points
    times = np.arange(0, total_time + dt, dt)

    # Expectation value operators (Z for first, middle, last qubit)
    # Indices: 0, L//2 -1, L-1
    # L=6 -> 0, 2, 5
    indices = [0, L//2 - 1, L-1]
    z_ops = []
    for i in indices:
        op_list_z = [si] * L
        op_list_z[i] = sz
        z_ops.append(qt.tensor(op_list_z))

    result_qutip = qt.mesolve(H_qutip, initial_state_qutip, times, c_ops, z_ops)
    print("QuTiP Simulation Finished.")

    # --- Load Benchmark Results ---
    julia_file = "experiments/julia_results_L6.csv"
    python_file = "experiments/python_results_L6.csv"
    
    if not os.path.exists(julia_file) or not os.path.exists(python_file):
        print("Error: Benchmark result files not found. Run benchmarks first.")
        return

    df_julia = pd.read_csv(julia_file)
    df_python = pd.read_csv(python_file)

    # --- Plotting ---
    plt.figure(figsize=(15, 10))
    
    labels = ["First", "Middle", "Last"]
    cols = ["Z_First", "Z_Middle", "Z_Last"]
    colors = ['blue', 'green', 'red']
    
    for i, col in enumerate(cols):
        c = colors[i]
        lbl = labels[i]
        
        # Julia (Solid)
        plt.plot(df_julia["Time"], df_julia[col], label=f"Julia Z_{lbl}", 
                 color=c, linestyle='-', linewidth=2.5, alpha=0.8)
        
        # Python (Dashed)
        plt.plot(df_python["Time"], df_python[col], label=f"Python Z_{lbl}", 
                 color=c, linestyle='--', linewidth=2.5, alpha=0.8)
        
        # QuTiP (Dotted + Markers)
        plt.plot(times, result_qutip.expect[i], label=f"QuTiP Z_{lbl}", 
                 color=c, linestyle=':', linewidth=2, marker='o', markersize=4, alpha=0.6)

    plt.xlabel("Time")
    plt.ylabel("Expectation Value <Z>")
    plt.title(f"Comparison: Julia TJM vs Python TJM vs QuTiP Exact\n(L={L}, Traj=200, Ising Chain)\nNoise: 0->1 Excitation (sigmam)")
    plt.legend(ncol=3)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_img = "experiments/comparison_L6_all.png"
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved to {output_img}")

if __name__ == "__main__":
    run_qutip_and_plot()
