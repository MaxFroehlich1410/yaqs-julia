import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import pandas as pd
import os

def run_qutip_and_plot():
    # --- Simulation Parameters (Matching Python L=6) ---
    L = 6
    J = 1.0
    h = 0.5
    gamma = 0.1
    dt = 0.05
    total_time = 2.0
    num_traj = 200
    
    print(f"Running QuTiP Exact Simulation (L={L})...")

    # --- QuTiP Setup ---
    si = qt.qeye(2)
    sx = qt.sigmax()
    sz = qt.sigmaz()
    
    # YAQS 'raising' is [[0,0],[1,0]] (0->1 excitation).
    # In QuTiP, sigmam = [[0,0],[1,0]] maps |0> -> |1>.
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

    # --- Load Python Results ---
    python_file = "experiments/python_results_L6.csv"
    
    if not os.path.exists(python_file):
        print(f"Error: Python results file not found at {python_file}")
        print("Please run: python3 experiments/benchmark_python.py")
        return

    df_python = pd.read_csv(python_file)

    # --- Plotting ---
    plt.figure(figsize=(15, 10))
    
    labels = ["First", "Middle", "Last"]
    cols = ["Z_First", "Z_Middle", "Z_Last"]
    colors = ['blue', 'green', 'red']
    
    for i, col in enumerate(cols):
        c = colors[i]
        lbl = labels[i]
        
        # Python (Dashed)
        plt.plot(df_python["Time"], df_python[col], label=f"Python TJM Z_{lbl}", 
                 color=c, linestyle='--', linewidth=2.5, alpha=0.8)
        
        # QuTiP (Solid + Markers)
        plt.plot(times, result_qutip.expect[i], label=f"QuTiP Exact Z_{lbl}", 
                 color=c, linestyle='-', linewidth=2, marker='o', markersize=4, alpha=0.6)

    plt.xlabel("Time")
    plt.ylabel("Expectation Value <Z>")
    plt.title(f"Python TJM vs QuTiP Exact\n(L={L}, Traj={num_traj}, Ising Chain, γ={gamma})")
    plt.legend(ncol=3, loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_img = "experiments/comparison_python_qutip.png"
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved to {output_img}")

if __name__ == "__main__":
    run_qutip_and_plot()

