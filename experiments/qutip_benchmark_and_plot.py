import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import csv

def run_qutip_benchmark():
    # Parameters
    L = 6
    J = 1.0
    h = 0.5
    gamma = 0.1
    T_total = 2.0
    dt = 0.05
    times = np.arange(0, T_total + dt/1000, dt) # Ensure we cover [0, 2.0]
    
    print(f"Running QuTiP Benchmark: L={L}, J={J}, h={h}, gamma={gamma}")
    
    # Operators
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    si = qt.qeye(2)
    
    # Construct Hamiltonian
    # H = -J sum Z Z - h sum X
    H_list = []
    
    # Interaction terms
    for i in range(L-1):
        op_list = [si] * L
        op_list[i] = sz
        op_list[i+1] = sz
        H_list.append(-J * qt.tensor(op_list))
        
    # Field terms
    for i in range(L):
        op_list = [si] * L
        op_list[i] = sx
        H_list.append(-h * qt.tensor(op_list))
        
    H = sum(H_list)
    
    # Construct Jump Operators
    # Raising operators (sigma_plus) on every site
    # sigma_plus = (sx + i sy) / 2 = Qobj([[0, 1], [0, 0]])
    sp = qt.create(2) # In QuTiP create(2) is [[0,0],[1,0]] (Lowering? No wait)
    # create(N) returns a^\dagger.
    # For N=2, basis is |0>, |1>. 
    # create(2) * |0> = |1>. create(2) * |1> = 0.
    # If |0> is ground state (0), |1> is excited state (1).
    # Then create is raising.
    # "raising" usually means sigma_+ = |0><1| ? No, |1><0|.
    # In QuTiP: basis(2,0) is |0>=(1,0)^T (up/excited usually in qubit? No, usually ground).
    # Wait, QuTiP convention: basis(2, 0) = |0>, basis(2, 1) = |1>.
    # sigmaz = diag(1, -1). So |0> is +1 (Up), |1> is -1 (Down).
    # J*Z*Z favors antiparallel? No, -J*Z*Z favors parallel (both +1 or both -1) -> Ferromagnetic.
    # Raising operator should flip -1 to +1. (|1> -> |0>).
    # Lowering should flip +1 to -1. (|0> -> |1>).
    
    # In Yaqs/Julia:
    # RaisingGate = [[0, 1], [0, 0]]. 
    # Z = [[1, 0], [0, -1]].
    # |0> = [1, 0] (Eigenval +1). |1> = [0, 1] (Eigenval -1).
    # Raising * |1> = [1, 0] = |0>. Correct.
    
    # In QuTiP:
    # sigmap() = [[0, 1], [0, 0]]. Matches.
    c_ops = []
    for i in range(L):
        op_list = [si] * L
        op_list[i] = qt.sigmap()
        c_ops.append(np.sqrt(gamma) * qt.tensor(op_list))
        
    # Initial State
    # All Zeros (|000...0>).
    # basis(2, 0) is |0>.
    psi0_list = [qt.basis(2, 0) for _ in range(L)]
    psi0 = qt.tensor(psi0_list)
    
    # Observables
    # Z1, Z3, Z6 (indices 0, 2, 5)
    e_ops = []
    obs_indices = [0, 2, 5]
    for idx in obs_indices:
        op_list = [si] * L
        op_list[idx] = sz
        e_ops.append(qt.tensor(op_list))
        
    # Solve Master Equation
    result = qt.mesolve(H, psi0, times, c_ops, e_ops)
    
    # Save Results
    # Time, Z1, Z3, Z6
    data = np.column_stack((times, np.array(result.expect).T))
    
    filename = "experiments/qutip_results.csv"
    np.savetxt(filename, data, delimiter=",", header="Time,Z1,Z3,Z6", comments="")
    print(f"Saved QuTiP results to {filename}")

def plot_comparison():
    # Load Data
    try:
        tjm_data = np.loadtxt("experiments/tjm_results.csv", delimiter=",")
        qutip_data = np.loadtxt("experiments/qutip_results.csv", delimiter=",", skiprows=1)
    except OSError:
        print("Error: Could not find results files. Run simulations first.")
        return

    times_tjm = tjm_data[:, 0]
    z1_tjm = tjm_data[:, 1]
    z3_tjm = tjm_data[:, 2]
    z6_tjm = tjm_data[:, 3]
    
    times_qt = qutip_data[:, 0]
    z1_qt = qutip_data[:, 1]
    z3_qt = qutip_data[:, 2]
    z6_qt = qutip_data[:, 3]
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Z1
    plt.plot(times_qt, z1_qt, 'k-', label='QuTiP Z1')
    plt.plot(times_tjm, z1_tjm, 'r--', label='TJM Z1')
    
    # Z3
    plt.plot(times_qt, z3_qt, 'b-', label='QuTiP Z3')
    plt.plot(times_tjm, z3_tjm, 'c--', label='TJM Z3')
    
    # Z6
    plt.plot(times_qt, z6_qt, 'g-', label='QuTiP Z6')
    plt.plot(times_tjm, z6_tjm, 'y--', label='TJM Z6')
    
    plt.xlabel("Time")
    plt.ylabel("<Z>")
    plt.title(f"TJM vs QuTiP Benchmark (Ising L=6, J=1, h=0.5, \u03b3=0.1 Raising)")
    plt.legend()
    plt.grid(True)
    
    plot_file = "experiments/benchmark_plot.png"
    plt.savefig(plot_file)
    print(f"Saved plot to {plot_file}")

if __name__ == "__main__":
    run_qutip_benchmark()
    plot_comparison()

