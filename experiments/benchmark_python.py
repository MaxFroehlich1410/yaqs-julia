import time
import os
import numpy as np
import pandas as pd

# Set environment variable for 7 workers (available_cpus - 1 logic in simulator)
os.environ["SLURM_CPUS_ON_NODE"] = "8"

try:
    from mqt.yaqs import simulator
    from mqt.yaqs.core.data_structures.networks import MPS, MPO
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
    from mqt.yaqs.core.libraries.gate_library import Z
except ImportError:
    print("Error: mqt.yaqs package not found. Please ensure it is installed.")
    exit(1)

def run_benchmark():
    # 1. Parameters
    L = 6
    J = 1.0
    h = 0.5
    dt = 0.05
    t_total = 1.0
    num_traj = 200  
    strength = 0.01

    print(f"Starting Python Analog TJM Benchmark (L=6)...")
    print(f"L={L}, J={J}, h={h}, dt={dt}, T={t_total}, Traj={num_traj}, Noise=Raising({strength})")
    print(f"Threads (Workers): 7 (via SLURM_CPUS_ON_NODE=8)")

    # 2. Initialize
    initial_state = MPS(L, state="zeros")

    # Hamiltonian
    H = MPO()
    H.init_ising(L, J, h)

    # Noise Model
    # Note: 'raising' in YAQS is [[0,0],[1,0]] (0->1 excitation)
    processes = [{"name": "raising", "sites": [i], "strength": strength} for i in range(L)]
    noise_model = NoiseModel(processes, num_qubits=L)

    # Observables: Z on first, middle, last
    # L=6 -> Julia: 1, 3, 6. Python: 0, 2, 5.
    
    sites_to_measure = [0, L//2 - 1, L-1]
    observables = [Observable(Z(), i) for i in sites_to_measure]

    # Config
    sim_params = AnalogSimParams(
        observables=observables,
        elapsed_time=t_total,
        dt=dt,
        num_traj=num_traj,
        max_bond_dim=32,
        threshold=1e-9,
        order=2,
        sample_timesteps=True
    )

    # 3. Run and Measure Time
    start_time = time.time()
    
    simulator.run(initial_state, H, sim_params, noise_model, parallel=True)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nPython Simulation Finished.")
    print(f"Elapsed Time: {elapsed:.4f} seconds")

    # 4. Save Results
    results_dict = {"Time": sim_params.times}
    labels = ["Z_First", "Z_Middle", "Z_Last"]
    
    for i, obs in enumerate(observables):
        results_dict[labels[i]] = obs.results

    df = pd.DataFrame(results_dict)
    output_file = "experiments/python_results_L6.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    run_benchmark()
