import time
import matplotlib.pyplot as plt
import pandas as pd
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.methods.tdvp import two_site_tdvp

def run_benchmark():
    # --- Experiment Parameters ---
    L = 30
    J = 1.0
    g = 0.5
    t_max = 2.0
    dt = 0.05
    max_bond_dim = 32
    sites_to_measure = [0, 15, 29]
    
    # Derived parameters
    num_steps = int(round(t_max / dt))
    
    print(f"Starting Benchmark: Ising Chain L={L}, J={J}, g={g}")
    print(f"Time Evolution: t=0 to {t_max}, dt={dt} ({num_steps} steps)")
    print(f"Algorithm: 2-Site TDVP, Max Bond Dim={max_bond_dim}")
    
    # --- 1. Initialization ---
    # Hamiltonian
    H = MPO()
    H.init_ising(L, J, g)
    
    # Initial State |00...0>
    psi = MPS(L, state="zeros")
    
    # Observables
    observables = [Observable(Z(), sites=i) for i in sites_to_measure]
    
    # Simulation Parameters
    # Note: elapsed_time here is just used to init the object/times array, 
    # but we will control the loop manually.
    sim_params = AnalogSimParams(
        observables=observables,
        elapsed_time=t_max,
        dt=dt,
        max_bond_dim=max_bond_dim,
        threshold=1e-12, # Default/High precision
        sample_timesteps=True
    )
    
    # Storage for results
    # Structure: list of dicts or just lists
    results_data = []
    
    def measure_observables(mps, obs_list, sites):
        """Measure observables while maintaining canonical form."""
        results = {}
        # 2-Site TDVP leaves the orthogonality center at site 0.
        # We shift it to the right as we measure to ensure correct expectation values.
        current_center = 0 
        
        # Ensure sites are sorted to minimize shifting
        # (sites_to_measure is already sorted [0, 15, 29])
        
        for i, site in enumerate(sites):
            # Shift center to the target site
            while current_center < site:
                mps.shift_orthogonality_center_right(current_center)
                current_center += 1
            
            val = mps.expect(obs_list[i])
            results[f"Site_{site}"] = val
        return results

    # --- 2. Simulation Loop ---
    total_exec_time = 0.0
    
    # Measure at t=0
    current_time = 0.0
    step_results = {"Time": current_time}
    step_results.update(measure_observables(psi, observables, sites_to_measure))
    results_data.append(step_results)
    
    print("Running simulation...")
    
    for step in range(1, num_steps + 1):
        t0 = time.perf_counter()
        
        # Perform one step of 2-Site TDVP
        two_site_tdvp(psi, H, sim_params)
        
        t1 = time.perf_counter()
        step_time = t1 - t0
        total_exec_time += step_time
        
        current_time = step * dt
        
        # Measure (efficiently, minimize I/O)
        step_results = {"Time": current_time}
        step_results.update(measure_observables(psi, observables, sites_to_measure))
        results_data.append(step_results)
        
        # Optional: Progress print
        if step % 10 == 0:
            print(f"Step {step}/{num_steps} completed. Time={current_time:.2f}")

    # --- 3. Metrics ---
    avg_time_per_step = total_exec_time / num_steps
    print("\n--- Benchmark Results ---")
    print(f"Total Execution Time: {total_exec_time:.4f} s")
    print(f"Average Time per Step: {avg_time_per_step:.4f} s")
    
    # --- 4. Save Results ---
    # Convert to DataFrame for easy CSV writing
    # Flattening the data for the requested format: Time, Site, ExpVal?
    # User asked for: "Save the results (Time, Site, ExpVal) to a CSV file"
    # But plotting implies trajectories.
    # Usually "Time, Site, ExpVal" implies a long format or wide format.
    # Given "Generate a plot comparing the trajectories...", wide format (Time, Site_0, Site_15...) is easier for plotting.
    # But "Time, Site, ExpVal" sounds like columns.
    # Let's save in a format that is easy to interpret. Wide format is standard for time series.
    # However, strictly reading "Time, Site, ExpVal" suggests 3 columns.
    # I will create a DataFrame with columns: Time, Site, ExpVal
    
    long_data = []
    for row in results_data:
        t = row["Time"]
        for site in sites_to_measure:
            long_data.append({
                "Time": t,
                "Site": site,
                "ExpVal": row[f"Site_{site}"]
            })
            
    df = pd.DataFrame(long_data)
    csv_filename = "python_large_ising_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    
    # --- 5. Plotting ---
    plt.figure(figsize=(10, 6))
    
    # We can use the wide data from results_data for easier plotting or pivot the df
    times = [r["Time"] for r in results_data]
    for site in sites_to_measure:
        vals = [r[f"Site_{site}"] for r in results_data]
        plt.plot(times, vals, label=f"Site {site}")
        
    plt.xlabel("Time")
    plt.ylabel("Expectation Value <Z>")
    plt.title(f"Ising Chain Dynamics (L={L}, 2-Site TDVP)")
    plt.legend()
    plt.grid(True)
    plt.savefig("python_ising_benchmark_plot.png")
    print("Plot saved to python_ising_benchmark_plot.png")

if __name__ == "__main__":
    run_benchmark()

