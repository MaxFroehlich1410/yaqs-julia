import os
import subprocess
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np

def _format_float_short(value):
    """Format floats compactly for filenames (e.g., 0.1 -> 0p1)."""
    return f"{value:.4g}".replace('.', 'p')

def get_error_suffix(model):
    channels = set()
    if "X" in model.upper(): channels.add("X")
    if "Y" in model.upper(): channels.add("Y")
    if "Z" in model.upper(): channels.add("Z")
    return "".join(sorted(channels)) if channels else "None"

def plot_results(
    num_qubits, num_layers, tau, noise_strength,
    model, circuit_label, fixed_trajectories,
    plot_qubits
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "../results")
    
    tau_str = _format_float_short(tau)
    noise_str = _format_float_short(noise_strength)
    err_suffix = get_error_suffix(model)
    
    # Construct filename pattern based on parameters
    # Note: Python benchmark now produces modeDM because we enabled run_density_matrix
    # Julia benchmark produces modeLarge because it doesn't compute density matrix
    # We need to check for both or specifically construct them.
    
    base_name_py = (
        f"unraveling_eff_N{num_qubits}_L{num_layers}_tau{tau_str}_"
        f"noise{noise_str}_basis{circuit_label}_obsZ_modeDM_"
        f"traj{fixed_trajectories}"
    )
    # Check if Python file has error suffix
    if err_suffix and err_suffix != "None":
        base_name_py += f"_err{err_suffix}"
    base_name_py += "_results.pkl"

    # Julia usually follows modeLarge if run_density_matrix was false in Julia
    base_name_jl = (
        f"JULIA_unraveling_eff_N{num_qubits}_L{num_layers}_tau{tau_str}_"
        f"noise{noise_str}_basis{circuit_label}_obsZ_modeLarge_"
        f"traj{fixed_trajectories}"
    )
    if err_suffix and err_suffix != "None":
        base_name_jl += f"_err{err_suffix}"
    base_name_jl += "_results.pkl"
    
    py_path = os.path.join(results_dir, base_name_py)
    jl_path = os.path.join(results_dir, base_name_jl)
    
    # Fallback: If Python file with suffix not found, try without
    if not os.path.exists(py_path) and err_suffix != "None":
        base_name_py_fallback = (
            f"unraveling_eff_N{num_qubits}_L{num_layers}_tau{tau_str}_"
            f"noise{noise_str}_basis{circuit_label}_obsZ_modeDM_"
            f"traj{fixed_trajectories}_results.pkl"
        )
        py_path_fallback = os.path.join(results_dir, base_name_py_fallback)
        if os.path.exists(py_path_fallback):
            py_path = py_path_fallback
            print(f"Note: Using fallback Python file: {py_path}")

    data_map = {}
    
    if os.path.exists(py_path):
        try:
            with open(py_path, "rb") as f:
                # Python dict has method names as keys
                loaded = pickle.load(f)
                # usually {"Qiskit MPS": {...}}
                data_map.update(loaded)
        except Exception as e:
            print(f"Error loading Python results: {e}")
    else:
        print(f"Python result file not found: {py_path}")

    if os.path.exists(jl_path):
        try:
            with open(jl_path, "rb") as f:
                loaded = pickle.load(f)
                # usually {"Julia V2": {...}}
                data_map.update(loaded)
        except Exception as e:
            print(f"Error loading Julia results: {e}")
    else:
        print(f"Julia result file not found: {jl_path}")

    if not data_map:
        print("No data to plot.")
        return

    # Plotting - now with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    t_axis = np.arange(num_layers + 1)
    
    # Colors/Styles for known keys, generic for others
    # Set opacity to 50% for Julia and Qiskit MPS to distinguish from Exact
    styles = {
        "Qiskit MPS": {"color": "blue", "ls": "-", "marker": "o", "markevery": 5, "alpha": 0.5},
        "Julia V2":   {"color": "red",  "ls": "-", "marker": "o", "markevery": 5, "alpha": 0.5},
        "Exact":      {"color": "black", "ls": "--", "linewidth": 2.0, "alpha": 1.0}
    }
    
    # Store local expectation values for MSE calculation
    local_expvals_dict = {}
    
    for method_name, res in data_map.items():
        style = styles.get(method_name, {})
        
        # 1. Staggered Magnetization
        if "staggered_magnetization" in res:
            stag = np.array(res["staggered_magnetization"])
            # Fix length if needed (prepend 1.0 for t=0 if missing, or crop)
            # Assuming 1.0 is the initial staggered mag for Neel/Zero state + X setup
            # But best to trust the data length or pad loosely.
            if len(stag) == num_layers:
                stag = np.concatenate(([1.0], stag))
            
            # Crop if too long
            stag = stag[:len(t_axis)]
            
            ax1.plot(t_axis[:len(stag)], stag, label=method_name, **style)
        
        # 2. Local Expectation Values
        if "local_expvals" in res:
            # List of arrays or 2D array (time, qubits) or (qubits, time)?
            # Python script: list of (num_qubits,) arrays -> (time, qubits)
            # Julia script: list of (num_qubits,) arrays -> (time, qubits)
            loc = np.array(res["local_expvals"]) # Shape (T, N)
            local_expvals_dict[method_name] = loc
            
            for q in plot_qubits:
                if q < loc.shape[1]:
                    # Differentiate qubits by transparency or line style variation if multiple
                    q_style = style.copy()
                    if len(plot_qubits) > 1:
                        q_style["label"] = f"{method_name} Q{q}"
                    else:
                        q_style["label"] = f"{method_name} Q{q}"
                    
                    # Slight visual offset for different qubits if needed, or just plot
                    y_data = loc[:, q]
                    ax2.plot(t_axis[:len(y_data)], y_data, **q_style)

    # 3. MSE Plot: Compare Julia V2 and Qiskit MPS against Exact
    if "Exact" in local_expvals_dict:
        exact_expvals = local_expvals_dict["Exact"]  # Shape (T, N)
        
        for method_name in ["Julia V2", "Qiskit MPS"]:
            if method_name in local_expvals_dict:
                method_expvals = local_expvals_dict[method_name]  # Shape (T, N)
                
                # Calculate MSE per qubit per time step, then average over selected qubits
                # MSE = mean((method - exact)^2) over selected qubits
                mse_per_time = []
                
                # Find minimum time dimension
                min_time = min(exact_expvals.shape[0], method_expvals.shape[0])
                
                for t in range(min_time):
                    # Extract values for selected qubits at time t
                    exact_vals = exact_expvals[t, plot_qubits]
                    method_vals = method_expvals[t, plot_qubits]
                    
                    # Calculate MSE: mean squared error over selected qubits
                    mse = np.mean((method_vals - exact_vals)**2)
                    mse_per_time.append(mse)
                
                # Plot MSE
                mse_style = styles.get(method_name, {}).copy()
                mse_style["label"] = f"{method_name} vs Exact"
                # Remove marker for cleaner MSE plot
                if "marker" in mse_style:
                    del mse_style["marker"]
                if "markevery" in mse_style:
                    del mse_style["markevery"]
                
                ax3.plot(t_axis[:len(mse_per_time)], mse_per_time, **mse_style)
    else:
        ax3.text(0.5, 0.5, "Exact reference not available", 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("MSE vs Exact\n(Exact data missing)")

    ax1.set_title(f"Staggered Magnetization\n(N={num_qubits}, Noise={noise_strength})")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Staggered Mag.")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title(f"Local Expectation Values (Z)\nQubits: {plot_qubits}")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("<Z>")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_title(f"MSE vs Exact\nQubits: {plot_qubits}")
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Mean Squared Error")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Use log scale for MSE to better visualize differences

    plt.tight_layout()
    
    out_name = f"comparison_plot_N{num_qubits}_noise{noise_str}_{model}.png"
    out_path = os.path.join(results_dir, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"✓ Comparison plot saved to: {out_path}")


def run_benchmarks(
    num_qubits, num_layers, tau, noise_strength,
    noise_model, circuit_label, fixed_trajectories,
    plot_qubits
):
    """
    Orchestrate running both Julia and Python benchmarks with synchronized parameters.
    """
    print("=" * 70)
    print(f"Running Comparison Benchmark")
    print(f"  Qubits: {num_qubits}")
    print(f"  Layers: {num_layers}")
    print(f"  Tau: {tau}")
    print(f"  Noise Strength: {noise_strength}")
    print(f"  Noise Model: {noise_model}")
    print(f"  Circuit: {circuit_label}")
    print(f"  Trajectories: {fixed_trajectories}")
    print("=" * 70)

    # 1. Update Python Script (run_benchmark.py)
    # Since we are in the same folder, use relative path or just filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    py_script = os.path.join(script_dir, "run_benchmark.py")
    jl_script = os.path.join(script_dir, "../benchmark_longrange_digitaltjm.jl")

    update_python_script(
        py_script,
        num_qubits, num_layers, tau, noise_strength,
        noise_model, circuit_label, fixed_trajectories
    )

    # 2. Update Julia Script (benchmark_periodic.jl)
    update_julia_script(
        jl_script,
        num_qubits, num_layers, tau, noise_strength,
        noise_model, circuit_label, fixed_trajectories
    )

    # 3. Run Python Benchmark
    print("\n>>> Running Python Qiskit Benchmark...")
    try:
        # Use python3 explicitely as 'python' might not be in path
        subprocess.run(["python3", py_script], check=True)
        print("✓ Python benchmark completed.")
    except subprocess.CalledProcessError as e:
        print(f"✗ Python benchmark failed with exit code {e.returncode}")
        return

    # 4. Run Julia Benchmark
    print("\n>>> Running Julia Benchmark...")
    try:
        # Best practice: assume we want to run julia with project at repo root.
        # Repo root is 4 levels up: ../../../../
        repo_root = os.path.abspath(os.path.join(script_dir, "../../../../"))
        
        subprocess.run(
            ["julia", "-t", "auto", f"--project={repo_root}", jl_script], 
            check=True
        )
        print("✓ Julia benchmark completed.")
    except subprocess.CalledProcessError as e:
        print(f"✗ Julia benchmark failed with exit code {e.returncode}")
        return

    # 5. Plot Results
    print("\n>>> Generating Comparison Plots...")
    plot_results(
        num_qubits, num_layers, tau, noise_strength,
        noise_model, circuit_label, fixed_trajectories,
        plot_qubits
    )

    print("\n" + "="*70)
    print("All benchmarks finished successfully.")
    print("Results are saved in 00_safety_checks_full_algorithms/digitalTJM_check/longrange_check/results/")
    print("="*70)


def update_python_script(filepath, n, L, tau, noise, model, circ, traj):
    """
    Search and replace parameters in the Python script using regex or direct string replacement.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    
    x_err = "True" if "X" in model.upper() else "False"
    y_err = "True" if "Y" in model.upper() else "False"
    z_err = "True" if "Z" in model.upper() else "False"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("num_qubits ="):
            new_lines.append(f"    num_qubits = {n}\n")
        elif stripped.startswith("num_layers ="):
            new_lines.append(f"    num_layers = {L}\n")
        elif stripped.startswith("tau ="):
            new_lines.append(f"    tau = {tau}\n")
        elif stripped.startswith("noise_strengths ="):
            # Python script expects a list
            new_lines.append(f"    noise_strengths = [{noise}]\n")
        elif stripped.startswith("fixed_trajectories ="):
            new_lines.append(f"    fixed_trajectories = {traj}\n")
        elif stripped.startswith("enable_qiskit_x_error ="):
            new_lines.append(f"    enable_qiskit_x_error = {x_err}\n")
        elif stripped.startswith("enable_qiskit_y_error ="):
            new_lines.append(f"    enable_qiskit_y_error = {y_err}\n")
        elif stripped.startswith("enable_qiskit_z_error ="):
            new_lines.append(f"    enable_qiskit_z_error = {z_err}\n")
        elif stripped.startswith('circuit_configs = [{"label":'):
            # Update circuit configs to include the requested circuit
            if circ == "longrange_test":
                new_lines.append(f'    circuit_configs = [\n')
                new_lines.append(f'        {{"label": "XY_longrange", "builder": lambda: xy_trotter_layer_longrange(num_qubits, tau, order="YX")}},\n')
                new_lines.append(f'        {{"label": "longrange_test", "builder": lambda: longrange_test_circuit(num_qubits, np.pi/4)}}\n')
                new_lines.append(f'    ]\n')
            else:
                new_lines.append(line) 
        else:
            new_lines.append(line)
            
    with open(filepath, 'w') as f:
        f.writelines(new_lines)


def update_julia_script(filepath, n, L, tau, noise, model, circ, traj):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    
    x_err = "true" if "X" in model.upper() else "false"
    y_err = "true" if "Y" in model.upper() else "false"
    z_err = "true" if "Z" in model.upper() else "false"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("NUM_QUBITS ="):
            new_lines.append(f"NUM_QUBITS = {n}\n")
        elif stripped.startswith("NUM_LAYERS ="):
            new_lines.append(f"NUM_LAYERS = {L}\n")
        elif stripped.startswith("TAU ="):
            new_lines.append(f"TAU = {tau}\n")
        elif stripped.startswith("NOISE_STRENGTH ="):
            new_lines.append(f"NOISE_STRENGTH = {noise}\n")
        elif stripped.startswith("NUM_TRAJECTORIES ="):
            indent = line[:line.find("NUM_TRAJECTORIES")]
            new_lines.append(f"{indent}NUM_TRAJECTORIES = {traj}\n")
        elif stripped.startswith("FIXED_TRAJECTORIES =") and "if SIMULATION_MODE" not in line:
             # Remove or ignore these as we simplified the scripts
             pass
        elif stripped.startswith("pauli_x_error ="):
            new_lines.append(f"pauli_x_error = {x_err}\n")
        elif stripped.startswith("pauli_y_error ="):
            new_lines.append(f"pauli_y_error = {y_err}\n")
        elif stripped.startswith("pauli_z_error ="):
            new_lines.append(f"pauli_z_error = {z_err}\n")
        elif stripped.startswith('CIRCUIT_TYPE ='):
            # Update CIRCUIT_TYPE, preserving any comment
            comment_part = ""
            if '#' in line:
                comment_idx = line.find('#')
                comment_part = " " + line[comment_idx:].rstrip()
            new_lines.append(f"CIRCUIT_TYPE = \"{circ}\"{comment_part}\n")
        else:
            # Handle indented lines
            if "NUM_TRAJECTORIES =" in line and "nothing" not in line:
                indent = line[:line.find("NUM_TRAJECTORIES")]
                new_lines.append(f"{indent}NUM_TRAJECTORIES = {traj}\n")
            elif "FIXED_TRAJECTORIES =" in line:
                 pass
            elif 'basis_label =' in line:
                 if circ != "XY_longrange":
                     pass
                 new_lines.append(line)
            else:
                new_lines.append(line)

    with open(filepath, 'w') as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qiskit vs Julia Benchmarks")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--layers", type=int, default=40, help="Number of layers")
    parser.add_argument("--tau", type=float, default=0.1, help="Time step tau")
    parser.add_argument("--noise", type=float, default=0.01, help="Noise strength")
    parser.add_argument("--model", type=str, default="Y", help="Noise model (e.g., 'Y', 'XY', 'XYZ')")
    parser.add_argument("--circuit", type=str, default="XY_longrange", help="Circuit label")
    parser.add_argument("--traj", type=int, default=10, help="Number of trajectories")
    parser.add_argument("--plot_qubits", type=int, nargs="+", default=[0,2,3], help="Indices of qubits to plot (space separated)")

    args = parser.parse_args()

    run_benchmarks(
        args.qubits, args.layers, args.tau, args.noise,
        args.model, args.circuit, args.traj, args.plot_qubits
    )
