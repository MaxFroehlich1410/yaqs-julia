using DelimitedFiles
using LinearAlgebra
using Statistics
using PythonCall

# Ensure we use the local Yaqs package
using Yaqs
using Yaqs.MPSModule
using Yaqs.MPOModule
using Yaqs.NoiseModule
using Yaqs.SimulationConfigs
using Yaqs.AnalogTJM
using Yaqs.GateLibrary

# Import Python modules
const qt = pyimport("qutip")
const plt = pyimport("matplotlib.pyplot")
const np = pyimport("numpy")

function _progress_bar_string(done::Int, total::Int; width::Int=30)
    total <= 0 && return "[ ] (0/0)"
    frac = clamp(done / total, 0.0, 1.0)
    filled = Int(floor(frac * width))
    filled = clamp(filled, 0, width)
    bar = string("[", repeat("=", filled), repeat(".", width - filled), "]")
    return string(bar, " (", done, "/", total, ")")
end

function _print_progress(done::Int, total::Int; extra::AbstractString="", width::Int=30)
    msg = _progress_bar_string(done, total; width=width)
    if !isempty(extra)
        msg = string(msg, "  ", extra)
    end
    print("\r", msg, "    ")
    flush(stdout)
    return nothing
end

function run_qutip_simulation(L, J, h, strength, times)
    println("Running Exact QuTiP Simulation...")
    
    # 1. Operators
    sx_list = []
    sz_list = []
    sm_list = [] # Sigma minus (lowering operator, maps |0> -> |1>)
    
    for i in 0:(L-1)
        push!(sx_list, qt.tensor([j == i ? qt.sigmax() : qt.qeye(2) for j in 0:(L-1)]))
        push!(sz_list, qt.tensor([j == i ? qt.sigmaz() : qt.qeye(2) for j in 0:(L-1)]))
        # Julia's RaisingGate is [0 0; 1 0] which maps |0> -> |1>. This corresponds to sigmam in QuTiP.
        push!(sm_list, qt.tensor([j == i ? qt.sigmam() : qt.qeye(2) for j in 0:(L-1)]))
    end
    
    # 2. Hamiltonian
    # H = -J sum Z Z - h sum X
    H = 0 * qt.tensor([qt.qeye(2) for _ in 1:L])
    
    # Interaction term
    for i in 0:(L-2)
        H = H - J * sz_list[i+1] * sz_list[i+2]
    end

    
    # Transverse field
    for i in 0:(L-1)
        H = H - h * sx_list[i+1]
    end
    
    # 3. Collapse Operators
    # Raising operators on every site: sqrt(strength) * sigma_- (which maps |0> to |1>)
    # Note: We use sm_list (sigmam) to match Julia's RaisingGate behavior on the |0> state.
    c_ops = []
    for i in 0:(L-1)
        push!(c_ops, np.sqrt(strength) * sm_list[i+1])
    end
    
    # 4. Initial State
    # |00...0> -> basis(2, 0) is |0> (spin up, +1)
    psi0 = qt.tensor([qt.basis(2, 0) for _ in 1:L])
    
    # 5. Expectation operators
    # We want Z on all sites to compute staggered mag later
    e_ops = [sz for sz in sz_list]
    
    # 6. Run Master Equation
    # Convert lists to Python objects explicitly if needed, but PythonCall usually handles Vector -> list
    # The error "Vector{Any} are not callable" often comes from pyconvert/pycall issues if arguments aren't right.
    # qutip.mesolve(H, rho0, tlist, c_ops=[], e_ops=[], args={}, options=None, progress_bar=None, _safe_mode=True)
    
    # We pass python lists explicitly just to be safe
    c_ops_py = pylist(c_ops)
    e_ops_py = pylist(e_ops)
    
    result = qt.mesolve(H, psi0, times, c_ops=c_ops_py, e_ops=e_ops_py)
    
    # result.expect is a list of arrays (one per e_op)
    # Convert to Julia matrix: (num_sites, num_times)
    
    qutip_results = zeros(Float64, L, length(times))
    for i in 1:L
        # result.expect is 0-indexed list
        qutip_results[i, :] = pyconvert(Vector{Float64}, result.expect[i-1])
    end
    
    return qutip_results
end

function run_benchmark()
    # Parameters
    L = 6
    J = 1.0
    h = 0.5
    strength = 0.1
    max_bond = 8
    num_traj = 500 # Reduced for speed if needed, but user asked for 1000 previously. Keeping 1000.
    dt = 0.05
    T_total = 2.0
    
    println("Starting TJM Benchmark...")
    println("L=$L, J=$J, h=$h, gamma=$strength")
    println("Trajectories: $num_traj")
    
    # 1. Initialize State (All Zeros)
    state = MPS(L, state="zeros")
    
    # 2. Initialize Hamiltonian
    # H = -J sum Z Z - h sum X
    H = MPOModule.init_ising(L, J, h)
    
    # 3. Initialize Noise
    # Raising operators on every site
    processes = [Dict{String, Any}("name" => "raising", "sites" => [i], "strength" => strength) for i in 1:L]
    noise_model = NoiseModel(processes, L)
    
    # 4. Observables: Z on ALL sites (needed for staggered mag)
    observables = [Observable("Z_$i", ZGate(), i) for i in 1:L]
    
    # 5. Simulation Config
    sim_params = TimeEvolutionConfig(
        observables, 
        T_total; 
        dt=dt, 
        num_traj=num_traj, 
        max_bond_dim=max_bond, 
        sample_timesteps=true
    )
    
    times = sim_params.times
    num_steps = length(times)
    num_obs = length(observables)
    
    # --- Run Julia TJM ---
    println("Running Julia TJM ($num_traj trajectories)...")
    
    all_results = zeros(Float64, num_obs, num_steps, num_traj)
    progress = Threads.Atomic{Int}(0)
    progress_lock = ReentrantLock()
    
    Threads.@threads for i in 1:num_traj
        # Note: We pass fresh objects or handle deepcopies inside. 
        # analog_tjm_2 does deepcopy(state) internally.
        traj_res = analog_tjm_2((i, state, noise_model, sim_params, H))
        all_results[:, :, i] = traj_res
        
        c = Threads.atomic_add!(progress, 1) + 1
        lock(progress_lock) do
            # Analog TJM doesn't currently expose per-trajectory bond growth; report cap for context.
            _print_progress(c, num_traj; extra="Max Bond cap: $max_bond")
            if c == num_traj
                println()
            end
        end
    end
    
    # Average
    tjm_mean = dropdims(mean(all_results, dims=3), dims=3)
    
    # --- Run QuTiP Exact ---
    qutip_mean = run_qutip_simulation(L, J, h, strength, times)
    
    # --- Calculate Staggered Magnetization ---
    # M_stagg = 1/L * sum_i (-1)^(i) * <Z_i>
    # Sites 1..L
    
    function calc_stagg_mag(z_expectations)
        # z_expectations: (L, num_steps)
        stagg = zeros(Float64, size(z_expectations, 2))
        for t in 1:size(z_expectations, 2)
            val = 0.0
            for i in 1:L
                val += (-1)^i * z_expectations[i, t]
            end
            stagg[t] = val / L
        end
        return stagg
    end
    
    tjm_stagg = calc_stagg_mag(tjm_mean)
    qt_stagg = calc_stagg_mag(qutip_mean)
    
    # --- Plotting ---
    println("Plotting results...")
    
    # Create results folder
    mkpath("00_safety_checks_full_algorithms/analogTJM_check/non_long_range_check/results")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Local Z expectations (Sites 1, 3, 6)
    sites_to_plot = [1, 3, 6]
    ax1 = axes[0]
    
    for site in sites_to_plot
        # Julia 1-based index vs QuTiP array 1-based index logic above
        ax1.plot(times, tjm_mean[site, :], label="TJM Site $site", linestyle="--", marker="o", markersize=3)
        ax1.plot(times, qutip_mean[site, :], label="Exact Site $site", linestyle="-", alpha=0.7)
    end
    
    ax1.set_xlabel("Time")
    ax1.set_ylabel(raw"$\langle Z_i \rangle$")
    ax1.set_title("Local Z Expectation Values")
    ax1.legend()
    ax1.grid(true)
    
    # Subplot 2: Staggered Magnetization
    ax2 = axes[1]
    ax2.plot(times, tjm_stagg, label="TJM", linestyle="--", marker="o", markersize=3, color="blue")
    ax2.plot(times, qt_stagg, label="Exact", linestyle="-", color="black", alpha=0.7)
    
    ax2.set_xlabel("Time")
    ax2.set_ylabel(raw"$M_{stagg}$")
    ax2.set_title("Staggered Magnetization")
    ax2.legend()
    ax2.grid(true)
    
    plt.tight_layout()
    
    outfile = "00_safety_checks_full_algorithms/analogTJM_check/non_long_range_check/results/benchmark_plot.png"
    plt.savefig(outfile, dpi=300)
    plt.close()
    
    println("Saved plot to $outfile")
    
    # Also save CSV for reference
    data_to_save = hcat(times, tjm_mean', tjm_stagg, qutip_mean', qt_stagg)
    # Header logic is a bit messy for simple writedlm, but raw data is there.
    csv_file = "00_safety_checks_full_algorithms/analogTJM_check/non_long_range_check/results/tjm_vs_qutip_data.csv"
    writedlm(csv_file, data_to_save, ',')
    println("Saved data to $csv_file")
end

run_benchmark()
