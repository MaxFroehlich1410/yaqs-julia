using LinearAlgebra
using Printf

# Helper to ensure modules are loaded
if !isdefined(Main, :GateLibrary)
    include("../src/GateLibrary.jl")
end
if !isdefined(Main, :Decompositions)
    include("../src/Decompositions.jl")
end
if !isdefined(Main, :MPSModule)
    include("../src/MPS.jl")
end
if !isdefined(Main, :MPOModule)
    include("../src/MPO.jl")
end
if !isdefined(Main, :SimulationConfigs)
    include("../src/SimulationConfigs.jl")
end
if !isdefined(Main, :Algorithms)
    include("../src/Algorithms.jl")
end

using .GateLibrary
using .MPSModule
using .MPOModule
using .SimulationConfigs
using .Algorithms

# --- Exact Simulation Helpers ---

function get_full_op(op, site, L)
    # I x I x ... x op x ... x I
    ops = [Matrix{ComplexF64}(I, 2, 2) for _ in 1:L]
    ops[site] = Matrix(matrix(op))
    res = ops[1]
    for i in 2:L
        res = kron(res, ops[i])
    end
    return res
end

function construct_ising_hamiltonian(L, J, g)
    # H = sum(-J_i Z_i Z_{i+1}) + sum(-g_i X_i)
    H = zeros(ComplexF64, 2^L, 2^L)
    
    get_J(i) = isa(J, Vector) ? (i <= length(J) ? J[i] : 0.0) : J
    get_g(i) = isa(g, Vector) ? g[i] : g
    
    Z = ZGate()
    X = XGate()
    
    # Interaction
    for i in 1:(L-1)
        val_J = get_J(i)
        term = get_full_op(Z, i, L) * get_full_op(Z, i+1, L)
        H .-= val_J .* term
    end
    
    # Field
    for i in 1:L
        val_g = get_g(i)
        term = get_full_op(X, i, L)
        H .-= val_g .* term
    end
    return H
end

function construct_heisenberg_hamiltonian(L, Jx, Jy, Jz, h)
    H = zeros(ComplexF64, 2^L, 2^L)
    X = XGate()
    Y = YGate()
    Z = ZGate()
    
    for i in 1:(L-1)
        H .-= Jx .* (get_full_op(X, i, L) * get_full_op(X, i+1, L))
        H .-= Jy .* (get_full_op(Y, i, L) * get_full_op(Y, i+1, L))
        H .-= Jz .* (get_full_op(Z, i, L) * get_full_op(Z, i+1, L))
    end
    
    for i in 1:L
        H .-= h .* get_full_op(Z, i, L)
    end
    return H
end

function measure_exact(psi, op, site, L)
    O = get_full_op(op, site, L)
    return real(dot(psi, O * psi))
end

# --- Benchmark Runner ---

function run_benchmark()
    println("Running TDVP Benchmark...")
    L = 6
    t_total = 2.0
    dt = 0.05
    steps = Int(t_total / dt)
    times = collect(0:steps) .* dt
    
    # Define Models
    models = []
    
    # 1. Ising Ferromagnetic
    push!(models, (
        name="Ising_Ferro",
        H_mpo=init_ising(L, 1.0, 0.5),
        H_exact=construct_ising_hamiltonian(L, 1.0, 0.5)
    ))
    
    # 2. Ising Disordered (Noisy)
    # Fixed parameters for reproducibility in Qutip verification
    # L=6
    # J (5 bonds): [0.6, 1.2, 0.8, 1.4, 0.9]
    # g (6 sites): [0.7, 1.3, 0.5, 1.1, 0.8, 1.2]
    J_rand = [0.6, 1.2, 0.8, 1.4, 0.9]
    g_rand = [0.7, 1.3, 0.5, 1.1, 0.8, 1.2]
    push!(models, (
        name="Ising_Disordered",
        H_mpo=init_ising(L, J_rand, g_rand),
        H_exact=construct_ising_hamiltonian(L, J_rand, g_rand)
    ))
    
    # 3. Heisenberg
    push!(models, (
        name="Heisenberg",
        H_mpo=init_heisenberg(L, 1.0, 1.0, 1.0, 0.5),
        H_exact=construct_heisenberg_hamiltonian(L, 1.0, 1.0, 1.0, 0.5)
    ))
    
    # Sites to measure
    sites = [1, L÷2, L]
    op_Z = ZGate()
    
    # Open CSV file
    open("tdvp_benchmark_results.csv", "w") do io
        write(io, "Model,Method,Time,Site,ExpVal\n")
        
        for model in models
            println("Simulating $(model.name)...")
            
            # 1. Exact Evolution
            psi_0 = zeros(ComplexF64, 2^L)
            psi_0[1] = 1.0 # |00...0> (assuming basis order)
            # Verify basis order: |0> is [1,0]. |00> is [1,0,0,0].
            # MPS "zeros" is |00...0>.
            
            # Pre-calculate Exact trajectory
            # exp(-i H t) * psi
            # We iterate steps
            psi_t = copy(psi_0)
            
            # Initial measure
            for s in sites
                val = measure_exact(psi_t, op_Z, s, L)
                write(io, "$(model.name),Exact,0.0,$s,$val\n")
            end
            
            U_step = exp(-1im * dt * model.H_exact)
            
            for t_idx in 1:steps
                psi_t = U_step * psi_t
                t = t_idx * dt
                for s in sites
                    val = measure_exact(psi_t, op_Z, s, L)
                    write(io, "$(model.name),Exact,$t,$s,$val\n")
                end
            end
            
            # 2. Single Site TDVP
            mps_1 = MPS(L; state="zeros")
            # Patch: Expand bond dimension for 1TDVP to allow entanglement growth
            pad_bond_dimension!(mps_1, 8)
            
            config_1 = TimeEvolutionConfig(Observable[], t_total; dt=dt)
            
            # Initial measure
            for s in sites
                # Use local_expect from MPSModule
                val = real(local_expect(mps_1, matrix(op_Z), s))
                write(io, "$(model.name),1TDVP,0.0,$s,$val\n")
            end
            
            for t_idx in 1:steps
                single_site_tdvp!(mps_1, model.H_mpo, config_1)
                t = t_idx * dt
                for s in sites
                    val = real(local_expect(mps_1, matrix(op_Z), s))
                    write(io, "$(model.name),1TDVP,$t,$s,$val\n")
                end
            end
            
            # 3. Two Site TDVP
            mps_2 = MPS(L; state="zeros")
            config_2 = TimeEvolutionConfig(Observable[], t_total; dt=dt, max_bond_dim=16) # Low bond dim to test truncation?
            
            # Initial measure
            for s in sites
                val = real(local_expect(mps_2, matrix(op_Z), s))
                write(io, "$(model.name),2TDVP,0.0,$s,$val\n")
            end
            
            for t_idx in 1:steps
                two_site_tdvp!(mps_2, model.H_mpo, config_2)
                t = t_idx * dt
                for s in sites
                    val = real(local_expect(mps_2, matrix(op_Z), s))
                    write(io, "$(model.name),2TDVP,$t,$s,$val\n")
                end
            end
            
        end
    end
    println("Benchmark complete. Results saved to tdvp_benchmark_results.csv")
end

run_benchmark()

