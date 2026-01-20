module NoiseModule

using LinearAlgebra
using StaticArrays
using ..GateLibrary
using ..MPOModule
using ..MPSModule

export NoiseModel, AbstractNoiseProcess, LocalNoiseProcess, MPONoiseProcess

const C128 = ComplexF64

"""
Abstract supertype for noise processes acting on an MPS/MPO.

This provides a common interface for local and MPO-based noise processes with a shared element type.

Args:
    None

Returns:
    AbstractNoiseProcess{T}: Abstract noise process type.
"""
abstract type AbstractNoiseProcess{T} end

"""
Represent a local noise process acting on one or two sites.

This stores the process name, target sites, strength, and the local operator matrix used for the
dissipative or stochastic updates.

Args:
    name (String): Noise process name.
    sites (Vector{Int}): Target site indices.
    strength (Float64): Noise strength parameter.
    matrix (AbstractMatrix{T}): Local operator matrix.

Returns:
    LocalNoiseProcess{T}: Local noise process description.
"""
struct LocalNoiseProcess{T, M<:AbstractMatrix{T}} <: AbstractNoiseProcess{T}
    name::String
    sites::Vector{Int}
    strength::Float64
    matrix::M
end

"""
Represent a noise process specified by an MPO.

This stores an MPO describing the jump operator, along with optional local factors for faster
application when the MPO factorizes into a tensor product.

Args:
    name (String): Noise process name.
    sites (Vector{Int}): Target site indices.
    strength (Float64): Noise strength parameter.
    mpo (MPO{T}): MPO representation of the noise operator.
    factors (Vector{Matrix{T}}): Optional local factor matrices.

Returns:
    MPONoiseProcess{T}: MPO-based noise process description.
"""
struct MPONoiseProcess{T} <: AbstractNoiseProcess{T}
    name::String
    sites::Vector{Int}
    strength::Float64
    mpo::MPO{T}
    factors::Vector{Matrix{T}} # Optional: Stores local factors for optimization if jump is a tensor product
end

"""
Construct an MPO-based noise process without explicit factors.

This convenience constructor initializes the `factors` field to an empty vector for backward
compatibility or when local factors are not available.

Args:
    name (String): Noise process name.
    sites (Vector{Int}): Target site indices.
    strength (Float64): Noise strength parameter.
    mpo (MPO{T}): MPO representation of the noise operator.

Returns:
    MPONoiseProcess{T}: MPO-based noise process description.
"""
# Constructor for backward compatibility or when factors are not available
function MPONoiseProcess(name::String, sites::Vector{Int}, strength::Float64, mpo::MPO{T}) where T
    return MPONoiseProcess(name, sites, strength, mpo, Vector{Matrix{T}}())
end

"""
Container for a collection of noise processes.

This bundles local and MPO-based noise processes under a single model used in simulations.

Args:
    processes (Vector{AbstractNoiseProcess{T}}): Noise processes included in the model.

Returns:
    NoiseModel{T}: Noise model container.
"""
struct NoiseModel{T}
    processes::Vector{AbstractNoiseProcess{T}}
end

# --- Operator Retrieval ---

"""
Retrieve a local operator matrix by name.

This maps a string identifier to a 2x2 or 4x4 operator matrix used in noise process construction,
including Pauli and crosstalk operators.

Args:
    name (String): Operator identifier.

Returns:
    AbstractMatrix: Operator matrix corresponding to the name.

Raises:
    ErrorException: If the operator name is unknown.
"""
function get_operator(name::String)
    if name == "pauli_x" return matrix(XGate()) end
    if name == "pauli_y" return matrix(YGate()) end
    if name == "pauli_z" return matrix(ZGate()) end
    if name == "raising" return matrix(RaisingGate()) end
    if name == "lowering" return matrix(LoweringGate()) end
    if name == "raising_two" return kron(matrix(RaisingGate()), matrix(RaisingGate())) end
    if name == "lowering_two" return kron(matrix(LoweringGate()), matrix(LoweringGate())) end
    
    if startswith(name, "crosstalk_")
        suffix = name[11:end] # remove "crosstalk_"
        if length(suffix) == 2
            a_char, b_char = suffix[1], suffix[2]
            op_a = get_pauli(a_char)
            op_b = get_pauli(b_char)
            return kron(op_a, op_b)
        end
    end
    
    error("Unknown operator: $name")
end

"""
Retrieve a single-qubit Pauli matrix by character code.

This maps `'x'`, `'y'`, or `'z'` to the corresponding Pauli operator matrix.

Args:
    c (Char): Pauli identifier character.

Returns:
    AbstractMatrix: Pauli matrix corresponding to the identifier.

Raises:
    ErrorException: If the character is not recognized.
"""
function get_pauli(c::Char)
    if c == 'x' return matrix(XGate()) end
    if c == 'y' return matrix(YGate()) end
    if c == 'z' return matrix(ZGate()) end
    error("Unknown Pauli: $c")
end

# --- MPO Construction Helpers (Long Range) ---

"""
Construct a long-range MPO for a two-site operator term.

This builds an MPO for `O = a * I + b * (sigma_i ⊗ tau_j)` with identity tensors elsewhere using
the `(Left, Out, In, Right)` MPO layout.

Args:
    L (Int): Number of sites.
    i (Int): Left site index (1-based).
    j (Int): Right site index (1-based).
    sigma (AbstractMatrix): Operator for site `i`.
    tau (AbstractMatrix): Operator for site `j`.
    a (Number): Identity coefficient.
    b (Number): Two-site operator coefficient.

Returns:
    MPO: MPO representing the long-range operator.

Raises:
    AssertionError: If `i` and `j` are not within bounds or ordered.
"""
function build_mpo_phys(L::Int, i::Int, j::Int, sigma::AbstractMatrix, tau::AbstractMatrix, a::Number, b::Number)
    # Note: Python i, j are 0-based. Julia 1-based.
    # Arguments i, j here are expected to be 1-based indices.
    
    @assert 1 <= i < j <= L "Require 1 <= i < j <= L"
    
    tensors = Vector{Array{C128, 4}}(undef, L)
    phys_dim = 2 # Assuming qubits
    
    Id2 = Matrix{C128}(I, 2, 2)
    
    # Sites < i: Identity 1x1
    # Shape: (1, 2, 2, 1) -> (Left, Out, In, Right)
    for k in 1:i-1
        T = zeros(C128, 1, 2, 2, 1)
        T[1, :, :, 1] = Id2
        tensors[k] = T
    end
    
    # Site i: row [I, sigma]
    # Python: (Left, Right, Out, In) -> (1, 2, 2, 2) with Left=1, Right=2
    # Julia: (Left, Out, In, Right) -> (1, 2, 2, 2)
    Wi = zeros(C128, 1, 2, 2, 2)
    Wi[1, :, :, 1] = Id2
    Wi[1, :, :, 2] = sigma
    tensors[i] = Wi
    
    # i < k < j: diag(I, I)
    # Shape: (2, 2, 2, 2)
    for k in i+1:j-1
        Wmid = zeros(C128, 2, 2, 2, 2)
        Wmid[1, :, :, 1] = Id2
        Wmid[2, :, :, 2] = Id2
        tensors[k] = Wmid
    end
    
    # Site j: column [a*I; b*tau]
    # Shape: (2, 2, 2, 1)
    Wj = zeros(C128, 2, 2, 2, 1)
    Wj[1, :, :, 1] = a * Id2
    Wj[2, :, :, 1] = b * tau
    tensors[j] = Wj
    
    # Sites > j: Identity 1x1
    for k in j+1:L
        T = zeros(C128, 1, 2, 2, 1)
        T[1, :, :, 1] = Id2
        tensors[k] = T
    end
    
    phys_dims = fill(2, L)
    return MPOModule.MPO(L, tensors, phys_dims)
end

# --- Unraveling Expansions ---

"""
Expand a projector noise process into +/- local channels.

This appends two LocalNoiseProcess entries corresponding to `(I ± P)` with half strength, matching
the projector-channel decomposition used in the stochastic unraveling.

Args:
    procs (Vector{AbstractNoiseProcess{C128}}): Output vector to append processes to.
    proc_info (Dict): Process metadata including `name` and `sites`.
    P (AbstractMatrix): Projector-defining operator matrix.
    gamma (Float64): Noise strength parameter.

Returns:
    Nothing: Processes are appended to `procs` in-place.
"""
function add_projector_expansion!(procs::Vector{AbstractNoiseProcess{C128}}, proc_info::Dict, P::AbstractMatrix, gamma::Float64)
    dim = size(P, 1)
    Id = Matrix{C128}(I, dim, dim)
    
    for (comp, sign_val) in [("plus", 1.0), ("minus", -1.0)]
        new_name = "projector_$(comp)_$(proc_info["name"])"
        new_strength = gamma / 2.0
        new_matrix = (Id + sign_val * P) # This is usually L†L ?? 
        # Python code says: matrix: (I + sign * P) which represents L^\dagger L = 2\gamma I ??
        # Wait, Python code comment: "L = sqrt(gamma/2) (I \pm P)".
        # "matrix": (I + sign * P).
        # In NoiseModel, 'matrix' usually stores the Jump Operator L.
        # If L = sqrt(gamma/2)(I+P), then L is the operator.
        # (I+P) is a projector (scaled). (I+P)/2 is projector.
        # (I+P)^2 = I + 2P + P^2. If P^2=I (Pauli), = 2(I+P).
        # So (I+P) is prop to projector.
        
        # If Python stores (I+P) in "matrix", then that is L (modulo sqrt(strength)).
        push!(procs, LocalNoiseProcess(new_name, proc_info["sites"], new_strength, SMatrix{dim,dim,C128}(new_matrix)))
    end
end

"""
Expand a long-range projector process into MPO-based +/- channels.

This constructs MPOs for `(I ± P)` over a long-range pair of sites and appends two MPONoiseProcess
entries with half strength.

Args:
    procs (Vector{AbstractNoiseProcess{C128}}): Output vector to append processes to.
    proc_info (Dict): Process metadata including `name`, `sites`, and optional `factors`.
    L (Int): Total number of sites.
    gamma (Float64): Noise strength parameter.

Returns:
    Nothing: Processes are appended to `procs` in-place.
"""
function add_projector_expansion_longrange!(procs::Vector{AbstractNoiseProcess{C128}}, proc_info::Dict, L::Int, gamma::Float64)
    i, j = sort(proc_info["sites"]) # These are likely 1-based in Julia usage?
    # We assume input dictionaries use 1-based indexing if coming from Julia, 
    # but if porting Python scripts, they might be 0-based.
    # The implementation of NoiseModel should standardise this. 
    # Let's assume the dicts passed to constructor use 1-BASED indexing for consistency with Julia.
    
    # Parse factors
    if haskey(proc_info, "factors")
        sigma, tau = proc_info["factors"]
    else
        # Parse from name
        name = proc_info["name"]
        suffix = split(name, "_")[end] # Assumes "crosstalk_xy" format
        sigma = get_pauli(suffix[1])
        tau = get_pauli(suffix[2])
    end
    
    for (comp, sign_val) in [("plus", 1.0), ("minus", -1.0)]
        new_name = "projector_$(comp)_$(proc_info["name"])"
        new_strength = gamma / 2.0
        
        # Construct MPO for (I ± P) where P = sigma_i tau_j
        # a = 1.0, b = sign_val
        mpo = build_mpo_phys(L, i, j, sigma, tau, 1.0, sign_val)
        
        push!(procs, MPONoiseProcess(new_name, [i, j], new_strength, mpo))
    end
end

"""
Expand a two-point unitary noise process into +/- local channels.

This constructs local unitary operators `exp(±i θ0 P)` and appends two LocalNoiseProcess entries
with strengths derived from `gamma` and `theta0`.

Args:
    procs (Vector{AbstractNoiseProcess{C128}}): Output vector to append processes to.
    proc_info (Dict): Process metadata including `name` and `sites`.
    P (AbstractMatrix): Generator operator matrix.
    gamma (Float64): Noise strength parameter.
    theta0 (Float64): Rotation angle used in the decomposition.

Returns:
    Nothing: Processes are appended to `procs` in-place.

Raises:
    AssertionError: If `theta0` is too small to define a stable decomposition.
"""
function add_unitary_2pt_expansion!(procs::Vector{AbstractNoiseProcess{C128}}, proc_info::Dict, P::AbstractMatrix, gamma::Float64, theta0::Float64)
    s_val = sin(theta0)^2
    @assert s_val > 0 "theta0 too small"
    lam = gamma / s_val
    
    for (comp, sign_val) in [("plus", 1.0), ("minus", -1.0)]
        U = exp(1im * sign_val * theta0 * P)
        new_name = "unitary2pt_$(comp)_$(proc_info["name"])"
        new_strength = lam / 2.0
        dim = size(P, 1)
        push!(procs, LocalNoiseProcess(new_name, proc_info["sites"], new_strength, SMatrix{dim,dim,C128}(U)))
    end
end

"""
Expand a long-range two-point unitary process into MPO channels.

This builds MPOs for `a I + b P` with coefficients derived from `theta0` and appends two
MPONoiseProcess entries with strengths derived from `gamma`.

Args:
    procs (Vector{AbstractNoiseProcess{C128}}): Output vector to append processes to.
    proc_info (Dict): Process metadata including `name`, `sites`, and optional `factors`.
    L (Int): Total number of sites.
    gamma (Float64): Noise strength parameter.
    theta0 (Float64): Rotation angle used in the decomposition.

Returns:
    Nothing: Processes are appended to `procs` in-place.
"""
function add_unitary_2pt_expansion_longrange!(procs::Vector{AbstractNoiseProcess{C128}}, proc_info::Dict, L::Int, gamma::Float64, theta0::Float64)
    i, j = sort(proc_info["sites"])
    
    if haskey(proc_info, "factors")
        sigma, tau = proc_info["factors"]
    else
        name = proc_info["name"]
        suffix = split(name, "_")[end]
        sigma = get_pauli(suffix[1])
        tau = get_pauli(suffix[2])
    end
    
    s_val = sin(theta0)^2
    lam = gamma / s_val
    
    for (comp, sign_val) in [("plus", 1.0), ("minus", -1.0)]
        a = cos(theta0)
        b = 1im * sign_val * sin(theta0)
        
        mpo = build_mpo_phys(L, i, j, sigma, tau, a, b)
        
        new_name = "unitary2pt_$(comp)_$(proc_info["name"])"
        new_strength = lam / 2.0
        
        push!(procs, MPONoiseProcess(new_name, [i, j], new_strength, mpo))
    end
end

"""
Expand a Gaussian-distributed unitary noise process into local channels.

This discretizes a Gaussian distribution over rotation angles and appends LocalNoiseProcess entries
with weights normalized to match the target noise strength.

Args:
    procs (Vector{AbstractNoiseProcess{C128}}): Output vector to append processes to.
    proc_info (Dict): Process metadata including `name`, `sites`, and optional parameters.
    P (AbstractMatrix): Generator operator matrix.
    gamma (Float64): Noise strength parameter.
    sigma (Float64): Standard deviation of the angle distribution.
    gauss_M (Int): Number of discretization points.
    gauss_k (Float64): Range multiplier for the angle grid.

Returns:
    Nothing: Processes are appended to `procs` in-place.

Raises:
    AssertionError: If the Gaussian weight normalization is ill-conditioned.
"""
function add_unitary_gauss_expansion!(procs::Vector{AbstractNoiseProcess{C128}}, proc_info::Dict, P::AbstractMatrix, gamma::Float64, sigma::Float64, gauss_M::Int, gauss_k::Float64)
    M = get(proc_info, "M", gauss_M)
    theta_max = get(proc_info, "theta_max", gauss_k * sigma)
    
    thetas_pos = range(0.0, theta_max, length=(M+1)÷2)
    # Discretization logic from Python
    # concatenate([-thetas_pos[:0:-1], thetas_pos])
    thetas = vcat(-reverse(thetas_pos[2:end]), thetas_pos)
    
    w = exp.(-0.5 .* (thetas ./ sigma).^2)
    w ./= sum(w)
    w = 0.5 .* (w .+ reverse(w))
    
    s_weight = sum(w .* (sin.(thetas).^2))
    @assert s_weight > 1e-12 "E[sin^2 theta] too small"
    
    lam = gamma / s_weight
    
    for (idx, (wk, th)) in enumerate(zip(w, thetas))
        if wk <= 0.0 continue end
        
        U = exp(1im * th * P)
        new_name = "unitary_gauss_$(idx-1)_$(proc_info["name"])"
        new_strength = lam * wk
        dim = size(P, 1)
        push!(procs, LocalNoiseProcess(new_name, proc_info["sites"], new_strength, SMatrix{dim,dim,C128}(U)))
    end
end

"""
Expand a Gaussian-distributed unitary noise process into MPO channels.

This discretizes a Gaussian distribution over rotation angles and constructs MPOs for each angle,
appending MPONoiseProcess entries with normalized weights.

Args:
    procs (Vector{AbstractNoiseProcess{C128}}): Output vector to append processes to.
    proc_info (Dict): Process metadata including `name`, `sites`, and optional `factors`.
    L (Int): Total number of sites.
    gamma (Float64): Noise strength parameter.
    sigma (Float64): Standard deviation of the angle distribution.
    gauss_M (Int): Number of discretization points.
    gauss_k (Float64): Range multiplier for the angle grid.

Returns:
    Nothing: Processes are appended to `procs` in-place.
"""
function add_unitary_gauss_expansion_longrange!(procs::Vector{AbstractNoiseProcess{C128}}, proc_info::Dict, L::Int, gamma::Float64, sigma::Float64, gauss_M::Int, gauss_k::Float64)
    i, j = sort(proc_info["sites"])
    
    if haskey(proc_info, "factors")
        sig, tau = proc_info["factors"]
    else
        name = proc_info["name"]
        suffix = split(name, "_")[end]
        sig = get_pauli(suffix[1])
        tau = get_pauli(suffix[2])
    end
    
    M = get(proc_info, "M", gauss_M)
    theta_max = get(proc_info, "theta_max", gauss_k * sigma)
    
    thetas_pos = range(0.0, theta_max, length=(M+1)÷2)
    thetas = vcat(-reverse(thetas_pos[2:end]), thetas_pos)
    
    w = exp.(-0.5 .* (thetas ./ sigma).^2)
    w ./= sum(w)
    w = 0.5 .* (w .+ reverse(w))
    
    s_weight = sum(w .* (sin.(thetas).^2))
    lam = gamma / s_weight
    
    for (idx, (wk, th)) in enumerate(zip(w, thetas))
        if wk <= 0.0 continue end
        
        a = cos(th)
        b = 1im * sin(th)
        
        mpo = build_mpo_phys(L, i, j, sig, tau, a, b)
        
        new_name = "unitary_gauss_$(idx-1)_$(proc_info["name"])"
        new_strength = lam * wk
        
        push!(procs, MPONoiseProcess(new_name, [i, j], new_strength, mpo))
    end
end

# --- Main Constructor ---

"""
Build a NoiseModel from process metadata dictionaries.

This parses process definitions, expands composite channels (projector/unitary/gaussian), and
returns a NoiseModel containing the resulting local or MPO noise processes.

Args:
    processes_info (Vector{Dict{String, Any}}): Process definition dictionaries.
    num_qubits (Int): Total number of qubits in the system.
    theta0 (Float64): Base rotation angle for unitary two-point channels.
    dt (Union{Float64, Nothing}): Optional time step for rate conversion.
    sigma (Float64): Standard deviation for Gaussian unitary channels.
    gauss_M (Int): Number of discretization points for Gaussian channels.
    gauss_k (Float64): Range multiplier for Gaussian angle grid.

Returns:
    NoiseModel{ComplexF64}: Constructed noise model with expanded processes.
"""
function NoiseModel(processes_info::Vector{Dict{String, Any}}, num_qubits::Int; 
                   theta0::Float64=pi/2, dt::Union{Float64, Nothing}=nothing,
                   sigma::Float64=1.0, gauss_M::Int=11, gauss_k::Float64=4.0)
    
    final_processes = Vector{AbstractNoiseProcess{C128}}()
    
    
    for proc in processes_info
        name = proc["name"]
        sites = proc["sites"]
        strength = Float64(proc["strength"])
        unravel = get(proc, "unraveling", "standard")
        
        # Normalize sites (sort)
        if length(sites) == 2
            sites = sort(sites)
        end
        
        # --- Unraveling ---
        if unravel == "projector"
            # Local or Long Range?
            is_long_range = (length(sites) == 2 && abs(sites[2] - sites[1]) > 1)
            
            if is_long_range
                add_projector_expansion_longrange!(final_processes, proc, num_qubits, strength)
            else
                P = get_operator_from_proc(proc)
                add_projector_expansion!(final_processes, proc, P, strength)
            end
            continue
        elseif unravel == "unitary_2pt"
             # Get theta0 from proc or default
             th = get(proc, "theta0", theta0)
             
             # Stability Check
             # lambda = gamma / sin(theta0)^2
             s_val = sin(th)^2
             if s_val > 1e-12
                 lam = strength / s_val
                 if !isnothing(dt) && (lam * dt > 0.1)
                     @warn "Stability Warning: theta0=$th is too small for dt=$dt (lambda*dt = $(lam*dt) > 0.1). Consider increasing theta0 or decreasing dt."
                 end
             end

             is_long_range = (length(sites) == 2 && abs(sites[2] - sites[1]) > 1)
             if is_long_range
                 add_unitary_2pt_expansion_longrange!(final_processes, proc, num_qubits, strength, th)
             else
                 P = get_operator_from_proc(proc)
                 add_unitary_2pt_expansion!(final_processes, proc, P, strength, th)
             end
             continue
        elseif unravel == "unitary_gauss"
            # Get params from proc or default
            sig = get(proc, "sigma", sigma)
            M = get(proc, "M", gauss_M)
            k_val = get(proc, "gauss_k", gauss_k)
            
            is_long_range = (length(sites) == 2 && abs(sites[2] - sites[1]) > 1)
            if is_long_range
                add_unitary_gauss_expansion_longrange!(final_processes, proc, num_qubits, strength, sig, M, k_val)
            else
                P = get_operator_from_proc(proc)
                add_unitary_gauss_expansion!(final_processes, proc, P, strength, sig, M, k_val)
            end
            continue
        elseif unravel == "analog_auto"
             @warn "Unraveling mode 'analog_auto' is deprecated. Please choose 'unitary_2pt' or 'unitary_gauss' manually."
        end
        
        # --- Standard (Pass-through) ---
        if isempty(sites) || length(sites) > 2
             error("Invalid sites")
        end
        
        is_long_range = (length(sites) == 2 && abs(sites[2] - sites[1]) > 1)
        
        if is_long_range && !haskey(proc, "mpo")
            # Long range standard -> Needs MPO? 
            # Or is it treated as a standard jump operator L?
            # Standard Lindblad L is local. 
            # If long range L, it must be MPO?
            # Python: "non-adjacent 2-site processes must specify 'factors'".
            # But it appends to self.processes without expansion?
            # Wait, if standard process has non-adjacent sites, how is it simulated?
            # The Python code seems to assume the backend handles it.
            # But 'NoiseModel' just stores it.
            # For Julia, we need explicit MPO or matrix.
            # If it's standard jump L = sqrt(gamma) * O.
            # If O is non-local, we need MPO.
            
            # Construct MPO for L = sqrt(gamma) * (sigma_i ⊗ tau_j)
            # Use build_mpo_phys with a=0, b=sqrt(gamma)?
            # No, NoiseProcess separates 'strength'.
            # So MPO should be just O.
            
            if haskey(proc, "factors")
                sigma, tau = proc["factors"]
            else
                suffix = split(name, "_")[end]
                sigma = get_pauli(suffix[1])
                tau = get_pauli(suffix[2])
            end
            
            # Build MPO for O = sigma_i ⊗ tau_j
            # a=0, b=1
            mpo = build_mpo_phys(num_qubits, sites[1], sites[2], sigma, tau, 0.0, 1.0)
            
            # Store factors for optimization
            factors = Matrix{C128}[Matrix(sigma), Matrix(tau)]
            push!(final_processes, MPONoiseProcess(name, sites, strength, mpo, factors))
            
        else
            # Local
            mat = get_operator_from_proc(proc)
            dim = size(mat, 1)
            push!(final_processes, LocalNoiseProcess(name, sites, strength, SMatrix{dim,dim,C128}(mat)))
        end
    end
    
    return NoiseModel(final_processes)
end

"""
Resolve an operator matrix from a process dictionary.

This returns the explicit matrix if provided, otherwise derives the operator from the process name,
including crosstalk naming conventions.

Args:
    proc (Dict): Process definition dictionary.

Returns:
    AbstractMatrix: Operator matrix for the process.
"""
function get_operator_from_proc(proc::Dict)
    if haskey(proc, "matrix")
        return proc["matrix"]
    end
    name = proc["name"]
    # Handle adjacent crosstalk
    if startswith(name, "crosstalk_")
        # ... logic ...
        return get_operator(name)
    end
    return get_operator(name)
end

end
