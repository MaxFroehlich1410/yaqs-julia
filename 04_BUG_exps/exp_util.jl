module BUGExpUtil

using LinearAlgebra
using Statistics
using Printf
using PythonCall

using Yaqs

const BUG = Yaqs.BUGModule
const Algo = Yaqs.Algorithms
const MPSMod = Yaqs.MPSModule
const MPOMod = Yaqs.MPOModule
const Cfg = Yaqs.SimulationConfigs
const GL = Yaqs.GateLibrary

export parse_kv_args,
       mid_site,
       qutip_expect_z_site,
       dense_expect_z_site,
       run_method_expect_z_site,
       rms_error,
       min_abs_error,
       max_abs_error

@inline mid_site(L::Int) = (L + 1) ÷ 2

function parse_kv_args(args::Vector{String})
    # Minimal `--key=value` parser (also supports `--flag` => "true")
    out = Dict{String, String}()
    for a in args
        startswith(a, "--") || continue
        s = a[3:end]
        if occursin("=", s)
            k, v = split(s, "=", limit=2)
            out[k] = v
        else
            out[s] = "true"
        end
    end
    return out
end

function _dense_initial_state(L::Int, state::AbstractString)
    if state == "zeros"
        v1 = ComplexF64[1.0, 0.0]
    elseif state == "ones"
        v1 = ComplexF64[0.0, 1.0]
    elseif state == "x+"
        v1 = ComplexF64[1.0, 1.0] ./ sqrt(2)
    else
        error("Unsupported initial state: $state (supported: zeros, ones, x+)")
    end
    v = v1
    for _ in 2:L
        v = kron(v, v1)
    end
    return v
end

function _dense_op_on_site(L::Int, op::AbstractMatrix{<:Complex}, site::Int)
    @assert 1 ≤ site ≤ L
    I2 = Matrix{ComplexF64}(I, 2, 2)
    acc = (site == 1) ? ComplexF64.(op) : I2
    for s in 2:L
        acc = kron(acc, (s == site) ? ComplexF64.(op) : I2)
    end
    return acc
end

function _dense_ising_hamiltonian(L::Int; J::Real=1.0, g::Real=0.5)
    X = ComplexF64.(Matrix(GL.matrix(GL.XGate())))
    Z = ComplexF64.(Matrix(GL.matrix(GL.ZGate())))

    H = zeros(ComplexF64, 2^L, 2^L)
    for i in 1:(L - 1)
        Zi = _dense_op_on_site(L, Z, i)
        Zj = _dense_op_on_site(L, Z, i + 1)
        H .+= (-J) .* (Zi * Zj)
    end
    for i in 1:L
        Xi = _dense_op_on_site(L, X, i)
        H .+= (-g) .* Xi
    end
    return H
end

"""
    dense_expect_z_site(L; J, g, dt, steps, initial_state, site) -> (times, z)

Compute exact real-time evolution for the transverse-field Ising Hamiltonian
and return ⟨Z_site⟩ at each time point using dense matrices (feasible for small L).
"""
function dense_expect_z_site(L::Int;
                             J::Real,
                             g::Real,
                             dt::Real,
                             steps::Int,
                             initial_state::AbstractString="x+",
                             site::Int=mid_site(L))
    @assert steps ≥ 0
    times = collect(0:dt:(steps * dt))

    ψ = _dense_initial_state(L, initial_state)
    H = _dense_ising_hamiltonian(L; J=J, g=g)
    U = exp((-1im * dt) .* H)

    Z = ComplexF64.(Matrix(GL.matrix(GL.ZGate())))
    Zsite = _dense_op_on_site(L, Z, site)

    out = Vector{Float64}(undef, steps + 1)
    out[1] = real(dot(conj(ψ), Zsite * ψ))
    for k in 1:steps
        ψ = U * ψ
        out[k + 1] = real(dot(conj(ψ), Zsite * ψ))
    end
    return times, out
end

"""
    qutip_expect_z_site(L; J, g, dt, steps, initial_state, site) -> (times, z)

Compute exact real-time evolution using qutip (via PythonCall) and return ⟨Z_site⟩ at each time point.
This matches the reference used by the original Python scripts.
"""
function qutip_expect_z_site(L::Int;
                             J::Real,
                             g::Real,
                             dt::Real,
                             steps::Int,
                             initial_state::AbstractString="zeros",
                             site::Int=mid_site(L))
    @assert steps ≥ 0
    @assert 1 ≤ site ≤ L

    builtins = pyimport("builtins")
    np = pyimport("numpy")

    # Import qutip and helpers
    qutip = pyimport("qutip")
    basis = qutip.basis
    sigmax = qutip.sigmax
    sigmaz = qutip.sigmaz
    tensor = qutip.tensor
    qeye = qutip.qeye
    mesolve = qutip.mesolve

    sx = sigmax()
    sz = sigmaz()
    id2 = qeye(2)

    # Build operators acting on the full space
    sx_list = Py[]
    sz_list = Py[]
    for i in 1:L
        ops_x_jl = Py[ id2 for _ in 1:L ]
        ops_z_jl = Py[ id2 for _ in 1:L ]
        ops_x_jl[i] = sx
        ops_z_jl[i] = sz
        push!(sx_list, tensor(builtins.list(ops_x_jl)))
        push!(sz_list, tensor(builtins.list(ops_z_jl)))
    end

    # Hamiltonian H = -J Σ Z_i Z_{i+1} - g Σ X_i
    H = 0 * sz_list[1]  # ensure Python/qutip type
    for i in 1:(L - 1)
        H = H + (-J) * (sz_list[i] * sz_list[i + 1])
    end
    for i in 1:L
        H = H + (-g) * sx_list[i]
    end

    # Initial state
    if initial_state == "zeros"
        psi0 = tensor(builtins.list(Py[ basis(2, 0) for _ in 1:L ]))
    elseif initial_state == "ones"
        psi0 = tensor(builtins.list(Py[ basis(2, 1) for _ in 1:L ]))
    elseif initial_state == "x+"
        # Keep consistent with our Julia exp default: start from |+>^⊗L
        plus = (basis(2, 0) + basis(2, 1)) / sqrt(2)
        psi0 = tensor(builtins.list(Py[ plus for _ in 1:L ]))
    else
        error("Unsupported initial state for qutip: $initial_state (supported: zeros, ones, x+)")
    end

    times = collect(0:dt:(steps * dt))
    tlist = Py(times)

    # Expectation values: qutip wants list of operators
    eops = builtins.list(Py[ sz_list[site] ])
    # Modern qutip prefers options as a Python dict.
    opts = builtins.dict()
    opts["store_states"] = false
    c_ops = builtins.list()
    res = mesolve(H, psi0, tlist, c_ops, eops; options=opts)
    # res.expect[0] is the first eop (python indexing)
    exp0 = res.expect[0]
    z = Vector{Float64}(pyconvert(Vector{Float64}, np.real(exp0)))

    return times, z
end

function _method_from_string(s::AbstractString)
    if s == "ADAPTIVE" || s == "bug"
        return :bug
    elseif s == "FIXED" || s == "fixed_bug"
        return :fixed_bug
    elseif s == "DOUBLEADAPTIVE" || s == "bug_second_order"
        return :bug_second_order
    elseif s == "DOUBLEFIXED" || s == "fixed_bug_second_order"
        return :fixed_bug_second_order
    elseif s == "HYBRID" || s == "hybrid_bug_second_order"
        return :hybrid_bug_second_order
    elseif s == "SINGLE_SITE_TDVP" || s == "single_site_tdvp"
        return :single_site_tdvp
    elseif s == "TWO_SITE_TDVP" || s == "two_site_tdvp"
        return :two_site_tdvp
    else
        error("Unknown method: $s")
    end
end

"""
    run_method_expect_z_site(method; ...) -> (times, z, wall_s)

Run one method (BUG variants or TDVP) for a small Ising chain and return ⟨Z_site⟩ vs time.
"""
function run_method_expect_z_site(method::Union{Symbol, AbstractString};
                                  L::Int,
                                  J::Real,
                                  g::Real,
                                  dt::Real,
                                  steps::Int,
                                  initial_state::AbstractString="x+",
                                  site::Int=mid_site(L),
                                  max_bond_dim::Int=128,
                                  threshold::Real=1e-12,
                                  numiter_lanczos::Int=25)
    method_sym = method isa AbstractString ? _method_from_string(method) : method

    times = collect(0:dt:(steps * dt))
    Z = ComplexF64.(Matrix(GL.matrix(GL.ZGate())))

    ψ = MPSMod.MPS(L; state=initial_state)
    H = MPOMod.init_ising(L, float(J), float(g))
    cfg = Cfg.TimeEvolutionConfig(Cfg.Observable[], float(dt); dt=float(dt),
                                  max_bond_dim=max_bond_dim,
                                  truncation_threshold=float(threshold),
                                  sample_timesteps=false)

    # Mirror the Python harness: pad for fixed methods and 1-site TDVP.
    if method_sym in (:fixed_bug, :fixed_bug_second_order, :single_site_tdvp)
        MPSMod.pad_bond_dimension!(ψ, max_bond_dim; noise_scale=1e-10)
    end

    out = Vector{Float64}(undef, steps + 1)
    out[1] = real(MPSMod.local_expect(ψ, Z, site))

    t0 = time()
    for k in 1:steps
        if method_sym === :bug
            BUG.bug!(ψ, H, cfg; numiter_lanczos=numiter_lanczos)
        elseif method_sym === :fixed_bug
            BUG.fixed_bug!(ψ, H, cfg; numiter_lanczos=numiter_lanczos)
            MPSMod.truncate!(ψ; threshold=cfg.truncation_threshold, max_bond_dim=cfg.max_bond_dim)
        elseif method_sym === :bug_second_order
            BUG.bug_second_order!(ψ, H, cfg; numiter_lanczos=numiter_lanczos)
        elseif method_sym === :fixed_bug_second_order
            BUG.fixed_bug_second_order!(ψ, H, cfg; numiter_lanczos=numiter_lanczos)
            MPSMod.truncate!(ψ; threshold=cfg.truncation_threshold, max_bond_dim=cfg.max_bond_dim)
        elseif method_sym === :hybrid_bug_second_order
            BUG.hybrid_bug_second_order!(ψ, H, cfg; numiter_lanczos=numiter_lanczos)
        elseif method_sym === :single_site_tdvp
            Algo.single_site_tdvp!(ψ, H, cfg; numiter_lanczos=numiter_lanczos)
            MPSMod.truncate!(ψ; threshold=cfg.truncation_threshold, max_bond_dim=cfg.max_bond_dim)
        elseif method_sym === :two_site_tdvp
            Algo.two_site_tdvp!(ψ, H, cfg; numiter_lanczos=numiter_lanczos)
            MPSMod.truncate!(ψ; threshold=cfg.truncation_threshold, max_bond_dim=cfg.max_bond_dim)
        else
            error("Unknown method_sym: $method_sym")
        end
        out[k + 1] = real(MPSMod.local_expect(ψ, Z, site))
    end
    wall = time() - t0

    return times, out, wall
end

@inline function rms_error(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; skip::Int=0)
    @assert length(x) == length(y)
    i0 = 1 + skip
    return sqrt(mean((x[i0:end] .- y[i0:end]) .^ 2))
end

@inline function min_abs_error(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; skip::Int=0)
    @assert length(x) == length(y)
    i0 = 1 + skip
    return minimum(abs.(x[i0:end] .- y[i0:end]))
end

@inline function max_abs_error(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; skip::Int=0)
    @assert length(x) == length(y)
    i0 = 1 + skip
    return maximum(abs.(x[i0:end] .- y[i0:end]))
end

end # module BUGExpUtil

