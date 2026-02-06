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

const _qutip_mods = Ref{Any}(nothing)
const _qutip_cache = Dict{Tuple{Int,String}, Any}()  # (L, initial_state) -> psi0
const _qutip_ops_cache = Dict{Int, Any}()           # L -> (sx_list, sy_list, sz_list)

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
    # Minimal `--key=value` parser (also supports `--flag` => "true").
    # Also accepts `key=value` / `flag` (useful when calling `main([...])`).
    out = Dict{String, String}()
    for a in args
        s = if startswith(a, "--")
            a[3:end]
        elseif startswith(a, "-")
            continue
        else
            a
        end

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

@inline function _get_param(p, i::Int)
    return isa(p, Vector) ? (i <= Base.length(p) ? p[i] : 0.0) : p
end

function _mpo_hamiltonian(model::AbstractString, L::Int;
                          # TFIM params
                          J::Real=1.0, g::Real=0.5,
                          # General params (used by `general`)
                          Jxx::Real=0.0, Jyy::Real=0.0, Jzz::Real=0.0,
                          hx::Real=0.0, hy::Real=0.0, hz::Real=0.0,
                          # Convenience params for common models
                          Delta::Real=1.0, gamma::Real=0.0)
    m = lowercase(strip(model))
    if m in ("tfim", "ising")
        return MPOMod.init_ising(L, float(J), float(g))
    elseif m in ("heisenberg", "xxx")
        return MPOMod.init_general_hamiltonian(L, float(J), float(J), float(J), float(hx), float(hy), float(hz))
    elseif m in ("xx",)
        return MPOMod.init_general_hamiltonian(L, float(J), float(J), 0.0, float(hx), float(hy), float(hz))
    elseif m in ("xy",)
        Jx = float(J) * (1 + float(gamma))
        Jy = float(J) * (1 - float(gamma))
        return MPOMod.init_general_hamiltonian(L, Jx, Jy, 0.0, float(hx), float(hy), float(hz))
    elseif m in ("xxz",)
        return MPOMod.init_general_hamiltonian(L, float(J), float(J), float(Delta) * float(J), float(hx), float(hy), float(hz))
    elseif m in ("general",)
        return MPOMod.init_general_hamiltonian(L, float(Jxx), float(Jyy), float(Jzz), float(hx), float(hy), float(hz))
    else
        error("Unsupported model=$model (supported: tfim/ising, xx, xy, xxz, heisenberg, general)")
    end
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
                             model::AbstractString="tfim",
                             # TFIM params
                             J::Real=1.0,
                             g::Real=0.5,
                             # General params (used by `general`)
                             Jxx::Real=0.0,
                             Jyy::Real=0.0,
                             Jzz::Real=0.0,
                             hx::Real=0.0,
                             hy::Real=0.0,
                             hz::Real=0.0,
                             # Convenience params for common models
                             Delta::Real=1.0,
                             gamma::Real=0.0,
                             dt::Real,
                             steps::Int,
                             initial_state::AbstractString="zeros",
                             site::Int=mid_site(L))
    @assert steps ≥ 0
    @assert 1 ≤ site ≤ L

    mods = _qutip_mods[]
    if mods === nothing
        builtins = pyimport("builtins")
        np = pyimport("numpy")
        qutip = pyimport("qutip")
        mods = (;
            builtins,
            np,
            qutip,
            basis = qutip.basis,
            tensor = qutip.tensor,
            qeye = qutip.qeye,
            sigmax = qutip.sigmax,
            sigmay = qutip.sigmay,
            sigmaz = qutip.sigmaz,
            sesolve = qutip.sesolve,
        )
        _qutip_mods[] = mods
    end

    builtins = mods.builtins
    np = mods.np
    basis = mods.basis
    tensor = mods.tensor
    qeye = mods.qeye
    sigmax = mods.sigmax
    sigmay = mods.sigmay
    sigmaz = mods.sigmaz
    sesolve = mods.sesolve

    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    id2 = qeye(2)

    # Build operators acting on the full space (cached per L).
    ops = get(_qutip_ops_cache, L, nothing)
    if ops === nothing
        sx_list = Py[]
        sy_list = Py[]
        sz_list = Py[]
        for i in 1:L
            ops_x_jl = Py[id2 for _ in 1:L]
            ops_y_jl = Py[id2 for _ in 1:L]
            ops_z_jl = Py[id2 for _ in 1:L]
            ops_x_jl[i] = sx
            ops_y_jl[i] = sy
            ops_z_jl[i] = sz
            push!(sx_list, tensor(builtins.list(ops_x_jl)))
            push!(sy_list, tensor(builtins.list(ops_y_jl)))
            push!(sz_list, tensor(builtins.list(ops_z_jl)))
        end
        ops = (; sx_list, sy_list, sz_list)
        _qutip_ops_cache[L] = ops
    end

    sx_list = ops.sx_list
    sy_list = ops.sy_list
    sz_list = ops.sz_list

    m = lowercase(strip(model))
    H = 0 * sz_list[1]  # ensure Python/qutip type
    if m in ("tfim", "ising")
        # H = -J Σ Z_i Z_{i+1} - g Σ X_i
        for i in 1:(L - 1)
            H = H + (-J) * (sz_list[i] * sz_list[i + 1])
        end
        for i in 1:L
            H = H + (-g) * sx_list[i]
        end
    else
        # General Hamiltonian:
        # H = Σ_i (Jxx X_i X_{i+1} + Jyy Y_i Y_{i+1} + Jzz Z_i Z_{i+1})
        #   + Σ_i (hx X_i + hy Y_i + hz Z_i)
        local Jxx_eff, Jyy_eff, Jzz_eff, hx_eff, hy_eff, hz_eff
        if m in ("heisenberg", "xxx")
            Jxx_eff, Jyy_eff, Jzz_eff = J, J, J
            hx_eff, hy_eff, hz_eff = hx, hy, hz
        elseif m in ("xx",)
            Jxx_eff, Jyy_eff, Jzz_eff = J, J, 0.0
            hx_eff, hy_eff, hz_eff = hx, hy, hz
        elseif m in ("xy",)
            Jxx_eff, Jyy_eff, Jzz_eff = J * (1 + gamma), J * (1 - gamma), 0.0
            hx_eff, hy_eff, hz_eff = hx, hy, hz
        elseif m in ("xxz",)
            Jxx_eff, Jyy_eff, Jzz_eff = J, J, Delta * J
            hx_eff, hy_eff, hz_eff = hx, hy, hz
        elseif m in ("general",)
            Jxx_eff, Jyy_eff, Jzz_eff = Jxx, Jyy, Jzz
            hx_eff, hy_eff, hz_eff = hx, hy, hz
        else
            error("Unsupported model=$model (supported: tfim/ising, xx, xy, xxz, heisenberg, general)")
        end

        for i in 1:(L - 1)
            if Jxx_eff != 0
                H = H + Jxx_eff * (sx_list[i] * sx_list[i + 1])
            end
            if Jyy_eff != 0
                H = H + Jyy_eff * (sy_list[i] * sy_list[i + 1])
            end
            if Jzz_eff != 0
                H = H + Jzz_eff * (sz_list[i] * sz_list[i + 1])
            end
        end
        for i in 1:L
            if hx_eff != 0
                H = H + hx_eff * sx_list[i]
            end
            if hy_eff != 0
                H = H + hy_eff * sy_list[i]
            end
            if hz_eff != 0
                H = H + hz_eff * sz_list[i]
            end
        end
    end

    # Initial state
    psi0 = get(_qutip_cache, (L, initial_state), nothing)
    if psi0 === nothing
        if initial_state == "zeros"
            psi0 = tensor(builtins.list(Py[basis(2, 0) for _ in 1:L]))
        elseif initial_state == "ones"
            psi0 = tensor(builtins.list(Py[basis(2, 1) for _ in 1:L]))
        elseif initial_state == "x+"
            plus = (basis(2, 0) + basis(2, 1)) / sqrt(2)
            psi0 = tensor(builtins.list(Py[plus for _ in 1:L]))
        elseif initial_state == "Neel"
            # Match `MPSModule.MPS(...; state="Neel")`: |0 1 0 1 ...> with site 1 = |0>
            jl = Py[]
            for i in 1:L
                push!(jl, isodd(i) ? basis(2, 0) : basis(2, 1))
            end
            psi0 = tensor(builtins.list(jl))
        else
            error("Unsupported initial state for qutip: $initial_state (supported: zeros, ones, x+, Neel)")
        end
        _qutip_cache[(L, initial_state)] = psi0
    end

    times = collect(0:dt:(steps * dt))
    tlist = Py(times)

    # Expectation values: qutip wants list of operators
    eops = builtins.list(Py[ sz_list[site] ])
    # Options as Python dict.
    opts = builtins.dict()
    opts["store_states"] = false
    res = sesolve(H, psi0, tlist, eops; options=opts)
    exp0 = res.expect[0] # first eop (python indexing)
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
                                  model::AbstractString="tfim",
                                  # TFIM params
                                  J::Real=1.0,
                                  g::Real=0.5,
                                  # General params (used by `general`)
                                  Jxx::Real=0.0,
                                  Jyy::Real=0.0,
                                  Jzz::Real=0.0,
                                  hx::Real=0.0,
                                  hy::Real=0.0,
                                  hz::Real=0.0,
                                  # Convenience params for common models
                                  Delta::Real=1.0,
                                  gamma::Real=0.0,
                                  dt::Real,
                                  steps::Int,
                                  initial_state::AbstractString="x+",
                                  site::Int=mid_site(L),
                                  max_bond_dim::Int=128,
                                  threshold::Real=1e-12,
                                  numiter_lanczos::Int=25,
                                  track_bond_dims::Bool=false)
    method_sym = method isa AbstractString ? _method_from_string(method) : method

    times = collect(0:dt:(steps * dt))
    Z = ComplexF64.(Matrix(GL.matrix(GL.ZGate())))

    ψ = MPSMod.MPS(L; state=initial_state)
    H = _mpo_hamiltonian(model, L;
                         J=J, g=g,
                         Jxx=Jxx, Jyy=Jyy, Jzz=Jzz,
                         hx=hx, hy=hy, hz=hz,
                         Delta=Delta, gamma=gamma)
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

    bond_dims = track_bond_dims ? Vector{Int}(undef, steps + 1) : Int[]
    if track_bond_dims
        bond_dims[1] = MPSMod.write_max_bond_dim(ψ)
    end

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
        if track_bond_dims
            bond_dims[k + 1] = MPSMod.write_max_bond_dim(ψ)
        end
    end
    wall = time() - t0

    return track_bond_dims ? (times, out, bond_dims, wall) : (times, out, wall)
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

