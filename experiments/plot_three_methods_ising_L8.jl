using LinearAlgebra
using Printf
using DelimitedFiles

# Minimal subset without PythonCall/CircuitIngestion (avoids CondaPkg/pixi init)
module YaqsLite
module Timing
export enable_timing!, set_timing_print_each_call!, reset_timing!, print_timing_summary!, begin_scope!, end_scope!, @t
enable_timing!(::Bool=true) = nothing
set_timing_print_each_call!(::Bool=true) = nothing
reset_timing!() = nothing
print_timing_summary!(; header::AbstractString="Timing summary", top::Int=20) = nothing
begin_scope!() = nothing
end_scope!(::Any; header::AbstractString="Timing scope") = nothing
macro t(_key, ex)
    return esc(ex)
end
end # Timing

include(joinpath(@__DIR__, "..", "src", "GateLibrary.jl"))
include(joinpath(@__DIR__, "..", "src", "Decompositions.jl"))
include(joinpath(@__DIR__, "..", "src", "MPS.jl"))
include(joinpath(@__DIR__, "..", "src", "MPO.jl"))
include(joinpath(@__DIR__, "..", "src", "SimulationConfigs.jl"))
include(joinpath(@__DIR__, "..", "src", "Algorithms.jl"))
include(joinpath(@__DIR__, "..", "src", "BUG.jl"))

using .GateLibrary
using .Decompositions
using .MPSModule
using .MPOModule
using .SimulationConfigs
using .Algorithms
using .BUGModule
end # module YaqsLite

const MPSMod = YaqsLite.MPSModule
const MPOMod = YaqsLite.MPOModule
const Algo   = YaqsLite.Algorithms
const BUG    = YaqsLite.BUGModule
const GL     = YaqsLite.GateLibrary
const Cfg    = YaqsLite.SimulationConfigs

# ----------------------------
# Problem setup
# ----------------------------

const L = 8
const steps = 1000
const dt = 1e-2
const J = 1.0
const g = 0.7
const init_state = "Neel"  # "zeros" | "ones" | "x+" | "Neel"

const chosen_sites = [1, 4, 8]  # change as you like (1-based)

function main()
@printf("Model: Ising L=%d, steps=%d, dt=%.3g, J=%.3g, g=%.3g\n", L, steps, dt, J, g)
@printf("Initial state: %s\n", init_state)
@printf("Chosen sites for <Z>: %s\n", string(chosen_sites))

# ----------------------------
# Helpers
# ----------------------------

@inline function max_bond_dim(psi::MPSMod.MPS)
    return MPSMod.write_max_bond_dim(psi)
end

function _basis_string_from_int(b::Int, L::Int)
    chars = Vector{Char}(undef, L)
    @inbounds for i in 1:L
        bit = (b >>> (i - 1)) & 0x1
        chars[i] = (bit == 1) ? '1' : '0'
    end
    return String(chars)
end

function mpo_to_dense(mpo::MPOMod.MPO{ComplexF64})
    L = mpo.length
    dim = 2^L
    H = Matrix{ComplexF64}(undef, dim, dim)
    for col in 0:(dim - 1)
        bs = _basis_string_from_int(col, L)
        ψ = MPSMod.MPS(L; state="basis", basis_string=bs)
        Hψ = MPOMod.contract_mpo_mps(mpo, ψ)
        H[:, col + 1] .= MPSMod.to_vec(Hψ)
    end
    return H
end

function z_eigs_for_sites(L::Int, sites::Vector{Int})
    dim = 2^L
    Zs = Vector{Vector{Float64}}(undef, length(sites))
    for (k, s) in pairs(sites)
        z = Vector{Float64}(undef, dim)
        @inbounds for b in 0:(dim - 1)
            bit = (b >>> (s - 1)) & 0x1
            z[b + 1] = (bit == 1) ? -1.0 : 1.0
        end
        Zs[k] = z
    end
    return Zs
end

function expZ_exact_diag(ψ::Vector{ComplexF64}, Zs::Vector{Vector{Float64}})
    p = abs2.(ψ)
    out = Vector{Float64}(undef, length(Zs))
    for k in eachindex(Zs)
        out[k] = sum(Zs[k] .* p)
    end
    return out
end

function expZ_mps(psi::MPSMod.MPS, Z::AbstractMatrix{ComplexF64}, sites::Vector{Int})
    out = Vector{Float64}(undef, length(sites))
    # local_expect shifts center, so operate on a copy to avoid interfering with time evolution center tracking
    tmp = deepcopy(psi)
    for (k, s) in pairs(sites)
        out[k] = real(MPSMod.local_expect(tmp, Z, s))
    end
    return out
end

# Write a simple CSV (header + numeric rows).
function write_csv(path::AbstractString, header::Vector{String}, data::AbstractMatrix{<:Real})
    open(path, "w") do io
        println(io, join(header, ","))
        for i in 1:size(data, 1)
            @inbounds println(io, join(data[i, :], ","))
        end
    end
    return nothing
end

# ----------------------------
# Build Hamiltonian and initial states
# ----------------------------

H_mpo = MPOMod.init_ising(L, J, g)
Zmat = Matrix(GL.matrix(GL.ZGate()))

function exact_product_state(L::Int, state::AbstractString)
    if state == "x+"
        v = ComplexF64[1, 1] ./ sqrt(2)
    elseif state == "zeros"
        v = ComplexF64[1, 0]
    elseif state == "ones"
        v = ComplexF64[0, 1]
    else
        error("Unsupported exact product state: $state")
    end
    # site 1 is rightmost in tensor product
    ψ = ComplexF64[1.0]
    for _ in 1:L
        ψ = kron(v, ψ)
    end
    return ψ
end

function exact_neel_state(L::Int)
    # Matches MPS constructor state="Neel": odd sites -> |0>, even sites -> |1>
    v0 = ComplexF64[1, 0]
    v1 = ComplexF64[0, 1]
    # site 1 is rightmost in tensor product
    ψ = ComplexF64[1.0]
    for i in L:-1:1
        v = isodd(i) ? v0 : v1
        ψ = kron(v, ψ)
    end
    return ψ
end

psi_tdvp = MPSMod.MPS(L; state=init_state)
psi_bug  = deepcopy(psi_tdvp)

# Exact reference built in the *same basis/order* as MPS (`to_vec`) and from the MPO itself.
@printf("Building dense H from MPO (dim=%d)...\n", 2^L)
H_dense = mpo_to_dense(H_mpo)
U_dense = exp(-1im * dt * H_dense)
ψ_exact = MPSMod.to_vec(deepcopy(psi_tdvp))
Zs = z_eigs_for_sites(L, chosen_sites)

cfg = Cfg.TimeEvolutionConfig(Cfg.Observable[], steps * dt;
                              dt=dt,
                              max_bond_dim=256,
                              truncation_threshold=1e-12,
                              order=2)

# ----------------------------
# Run evolutions and record data
# ----------------------------

times = collect(0:dt:(steps * dt))
nsamp = length(times)

z_exact = zeros(Float64, nsamp, length(chosen_sites))
z_tdvp  = zeros(Float64, nsamp, length(chosen_sites))
z_bug   = zeros(Float64, nsamp, length(chosen_sites))
err2_tdvp = zeros(Float64, nsamp, length(chosen_sites))
err2_bug  = zeros(Float64, nsamp, length(chosen_sites))

chi_tdvp = zeros(Int, nsamp)
chi_bug  = zeros(Int, nsamp)

# t=0
z_exact[1, :] .= expZ_exact_diag(ψ_exact, Zs)
z_tdvp[1, :]  .= expZ_mps(psi_tdvp, Zmat, chosen_sites)
z_bug[1, :]   .= expZ_mps(psi_bug,  Zmat, chosen_sites)
err2_tdvp[1, :] .= (z_tdvp[1, :] .- z_exact[1, :]).^2
err2_bug[1, :]  .= (z_bug[1, :]  .- z_exact[1, :]).^2
chi_tdvp[1] = max_bond_dim(psi_tdvp)
chi_bug[1]  = max_bond_dim(psi_bug)

@printf("\nEvolving...\n")

# Exact
t0 = time_ns()
ψ = copy(ψ_exact)
for n in 1:steps
    ψ = U_dense * ψ
    z_exact[n+1, :] .= expZ_exact_diag(ψ, Zs)
end
t_exact = (time_ns() - t0) / 1e9

# 2-site TDVP
t0 = time_ns()
for n in 1:steps
    Algo.two_site_tdvp!(psi_tdvp, H_mpo, cfg)
    z_tdvp[n+1, :] .= expZ_mps(psi_tdvp, Zmat, chosen_sites)
    err2_tdvp[n+1, :] .= (z_tdvp[n+1, :] .- z_exact[n+1, :]).^2
    chi_tdvp[n+1] = max_bond_dim(psi_tdvp)
end
t_tdvp = (time_ns() - t0) / 1e9

# 2nd-order BUG (adaptive)
t0 = time_ns()
for n in 1:steps
    BUG.bug_second_order!(psi_bug, H_mpo, cfg; numiter_lanczos=25)
    z_bug[n+1, :] .= expZ_mps(psi_bug, Zmat, chosen_sites)
    err2_bug[n+1, :] .= (z_bug[n+1, :] .- z_exact[n+1, :]).^2
    chi_bug[n+1] = max_bond_dim(psi_bug)
end
t_bug = (time_ns() - t0) / 1e9

@printf("\nWalltimes (total, %d steps):\n", steps)
@printf("  exact   : %.3f s\n", t_exact)
@printf("  2TDVP   : %.3f s\n", t_tdvp)
@printf("  BUG2nd  : %.3f s\n", t_bug)

max_err2_tdvp = maximum(err2_tdvp)
max_err2_bug  = maximum(err2_bug)
@printf("\nMax squared abs error vs exact (over all chosen sites and times):\n")
@printf("  2TDVP : %.3e\n", max_err2_tdvp)
@printf("  BUG2nd: %.3e\n", max_err2_bug)

# ----------------------------
# Save CSV + plot via system python (avoids PythonCall/CondaPkg)
# ----------------------------

csv_path = joinpath(@__DIR__, "three_methods_ising_L$(L)_steps$(steps).csv")
png_path = joinpath(@__DIR__, "three_methods_ising_L$(L)_steps$(steps).png")

header = String["t", "chi_tdvp", "chi_bug"]
for s in chosen_sites
    push!(header, "z_exact_$s")
    push!(header, "z_tdvp_$s")
    push!(header, "z_bug_$s")
    push!(header, "err2_tdvp_$s")
    push!(header, "err2_bug_$s")
end

data = zeros(Float64, nsamp, length(header))
data[:, 1] .= times
data[:, 2] .= chi_tdvp
data[:, 3] .= chi_bug
col = 4
for k in 1:length(chosen_sites)
    data[:, col]     .= z_exact[:, k]; col += 1
    data[:, col]     .= z_tdvp[:,  k]; col += 1
    data[:, col]     .= z_bug[:,   k]; col += 1
    data[:, col]     .= err2_tdvp[:, k]; col += 1
    data[:, col]     .= err2_bug[:,  k]; col += 1
end

write_csv(csv_path, header, data)
@printf("\nWrote data to: %s\n", csv_path)

python_code = """
import numpy as np
import matplotlib.pyplot as plt

csv_path = r\"$csv_path\"
png_path = r\"$png_path\"
chosen_sites = $chosen_sites
L = $L
steps = $steps
dt = $dt
t_exact = $t_exact
t_tdvp = $t_tdvp
t_bug = $t_bug

arr = np.genfromtxt(csv_path, delimiter=',', names=True)
t = arr['t']

fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(3, 2)

ax1 = fig.add_subplot(gs[0, :])
for s in chosen_sites:
    ax1.plot(t, arr[f'z_exact_{s}'], '-',  label=f'exact  <Z_{s}>')
    ax1.plot(t, arr[f'z_tdvp_{s}'],  '--', label=f'2TDVP  <Z_{s}>')
    ax1.plot(t, arr[f'z_bug_{s}'],   ':',  label=f'BUG2nd <Z_{s}>')
ax1.set_xlabel('t')
ax1.set_ylabel('<Z>')
ax1.set_title(f'Local Z expectations (Ising L={L}, steps={steps}, dt={dt})')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8, ncol=3)

ax2 = fig.add_subplot(gs[1, 0])
for s in chosen_sites:
    ax2.semilogy(t, arr[f'err2_tdvp_{s}'], '--', label=f'2TDVP err^2 Z_{s}')
    ax2.semilogy(t, arr[f'err2_bug_{s}'],  ':', label=f'BUG2nd err^2 Z_{s}')
ax2.set_xlabel('t')
ax2.set_ylabel('squared abs error')
ax2.set_title('Squared error vs exact')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=8, ncol=2)

ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(t, arr['chi_tdvp'], label='2TDVP max chi')
ax3.plot(t, arr['chi_bug'],  label='BUG2nd max chi')
ax3.set_xlabel('t')
ax3.set_ylabel('max bond dim chi')
ax3.set_title('Bond dimension growth')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

ax4 = fig.add_subplot(gs[2, 1])
ax4.bar(['exact', '2TDVP', 'BUG2nd'], [t_exact, t_tdvp, t_bug])
ax4.set_ylabel('walltime (s)')
ax4.set_title('Walltime (total)')
ax4.grid(True, axis='y', alpha=0.3)

fig.tight_layout()
plt.savefig(png_path, dpi=200)
print('Saved plot to:', png_path)
"""

try
    run(`python3 -c $python_code`)
catch e
    @warn "Plotting via system python failed; data CSV was still written." exception=e
    @printf("To plot manually, run: python3 -c '<plot code inside experiments/plot_three_methods_ising_L8.jl>'\n")
end

end # main()

main()


