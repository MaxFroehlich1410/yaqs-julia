#!/usr/bin/env julia
"""
2-qubit variance experiment (Julia / Yaqs DigitalTJM).

Replicates the intent of `01_PaperExps/2sites_variance_exp.py`:

- Two qubits, repeated "identity layer" (implemented as `RxxGate(0.0)` so the
  DigitalTJM noise hook is exercised every layer).
- Sparse Pauli-Lindblad noise on {X⊗I, I⊗X, X⊗X} with equal rates γ.
- Compare simulated trajectory variance of ⟨Z₁⟩, ⟨Z₂⟩ against closed-form formulas
  for different unravelings:
  - standard
  - projector
  - unitary_2pt (two-point law) with s = E[sin²θ] = 1/3
  - unitary_gauss (Gaussian law) tuned to match s = 1/3 for the chosen discretization

Run (from repo root):

    julia --project=. 01_PaperExps/2sites_variance_exp_digitaltjm.jl

This script uses PythonCall+matplotlib for plotting (no extra Julia plotting deps).
"""

using Random
using Statistics
using Printf
using Dates

using Yaqs
using Yaqs.GateLibrary
using Yaqs.MPSModule
using Yaqs.NoiseModule
using Yaqs.SimulationConfigs
using Yaqs.DigitalTJM
using Yaqs.Simulator

function _parse_args(args::Vector{String})
    d = Dict{String, String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            key = a[3:end]
            if i == length(args) || startswith(args[i+1], "--")
                d[key] = "true"
                i += 1
            else
                d[key] = args[i+1]
                i += 2
            end
        else
            i += 1
        end
    end
    return d
end

_getint(d, k, default) = haskey(d, k) ? parse(Int, d[k]) : default
_getfloat(d, k, default) = haskey(d, k) ? parse(Float64, d[k]) : default
_getstr(d, k, default) = haskey(d, k) ? d[k] : default
_getbool(d, k, default=false) = haskey(d, k) ? (lowercase(d[k]) in ("1", "true", "yes", "y", "on")) : default


# -----------------------------
# Theoretical formulas (from the python script)
# -----------------------------

"""
    theoretical_variances(t::AbstractVector{<:Real}, γ::Real; s=1/3) -> Dict{String, Vector{Float64}}

Return theoretical variance curves for ⟨Z_i⟩ in the 2-qubit IX/XI/XX noise model.
"""
function theoretical_variances(t::AbstractVector{<:Real}, γ::Real; s::Float64 = 1 / 3)
    tt = Float64.(t)
    gamma = Float64(γ)

    # mean² = exp(-8γ t)
    mean_sq = @. exp(-8.0 * gamma * tt)

    # Standard unraveling: Var = 1 - exp(-8γ t)
    var_std = @. 1.0 - exp(-8.0 * gamma * tt)

    # Projector unraveling: Var = exp(-4γ t) * (1 - exp(-4γ t))
    var_proj = @. exp(-4.0 * gamma * tt) * (1.0 - exp(-4.0 * gamma * tt))

    # Analog / unitary 2-point:
    # Var = 1/4 + 1/2 e^{-8γ(1-s)t} + 1/4 e^{-16γ(1-s)t} - e^{-8γ t}
    var_2pt = @. 0.25 + 0.5 * exp(-8.0 * gamma * (1.0 - s) * tt) + 0.25 * exp(-16.0 * gamma * (1.0 - s) * tt) - mean_sq

    # Analog / unitary Gaussian:
    # b = (γ/s) * (1 - (1 - 2s)^4) / 2
    one_minus_2s = 1.0 - 2.0 * s
    b = (gamma / max(s, 1e-16)) * (1.0 - one_minus_2s^4) / 2.0
    var_gauss = @. 0.25 + 0.5 * exp(-2.0 * b * tt) + 0.25 * exp(-4.0 * b * tt) - mean_sq

    return Dict(
        "standard" => var_std,
        "projector" => var_proj,
        "unitary_2pt" => var_2pt,
        "unitary_gauss" => var_gauss,
    )
end

"""
    mean_Z(t, γ) -> Vector{Float64}

Theoretical mean for ⟨Z_i⟩ under IX/XI/XX noise: E[⟨Z⟩] = exp(-4γ t).
"""
mean_Z(t::AbstractVector{<:Real}, γ::Real) = @. exp(-4.0 * Float64(γ) * Float64.(t))


# -----------------------------
# Matching the "s = 1/3" choices for unitary expansions
# -----------------------------

const S_TARGET = 1.0 / 3.0
const THETA0_2PT = asin(sqrt(S_TARGET)) # sin²(theta0) = 1/3

@inline function _gauss_s_weight(σ::Float64; M::Int = 11, k::Float64 = 4.0)
    # Mirrors the discretization in `NoiseModule.add_unitary_gauss_expansion!`
    theta_max = k * σ
    npos = (M + 1) ÷ 2
    thetas_pos = range(0.0, theta_max; length = npos)
    thetas = vcat(-reverse(collect(thetas_pos[2:end])), collect(thetas_pos))

    w = exp.(-0.5 .* (thetas ./ σ) .^ 2)
    w ./= sum(w)
    w = 0.5 .* (w .+ reverse(w))

    return sum(w .* (sin.(thetas) .^ 2))
end

function sigma_for_s_target(; s_target::Float64 = S_TARGET, M::Int = 11, k::Float64 = 4.0)
    # Find σ such that s_weight(σ) ≈ s_target via bisection.
    # s_weight(σ) is increasing for the ranges we care about.
    σ_lo = 1e-4
    σ_hi = 0.1
    s_lo = _gauss_s_weight(σ_lo; M=M, k=k)
    s_hi = _gauss_s_weight(σ_hi; M=M, k=k)

    while s_hi < s_target
        σ_hi *= 2
        s_hi = _gauss_s_weight(σ_hi; M=M, k=k)
        if σ_hi > 50
            error("Failed to bracket sigma for s_target=$(s_target). Got s_hi=$(s_hi) at σ_hi=$(σ_hi).")
        end
    end

    for _ in 1:80
        σ_mid = 0.5 * (σ_lo + σ_hi)
        s_mid = _gauss_s_weight(σ_mid; M=M, k=k)
        if s_mid < s_target
            σ_lo = σ_mid
            s_lo = s_mid
        else
            σ_hi = σ_mid
            s_hi = s_mid
        end
        if abs(s_mid - s_target) < 1e-6
            return σ_mid
        end
    end

    return 0.5 * (σ_lo + σ_hi)
end


# -----------------------------
# Circuit + simulation helpers
# -----------------------------

function build_identity_step_circuit(L::Int)
    # We add SAMPLE_OBSERVABLES barriers so DigitalTJM records results at:
    # - t=0 (barrier at start)
    # - after each layer (barrier after the 2q gate)
    step = DigitalCircuit(L)
    add_gate!(step, Barrier("SAMPLE_OBSERVABLES"), Int[]) # map idx 0
    add_gate!(step, RxxGate(0.0), [1, 2])
    add_gate!(step, Barrier("SAMPLE_OBSERVABLES"), Int[]) # map idx 1
    return step
end

function build_identity_step_circuit(L::Int, substeps::Int)
    @assert substeps >= 1
    step = DigitalCircuit(L)
    add_gate!(step, Barrier("SAMPLE_OBSERVABLES"), Int[]) # sample at macro-step start (t=0 or after previous macro)
    for _ in 1:substeps
        add_gate!(step, RxxGate(0.0), [1, 2]) # micro-step; triggers noise update
    end
    add_gate!(step, Barrier("SAMPLE_OBSERVABLES"), Int[]) # sample at macro-step end
    return step
end

function base_processes(gamma::Float64)
    return [
        Dict{String, Any}("name" => "pauli_x", "sites" => [1], "strength" => gamma),
        Dict{String, Any}("name" => "crosstalk_xx", "sites" => [1, 2], "strength" => gamma),
        Dict{String, Any}("name" => "pauli_x", "sites" => [2], "strength" => gamma),
    ]
end

function base_processes(gamma::Float64, substeps::Int)
    # DigitalTJM uses dt=1.0 internally for each gate's noise update.
    # To approximate continuous-time evolution over one macro-layer (Δt = 1),
    # we split it into `substeps` micro-steps with effective rate γ/substeps.
    γeff = gamma / substeps
    return [
        Dict{String, Any}("name" => "pauli_x", "sites" => [1], "strength" => γeff),
        Dict{String, Any}("name" => "crosstalk_xx", "sites" => [1, 2], "strength" => γeff),
        Dict{String, Any}("name" => "pauli_x", "sites" => [2], "strength" => γeff),
    ]
end

function processes_with_unraveling(gamma::Float64, unraveling::String; theta0::Float64 = THETA0_2PT, sigma::Float64 = 1.0, M::Int = 11, k::Float64 = 4.0)
    procs = base_processes(gamma)
    for p in procs
        p["unraveling"] = unraveling
        if unraveling == "unitary_2pt"
            p["theta0"] = theta0
        elseif unraveling == "unitary_gauss"
            p["sigma"] = sigma
            p["M"] = M
            p["gauss_k"] = k
        end
    end
    return procs
end

function processes_with_unraveling(gamma::Float64, substeps::Int, unraveling::String; theta0::Float64 = THETA0_2PT, sigma::Float64 = 1.0, M::Int = 11, k::Float64 = 4.0)
    procs = base_processes(gamma, substeps)
    for p in procs
        p["unraveling"] = unraveling
        if unraveling == "unitary_2pt"
            p["theta0"] = theta0
        elseif unraveling == "unitary_gauss"
            p["sigma"] = sigma
            p["M"] = M
            p["gauss_k"] = k
        end
    end
    return procs
end

function run_one_method(; L::Int, num_layers::Int, num_traj::Int, dt::Float64, gamma::Float64, method::String, theta0_2pt::Float64, sigma_gauss::Float64, gauss_M::Int, gauss_k::Float64, seed::Int, substeps::Int)
    Random.seed!(seed)

    # Initial |00⟩
    ψ0 = MPS(L; state="zeros")

    # Observables: ⟨Z₁⟩ and ⟨Z₂⟩
    obs = Observable[
        Observable("Z1", ZGate(), 1),
        Observable("Z2", ZGate(), 2),
    ]

    total_time = num_layers * dt
    sim = TimeEvolutionConfig(obs, total_time; dt=dt, num_traj=num_traj, sample_timesteps=true, max_bond_dim=64, truncation_threshold=1e-12)

    # Circuit: repeated identity layer
    step = build_identity_step_circuit(L, substeps)
    circ = RepeatedDigitalCircuit(step, num_layers)

    # Noise model
    if method == "standard"
        procs = base_processes(gamma, substeps)
        noise = NoiseModel(procs, L; dt=dt)
    elseif method == "projector"
        procs = processes_with_unraveling(gamma, substeps, "projector")
        noise = NoiseModel(procs, L; dt=dt)
    elseif method == "unitary_2pt"
        procs = processes_with_unraveling(gamma, substeps, "unitary_2pt"; theta0=theta0_2pt)
        noise = NoiseModel(procs, L; theta0=theta0_2pt, dt=dt)
    elseif method == "unitary_gauss"
        procs = processes_with_unraveling(gamma, substeps, "unitary_gauss"; sigma=sigma_gauss, M=gauss_M, k=gauss_k)
        noise = NoiseModel(procs, L; sigma=sigma_gauss, gauss_M=gauss_M, gauss_k=gauss_k, dt=dt)
    else
        error("Unknown method: $method")
    end

    # Run DigitalTJM trajectories and aggregate means into obs.results
    Simulator.run(ψ0, circ, sim, noise; parallel=true)

    # Extract per-qubit trajectories: size (num_traj, num_steps)
    Z1_traj = real.(obs[1].trajectories)
    Z2_traj = real.(obs[2].trajectories)

    # Population variance across trajectories at each time (dims=1 -> per-column)
    varZ1 = vec(var(Z1_traj; dims=1, corrected=false))
    varZ2 = vec(var(Z2_traj; dims=1, corrected=false))

    meanZ1 = obs[1].results
    meanZ2 = obs[2].results

    return (meanZ1=meanZ1, meanZ2=meanZ2, varZ1=varZ1, varZ2=varZ2)
end


function write_results_csv(path::AbstractString, t_layers::Vector{Float64}, gammas::Vector{Float64}, theo_list, sim_list)
    open(path, "w") do io
        println(io, "gamma,method,t,meanZ1,meanZ2,varZ1,varZ2,theory_mean,theory_var")
        methods = ["standard", "projector", "unitary_2pt", "unitary_gauss"]
        for (j, γ) in enumerate(gammas)
            theo = theo_list[j]
            sim = sim_list[j]
            theory_mean = mean_Z(t_layers, γ)
            for m in methods
                @assert length(theo[m]) == length(t_layers)
                s = sim[m]
                @assert length(s[:meanZ1]) == length(t_layers)
                for k in eachindex(t_layers)
                    println(
                        io,
                        string(γ), ",",
                        m, ",",
                        t_layers[k], ",",
                        s[:meanZ1][k], ",",
                        s[:meanZ2][k], ",",
                        s[:varZ1][k], ",",
                        s[:varZ2][k], ",",
                        theory_mean[k], ",",
                        theo[m][k],
                    )
                end
            end
        end
    end
    return nothing
end

function maybe_plot_with_pythoncall(t_layers::Vector{Float64}, gammas::Vector{Float64}, theo_list, sim_list; outfile::Union{Nothing,String}=nothing)
    # Optional plotting. If PythonCall/Matplotlib cannot initialize (e.g. CondaPkg/Pixi issues),
    # we fall back to CSV output.
    try
        # `import` must happen at top-level in Julia, so we do it via `@eval` and
        # then use the returned module object.
        pyc = @eval begin
            import PythonCall
            PythonCall
        end
        plt = pyc.pyimport("matplotlib.pyplot")

        fig, axs = plt.subplots(2, 3; figsize=(18, 8), sharex=true)
        colors = Dict(
            "standard" => "tab:blue",
            "projector" => "tab:red",
            "unitary_2pt" => "tab:green",
            "unitary_gauss" => "tab:orange",
            "theory_mean" => "black",
        )
        methods = ["standard", "projector", "unitary_2pt", "unitary_gauss"]

        for (j, γ) in enumerate(gammas)
            ax_var = axs[1, j]
            ax_exp = axs[2, j]

            ax_var.set_title(@sprintf("γ=%.3g", γ))
            ax_var.grid(true, alpha=0.3)
            ax_var.set_ylim(0, 1.1)
            if j == 1
                ax_var.set_ylabel("Variance")
            end

            ax_exp.grid(true, alpha=0.3)
            ax_exp.set_ylim(0, 1.1)
            ax_exp.set_xlabel("Physical Time (t)")
            if j == 1
                ax_exp.set_ylabel("⟨Z⟩")
            end

            theo = theo_list[j]
            sim = sim_list[j]

            # Variance (qubit 1 only, to match the python figure)
            for m in methods
                ax_var.plot(t_layers, theo[m], "--"; color=colors[m], linewidth=2, label=(j == 1 ? "$m (theory)" : nothing))
                ax_var.plot(t_layers, sim[m][:varZ1], "-"; color=colors[m], linewidth=1.5, alpha=0.85, label=(j == 1 ? "$m (sim)" : nothing))
            end

            # Mean expectation (qubit 1)
            ax_exp.plot(t_layers, mean_Z(t_layers, γ), "--"; color=colors["theory_mean"], linewidth=2, label=(j == 1 ? "theory" : nothing))
            for m in methods
                ax_exp.plot(t_layers, sim[m][:meanZ1], "-"; color=colors[m], linewidth=1.5, alpha=0.85, label=(j == 1 ? m : nothing))
            end
        end

        handles, labels = axs[1, 1].get_legend_handles_labels()
        fig.legend(handles, labels; loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout(rect=[0, 0, 0.88, 1])

        if outfile !== nothing
            fig.savefig(outfile; dpi=200)
            println("Saved plot to: $outfile")
        else
            plt.show()
        end
        return true
    catch err
        @warn "Plotting disabled (PythonCall/matplotlib failed to initialize). Writing CSV only." exception=(err, catch_backtrace())
        return false
    end
end

function _python_executable()
    # Prefer CondaPkg/Pixi python if present (reproducible), else fall back to system python.
    p = joinpath(@__DIR__, "..", ".CondaPkg", ".pixi", "envs", "default", "bin", "python")
    return isfile(p) ? p : "python3"
end

function plot_from_csv_with_python(csv_path::AbstractString, png_path::AbstractString; python::AbstractString=_python_executable())
    # Use a tiny inline python script with Agg backend (non-interactive) to avoid hangs.
    py = """
import csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = r\"\"\"$csv_path\"\"\"
PNG_PATH = r\"\"\"$png_path\"\"\"

rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

def f(x):
    try:
        return float(x)
    except Exception:
        return math.nan

methods = ["standard","projector","unitary_2pt","unitary_gauss"]
colors = {
    "standard": "tab:blue",
    "projector": "tab:red",
    "unitary_2pt": "tab:green",
    "unitary_gauss": "tab:orange",
    "theory_mean": "black",
}

# group by gamma, method
by = {}
for r in rows:
    g = f(r["gamma"])
    m = r["method"]
    by.setdefault(g, {}).setdefault(m, []).append(r)

gammas = sorted(by.keys(), reverse=False)
fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True)

for j, g in enumerate(gammas[:3]):
    ax_var = axs[0, j]
    ax_exp = axs[1, j]
    ax_var.set_title(f"γ={g:g}")
    ax_var.grid(True, alpha=0.3)
    ax_var.set_ylim(0, 1.1)
    if j == 0:
        ax_var.set_ylabel("Variance")
    ax_exp.grid(True, alpha=0.3)
    ax_exp.set_ylim(0, 1.1)
    ax_exp.set_xlabel("Physical Time (t)")
    if j == 0:
        ax_exp.set_ylabel("⟨Z⟩")

    # Mean theory curve (same for all methods) - plot last so it stays visible.
    t_theory = None
    mean_theory = None

    for m in methods:
        rs = sorted(by[g][m], key=lambda r: f(r["t"]))
        t = [f(r["t"]) for r in rs]
        var_sim = [f(r["varZ1"]) for r in rs]
        mean_sim = [f(r["meanZ1"]) for r in rs]
        var_theo = [f(r["theory_var"]) for r in rs]
        mean_theo = [f(r["theory_mean"]) for r in rs]
        t_theory = t
        mean_theory = mean_theo

        ax_var.plot(t, var_theo, "--", color=colors[m], linewidth=2, label=f"{m} (theory)" if j == 0 else None)
        ax_var.plot(t, var_sim, "-", color=colors[m], linewidth=1.5, alpha=0.85, label=f"{m} (sim)" if j == 0 else None)

        # Mean expectation (qubit 1) - simulation
        ax_exp.plot(t, mean_sim, "-", color=colors[m], linewidth=1.5, alpha=0.85, label=m if j == 0 else None)

    # Plot mean theory once per gamma, on top.
    if t_theory is not None and mean_theory is not None:
        ax_exp.plot(t_theory, mean_theory, "--", color=colors["theory_mean"], linewidth=2.2, zorder=10, label="theory" if j == 0 else None)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout(rect=[0, 0, 0.88, 1])
fig.savefig(PNG_PATH, dpi=200)
plt.close(fig)
print(f"Saved plot to: {PNG_PATH}")
"""
    Base.run(`$python -c $py`)
    return nothing
end


# -----------------------------
# Main
# -----------------------------

function main()
    argd = _parse_args(ARGS)
    if _getbool(argd, "help", false)
        println("""
Usage:
  julia --project=. 01_PaperExps/2sites_variance_exp_digitaltjm.jl [options]

Options:
  --traj N        number of trajectories (default: 10)
  --layers N      number of layers (default: 150)
  --dt DT         time step per layer (default: 1.0)
  --plot-out PATH save plot to PATH (default: 01_PaperExps/variance_comparison_digitaltjm.png)
  --no-plot       do not attempt matplotlib plotting

Notes:
  - For debugging, use --traj 10 to run quickly.
  - If theory-vs-sim mismatches persist, reduce --dt (and increase --layers if you want same total time).
""")
        return nothing
    end

    # Match python defaults
    L = 2
    num_layers = _getint(argd, "layers", 150)
    num_traj = _getint(argd, "traj", 10) # default reduced for fast iteration
    dt = _getfloat(argd, "dt", 1.0)
    substeps = _getint(argd, "substeps", 1)
    gammas = Float64[0.1, 0.01, 0.001]

    t_layers = collect(0.0:dt:(num_layers * dt))

    # Match the python script's analog choice s=1/3
    theta0 = THETA0_2PT
    gauss_M = 11
    gauss_k = 4.0
    sigma_gauss = sigma_for_s_target(; s_target=S_TARGET, M=gauss_M, k=gauss_k)

    @printf "2-qubit variance experiment (DigitalTJM)\n"
    @printf "  L=%d, layers=%d, dt=%.3g, num_traj=%d\n" L num_layers dt num_traj
    @printf "  methods: standard, projector, unitary_2pt, unitary_gauss\n"
    @printf "  substeps=%d (micro-steps per layer; set e.g. 20–100 to better match continuous-time theory at dt=1)\n" substeps
    @printf "  unitary_2pt: theta0=%.6f (sin^2=%.6f)\n" theta0 (sin(theta0)^2)
    @printf "  unitary_gauss: M=%d, k=%.3g, sigma≈%.6f (s_weight≈%.6f)\n" gauss_M gauss_k sigma_gauss _gauss_s_weight(sigma_gauss; M=gauss_M, k=gauss_k)

    theo_list = Vector{Dict{String, Vector{Float64}}}(undef, length(gammas))
    sim_list = Vector{Dict{String, NamedTuple}}(undef, length(gammas))

    methods = ["standard", "projector", "unitary_2pt", "unitary_gauss"]

    for (idx, γ) in enumerate(gammas)
        @printf "\n=== γ = %.3g ===\n" γ
        theo_list[idx] = theoretical_variances(t_layers, γ; s=S_TARGET)

        sim_for_gamma = Dict{String, NamedTuple}()
        for (mi, method) in enumerate(methods)
            # Vary seed per (γ, method) deterministically
            seed = 12345 + 1000 * idx + 10 * mi
            @printf "  running %-12s ...\n" method
            sim_for_gamma[method] = run_one_method(
                ; L=L,
                num_layers=num_layers,
                num_traj=num_traj,
                dt=dt,
                gamma=γ,
                method=method,
                theta0_2pt=theta0,
                sigma_gauss=sigma_gauss,
                gauss_M=gauss_M,
                gauss_k=gauss_k,
                seed=seed,
                substeps=substeps,
            )
        end
        sim_list[idx] = sim_for_gamma
    end

    # Always write CSV so results are usable even without plotting.
    ts = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    out_csv = joinpath(@__DIR__, "variance_comparison_digitaltjm_$ts.csv")
    write_results_csv(out_csv, t_layers, gammas, theo_list, sim_list)
    println("Wrote results CSV to: $out_csv")

    # Plot from CSV using external python (Agg backend). This avoids PythonCall hangs/segfaults.
    if !_getbool(argd, "no-plot", false)
        plot_out = _getstr(argd, "plot-out", joinpath(@__DIR__, "variance_comparison_digitaltjm.png"))
        plot_from_csv_with_python(out_csv, plot_out)
    end
    return nothing
end

main()

