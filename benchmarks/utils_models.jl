"""
    BenchmarkModels

Thin wrappers around the existing MPOModule Hamiltonian constructors
and MPSModule initial-state constructors.  No new physics here —
just a clean API for the benchmark runner.
"""
module BenchmarkModels

using Yaqs
const MPSMod = Yaqs.MPSModule
const MPOMod = Yaqs.MPOModule

export build_hamiltonian, build_initial_state, model_label

"""
    build_hamiltonian(model_name, N; kwargs...) -> MPO

Build an MPO Hamiltonian using the existing `MPOModule` constructors.

Supported `model_name` values:
- `"tfim"` / `"ising"` : calls `init_ising(N, J, g)`
- `"general"`          : calls `init_general_hamiltonian(N, Jxx, Jyy, Jzz, hx, hy, hz)`
"""
function build_hamiltonian(model_name::AbstractString, N::Int;
                           J::Float64=1.0, g::Float64=1.05,
                           Jxx::Float64=0.0, Jyy::Float64=0.0, Jzz::Float64=0.0,
                           hx::Float64=0.0, hy::Float64=0.0, hz::Float64=0.0)
    m = lowercase(strip(model_name))
    if m in ("tfim", "ising")
        return MPOMod.init_ising(N, J, g)
    elseif m == "general"
        return MPOMod.init_general_hamiltonian(N, Jxx, Jyy, Jzz, hx, hy, hz)
    else
        error("Unsupported model_name=$model_name (supported: tfim, general)")
    end
end

"""
    build_initial_state(model_name, N) -> MPS

Return a product-state MPS matching the model convention:
- TFIM  → |+>^⊗N
- general (XXZ+hz) → Néel |0101…>
"""
function build_initial_state(model_name::AbstractString, N::Int)
    m = lowercase(strip(model_name))
    if m in ("tfim", "ising")
        return MPSMod.MPS(N; state="x+")
    elseif m == "general"
        return MPSMod.MPS(N; state="Neel")
    else
        error("Unsupported model_name=$model_name for initial state")
    end
end

"""
    model_label(model_name) -> String

Short human-readable label for directory / filename use.
"""
function model_label(model_name::AbstractString)
    m = lowercase(strip(model_name))
    if m in ("tfim", "ising")
        return "tfim"
    elseif m == "general"
        return "xxz"
    else
        return m
    end
end

end # module BenchmarkModels
