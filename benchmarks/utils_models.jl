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
- `"tfim"` / `"ising"`          : calls `init_ising(N, J, g)`
- `"general"`                   : calls `init_general_hamiltonian(N, Jxx, Jyy, Jzz, hx, hy, hz)`
- `"haldane_shastry"` / `"hs"` : calls `init_haldane_shastry(N; J, pbc)`
"""
function build_hamiltonian(model_name::AbstractString, N::Int;
                           J::Float64=1.0, g::Float64=1.05,
                           Jxx::Float64=0.0, Jyy::Float64=0.0, Jzz::Float64=0.0,
                           hx::Float64=0.0, hy::Float64=0.0, hz::Float64=0.0,
                           pbc::Bool=true)
    m = lowercase(strip(model_name))
    if m in ("tfim", "ising")
        return MPOMod.init_ising(N, J, g)
    elseif m == "general"
        return MPOMod.init_general_hamiltonian(N, Jxx, Jyy, Jzz, hx, hy, hz)
    elseif m in ("haldane_shastry", "hs")
        return MPOMod.init_haldane_shastry(N; J=J, pbc=pbc)
    else
        error("Unsupported model_name=$model_name (supported: tfim, general, haldane_shastry)")
    end
end

"""
    build_initial_state(model_name, N; state="") -> MPS

Return a product-state MPS. When `state` is non-empty it overrides the model
default; otherwise the model convention is used:
- TFIM                 → |+⟩^⊗N  (`"x+"`)
- general (XXZ+hz)     → Néel     (`"Neel"`)
- haldane_shastry / hs → Néel     (`"Neel"`)  or any MPS-supported state
"""
function build_initial_state(model_name::AbstractString, N::Int; state::String="")
    m = lowercase(strip(model_name))
    s = isempty(state) ? _default_state(m) : state
    return MPSMod.MPS(N; state=s)
end

function _default_state(m::AbstractString)
    if m in ("tfim", "ising")
        return "x+"
    elseif m == "general"
        return "Neel"
    elseif m in ("haldane_shastry", "hs")
        return "Neel"
    else
        error("No default initial state for model $m")
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
    elseif m in ("haldane_shastry", "hs")
        return "hs"
    else
        return m
    end
end

end # module BenchmarkModels
