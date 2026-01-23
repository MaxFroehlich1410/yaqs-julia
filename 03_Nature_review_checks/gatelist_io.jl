"""
Utilities to serialize a `DigitalCircuit` (from `src/DigitalTJM.jl`) into a compact,
backend-agnostic gate list that can be consumed by Python (Qiskit/TenPy).

CSV format (header included):

    op,qubits,params,label

- `op`: lowercase op name (e.g. `rx`, `cx`, `rzz`, `barrier`)
- `qubits`: `i` or `i;j` (1-based indices)
- `params`: empty or `p1` or `p1;p2;p3` (Float64 printed with full precision)
- `label`: only for `barrier`, otherwise empty

This is intentionally simple to parse without extra Julia dependencies.
"""

module GateListIO

export write_gatelist_csv

_join_ints(xs::AbstractVector{<:Integer}) = join(string.(xs), ";")
_join_floats(xs::Vector{Float64}) = join(string.(xs), ";")

function _op_record(op)::Tuple{String, Vector{Float64}, String}
    # returns (op_name, params, label)
    t = nameof(typeof(op))
    if t === :Barrier
        return ("barrier", Float64[], getfield(op, :label))
    elseif t === :IdGate
        return ("id", Float64[], "")
    elseif t === :XGate
        return ("x", Float64[], "")
    elseif t === :YGate
        return ("y", Float64[], "")
    elseif t === :ZGate
        return ("z", Float64[], "")
    elseif t === :HGate
        return ("h", Float64[], "")
    elseif t === :SGate
        return ("s", Float64[], "")
    elseif t === :TGate
        return ("t", Float64[], "")
    elseif t === :SdgGate
        return ("sdg", Float64[], "")
    elseif t === :TdgGate
        return ("tdg", Float64[], "")
    elseif t === :RxGate
        return ("rx", [getfield(op, :theta)], "")
    elseif t === :RyGate
        return ("ry", [getfield(op, :theta)], "")
    elseif t === :RzGate
        return ("rz", [getfield(op, :theta)], "")
    elseif t === :PhaseGate
        return ("p", [getfield(op, :theta)], "")
    elseif t === :UGate
        return ("u", [getfield(op, :theta), getfield(op, :phi), getfield(op, :lam)], "")
    elseif t === :CXGate
        return ("cx", Float64[], "")
    elseif t === :CYGate
        return ("cy", Float64[], "")
    elseif t === :CZGate
        return ("cz", Float64[], "")
    elseif t === :CHGate
        return ("ch", Float64[], "")
    elseif t === :CPhaseGate
        return ("cp", [getfield(op, :theta)], "")
    elseif t === :SWAPGate
        return ("swap", Float64[], "")
    elseif t === :iSWAPGate
        return ("iswap", Float64[], "")
    elseif t === :RxxGate
        return ("rxx", [getfield(op, :theta)], "")
    elseif t === :RyyGate
        return ("ryy", [getfield(op, :theta)], "")
    elseif t === :RzzGate
        return ("rzz", [getfield(op, :theta)], "")
    elseif t === :RaisingGate
        error("Cannot export non-unitary operator RaisingGate to gate list.")
    elseif t === :LoweringGate
        error("Cannot export non-unitary operator LoweringGate to gate list.")
    else
        error("Unsupported gate type for export: $(typeof(op))")
    end
end

"""
    write_gatelist_csv(circ, path)

Write `circ.gates` to CSV at `path`.
"""
function write_gatelist_csv(circ, path::AbstractString)
    open(path, "w") do io
        println(io, "op,qubits,params,label")
        for g in getfield(circ, :gates)
            op_name, params, label = _op_record(g.op)
            qubits = _join_ints(g.sites)
            param_str = isempty(params) ? "" : _join_floats(params)
            # labels can contain commas? keep it simple: replace commas with spaces
            safe_label = replace(label, "," => " ")
            println(io, join((op_name, qubits, param_str, safe_label), ","))
        end
    end
    return path
end

end # module

