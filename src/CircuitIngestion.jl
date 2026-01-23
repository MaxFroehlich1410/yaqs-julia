module CircuitIngestion

using ..CircuitTJM
using ..GateLibrary

export ingest_qiskit_circuit, map_qiskit_name, convert_instruction_to_gate

@inline function _ensure_pythoncall()
    # Load PythonCall lazily so normal TJM runs don't pay CondaPkg/Pixi startup.
    # We attach it to `Main` to avoid world-age issues from injecting new globals into this module.
    if !isdefined(Main, :PythonCall)
        Base.eval(Main, :(using PythonCall))
    end
    return getfield(Main, :PythonCall)
end

"""
Convert a Qiskit circuit into a CircuitTJM circuit representation.

This ingests a `QuantumCircuit` from Python, converts it to a DAG, and groups front-layer gates into
commuting layers for CircuitTJM execution. Measurements and barriers are filtered or mapped based
on the ingestion logic.

Args:
    qc: Qiskit `QuantumCircuit` object (PythonCall `Py`).

Returns:
    DigitalCircuit: Circuit with layers and flat gate list populated.
"""
function ingest_qiskit_circuit(qc)
    PC = _ensure_pythoncall()
    qiskit = PC.pyimport("qiskit")
    dag_converter = PC.pyimport("qiskit.converters")
    
    # 1. Convert to DAG
    dag = dag_converter.circuit_to_dag(qc)
    
    # Determine number of qubits
    num_qubits = PC.pyconvert(Int, qc.num_qubits)
    
    # Create DigitalCircuit
    circ = DigitalCircuit(num_qubits)
    
    # 2. Process DAG into Layers
    # We replicate process_layer logic from Python, but instead of running simulation,
    # we store the gates into DigitalCircuit.layers.
    
    processed_layers = Vector{Vector{DigitalGate}}()
    
    # Loop while DAG has op nodes
    while PC.pyconvert(Int, PC.pybuiltins.len(dag.op_nodes())) > 0
        layer_gates = Vector{DigitalGate}()
        
        # Extract front layer
        # In Python: single, even, odd, barriers = process_layer(dag)
        # We implement the logic here in Julia using PythonCall.
        
        current_layer = dag.front_layer() # List of nodes
        
        single_nodes = []
        even_nodes = []
        odd_nodes = []
        
        nodes_to_remove = []
        
        for node in current_layer
            op = node.op
            name = PC.pyconvert(String, op.name)
            
            if name == "measure"
                push!(nodes_to_remove, node)
                continue
            end
            
            if name == "barrier"
                # Check label
                label = PC.pygetattr(op, "label", nothing)
                if !PC.pyis(label, nothing) && uppercase(PC.pyconvert(String, PC.pybuiltins.str(label))) == "SAMPLE_OBSERVABLES"
                    push!(nodes_to_remove, node)
                else
                    push!(nodes_to_remove, node)
                end
                continue
            end
            
            qargs = node.qargs
            num_q = length(qargs)
            
            if num_q == 1
                push!(single_nodes, node)
            elseif num_q == 2
                # These are not used in the logic below, but kept for parity with earlier comments.
                _ = PC.pyconvert(Int, qargs[0]._index) # 0-based
                _ = PC.pyconvert(Int, qargs[1]._index) # 0-based
                push!(even_nodes, node) # Just adding to list to process
            else
                error("Gates with >2 qubits not supported")
            end
        end
        
        # Process nodes found
        all_nodes = vcat(single_nodes, even_nodes, odd_nodes)
        
        if isempty(all_nodes) && isempty(nodes_to_remove)
            # If we have op nodes but front layer is empty or only contains things we don't handle?
            # If only measures/barriers left, we remove them and continue.
            if !PC.pyconvert(Bool, dag.op_nodes())
                break
            end
        end
        
        for node in all_nodes
            gate = convert_node_to_gate(node)
            push!(layer_gates, gate)
            push!(circ.gates, gate) # Populate flat list for process_circuit
            push!(nodes_to_remove, node)
        end
        
        # Remove from DAG
        for node in nodes_to_remove
            dag.remove_op_node(node)
        end
        
        if !isempty(layer_gates)
            push!(processed_layers, layer_gates)
        end
    end
    
    circ.layers = processed_layers
    return circ
end

"""
Convert a Qiskit CircuitInstruction into a DigitalGate.

This maps the Qiskit instruction name and parameters to a Julia gate, resolves qubit indices, and
constructs a `DigitalGate` with an optional generator. Unsupported gates return `nothing`.

Args:
    instr: Qiskit `CircuitInstruction` object (PythonCall `Py`).
    circuit: Qiskit `QuantumCircuit` owning the instruction (PythonCall `Py`), or `None`.

Returns:
    Union{DigitalGate, Nothing}: Converted gate or `nothing` if unsupported.
"""
function convert_instruction_to_gate(instr, circuit)
    PC = _ensure_pythoncall()
    op = instr.operation
    
    qubits_py = instr.qubits
    sites = Int[]
    
    function get_idx(q)
        if PC.pyhasattr(q, "_index")
            return PC.pyconvert(Int, q._index)
        else
            return PC.pyconvert(Int, circuit.find_bit(q).index)
        end
    end
    
    for q in qubits_py
        push!(sites, get_idx(q) + 1) # 1-based indexing
    end
    
    name = PC.pyconvert(String, op.name)
    params = [PC.pyconvert(Float64, p) for p in op.params]
    
    if name == "barrier"
        label = PC.pygetattr(op, "label", nothing)
        label_str = PC.pyis(label, nothing) ? "" : PC.pyconvert(String, PC.pybuiltins.str(label))
        if isempty(label_str)
            label_str = "SAMPLE_OBSERVABLES"
        end
        return DigitalGate(GateLibrary.Barrier(label_str), sites, nothing)
    end
    
    julia_op = nothing
    try
        julia_op = map_qiskit_name(name, params)
    catch
        return nothing
    end
    
    if isnothing(julia_op)
        return nothing
    end
    
    gen = nothing
    try
        gen = GateLibrary.generator(julia_op)
    catch
    end
    
    return DigitalGate(julia_op, sites, gen)
end

"""
Convert a Qiskit DAG node into a DigitalGate.

Args:
    node: Qiskit DAG node representing an operation (PythonCall `Py`).

Returns:
    DigitalGate: Converted gate for use in `DigitalCircuit`.
"""
function convert_node_to_gate(node)::DigitalGate
    PC = _ensure_pythoncall()
    op = node.op
    name = PC.pyconvert(String, op.name)
    params = [PC.pyconvert(Float64, p) for p in op.params]
    
    sites = Int[]
    for q in node.qargs
        push!(sites, PC.pyconvert(Int, q._index) + 1) # Convert to 1-based
    end
    
    julia_op = map_qiskit_name(name, params)
    
    gen = nothing
    try
        gen = GateLibrary.generator(julia_op)
    catch
    end
    
    return DigitalGate(julia_op, sites, gen)
end

"""
Map a Qiskit gate name and parameters to a Julia gate type.
"""
function map_qiskit_name(name::String, params::Vector{Float64})
    if name == "x" || name == "cx" && length(params) == 0 # CNOT is usually 'cx'
        if name == "cx" return CXGate() end
        return XGate()
    elseif name == "y"
        return YGate()
    elseif name == "z" || name == "cz"
        if name == "cz" return CZGate() end
        return ZGate()
    elseif name == "h"
        return HGate()
    elseif name == "id"
        return IdGate()
    elseif name == "s"
        return SGate()
    elseif name == "t"
        return TGate()
    elseif name == "sx"
        # SX = sqrt(X) = Rx(π/2)
        return RxGate(π/2)
    elseif name == "rx"
        return RxGate(params[1])
    elseif name == "ry"
        return RyGate(params[1])
    elseif name == "rz"
        return RzGate(params[1])
    elseif name == "p" || name == "phase" # Phase gate
        return PhaseGate(params[1])
    elseif name == "u" || name == "u3"
        return UGate(params[1], params[2], params[3])
    elseif name == "swap"
        return SWAPGate()
    elseif name == "rxx"
        return RxxGate(params[1])
    elseif name == "ryy"
        return RyyGate(params[1])
    elseif name == "rzz"
        return RzzGate(params[1])
    else
        error("Unsupported Qiskit gate: $name")
    end
end

end
