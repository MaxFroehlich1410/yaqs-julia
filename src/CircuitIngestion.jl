module CircuitIngestion

using PythonCall
using ..CircuitTJM
using ..GateLibrary

export ingest_qiskit_circuit, map_qiskit_name, convert_instruction_to_gate

"""
Convert a Qiskit circuit into a CircuitTJM circuit representation.

This ingests a `QuantumCircuit` from Python, converts it to a DAG, and groups front-layer gates into
commuting layers for CircuitTJM execution. Measurements and barriers are filtered or mapped based
on the ingestion logic.

Args:
    qc (Py): Qiskit `QuantumCircuit` object.

Returns:
    DigitalCircuit: Circuit with layers and flat gate list populated.
"""
function ingest_qiskit_circuit(qc::Py)
    qiskit = pyimport("qiskit")
    dag_converter = pyimport("qiskit.converters")
    
    # 1. Convert to DAG
    dag = dag_converter.circuit_to_dag(qc)
    
    # Determine number of qubits
    num_qubits = pyconvert(Int, qc.num_qubits)
    
    # Create DigitalCircuit
    circ = DigitalCircuit(num_qubits)
    
    # 2. Process DAG into Layers
    # We replicate process_layer logic from Python, but instead of running simulation,
    # we store the gates into DigitalCircuit.layers.
    
    processed_layers = Vector{Vector{DigitalGate}}()
    
    # Loop while DAG has op nodes
    while pyconvert(Int, pybuiltins.len(dag.op_nodes())) > 0
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
            name = pyconvert(String, op.name)
            
            if name == "measure"
                push!(nodes_to_remove, node)
                continue
            end
            
            if name == "barrier"
                # Check label
                label = pygetattr(op, "label", nothing)
                if !pyis(label, nothing) && uppercase(pyconvert(String, pybuiltins.str(label))) == "SAMPLE_OBSERVABLES"
                    # Keep barrier? Python code returns it to trigger measurement.
                    # In this ingestion phase, we might want to store it as a "MeasurementGate" or split layers?
                    # The current CircuitTJM structure assumes standard time evolution.
                    # If we want mid-circuit measurements, we'd need to support them.
                    # For now, let's just drop barriers or treat them as layer separators if needed.
                    # The Python code removes them in the measurement phase.
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
                q0_idx = pyconvert(Int, qargs[0]._index) # 0-based
                q1_idx = pyconvert(Int, qargs[1]._index) # 0-based
                
                # Python logic: min(q0, q1) % 2 == 0 -> Even.
                # This corresponds to Bond 1 (0-1), Bond 3 (2-3).
                # In Julia (1-based): Bond 1 is (1,2). min(1,2)=1 (Odd).
                # So Python Even (0) -> Julia Odd min-index.
                
    # However, CircuitTJM.jl expects a flat list of gates for the layer,
                # and it does the even/odd splitting internally in `run_circuit_tjm`.
                # BUT `process_layer` in Python returns separate lists (single, even, odd) 
                # and processes them sequentially (single then even/odd).
                # It removes them from the DAG.
                # So we should collect ALL valid front-layer operations (single + appropriate 2-qubit),
                # convert them to DigitalGate, add to `layer_gates`, and remove from DAG.
                
                # Wait, Python `process_layer` separates Even/Odd *to be executed*.
                # It removes ALL of them from the DAG in that step?
                # Let's check python code:
                # `for node in group: ... dag.remove_op_node(node)`
                # Yes, it removes both even and odd groups in one "while loop" iteration (if they are in front layer).
                # BUT `front_layer()` only returns nodes that have NO dependencies.
                # If Even gate on (0,1) and Odd gate on (1,2) exist, can they both be in front layer?
                # No, because they share qubit 1. One must depend on other.
                # So `front_layer()` naturally handles commutation constraints.
                # If (0,1) and (2,3) are there, they commute.
                # So we can just take EVERYTHING from `front_layer()` (excluding measures/barriers)
                # and put them into one "Layer" in `DigitalCircuit`.
                # `CircuitTJM.run_circuit_tjm` will then split them into parallelizable groups if needed.
                
                # So here we just categorize to differentiate 2-qubit vs 1-qubit for our internal logic if needed,
                # but mostly we just convert everything.
                
                # One detail: Python code groups Even and Odd and runs them.
                # Does it run Even then Odd?
                # `for _, group in [("even", even_nodes), ("odd", odd_nodes)]: ...`
                # Yes.
                # Since they are in front_layer, they are disjoint (don't share qubits).
                # So order between Even/Odd groups in front layer doesn't matter?
                # Wait, if (0,1) and (1,2) are both in circuit.
                # (0,1) is Even. (1,2) is Odd.
                # Dependency: if (0,1) is first, (1,2) is not in front layer yet.
                # So `front_layer` guarantees disjointness on shared qubits?
                # Yes.
                # So we can just take all of them.
                
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
            if !pyconvert(Bool, dag.op_nodes())
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
    instr (Py): Qiskit `CircuitInstruction` object.
    circuit (Py): Qiskit `QuantumCircuit` owning the instruction.

Returns:
    Union{DigitalGate, Nothing}: Converted gate or `nothing` if unsupported.
"""
function convert_instruction_to_gate(instr::Py, circuit::Py)
    op = instr.operation
    
    # Try to extract qubits. instr.qubits is a tuple/list of Qubit objects.
    # Qubit objects usually have `_index`?
    # circuit.find_bit(q).index is safer if Qubit is detached, but usually it works.
    # We will try `_index` first, then fallback to `circuit.find_bit`.
    
    qubits_py = instr.qubits
    sites = Int[]
    
    """
    Resolve a Qiskit qubit object to its integer index.

    This prefers the `_index` attribute when available and falls back to `circuit.find_bit` when
    necessary, returning a zero-based index.

    Args:
        q (Py): Qiskit qubit object.

    Returns:
        Int: Zero-based qubit index.
    """
    function get_idx(q)
        # Try _index attribute
        if pyhasattr(q, "_index")
            return pyconvert(Int, q._index)
        else
            # Use circuit.find_bit
            return pyconvert(Int, circuit.find_bit(q).index)
        end
    end
    
    for q in qubits_py
        push!(sites, get_idx(q) + 1) # 1-based indexing
    end
    
    name = pyconvert(String, op.name)
    params = [pyconvert(Float64, p) for p in op.params]
    
    # Handle Barrier specifically
    if name == "barrier"
        label = pygetattr(op, "label", nothing)
        label_str = pyis(label, nothing) ? "" : pyconvert(String, pybuiltins.str(label))
        # Use SAMPLE_OBSERVABLES as default if empty?
        if isempty(label_str)
            label_str = "SAMPLE_OBSERVABLES" 
        end
        return DigitalGate(GateLibrary.Barrier(label_str), sites, nothing)
    end
    
    # Map Name
    julia_op = nothing
    try
        julia_op = map_qiskit_name(name, params)
    catch
        # If unknown gate, return nothing?
        # Benchmark scripts rely on returning nothing for unknown gates (like 'measure' or unsupported ones).
        return nothing
    end
    
    # Safety check: if julia_op is still nothing, return nothing
    if isnothing(julia_op)
        return nothing
    end
    
    # Generator
    gen = nothing
    try
        gen = GateLibrary.generator(julia_op)
    catch
    end
    
    return DigitalGate(julia_op, sites, gen)
end


"""
Convert a Qiskit DAG node into a DigitalGate.

This extracts the gate name, parameters, and qubit indices from a DAG node and maps them to the
corresponding Julia gate in the `GateLibrary`.

Args:
    node (Py): Qiskit DAG node representing an operation.

Returns:
    DigitalGate: Converted gate for use in `DigitalCircuit`.
"""
function convert_node_to_gate(node::Py)::DigitalGate
    op = node.op
    name = pyconvert(String, op.name)
    params = [pyconvert(Float64, p) for p in op.params]
    
    # Qubits
    # node.qargs is list of Qubit objects
    # q._index gives 0-based index
    sites = Int[]
    for q in node.qargs
        push!(sites, pyconvert(Int, q._index) + 1) # Convert to 1-based
    end
    
    # Map name to AbstractOperator
    julia_op = map_qiskit_name(name, params)
    
    # Create DigitalGate
    # We let DigitalGate constructor handle generator if needed (it calls GateLibrary.generator)
    # But we need to pass `nothing` if we want it to auto-generate.
    
    gen = nothing
    try
        gen = GateLibrary.generator(julia_op)
    catch
        # Generator not defined
    end
    
    return DigitalGate(julia_op, sites, gen)
end

"""
Map a Qiskit gate name and parameters to a Julia gate type.

This dispatches common Qiskit instruction names to the corresponding Julia gate constructors,
including parameterized rotations and two-qubit gates.

Args:
    name (String): Qiskit gate name.
    params (Vector{Float64}): Parameter list for the gate.

Returns:
    AbstractGate: Julia gate instance mapped from Qiskit.

Raises:
    ErrorException: If the gate name is unsupported.
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
