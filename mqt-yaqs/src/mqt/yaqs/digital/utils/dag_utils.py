from __future__ import annotations

from typing import TYPE_CHECKING, List, Union
import numpy as np

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit, DAGOpNode
    from ...core.libraries.gate_library import BaseGate

from ...core.libraries.gate_library import GateLibrary

def convert_dag_to_tensor_algorithm(node: Union[DAGOpNode, DAGCircuit]) -> List[BaseGate]:
    """
    Converts a Qiskit DAGOpNode to a list of YAQS Gates.
    Currently only supports single nodes (used in digital_tjm).
    """
    if not hasattr(node, "op"):
         # Fallback for DAGCircuit if ever called (not expected for digital_tjm)
         raise NotImplementedError("convert_dag_to_tensor_algorithm currently only supports DAGOpNode.")

    op = node.op
    name = op.name
    params = op.params
    
    # Name mapping for discrepancies
    name_map = {
        "u3": "u",
        "u1": "p",
        "cp": "cp", # CPhase
        "mcp": "cp", # Multicontrol phase?
    }
    
    yaqs_name = name_map.get(name, name)
    
    gate_cls = getattr(GateLibrary, yaqs_name, None)
    
    if gate_cls is None:
        # Handle special cases like S, Sdg, T, Tdg which map to Phase
        if name == "s":
            gate_cls = GateLibrary.p
            params = [np.pi / 2]
        elif name == "sdg":
            gate_cls = GateLibrary.p
            params = [-np.pi / 2]
        elif name == "t":
            gate_cls = GateLibrary.p
            params = [np.pi / 4]
        elif name == "tdg":
            gate_cls = GateLibrary.p
            params = [-np.pi / 4]
        else:
            raise ValueError(f"Gate '{name}' not supported by YAQS GateLibrary.")

    # Instantiate gate
    try:
        if len(params) > 0:
            gate = gate_cls(params)
        else:
            gate = gate_cls()
    except TypeError:
        # If initialization failed, check if we provided params to a gate that doesn't take them
        # or vice-versa.
        if len(params) > 0:
             # Maybe gate doesn't take params? (e.g. X, but we passed something?)
             # Try without
             gate = gate_cls()
        else:
             raise ValueError(f"Gate {name} requires parameters but none provided.")

    # Set sites
    # Assumes node.qargs contains Qubits with _index attribute (standard in Qiskit)
    sites = [q._index for q in node.qargs]
    gate.set_sites(sites)
    
    return [gate]

