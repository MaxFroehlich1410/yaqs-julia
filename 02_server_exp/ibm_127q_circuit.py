import numpy as np
from collections import defaultdict
from qiskit.circuit import QuantumCircuit

def edge_colored_layers_from_backend(backend):
    """
    Returns a list of edge-disjoint layers [(q1,q2), ...] for the backend coupling graph.
    For heavy-hex (max degree 3), this yields 3 layers as used in the paper. :contentReference[oaicite:2]{index=2}
    """
    # rustworkx is a Qiskit dependency in typical installs
    import rustworkx as rx

    coupling_graph = backend.coupling_map.graph.to_undirected(multigraph=False)
    edge_coloring = rx.graph_bipartite_edge_color(coupling_graph)

    layer_couplings = defaultdict(list)
    for edge_idx, color in edge_coloring.items():
        layer_couplings[color].append(
            coupling_graph.get_edge_endpoints_by_index(edge_idx)
        )

    # deterministic ordering
    return [sorted(layer_couplings[i]) for i in sorted(layer_couplings.keys())]

def build_untwirled_kicked_ising_circuit(backend, num_trotter_steps, theta_h, theta_J=-np.pi/2, add_barriers=True):
    """
    Untwirled benchmark circuit from Kim et al. (Nature 618, 500–505 (2023)):
      for each step: RX(theta_h) on all qubits, then RZZ(theta_J) on all neighbor edges
    with theta_J = -pi/2 in the experiment. :contentReference[oaicite:3]{index=3}
    """
    n = backend.num_qubits
    layers = edge_colored_layers_from_backend(backend)

    qc = QuantumCircuit(n)
    for _ in range(num_trotter_steps):
        qc.rx(theta_h, range(n))
        for layer in layers:
            for (i, j) in layer:
                qc.rzz(theta_J, i, j)
        if add_barriers:
            qc.barrier()

    return qc

# --- Example usage  ---

# If you just want a topology-compatible offline object:
# from qiskit_ibm_runtime.fake_provider import FakeKyiv
# backend = FakeKyiv()
# qc = build_untwirled_kicked_ising_circuit(backend, num_trotter_steps=20, theta_h=0.7)
