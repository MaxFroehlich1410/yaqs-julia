using Yaqs
using Yaqs.MPSModule
using Yaqs.GateLibrary
using Yaqs.SimulationConfigs
using Yaqs.CircuitLibrary
using Yaqs.Simulator

L = 6
psi = MPSModule.MPS(L, state="zeros")

obs = SimulationConfigs.Observable("Z1", GateLibrary.ZGate(), 1)
cfg = SimulationConfigs.TimeEvolutionConfig([obs], 1.0; dt=0.1, num_traj=10)

circ = CircuitLibrary.create_ising_circuit(L, 1.0, 1.0, 0.1, 10)
Simulator.run(psi, circ, cfg)

println(obs.results)
