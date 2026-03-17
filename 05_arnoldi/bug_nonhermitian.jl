using LinearAlgebra

"""
    bug_second_order_nonhermitian!(state, mpo, config; numiter_lanczos=25, truncation_granularity=:after_site)

Second-order BUG integrator specialized for non-Hermitian linear ODE benchmarks.

It composes two first-order BUG half-steps with opposite sweep directions (Strang-like),
but intentionally avoids the `BUGModule.bug_second_order!` path that may introduce
implicit normalization through end-of-window truncation behavior.

This routine uses `BUGModule.bug!` with:
- half-step `dt/2` on each directional sweep,
- per-site truncation (`truncation_granularity=:after_site`) by default,
- no additional full-sweep/post-window truncation pass.
"""
function bug_second_order_nonhermitian!(state::Yaqs.MPSModule.MPS{ComplexF64},
                                        mpo::Yaqs.MPOModule.MPO{ComplexF64},
                                        config::Yaqs.SimulationConfigs.AbstractSimConfig;
                                        numiter_lanczos::Int=25,
                                        truncation_granularity::Symbol=:after_site)
    @assert truncation_granularity === :after_site || truncation_granularity === :after_sweep
    L = state.length
    @assert mpo.length == L

    # Preserve original dt and use symmetric half-steps.
    dt_orig = config isa Yaqs.SimulationConfigs.TimeEvolutionConfig ? config.dt : 1.0
    dt_half = config isa Yaqs.SimulationConfigs.TimeEvolutionConfig ? 0.5 * dt_orig : 1.0
    if config isa Yaqs.SimulationConfigs.TimeEvolutionConfig
        config.dt = dt_half
    end

    # First half-step (right-to-left sweep in BUG implementation).
    Yaqs.BUGModule.bug!(state, mpo, config;
                        numiter_lanczos=numiter_lanczos,
                        do_truncate=true,
                        truncation_granularity=truncation_granularity)

    # Second half-step with opposite direction via explicit flip (same strategy as core BUG2).
    _flip_mps_nonherm!(state)
    _flip_mpo_nonherm!(mpo)
    Yaqs.MPSModule.shift_orthogonality_center!(state, state.length)
    Yaqs.BUGModule.bug!(state, mpo, config;
                        numiter_lanczos=numiter_lanczos,
                        do_truncate=true,
                        truncation_granularity=truncation_granularity)
    _flip_mps_nonherm!(state)
    _flip_mpo_nonherm!(mpo)

    if config isa Yaqs.SimulationConfigs.TimeEvolutionConfig
        config.dt = dt_orig
    end
    return nothing
end

@inline function _flip_mps_nonherm!(state::Yaqs.MPSModule.MPS{ComplexF64})
    L = state.length
    reverse!(state.tensors)
    for i in 1:L
        state.tensors[i] = permutedims(state.tensors[i], (3, 2, 1)) # (Dl,d,Dr) -> (Dr,d,Dl)
    end
    state.orth_center = (state.orth_center == 0) ? 0 : (L + 1 - state.orth_center)
    return nothing
end

@inline function _flip_mpo_nonherm!(mpo::Yaqs.MPOModule.MPO{ComplexF64})
    L = mpo.length
    reverse!(mpo.tensors)
    for i in 1:L
        mpo.tensors[i] = permutedims(mpo.tensors[i], (4, 2, 3, 1)) # (Dl,po,pi,Dr) -> (Dr,po,pi,Dl)
    end
    mpo.orth_center = (mpo.orth_center == 0) ? 0 : (L + 1 - mpo.orth_center)
    return nothing
end
