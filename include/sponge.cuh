#pragma once

#include "grid.cuh"

// ============================================================
// GPU-WM: Sponge Layer Damping
//
// Rayleigh damping sponge for the upper boundary. Prevents
// spurious gravity wave reflection off the rigid lid.
//
// Upper sponge: Newtonian relaxation in the top 30% of the
// domain (above 0.7 * ztop). Damping coefficient ramps
// quadratically from zero at 0.7*ztop to 1/(20*dt) at ztop.
//
// Lateral damping is handled by open-boundary relaxation in
// boundaries.cu and is not duplicated here.
// ============================================================

namespace gpuwm {

/// Apply upper Rayleigh sponge damping.
/// @param state     Current model state (modified in-place)
/// @param state_init Initial/reference state for relaxation targets
/// @param grid      Grid configuration (nx, ny, nz, dx, dy, ztop)
/// @param dt        Current timestep (seconds)
void apply_sponge(StateGPU& state, StateGPU& state_init,
                  const GridConfig& grid, double dt);

} // namespace gpuwm
