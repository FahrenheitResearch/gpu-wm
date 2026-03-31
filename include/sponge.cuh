#pragma once

#include "grid.cuh"

// ============================================================
// GPU-WM: Sponge Layer Damping
//
// Rayleigh damping sponge for the upper boundary and lateral
// boundaries. Prevents spurious gravity wave reflection off the
// rigid lid and absorbs outgoing waves near domain edges when
// periodic boundary conditions are used.
//
// Upper sponge: Newtonian relaxation in the top 30% of the
// domain (above 0.7 * ztop). Damping coefficient ramps
// quadratically from zero at 0.7*ztop to 1/(20*dt) at ztop.
//
// Lateral sponge: Cosine-tapered relaxation in the outermost
// 15 grid cells on each side toward the initial state.
// ============================================================

namespace gpuwm {

/// Apply both upper Rayleigh and lateral sponge damping.
/// @param state     Current model state (modified in-place)
/// @param state_init Initial/reference state for relaxation targets
/// @param grid      Grid configuration (nx, ny, nz, dx, dy, ztop)
/// @param dt        Current timestep (seconds)
void apply_sponge(StateGPU& state, StateGPU& state_init,
                  const GridConfig& grid, double dt);

} // namespace gpuwm
