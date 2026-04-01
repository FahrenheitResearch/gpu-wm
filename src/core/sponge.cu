// ============================================================
// GPU-WM: Sponge Layer Damping
//
// Upper Rayleigh sponge -- damps fields toward the base state
// (theta, w) or initial values (u, v) in the top 30% of the
// domain, preventing gravity wave reflection off the rigid-lid
// model top.
//
// Uses Newtonian relaxation:
//   field = field - alpha * dt * (field - field_target)
//
// Lateral damping is handled separately by the boundary
// relaxation in boundaries.cu (apply_open_boundaries).
// ============================================================

#include "../../include/constants.cuh"
#include "../../include/grid.cuh"

namespace gpuwm {

__device__ inline double sponge_column_terrain(
    const real_t* __restrict__ terrain,
    int i, int j, int nx,
    double ztop
) {
    double terrain_val = (double)terrain[idx2(i, j, nx)];
    return fmin(terrain_val, ztop - 1.0);
}

__device__ inline double sponge_theta_reference(
    const double* __restrict__ theta_base,
    const double* __restrict__ z_levels,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k,
    int nx, int ny, int nz,
    double ztop
) {
    (void)ny;
    if (nz <= 1) return theta_base[0];

    double terrain_val = sponge_column_terrain(terrain, i, j, nx, ztop);
    double z_local = terrain_following_height(terrain_val, eta_m[k], ztop);
    if (z_local <= z_levels[0]) {
        double dz = fmax(z_levels[1] - z_levels[0], 1.0);
        return theta_base[0] + (z_local - z_levels[0]) * (theta_base[1] - theta_base[0]) / dz;
    }
    if (z_local >= z_levels[nz - 1]) {
        double dz = fmax(z_levels[nz - 1] - z_levels[nz - 2], 1.0);
        return theta_base[nz - 1] + (z_local - z_levels[nz - 1]) *
            (theta_base[nz - 1] - theta_base[nz - 2]) / dz;
    }

    int lo = 0;
    int hi = nz - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (z_levels[mid] <= z_local) lo = mid;
        else hi = mid;
    }

    double frac = (z_local - z_levels[lo]) / fmax(z_levels[hi] - z_levels[lo], 1.0);
    return theta_base[lo] + frac * (theta_base[hi] - theta_base[lo]);
}

// ----------------------------------------------------------
// Upper Rayleigh sponge kernel
//
// Active in the top 30% of the domain (z > 0.7 * ztop).
// Damping coefficient ramps quadratically:
//   alpha = (1 / (20*dt)) * ((z - z_damp) / (ztop - z_damp))^2
//
// Fields are relaxed toward:
//   theta   -> theta_base[k]
//   u, v    -> initial values (state_init)
//
// p' is NOT damped here.  Relaxing p' toward zero in the sponge
// layer over-constrains the mass field and creates an artificial
// pressure sink that generates spurious vertical circulations.
// Standard practice (WRF's w_damping + Rayleigh sponge, MPAS-LAM)
// is to damp w and theta/u/v but leave p' free to adjust via the
// continuity equation.
// ----------------------------------------------------------
__global__ void rayleigh_sponge_kernel(
    real_t* __restrict__ u,
    real_t* __restrict__ v,
    real_t* __restrict__ theta,
    const real_t* __restrict__ u_init,
    const real_t* __restrict__ v_init,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ theta_base,
    const double* __restrict__ z_levels,
    int nx, int ny, int nz,
    double ztop, double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    double terrain_val = sponge_column_terrain(terrain, i, j, nx, ztop);
    double z = terrain_following_height(terrain_val, eta_m[k], ztop);
    double z_damp = 0.7 * ztop;

    // No damping below the sponge onset
    if (z <= z_damp) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);

    // Quadratic ramp: 0 at z_damp, 1/(20*dt) at ztop
    double frac = (z - z_damp) / (ztop - z_damp);  // 0 -> 1
    double alpha = (1.0 / (20.0 * dt)) * frac * frac;
    double adt = alpha * dt;  // dimensionless damping factor per step
    double theta_ref = sponge_theta_reference(
        theta_base, z_levels, terrain, eta_m, i, j, k, nx, ny, nz, ztop
    );

    // Newtonian relaxation: field = field - alpha*dt*(field - target)
    // theta -> theta_base[k]
    theta[ijk]  = (real_t)((double)theta[ijk] - adt * ((double)theta[ijk] - theta_ref));

    // u -> initial u
    u[ijk]      = (real_t)((double)u[ijk] - adt * ((double)u[ijk] - (double)u_init[ijk]));

    // v -> initial v
    v[ijk]      = (real_t)((double)v[ijk] - adt * ((double)v[ijk] - (double)v_init[ijk]));
}

__global__ void rayleigh_sponge_w_kernel(
    real_t* __restrict__ w,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_w,
    int nx, int ny, int nz,
    double ztop, double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k <= 0 || k >= nz) return;

    double terrain_val = sponge_column_terrain(terrain, i, j, nx, ztop);
    double z = terrain_following_height(terrain_val, eta_w[k], ztop);
    double z_damp = 0.7 * ztop;
    if (z <= z_damp) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk_w = idx3w(i, j, k, nx_h, ny_h);

    double frac = (z - z_damp) / (ztop - z_damp);
    double alpha = (1.0 / (20.0 * dt)) * frac * frac;
    double adt = alpha * dt;
    w[ijk_w] = (real_t)((double)w[ijk_w] * (1.0 - adt));
}

// ----------------------------------------------------------
// Lateral sponge kernel
//
// Retained for mass fields in open-boundary runs, but not for
// interface w or p'. Those are handled by the dedicated open
// boundary relaxation and the continuity/acoustic system.
// ----------------------------------------------------------
static constexpr int SPONGE_WIDTH = 15;

__global__ void lateral_sponge_kernel(
    real_t* __restrict__ field,
    const real_t* __restrict__ field_init,
    int nx, int ny, int nz,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int dist_w = i;
    int dist_e = nx - 1 - i;
    int dist_s = j;
    int dist_n = ny - 1 - j;
    int dist = min(min(dist_w, dist_e), min(dist_s, dist_n));
    if (dist >= SPONGE_WIDTH) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);
    double weight = 0.5 * (1.0 + cos(PI * (double)dist / (double)SPONGE_WIDTH));
    double alpha_lat = 1.0 / (10.0 * dt);
    double adt = weight * alpha_lat * dt;
    field[ijk] = (real_t)((double)field[ijk] - adt * ((double)field[ijk] - (double)field_init[ijk]));
}

// ----------------------------------------------------------
// Host driver: apply sponge layers
//
// The upper Rayleigh sponge is always applied here.  A lighter
// lateral sponge is retained for mass fields only; interface w
// and p' are left to the dedicated boundary relaxation plus the
// pressure/continuity system.
// ----------------------------------------------------------
void apply_sponge(StateGPU& state, StateGPU& state_init,
                  const GridConfig& grid, double dt) {
    int nx = grid.nx;
    int ny = grid.ny;
    int nz = grid.nz;

    dim3 block(8, 8, 4);
    dim3 grid3d((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
    dim3 grid3d_w((nx + 7) / 8, (ny + 7) / 8, ((nz + 1) + 3) / 4);

    // --- Upper Rayleigh sponge ---
    rayleigh_sponge_kernel<<<grid3d, block>>>(
        state.u, state.v, state.theta,
        state_init.u, state_init.v,
        state.terrain, state.eta_m, state.theta_base, state.z_levels,
        nx, ny, nz, grid.ztop, dt
    );
    rayleigh_sponge_w_kernel<<<grid3d_w, block>>>(
        state.w, state.terrain, state.eta,
        nx, ny, nz, grid.ztop, dt
    );

    real_t* fields[] = {
        state.u, state.v, state.theta,
        state.qv, state.qc, state.qr
    };
    real_t* fields_init[] = {
        state_init.u, state_init.v, state_init.theta,
        state_init.qv, state_init.qc, state_init.qr
    };

    for (int f = 0; f < 6; ++f) {
        lateral_sponge_kernel<<<grid3d, block>>>(
            fields[f], fields_init[f],
            nx, ny, nz, dt
        );
    }
}

} // namespace gpuwm
