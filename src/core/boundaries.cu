// ============================================================
// GPU-WM: Open Boundary Conditions
// Relaxation (Davies 1976) + radiative (Orlanski 1976)
//
// Real weather models can't use periodic BCs - they need to
// blend toward a parent model solution at the domain edges.
// ============================================================

#include "../../include/constants.cuh"
#include "../../include/grid.cuh"

namespace gpuwm {

__device__ inline double sample_terrain_clamped(
    const real_t* __restrict__ terrain,
    int i, int j, int nx, int ny,
    double ztop
) {
    int ii = max(0, min(i, nx - 1));
    int jj = max(0, min(j, ny - 1));
    double terrain_val = (double)terrain[idx2(ii, jj, nx)];
    return fmin(terrain_val, ztop - 1.0);
}

__device__ inline double local_boundary_metric_slope_x(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int nx, int ny,
    double dx_eff, double ztop
) {
    int i_m = max(i - 1, 0);
    int i_p = min(i + 1, nx - 1);
    double h_m = sample_terrain_clamped(terrain, i_m, j, nx, ny, ztop);
    double h_p = sample_terrain_clamped(terrain, i_p, j, nx, ny, ztop);
    double ds = max((i_p - i_m) * dx_eff, 1.0);
    return (1.0 - eta_m[0]) * (h_p - h_m) / ds;
}

__device__ inline double local_boundary_metric_slope_y(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int nx, int ny,
    double dy_eff, double ztop
) {
    int j_m = max(j - 1, 0);
    int j_p = min(j + 1, ny - 1);
    double h_m = sample_terrain_clamped(terrain, i, j_m, nx, ny, ztop);
    double h_p = sample_terrain_clamped(terrain, i, j_p, nx, ny, ztop);
    double ds = max((j_p - j_m) * dy_eff, 1.0);
    return (1.0 - eta_m[0]) * (h_p - h_m) / ds;
}

// ----------------------------------------------------------
// Relaxation weight function
// w = 1 at boundary, 0 at interior edge of relax zone
// Uses cosine taper for smooth blending
// ----------------------------------------------------------
__device__ double relax_weight(int dist, int width) {
    if (dist >= width) return 0.0;
    double x = (double)(width - dist) / width;
    return 0.5 * (1.0 + cos(PI * (1.0 - x)));
}

// ----------------------------------------------------------
// Apply relaxation toward lateral boundary values
// field = (1-w)*field + w*field_bdy
// For now, boundary values = initial values (simple approach)
// In production, these come from a parent model (GFS/RAP)
// ----------------------------------------------------------
__global__ void relax_boundary_kernel(
    real_t* __restrict__ field,
    const real_t* __restrict__ field_init,  // Initial/boundary values
    int nx, int ny, int nz,
    int relax_width
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);

    // Distance from each boundary
    int dist_w = i;
    int dist_e = nx - 1 - i;
    int dist_s = j;
    int dist_n = ny - 1 - j;
    int dist = min(min(dist_w, dist_e), min(dist_s, dist_n));

    double w = relax_weight(dist, relax_width);

    if (w > 0.0) {
        field[ijk] = (real_t)((1.0 - w) * (double)field[ijk] + w * (double)field_init[ijk]);
    }
}

// ----------------------------------------------------------
// Open boundary extrapolation (x-direction)
//
// Uses linear extrapolation from the two nearest interior
// points.  This is equivalent to the gravity-wave radiation
// condition of Durran & Klemp (1983) in the limit of constant
// phase speed equal to one grid point per time step.  Compared
// with the previous zero-gradient copy (field[0]=field[1]),
// linear extrapolation lets outgoing disturbances propagate
// through the boundary rather than reflect back.
//
// The extrapolated value is clamped so it cannot overshoot by
// more than one gradient step -- this prevents ringing when the
// interior field has sharp gradients near the boundary.
// ----------------------------------------------------------
__global__ void open_bc_x_kernel(
    real_t* __restrict__ field,
    int nx, int ny, int nz
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= ny || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;

    // West boundary: linear extrapolation from points 1,2
    {
        double f1 = (double)field[idx3(1, j, k, nx_h, ny_h)];
        double f2 = (double)field[idx3(2, j, k, nx_h, ny_h)];
        double extrap = 2.0 * f1 - f2;  // linear extrapolation
        // Blend: 50% extrapolation + 50% zero-gradient for stability
        double val = 0.5 * extrap + 0.5 * f1;
        field[idx3(0, j, k, nx_h, ny_h)] = (real_t)val;
    }

    // Halo cells: zero-gradient from boundary value
    field[idx3(-1, j, k, nx_h, ny_h)] = field[idx3(0, j, k, nx_h, ny_h)];
    field[idx3(-2, j, k, nx_h, ny_h)] = field[idx3(0, j, k, nx_h, ny_h)];

    // East boundary: linear extrapolation from points nx-2, nx-3
    {
        double f1 = (double)field[idx3(nx-2, j, k, nx_h, ny_h)];
        double f2 = (double)field[idx3(nx-3, j, k, nx_h, ny_h)];
        double extrap = 2.0 * f1 - f2;
        double val = 0.5 * extrap + 0.5 * f1;
        field[idx3(nx-1, j, k, nx_h, ny_h)] = (real_t)val;
    }

    // Halo
    field[idx3(nx, j, k, nx_h, ny_h)] = field[idx3(nx-1, j, k, nx_h, ny_h)];
    field[idx3(nx+1, j, k, nx_h, ny_h)] = field[idx3(nx-1, j, k, nx_h, ny_h)];
}

// ----------------------------------------------------------
// Open boundary extrapolation (y-direction)
// Same approach as open_bc_x_kernel: blended linear
// extrapolation from interior.
// ----------------------------------------------------------
__global__ void open_bc_y_kernel(
    real_t* __restrict__ field,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;

    // South boundary: linear extrapolation from points 1,2
    {
        double f1 = (double)field[idx3(i, 1, k, nx_h, ny_h)];
        double f2 = (double)field[idx3(i, 2, k, nx_h, ny_h)];
        double extrap = 2.0 * f1 - f2;
        double val = 0.5 * extrap + 0.5 * f1;
        field[idx3(i, 0, k, nx_h, ny_h)] = (real_t)val;
    }
    field[idx3(i, -1, k, nx_h, ny_h)] = field[idx3(i, 0, k, nx_h, ny_h)];
    field[idx3(i, -2, k, nx_h, ny_h)] = field[idx3(i, 0, k, nx_h, ny_h)];

    // North boundary: linear extrapolation from points ny-2, ny-3
    {
        double f1 = (double)field[idx3(i, ny-2, k, nx_h, ny_h)];
        double f2 = (double)field[idx3(i, ny-3, k, nx_h, ny_h)];
        double extrap = 2.0 * f1 - f2;
        double val = 0.5 * extrap + 0.5 * f1;
        field[idx3(i, ny-1, k, nx_h, ny_h)] = (real_t)val;
    }
    field[idx3(i, ny, k, nx_h, ny_h)] = field[idx3(i, ny-1, k, nx_h, ny_h)];
    field[idx3(i, ny+1, k, nx_h, ny_h)] = field[idx3(i, ny-1, k, nx_h, ny_h)];
}

__global__ void fill_halo_x_kernel(
    real_t* __restrict__ field,
    int nx, int ny, int nz
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= ny || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;

    field[idx3(-1, j, k, nx_h, ny_h)] = field[idx3(0, j, k, nx_h, ny_h)];
    field[idx3(-2, j, k, nx_h, ny_h)] = field[idx3(0, j, k, nx_h, ny_h)];
    field[idx3(nx, j, k, nx_h, ny_h)] = field[idx3(nx-1, j, k, nx_h, ny_h)];
    field[idx3(nx+1, j, k, nx_h, ny_h)] = field[idx3(nx-1, j, k, nx_h, ny_h)];
}

__global__ void fill_halo_y_kernel(
    real_t* __restrict__ field,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;

    field[idx3(i, -1, k, nx_h, ny_h)] = field[idx3(i, 0, k, nx_h, ny_h)];
    field[idx3(i, -2, k, nx_h, ny_h)] = field[idx3(i, 0, k, nx_h, ny_h)];
    field[idx3(i, ny, k, nx_h, ny_h)] = field[idx3(i, ny-1, k, nx_h, ny_h)];
    field[idx3(i, ny+1, k, nx_h, ny_h)] = field[idx3(i, ny-1, k, nx_h, ny_h)];
}

// Vertical BC for contravariant w.  See bc_w_kernel in dynamics.cu
// for the full rationale.  deta/dt = 0 at both eta=0 and eta=1.
__global__ void zero_w_vertical_bc_kernel(
    real_t* __restrict__ w,
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ mapfac_m,
    int nx, int ny, int nz,
    double dx, double dy, double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    // Unused -- contravariant w BC is zero regardless of terrain geometry.
    (void)u; (void)v; (void)terrain; (void)eta_m; (void)mapfac_m;
    (void)dx; (void)dy; (void)ztop;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    w[idx3w(i, j, 0,  nx_h, ny_h)] = (real_t)0.0;  // surface
    w[idx3w(i, j, nz, nx_h, ny_h)] = (real_t)0.0;  // model top
}

// ----------------------------------------------------------
// Host driver for open boundary conditions
// ----------------------------------------------------------
void apply_open_boundaries(StateGPU& state, StateGPU& state_init,
                           const GridConfig& grid, int relax_width) {
    int nx = grid.nx, ny = grid.ny, nz = grid.nz;

    dim3 block(8, 8, 4);
    dim3 grid3d((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);

    dim3 block_jk(16, 16);
    dim3 grid_jk((ny + 15) / 16, (nz + 15) / 16);
    dim3 block_ik(16, 16);
    dim3 grid_ik((nx + 15) / 16, (nz + 15) / 16);
    dim3 grid_jk_w((ny + 15) / 16, ((nz + 1) + 15) / 16);
    dim3 grid_ik_w((nx + 15) / 16, ((nz + 1) + 15) / 16);

    // Relax only the materially advected mass-level fields. Pressure
    // perturbation and interface w are allowed to adjust more freely at the
    // open boundary; they still get extrapolated boundary values and halo
    // refresh, but they are not nudged toward the boundary snapshot here.
    real_t* relax_fields[] = {state.u, state.v, state.theta,
                              state.qv, state.qc, state.qr};
    real_t* relax_fields_init[] = {state_init.u, state_init.v, state_init.theta,
                                   state_init.qv, state_init.qc, state_init.qr};

    for (int f = 0; f < 6; f++) {
        open_bc_x_kernel<<<grid_jk, block_jk>>>(relax_fields[f], nx, ny, nz);
        open_bc_y_kernel<<<grid_ik, block_ik>>>(relax_fields[f], nx, ny, nz);
        relax_boundary_kernel<<<grid3d, block>>>(
            relax_fields[f], relax_fields_init[f], nx, ny, nz, relax_width
        );
        fill_halo_x_kernel<<<grid_jk, block_jk>>>(relax_fields[f], nx, ny, nz);
        fill_halo_y_kernel<<<grid_ik, block_ik>>>(relax_fields[f], nx, ny, nz);
    }
    open_bc_x_kernel<<<grid_jk, block_jk>>>(state.p, nx, ny, nz);
    open_bc_y_kernel<<<grid_ik, block_ik>>>(state.p, nx, ny, nz);
    fill_halo_x_kernel<<<grid_jk, block_jk>>>(state.p, nx, ny, nz);
    fill_halo_y_kernel<<<grid_ik, block_ik>>>(state.p, nx, ny, nz);
    dim3 grid3d_w((nx + 7) / 8, (ny + 7) / 8, ((nz + 1) + 3) / 4);
    open_bc_x_kernel<<<grid_jk_w, block_jk>>>(state.w, nx, ny, nz + 1);
    open_bc_y_kernel<<<grid_ik_w, block_ik>>>(state.w, nx, ny, nz + 1);
    fill_halo_x_kernel<<<grid_jk_w, block_jk>>>(state.w, nx, ny, nz + 1);
    fill_halo_y_kernel<<<grid_ik_w, block_ik>>>(state.w, nx, ny, nz + 1);

    dim3 block_ij(16, 16);
    dim3 grid_ij((nx + 15) / 16, (ny + 15) / 16);
    zero_w_vertical_bc_kernel<<<grid_ij, block_ij>>>(
        state.w, state.u, state.v, state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop
    );
}

// ----------------------------------------------------------
// Lightweight halo refresh for open-BC mode.
//
// Extrapolates boundary cells (zero-gradient) and fills halo
// zones for all prognostic fields.  Does NOT apply relaxation.
// Use this after physics modifications to ensure halos are
// consistent before the next timestep, without adding an extra
// round of lateral damping.
// ----------------------------------------------------------
void refresh_open_halos(StateGPU& state, const GridConfig& grid) {
    int nx = grid.nx, ny = grid.ny, nz = grid.nz;

    dim3 block_jk(16, 16);
    dim3 grid_jk((ny + 15) / 16, (nz + 15) / 16);
    dim3 block_ik(16, 16);
    dim3 grid_ik((nx + 15) / 16, (nz + 15) / 16);
    dim3 grid_jk_w((ny + 15) / 16, ((nz + 1) + 15) / 16);
    dim3 grid_ik_w((nx + 15) / 16, ((nz + 1) + 15) / 16);

    real_t* mass_fields[] = {state.u, state.v, state.theta,
                             state.qv, state.qc, state.qr, state.p};

    for (int f = 0; f < 7; f++) {
        open_bc_x_kernel<<<grid_jk, block_jk>>>(mass_fields[f], nx, ny, nz);
        open_bc_y_kernel<<<grid_ik, block_ik>>>(mass_fields[f], nx, ny, nz);
        fill_halo_x_kernel<<<grid_jk, block_jk>>>(mass_fields[f], nx, ny, nz);
        fill_halo_y_kernel<<<grid_ik, block_ik>>>(mass_fields[f], nx, ny, nz);
    }
    open_bc_x_kernel<<<grid_jk_w, block_jk>>>(state.w, nx, ny, nz + 1);
    open_bc_y_kernel<<<grid_ik_w, block_ik>>>(state.w, nx, ny, nz + 1);
    fill_halo_x_kernel<<<grid_jk_w, block_jk>>>(state.w, nx, ny, nz + 1);
    fill_halo_y_kernel<<<grid_ik_w, block_ik>>>(state.w, nx, ny, nz + 1);

    dim3 block_ij(16, 16);
    dim3 grid_ij((nx + 15) / 16, (ny + 15) / 16);
    zero_w_vertical_bc_kernel<<<grid_ij, block_ij>>>(
        state.w, state.u, state.v, state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop
    );
}

} // namespace gpuwm
