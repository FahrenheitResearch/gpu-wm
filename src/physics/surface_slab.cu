// ============================================================
// GPU-WM: Minimal slab surface energy-balance model
//
// This is intentionally small. It adds a single 2D prognostic
// skin-temperature-like reservoir that can feed the existing
// surface-layer/PBL exchange path without dragging in a full
// land-surface model.
// ============================================================

#include "../../include/constants.cuh"
#include "../../include/grid.cuh"
#include "../../include/surface_layer.cuh"

namespace gpuwm {

namespace {

__device__ inline double clamp_skin_theta(double value) {
    return fmin(330.0, fmax(250.0, value));
}

__device__ inline double column_depth_from_terrain(double terrain, double ztop) {
    double col_depth = ztop - terrain;
    return (col_depth < 1.0) ? 1.0 : col_depth;
}

__device__ inline double terrain_relief_3x3(const real_t* __restrict__ terrain,
                                            int i, int j, int nx, int ny) {
    double zmin = 1.0e30;
    double zmax = -1.0e30;
    for (int jj = max(0, j - 1); jj <= min(ny - 1, j + 1); ++jj) {
        for (int ii = max(0, i - 1); ii <= min(nx - 1, i + 1); ++ii) {
            double z = (double)terrain[idx2(ii, jj, nx)];
            zmin = fmin(zmin, z);
            zmax = fmax(zmax, z);
        }
    }
    return fmax(0.0, zmax - zmin);
}

__device__ inline double terrain_slope_2d(const real_t* __restrict__ terrain,
                                          int i, int j, int nx, int ny,
                                          double dx, double dy) {
    int im = max(i - 1, 0);
    int ip = min(i + 1, nx - 1);
    int jm = max(j - 1, 0);
    int jp = min(j + 1, ny - 1);

    double dzdx = ((double)terrain[idx2(ip, j, nx)] - (double)terrain[idx2(im, j, nx)]) /
                  fmax((ip - im) * dx, 1.0);
    double dzdy = ((double)terrain[idx2(i, jp, nx)] - (double)terrain[idx2(i, jm, nx)]) /
                  fmax((jp - jm) * dy, 1.0);
    return sqrt(dzdx * dzdx + dzdy * dzdy);
}

__device__ inline double skin_theta_range_3x3(const real_t* __restrict__ tskin,
                                              int i, int j, int nx, int ny,
                                              double fallback_theta) {
    double tmin = fallback_theta;
    double tmax = fallback_theta;
    for (int jj = max(0, j - 1); jj <= min(ny - 1, j + 1); ++jj) {
        for (int ii = max(0, i - 1); ii <= min(nx - 1, i + 1); ++ii) {
            double th = (double)tskin[idx2(ii, jj, nx)];
            if (!(th > 0.0)) th = fallback_theta;
            tmin = fmin(tmin, th);
            tmax = fmax(tmax, th);
        }
    }
    return fmax(0.0, tmax - tmin);
}

__device__ inline double extrapolated_surface_theta(
    const real_t* __restrict__ theta,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int nx, int ny, int nz,
    int nx_h, int ny_h, double ztop,
    double theta_sfc_prior
) {
    int ijk1 = idx3(i, j, min(1, nz - 1), nx_h, ny_h);
    double th1 = (double)theta[ijk1];
    if (nz <= 2) {
        return clamp_skin_theta(0.75 * th1 + 0.25 * theta_sfc_prior);
    }

    int ijk2 = idx3(i, j, 2, nx_h, ny_h);
    double th2 = (double)theta[ijk2];
    double terrain_ij = (double)terrain[idx2(i, j, nx)];
    double col_depth = column_depth_from_terrain(terrain_ij, ztop);
    double z1 = eta_m[1] * col_depth;
    double z2 = eta_m[2] * col_depth;
    double dz12 = fmax(z2 - z1, 1.0);
    double theta_extrap = th1 - z1 * (th2 - th1) / dz12;
    theta_extrap = clamp_skin_theta(theta_extrap);
    return clamp_skin_theta(0.75 * theta_extrap + 0.25 * theta_sfc_prior);
}

} // namespace

__global__ void initialize_tskin_from_surface_layer_kernel(
    real_t* __restrict__ tskin,
    const real_t* __restrict__ theta,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int nx, int ny, int nz,
    double ztop,
    double theta_sfc_prior
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny || nz < 2) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    double theta_init = extrapolated_surface_theta(
        theta, terrain, eta_m, i, j, nx, ny, nz, nx_h, ny_h, ztop, theta_sfc_prior
    );
    tskin[idx2(i, j, nx)] = (real_t)theta_init;
}

__global__ void initialize_surface_moisture_memory_kernel(
    real_t* __restrict__ moistmem,
    const real_t* __restrict__ tskin,
    const real_t* __restrict__ terrain,
    int nx, int ny,
    double dx,
    double dy,
    double gate_strength
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    double theta_skin = (double)tskin[idx2(i, j, nx)];
    if (!(theta_skin > 0.0)) theta_skin = 300.0;
    double thermal_range = skin_theta_range_3x3(tskin, i, j, nx, ny, theta_skin);
    double terrain_relief = terrain_relief_3x3(terrain, i, j, nx, ny);
    double terrain_slope = terrain_slope_2d(terrain, i, j, nx, ny, dx, dy);
    double activation = admittance_seam_factor(
        thermal_range, terrain_relief, terrain_slope, 1.0
    );
    moistmem[idx2(i, j, nx)] = (real_t)moisture_availability_scale(
        activation, fmax(gate_strength, 0.0)
    );
}

__global__ void update_tskin_slab_kernel(
    real_t* __restrict__ tskin,
    real_t* __restrict__ moistmem,
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ theta,
    const real_t* __restrict__ qv,
    const real_t* __restrict__ rho,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ eta_w,
    int nx, int ny, int nz,
    double ztop,
    double dx,
    double dy,
    double z0,
    double theta_sfc_prior,
    double qv_sfc_prior,
    double skin_heat_capacity,
    double ground_restore_coeff,
    double anchor_weight,
    double admittance_seam_strength,
    double moisture_gate_strength,
    double moisture_memory_timescale,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny || nz < 2) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk1 = idx3(i, j, min(1, nz - 1), nx_h, ny_h);

    double u1 = (double)u[ijk1];
    double v1 = (double)v[ijk1];
    double th1 = (double)theta[ijk1];
    double qv1 = fmax((double)qv[ijk1], 0.0);
    double rho1 = fmax((double)rho[ijk1], 0.1);

    double theta_diag = extrapolated_surface_theta(
        theta, terrain, eta_m, i, j, nx, ny, nz, nx_h, ny_h, ztop, theta_sfc_prior
    );

    double qv_sfc_local = fmax(qv_sfc_prior, 0.0);
    if (nz > 2) {
        int ijk2 = idx3(i, j, 2, nx_h, ny_h);
        double qv2 = fmax((double)qv[ijk2], 0.0);
        double terrain_ij = (double)terrain[idx2(i, j, nx)];
        double col_depth = column_depth_from_terrain(terrain_ij, ztop);
        double z1 = eta_m[1] * col_depth;
        double z2 = eta_m[2] * col_depth;
        double dz12 = fmax(z2 - z1, 1.0);
        double qv_extrap = qv1 - z1 * (qv2 - qv1) / dz12;
        qv_extrap = fmin(0.03, fmax(0.0, qv_extrap));
        qv_sfc_local = 0.75 * qv_extrap + 0.25 * qv_sfc_local;
    }

    double terrain_ij = (double)terrain[idx2(i, j, nx)];
    double col_depth = column_depth_from_terrain(terrain_ij, ztop);
    double z1 = eta_m[min(1, nz - 1)] * col_depth;
    double log_z_z0 = log(fmax(z1, z0 + 1.0) / z0);
    if (log_z_z0 < 0.5) log_z_z0 = 0.5;
    double Cd = (KARMAN / log_z_z0) * (KARMAN / log_z_z0);

    double wspd = sqrt(u1 * u1 + v1 * v1);
    if (wspd < 0.1) wspd = 0.1;

    double theta_skin_old = (double)tskin[idx2(i, j, nx)];
    if (!(theta_skin_old > 0.0)) theta_skin_old = theta_diag;

    double seam_strength = fmax(admittance_seam_strength, 0.0);
    double thermal_range = skin_theta_range_3x3(tskin, i, j, nx, ny, theta_skin_old);
    double terrain_relief = terrain_relief_3x3(terrain, i, j, nx, ny);
    double terrain_slope = terrain_slope_2d(terrain, i, j, nx, ny, dx, dy);
    double moisture_activation = admittance_seam_factor(
        thermal_range, terrain_relief, terrain_slope, 1.0
    );
    double admittance_seam = admittance_seam_factor(
        thermal_range, terrain_relief, terrain_slope, seam_strength
    );
    double moisture_target = moisture_availability_scale(
        moisture_activation, fmax(moisture_gate_strength, 0.0)
    );
    double moisture_scale_old = (double)moistmem[idx2(i, j, nx)];
    if (!(moisture_scale_old > 0.0)) moisture_scale_old = moisture_target;
    double moisture_scale_new = update_moisture_memory_scale(
        moisture_scale_old, moisture_target, dt, moisture_memory_timescale
    );
    moistmem[idx2(i, j, nx)] = (real_t)moisture_scale_new;
    double qv_sfc_effective = apply_surface_moisture_scale(
        qv_sfc_local, qv1, moisture_scale_new
    );

    double sensible_flux = rho1 * CP_D * Cd * wspd * (theta_skin_old - th1);
    sensible_flux = fmax(-1000.0, fmin(sensible_flux, 1000.0));

    double latent_flux = rho1 * LV * Cd * wspd * (qv_sfc_effective - qv1);
    latent_flux = fmax(-800.0, fmin(latent_flux, 800.0));

    double effective_heat_capacity = fmax(
        skin_heat_capacity * fmax(0.45, 1.0 - 0.40 * admittance_seam),
        1.0e3
    );
    double effective_restore = fmax(
        ground_restore_coeff * fmax(0.55, 1.0 - 0.25 * admittance_seam),
        0.0
    );
    double effective_anchor = fmin(
        fmax(anchor_weight * fmax(0.35, 1.0 - 0.50 * admittance_seam), 0.0),
        1.0
    );

    double ground_restore = effective_restore * (theta_sfc_prior - theta_skin_old);
    double net_flux = ground_restore - sensible_flux - latent_flux;
    double theta_skin_new = theta_skin_old + dt * net_flux / effective_heat_capacity;

    // Keep the slab anchored to the local column enough to avoid decoupled drift.
    theta_skin_new = (1.0 - effective_anchor) * theta_skin_new + effective_anchor * theta_diag;
    tskin[idx2(i, j, nx)] = (real_t)clamp_skin_theta(theta_skin_new);
}

void initialize_tskin_from_surface_layer(StateGPU& state, const GridConfig& grid,
                                         double theta_sfc) {
    dim3 block2d(16, 16);
    dim3 grid2d((grid.nx + 15) / 16, (grid.ny + 15) / 16);
    initialize_tskin_from_surface_layer_kernel<<<grid2d, block2d>>>(
        state.tskin, state.theta, state.terrain, state.eta_m,
        grid.nx, grid.ny, grid.nz, grid.ztop, theta_sfc
    );
}

void initialize_surface_moisture_memory(StateGPU& state, const GridConfig& grid,
                                        double moisture_gate_strength) {
    dim3 block2d(16, 16);
    dim3 grid2d((grid.nx + 15) / 16, (grid.ny + 15) / 16);
    initialize_surface_moisture_memory_kernel<<<grid2d, block2d>>>(
        state.moistmem, state.tskin, state.terrain,
        grid.nx, grid.ny, grid.dx, grid.dy, moisture_gate_strength
    );
}

void update_tskin_slab(StateGPU& state, const GridConfig& grid,
                       double z0, double theta_sfc, double qv_sfc,
                       double skin_heat_capacity, double ground_restore_coeff,
                       double anchor_weight, double admittance_seam_strength,
                       double moisture_gate_strength,
                       double moisture_memory_timescale,
                       double dt) {
    dim3 block2d(16, 16);
    dim3 grid2d((grid.nx + 15) / 16, (grid.ny + 15) / 16);
    update_tskin_slab_kernel<<<grid2d, block2d>>>(
        state.tskin, state.moistmem, state.u, state.v, state.theta, state.qv, state.rho,
        state.terrain, state.eta_m, state.eta,
        grid.nx, grid.ny, grid.nz, grid.ztop, grid.dx, grid.dy,
        z0, theta_sfc, qv_sfc,
        skin_heat_capacity, ground_restore_coeff, anchor_weight,
        admittance_seam_strength, moisture_gate_strength, moisture_memory_timescale, dt
    );
}

} // namespace gpuwm
