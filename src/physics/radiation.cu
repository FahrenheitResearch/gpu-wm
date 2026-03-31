// ============================================================
// GPU-WM: Simplified Radiation Scheme
// Longwave and shortwave radiation parameterization
// Based on simplified gray-atmosphere approach
// ============================================================

#include "../../include/constants.cuh"
#include "../../include/grid.cuh"
#include "../../include/microphysics.cuh"

namespace gpuwm {

// ----------------------------------------------------------
// Longwave radiation (cooling profile)
// Simplified Newtonian cooling toward a radiative equilibrium
// ----------------------------------------------------------
__global__ void longwave_kernel(
    real_t* __restrict__ theta_tend,
    const real_t* __restrict__ theta,
    const double* __restrict__ p_base,
    const double* __restrict__ z_levels,
    const real_t* __restrict__ qv,
    const real_t* __restrict__ qc,
    const real_t* __restrict__ qr,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);

    double p = p_base[k];
    double th = (double)theta[ijk];
    double T = temperature_from_theta(th, p);
    double qv_val = fmax((double)qv[ijk], 0.0);
    double cloud_val = fmax((double)qc[ijk] + (double)qr[ijk], 0.0);
    double dz = (k == 0) ? z_levels[0] : (z_levels[k] - z_levels[k - 1]);
    dz = fmax(dz, 1.0);
    double air_density = p / (R_D * fmax(T, 180.0));

    // Radiative equilibrium temperature profile
    // Troposphere: ~200K at tropopause to ~300K at surface
    double p_trop = 20000.0;  // tropopause pressure (Pa)
    double th_eq;
    if (p > p_trop) {
        // Troposphere: linear in log-pressure
        th_eq = 300.0 * pow(p / P0, R_D * 0.0065 / G);
    } else {
        // Stratosphere: isothermal
        th_eq = 300.0 * pow(p_trop / P0, R_D * 0.0065 / G);
    }

    // Humidity and cloud feedbacks reduce longwave cooling in moist columns.
    double humidity_path = qv_val * dz * fmax(air_density, 0.1);
    double cloud_path = cloud_val * dz * fmax(air_density, 0.1);
    double greenhouse_boost = 3.0 * (1.0 - exp(-0.10 * humidity_path))
                            + 8.0 * (1.0 - exp(-18.0 * cloud_path));
    th_eq += greenhouse_boost;

    // Newtonian cooling rate (timescale ~20 days in free troposphere)
    double tau_rad = 20.0 * 86400.0 / (1.0 + 2.5 * (1.0 - exp(-0.10 * humidity_path))
                                                     + 6.0 * (1.0 - exp(-18.0 * cloud_path)));

    // Faster relaxation near surface (boundary layer, ~2 days)
    if (p > 85000.0) {
        tau_rad = 2.0 * 86400.0;
    }

    double dthdt = -(th - th_eq) / tau_rad;

    dthdt = fmax(fmin(dthdt, 3.0 / 86400.0), -3.0 / 86400.0);

    theta_tend[ijk] = (real_t)((double)theta_tend[ijk] + dthdt);
}

// ----------------------------------------------------------
// Shortwave radiation (solar heating)
// Simplified: exponential absorption with height
// ----------------------------------------------------------
__global__ void shortwave_kernel(
    real_t* __restrict__ theta_tend,
    const real_t* __restrict__ theta,
    const double* __restrict__ p_base,
    const double* __restrict__ rho_base,
    const double* __restrict__ z_levels,
    const real_t* __restrict__ qv,
    const real_t* __restrict__ qc,
    const real_t* __restrict__ qr,
    int nx, int ny, int nz,
    double solar_zenith_cos,  // cosine of solar zenith angle
    double solar_constant     // W/m^2 at TOA
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;
    if (solar_zenith_cos <= 0.0) return;  // Nighttime

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);

    double rho = rho_base[k];
    double dz = (k == 0) ? z_levels[0] : (z_levels[k] - z_levels[k - 1]);
    dz = fmax(dz, 1.0);

    // Column optical depth from the top of atmosphere to this layer.
    // Cloud condensate attenuates much more strongly than vapor.
    double tau_above = 0.0;
    for (int kk = nz - 1; kk >= k; --kk) {
        int jkk = idx3(i, j, kk, nx_h, ny_h);
        double dzk = (kk == 0) ? z_levels[0] : (z_levels[kk] - z_levels[kk - 1]);
        dzk = fmax(dzk, 1.0);
        double air_mass = rho_base[kk] * dzk;
        double vapor = fmax((double)qv[jkk], 0.0);
        double cloud = fmax((double)qc[jkk] + (double)qr[jkk], 0.0);
        tau_above += 0.08 * air_mass * vapor + 14.0 * air_mass * cloud;
    }

    double pressure_factor = sqrt(fmax(p_base[k] / P0, 0.05));
    double layer_cloud = fmax((double)qc[ijk] + (double)qr[ijk], 0.0);
    double layer_vapor = fmax((double)qv[ijk], 0.0);
    double layer_tau = pressure_factor * (
        0.05 * rho * dz * layer_vapor +
        12.0 * rho * dz * layer_cloud
    );

    double incoming = solar_constant * solar_zenith_cos;
    double trans_above = exp(-tau_above / fmax(solar_zenith_cos, 0.15));
    double absorbed = incoming * 0.78 * trans_above * (1.0 - exp(-layer_tau));

    double dthdt = absorbed / (CP_D * fmax(rho * dz, 1.0));

    theta_tend[ijk] = (real_t)((double)theta_tend[ijk] + dthdt);
}

// ----------------------------------------------------------
// Host driver
// ----------------------------------------------------------
void run_radiation(StateGPU& state, const GridConfig& grid,
                   double solar_zenith_cos, double solar_constant) {
    int nx = grid.nx, ny = grid.ny, nz = grid.nz;

    dim3 block(8, 8, 4);
    dim3 grid3d((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);

    longwave_kernel<<<grid3d, block>>>(
        state.theta_tend, state.theta, state.p_base, state.z_levels,
        state.qv, state.qc, state.qr,
        nx, ny, nz
    );

    if (solar_zenith_cos > 0.0) {
        shortwave_kernel<<<grid3d, block>>>(
            state.theta_tend, state.theta, state.p_base, state.rho_base,
            state.z_levels, state.qv, state.qc, state.qr,
            nx, ny, nz,
            solar_zenith_cos, solar_constant
        );
    }
}

} // namespace gpuwm
