// ============================================================
// GPU-WM: Kessler Warm Rain Microphysics
// Single-moment bulk microphysics scheme
// Handles: saturation adjustment, autoconversion,
//          accretion, rain evaporation, sedimentation
// ============================================================

#include "../../include/constants.cuh"
#include "../../include/grid.cuh"
#include "../../include/microphysics.cuh"

namespace gpuwm {

// ----------------------------------------------------------
// Kessler microphysics kernel
// Terrain-following: computes local pressure and density
// from the hydrostatic relation at the terrain-following
// coordinate height, rather than using flat 1D base-state
// arrays.
// ----------------------------------------------------------
__global__ void kessler_kernel(
    real_t* __restrict__ theta,
    real_t* __restrict__ qv,
    real_t* __restrict__ qc,
    real_t* __restrict__ qr,
    const double* __restrict__ rho_base,
    const double* __restrict__ p_base,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int nx, int ny, int nz,
    double ztop,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);

    double th = (double)theta[ijk];
    double qv_val = (double)qv[ijk];
    double qc_val = (double)qc[ijk];
    double qr_val = (double)qr[ijk];

    // --- Terrain-following pressure and density ---
    // The 1D base-state arrays assume flat terrain (z_phys = z_levels[k]).
    // Over elevated terrain the physical height of level k is higher,
    // so pressure and density are lower.  Correct using the hypsometric
    // equation applied to the height difference.
    double terrain_h = (double)terrain[idx2(i, j, nx)];
    double z_phys = terrain_following_height(terrain_h, eta_m[k], ztop);
    // Height the 1D base state was defined at (flat terrain => terrain=0)
    double z_flat = terrain_following_height(0.0, eta_m[k], ztop);
    double dz_shift = z_phys - z_flat;   // positive over mountains

    double p_flat = p_base[k];
    double rho_flat = rho_base[k];
    // Scale-height correction:  p(z) = p0 * exp(-dz / H)
    // where H = R_d * T / g.  Use base-state T for the scale height.
    double T_base_est = p_flat / (rho_flat * R_D);
    double H_scale = R_D * T_base_est / G;
    double correction = exp(-dz_shift / H_scale);

    double p_val   = p_flat * correction;
    double rho_val = rho_flat * correction;

    // Skip microphysics where theta is unphysical
    if (th > 500.0 || th < 200.0 || isnan(th)) return;

    // Guard against negative moisture on entry
    qv_val = fmax(qv_val, 0.0);
    qc_val = fmax(qc_val, 0.0);
    qr_val = fmax(qr_val, 0.0);

    // Save original theta for clamping later
    double theta_old = th;

    // Temperature from potential temperature
    double exner = exner_from_pressure(p_val);
    double T = th * exner;

    // Guard - skip unphysical values
    if (T < 150.0 || T > 350.0) return;

    // --- Instantaneous saturation adjustment (Newton iteration) ---
    // Standard NWP practice: iterate T and qv to exact equilibrium.
    // The Clausius-Clapeyron denominator accounts for the latent
    // heating feedback on saturation, so 3 iterations suffice.
    for (int iter = 0; iter < 3; ++iter) {
        double qvs = saturation_mixing_ratio_liquid(p_val, T);
        double excess = qv_val - qvs;
        // Denominator from implicit linearisation of Clausius-Clapeyron:
        //   1 + (L_v^2 * qvs) / (c_p * R_v * T^2)
        double denominator = 1.0 + LV * LV * qvs / (CP_D * R_V * T * T);

        if (excess > 0.0) {
            // Supersaturated: condense
            double cond = excess / denominator;
            cond = fmin(cond, qv_val);        // cannot condense more than available vapor
            cond = fmax(cond, 0.0);
            qc_val += cond;
            qv_val -= cond;
            T += LV * cond / CP_D;
        } else if (qc_val > 0.0 && excess < 0.0) {
            // Sub-saturated with cloud water: evaporate
            double evap = -excess / denominator;
            evap = fmin(evap, qc_val);        // cannot evaporate more than cloud water
            evap = fmax(evap, 0.0);
            qc_val -= evap;
            qv_val += evap;
            T -= LV * evap / CP_D;
        } else {
            break;  // converged (no excess and no cloud water to evaporate)
        }
    }

    // --- Autoconversion (cloud -> rain) ---
    double qc_threshold = 8.0e-4;  // slightly easier warm-rain trigger for real-data moisture
    double autoconv = 0.0;
    if (qc_val > qc_threshold) {
        double cloud_excess = qc_val - qc_threshold;
        autoconv = 1.2e-3 * cloud_excess;
        autoconv = fmin(autoconv * dt, qc_val * 0.4) / dt;
    }

    // --- Accretion (collection of cloud water by rain) ---
    double accretion = 0.0;
    if (qr_val > 0.0 && qc_val > 0.0) {
        accretion = 2.2 * qc_val * pow(fmax(qr_val, 0.0), 0.875);
        accretion = fmin(accretion * dt, qc_val * 0.5) / dt;
    }

    // --- Rain evaporation ---
    double rain_evap = 0.0;
    double qvs = saturation_mixing_ratio_liquid(p_val, T);
    if (qr_val > 1.0e-8 && qv_val < qvs) {
        double ventilation = 1.6 + 30.3922 * pow(fmax(rho_val * qr_val, 0.0), 0.2046);
        double sat_deficit = 1.0 - qv_val / fmax(qvs, SMALL);
        rain_evap = (1.0 / rho_val) * sat_deficit * ventilation *
                    pow(fmax(rho_val * qr_val, 0.0), 0.525) /
                    (2.55e8 / fmax(p_val * qvs, SMALL) + 5.4e5);
        rain_evap = fmax(rain_evap, 0.0);
        rain_evap = fmin(rain_evap * dt, qr_val * 0.5) / dt;
        rain_evap = fmin(rain_evap, (qvs - qv_val) / fmax(dt, 1.0));
    }

    // Apply tendencies
    qc_val -= (autoconv + accretion) * dt;
    qr_val += (autoconv + accretion - rain_evap) * dt;
    qv_val += rain_evap * dt;
    T -= (LV / CP_D) * rain_evap * dt;

    // Ensure non-negative moisture
    qc_val = fmax(qc_val, 0.0);
    qr_val = fmax(qr_val, 0.0);
    qv_val = fmax(qv_val, 0.0);

    // Convert the final thermodynamic state back to theta so latent heating
    // from saturation adjustment and evaporation both feed buoyancy.
    th = T / exner;

    // Safety cap: allow realistic convective latent heating.
    // Vigorous convection routinely produces 5-10 K theta changes per
    // timestep (dt ~ 1-10s); the old 1.5 K cap suppressed real storms.
    // Scale the limit with dt so it is ~10 K/s * dt but at least 5 K.
    double max_dtheta = fmax(10.0 * dt, 5.0);
    double dtheta = th - theta_old;
    if (fabs(dtheta) > max_dtheta) {
        th = theta_old + copysign(max_dtheta, dtheta);
    }

    // Write back (cast to real_t for storage)
    theta[ijk] = (real_t)th;
    qv[ijk] = (real_t)qv_val;
    qc[ijk] = (real_t)qc_val;
    qr[ijk] = (real_t)qr_val;
}

// ----------------------------------------------------------
// Rain sedimentation kernel (terrain-following)
// Uses per-column terrain height and eta coordinates to
// compute the true layer thickness dz at each (i,j,k).
// ----------------------------------------------------------
__global__ void rain_sedimentation_kernel(
    real_t* __restrict__ qr,
    const double* __restrict__ rho_base,
    const double* __restrict__ eta,        // w-level eta [nz+1]
    const real_t* __restrict__ terrain,
    int nx, int ny, int nz,
    double ztop,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;

    double terrain_h = (double)terrain[idx2(i, j, nx)];

    // Process column top-down
    for (int k = nz - 2; k >= 1; k--) {
        int ijk = idx3(i, j, k, nx_h, ny_h);

        double qr_val = (double)qr[ijk];
        if (qr_val <= 1.0e-10) continue;

        // Terrain-following layer thickness for this column
        double dz = terrain_following_layer_thickness(terrain_h, eta[k], eta[k+1], ztop);
        dz = fmax(dz, 1.0);  // safety floor

        // Terrain-corrected density (same hypsometric approach as kessler_kernel)
        double eta_m_k = 0.5 * (eta[k] + eta[k+1]);
        double z_phys = terrain_following_height(terrain_h, eta_m_k, ztop);
        double z_flat = terrain_following_height(0.0, eta_m_k, ztop);
        double dz_shift = z_phys - z_flat;
        double rho_flat = rho_base[k];
        // Scale height using mid-troposphere temperature (~250 K)
        double H_scale = R_D * 250.0 / G;
        double rho_val = rho_flat * exp(-dz_shift / H_scale);

        // Terminal velocity (Marshall-Palmer) with terrain-corrected density
        double vt = 36.34 * pow(fmax(rho_val * qr_val * 1000.0, 0.0), 0.1364) *
                    sqrt(fmax(RHO0 / rho_val, 0.1));
        vt = fmin(vt, 0.5 * dz / dt);  // CFL limit

        // Flux out of this cell
        double flux = qr_val * vt * dt / dz;
        flux = fmin(flux, qr_val * 0.9);  // Don't remove more than 90%

        qr[ijk] = (real_t)(qr_val - flux);

        // Rain goes to cell below -- use terrain-following dz for receiving cell
        int ijkm = idx3(i, j, k-1, nx_h, ny_h);
        double dz_below = terrain_following_layer_thickness(terrain_h, eta[k-1], eta[k], ztop);
        dz_below = fmax(dz_below, 1.0);

        double eta_m_below = 0.5 * (eta[k-1] + eta[k]);
        double z_phys_below = terrain_following_height(terrain_h, eta_m_below, ztop);
        double z_flat_below = terrain_following_height(0.0, eta_m_below, ztop);
        double rho_below = rho_base[k > 0 ? k-1 : 0] * exp(-(z_phys_below - z_flat_below) / H_scale);

        qr[ijkm] = (real_t)((double)qr[ijkm] + flux * (rho_val * dz) / fmax(rho_below * dz_below, 1.0));
    }

    // Remove rain at ground level (accumulated precipitation)
    int ijk0 = idx3(i, j, 0, nx_h, ny_h);
    qr[ijk0] = (real_t)0.0;
}

// ----------------------------------------------------------
// Host driver
// ----------------------------------------------------------
void run_microphysics(StateGPU& state, const GridConfig& grid, double dt) {
    int nx = grid.nx, ny = grid.ny, nz = grid.nz;

    dim3 block(8, 8, 4);
    dim3 grid3d((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);

    kessler_kernel<<<grid3d, block>>>(
        state.theta, state.qv, state.qc, state.qr,
        state.rho_base, state.p_base,
        state.terrain, state.eta_m,
        nx, ny, nz, grid.ztop, dt
    );

    // Sedimentation
    dim3 block2d(16, 16);
    dim3 grid2d((nx + 15) / 16, (ny + 15) / 16);

    rain_sedimentation_kernel<<<grid2d, block2d>>>(
        state.qr, state.rho_base, state.eta,
        state.terrain,
        nx, ny, nz, grid.ztop, dt
    );
}

} // namespace gpuwm
