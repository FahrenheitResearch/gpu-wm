// ============================================================
// GPU-WM: Planetary Boundary Layer Parameterization
// K-profile first-order closure with implicit vertical diffusion
//
// Components:
//   1. Surface layer: Monin-Obukhov similarity with stability functions
//   2. PBL height diagnosis: Bulk Richardson number method
//   3. K-profile: u* * kappa * z * (1 - z/h)^2 within PBL
//   4. Free atmosphere: Local Richardson-limited Smagorinsky
//   5. Implicit tridiagonal solver for vertical diffusion
// ============================================================

#include "../../include/constants.cuh"
#include "../../include/grid.cuh"
#include "../../include/surface_layer.cuh"

namespace gpuwm {

// Maximum vertical levels for stack-allocated tridiagonal solver
static constexpr int MAX_NZ = 256;

// ----------------------------------------------------------
// Device helper: Bulk Richardson number at height z
// Used to diagnose PBL height
// ----------------------------------------------------------
__device__ double bulk_richardson(
    double theta_v_z, double theta_v_sfc, double z,
    double u_z, double v_z
) {
    double dth = theta_v_z - theta_v_sfc;
    double wspd2 = u_z * u_z + v_z * v_z + 0.1;  // min wind for stability
    return (G / fmax(theta_v_sfc, 200.0)) * dth * z / wspd2;
}

// ----------------------------------------------------------
// PBL kernel: surface layer + K-profile + implicit diffusion
// One thread per (i,j) column
// ----------------------------------------------------------
__global__ void pbl_column_kernel(
    real_t* __restrict__ u,
    real_t* __restrict__ v,
    real_t* __restrict__ theta,
    real_t* __restrict__ qv,
    const real_t* __restrict__ rho,
    const real_t* __restrict__ tskin,
    const real_t* __restrict__ moistmem,
    const real_t* __restrict__ terrain,  // 2D terrain height (m) [nx*ny]
    const double* __restrict__ eta_m,    // eta at mass levels [nz]
    const double* __restrict__ eta_w,    // eta at w-levels [nz+1]
    int nx, int ny, int nz,
    double dx, double dy,
    double ztop,       // model top height (m)
    double z0,         // roughness length (m)
    double qv_sfc,     // surface moisture mixing ratio (kg/kg)
    double moisture_gate_strength,  // heterogeneity-linked surface qv reduction
    double cs,         // Smagorinsky coefficient
    double dt          // timestep for implicit solver
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny || nz < 3) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;

    // Stack arrays for this column
    double z_agl[MAX_NZ];      // height AGL at mass levels
    double dz_col[MAX_NZ];     // layer thickness between w-level k and k+1
    double u_col[MAX_NZ];
    double v_col[MAX_NZ];
    double th_col[MAX_NZ];
    double qv_col[MAX_NZ];
    double rho_col[MAX_NZ];
    double Km_col[MAX_NZ];     // eddy viscosity at level interfaces

    if (nz > MAX_NZ) return;

    // ----------------------------------------------------------
    // 1. Load column data, compute terrain-following heights AGL
    // ----------------------------------------------------------
    double ter = (double)terrain[idx2(i, j, nx)];
    double col_depth = ztop - ter;
    if (col_depth < 1.0) col_depth = 1.0;  // safety over extreme terrain

    bool has_nan = false;
    for (int k = 0; k < nz; k++) {
        int ijk = idx3(i, j, k, nx_h, ny_h);
        u_col[k]   = (double)u[ijk];
        v_col[k]   = (double)v[ijk];
        th_col[k]  = (double)theta[ijk];
        qv_col[k]  = (double)qv[ijk];
        rho_col[k] = (double)rho[ijk];

        // Terrain-following height AGL at mass level k:
        //   z_ASL = terrain + eta_m[k] * (ztop - terrain)
        //   z_AGL = eta_m[k] * (ztop - terrain)
        z_agl[k] = eta_m[k] * col_depth;

        // NaN guard: skip entire column if any input is bad
        if (isnan(u_col[k]) || isnan(v_col[k]) || isnan(th_col[k]) ||
            isnan(rho_col[k])) {
            has_nan = true;
        }
    }
    if (has_nan) return;

    // Compute dz at interfaces using w-level eta spacing.
    // dz_col[k] = thickness between w-level k and w-level k+1
    //           = (eta_w[k+1] - eta_w[k]) * (ztop - terrain)
    for (int k = 0; k < nz; k++) {
        dz_col[k] = (eta_w[k + 1] - eta_w[k]) * col_depth;
        if (dz_col[k] < 1.0) dz_col[k] = 1.0;  // safety
    }

    // ----------------------------------------------------------
    // 2. Surface layer: Monin-Obukhov with stability functions
    //    Applied at k=1 (first level above surface)
    //    All heights are AGL (above ground level)
    // ----------------------------------------------------------
    double z1 = z_agl[1];       // first mass level height AGL
    if (z1 < z0) z1 = z0 + 1.0;  // safety

    double u1 = u_col[1];
    double v1 = v_col[1];
    double wspd = sqrt(u1 * u1 + v1 * v1);
    if (wspd < 0.1) wspd = 0.1;  // minimum wind speed

    double th1 = th_col[1];
    double qv1 = qv_col[1];
    double rho1 = rho_col[1];

    double theta_skin = (double)tskin[idx2(i, j, nx)];
    bool has_second_level = (nz > 2);
    double th2 = has_second_level ? th_col[2] : th1;
    double qv2 = has_second_level ? qv_col[2] : qv1;
    double z2 = has_second_level ? z_agl[2] : (z1 + 1.0);
    SurfaceLayerState sfc = diagnose_surface_layer_state(
        theta_skin, qv_sfc, u1, v1, th1, qv1, z1,
        has_second_level, th2, qv2, z2, z0
    );
    double theta_sfc_local = sfc.theta_sfc_local;
    double qv_sfc_local = sfc.qv_sfc_local;
    double moisture_scale = (double)moistmem[idx2(i, j, nx)];
    if (!(moisture_scale > 0.0)) {
        double thermal_range = skin_theta_range_3x3_field(tskin, i, j, nx, ny, theta_skin);
        double terrain_relief = terrain_relief_3x3_field(terrain, i, j, nx, ny);
        double terrain_slope = terrain_slope_2d_field(terrain, i, j, nx, ny, dx, dy);
        double moisture_activation = admittance_seam_factor(
            thermal_range, terrain_relief, terrain_slope, 1.0
        );
        moisture_scale = moisture_availability_scale(
            moisture_activation, fmax(moisture_gate_strength, 0.0)
        );
    }
    qv_sfc_local = apply_surface_moisture_scale(qv_sfc_local, qv1, moisture_scale);
    double Ri_sfc = sfc.ri_sfc;
    double Cd = sfc.cd;
    wspd = sfc.wspd;

    // Friction velocity
    double ustar = sqrt(Cd) * wspd;
    ustar = fmax(ustar, 0.001);  // min u*

    // Surface stress: tau = rho * Cd * wspd * wind_component
    // Limit |tau| < 5 Pa
    double tau_x = rho1 * Cd * wspd * u1;
    double tau_y = rho1 * Cd * wspd * v1;
    double tau_mag = sqrt(tau_x * tau_x + tau_y * tau_y);
    if (tau_mag > 5.0) {
        double scale = 5.0 / tau_mag;
        tau_x *= scale;
        tau_y *= scale;
    }

    // Surface heat flux: H = rho * Cp * Ch * wspd * (theta_sfc - theta1)
    // Use Ch = Cd (neutral Prandtl number = 1)
    double Ch = Cd;
    double shf = rho1 * CP_D * Ch * wspd * (theta_sfc_local - th1);
    shf = fmax(-1000.0, fmin(shf, 1000.0));  // limit |H| < 1000 W/m^2

    // Convert surface flux to theta tendency for k=1
    // d(theta)/dt = H / (rho * Cp * dz)
    double dz1 = dz_col[0];  // thickness of first layer

    // Apply surface drag as direct modification to u, v at k=1
    // d(u)/dt = -tau_x / (rho * dz)
    u_col[1] -= dt * tau_x / (rho1 * dz1);
    v_col[1] -= dt * tau_y / (rho1 * dz1);

    // Apply surface heat flux
    th_col[1] += dt * shf / (rho1 * CP_D * dz1);

    // Surface moisture flux
    double moist_flux = Ch * wspd * (qv_sfc_local - qv1);
    moist_flux = fmax(-0.01, fmin(moist_flux, 0.01));  // limit
    qv_col[1] += dt * moist_flux / dz1;

    // qv_col[1] already updated above; written back in step 6

    // ----------------------------------------------------------
    // 3. Diagnose PBL height from bulk Richardson number
    //    h = height AGL where Ri_bulk > 0.25
    //    All heights are AGL so PBL depth is relative to terrain
    // ----------------------------------------------------------
    double pbl_h = z_agl[1];  // minimum PBL height = first level AGL
    double theta_ref = surface_layer_virtual_theta(th_col[1], qv_col[1]);  // reference theta near surface

    for (int k = 2; k < nz; k++) {
        double Ri_bulk = bulk_richardson(
            surface_layer_virtual_theta(th_col[k], qv_col[k]), theta_ref, z_agl[k],
            u_col[k] - u_col[1], v_col[k] - v_col[1]
        );
        if (Ri_bulk > 0.25) {
            // Interpolate between k-1 and k
            double Ri_prev = bulk_richardson(
                surface_layer_virtual_theta(th_col[k-1], qv_col[k-1]), theta_ref, z_agl[k-1],
                u_col[k-1] - u_col[1], v_col[k-1] - v_col[1]
            );
            double frac = (0.25 - Ri_prev) / fmax(Ri_bulk - Ri_prev, 1.0e-6);
            frac = fmax(0.0, fmin(frac, 1.0));
            pbl_h = z_agl[k-1] + frac * (z_agl[k] - z_agl[k-1]);
            break;
        }
        pbl_h = z_agl[k];
    }

    // Bound PBL height AGL: at least 100m, at most 5000m
    pbl_h = fmax(pbl_h, 100.0);
    pbl_h = fmin(pbl_h, 5000.0);

    // ----------------------------------------------------------
    // 4. Compute eddy viscosity Km at each level interface
    //    - Within PBL (z_AGL < h): K-profile = kappa * u* * z * (1 - z/h)^2
    //    - Above PBL: local Ri-based Smagorinsky (limited)
    //    Heights are all AGL so K-profile shape is relative to terrain
    // ----------------------------------------------------------
    Km_col[0] = 0.0;  // surface: no diffusion through bottom

    for (int k = 1; k < nz - 1; k++) {
        // Interface height AGL -- midpoint between mass levels k and k+1
        double z_k = 0.5 * (z_agl[k] + z_agl[k + 1]);

        if (z_k < pbl_h) {
            // Within PBL: K-profile parameterization
            double zoh = z_k / pbl_h;
            double one_minus = 1.0 - zoh;
            if (one_minus < 0.0) one_minus = 0.0;
            Km_col[k] = KARMAN * ustar * z_k * one_minus * one_minus;
        } else {
            // Above PBL: local Smagorinsky with Richardson damping
            // dz_col[k] is the terrain-following layer thickness
            double dz_k = dz_col[k];
            double dudz = (u_col[k + 1] - u_col[k]) / dz_k;
            double dvdz = (v_col[k + 1] - v_col[k]) / dz_k;
            double S2 = dudz * dudz + dvdz * dvdz;
            double S = sqrt(fmax(S2, 0.0));

            double delta = cbrt(dx * dy * dz_k);
            Km_col[k] = cs * cs * delta * delta * S;

            // Local Richardson number stability correction
            double thv_k = surface_layer_virtual_theta(th_col[k], qv_col[k]);
            double thv_kp1 = surface_layer_virtual_theta(th_col[k + 1], qv_col[k + 1]);
            double dthvdz = (thv_kp1 - thv_k) / dz_k;
            double thv_avg = 0.5 * (thv_k + thv_kp1);
            double Ri_loc = (G / fmax(thv_avg, 200.0)) * dthvdz / fmax(S2, SMALL);

            if (Ri_loc > 0.0) {
                // Stable: reduce mixing, shut off above Ri_crit = 0.25
                Km_col[k] *= fmax(1.0 - Ri_loc / 0.25, 0.0);
            }
            // Unstable: no enhancement (keep Smagorinsky value)
        }

        // Clamp Km: 0 to 500 m^2/s
        Km_col[k] = fmax(Km_col[k], 0.0);
        Km_col[k] = fmin(Km_col[k], 500.0);
    }
    Km_col[nz - 1] = 0.0;  // no diffusion through top

    // ----------------------------------------------------------
    // 5. Implicit vertical diffusion using Thomas algorithm
    //    Solve: phi_new = phi_old + dt * d/dz(Km * d(phi)/dz)
    //
    //    Discretized on the column interior (k=1 to k=nz-2):
    //      phi_new[k] - dt/(dz_m[k]) * (
    //          Km[k]  *(phi[k+1]-phi[k])  /dz[k]
    //        - Km[k-1]*(phi[k]-phi[k-1])  /dz[k-1]
    //      ) = phi_old[k]
    //
    //    This gives a tridiagonal system: a[k]*x[k-1] + b[k]*x[k] + c[k]*x[k+1] = d[k]
    //
    //    k=0 and k=nz-1 are NOT modified (boundary levels).
    // ----------------------------------------------------------

    // Tridiagonal coefficients
    double a[MAX_NZ], b[MAX_NZ], c_arr[MAX_NZ], d_arr[MAX_NZ];

    // Helper: dz at mass level k (distance from k-0.5 to k+0.5)
    // Approximate as average of surrounding interface spacings
    auto dz_mass = [&](int k) -> double {
        if (k <= 0) return dz_col[0];
        if (k >= nz - 1) return dz_col[nz - 2];
        return 0.5 * (dz_col[k - 1] + dz_col[k]);
    };

    // Compute Kh from Km (eddy diffusivity for heat)
    // Prandtl number: ~3 in convective (unstable), ~1 in stable/neutral
    auto Kh_from_Km = [&](int k) -> double {
        // Within PBL and unstable => Kh = 3 * Km
        // Otherwise Kh = Km
        if (z_agl[k] < pbl_h && Ri_sfc < 0.0) {
            return fmin(3.0 * Km_col[k], 500.0);
        }
        return Km_col[k];
    };

    // --- Solve for u ---
    // Build tridiagonal system
    for (int k = 0; k < nz; k++) {
        if (k == 0 || k == nz - 1) {
            // Boundary: no change
            a[k] = 0.0;
            b[k] = 1.0;
            c_arr[k] = 0.0;
            d_arr[k] = u_col[k];
        } else {
            double dzm = dz_mass(k);
            double alpha_lo = dt * Km_col[k - 1] / (dz_col[k - 1] * dzm);
            double alpha_hi = dt * Km_col[k]     / (dz_col[k]     * dzm);

            a[k]     = -alpha_lo;
            c_arr[k] = -alpha_hi;
            b[k]     = 1.0 + alpha_lo + alpha_hi;
            d_arr[k] = u_col[k];
        }
    }

    // Thomas algorithm forward sweep
    for (int k = 1; k < nz; k++) {
        if (fabs(b[k - 1]) < SMALL) continue;  // safety
        double m = a[k] / b[k - 1];
        b[k]     -= m * c_arr[k - 1];
        d_arr[k] -= m * d_arr[k - 1];
    }
    // Back substitution
    if (fabs(b[nz - 1]) > SMALL) {
        u_col[nz - 1] = d_arr[nz - 1] / b[nz - 1];
    }
    for (int k = nz - 2; k >= 0; k--) {
        if (fabs(b[k]) > SMALL) {
            u_col[k] = (d_arr[k] - c_arr[k] * u_col[k + 1]) / b[k];
        }
    }

    // --- Solve for v ---
    for (int k = 0; k < nz; k++) {
        if (k == 0 || k == nz - 1) {
            a[k] = 0.0;
            b[k] = 1.0;
            c_arr[k] = 0.0;
            d_arr[k] = v_col[k];
        } else {
            double dzm = dz_mass(k);
            double alpha_lo = dt * Km_col[k - 1] / (dz_col[k - 1] * dzm);
            double alpha_hi = dt * Km_col[k]     / (dz_col[k]     * dzm);

            a[k]     = -alpha_lo;
            c_arr[k] = -alpha_hi;
            b[k]     = 1.0 + alpha_lo + alpha_hi;
            d_arr[k] = v_col[k];
        }
    }
    for (int k = 1; k < nz; k++) {
        if (fabs(b[k - 1]) < SMALL) continue;
        double m = a[k] / b[k - 1];
        b[k]     -= m * c_arr[k - 1];
        d_arr[k] -= m * d_arr[k - 1];
    }
    if (fabs(b[nz - 1]) > SMALL) {
        v_col[nz - 1] = d_arr[nz - 1] / b[nz - 1];
    }
    for (int k = nz - 2; k >= 0; k--) {
        if (fabs(b[k]) > SMALL) {
            v_col[k] = (d_arr[k] - c_arr[k] * v_col[k + 1]) / b[k];
        }
    }

    // --- Solve for theta (use Kh instead of Km) ---
    for (int k = 0; k < nz; k++) {
        if (k == 0 || k == nz - 1) {
            a[k] = 0.0;
            b[k] = 1.0;
            c_arr[k] = 0.0;
            d_arr[k] = th_col[k];
        } else {
            double dzm = dz_mass(k);
            double Kh_lo = Kh_from_Km(k - 1);
            double Kh_hi = Kh_from_Km(k);
            double alpha_lo = dt * Kh_lo / (dz_col[k - 1] * dzm);
            double alpha_hi = dt * Kh_hi / (dz_col[k]     * dzm);

            a[k]     = -alpha_lo;
            c_arr[k] = -alpha_hi;
            b[k]     = 1.0 + alpha_lo + alpha_hi;
            d_arr[k] = th_col[k];
        }
    }
    for (int k = 1; k < nz; k++) {
        if (fabs(b[k - 1]) < SMALL) continue;
        double m = a[k] / b[k - 1];
        b[k]     -= m * c_arr[k - 1];
        d_arr[k] -= m * d_arr[k - 1];
    }
    if (fabs(b[nz - 1]) > SMALL) {
        th_col[nz - 1] = d_arr[nz - 1] / b[nz - 1];
    }
    for (int k = nz - 2; k >= 0; k--) {
        if (fabs(b[k]) > SMALL) {
            th_col[k] = (d_arr[k] - c_arr[k] * th_col[k + 1]) / b[k];
        }
    }

    // --- Solve for qv using Kh ---
    for (int k = 0; k < nz; k++) {
        if (k == 0 || k == nz - 1) {
            a[k] = 0.0;
            b[k] = 1.0;
            c_arr[k] = 0.0;
            d_arr[k] = qv_col[k];
        } else {
            double dzm = dz_mass(k);
            double Kh_lo = Kh_from_Km(k - 1);
            double Kh_hi = Kh_from_Km(k);
            double alpha_lo = dt * Kh_lo / (dz_col[k - 1] * dzm);
            double alpha_hi = dt * Kh_hi / (dz_col[k]     * dzm);

            a[k]     = -alpha_lo;
            c_arr[k] = -alpha_hi;
            b[k]     = 1.0 + alpha_lo + alpha_hi;
            d_arr[k] = qv_col[k];
        }
    }
    for (int k = 1; k < nz; k++) {
        if (fabs(b[k - 1]) < SMALL) continue;
        double m = a[k] / b[k - 1];
        b[k]     -= m * c_arr[k - 1];
        d_arr[k] -= m * d_arr[k - 1];
    }
    if (fabs(b[nz - 1]) > SMALL) {
        qv_col[nz - 1] = d_arr[nz - 1] / b[nz - 1];
    }
    for (int k = nz - 2; k >= 0; k--) {
        if (fabs(b[k]) > SMALL) {
            qv_col[k] = (d_arr[k] - c_arr[k] * qv_col[k + 1]) / b[k];
        }
        qv_col[k] = fmax(qv_col[k], 0.0);
    }

    // ----------------------------------------------------------
    // 6. Write updated fields back to global memory
    //    Don't modify k=0 or k=nz-1
    // ----------------------------------------------------------
    for (int k = 1; k < nz - 1; k++) {
        int ijk = idx3(i, j, k, nx_h, ny_h);
        u[ijk]     = (real_t)u_col[k];
        v[ijk]     = (real_t)v_col[k];
        theta[ijk] = (real_t)th_col[k];
        qv[ijk]    = (real_t)qv_col[k];
    }
}

// ----------------------------------------------------------
// Host driver
// ----------------------------------------------------------
void run_pbl(StateGPU& state, const GridConfig& grid,
             double z0, double qv_sfc, double moisture_gate_strength, double cs, double dt) {
    int nx = grid.nx, ny = grid.ny, nz = grid.nz;

    dim3 block2d(16, 16);
    dim3 grid2d((nx + 15) / 16, (ny + 15) / 16);

    pbl_column_kernel<<<grid2d, block2d>>>(
        state.u, state.v, state.theta, state.qv,
        state.rho, state.tskin, state.moistmem,
        state.terrain,       // 2D terrain height field
        state.eta_m,         // eta at mass levels [nz]
        state.eta,           // eta at w-levels [nz+1]
        nx, ny, nz, grid.dx, grid.dy,
        grid.ztop,           // model top height
        z0, qv_sfc, moisture_gate_strength, cs,
        dt
    );
}

} // namespace gpuwm
