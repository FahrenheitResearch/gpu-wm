// ============================================================
// GPU-WM: Simplified Thompson Microphysics
// Mixed-phase bulk microphysics scheme with ice processes
//
// Extends Kessler warm-rain by adding:
//   - Cloud ice (vapor deposition when T < 273.15 K)
//   - Snow (ice aggregation and riming)
//   - Graupel (heavy riming of snow by supercooled water)
//   - Bergeron process (ice growth at expense of water)
//   - Ice nucleation (Cooper 1986)
//   - Melting / freezing transitions
//   - Multi-species sedimentation
//
// Field packing (reuses existing StateGPU arrays):
//   qc = cloud water (T >= 273.15 K) or cloud ice  (T < 273.15 K)
//   qr = rain        (T >= 273.15 K) or snow+graupel(T < 273.15 K)
// The kernel tracks qi, qs, qg internally per grid point and
// writes the combined value back to qc/qr at the end.
// ============================================================

#include "../../include/constants.cuh"
#include "../../include/grid.cuh"
#include "../../include/microphysics.cuh"

namespace gpuwm {

// -------------------------------------------------------
// Temperature thresholds
// -------------------------------------------------------
static constexpr double T_FREEZE   = 273.15;   // 0 C
static constexpr double T_HOMOG    = 233.15;   // -40 C  (homogeneous freezing)
static constexpr double T_ICE_ONLY = 253.15;   // -20 C  (pure ice saturation)

// -------------------------------------------------------
// Saturation helpers
// -------------------------------------------------------

// Saturation vapor pressure over liquid water (Bolton 1980)
__device__ double esl_bolton(double T) {
    T = fmax(fmin(T, 350.0), 180.0);
    return 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65));
}

// Saturation vapor pressure over ice
//   esi = 611.2 * exp(21.8745584 * (T - 273.16) / (T - 7.66))
__device__ double esi_murphy(double T) {
    T = fmax(fmin(T, 273.16), 100.0);
    return 611.2 * exp(21.8745584 * (T - 273.16) / (T - 7.66));
}

// Blended saturation pressure: liquid above T_FREEZE,
// ice below T_ICE_ONLY, linear mix in between.
__device__ double esat_blend(double T) {
    double es_liq = esl_bolton(T);
    double es_ice = esi_murphy(T);

    if (T >= T_FREEZE)   return es_liq;
    if (T <= T_ICE_ONLY) return es_ice;

    // Linear blend in mixed-phase zone
    double frac_liq = (T - T_ICE_ONLY) / (T_FREEZE - T_ICE_ONLY);
    return frac_liq * es_liq + (1.0 - frac_liq) * es_ice;
}

// Saturation mixing ratio using blended es
__device__ double qvs_blend(double p, double T) {
    double es = esat_blend(T);
    if (es >= p * 0.5) return 0.1;
    return 0.622 * es / fmax(p - es, 1.0);
}

// Saturation mixing ratio over ice only
__device__ double qvs_ice(double p, double T) {
    double es = esi_murphy(T);
    if (es >= p * 0.5) return 0.1;
    return 0.622 * es / fmax(p - es, 1.0);
}

// Saturation mixing ratio over liquid only
__device__ double qvs_liquid(double p, double T) {
    double es = esl_bolton(T);
    if (es >= p * 0.5) return 0.1;
    return 0.622 * es / fmax(p - es, 1.0);
}

// -------------------------------------------------------
// Cooper (1986) ice nucleation: number conc. per m^3
//   N_ice = 5e3 * exp(0.304 * (273.15 - T))  [per m^3]
//   (original is 5 per liter = 5e3 per m^3)
// -------------------------------------------------------
__device__ double cooper_ice_number(double T) {
    if (T >= T_FREEZE) return 0.0;
    double dT = fmin(T_FREEZE - T, 50.0);   // clamp to avoid overflow
    return 5.0e3 * exp(0.304 * dT);
}

// -------------------------------------------------------
// Thompson microphysics kernel (terrain-following)
// -------------------------------------------------------
__global__ void thompson_kernel(
    real_t* __restrict__ theta,
    real_t* __restrict__ qv,
    real_t* __restrict__ qc,   // cloud water OR cloud ice (packed)
    real_t* __restrict__ qr,   // rain OR snow+graupel (packed)
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
    int ijk  = idx3(i, j, k, nx_h, ny_h);

    double th      = (double)theta[ijk];
    double qv_val  = (double)qv[ijk];
    double qc_val  = (double)qc[ijk];
    double qr_val  = (double)qr[ijk];

    // --- Terrain-following pressure and density ---
    double terrain_h = (double)terrain[idx2(i, j, nx)];
    double z_phys = terrain_following_height(terrain_h, eta_m[k], ztop);
    double z_flat = terrain_following_height(0.0, eta_m[k], ztop);
    double dz_shift = z_phys - z_flat;

    double p_flat   = p_base[k];
    double rho_flat = rho_base[k];
    double T_base_est = p_flat / (rho_flat * R_D);
    double H_scale = R_D * T_base_est / G;
    double correction = exp(-dz_shift / H_scale);

    double p_val   = p_flat * correction;
    double rho_val = rho_flat * correction;

    // Temperature from potential temperature
    double exner = exner_from_pressure(p_val);
    double T     = th * exner;

    // Guard against unphysical values
    if (T < 100.0 || T > 350.0 || isnan(th)) return;

    // ----- Unpack qc / qr into warm and cold species -----
    // Warm regime (T >= T_FREEZE): qc is cloud water, qr is rain
    // Cold regime (T <  T_FREEZE): qc is cloud ice,   qr is snow+graupel
    double qc_water = 0.0, qi_ice = 0.0;
    double qr_rain  = 0.0, qs_snow = 0.0, qg_graupel = 0.0;

    double warm_frac = clamp01((T - T_ICE_ONLY) / (T_FREEZE - T_ICE_ONLY));
    double cold_frac = 1.0 - warm_frac;

    if (T >= T_FREEZE) {
        qc_water = qc_val;
        qr_rain  = qr_val;
    } else if (T <= T_ICE_ONLY) {
        qi_ice   = qc_val;
        // Split qr into snow (60 %) and graupel (40 %) as a simple heuristic
        qs_snow    = qr_val * 0.6;
        qg_graupel = qr_val * 0.4;
    } else {
        qc_water   = warm_frac * qc_val;
        qi_ice     = cold_frac * qc_val;
        qr_rain    = warm_frac * qr_val;
        qs_snow    = cold_frac * qr_val * 0.6;
        qg_graupel = cold_frac * qr_val * 0.4;
    }

    // Maximum tendency per step (stability guard)
    double max_tend = 0.002 * dt;   // 2 g/kg per second

    // ===================================================
    // 1.  Instantaneous saturation adjustment (Newton iteration)
    //     Standard NWP practice: iterate to exact equilibrium.
    //     3 Newton iterations with the implicit Clausius-Clapeyron
    //     denominator converge to machine precision for typical
    //     atmospheric conditions.
    // ===================================================
    if (T >= T_FREEZE) {
        // --- Warm: condense / evaporate cloud water ---
        for (int iter = 0; iter < 3; ++iter) {
            double qvs_liq = saturation_mixing_ratio_liquid(p_val, T);
            double excess_w = qv_val - qvs_liq;
            double denom = 1.0 + LV * LV * qvs_liq / (CP_D * R_V * T * T);

            if (excess_w > 0.0) {
                double cond = excess_w / denom;
                cond = fmin(cond, qv_val);
                cond = fmax(cond, 0.0);
                qc_water += cond;
                qv_val   -= cond;
                T        += LV * cond / CP_D;
            } else if (qc_water > 0.0 && excess_w < 0.0) {
                double evap = -excess_w / denom;
                evap = fmin(evap, qc_water);
                evap = fmax(evap, 0.0);
                qc_water -= evap;
                qv_val   += evap;
                T        -= LV * evap / CP_D;
            } else {
                break;
            }
        }
    } else {
        // --- Cold: deposit / sublimate cloud ice ---
        for (int iter = 0; iter < 3; ++iter) {
            double qvsi = saturation_mixing_ratio_ice(p_val, T);
            double excess_ice = qv_val - qvsi;
            double denom_ice  = 1.0 + LS * LS * qvsi / (CP_D * R_V * T * T);

            if (excess_ice > 0.0) {
                double dep = excess_ice / denom_ice;
                dep = fmin(dep, qv_val);
                dep = fmax(dep, 0.0);
                qi_ice += dep;
                qv_val -= dep;
                T      += LS * dep / CP_D;
            } else if (qi_ice > 0.0 && excess_ice < 0.0) {
                double subl = -excess_ice / denom_ice;
                subl = fmin(subl, qi_ice);
                subl = fmax(subl, 0.0);
                qi_ice -= subl;
                qv_val += subl;
                T      -= LS * subl / CP_D;
            } else {
                break;
            }
        }
    }

    // ===================================================
    // 2.  Ice nucleation (Cooper 1986)
    //     Only in the cold regime where super-saturated
    //     w.r.t. ice and qi is very small.
    // ===================================================
    if (T < T_FREEZE && T > T_HOMOG) {
        double qvsi = saturation_mixing_ratio_ice(p_val, T);
        if (qv_val > qvsi && qi_ice < 1.0e-6) {
            double N_ice = cooper_ice_number(T);        // per m^3
            double m_ice_crystal = 1.0e-12;             // mass of one crystal ~1e-12 kg
            double qi_nucleated  = N_ice * m_ice_crystal / rho_val;
            qi_nucleated = fmin(qi_nucleated, qv_val);
            qi_nucleated = fmin(qi_nucleated, max_tend);
            qi_nucleated = fmax(qi_nucleated, 0.0);
            qi_ice += qi_nucleated;
            qv_val -= qi_nucleated;
            T      += LS * qi_nucleated / CP_D;
        }
    }

    // ===================================================
    // 3.  Bergeron process
    //     Mixed-phase zone: -40 C < T < 0 C
    //     Ice grows at the expense of liquid water because
    //     esl > esi => environment super-saturated w.r.t. ice
    //     while sub-saturated w.r.t. liquid.
    // ===================================================
    if (T > T_HOMOG && T < T_FREEZE && qc_water > 1.0e-8 && qi_ice > 1.0e-8) {
        double esl = esl_bolton(T);
        double esi = esi_murphy(T);
        // Rate proportional to saturation difference
        double bergeron_rate = 1.0e-3 * (esl - esi) / fmax(esl, 1.0);
        double transfer = bergeron_rate * qc_water * dt;
        transfer = fmin(transfer, qc_water * 0.5);
        transfer = fmin(transfer, max_tend);
        transfer = fmax(transfer, 0.0);
        qc_water -= transfer;
        qi_ice   += transfer;
        // Latent heat: sublimation releases LS, evaporation absorbs LV
        // Net release = LF per unit mass transferred
        T += LF * transfer / CP_D;
    }

    // ===================================================
    // 4.  Autoconversion
    // ===================================================

    // 4a. Warm rain autoconversion (Kessler)
    double autoconv_rain = 0.0;
    if (qc_water > 1.0e-3) {
        autoconv_rain = 1.0e-3 * (qc_water - 1.0e-3);
        autoconv_rain = fmin(autoconv_rain * dt, qc_water * 0.5) / dt;
    }

    // 4b. Ice -> snow autoconversion (threshold-based)
    double autoconv_snow = 0.0;
    double qi_threshold = 5.0e-4;   // 0.5 g/kg
    if (qi_ice > qi_threshold) {
        autoconv_snow = 5.0e-4 * (qi_ice - qi_threshold);
        autoconv_snow = fmin(autoconv_snow * dt, qi_ice * 0.5) / dt;
    }

    // ===================================================
    // 5.  Accretion / collection
    // ===================================================

    // 5a. Rain collecting cloud water
    double accr_rain_cw = 0.0;
    if (qr_rain > 0.0 && qc_water > 0.0) {
        accr_rain_cw = 2.2 * qc_water * pow(fmax(qr_rain, 0.0), 0.875);
        accr_rain_cw = fmin(accr_rain_cw * dt, qc_water * 0.5) / dt;
    }

    // 5b. Snow collecting cloud water = riming
    //     If riming is heavy, product is graupel instead
    double riming_snow   = 0.0;
    double riming_graupel = 0.0;
    if (qs_snow > 1.0e-8 && qc_water > 1.0e-8 && T < T_FREEZE) {
        double rime_rate = 1.5 * qc_water * pow(fmax(qs_snow, 0.0), 0.875);
        rime_rate = fmin(rime_rate * dt, qc_water * 0.5) / dt;

        // Split: if lots of supercooled water, product -> graupel
        if (qc_water > 5.0e-4) {
            // Heavy riming -> graupel
            riming_graupel = rime_rate * 0.7;
            riming_snow    = rime_rate * 0.3;
        } else {
            // Light riming -> snow
            riming_snow    = rime_rate * 0.9;
            riming_graupel = rime_rate * 0.1;
        }
    }

    // 5c. Snow collecting cloud ice (aggregation)
    double aggr_snow = 0.0;
    if (qs_snow > 1.0e-8 && qi_ice > 1.0e-8) {
        aggr_snow = 0.5 * qi_ice * pow(fmax(qs_snow, 0.0), 0.5);
        aggr_snow = fmin(aggr_snow * dt, qi_ice * 0.5) / dt;
    }

    // ===================================================
    // 6.  Rain evaporation (same as Kessler, warm only)
    // ===================================================
    double rain_evap = 0.0;
    if (qr_rain > 1.0e-8 && T >= T_FREEZE) {
        double qvs_liq = qvs_liquid(p_val, T);
        if (qv_val < qvs_liq) {
            double ventilation = 1.6 + 30.3922 * pow(fmax(rho_val * qr_rain, 0.0), 0.2046);
            double sat_deficit = 1.0 - qv_val / fmax(qvs_liq, SMALL);
            rain_evap = (1.0 / rho_val) * sat_deficit * ventilation *
                        pow(fmax(rho_val * qr_rain, 0.0), 0.525) /
                        (2.55e8 / fmax(p_val * qvs_liq, SMALL) + 5.4e5);
            rain_evap = fmax(rain_evap, 0.0);
            rain_evap = fmin(rain_evap * dt, qr_rain * 0.5) / dt;
            rain_evap = fmin(rain_evap, (qvs_liq - qv_val) / fmax(dt, 1.0));
        }
    }

    // ===================================================
    // 7.  Snow / ice sublimation (cold regime, sub-saturated)
    // ===================================================
    double snow_subl = 0.0;
    if (qs_snow > 1.0e-8 && T < T_FREEZE) {
        double qvsi = qvs_ice(p_val, T);
        if (qv_val < qvsi) {
            double deficit = 1.0 - qv_val / fmax(qvsi, SMALL);
            snow_subl = 5.0e-4 * deficit * qs_snow;
            snow_subl = fmin(snow_subl * dt, qs_snow * 0.5) / dt;
        }
    }

    // ===================================================
    // 8.  Melting  (snow/ice/graupel when T > T_FREEZE)
    // ===================================================
    double melt_ice = 0.0, melt_snow = 0.0, melt_graupel = 0.0;
    if (T > T_FREEZE) {
        double dT_above = T - T_FREEZE;
        // Melt rate proportional to temperature excess
        double melt_rate = 5.0e-3 * dT_above;

        if (qi_ice > 1.0e-10) {
            melt_ice = melt_rate * qi_ice;
            melt_ice = fmin(melt_ice * dt, qi_ice) / dt;
        }
        if (qs_snow > 1.0e-10) {
            melt_snow = melt_rate * qs_snow;
            melt_snow = fmin(melt_snow * dt, qs_snow) / dt;
        }
        if (qg_graupel > 1.0e-10) {
            melt_graupel = melt_rate * qg_graupel;
            melt_graupel = fmin(melt_graupel * dt, qg_graupel) / dt;
        }
    }

    // ===================================================
    // 9.  Freezing
    //     Homogeneous:   T < -40 C => all liquid -> ice
    //     Heterogeneous: T < 0 C   => probabilistic
    // ===================================================
    double freeze_cw = 0.0;
    double freeze_rain = 0.0;

    if (T < T_HOMOG) {
        // Homogeneous freezing - all liquid instantly
        if (qc_water > 1.0e-10) {
            freeze_cw = qc_water / dt;
        }
        if (qr_rain > 1.0e-10) {
            freeze_rain = qr_rain / dt;
        }
    } else if (T < T_FREEZE) {
        // Heterogeneous freezing (probabilistic, increases with supercooling)
        double supercool = T_FREEZE - T;
        double freeze_frac = 1.0e-4 * supercool * supercool;  // quadratic ramp
        freeze_frac = fmin(freeze_frac, 0.1);  // cap at 10 % per step

        if (qc_water > 1.0e-8) {
            freeze_cw = freeze_frac * qc_water / dt;
        }
        if (qr_rain > 1.0e-8) {
            freeze_rain = freeze_frac * qr_rain / dt;
        }
    }

    // ===================================================
    // Apply all tendencies
    // ===================================================

    // Warm cloud water
    qc_water -= (autoconv_rain + accr_rain_cw + riming_snow + riming_graupel + freeze_cw) * dt;
    qc_water += melt_ice * dt;
    qc_water  = fmax(qc_water, 0.0);

    // Cloud ice
    qi_ice -= (autoconv_snow + aggr_snow + melt_ice) * dt;
    qi_ice += freeze_cw * dt;
    qi_ice  = fmax(qi_ice, 0.0);

    // Rain
    qr_rain += (autoconv_rain + accr_rain_cw - rain_evap - freeze_rain) * dt;
    qr_rain += (melt_snow + melt_graupel) * dt;
    qr_rain  = fmax(qr_rain, 0.0);

    // Snow
    qs_snow += (autoconv_snow + aggr_snow + riming_snow - melt_snow - snow_subl) * dt;
    qs_snow  = fmax(qs_snow, 0.0);

    // Graupel
    qg_graupel += (riming_graupel + freeze_rain - melt_graupel) * dt;
    qg_graupel  = fmax(qg_graupel, 0.0);

    // Vapor adjustments from evaporation / sublimation
    qv_val += (rain_evap + snow_subl) * dt;
    T      -= LV * rain_evap * dt / CP_D;
    T      -= LS * snow_subl * dt / CP_D;

    // Latent heat from melting (absorbs LF) and freezing (releases LF)
    T -= LF * (melt_ice + melt_snow + melt_graupel) * dt / CP_D;
    T += LF * (freeze_cw + freeze_rain) * dt / CP_D;

    // Ensure non-negative vapor
    qv_val = fmax(qv_val, 0.0);

    // Update theta from modified temperature
    double theta_old = th;
    th = T / exner;

    // Safety cap: allow realistic convective latent heating.
    // With ice processes (freezing + deposition) the total latent heating
    // can exceed warm-rain-only rates.  Scale with dt, floor at 5 K.
    double max_dtheta = fmax(10.0 * dt, 5.0);
    double dtheta = th - theta_old;
    if (fabs(dtheta) > max_dtheta) {
        th = theta_old + copysign(max_dtheta, dtheta);
    }

    // ----- Re-pack into qc / qr based on updated temperature -----
    // After all processes, temperature may have changed regime.
    // Pack: qc gets the dominant cloud condensate, qr gets precip.
    double packed_qc, packed_qr;
    if (T >= T_FREEZE) {
        // Warm: residual ice should have melted (handled above)
        packed_qc = qc_water + qi_ice;
        packed_qr = qr_rain + qs_snow + qg_graupel;
    } else if (T <= T_ICE_ONLY) {
        // Cold: residual water should have frozen (handled above)
        packed_qc = qi_ice + qc_water;
        packed_qr = qs_snow + qg_graupel + qr_rain;
    } else {
        // Mixed phase: preserve total condensate, but keep the split smooth
        packed_qc = qc_water + qi_ice;
        packed_qr = qr_rain + qs_snow + qg_graupel;
    }

    // Write back (cast to real_t for storage)
    theta[ijk] = (real_t)th;
    qv[ijk]    = (real_t)qv_val;
    qc[ijk]    = (real_t)fmax(packed_qc, 0.0);
    qr[ijk]    = (real_t)fmax(packed_qr, 0.0);
}

// -------------------------------------------------------
// Multi-species sedimentation kernel (terrain-following)
//   Rain   ~ 5 m/s
//   Snow   ~ 1 m/s
//   Graupel~ 4 m/s
// Uses per-column terrain height and eta coordinates to
// compute the true layer thickness dz at each (i,j,k).
// -------------------------------------------------------
__global__ void thompson_sedimentation_kernel(
    real_t* __restrict__ qr,
    real_t* __restrict__ qc,
    const double* __restrict__ rho_base,
    const double* __restrict__ p_base,
    const double* __restrict__ eta,        // w-level eta [nz+1]
    const real_t* __restrict__ terrain,
    const real_t* __restrict__ theta_field,
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
    // Pre-compute scale height for density correction (mid-troposphere approx)
    double H_scale = R_D * 250.0 / G;

    // --- Precipitating species (qr): rain / snow / graupel ---
    for (int k = nz - 2; k >= 1; k--) {
        int ijk = idx3(i, j, k, nx_h, ny_h);
        double qr_val = (double)qr[ijk];
        if (qr_val <= 1.0e-10) continue;

        // Terrain-following layer thickness
        double dz = terrain_following_layer_thickness(terrain_h, eta[k], eta[k+1], ztop);
        dz = fmax(dz, 1.0);

        // Terrain-corrected density
        double eta_m_k = 0.5 * (eta[k] + eta[k+1]);
        double z_phys = terrain_following_height(terrain_h, eta_m_k, ztop);
        double z_flat = terrain_following_height(0.0, eta_m_k, ztop);
        double rho_val = rho_base[k] * exp(-(z_phys - z_flat) / H_scale);

        // Terrain-corrected pressure for temperature computation
        double p_val = p_base[k] * exp(-(z_phys - z_flat) / H_scale);
        double exner   = pow(p_val / P0, KAPPA);
        double T       = (double)theta_field[ijk] * exner;

        // Temperature-dependent terminal velocity
        double vt_rain = 36.34 * pow(fmax(rho_val * qr_val * 1000.0, 0.0), 0.1364) *
                         sqrt(fmax(RHO0 / rho_val, 0.1));
        double vt_snow = 1.0 * pow(fmax(rho_val * qr_val * 1000.0, 0.0), 0.1) *
                         sqrt(fmax(RHO0 / rho_val, 0.1));
        double vt_graupel = 4.0 * pow(fmax(rho_val * qr_val * 1000.0, 0.0), 0.12) *
                            sqrt(fmax(RHO0 / rho_val, 0.1));

        double vt;
        if (T >= T_FREEZE) {
            vt = vt_rain;
        } else if (T <= T_ICE_ONLY) {
            vt = 0.6 * vt_snow + 0.4 * vt_graupel;
        } else {
            double frac_warm = (T - T_ICE_ONLY) / (T_FREEZE - T_ICE_ONLY);
            double vt_cold   = 0.6 * vt_snow + 0.4 * vt_graupel;
            vt = frac_warm * vt_rain + (1.0 - frac_warm) * vt_cold;
        }

        vt = fmin(vt, 0.5 * dz / dt);   // CFL limit

        double flux = qr_val * vt * dt / dz;
        flux = fmin(flux, qr_val * 0.9);

        qr[ijk] = (real_t)(qr_val - flux);

        int ijkm = idx3(i, j, k - 1, nx_h, ny_h);
        double dz_below = terrain_following_layer_thickness(terrain_h, eta[k-1], eta[k], ztop);
        dz_below = fmax(dz_below, 1.0);
        double eta_m_below = 0.5 * (eta[k-1] + eta[k]);
        double z_phys_below = terrain_following_height(terrain_h, eta_m_below, ztop);
        double z_flat_below = terrain_following_height(0.0, eta_m_below, ztop);
        double rho_below = rho_base[k > 0 ? k-1 : 0] * exp(-(z_phys_below - z_flat_below) / H_scale);

        qr[ijkm] = (real_t)((double)qr[ijkm] + flux * (rho_val * dz) / fmax(rho_below * dz_below, 1.0));
    }

    // Remove precipitation at ground level
    int ijk0 = idx3(i, j, 0, nx_h, ny_h);
    qr[ijk0] = (real_t)0.0;

    // --- Cloud ice sedimentation (very slow, ~0.3 m/s) ---
    for (int k = nz - 2; k >= 1; k--) {
        int ijk = idx3(i, j, k, nx_h, ny_h);
        double qc_val = (double)qc[ijk];
        if (qc_val <= 1.0e-10) continue;

        // Need terrain-corrected p for temperature
        double eta_m_k = 0.5 * (eta[k] + eta[k+1]);
        double z_phys = terrain_following_height(terrain_h, eta_m_k, ztop);
        double z_flat = terrain_following_height(0.0, eta_m_k, ztop);
        double p_val = p_base[k] * exp(-(z_phys - z_flat) / H_scale);
        double exner = pow(p_val / P0, KAPPA);
        double T     = (double)theta_field[ijk] * exner;

        // Only sediment if cold (ice in qc slot)
        if (T >= T_FREEZE) continue;

        double rho_val = rho_base[k] * exp(-(z_phys - z_flat) / H_scale);
        double dz = terrain_following_layer_thickness(terrain_h, eta[k], eta[k+1], ztop);
        dz = fmax(dz, 1.0);

        double vt_ice = 0.3;  // cloud ice falls very slowly
        vt_ice = fmin(vt_ice, 0.5 * dz / dt);

        double flux = qc_val * vt_ice * dt / dz;
        flux = fmin(flux, qc_val * 0.5);

        qc[ijk] = (real_t)(qc_val - flux);

        int ijkm = idx3(i, j, k - 1, nx_h, ny_h);
        double dz_below = terrain_following_layer_thickness(terrain_h, eta[k-1], eta[k], ztop);
        dz_below = fmax(dz_below, 1.0);
        double eta_m_below = 0.5 * (eta[k-1] + eta[k]);
        double z_phys_below = terrain_following_height(terrain_h, eta_m_below, ztop);
        double z_flat_below = terrain_following_height(0.0, eta_m_below, ztop);
        double rho_below = rho_base[k > 0 ? k-1 : 0] * exp(-(z_phys_below - z_flat_below) / H_scale);

        qc[ijkm] = (real_t)((double)qc[ijkm] + flux * (rho_val * dz) / fmax(rho_below * dz_below, 1.0));
    }
}

// -------------------------------------------------------
// Host driver
// -------------------------------------------------------
void run_microphysics_thompson(StateGPU& state, const GridConfig& grid, double dt) {
    int nx = grid.nx, ny = grid.ny, nz = grid.nz;

    // 3-D kernel: microphysical processes
    dim3 block(8, 8, 4);
    dim3 grid3d((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);

    thompson_kernel<<<grid3d, block>>>(
        state.theta, state.qv, state.qc, state.qr,
        state.rho_base, state.p_base,
        state.terrain, state.eta_m,
        nx, ny, nz, grid.ztop, dt
    );

    // 2-D kernel: sedimentation (column-wise)
    dim3 block2d(16, 16);
    dim3 grid2d((nx + 15) / 16, (ny + 15) / 16);

    thompson_sedimentation_kernel<<<grid2d, block2d>>>(
        state.qr, state.qc,
        state.rho_base, state.p_base, state.eta,
        state.terrain, state.theta,
        nx, ny, nz, grid.ztop, dt
    );
}

} // namespace gpuwm
