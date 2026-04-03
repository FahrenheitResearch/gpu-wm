#pragma once

#include <cmath>

#include "constants.cuh"
#include "grid.cuh"

namespace gpuwm {

struct SurfaceLayerState {
    double z1_agl;
    double z2_agl;
    double wspd;
    double theta_sfc_local;
    double qv_sfc_local;
    double theta_v1;
    double theta_v_sfc;
    double ri_sfc;
    double cd_neutral;
    double cd;
    double ch;
};

__host__ __device__ inline double surface_layer_virtual_theta(double theta, double qv) {
    return theta * (1.0 + 0.61 * fmax(qv, 0.0));
}

__host__ __device__ inline double clamp_surface_theta(double value) {
    return fmin(330.0, fmax(250.0, value));
}

__host__ __device__ inline double clamp_surface_qv(double value) {
    return fmin(0.03, fmax(0.0, value));
}

__host__ __device__ inline double blend_surface_theta(double theta_skin,
                                                      double th1,
                                                      double th2,
                                                      double z1,
                                                      double z2) {
    double theta_sfc_local = clamp_surface_theta((theta_skin > 0.0) ? theta_skin : th1);
    double dz12 = fmax(z2 - z1, 1.0);
    double theta_extrap = th1 - z1 * (th2 - th1) / dz12;
    theta_extrap = clamp_surface_theta(theta_extrap);
    return clamp_surface_theta(0.75 * theta_extrap + 0.25 * theta_sfc_local);
}

__host__ __device__ inline double blend_surface_qv(double qv_sfc_prior,
                                                   double qv1,
                                                   double qv2,
                                                   double z1,
                                                   double z2) {
    double qv_sfc_local = clamp_surface_qv(qv_sfc_prior);
    double dz12 = fmax(z2 - z1, 1.0);
    double qv_extrap = qv1 - z1 * (qv2 - qv1) / dz12;
    qv_extrap = clamp_surface_qv(qv_extrap);
    return clamp_surface_qv(0.75 * qv_extrap + 0.25 * qv_sfc_local);
}

__host__ __device__ inline double neutral_surface_exchange_coeff(double z1, double z0) {
    double z0_safe = fmax(z0, 0.01);
    double z1_safe = fmax(z1, z0_safe + 1.0);
    double log_z_z0 = log(z1_safe / z0_safe);
    if (log_z_z0 < 0.5) log_z_z0 = 0.5;
    double coeff = KARMAN / log_z_z0;
    return coeff * coeff;
}

__host__ __device__ inline double surface_layer_stability_factor(double ri_sfc) {
    if (ri_sfc > 0.0) {
        double fac = fmax(1.0 - 5.0 * ri_sfc, 0.0);
        return fac * fac;
    }
    return sqrt(fmax(1.0 - 16.0 * ri_sfc, 0.0));
}

__host__ __device__ inline SurfaceLayerState diagnose_surface_layer_state(
    double theta_skin,
    double qv_sfc_prior,
    double u1,
    double v1,
    double th1,
    double qv1,
    double z1,
    bool has_second_level,
    double th2,
    double qv2,
    double z2,
    double z0
) {
    SurfaceLayerState state{};
    double z0_safe = fmax(z0, 0.01);
    state.z1_agl = fmax(z1, z0_safe + 1.0);
    state.z2_agl = has_second_level ? fmax(z2, state.z1_agl + 1.0) : (state.z1_agl + 1.0);

    state.wspd = sqrt(u1 * u1 + v1 * v1);
    if (state.wspd < 0.1) state.wspd = 0.1;

    state.theta_sfc_local = has_second_level
        ? blend_surface_theta(theta_skin, th1, th2, state.z1_agl, state.z2_agl)
        : clamp_surface_theta((theta_skin > 0.0) ? theta_skin : th1);
    state.qv_sfc_local = has_second_level
        ? blend_surface_qv(qv_sfc_prior, qv1, qv2, state.z1_agl, state.z2_agl)
        : clamp_surface_qv(qv_sfc_prior);

    state.theta_v1 = surface_layer_virtual_theta(th1, qv1);
    state.theta_v_sfc = surface_layer_virtual_theta(state.theta_sfc_local, state.qv_sfc_local);
    state.ri_sfc = (G / fmax(state.theta_v1, 200.0)) *
                   (state.theta_v1 - state.theta_v_sfc) *
                   state.z1_agl / (state.wspd * state.wspd);
    state.ri_sfc = fmax(-10.0, fmin(state.ri_sfc, 10.0));

    state.cd_neutral = neutral_surface_exchange_coeff(state.z1_agl, z0_safe);
    state.cd = state.cd_neutral * surface_layer_stability_factor(state.ri_sfc);
    state.ch = state.cd;
    return state;
}

__host__ __device__ inline double screen_log_fraction(double z_target,
                                                      double z1_agl,
                                                      double z0) {
    double z0_safe = fmax(z0, 0.01);
    double z1_safe = fmax(z1_agl, z0_safe + 1.0);
    double z_target_safe = fmax(z_target, z0_safe + 1.0e-3);
    double denom = log(z1_safe / z0_safe);
    if (denom <= 1.0e-6) return 1.0;
    double frac = log(z_target_safe / z0_safe) / denom;
    return fmin(1.0, fmax(0.0, frac));
}

__host__ __device__ inline double flow_heterogeneity_gate(double thermal_anomaly_k,
                                                          double thermal_range_k,
                                                          double terrain_slope,
                                                          double ri_sfc,
                                                          double wspd) {
    double anomaly_score = fmin(1.0, fmax(0.0, thermal_anomaly_k / 2.0));
    double range_score = fmin(1.0, fmax(0.0, thermal_range_k / 5.0));
    double terrain_score = fmin(1.0, fmax(0.0, (terrain_slope - 0.015) / 0.065));
    double sheltered_score = fmin(1.0, fmax(0.0, (6.0 - wspd) / 4.0));
    double stable_score = fmin(1.0, fmax(0.0, ri_sfc / 0.15));

    double gate = 0.40 * anomaly_score +
                  0.20 * range_score +
                  0.15 * terrain_score +
                  0.15 * sheltered_score +
                  0.10 * stable_score;
    return fmin(1.0, fmax(0.0, gate));
}

__host__ __device__ inline double admittance_seam_factor(double thermal_range_k,
                                                         double terrain_relief_m,
                                                         double terrain_slope,
                                                         double strength) {
    double thermal_score = fmin(1.0, fmax(0.0, thermal_range_k / 4.0));
    double relief_score = fmin(1.0, fmax(0.0, terrain_relief_m / 350.0));
    double slope_score = fmin(1.0, fmax(0.0, (terrain_slope - 0.01) / 0.06));
    double base = 0.50 * thermal_score +
                  0.30 * relief_score +
                  0.20 * slope_score;
    return fmin(1.0, fmax(0.0, strength * base));
}

__host__ __device__ inline double moisture_availability_scale(double activation,
                                                              double strength) {
    double gate = fmin(1.0, fmax(0.0, activation));
    double scale = 1.0 - strength * gate;
    return fmin(1.0, fmax(0.20, scale));
}

__host__ __device__ inline double apply_surface_moisture_scale(double qv_sfc_local,
                                                               double qv_level1,
                                                               double scale) {
    double scale_clamped = fmin(1.0, fmax(0.20, scale));
    return clamp_surface_qv(qv_level1 + (qv_sfc_local - qv_level1) * scale_clamped);
}

__host__ __device__ inline double update_moisture_memory_scale(double current_scale,
                                                               double target_scale,
                                                               double dt,
                                                               double timescale_seconds) {
    double current_clamped = fmin(1.0, fmax(0.20, current_scale));
    double target_clamped = fmin(1.0, fmax(0.20, target_scale));
    double tau = fmax(timescale_seconds, dt);
    double alpha = fmin(1.0, fmax(0.0, dt / tau));
    return fmin(1.0, fmax(0.20, current_clamped + alpha * (target_clamped - current_clamped)));
}

__host__ __device__ inline double apply_surface_moisture_availability(double qv_sfc_local,
                                                                      double qv_level1,
                                                                      double activation,
                                                                      double strength) {
    double scale = moisture_availability_scale(activation, strength);
    return apply_surface_moisture_scale(qv_sfc_local, qv_level1, scale);
}

__host__ __device__ inline double terrain_relief_3x3_field(const real_t* __restrict__ terrain,
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

__host__ __device__ inline double terrain_slope_2d_field(const real_t* __restrict__ terrain,
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

__host__ __device__ inline double skin_theta_range_3x3_field(const real_t* __restrict__ tskin,
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

__host__ __device__ inline double gated_scalar_screen_fraction(double z_target,
                                                               double z1_agl,
                                                               double z0,
                                                               double gate) {
    double frac = screen_log_fraction(z_target, z1_agl, z0);
    return fmin(1.0, fmax(0.0, frac * (1.0 - 0.25 * gate)));
}

__host__ __device__ inline double diagnose_screen_theta(double theta_sfc_local,
                                                        double theta_level1,
                                                        double z_target,
                                                        double z1_agl,
                                                        double z0,
                                                        double gate = 0.0) {
    double frac = gated_scalar_screen_fraction(z_target, z1_agl, z0, gate);
    return clamp_surface_theta(theta_sfc_local + (theta_level1 - theta_sfc_local) * frac);
}

__host__ __device__ inline double diagnose_screen_qv(double qv_sfc_local,
                                                     double qv_level1,
                                                     double z_target,
                                                     double z1_agl,
                                                     double z0,
                                                     double gate = 0.0) {
    double frac = gated_scalar_screen_fraction(z_target, z1_agl, z0, gate);
    return clamp_surface_qv(qv_sfc_local + (qv_level1 - qv_sfc_local) * frac);
}

__host__ __device__ inline double diagnose_screen_wind_component(double wind_level1,
                                                                 double z_target,
                                                                 double z1_agl,
                                                                 double z0) {
    double frac = screen_log_fraction(z_target, z1_agl, z0);
    return wind_level1 * frac;
}

__host__ __device__ inline double surface_saturation_vapor_pressure_liquid(double temperature_k) {
    double temp_c = temperature_k - 273.15;
    return 611.2 * exp(17.67 * temp_c / (temp_c + 243.5));
}

__host__ __device__ inline double surface_saturation_mixing_ratio(double temperature_k,
                                                                  double pressure_pa) {
    double es = surface_saturation_vapor_pressure_liquid(temperature_k);
    double eps = R_D / R_V;
    return eps * es / fmax(pressure_pa - es, 1.0);
}

__host__ __device__ inline double diagnose_screen_relative_humidity(double temperature_k,
                                                                    double pressure_pa,
                                                                    double qv_target) {
    double qvs = surface_saturation_mixing_ratio(temperature_k, pressure_pa);
    double rh = 100.0 * clamp_surface_qv(qv_target) / fmax(qvs, 1.0e-12);
    return fmin(100.0, fmax(0.0, rh));
}

} // namespace gpuwm
