#pragma once
#include <cmath>
#include "grid.cuh"

// ============================================================
// GPU-WM: Microphysics scheme declarations
// ============================================================

namespace gpuwm {

__host__ __device__ inline double clamp01(double x) {
    return fmax(0.0, fmin(1.0, x));
}

__host__ __device__ inline double exner_from_pressure(double p) {
    return pow(fmax(p, 1.0) / P0, KAPPA);
}

__host__ __device__ inline double temperature_from_theta(double theta, double p) {
    return theta * exner_from_pressure(p);
}

__host__ __device__ inline double saturation_vapor_pressure_liquid(double T) {
    T = fmax(fmin(T, 350.0), 180.0);
    return 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65));
}

__host__ __device__ inline double saturation_vapor_pressure_ice(double T) {
    T = fmax(fmin(T, 273.16), 100.0);
    return 611.2 * exp(21.8745584 * (T - 273.16) / (T - 7.66));
}

__host__ __device__ inline double saturation_mixing_ratio_liquid(double p, double T) {
    double es = saturation_vapor_pressure_liquid(T);
    if (es >= p * 0.5) return 0.1;
    return 0.622 * es / fmax(p - es, 1.0);
}

__host__ __device__ inline double saturation_mixing_ratio_ice(double p, double T) {
    double es = saturation_vapor_pressure_ice(T);
    if (es >= p * 0.5) return 0.1;
    return 0.622 * es / fmax(p - es, 1.0);
}

__host__ __device__ inline double saturation_mixing_ratio_blend(double p, double T) {
    constexpr double T_FREEZE = 273.15;
    constexpr double T_ICE_ONLY = 253.15;
    double qv_liq = saturation_mixing_ratio_liquid(p, T);
    double qv_ice = saturation_mixing_ratio_ice(p, T);
    if (T >= T_FREEZE) return qv_liq;
    if (T <= T_ICE_ONLY) return qv_ice;
    double frac_liq = clamp01((T - T_ICE_ONLY) / (T_FREEZE - T_ICE_ONLY));
    return frac_liq * qv_liq + (1.0 - frac_liq) * qv_ice;
}

// Kessler warm-rain microphysics (single-moment)
// Handles: saturation adjustment, autoconversion, accretion,
//          rain evaporation, sedimentation
void run_microphysics(StateGPU& state, const GridConfig& grid, double dt);

// Simplified Thompson microphysics (mixed-phase)
// Adds ice-phase processes: vapor deposition, Bergeron process,
// ice nucleation, snow autoconversion, riming, melting, freezing,
// and multi-species sedimentation (rain, snow, graupel)
void run_microphysics_thompson(StateGPU& state, const GridConfig& grid, double dt);

} // namespace gpuwm
