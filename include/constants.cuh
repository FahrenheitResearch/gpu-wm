#pragma once

// ============================================================
// GPU-WM: GPU Weather Model
// Physical and numerical constants
// ============================================================

// Precision control: use float for memory efficiency, double for accuracy
// Build with -DUSE_DOUBLE to switch all 3D fields to float64
#ifdef USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

namespace gpuwm {

// Fundamental physical constants
constexpr double R_D     = 287.04;      // Gas constant for dry air (J/kg/K)
constexpr double R_V     = 461.6;       // Gas constant for water vapor (J/kg/K)
constexpr double CP_D    = 1004.5;      // Specific heat at constant pressure, dry air (J/kg/K)
constexpr double CV_D    = 717.46;      // Specific heat at constant volume, dry air (J/kg/K)
constexpr double CP_V    = 1846.0;      // Specific heat of water vapor (J/kg/K)
constexpr double GAMMA   = CP_D / CV_D; // Ratio of specific heats
constexpr double KAPPA   = R_D / CP_D;  // Poisson constant
constexpr double G       = 9.81;        // Gravitational acceleration (m/s^2)
constexpr double P0      = 100000.0;    // Reference pressure (Pa)
constexpr double T0      = 300.0;       // Reference temperature (K)
constexpr double RHO0    = P0 / (R_D * T0); // Reference density
constexpr double PI      = 3.14159265358979323846;
constexpr double OMEGA   = 7.292e-5;    // Earth's angular velocity (rad/s)
constexpr double RE      = 6.371e6;     // Earth's radius (m)
constexpr double LV      = 2.501e6;     // Latent heat of vaporization (J/kg)
constexpr double LF      = 3.337e5;     // Latent heat of fusion (J/kg)
constexpr double LS      = LV + LF;     // Latent heat of sublimation (J/kg)
constexpr double STEFAN_BOLTZMANN = 5.67e-8; // Stefan-Boltzmann constant
constexpr double KARMAN  = 0.4;         // Von Karman constant

// Reference atmosphere (US Standard Atmosphere lapse rate)
constexpr double LAPSE_RATE = 0.0065;   // K/m in troposphere

// Numerical limits
constexpr double SMALL   = 1.0e-12;
constexpr double BIG     = 1.0e30;

} // namespace gpuwm
