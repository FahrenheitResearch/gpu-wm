#pragma once
#include <string>

#include "stability_control.cuh"

// ============================================================
// GPU-WM: Model Configuration
// ============================================================

namespace gpuwm {

struct ModelConfig {
    // Grid
    int nx = 512;
    int ny = 512;
    int nz = 50;
    double dx = 3000.0;     // m
    double dy = 3000.0;
    double ztop = 25000.0;  // m

    // Projection
    double truelat1 = 38.5;
    double truelat2 = 38.5;
    double stand_lon = -97.5;
    double ref_lat = 38.5;
    double ref_lon = -97.5;

    // Time
    double dt = 10.0;       // s
    double t_end = 21600.0; // 6 hours
    double output_interval = 900.0;  // 15 min
    int diag_interval = 100;

    // Physics
    double kdiff = 100.0;       // Horizontal diffusion (m^2/s)
    double cs_smag = 0.18;      // Smagorinsky coefficient
    double z0 = 0.1;            // Surface roughness (m)
    double theta_sfc = 300.0;   // Surface temperature (K)
    double qv_sfc = 0.015;
    double tskin_heat_capacity = 2.0e5;   // J m^-2 K^-1
    double tskin_restore_coeff = 12.0;    // W m^-2 K^-1
    double tskin_anchor_weight = 0.15;    // blend toward diagnosed surface theta
    double tskin_admittance_seam_strength = 0.0; // 0 disables spatial admittance modulation
    double tskin_moisture_gate_strength = 0.0;   // 0 disables heterogeneity-linked surface qv reduction
    double tskin_moisture_memory_timescale = 1800.0; // s, lagged surface moisture-availability response time
    StabilityControlConfig stability;

    // Boundary conditions
    // 0 = periodic, 1 = open/relaxation
    int bc_type = 1;
    int relax_width = 10;       // Relaxation zone width (grid points)

    // Data
    std::string init_data = "";     // GFS/HRRR GRIB2 file for IC
    std::string terrain_file = "";  // Terrain data file
    std::string output_dir = "output";
    std::string output_format = "netcdf"; // "netcdf" or "binary"

    // Test case (0 = real data, 1-3 = idealized)
    int test_case = 0;
};

} // namespace gpuwm
