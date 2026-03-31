#pragma once
#include <cmath>
#include "grid.cuh"

// ============================================================
// GPU-WM: Lambert Conformal Conic Map Projection
// Same projection used by HRRR/RAP/NAM
// ============================================================

namespace gpuwm {

struct LambertConformal {
    double truelat1;    // First true latitude (degrees)
    double truelat2;    // Second true latitude (degrees)
    double stand_lon;   // Standard longitude (degrees)
    double ref_lat;     // Reference latitude (degrees)
    double ref_lon;     // Reference longitude (degrees)
    double dx, dy;      // Grid spacing at true latitudes (m)
    int nx, ny;         // Grid dimensions

    // Derived constants (computed once)
    double n;           // Cone constant
    double F;           // Mapping factor
    double rho0;        // Polar distance to reference latitude
    double x0;          // X offset of the reference point
    double y0;          // Y offset of the reference point

    void init() {
        double phi1 = truelat1 * PI / 180.0;
        double phi2 = truelat2 * PI / 180.0;

        if (fabs(truelat1 - truelat2) < 1e-6) {
            n = sin(phi1);
        } else {
            n = log(cos(phi1) / cos(phi2)) /
                log(tan(PI / 4.0 + phi2 / 2.0) / tan(PI / 4.0 + phi1 / 2.0));
        }

        F = cos(phi1) * pow(tan(PI / 4.0 + phi1 / 2.0), n) / n;
        double phi0 = ref_lat * PI / 180.0;
        double lam0 = (ref_lon - stand_lon) * PI / 180.0;
        rho0 = RE * F / pow(tan(PI / 4.0 + phi0 / 2.0), n);
        double theta0 = n * lam0;
        x0 = rho0 * sin(theta0);
        y0 = rho0 * cos(theta0);
    }

    // Grid (i,j) -> lat/lon
    void ij_to_latlon(double i, double j, double& lat, double& lon) const {
        // Grid coords relative to center
        double x = (i - nx / 2.0) * dx;
        double y = (j - ny / 2.0) * dy;

        double x_shift = x + x0;
        double y_shift = y0 - y;
        double rho = copysign(sqrt(x_shift * x_shift + y_shift * y_shift), n);
        double theta = atan2(x_shift, y_shift);

        lat = (2.0 * atan(pow(RE * F / rho, 1.0 / n)) - PI / 2.0) * 180.0 / PI;
        lon = stand_lon + theta / n * 180.0 / PI;
    }

    // lat/lon -> grid (i,j)
    void latlon_to_ij(double lat, double lon, double& i, double& j) const {
        double phi = lat * PI / 180.0;
        double lam = (lon - stand_lon) * PI / 180.0;

        double rho = RE * F / pow(tan(PI / 4.0 + phi / 2.0), n);
        double theta = n * lam;

        double x = rho * sin(theta) - x0;
        double y = y0 - rho * cos(theta);

        i = x / dx + nx / 2.0;
        j = y / dy + ny / 2.0;
    }

    // Map scale factor at a given latitude
    __host__ __device__
    static double map_factor(double lat_deg, double truelat1_deg, double n_val) {
        double phi = lat_deg * PI / 180.0;
        double phi1 = truelat1_deg * PI / 180.0;
        double m = cos(phi1) / cos(phi) *
                   pow(tan(PI / 4.0 + phi / 2.0) / tan(PI / 4.0 + phi1 / 2.0), n_val);
        return m;
    }
};

inline LambertConformal make_lambert_projection(
    int nx, int ny, double dx, double dy,
    double truelat1, double truelat2,
    double stand_lon, double ref_lat, double ref_lon
) {
    LambertConformal lc;
    lc.truelat1 = truelat1;
    lc.truelat2 = truelat2;
    lc.stand_lon = stand_lon;
    lc.ref_lat = ref_lat;
    lc.ref_lon = ref_lon;
    lc.dx = dx;
    lc.dy = dy;
    lc.nx = nx;
    lc.ny = ny;
    lc.init();
    return lc;
}

inline LambertConformal projection_from_grid(const GridConfig& grid) {
    return make_lambert_projection(
        grid.nx, grid.ny, grid.dx, grid.dy,
        grid.truelat1, grid.truelat2,
        grid.stand_lon, grid.ref_lat, grid.ref_lon
    );
}

// HRRR-like CONUS domain configuration
inline LambertConformal hrrr_projection() {
    return make_lambert_projection(
        1799, 1059, 3000.0, 3000.0,
        38.5, 38.5, -97.5, 38.5, -97.5
    );
}

// Smaller test domain (e.g., central US 500x500 at 3km)
inline LambertConformal conus_test_projection(int nx, int ny, double dx) {
    return make_lambert_projection(
        nx, ny, dx, dx,
        38.5, 38.5, -97.5, 38.5, -97.5
    );
}

inline void setup_projection_metrics(GridConfig& grid, const LambertConformal& proj) {
    free_grid_metrics(grid);

    if (grid.ny <= 0) return;

    double* lat_h = new double[grid.ny];
    double* mapfac_h = new double[grid.ny];
    double* coriolis_h = new double[grid.ny];

    double i_center = 0.5 * (grid.nx - 1);
    for (int j = 0; j < grid.ny; j++) {
        double lat = proj.ref_lat;
        double lon = proj.ref_lon;
        proj.ij_to_latlon(i_center, (double)j, lat, lon);

        double mapfac = LambertConformal::map_factor(lat, proj.truelat1, proj.n);
        if (!std::isfinite(mapfac) || mapfac <= 0.0) {
            mapfac = 1.0;
        }

        lat_h[j] = lat;
        mapfac_h[j] = mapfac;
        coriolis_h[j] = 2.0 * OMEGA * sin(lat * PI / 180.0);
    }

    CUDA_CHECK(cudaMalloc(&grid.latitudes, grid.ny * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&grid.mapfac_m, grid.ny * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&grid.coriolis_f, grid.ny * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(grid.latitudes, lat_h, grid.ny * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grid.mapfac_m, mapfac_h, grid.ny * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grid.coriolis_f, coriolis_h, grid.ny * sizeof(double), cudaMemcpyHostToDevice));

    delete[] lat_h;
    delete[] mapfac_h;
    delete[] coriolis_h;
}

} // namespace gpuwm
