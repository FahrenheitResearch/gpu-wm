#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include "constants.cuh"

// ============================================================
// GPU-WM: Grid definition
// Arakawa-C staggered grid with terrain-following sigma coordinates
// ============================================================

namespace gpuwm {

// Check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct GridConfig {
    // Domain dimensions (number of mass points)
    int nx, ny, nz;

    // Grid spacing
    double dx, dy;           // Horizontal spacing (m)

    // Domain physical extent
    double lx, ly;           // Domain size (m)
    double ztop;             // Model top height (m)

    // Latitude/longitude of domain center
    double clat, clon;       // Center lat/lon (degrees)

    // Lambert conformal metadata for regional runs
    double truelat1 = 38.5;
    double truelat2 = 38.5;
    double stand_lon = -97.5;
    double ref_lat = 38.5;
    double ref_lon = -97.5;

    // Valid/reference times for real-data initial conditions, UTC seconds since Unix epoch.
    int64_t init_valid_time_unix = -1;
    int64_t init_reference_time_unix = -1;
    int32_t init_forecast_hour = 0;

    // Vertical coordinate parameters
    // eta levels from 0 (surface) to 1 (model top)
    double* eta = nullptr;             // host-side eta values at w-levels [nz+1]
    double* eta_m = nullptr;           // host-side eta values at mass levels [nz]

    // Row-wise Lambert metrics on device.
    // These let the solver vary Coriolis and horizontal metric scale with y.
    double* latitudes = nullptr;       // latitude at row centers [ny]
    double* mapfac_m = nullptr;        // map factor at row centers [ny]
    double* coriolis_f = nullptr;      // Coriolis parameter by row [ny]
};

// State variables on GPU - all prognostic variables
// Using perturbation form: variable = base_state + perturbation
// 3D fields use real_t (float32 by default) for memory efficiency
// Base state arrays stay double (small, need precision)
struct StateGPU {
    // Prognostic variables (3D fields: nx * ny * nz) - real_t for memory savings
    real_t* u;               // x-wind component (m/s) - staggered in x
    real_t* v;               // y-wind component (m/s) - staggered in y
    real_t* w;               // contravariant vertical transport velocity (m/s)
    real_t* theta;           // potential temperature (K)
    real_t* qv;              // water vapor mixing ratio (kg/kg)
    real_t* qc;              // cloud water mixing ratio (kg/kg)
    real_t* qr;              // rain water mixing ratio (kg/kg)

    // Diagnostic variables (3D) - real_t
    real_t* p;               // pressure perturbation (Pa)
    real_t* rho;             // solver reference density field (kg/m^3)
    real_t* phi;             // geopotential / scratch (m^2/s^2)

    // Base state (1D, function of z only) - stays double (small arrays, need precision)
    double* theta_base;      // base state potential temperature [nz]
    double* p_base;          // base state pressure [nz]
    double* rho_base;        // base state density [nz]
    double* qv_base;         // base state water vapor mixing ratio [nz]
    double* z_levels;        // height of each level [nz]
    double* z_w_levels;      // height of each w-level [nz+1]
    double* eta;             // device copy of eta values at w-levels [nz+1]
    double* eta_m;           // device copy of eta values at mass levels [nz]

    // Terrain height (2D) - real_t
    real_t* terrain;         // terrain height (m) [nx * ny]
    real_t* tskin;           // prognostic surface skin temperature surrogate (K) [nx * ny]
    real_t* moistmem;        // lagged surface moisture-availability scale [0..1] [nx * ny]

    // Tendency arrays (for RK3 time integration) - real_t
    real_t* u_tend;
    real_t* v_tend;
    real_t* w_tend;
    real_t* theta_tend;
    real_t* qv_tend;
    real_t* qc_tend;
    real_t* qr_tend;
};

// Allocate state on GPU
inline void allocate_state(StateGPU& state, const GridConfig& grid) {
    size_t n2d = (size_t)grid.nx * grid.ny;

    // 3D prognostic fields (include halos: +2 in each horizontal direction)
    int nx_h = grid.nx + 4;  // 2 halo cells on each side
    int ny_h = grid.ny + 4;
    size_t n3d_h = (size_t)nx_h * ny_h * grid.nz;
    size_t n3d_w = (size_t)nx_h * ny_h * (grid.nz + 1);

    // 3D fields use real_t (float32 by default, float64 with USE_DOUBLE)
    CUDA_CHECK(cudaMalloc(&state.u,     n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.v,     n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.w,     n3d_w * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.theta, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.qv,    n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.qc,    n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.qr,    n3d_h * sizeof(real_t)));

    CUDA_CHECK(cudaMalloc(&state.p,     n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.rho,   n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.phi,   n3d_h * sizeof(real_t)));

    // Tendency arrays - real_t
    CUDA_CHECK(cudaMalloc(&state.u_tend,     n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.v_tend,     n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.w_tend,     n3d_w * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.theta_tend, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.qv_tend,    n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.qc_tend,    n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.qr_tend,    n3d_h * sizeof(real_t)));

    // Base state (1D) - stays double
    CUDA_CHECK(cudaMalloc(&state.theta_base, grid.nz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.p_base,     grid.nz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.rho_base,   grid.nz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.qv_base,    grid.nz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.z_levels,   grid.nz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.z_w_levels, (grid.nz + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.eta,       (grid.nz + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.eta_m,      grid.nz * sizeof(double)));

    if (grid.eta && grid.eta_m) {
        CUDA_CHECK(cudaMemcpy(state.eta,   grid.eta,   (grid.nz + 1) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.eta_m,  grid.eta_m, grid.nz * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        double* eta_h = new double[grid.nz + 1];
        double* eta_m_h = new double[grid.nz];
        for (int k = 0; k <= grid.nz; ++k) {
            eta_h[k] = (double)k / (double)grid.nz;
        }
        for (int k = 0; k < grid.nz; ++k) {
            eta_m_h[k] = 0.5 * (eta_h[k] + eta_h[k + 1]);
        }
        CUDA_CHECK(cudaMemcpy(state.eta,   eta_h,   (grid.nz + 1) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.eta_m,  eta_m_h, grid.nz * sizeof(double), cudaMemcpyHostToDevice));
        delete[] eta_h;
        delete[] eta_m_h;
    }

    // Terrain (2D) - real_t
    CUDA_CHECK(cudaMalloc(&state.terrain, n2d * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.tskin,   n2d * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&state.moistmem, n2d * sizeof(real_t)));

    // Zero everything
    CUDA_CHECK(cudaMemset(state.u,     0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.v,     0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.w,     0, n3d_w * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.theta, 0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.qv,    0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.qc,    0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.qr,    0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.p,     0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.rho,   0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.phi,   0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.u_tend,     0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.v_tend,     0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.w_tend,     0, n3d_w * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.theta_tend, 0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.qv_tend,    0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.qc_tend,    0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.qr_tend,    0, n3d_h * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.terrain, 0, n2d * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.tskin,   0, n2d * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(state.moistmem, 0, n2d * sizeof(real_t)));
}

inline void free_grid_metrics(GridConfig& grid) {
    if (grid.latitudes) {
        cudaFree(grid.latitudes);
        grid.latitudes = nullptr;
    }
    if (grid.mapfac_m) {
        cudaFree(grid.mapfac_m);
        grid.mapfac_m = nullptr;
    }
    if (grid.coriolis_f) {
        cudaFree(grid.coriolis_f);
        grid.coriolis_f = nullptr;
    }
}

inline void free_state(StateGPU& state) {
    cudaFree(state.u);     cudaFree(state.v);     cudaFree(state.w);
    cudaFree(state.theta);  cudaFree(state.qv);    cudaFree(state.qc);
    cudaFree(state.qr);     cudaFree(state.p);     cudaFree(state.rho);
    cudaFree(state.phi);
    cudaFree(state.u_tend);     cudaFree(state.v_tend);     cudaFree(state.w_tend);
    cudaFree(state.theta_tend); cudaFree(state.qv_tend);    cudaFree(state.qc_tend);
    cudaFree(state.qr_tend);
    cudaFree(state.theta_base); cudaFree(state.p_base);  cudaFree(state.rho_base);
    cudaFree(state.qv_base);
    cudaFree(state.z_levels);   cudaFree(state.z_w_levels);
    cudaFree(state.eta);        cudaFree(state.eta_m);
    cudaFree(state.terrain);
    cudaFree(state.tskin);
    cudaFree(state.moistmem);
}

// Index helpers for 3D arrays with halo
__host__ __device__ inline int idx3(int i, int j, int k, int nx_h, int ny_h) {
    return (k * ny_h + (j + 2)) * nx_h + (i + 2);  // +2 for halo offset
}

__host__ __device__ inline int idx3w(int i, int j, int k, int nx_h, int ny_h) {
    return (k * ny_h + (j + 2)) * nx_h + (i + 2);  // identical to idx3, for w-staggered fields
}

__host__ __device__ inline int idx2(int i, int j, int nx) {
    return j * nx + i;
}

// Terrain-following helpers for kernels that need local column geometry.
__host__ __device__ inline double terrain_following_height(double terrain, double eta, double ztop) {
    if (eta < 0.0) eta = 0.0;
    if (eta > 1.0) eta = 1.0;
    double column_depth = ztop - terrain;
    if (column_depth < 1.0) column_depth = 1.0;
    return terrain + eta * column_depth;
}

__host__ __device__ inline double terrain_following_layer_thickness(
    double terrain, double eta_lower, double eta_upper, double ztop
) {
    return terrain_following_height(terrain, eta_upper, ztop) -
           terrain_following_height(terrain, eta_lower, ztop);
}

} // namespace gpuwm
