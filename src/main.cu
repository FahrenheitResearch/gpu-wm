// ============================================================
// GPU-WM: GPU Weather Model
// All-GPU non-hydrostatic atmospheric model
//
// Operational CONUS-scale model targeting 1-3km resolution
// on NVIDIA GPUs. Equivalent capability to WRF/HRRR.
//
// Dynamics: Split-explicit compressible equations (no FFT needed)
// Physics: Kessler microphysics, Smagorinsky PBL, simplified radiation
// Grid: Lambert Conformal (HRRR-compatible), open lateral BCs
// I/O: NetCDF output, GFS/HRRR GRIB2 initialization
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <chrono>
#include <string>
#include <vector>

#include "../include/constants.cuh"
#include "../include/grid.cuh"
#include "../include/projection.cuh"
#include "../include/config.cuh"
#include "../include/sponge.cuh"

namespace gpuwm {

// Forward declarations
void initialize_model(StateGPU& state, GridConfig& grid, int test_case);
void rk3_step(StateGPU& state, StateGPU& state_old, StateGPU& state_init,
              const GridConfig& grid, double dt, double kdiff, int rk_stage,
              bool use_open_bc, const StabilityControlConfig& stability_cfg,
              int relax_width);
void apply_boundary_conditions(StateGPU& state, const GridConfig& grid);
void apply_open_boundaries(StateGPU& state, StateGPU& state_init,
                           const GridConfig& grid, int relax_width);
void refresh_open_halos(StateGPU& state, const GridConfig& grid);
void convert_w_to_contravariant(StateGPU& state, const GridConfig& grid);
void initialize_w_from_continuity(StateGPU& state, const GridConfig& grid, const char* label);
void run_microphysics(StateGPU& state, const GridConfig& grid, double dt);
void run_microphysics_thompson(StateGPU& state, const GridConfig& grid, double dt);
void run_radiation(StateGPU& state, const GridConfig& grid,
                   double solar_zenith_cos, double solar_constant);
void run_pbl(StateGPU& state, const GridConfig& grid,
             double z0, double qv_sfc, double cs, double dt);
void initialize_tskin_from_surface_layer(StateGPU& state, const GridConfig& grid,
                                         double theta_sfc);
void update_tskin_slab(StateGPU& state, const GridConfig& grid,
                       double z0, double theta_sfc, double qv_sfc, double dt);
void print_diagnostics(const StateGPU& state, const GridConfig& grid,
                       double time, int step,
                       const StabilityControlConfig& stability_cfg);
void write_output(const StateGPU& state, const GridConfig& grid,
                  double time, int output_num);
void write_netcdf(const StateGPU& state, const GridConfig& grid,
                  const LambertConformal& proj,
                  double time, int output_num);
bool init_from_gfs(StateGPU& state, const GridConfig& grid,
                   const LambertConformal& proj, const char* gfs_file);
bool load_gfs_binary(StateGPU& state, GridConfig& grid, const char* filename);
bool download_gfs(const char* date, const char* cycle, const char* output_file);

// Copy state for RK3
__global__ void copy_field_kernel(real_t* __restrict__ dst,
                                   const real_t* __restrict__ src,
                                   int n_total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    dst[idx] = src[idx];
}

void copy_state(StateGPU& dst, const StateGPU& src, const GridConfig& grid) {
    int nx_h = grid.nx + 4;
    int ny_h = grid.ny + 4;
    int n_total = nx_h * ny_h * grid.nz;
    int n_total_w = nx_h * ny_h * (grid.nz + 1);
    int n2d = grid.nx * grid.ny;
    int block = 256;
    int grid1d = (n_total + 255) / 256;
    int grid1d_w = (n_total_w + 255) / 256;
    int grid2d = (n2d + 255) / 256;

    copy_field_kernel<<<grid1d, block>>>(dst.u, src.u, n_total);
    copy_field_kernel<<<grid1d, block>>>(dst.v, src.v, n_total);
    copy_field_kernel<<<grid1d_w, block>>>(dst.w, src.w, n_total_w);
    copy_field_kernel<<<grid1d, block>>>(dst.theta, src.theta, n_total);
    copy_field_kernel<<<grid1d, block>>>(dst.qv, src.qv, n_total);
    copy_field_kernel<<<grid1d, block>>>(dst.qc, src.qc, n_total);
    copy_field_kernel<<<grid1d, block>>>(dst.qr, src.qr, n_total);
    copy_field_kernel<<<grid1d, block>>>(dst.p, src.p, n_total);
    copy_field_kernel<<<grid2d, block>>>(dst.tskin, src.tskin, n2d);
}

__global__ void blend_field_kernel(
    real_t* __restrict__ dst,
    const real_t* __restrict__ src_a,
    const real_t* __restrict__ src_b,
    double alpha,
    int n_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    double a = (double)src_a[idx];
    double b = (double)src_b[idx];
    dst[idx] = (real_t)((1.0 - alpha) * a + alpha * b);
}

__global__ void blend_double_kernel(
    double* __restrict__ dst,
    const double* __restrict__ src_a,
    const double* __restrict__ src_b,
    double alpha,
    int n_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    dst[idx] = (1.0 - alpha) * src_a[idx] + alpha * src_b[idx];
}

void copy_base_state(StateGPU& dst, const StateGPU& src, const GridConfig& grid) {
    CUDA_CHECK(cudaMemcpy(dst.theta_base, src.theta_base, grid.nz * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst.p_base, src.p_base, grid.nz * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst.rho_base, src.rho_base, grid.nz * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst.qv_base, src.qv_base, grid.nz * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst.z_levels, src.z_levels, grid.nz * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst.z_w_levels, src.z_w_levels, (grid.nz + 1) * sizeof(double), cudaMemcpyDeviceToDevice));
}

void blend_boundary_state(
    StateGPU& dst,
    const StateGPU& src_a,
    const StateGPU& src_b,
    const GridConfig& grid,
    double alpha
) {
    if (alpha <= 0.0) {
        copy_state(dst, src_a, grid);
        copy_base_state(dst, src_a, grid);
        return;
    }
    if (alpha >= 1.0) {
        copy_state(dst, src_b, grid);
        copy_base_state(dst, src_b, grid);
        return;
    }

    int nx_h = grid.nx + 4;
    int ny_h = grid.ny + 4;
    int n_total = nx_h * ny_h * grid.nz;
    int n_total_w = nx_h * ny_h * (grid.nz + 1);
    int n2d = grid.nx * grid.ny;
    int block = 256;
    int grid1d = (n_total + block - 1) / block;
    int grid1d_w = (n_total_w + block - 1) / block;
    int grid2d = (n2d + block - 1) / block;

    blend_field_kernel<<<grid1d, block>>>(dst.u, src_a.u, src_b.u, alpha, n_total);
    blend_field_kernel<<<grid1d, block>>>(dst.v, src_a.v, src_b.v, alpha, n_total);
    blend_field_kernel<<<grid1d_w, block>>>(dst.w, src_a.w, src_b.w, alpha, n_total_w);
    blend_field_kernel<<<grid1d, block>>>(dst.theta, src_a.theta, src_b.theta, alpha, n_total);
    blend_field_kernel<<<grid1d, block>>>(dst.qv, src_a.qv, src_b.qv, alpha, n_total);
    blend_field_kernel<<<grid1d, block>>>(dst.qc, src_a.qc, src_b.qc, alpha, n_total);
    blend_field_kernel<<<grid1d, block>>>(dst.qr, src_a.qr, src_b.qr, alpha, n_total);
    blend_field_kernel<<<grid1d, block>>>(dst.p, src_a.p, src_b.p, alpha, n_total);
    blend_field_kernel<<<grid2d, block>>>(dst.tskin, src_a.tskin, src_b.tskin, alpha, n2d);

    int nz = grid.nz;
    int grid1d_z = (nz + block - 1) / block;
    blend_double_kernel<<<grid1d_z, block>>>(dst.theta_base, src_a.theta_base, src_b.theta_base, alpha, nz);
    blend_double_kernel<<<grid1d_z, block>>>(dst.p_base, src_a.p_base, src_b.p_base, alpha, nz);
    blend_double_kernel<<<grid1d_z, block>>>(dst.rho_base, src_a.rho_base, src_b.rho_base, alpha, nz);
    blend_double_kernel<<<grid1d_z, block>>>(dst.qv_base, src_a.qv_base, src_b.qv_base, alpha, nz);
    CUDA_CHECK(cudaMemcpy(dst.z_levels, src_a.z_levels, nz * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst.z_w_levels, src_a.z_w_levels, (nz + 1) * sizeof(double), cudaMemcpyDeviceToDevice));
}

bool boundary_grids_compatible(const GridConfig& a, const GridConfig& b) {
    auto nearly_equal = [](double lhs, double rhs) {
        return fabs(lhs - rhs) <= 1.0e-6 * fmax(1.0, fmax(fabs(lhs), fabs(rhs)));
    };
    return a.nx == b.nx &&
           a.ny == b.ny &&
           a.nz == b.nz &&
           nearly_equal(a.dx, b.dx) &&
           nearly_equal(a.dy, b.dy) &&
           nearly_equal(a.ztop, b.ztop) &&
           nearly_equal(a.truelat1, b.truelat1) &&
           nearly_equal(a.truelat2, b.truelat2) &&
           nearly_equal(a.stand_lon, b.stand_lon) &&
           nearly_equal(a.ref_lat, b.ref_lat) &&
           nearly_equal(a.ref_lon, b.ref_lon);
}

struct BoundarySpec {
    std::string path;
    double time_seconds = 0.0;
    bool has_explicit_time = false;
};

struct BoundaryNode {
    StateGPU state{};
    GridConfig grid{};
    std::string path;
    double time_seconds = 0.0;
};

bool parse_boundary_spec(const char* token, BoundarySpec& spec) {
    if (!token || token[0] == '\0') {
        return false;
    }

    spec = BoundarySpec{};
    std::string value(token);
    std::size_t at = value.rfind('@');
    if (at == std::string::npos) {
        spec.path = value;
        return true;
    }

    std::string maybe_time = value.substr(at + 1);
    char* endptr = nullptr;
    double parsed_time = strtod(maybe_time.c_str(), &endptr);
    if (endptr && *endptr == '\0') {
        spec.path = value.substr(0, at);
        spec.time_seconds = parsed_time;
        spec.has_explicit_time = true;
        return !spec.path.empty();
    }

    spec.path = value;
    return true;
}

void reset_boundary_grid(GridConfig& grid, const GridConfig& template_grid) {
    grid = template_grid;
    grid.eta = nullptr;
    grid.eta_m = nullptr;
    grid.latitudes = nullptr;
    grid.mapfac_m = nullptr;
    grid.coriolis_f = nullptr;
}

void free_boundary_grid(GridConfig& grid) {
    delete[] grid.eta;
    delete[] grid.eta_m;
    grid.eta = nullptr;
    grid.eta_m = nullptr;
}

double infer_boundary_time_seconds(const GridConfig& initial_grid,
                                   const GridConfig& boundary_grid,
                                   const BoundarySpec& spec,
                                   double fallback_seconds) {
    if (spec.has_explicit_time) {
        return spec.time_seconds;
    }
    if (initial_grid.init_valid_time_unix >= 0 && boundary_grid.init_valid_time_unix >= 0) {
        return static_cast<double>(boundary_grid.init_valid_time_unix - initial_grid.init_valid_time_unix);
    }
    return fallback_seconds;
}

} // namespace gpuwm

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    using namespace gpuwm;

    printf("============================================\n");
    printf("  GPU-WM: GPU Weather Model v0.2\n");
    printf("  All-GPU Non-Hydrostatic Atmospheric Model\n");
    printf("============================================\n\n");

    // GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%.1f GB, SM %d.%d)\n\n",
           prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);

    // Configuration
    ModelConfig cfg;
    int use_netcdf = 0;
    int use_open_bc = 0;
    int use_thompson = 0;
    char gfs_file[512] = "";
    char boundary_next_file[512] = "";
    double boundary_interval = 0.0;
    std::vector<BoundarySpec> boundary_specs;

    // Parse command line
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nx") == 0 && i+1 < argc) cfg.nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ny") == 0 && i+1 < argc) cfg.ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nz") == 0 && i+1 < argc) cfg.nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dx") == 0 && i+1 < argc) { cfg.dx = atof(argv[++i]); cfg.dy = cfg.dx; }
        else if (strcmp(argv[i], "--truelat1") == 0 && i+1 < argc) cfg.truelat1 = atof(argv[++i]);
        else if (strcmp(argv[i], "--truelat2") == 0 && i+1 < argc) cfg.truelat2 = atof(argv[++i]);
        else if (strcmp(argv[i], "--stand-lon") == 0 && i+1 < argc) cfg.stand_lon = atof(argv[++i]);
        else if (strcmp(argv[i], "--ref-lat") == 0 && i+1 < argc) cfg.ref_lat = atof(argv[++i]);
        else if (strcmp(argv[i], "--ref-lon") == 0 && i+1 < argc) cfg.ref_lon = atof(argv[++i]);
        else if (strcmp(argv[i], "--ztop") == 0 && i+1 < argc) cfg.ztop = atof(argv[++i]);
        else if (strcmp(argv[i], "--dt") == 0 && i+1 < argc) cfg.dt = atof(argv[++i]);
        else if (strcmp(argv[i], "--tend") == 0 && i+1 < argc) cfg.t_end = atof(argv[++i]);
        else if (strcmp(argv[i], "--test") == 0 && i+1 < argc) cfg.test_case = atoi(argv[++i]);
        else if (strcmp(argv[i], "--output-interval") == 0 && i+1 < argc) cfg.output_interval = atof(argv[++i]);
        else if (strcmp(argv[i], "--diag-interval") == 0 && i+1 < argc) cfg.diag_interval = atoi(argv[++i]);
        else if (strcmp(argv[i], "--netcdf") == 0) use_netcdf = 1;
        else if (strcmp(argv[i], "--open-bc") == 0) use_open_bc = 1;
        else if (strcmp(argv[i], "--thompson") == 0) use_thompson = 1;
        else if (strcmp(argv[i], "--gfs") == 0 && i+1 < argc) strncpy(gfs_file, argv[++i], sizeof(gfs_file)-1);
        else if (strcmp(argv[i], "--boundary-state") == 0 && i+1 < argc) {
            BoundarySpec spec;
            if (!parse_boundary_spec(argv[++i], spec)) {
                fprintf(stderr, "FATAL: Invalid --boundary-state spec\n");
                return 1;
            }
            boundary_specs.push_back(spec);
        }
        else if (strcmp(argv[i], "--boundary-next") == 0 && i+1 < argc) strncpy(boundary_next_file, argv[++i], sizeof(boundary_next_file)-1);
        else if (strcmp(argv[i], "--boundary-interval") == 0 && i+1 < argc) boundary_interval = atof(argv[++i]);
        else if (strcmp(argv[i], "--kdiff") == 0 && i+1 < argc) cfg.kdiff = atof(argv[++i]);
        else if (strcmp(argv[i], "--no-adaptive-stab") == 0) cfg.stability.enabled = 0;
        else if (strcmp(argv[i], "--w-damp") == 0) cfg.stability.w_cfl_damping = 1;
        else if (strcmp(argv[i], "--w-damp-alpha") == 0 && i+1 < argc) cfg.stability.w_damping_alpha = atof(argv[++i]);
        else if (strcmp(argv[i], "--w-damp-beta") == 0 && i+1 < argc) cfg.stability.w_damping_beta = atof(argv[++i]);
        else if (strcmp(argv[i], "--w-transport-blend") == 0 && i+1 < argc) {
            cfg.stability.w_transport_blend = atof(argv[++i]);
            cfg.stability.w_transport_blend = std::max(0.0, std::min(1.0, cfg.stability.w_transport_blend));
        }
        else if (strcmp(argv[i], "--w-transport-diagnostics") == 0) cfg.stability.w_transport_diagnostics = 1;
        else if (strcmp(argv[i], "--semiimplicit-pw") == 0) cfg.stability.pw_column_implicit = 1;
        else if (strcmp(argv[i], "--semiimplicit-pw-diagnostics") == 0) {
            cfg.stability.pw_column_implicit = 1;
            cfg.stability.pw_column_diagnostics = 1;
        }
        else if (strcmp(argv[i], "--hrrr") == 0) {
            // HRRR-like CONUS domain
            cfg.nx = 1799; cfg.ny = 1059; cfg.nz = 50;
            cfg.dx = 3000.0; cfg.dy = 3000.0;
            cfg.ztop = 25000.0; cfg.dt = 15.0;
            use_open_bc = 1; use_netcdf = 1;
        }
        else if (strcmp(argv[i], "--conus-test") == 0) {
            // Smaller CONUS test domain
            cfg.nx = 512; cfg.ny = 512; cfg.nz = 50;
            cfg.dx = 3000.0; cfg.dy = 3000.0;
            cfg.ztop = 25000.0; cfg.dt = 10.0;
            cfg.test_case = 3;
            use_open_bc = 1; use_netcdf = 1;
        }
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: gpu-wm [options]\n\n");
            printf("Grid:\n");
            printf("  --nx N              Grid points in x (default: 512)\n");
            printf("  --ny N              Grid points in y (default: 512)\n");
            printf("  --nz N              Grid points in z (default: 50)\n");
            printf("  --dx D              Grid spacing (m, default: 3000)\n");
            printf("  --truelat1 LAT      Lambert true latitude 1 (default: 38.5)\n");
            printf("  --truelat2 LAT      Lambert true latitude 2 (default: 38.5)\n");
            printf("  --stand-lon LON     Lambert standard longitude (default: -97.5)\n");
            printf("  --ref-lat LAT       Domain center latitude (default: 38.5)\n");
            printf("  --ref-lon LON       Domain center longitude (default: -97.5)\n");
            printf("  --ztop H            Model top (m, default: 25000)\n\n");
            printf("Time:\n");
            printf("  --dt D              Time step (s, default: 10)\n");
            printf("  --tend T            End time (s, default: 21600)\n");
            printf("  --output-interval S Output interval (s, default: 900)\n\n");
            printf("  --diag-interval N   Print runtime diagnostics every N steps (default: 100)\n\n");
            printf("Physics:\n");
            printf("  --kdiff K           Horizontal diffusion (m^2/s, default: 100)\n");
            printf("  --thompson          Use Thompson mixed-phase microphysics (default: Kessler)\n");
            printf("  --no-adaptive-stab  Disable norm-based adaptive stabilization\n");
            printf("  --w-damp            Enable WRF-style vertical CFL damping on interface w\n");
            printf("  --w-damp-alpha A    w damping strength (m/s/s, default: 0.3)\n");
            printf("  --w-damp-beta B     w damping activation CFL (default: 1.0)\n");
            printf("  --w-transport-blend B  Blend legacy and ERF-style w transport (0..1, default: 1.0)\n");
            printf("  --w-transport-diagnostics  Print interval diagnostics for old/new w transport terms\n");
            printf("  --semiimplicit-pw   Replace the explicit fast vertical p-w pair with a column implicit solve\n");
            printf("  --semiimplicit-pw-diagnostics  Enable the semi-implicit p-w path and extra diagnostics\n");
            printf("  --test N            Idealized test: 1=bubble 2=density-current 3=convection 4=free-stream-terrain\n\n");
            printf("Operational:\n");
            printf("  --hrrr              Full HRRR CONUS domain (1799x1059 @ 3km)\n");
            printf("  --conus-test        Test CONUS domain (512x512 @ 3km)\n");
            printf("  --gfs FILE          Initialize from GFS GRIB2 file\n");
            printf("  --boundary-state S  Parent boundary snapshot: FILE or FILE@SECONDS (repeatable)\n");
            printf("  --boundary-next F   Next parent state for time-varying boundaries\n");
            printf("  --boundary-interval S  Seconds spanned by start->next boundary interpolation\n");
            printf("  --open-bc           Use open lateral boundary conditions\n");
            printf("  --netcdf            Output in NetCDF format\n");
            return 0;
        }
    }

    // Setup grid
    GridConfig grid;
    grid.nx = cfg.nx; grid.ny = cfg.ny; grid.nz = cfg.nz;
    grid.dx = cfg.dx; grid.dy = cfg.dy;
    grid.lx = grid.nx * grid.dx;
    grid.ly = grid.ny * grid.dy;
    grid.ztop = cfg.ztop;
    grid.clat = cfg.ref_lat;
    grid.clon = cfg.ref_lon;
    grid.truelat1 = cfg.truelat1;
    grid.truelat2 = cfg.truelat2;
    grid.stand_lon = cfg.stand_lon;
    grid.ref_lat = cfg.ref_lat;
    grid.ref_lon = cfg.ref_lon;

    double dt = cfg.dt;
    double t_end = cfg.t_end;
    if (boundary_next_file[0] != '\0') {
        BoundarySpec spec;
        spec.path = boundary_next_file;
        if (boundary_interval > 0.0) {
            spec.time_seconds = boundary_interval;
            spec.has_explicit_time = true;
        }
        boundary_specs.push_back(spec);
    }

    if (!boundary_specs.empty() && boundary_interval <= 0.0 && boundary_next_file[0] != '\0') {
        boundary_interval = t_end;
    }

    // Setup projection
    LambertConformal proj = projection_from_grid(grid);

    auto print_case_summary = [&](const GridConfig& active_grid) {
        // 3D fields use real_t (17 fields per state: 7 prognostic + 3 diagnostic + 7 tendency)
        // Plus base state (5 * nz * double) and terrain (nx*ny*real_t)
        // We have 3 states: state, state_old, state_init
        size_t n3d = (size_t)(active_grid.nx + 4) * (active_grid.ny + 4) * active_grid.nz;
        int boundary_state_count = static_cast<int>(boundary_specs.size());
        int state_count = 3 + boundary_state_count + (boundary_state_count > 0 ? 1 : 0);
        size_t mem_3d_fields = n3d * 17 * sizeof(real_t) * state_count;
        size_t mem_base = (size_t)active_grid.nz * 5 * sizeof(double) * state_count;
        size_t mem_est = mem_3d_fields + mem_base;

        printf("Domain: %d x %d x %d (%.1f M points)\n",
               active_grid.nx, active_grid.ny, active_grid.nz,
               (double)active_grid.nx * active_grid.ny * active_grid.nz / 1e6);
        printf("Domain size: %.0f x %.0f km\n", active_grid.lx / 1000.0, active_grid.ly / 1000.0);
        printf("Resolution: %.0f m, dt=%.1f s\n", active_grid.dx, dt);
        printf("Projection: tl1=%.2f tl2=%.2f stand_lon=%.2f ref=(%.2f, %.2f)\n",
               active_grid.truelat1, active_grid.truelat2, active_grid.stand_lon,
               active_grid.ref_lat, active_grid.ref_lon);
        printf("Precision: %s (real_t = %zu bytes)\n",
               sizeof(real_t) == 4 ? "float32" : "float64", sizeof(real_t));
        printf("Est. GPU memory: %.2f GB (3D fields: %.2f GB)\n",
               mem_est / 1e9, mem_3d_fields / 1e9);
        printf("BCs: %s\n", use_open_bc ? "open (relaxation)" : "periodic");
        if (!boundary_specs.empty()) {
            printf("Parent boundaries: %zu scheduled snapshots\n", boundary_specs.size());
            for (const BoundarySpec& spec : boundary_specs) {
                if (spec.has_explicit_time) {
                    printf("  - %s @ %.0f s\n", spec.path.c_str(), spec.time_seconds);
                } else {
                    printf("  - %s @ inferred valid time\n", spec.path.c_str());
                }
            }
        }
        printf("Microphysics: %s\n", use_thompson ? "Thompson (mixed-phase)" : "Kessler (warm rain)");
        printf("Adaptive stability: %s\n", cfg.stability.enabled ? "on" : "off");
        printf("Vertical w damping: %s", cfg.stability.w_cfl_damping ? "on" : "off");
        if (cfg.stability.w_cfl_damping) {
            printf(" (alpha=%.2f beta=%.2f)", cfg.stability.w_damping_alpha, cfg.stability.w_damping_beta);
        }
        printf("\n");
        printf("w transport blend: %.2f", cfg.stability.w_transport_blend);
        if (cfg.stability.w_transport_diagnostics) {
            printf(" (diagnostics on)");
        }
        printf("\n");
        printf("semi-implicit p-w: %s", cfg.stability.pw_column_implicit ? "on" : "off");
        if (cfg.stability.pw_column_diagnostics) {
            printf(" (diagnostics on)");
        }
        printf("\n");
        printf("Output: %s\n\n", use_netcdf ? "NetCDF" : "binary");

        if (mem_est > prop.totalGlobalMem * 0.9) {
            fprintf(stderr, "WARNING: Domain may exceed GPU memory (%.1f GB available)\n",
                    prop.totalGlobalMem / 1e9);
        }
    };

    // Physics parameters
    double kdiff = cfg.kdiff;
    double z0 = cfg.z0;
    double theta_sfc = cfg.theta_sfc;
    double qv_sfc = cfg.qv_sfc;
    double cs_smag = cfg.cs_smag;

    // Create output directory
    mkdir("output", 0755);

    // Initialize
    StateGPU state{}, state_old{}, state_init{}, state_bdy{};
    std::vector<BoundaryNode> boundary_nodes;

    int gfs_mode = 0;  // 1 if running from GFS data
    int use_boundary_stream = 0;

    if (strlen(gfs_file) > 0) {
        // GFS binary initialization: load_gfs_binary handles allocation,
        // vertical levels, field loading, and base state computation.
        if (!load_gfs_binary(state, grid, gfs_file)) {
            fprintf(stderr, "FATAL: Failed to load GFS binary file: %s\n", gfs_file);
            return 1;
        }
        // Rebuild projection to match the loaded metadata
        proj = projection_from_grid(grid);
        gfs_mode = 1;
        use_open_bc = 1;  // GFS runs always use open BCs
        printf("GFS mode: open BCs enabled, microphysics+PBL+sponge active\n");
    } else {
        initialize_model(state, grid, cfg.test_case);
    }

    setup_projection_metrics(grid, proj);
    convert_w_to_contravariant(state, grid);
    if (gfs_mode) {
        initialize_w_from_continuity(state, grid, "primary-init");
    }
    initialize_tskin_from_surface_layer(state, grid, theta_sfc);
    print_case_summary(grid);

    allocate_state(state_old, grid);
    allocate_state(state_init, grid);
    copy_base_state(state_old, state, grid);
    copy_base_state(state_init, state, grid);

    // Save initial state for boundary relaxation
    copy_state(state_init, state, grid);

    if (!boundary_specs.empty()) {
        boundary_nodes.reserve(boundary_specs.size());
        for (std::size_t idx = 0; idx < boundary_specs.size(); ++idx) {
            const BoundarySpec& spec = boundary_specs[idx];
            boundary_nodes.emplace_back();
            BoundaryNode& node = boundary_nodes.back();
            node.path = spec.path;
            reset_boundary_grid(node.grid, grid);

            if (!load_gfs_binary(node.state, node.grid, spec.path.c_str())) {
                fprintf(stderr, "FATAL: Failed to load boundary state: %s\n", spec.path.c_str());
                free_boundary_grid(node.grid);
                return 1;
            }
            if (!boundary_grids_compatible(grid, node.grid)) {
                fprintf(stderr, "FATAL: Boundary state grid/projection does not match the primary init: %s\n",
                        spec.path.c_str());
                free_state(node.state);
                free_boundary_grid(node.grid);
                return 1;
            }

            double fallback_seconds = (idx + 1 == boundary_specs.size() && boundary_interval > 0.0)
                ? boundary_interval
                : -1.0;
            node.time_seconds = infer_boundary_time_seconds(grid, node.grid, spec, fallback_seconds);
            if (node.time_seconds <= 0.0) {
                fprintf(stderr,
                        "FATAL: Could not infer a positive boundary time for %s. "
                        "Use FILE@SECONDS or provide timestamped init binaries.\n",
                        spec.path.c_str());
                free_state(node.state);
                free_boundary_grid(node.grid);
                return 1;
            }
            convert_w_to_contravariant(node.state, grid);
            initialize_w_from_continuity(node.state, grid, spec.path.c_str());
        }

        std::sort(boundary_nodes.begin(), boundary_nodes.end(),
                  [](const BoundaryNode& a, const BoundaryNode& b) {
                      return a.time_seconds < b.time_seconds;
                  });
        for (std::size_t idx = 1; idx < boundary_nodes.size(); ++idx) {
            if (boundary_nodes[idx].time_seconds <= boundary_nodes[idx - 1].time_seconds) {
                fprintf(stderr, "FATAL: Boundary states must have strictly increasing times\n");
                return 1;
            }
        }

        allocate_state(state_bdy, grid);
        copy_base_state(state_bdy, state_init, grid);
        copy_state(state_bdy, state_init, grid);
        use_boundary_stream = 1;
    }

    // Initial output
    if (use_netcdf) {
        write_netcdf(state, grid, proj, 0.0, 0);
    } else {
        write_output(state, grid, 0.0, 0);
    }
    print_diagnostics(state, grid, 0.0, 0, cfg.stability);

    // Main time loop
    printf("\nStarting time integration...\n");
    printf("  dt=%.1f s, t_end=%.0f s (%s), steps=%d\n\n",
           dt, t_end,
           t_end >= 3600 ? (t_end >= 86400 ?
               (std::string(std::to_string((int)(t_end/86400))) + " days").c_str() :
               (std::string(std::to_string((int)(t_end/3600))) + " hours").c_str()) :
               (std::string(std::to_string((int)(t_end/60))) + " min").c_str(),
           (int)(t_end / dt));

    int total_steps = (int)(t_end / dt);
    int output_num = 1;
    double next_output = cfg.output_interval;

    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int step = 1; step <= total_steps; step++) {
        double time = step * dt;
        if (use_boundary_stream) {
            const StateGPU* blend_a = &state_init;
            const StateGPU* blend_b = &boundary_nodes.front().state;
            double time_a = 0.0;
            double time_b = boundary_nodes.front().time_seconds;

            if (time >= boundary_nodes.back().time_seconds) {
                blend_a = &boundary_nodes.back().state;
                blend_b = &boundary_nodes.back().state;
                time_a = boundary_nodes.back().time_seconds;
                time_b = boundary_nodes.back().time_seconds;
            } else {
                for (std::size_t idx = 0; idx < boundary_nodes.size(); ++idx) {
                    if (time <= boundary_nodes[idx].time_seconds) {
                        blend_b = &boundary_nodes[idx].state;
                        time_b = boundary_nodes[idx].time_seconds;
                        if (idx > 0) {
                            blend_a = &boundary_nodes[idx - 1].state;
                            time_a = boundary_nodes[idx - 1].time_seconds;
                        }
                        break;
                    }
                }
            }

            double alpha = 0.0;
            if (time_b > time_a) {
                alpha = fmin(fmax((time - time_a) / (time_b - time_a), 0.0), 1.0);
            }
            blend_boundary_state(state_bdy, *blend_a, *blend_b, grid, alpha);
        }
        StateGPU& boundary_target = use_boundary_stream ? state_bdy : state_init;

        // RK3 time integration
        copy_state(state_old, state, grid);

        for (int rk = 0; rk < 3; rk++) {
            rk3_step(state, state_old, boundary_target, grid, dt, kdiff, rk,
                     use_open_bc != 0, cfg.stability, cfg.relax_width);
        }

        // Sponge layers: damp upper boundary and lateral edges
        // Applied right after dynamics, before physics
        if (gfs_mode || use_open_bc) {
            apply_sponge(state, boundary_target, grid, dt);
        }

        // Physics
        // Microphysics: run for convective tests and GFS (which has moisture)
        if (cfg.test_case == 3 || gfs_mode) {
            if (use_thompson) {
                run_microphysics_thompson(state, grid, dt);
            } else {
                run_microphysics(state, grid, dt);
            }
        }

        // PBL: implicit vertical diffusion + surface layer
        if (cfg.test_case == 3 || gfs_mode) {
            update_tskin_slab(state, grid, z0, theta_sfc, qv_sfc, dt);
            run_pbl(state, grid, z0, qv_sfc, cs_smag, dt);
        }

        // Boundary conditions after physics
        // The full relaxation+extrapolation cycle already ran at each RK
        // sub-stage inside rk3_step.  After physics we only need to refresh
        // halos so that any physics-induced changes propagate to the halo
        // cells before the next timestep's stencil work.  Running the full
        // apply_open_boundaries here would apply an extra round of lateral
        // relaxation (on top of the 3 rounds inside RK3), over-constraining
        // the solution near the boundaries.
        if (use_open_bc) {
            refresh_open_halos(state, grid);
        }

        // Diagnostics
        if (step % cfg.diag_interval == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            print_diagnostics(state, grid, time, step, cfg.stability);
        }

        // Output
        if (time >= next_output - 0.5 * dt) {
            CUDA_CHECK(cudaDeviceSynchronize());
            if (use_netcdf) {
                write_netcdf(state, grid, proj, time, output_num);
            } else {
                write_output(state, grid, time, output_num);
            }
            output_num++;
            next_output += cfg.output_interval;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_secs = std::chrono::duration<double>(wall_end - wall_start).count();

    printf("\n============================================\n");
    printf("  Simulation complete!\n");
    printf("  Simulated: %.0f s (%.1f hours)\n", t_end, t_end / 3600.0);
    printf("  Wall clock: %.2f s\n", wall_secs);
    printf("  Speed: %.1fx real-time\n", t_end / wall_secs);
    printf("  Grid: %d x %d x %d = %.1f M points\n",
           grid.nx, grid.ny, grid.nz,
           (double)grid.nx * grid.ny * grid.nz / 1e6);
    printf("  Throughput: %.1f M pts/s\n",
           (double)grid.nx * grid.ny * grid.nz * total_steps / wall_secs / 1e6);
    printf("============================================\n");

    // Cleanup
    free_state(state);
    free_state(state_old);
    free_state(state_init);
    free_state(state_bdy);
    for (BoundaryNode& node : boundary_nodes) {
        free_state(node.state);
        free_boundary_grid(node.grid);
    }
    free_grid_metrics(grid);
    delete[] grid.eta;
    delete[] grid.eta_m;

    return 0;
}
