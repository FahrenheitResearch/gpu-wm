// ============================================================
// GPU-WM: Model Initialization
// Sets up base state, vertical levels, and initial conditions
// Includes standard test cases
// ============================================================

#include "../../include/constants.cuh"
#include "../../include/grid.cuh"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace gpuwm {

static constexpr char GFS_PROJ_MAGIC[8] = {'G','W','M','P','R','J','1','\0'};
static constexpr char GFS_PRESSURE_MAGIC[8] = {'G','W','M','P','R','E','S','1'};
static constexpr char GFS_TERRAIN_MAGIC[8] = {'G','W','M','T','E','R','R','1'};
static constexpr char GFS_INIT_MODE_MAGIC[8] = {'G','W','M','I','N','I','T','1'};
static constexpr char GFS_TIME_MAGIC[8] = {'G','W','M','T','I','M','E','1'};

// Forward declarations from dynamics.cu
void apply_boundary_conditions(StateGPU& state, const GridConfig& grid);

static void release_vertical_levels(GridConfig& grid) {
    delete[] grid.eta;
    delete[] grid.eta_m;
    grid.eta = nullptr;
    grid.eta_m = nullptr;
}

static inline double clamp_loaded_terrain_host(double terrain, double ztop) {
    return fmin(terrain, ztop - 1.0);
}

static inline double interpolate_profile_linear_host(
    const double* z_ref,
    const double* profile,
    int nz,
    double z_target
) {
    if (nz <= 1) return profile[0];
    if (z_target <= z_ref[0]) {
        double dz = fmax(z_ref[1] - z_ref[0], 1.0);
        double slope = (profile[1] - profile[0]) / dz;
        return profile[0] + slope * (z_target - z_ref[0]);
    }
    if (z_target >= z_ref[nz - 1]) {
        double dz = fmax(z_ref[nz - 1] - z_ref[nz - 2], 1.0);
        double slope = (profile[nz - 1] - profile[nz - 2]) / dz;
        return profile[nz - 1] + slope * (z_target - z_ref[nz - 1]);
    }

    int k_hi = 1;
    while (k_hi < nz && z_ref[k_hi] < z_target) {
        ++k_hi;
    }
    int k_lo = k_hi - 1;
    double frac = (z_target - z_ref[k_lo]) / fmax(z_ref[k_hi] - z_ref[k_lo], 1.0);
    return profile[k_lo] + frac * (profile[k_hi] - profile[k_lo]);
}

static inline double interpolate_profile_log_host(
    const double* z_ref,
    const double* profile,
    int nz,
    double z_target
) {
    if (nz <= 1) return fmax(profile[0], 1.0);
    if (z_target <= z_ref[0]) {
        double dz = fmax(z_ref[1] - z_ref[0], 1.0);
        double slope = (log(fmax(profile[1], 1.0)) - log(fmax(profile[0], 1.0))) / dz;
        return exp(log(fmax(profile[0], 1.0)) + slope * (z_target - z_ref[0]));
    }
    if (z_target >= z_ref[nz - 1]) {
        double dz = fmax(z_ref[nz - 1] - z_ref[nz - 2], 1.0);
        double slope = (log(fmax(profile[nz - 1], 1.0)) - log(fmax(profile[nz - 2], 1.0))) / dz;
        return exp(log(fmax(profile[nz - 1], 1.0)) + slope * (z_target - z_ref[nz - 1]));
    }

    int k_hi = 1;
    while (k_hi < nz && z_ref[k_hi] < z_target) {
        ++k_hi;
    }
    int k_lo = k_hi - 1;
    double frac = (z_target - z_ref[k_lo]) / fmax(z_ref[k_hi] - z_ref[k_lo], 1.0);
    double log_lo = log(fmax(profile[k_lo], 1.0));
    double log_hi = log(fmax(profile[k_hi], 1.0));
    return exp(log_lo + frac * (log_hi - log_lo));
}

// ----------------------------------------------------------
// Setup vertical levels (stretched grid)
// ----------------------------------------------------------
void setup_vertical_levels(GridConfig& grid) {
    release_vertical_levels(grid);

    double* eta_h = new double[grid.nz + 1];
    double* eta_m_h = new double[grid.nz];

    // Two-zone stretched eta: mildly packed near the surface, then
    // quasi-uniform aloft. This keeps lower-PBL resolution without
    // creating an excessively thin first layer.
    int n_sfc = grid.nz / 5;
    if (n_sfc < 3) n_sfc = 3;
    if (n_sfc > 8) n_sfc = 8;
    if (grid.nz > 1) {
        if (n_sfc >= grid.nz) n_sfc = grid.nz - 1;
    } else {
        n_sfc = 1;
    }
    double eta_sfc = 1000.0 / fmax(grid.ztop, 1000.0);
    eta_sfc = fmin(fmax(eta_sfc, 0.02), 0.08);
    constexpr double sfc_power = 1.4;

    eta_h[0] = 0.0;
    for (int k = 1; k <= grid.nz; ++k) {
        if (k <= n_sfc) {
            double frac = (double)k / (double)n_sfc;
            eta_h[k] = eta_sfc * pow(frac, sfc_power);
        } else {
            double frac = (double)(k - n_sfc) / (double)(grid.nz - n_sfc);
            eta_h[k] = eta_sfc + (1.0 - eta_sfc) * frac;
        }
    }
    eta_h[grid.nz] = 1.0;

    for (int k = 0; k < grid.nz; ++k) {
        eta_m_h[k] = 0.5 * (eta_h[k] + eta_h[k + 1]);
    }

    grid.eta = new double[grid.nz + 1];
    grid.eta_m = new double[grid.nz];
    memcpy(grid.eta, eta_h, (grid.nz + 1) * sizeof(double));
    memcpy(grid.eta_m, eta_m_h, grid.nz * sizeof(double));

    delete[] eta_h;
    delete[] eta_m_h;
}

// ----------------------------------------------------------
// Initialize base state (hydrostatic, US Standard Atmosphere)
// ----------------------------------------------------------
void init_base_state(StateGPU& state, const GridConfig& grid) {
    int nz = grid.nz;

    double* z_w = new double[nz + 1];
    double* z_m = new double[nz];
    double* theta_b = new double[nz];
    double* p_b = new double[nz];
    double* rho_b = new double[nz];
    double* qv_b = new double[nz];

    // Height levels from eta
    for (int k = 0; k <= nz; k++) {
        z_w[k] = grid.eta[k] * grid.ztop;
    }
    for (int k = 0; k < nz; k++) {
        z_m[k] = 0.5 * (z_w[k] + z_w[k + 1]);
    }

    // Hydrostatic base state
    for (int k = 0; k < nz; k++) {
        double z = z_m[k];

        // Temperature profile with lapse rate
        double T;
        if (z < 11000.0) {
            T = T0 - LAPSE_RATE * z;
        } else {
            T = T0 - LAPSE_RATE * 11000.0;  // Isothermal above tropopause
        }

        // Pressure from hypsometric equation
        if (z < 11000.0) {
            p_b[k] = P0 * pow(T / T0, G / (R_D * LAPSE_RATE));
        } else {
            double p_trop = P0 * pow((T0 - LAPSE_RATE * 11000.0) / T0, G / (R_D * LAPSE_RATE));
            p_b[k] = p_trop * exp(-G * (z - 11000.0) / (R_D * T));
        }

        // Potential temperature
        theta_b[k] = T * pow(P0 / p_b[k], KAPPA);

        // Density
        rho_b[k] = p_b[k] / (R_D * T);
        qv_b[k] = 0.0;
    }

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(state.z_w_levels, z_w, (nz + 1) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.z_levels, z_m, nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.theta_base, theta_b, nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.p_base, p_b, nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.rho_base, rho_b, nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.qv_base, qv_b, nz * sizeof(double), cudaMemcpyHostToDevice));

    delete[] z_w;
    delete[] z_m;
    delete[] theta_b;
    delete[] p_b;
    delete[] rho_b;
    delete[] qv_b;
}

// ----------------------------------------------------------
// Initialize fields kernel
// ----------------------------------------------------------
__global__ void init_fields_kernel(
    real_t* __restrict__ theta,
    real_t* __restrict__ u,
    real_t* __restrict__ v,
    real_t* __restrict__ w,
    real_t* __restrict__ qv,
    real_t* __restrict__ qc,
    real_t* __restrict__ qr,
    real_t* __restrict__ p,
    real_t* __restrict__ rho,
    const double* __restrict__ theta_base,
    const double* __restrict__ p_base,
    const double* __restrict__ rho_base,
    const double* __restrict__ z_levels,
    int nx, int ny, int nz,
    double dx, double dy,
    int test_case  // 0=quiescent, 1=thermal bubble, 2=density current, 3=supercell
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int nx_h = nx + 4;
    int ny_h = ny + 4;

    // Handle halo region too
    int i_actual = i - 2;
    int j_actual = j - 2;

    if (i >= nx_h || j >= ny_h || k >= nz) return;

    int ijk = k * ny_h * nx_h + j * nx_h + i;

    double z = z_levels[k];

    // Compute everything in double, write as real_t at the end
    double th_val = theta_base[k];
    double p_val = 0.0;
    double rho_val = rho_base[k];
    double u_val = 0.0, v_val = 0.0, w_val = 0.0;
    double qv_val = 0.0, qc_val = 0.0, qr_val = 0.0;

    // Physical coordinates (centered domain)
    double x = (i_actual - nx / 2.0) * dx;
    double y = (j_actual - ny / 2.0) * dy;

    if (test_case == 1) {
        // --- Warm bubble (Robert 1993) ---
        double xc = 0.0, yc = 0.0, zc = 3000.0;
        double rx = 4000.0, ry = 4000.0, rz = 2000.0;
        double r = sqrt((x-xc)*(x-xc)/(rx*rx) + (y-yc)*(y-yc)/(ry*ry) + (z-zc)*(z-zc)/(rz*rz));
        if (r <= 1.0) {
            th_val += 2.0 * cos(r * PI / 2.0) * cos(r * PI / 2.0);
        }
        // Background moisture
        if (z < 8000.0) {
            qv_val = 0.014 * exp(-z / 3000.0);
        }
    }
    else if (test_case == 2) {
        // --- Cold density current (Straka et al. 1993) ---
        double xc = 0.0, zc = 3000.0;
        double rx = 4000.0, rz = 2000.0;
        double r = sqrt((x-xc)*(x-xc)/(rx*rx) + (z-zc)*(z-zc)/(rz*rz));
        if (r <= 1.0) {
            th_val -= 15.0 * cos(r * PI / 2.0) * cos(r * PI / 2.0);
        }
    }
    else if (test_case == 3) {
        // --- Multicell convective storm environment ---
        if (z < 1000.0) {
            u_val = -5.0 + 15.0 * z / 1000.0;
            v_val = -3.0 + 6.0 * z / 1000.0;
        } else if (z < 6000.0) {
            u_val = 10.0 + 10.0 * (z - 1000.0) / 5000.0;
            v_val = 3.0 + 5.0 * (z - 1000.0) / 5000.0;
        } else {
            u_val = 20.0;
            v_val = 8.0;
        }

        if (z < 1500.0) {
            qv_val = 0.018 - 0.002 * z / 1500.0;
        } else if (z < 3000.0) {
            qv_val = 0.016 * exp(-(z - 1500.0) / 2000.0);
        } else if (z < 8000.0) {
            qv_val = 0.004 * exp(-(z - 3000.0) / 4000.0);
        } else {
            qv_val = 0.0005;
        }

        if (z < 2000.0) {
            th_val -= 2.0 * (1.0 - z / 2000.0);
        }

        double lx = nx * dx;
        double ly = ny * dy;
        double x_abs = x + lx / 2.0;
        double y_abs = y + ly / 2.0;

        double line_x = lx * 0.35;
        double dist_from_line = x_abs - line_x - 3000.0 * sin(2.0 * PI * y_abs / (ly * 0.5));
        if (fabs(dist_from_line) < 8000.0 && z < 2000.0) {
            double env = cos(dist_from_line / 8000.0 * PI / 2.0);
            double zenv = cos(z / 2000.0 * PI / 2.0);
            th_val += 3.0 * env * env * zenv;
            qv_val += 0.002 * env * env * zenv;
        }

        for (int ti = 0; ti < 6; ti++) {
            double tx = lx * (0.15 + 0.12 * ti + 0.05 * sin(ti * 2.7));
            double ty = ly * (0.2 + 0.1 * ti + 0.08 * cos(ti * 1.9));
            double tr = 5000.0 + 2000.0 * sin(ti * 3.1);
            double tz = 1500.0;

            double R = sqrt((x_abs - tx) * (x_abs - tx) +
                           (y_abs - ty) * (y_abs - ty) +
                           (z - tz) * (z - tz));
            if (R < tr && z < 2500.0) {
                double env = cos(R / tr * PI / 2.0);
                th_val += 2.5 * env * env;
                qv_val += 0.001 * env * env;
            }
        }

        qv_val = fmax(qv_val, 0.0);
    }

    // Write as real_t
    theta[ijk] = (real_t)th_val;
    p[ijk]     = (real_t)p_val;
    rho[ijk]   = (real_t)rho_val;
    u[ijk]     = (real_t)u_val;
    v[ijk]     = (real_t)v_val;
    w[ijk]     = (real_t)w_val;
    qv[ijk]    = (real_t)qv_val;
    qc[ijk]    = (real_t)qc_val;
    qr[ijk]    = (real_t)qr_val;
}

// ----------------------------------------------------------
// Host driver for initialization
// ----------------------------------------------------------
void initialize_model(StateGPU& state, GridConfig& grid, int test_case) {
    printf("GPU-WM: Initializing model...\n");
    printf("  Grid: %d x %d x %d\n", grid.nx, grid.ny, grid.nz);
    printf("  dx=%.0f m, dy=%.0f m, ztop=%.0f m\n", grid.dx, grid.dy, grid.ztop);

    // Setup vertical levels
    setup_vertical_levels(grid);

    // Allocate GPU state
    allocate_state(state, grid);

    // Initialize base state
    init_base_state(state, grid);

    // Initialize fields
    int nx_h = grid.nx + 4;
    int ny_h = grid.ny + 4;

    dim3 block(8, 8, 4);
    dim3 grid3d((nx_h + 7) / 8, (ny_h + 7) / 8, (grid.nz + 3) / 4);

    init_fields_kernel<<<grid3d, block>>>(
        state.theta, state.u, state.v, state.w,
        state.qv, state.qc, state.qr, state.p, state.rho,
        state.theta_base, state.p_base, state.rho_base, state.z_levels,
        grid.nx, grid.ny, grid.nz,
        grid.dx, grid.dy,
        test_case
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply boundary conditions
    apply_boundary_conditions(state, grid);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  Test case: %d\n", test_case);
    printf("  Initialization complete.\n");
}

// ----------------------------------------------------------
// Load GFS binary initialization file
// Format: header(nx,ny,nz as int32; dx,dy,ztop as float64)
//         z_levels[nz] float64
//         7 fields [nz,ny,nx] float64: u, v, w, theta, qv, qc, qr
// ----------------------------------------------------------
bool load_gfs_binary(StateGPU& state, GridConfig& grid, const char* filename) {
    printf("GPU-WM: Loading GFS binary from %s\n", filename);

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open GFS binary file: %s\n", filename);
        return false;
    }

    // --- Read header ---
    int32_t file_nx, file_ny, file_nz;
    double file_dx, file_dy, file_ztop;

    if (fread(&file_nx, sizeof(int32_t), 1, fp) != 1 ||
        fread(&file_ny, sizeof(int32_t), 1, fp) != 1 ||
        fread(&file_nz, sizeof(int32_t), 1, fp) != 1) {
        fprintf(stderr, "ERROR: Failed to read header dimensions\n");
        fclose(fp);
        return false;
    }
    if (fread(&file_dx, sizeof(double), 1, fp) != 1 ||
        fread(&file_dy, sizeof(double), 1, fp) != 1 ||
        fread(&file_ztop, sizeof(double), 1, fp) != 1) {
        fprintf(stderr, "ERROR: Failed to read header grid spacing\n");
        fclose(fp);
        return false;
    }

    printf("  Binary header: nx=%d ny=%d nz=%d dx=%.0f dy=%.0f ztop=%.0f\n",
           file_nx, file_ny, file_nz, file_dx, file_dy, file_ztop);

    // Resize grid to match the file
    if (grid.nx != file_nx || grid.ny != file_ny || grid.nz != file_nz) {
        printf("  Adjusting grid from %dx%dx%d to %dx%dx%d to match binary file\n",
               grid.nx, grid.ny, grid.nz, file_nx, file_ny, file_nz);
        grid.nx = file_nx;
        grid.ny = file_ny;
        grid.nz = file_nz;
        grid.lx = grid.nx * grid.dx;
        grid.ly = grid.ny * grid.dy;
    }
    grid.dx = file_dx;
    grid.dy = file_dy;
    grid.ztop = file_ztop;
    grid.lx = grid.nx * grid.dx;
    grid.ly = grid.ny * grid.dy;

    int nx = grid.nx;
    int ny = grid.ny;
    int nz = grid.nz;
    int nx_h = nx + 4;
    int ny_h = ny + 4;
    size_t n3d_h = (size_t)nx_h * ny_h * nz;
    size_t n3d_file = (size_t)nx * ny * nz;

    // --- Read z_levels ---
    double* z_levels_h = new double[nz];
    if (fread(z_levels_h, sizeof(double), nz, fp) != (size_t)nz) {
        fprintf(stderr, "ERROR: Failed to read z_levels\n");
        delete[] z_levels_h;
        fclose(fp);
        return false;
    }
    printf("  z_levels: [%.1f, %.1f, ..., %.1f] m\n",
           z_levels_h[0], z_levels_h[1], z_levels_h[nz - 1]);

    // --- Setup vertical levels (eta) from z_levels ---
    // Derive eta from z_levels: eta = z / ztop
    release_vertical_levels(grid);
    grid.eta = new double[nz + 1];
    grid.eta_m = new double[nz];
    // z_levels are cell midpoints. Reconstruct w-levels (cell edges).
    double* z_w = new double[nz + 1];
    z_w[0] = 0.0;
    for (int k = 0; k < nz; k++) {
        if (k < nz - 1) {
            z_w[k + 1] = 0.5 * (z_levels_h[k] + z_levels_h[k + 1]);
        } else {
            z_w[k + 1] = grid.ztop;
        }
        grid.eta_m[k] = z_levels_h[k] / grid.ztop;
    }
    for (int k = 0; k <= nz; k++) {
        grid.eta[k] = z_w[k] / grid.ztop;
    }

    // --- Allocate GPU state ---
    allocate_state(state, grid);

    // --- Copy z_levels and z_w to GPU ---
    CUDA_CHECK(cudaMemcpy(state.z_levels, z_levels_h, nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.z_w_levels, z_w, (nz + 1) * sizeof(double), cudaMemcpyHostToDevice));

    // --- Read 3D fields and copy into halo-padded GPU arrays ---
    double* field_buf = new double[n3d_file];  // [nz, ny, nx] from file
    double* halo_buf = new double[n3d_h];      // [nz, ny_h, nx_h] with halos

    // Helper lambda: read one field from the file, pad into halo layout, copy to GPU
    // Converts from double (file format) to real_t (GPU storage)
    real_t* halo_buf_rt = new real_t[n3d_h];

    auto read_and_upload = [&](real_t* d_field, const char* name) -> bool {
        if (fread(field_buf, sizeof(double), n3d_file, fp) != n3d_file) {
            fprintf(stderr, "ERROR: Failed to read field %s\n", name);
            return false;
        }

        // Zero the halo buffer
        memset(halo_buf_rt, 0, n3d_h * sizeof(real_t));

        // Copy interior: file layout is [k][j][i] with i fastest
        // GPU layout is [k][j_h][i_h] where i_h = i+2, j_h = j+2
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int src = k * ny * nx + j * nx + i;
                    int dst = k * ny_h * nx_h + (j + 2) * nx_h + (i + 2);
                    halo_buf_rt[dst] = (real_t)field_buf[src];
                }
            }
        }

        CUDA_CHECK(cudaMemcpy(d_field, halo_buf_rt, n3d_h * sizeof(real_t), cudaMemcpyHostToDevice));
        return true;
    };

    bool ok = true;
    ok = ok && read_and_upload(state.u, "u");
    ok = ok && read_and_upload(state.v, "v");
    ok = ok && read_and_upload(state.w, "w");
    ok = ok && read_and_upload(state.theta, "theta");
    ok = ok && read_and_upload(state.qv, "qv");
    ok = ok && read_and_upload(state.qc, "qc");
    ok = ok && read_and_upload(state.qr, "qr");

    if (!ok) {
        delete[] z_levels_h;
        delete[] z_w;
        delete[] field_buf;
        delete[] halo_buf;
        delete[] halo_buf_rt;
        fclose(fp);
        return false;
    }

    double* pressure_h = nullptr;
    bool have_pressure_trailer = false;
    bool have_terrain_trailer = false;
    bool file_terrain_following_init = false;
    size_t n2d = (size_t)nx * ny;
    double* terrain_host = new double[n2d];
    memset(terrain_host, 0, n2d * sizeof(double));

    while (true) {
        char trailer_magic[8];
        size_t trailer_bytes = fread(trailer_magic, 1, sizeof(trailer_magic), fp);
        if (trailer_bytes == 0) {
            break;
        }
        if (trailer_bytes != sizeof(trailer_magic)) {
            fprintf(stderr, "ERROR: Truncated trailer in %s\n", filename);
            delete[] z_levels_h;
            delete[] z_w;
            delete[] field_buf;
            delete[] halo_buf;
            delete[] halo_buf_rt;
            fclose(fp);
            return false;
        }

        if (memcmp(trailer_magic, GFS_PROJ_MAGIC, sizeof(trailer_magic)) == 0) {
            double proj_meta[5];
            if (fread(proj_meta, sizeof(double), 5, fp) != 5) {
                fprintf(stderr, "ERROR: Failed to read projection metadata in %s\n", filename);
                delete[] z_levels_h;
                delete[] z_w;
                delete[] field_buf;
                delete[] halo_buf;
                delete[] halo_buf_rt;
                fclose(fp);
                return false;
            }
            grid.truelat1 = proj_meta[0];
            grid.truelat2 = proj_meta[1];
            grid.stand_lon = proj_meta[2];
            grid.ref_lat = proj_meta[3];
            grid.ref_lon = proj_meta[4];
            grid.clat = grid.ref_lat;
            grid.clon = grid.ref_lon;
            printf("  Projection metadata: truelat1=%.2f truelat2=%.2f stand_lon=%.2f ref=(%.2f, %.2f)\n",
                   grid.truelat1, grid.truelat2, grid.stand_lon, grid.ref_lat, grid.ref_lon);
        } else if (memcmp(trailer_magic, GFS_PRESSURE_MAGIC, sizeof(trailer_magic)) == 0) {
            if (!pressure_h) {
                pressure_h = new double[n3d_file];
            }
            if (fread(pressure_h, sizeof(double), n3d_file, fp) != n3d_file) {
                fprintf(stderr, "ERROR: Failed to read pressure metadata in %s\n", filename);
                delete[] pressure_h;
                delete[] z_levels_h;
                delete[] z_w;
                delete[] field_buf;
                delete[] halo_buf;
                delete[] halo_buf_rt;
                fclose(fp);
                return false;
            }
            have_pressure_trailer = true;
            double p_min = pressure_h[0];
            double p_max = pressure_h[0];
            for (size_t idx = 1; idx < n3d_file; idx++) {
                p_min = fmin(p_min, pressure_h[idx]);
                p_max = fmax(p_max, pressure_h[idx]);
            }
            printf("  Pressure metadata: min=%.0f Pa max=%.0f Pa\n",
                   p_min, p_max);
        } else if (memcmp(trailer_magic, GFS_TERRAIN_MAGIC, sizeof(trailer_magic)) == 0) {
            double* terrain_h = new double[n2d];
            real_t* terrain_rt = new real_t[n2d];
            if (fread(terrain_h, sizeof(double), n2d, fp) != n2d) {
                fprintf(stderr, "ERROR: Failed to read terrain metadata in %s\n", filename);
                delete[] terrain_h;
                delete[] terrain_rt;
                delete[] z_levels_h;
                delete[] z_w;
                delete[] field_buf;
                delete[] halo_buf;
                delete[] halo_buf_rt;
                fclose(fp);
                return false;
            }
            double terrain_min = terrain_h[0];
            double terrain_max = terrain_h[0];
            for (size_t idx = 0; idx < n2d; idx++) {
                terrain_min = fmin(terrain_min, terrain_h[idx]);
                terrain_max = fmax(terrain_max, terrain_h[idx]);
                terrain_host[idx] = terrain_h[idx];
                terrain_rt[idx] = (real_t)terrain_h[idx];
            }
            CUDA_CHECK(cudaMemcpy(state.terrain, terrain_rt, n2d * sizeof(real_t), cudaMemcpyHostToDevice));
            have_terrain_trailer = true;
            printf("  Terrain metadata: min=%.1f m max=%.1f m\n",
                   terrain_min, terrain_max);
            delete[] terrain_h;
            delete[] terrain_rt;
        } else if (memcmp(trailer_magic, GFS_INIT_MODE_MAGIC, sizeof(trailer_magic)) == 0) {
            int32_t init_meta[2];
            if (fread(init_meta, sizeof(int32_t), 2, fp) != 2) {
                fprintf(stderr, "ERROR: Failed to read init mode metadata in %s\n", filename);
                delete[] terrain_host;
                delete[] pressure_h;
                delete[] z_levels_h;
                delete[] z_w;
                delete[] field_buf;
                delete[] halo_buf;
                delete[] halo_buf_rt;
                fclose(fp);
                return false;
            }
            file_terrain_following_init = init_meta[0] != 0;
            printf("  Init metadata: sampling=%s\n",
                   file_terrain_following_init ? "terrain-following" : "flat-height");
        } else if (memcmp(trailer_magic, GFS_TIME_MAGIC, sizeof(trailer_magic)) == 0) {
            int64_t time_meta[2];
            int32_t aux_meta[2];
            if (fread(time_meta, sizeof(int64_t), 2, fp) != 2 ||
                fread(aux_meta, sizeof(int32_t), 2, fp) != 2) {
                fprintf(stderr, "ERROR: Failed to read time metadata in %s\n", filename);
                delete[] terrain_host;
                delete[] pressure_h;
                delete[] z_levels_h;
                delete[] z_w;
                delete[] field_buf;
                delete[] halo_buf;
                delete[] halo_buf_rt;
                fclose(fp);
                return false;
            }
            grid.init_valid_time_unix = time_meta[0];
            grid.init_reference_time_unix = time_meta[1];
            grid.init_forecast_hour = aux_meta[0];
            printf("  Time metadata: valid_unix=%lld reference_unix=%lld forecast_hour=%d\n",
                   (long long)grid.init_valid_time_unix,
                   (long long)grid.init_reference_time_unix,
                   (int)grid.init_forecast_hour);
        } else {
            fprintf(stderr, "WARNING: Unknown trailer in %s; stopping trailer parse\n", filename);
            break;
        }
    }

    fclose(fp);

    // --- Recompute base state from the loaded 3D fields ---
    // When pressure metadata is available, use it to derive a balanced
    // perturbation pressure instead of forcing p' = 0 everywhere.
    printf("  Recomputing base state from loaded fields...\n");

    double* theta_base_h = new double[nz];
    double* p_base_h = new double[nz];
    double* rho_base_h = new double[nz];
    double* qv_base_h = new double[nz];

    real_t* theta_gpu_h = new real_t[n3d_h];
    real_t* qv_gpu_h = new real_t[n3d_h];
    CUDA_CHECK(cudaMemcpy(theta_gpu_h, state.theta, n3d_h * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qv_gpu_h, state.qv, n3d_h * sizeof(real_t), cudaMemcpyDeviceToHost));

    for (int k = 0; k < nz; k++) {
        double theta_sum = 0.0;
        double qv_sum = 0.0;
        double p_sum = 0.0;
        int count = 0;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = k * ny_h * nx_h + (j + 2) * nx_h + (i + 2);
                int src = k * ny * nx + j * nx + i;
                theta_sum += (double)theta_gpu_h[idx];
                qv_sum += (double)qv_gpu_h[idx];
                if (have_pressure_trailer) {
                    p_sum += pressure_h[src];
                }
                count++;
            }
        }
        theta_base_h[k] = theta_sum / count;
        qv_base_h[k] = qv_sum / count;
        if (have_pressure_trailer) {
            p_base_h[k] = p_sum / count;
        }
    }

    bool use_terrain_aware_reference =
        file_terrain_following_init && have_pressure_trailer && have_terrain_trailer;
    if (use_terrain_aware_reference) {
        printf("  Building terrain-aware hydrostatic reference from terrain-following init...\n");

        auto sample_halo_column = [&](const real_t* field, int i, int j, double terrain, double z_target) -> double {
            double column_depth = fmax(grid.ztop - terrain, 1.0);
            double dz_col = column_depth / nz;
            double z0 = terrain + 0.5 * dz_col;
            double src_pos = (z_target - z0) / dz_col;
            int k0;
            double frac;
            if (src_pos <= 0.0) {
                k0 = 0;
                frac = src_pos;
            } else if (src_pos >= nz - 1) {
                k0 = nz - 2;
                frac = src_pos - (nz - 2);
            } else {
                k0 = (int)floor(src_pos);
                frac = src_pos - k0;
            }
            int idx0 = k0 * ny_h * nx_h + (j + 2) * nx_h + (i + 2);
            int idx1 = (k0 + 1) * ny_h * nx_h + (j + 2) * nx_h + (i + 2);
            double v0 = (double)field[idx0];
            double v1 = (double)field[idx1];
            return v0 + frac * (v1 - v0);
        };

        auto sample_file_column = [&](const double* field, int i, int j, double terrain, double z_target) -> double {
            double column_depth = fmax(grid.ztop - terrain, 1.0);
            double dz_col = column_depth / nz;
            double z0 = terrain + 0.5 * dz_col;
            double src_pos = (z_target - z0) / dz_col;
            int k0;
            double frac;
            if (src_pos <= 0.0) {
                k0 = 0;
                frac = src_pos;
            } else if (src_pos >= nz - 1) {
                k0 = nz - 2;
                frac = src_pos - (nz - 2);
            } else {
                k0 = (int)floor(src_pos);
                frac = src_pos - k0;
            }
            int idx0 = k0 * ny * nx + j * nx + i;
            int idx1 = (k0 + 1) * ny * nx + j * nx + i;
            double v0 = field[idx0];
            double v1 = field[idx1];
            return v0 + frac * (v1 - v0);
        };

        double* theta_ref_sum = new double[nz];
        double* qv_ref_sum = new double[nz];
        double* p_ref_sum = new double[nz];
        int* ref_count = new int[nz];
        memset(theta_ref_sum, 0, nz * sizeof(double));
        memset(qv_ref_sum, 0, nz * sizeof(double));
        memset(p_ref_sum, 0, nz * sizeof(double));
        memset(ref_count, 0, nz * sizeof(int));

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double terrain = clamp_loaded_terrain_host(terrain_host[idx2(i, j, nx)], grid.ztop);
                for (int k = 0; k < nz; k++) {
                    double z_target = z_levels_h[k];
                    if (z_target <= terrain) continue;

                    theta_ref_sum[k] += sample_halo_column(theta_gpu_h, i, j, terrain, z_target);
                    qv_ref_sum[k] += fmax(sample_halo_column(qv_gpu_h, i, j, terrain, z_target), 0.0);
                    p_ref_sum[k] += sample_file_column(pressure_h, i, j, terrain, z_target);
                    ref_count[k]++;
                }
            }
        }

        for (int k = 0; k < nz; k++) {
            if (ref_count[k] > 0) {
                theta_base_h[k] = theta_ref_sum[k] / ref_count[k];
                qv_base_h[k] = qv_ref_sum[k] / ref_count[k];
            } else if (k > 0) {
                theta_base_h[k] = theta_base_h[k - 1];
                qv_base_h[k] = qv_base_h[k - 1];
            }
        }

        double p_bottom = (ref_count[0] > 0) ? (p_ref_sum[0] / ref_count[0]) : p_base_h[0];
        p_base_h[0] = p_bottom;
        for (int k = 1; k < nz; k++) {
            double dz = fmax(z_levels_h[k] - z_levels_h[k - 1], 1.0);
            double qv_layer = fmax(0.5 * (qv_base_h[k - 1] + qv_base_h[k]), 0.0);
            double p_guess = p_base_h[k - 1];
            for (int iter = 0; iter < 6; iter++) {
                double p_mid = 0.5 * (p_base_h[k - 1] + p_guess);
                double t_low = theta_base_h[k - 1] * pow(p_mid / P0, KAPPA);
                double t_high = theta_base_h[k] * pow(p_mid / P0, KAPPA);
                double tv_layer = 0.5 * (t_low + t_high) * (1.0 + 0.61 * qv_layer);
                p_guess = p_base_h[k - 1] * exp(-G * dz / (R_D * fmax(tv_layer, 150.0)));
            }
            p_base_h[k] = p_guess;
        }

        delete[] theta_ref_sum;
        delete[] qv_ref_sum;
        delete[] p_ref_sum;
        delete[] ref_count;
    }

    if (!have_pressure_trailer) {
        for (int k = 0; k < nz; k++) {
            double z = z_levels_h[k];
            double T_est;
            if (z < 11000.0) {
                T_est = T0 - LAPSE_RATE * z;
            } else {
                T_est = T0 - LAPSE_RATE * 11000.0;
            }

            double p_est;
            if (z < 11000.0) {
                p_est = P0 * pow(T_est / T0, G / (R_D * LAPSE_RATE));
            } else {
                double p_trop = P0 * pow((T0 - LAPSE_RATE * 11000.0) / T0, G / (R_D * LAPSE_RATE));
                p_est = p_trop * exp(-G * (z - 11000.0) / (R_D * T_est));
            }
            p_base_h[k] = p_est;
        }
    }

    for (int k = 0; k < nz; k++) {
        double theta_b = theta_base_h[k];
        double p_b = p_base_h[k];
        double T_b = theta_b * pow(p_b / P0, KAPPA);
        double Tv_b = T_b * (1.0 + 0.61 * fmax(qv_base_h[k], 0.0));
        rho_base_h[k] = p_b / (R_D * fmax(Tv_b, 150.0));
    }

    // Copy base state to GPU
    CUDA_CHECK(cudaMemcpy(state.theta_base, theta_base_h, nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.p_base, p_base_h, nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.rho_base, rho_base_h, nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.qv_base, qv_base_h, nz * sizeof(double), cudaMemcpyHostToDevice));

    memset(halo_buf_rt, 0, n3d_h * sizeof(real_t));
    real_t* rho_ref_rt = new real_t[n3d_h];
    memset(rho_ref_rt, 0, n3d_h * sizeof(real_t));

    if (have_pressure_trailer) {
        double* theta_ref_col = new double[nz];
        double* qv_ref_col = new double[nz];
        double* z_ref_col = new double[nz];
        double* p_ref_col = new double[nz];

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double terrain = clamp_loaded_terrain_host(terrain_host[idx2(i, j, nx)], grid.ztop);
                double column_depth = fmax(grid.ztop - terrain, 1.0);

                for (int k = 0; k < nz; k++) {
                    z_ref_col[k] = terrain + grid.eta_m[k] * column_depth;
                    theta_ref_col[k] = interpolate_profile_linear_host(
                        z_levels_h, theta_base_h, nz, z_ref_col[k]
                    );
                    qv_ref_col[k] = fmax(interpolate_profile_linear_host(
                        z_levels_h, qv_base_h, nz, z_ref_col[k]
                    ), 0.0);
                }

                if (use_terrain_aware_reference) {
                    p_ref_col[nz - 1] = interpolate_profile_log_host(
                        z_levels_h, p_base_h, nz, z_ref_col[nz - 1]
                    );
                    for (int k = nz - 2; k >= 0; --k) {
                        double dz = fmax(z_ref_col[k + 1] - z_ref_col[k], 1.0);
                        double qv_layer = fmax(0.5 * (qv_ref_col[k] + qv_ref_col[k + 1]), 0.0);
                        double p_guess = p_ref_col[k + 1] * 1.05;
                        for (int iter = 0; iter < 6; ++iter) {
                            double p_mid = sqrt(fmax(p_guess * p_ref_col[k + 1], 1.0));
                            double t_low = theta_ref_col[k] * pow(p_mid / P0, KAPPA);
                            double t_high = theta_ref_col[k + 1] * pow(p_mid / P0, KAPPA);
                            double tv_layer = 0.5 * (t_low + t_high) * (1.0 + 0.61 * qv_layer);
                            p_guess = p_ref_col[k + 1] * exp(G * dz / (R_D * fmax(tv_layer, 150.0)));
                        }
                        p_ref_col[k] = p_guess;
                    }
                } else {
                    for (int k = 0; k < nz; ++k) {
                        p_ref_col[k] = interpolate_profile_log_host(
                            z_levels_h, p_base_h, nz, z_ref_col[k]
                        );
                    }
                }

                for (int k = 0; k < nz; k++) {
                    int src = k * ny * nx + j * nx + i;
                    int dst = k * ny_h * nx_h + (j + 2) * nx_h + (i + 2);
                    halo_buf_rt[dst] = (real_t)(pressure_h[src] - p_ref_col[k]);

                    double t_ref = theta_ref_col[k] * pow(fmax(p_ref_col[k], 1.0) / P0, KAPPA);
                    double tv_ref = t_ref * (1.0 + 0.61 * qv_ref_col[k]);
                    rho_ref_rt[dst] = (real_t)(p_ref_col[k] / (R_D * fmax(tv_ref, 150.0)));
                }
            }
        }

        delete[] theta_ref_col;
        delete[] qv_ref_col;
        delete[] z_ref_col;
        delete[] p_ref_col;
    } else {
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny_h; j++) {
                for (int i = 0; i < nx_h; i++) {
                    int idx = k * ny_h * nx_h + j * nx_h + i;
                    rho_ref_rt[idx] = (real_t)rho_base_h[k];
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(state.p, halo_buf_rt, n3d_h * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.rho, rho_ref_rt, n3d_h * sizeof(real_t), cudaMemcpyHostToDevice));

    // Apply boundary conditions to fill halos
    apply_boundary_conditions(state, grid);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  Base state: theta_base[0]=%.2f K, theta_base[%d]=%.2f K\n",
           theta_base_h[0], nz - 1, theta_base_h[nz - 1]);
    printf("  Base state: p_base[0]=%.0f Pa, rho_base[0]=%.4f kg/m3\n",
           p_base_h[0], rho_base_h[0]);
    printf("  GFS binary load complete.\n");

    // Cleanup
    delete[] z_levels_h;
    delete[] z_w;
    delete[] field_buf;
    delete[] halo_buf;
    delete[] halo_buf_rt;
    delete[] theta_base_h;
    delete[] p_base_h;
    delete[] rho_base_h;
    delete[] qv_base_h;
    delete[] theta_gpu_h;
    delete[] qv_gpu_h;
    delete[] pressure_h;
    delete[] rho_ref_rt;
    delete[] terrain_host;

    return true;
}

} // namespace gpuwm
