// ============================================================
// GPU-WM: Compressible Dynamics Core with Prognostic Pressure
//
// Solves the fully compressible Euler equations with RK3 time
// integration. Pressure is prognostic, evolving via:
//   dp'/dt = -rho0 * cs^2 * div(u)
//
// The pressure gradient force is included in the momentum
// tendencies along with buoyancy, advection, Coriolis, and
// diffusion. All terms are integrated together in the RK3 scheme.
//
// To avoid the acoustic CFL restriction (cs*dt/dx < 1), a
// reduced effective sound speed is used for the pressure update
// while maintaining the correct pressure gradient force. This is
// the "pseudo-incompressible" approach that filters acoustic
// waves while retaining gravity waves and convective dynamics.
// ============================================================

#include "../../include/constants.cuh"
#include "../../include/grid.cuh"
#include "../../include/stability_control.cuh"

namespace gpuwm {

void apply_open_boundaries(StateGPU& state, StateGPU& state_init,
                           const GridConfig& grid, int relax_width);

static constexpr double ACOUSTIC_SMDIV = 0.10;

namespace {

constexpr int W_TRANSPORT_DIAG_SUM_ABS_OLD = 0;
constexpr int W_TRANSPORT_DIAG_SUM_ABS_NEW = 1;
constexpr int W_TRANSPORT_DIAG_SUM_ABS_DELTA = 2;
constexpr int W_TRANSPORT_DIAG_SUM_DELTA = 3;
constexpr int W_TRANSPORT_DIAG_SUM_DIV = 4;
constexpr int W_TRANSPORT_DIAG_SUM_DELTA_DIV = 5;
constexpr int W_TRANSPORT_DIAG_SUM_DELTA2 = 6;
constexpr int W_TRANSPORT_DIAG_SUM_DIV2 = 7;
constexpr int W_TRANSPORT_DIAG_SAMPLES = 8;
constexpr int W_TRANSPORT_DIAG_COUNT = 9;

double* g_w_transport_diag_device = nullptr;
double g_w_transport_tendency_calls = 0.0;

double* ensure_w_transport_diag_buffer() {
    if (!g_w_transport_diag_device) {
        CUDA_CHECK(cudaMalloc(&g_w_transport_diag_device, W_TRANSPORT_DIAG_COUNT * sizeof(double)));
        CUDA_CHECK(cudaMemset(g_w_transport_diag_device, 0, W_TRANSPORT_DIAG_COUNT * sizeof(double)));
    }
    return g_w_transport_diag_device;
}

} // namespace

// ----------------------------------------------------------
// 3rd-order upwind-biased derivative (double precision compute)
// ----------------------------------------------------------
__device__ double advect_3rd(double vel,
    double fm2, double fm1, double f0, double fp1, double fp2,
    double ds)
{
    if (vel > 0.0) {
        return vel * (fm2 - 6.0*fm1 + 3.0*f0 + 2.0*fp1) / (6.0 * ds);
    } else {
        return vel * (-2.0*fm1 - 3.0*f0 + 6.0*fp1 - fp2) / (6.0 * ds);
    }
}

__device__ double upwind_flux(double vel, double f_m, double f_c, double f_p, double ds) {
    if (vel > 0.0) {
        return vel * (f_c - f_m) / ds;
    } else {
        return vel * (f_p - f_c) / ds;
    }
}

__device__ double upwind_face_flux(double vel, double left_state, double right_state) {
    return vel > 0.0 ? vel * left_state : vel * right_state;
}

__device__ inline double clamped_column_terrain(
    const real_t* __restrict__ terrain,
    int i, int j, int nx,
    double ztop
) {
    double terrain_val = (double)terrain[idx2(i, j, nx)];
    return fmin(terrain_val, ztop - 1.0);
}

__device__ inline double local_mass_level_height(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k, int nx,
    double ztop
) {
    double terrain_val = clamped_column_terrain(terrain, i, j, nx, ztop);
    return terrain_following_height(terrain_val, eta_m[k], ztop);
}

__device__ inline double local_column_depth(
    const real_t* __restrict__ terrain,
    int i, int j, int nx, int ny,
    double ztop
) {
    int ii = max(0, min(i, nx - 1));
    int jj = max(0, min(j, ny - 1));
    return fmax(ztop - clamped_column_terrain(terrain, ii, jj, nx, ztop), 1.0);
}

__device__ inline double local_centered_mass_dz(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k, int nx,
    double ztop
) {
    double terrain_val = clamped_column_terrain(terrain, i, j, nx, ztop);
    return fmax(
        0.5 * terrain_following_layer_thickness(
            terrain_val, eta_m[k - 1], eta_m[k + 1], ztop
        ),
        1.0
    );
}

__device__ inline double local_mass_center_spacing(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k_if, int nx,
    double ztop
) {
    double terrain_val = clamped_column_terrain(terrain, i, j, nx, ztop);
    return fmax(
        terrain_following_layer_thickness(
            terrain_val, eta_m[k_if - 1], eta_m[k_if], ztop
        ),
        1.0
    );
}

__device__ inline double local_mass_cell_thickness(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_w,
    int i, int j, int k, int nx,
    double ztop
) {
    double terrain_val = clamped_column_terrain(terrain, i, j, nx, ztop);
    return fmax(
        terrain_following_layer_thickness(
            terrain_val, eta_w[k], eta_w[k + 1], ztop
        ),
        1.0
    );
}

__device__ inline double local_centered_interface_dz(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_w,
    int i, int j, int k_if, int nx,
    double ztop
) {
    double terrain_val = clamped_column_terrain(terrain, i, j, nx, ztop);
    return fmax(
        0.5 * terrain_following_layer_thickness(
            terrain_val, eta_w[k_if - 1], eta_w[k_if + 1], ztop
        ),
        1.0
    );
}

// Half-level dz for upwind vertical fluxes: distance between mass level k
// and mass level k-1 (lower interface) or k+1 (upper interface).
// The upwind scheme selects f[k]-f[k-1] when w>0 and f[k+1]-f[k] when w<0,
// so the correct denominator is the actual distance between those two mass
// levels, NOT the centered average used for centered differences.
__device__ inline double local_upwind_dz(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k, int nz, int nx,
    double ztop, double vel
) {
    double terrain_val = clamped_column_terrain(terrain, i, j, nx, ztop);
    if (vel > 0.0 && k > 0) {
        // flux comes from below: distance from mass level k-1 to k
        return fmax(
            terrain_following_height(terrain_val, eta_m[k], ztop) -
            terrain_following_height(terrain_val, eta_m[k - 1], ztop),
            1.0
        );
    } else if (vel <= 0.0 && k < nz - 1) {
        // flux comes from above: distance from mass level k to k+1
        return fmax(
            terrain_following_height(terrain_val, eta_m[k + 1], ztop) -
            terrain_following_height(terrain_val, eta_m[k], ztop),
            1.0
        );
    }
    // fallback for boundaries
    return fmax(
        local_centered_mass_dz(terrain, eta_m, i, j,
                               max(1, min(k, nz - 2)), nx, ztop),
        1.0
    );
}

__device__ inline double local_upwind_dz_interface(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_w,
    int i, int j, int k_if, int nz, int nx,
    double ztop, double vel
) {
    if (vel > 0.0 && k_if > 0) {
        return local_mass_cell_thickness(terrain, eta_w, i, j, k_if - 1, nx, ztop);
    } else if (vel <= 0.0 && k_if < nz) {
        return local_mass_cell_thickness(terrain, eta_w, i, j, k_if, nx, ztop);
    }

    return local_mass_cell_thickness(
        terrain, eta_w, i, j, max(0, min(k_if, nz - 1)), nx, ztop
    );
}

__device__ inline double sample_terrain_clamped(
    const real_t* __restrict__ terrain,
    int i, int j, int nx, int ny,
    double ztop
) {
    int ii = max(0, min(i, nx - 1));
    int jj = max(0, min(j, ny - 1));
    return clamped_column_terrain(terrain, ii, jj, nx, ztop);
}

__device__ inline double local_terrain_slope_x(
    const real_t* __restrict__ terrain,
    int i, int j, int nx, int ny,
    double dx_eff, double ztop
) {
    int i_m = max(i - 1, 0);
    int i_p = min(i + 1, nx - 1);
    double h_m = sample_terrain_clamped(terrain, i_m, j, nx, ny, ztop);
    double h_p = sample_terrain_clamped(terrain, i_p, j, nx, ny, ztop);
    double ds = max((i_p - i_m) * dx_eff, 1.0);
    return (h_p - h_m) / ds;
}

__device__ inline double local_terrain_slope_y(
    const real_t* __restrict__ terrain,
    int i, int j, int nx, int ny,
    double dy_eff, double ztop
) {
    int j_m = max(j - 1, 0);
    int j_p = min(j + 1, ny - 1);
    double h_m = sample_terrain_clamped(terrain, i, j_m, nx, ny, ztop);
    double h_p = sample_terrain_clamped(terrain, i, j_p, nx, ny, ztop);
    double ds = max((j_p - j_m) * dy_eff, 1.0);
    return (h_p - h_m) / ds;
}

__device__ inline double local_metric_slope_x(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k, int nx, int ny,
    double dx_eff, double ztop
) {
    return (1.0 - eta_m[k]) * local_terrain_slope_x(terrain, i, j, nx, ny, dx_eff, ztop);
}

__device__ inline double local_metric_slope_y(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k, int nx, int ny,
    double dy_eff, double ztop
) {
    return (1.0 - eta_m[k]) * local_terrain_slope_y(terrain, i, j, nx, ny, dy_eff, ztop);
}

__device__ inline double local_metric_slope_x_at_eta(
    const real_t* __restrict__ terrain,
    double eta_value,
    int i, int j, int nx, int ny,
    double dx_eff, double ztop
) {
    return (1.0 - eta_value) * local_terrain_slope_x(terrain, i, j, nx, ny, dx_eff, ztop);
}

__device__ inline double local_metric_slope_y_at_eta(
    const real_t* __restrict__ terrain,
    double eta_value,
    int i, int j, int nx, int ny,
    double dy_eff, double ztop
) {
    return (1.0 - eta_value) * local_terrain_slope_y(terrain, i, j, nx, ny, dy_eff, ztop);
}

__device__ inline double mass_field_at_interface(
    const real_t* __restrict__ field,
    int i, int j, int k_if, int nz,
    int nx_h, int ny_h
) {
    int k_lower = max(0, min(k_if - 1, nz - 1));
    int k_upper = max(0, min(k_if, nz - 1));
    return 0.5 * (
        (double)field[idx3(i, j, k_lower, nx_h, ny_h)] +
        (double)field[idx3(i, j, k_upper, nx_h, ny_h)]
    );
}

__device__ inline double interface_field_at_mass_level(
    const real_t* __restrict__ field,
    int i, int j, int k,
    int nx_h, int ny_h
) {
    return 0.5 * (
        (double)field[idx3w(i, j, k, nx_h, ny_h)] +
        (double)field[idx3w(i, j, k + 1, nx_h, ny_h)]
    );
}

__device__ inline double physical_vertical_velocity_at_interface(
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ omega,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_w,
    const double* __restrict__ mapfac_m,
    int i, int j, int k_if,
    int nx, int ny, int nz,
    int nx_h, int ny_h,
    double dx, double dy, double ztop
) {
    double mapfac = mapfac_m ? mapfac_m[max(0, min(j, ny - 1))] : 1.0;
    double dx_eff = dx / mapfac;
    double dy_eff = dy / mapfac;
    double eta_if = eta_w[k_if];
    double u_if = mass_field_at_interface(u, i, j, k_if, nz, nx_h, ny_h);
    double v_if = mass_field_at_interface(v, i, j, k_if, nz, nx_h, ny_h);
    double zx = local_metric_slope_x_at_eta(terrain, eta_if, i, j, nx, ny, dx_eff, ztop);
    double zy = local_metric_slope_y_at_eta(terrain, eta_if, i, j, nx, ny, dy_eff, ztop);
    return (double)omega[idx3w(i, j, k_if, nx_h, ny_h)] + u_if * zx + v_if * zy;
}

__device__ inline double centered_vertical_derivative(
    const real_t* __restrict__ field,
    int i, int j, int k,
    int nx_h, int ny_h,
    double dz_half
) {
    return ((double)field[idx3(i, j, k + 1, nx_h, ny_h)] -
            (double)field[idx3(i, j, k - 1, nx_h, ny_h)]) / (2.0 * dz_half);
}

__device__ inline double reference_profile_at_local_height(
    const double* __restrict__ profile,
    const double* __restrict__ z_levels,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k,
    int nx, int ny, int nz,
    double ztop
) {
    if (nz <= 1) return profile[0];

    double terrain_val = sample_terrain_clamped(terrain, i, j, nx, ny, ztop);
    double z_local = terrain_following_height(terrain_val, eta_m[k], ztop);
    if (z_local <= z_levels[0]) {
        double dz = fmax(z_levels[1] - z_levels[0], 1.0);
        return profile[0] + (z_local - z_levels[0]) * (profile[1] - profile[0]) / dz;
    }
    if (z_local >= z_levels[nz - 1]) {
        double dz = fmax(z_levels[nz - 1] - z_levels[nz - 2], 1.0);
        return profile[nz - 1] + (z_local - z_levels[nz - 1]) *
            (profile[nz - 1] - profile[nz - 2]) / dz;
    }

    int lo = 0;
    int hi = nz - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (z_levels[mid] <= z_local) lo = mid;
        else hi = mid;
    }

    double frac = (z_local - z_levels[lo]) / fmax(z_levels[hi] - z_levels[lo], 1.0);
    return profile[lo] + frac * (profile[hi] - profile[lo]);
}

__device__ inline double reference_density_at_local_height(
    const double* __restrict__ theta_base,
    const double* __restrict__ p_base,
    const double* __restrict__ qv_base,
    const double* __restrict__ z_levels,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k,
    int nx, int ny, int nz,
    double ztop
) {
    double theta_ref = reference_profile_at_local_height(
        theta_base, z_levels, terrain, eta_m, i, j, k, nx, ny, nz, ztop
    );
    double p_ref = reference_profile_at_local_height(
        p_base, z_levels, terrain, eta_m, i, j, k, nx, ny, nz, ztop
    );
    double qv_ref = reference_profile_at_local_height(
        qv_base, z_levels, terrain, eta_m, i, j, k, nx, ny, nz, ztop
    );
    double t_ref = theta_ref * pow(fmax(p_ref, 1.0) / P0, KAPPA);
    double tv_ref = t_ref * (1.0 + (R_V / R_D - 1.0) * fmax(qv_ref, 0.0));
    return fmax(p_ref / (R_D * fmax(tv_ref, 150.0)), 1.0e-6);
}

__device__ inline double reference_density_from_field(
    const real_t* __restrict__ rho_ref,
    int i, int j, int k,
    int nx_h, int ny_h
) {
    return fmax((double)rho_ref[idx3(i, j, k, nx_h, ny_h)], 1.0e-6);
}

__device__ inline double reference_density_at_interface_from_field(
    const real_t* __restrict__ rho_ref,
    int i, int j, int k_if, int nz,
    int nx_h, int ny_h
) {
    int k_clamped = max(1, min(k_if, nz - 1));
    double rho_lower = reference_density_from_field(rho_ref, i, j, k_clamped - 1, nx_h, ny_h);
    double rho_upper = reference_density_from_field(rho_ref, i, j, k_clamped, nx_h, ny_h);
    return fmax(0.5 * (rho_lower + rho_upper), 1.0e-6);
}

__device__ inline double vertical_derivative_mass_field(
    const real_t* __restrict__ field,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k,
    int nx, int nz, int nx_h, int ny_h,
    double ztop
) {
    if (nz <= 1) return 0.0;

    if (k <= 0) {
        double dz = fmax(
            local_mass_level_height(terrain, eta_m, i, j, 1, nx, ztop) -
            local_mass_level_height(terrain, eta_m, i, j, 0, nx, ztop),
            1.0
        );
        return ((double)field[idx3(i, j, 1, nx_h, ny_h)] -
                (double)field[idx3(i, j, 0, nx_h, ny_h)]) / dz;
    }

    if (k >= nz - 1) {
        double dz = fmax(
            local_mass_level_height(terrain, eta_m, i, j, nz - 1, nx, ztop) -
            local_mass_level_height(terrain, eta_m, i, j, nz - 2, nx, ztop),
            1.0
        );
        return ((double)field[idx3(i, j, nz - 1, nx_h, ny_h)] -
                (double)field[idx3(i, j, nz - 2, nx_h, ny_h)]) / dz;
    }

    // Use proper non-uniform centered difference (2nd-order accurate on
    // stretched grids). The naive (f[k+1]-f[k-1])/(z[k+1]-z[k-1]) is only
    // 1st-order when dz_above != dz_below.
    double z_above  = local_mass_level_height(terrain, eta_m, i, j, k + 1, nx, ztop);
    double z_center = local_mass_level_height(terrain, eta_m, i, j, k,     nx, ztop);
    double z_below  = local_mass_level_height(terrain, eta_m, i, j, k - 1, nx, ztop);
    double dz_p = fmax(z_above - z_center, 0.5);   // dz above
    double dz_m = fmax(z_center - z_below, 0.5);   // dz below
    double f_p = (double)field[idx3(i, j, k + 1, nx_h, ny_h)];
    double f_c = (double)field[idx3(i, j, k,     nx_h, ny_h)];
    double f_m = (double)field[idx3(i, j, k - 1, nx_h, ny_h)];
    // Standard 2nd-order non-uniform finite difference:
    //   df/dz = (f_p*dz_m^2 + f_c*(dz_p^2-dz_m^2) - f_m*dz_p^2)
    //           / (dz_p * dz_m * (dz_p + dz_m))
    return (f_p * dz_m * dz_m + f_c * (dz_p * dz_p - dz_m * dz_m) - f_m * dz_p * dz_p)
           / (dz_p * dz_m * (dz_p + dz_m));
}

__device__ inline double physical_vertical_velocity_from_contravariant(
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ w_contra,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k,
    int nx, int ny, int nx_h, int ny_h,
    double dx_eff, double dy_eff, double ztop
) {
    double zx = local_metric_slope_x(terrain, eta_m, i, j, k, nx, ny, dx_eff, ztop);
    double zy = local_metric_slope_y(terrain, eta_m, i, j, k, nx, ny, dy_eff, ztop);
    int ijk = idx3(i, j, k, nx_h, ny_h);
    return (double)w_contra[idx3(i,j,k,nx_h,ny_h)] + (double)u[ijk] * zx + (double)v[ijk] * zy;
}

__device__ inline double generalized_horizontal_divergence(
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ terrain,
    int i, int j, int k,
    int nx, int ny, int nx_h, int ny_h,
    double dx_eff, double dy_eff, double ztop
) {
    double h_c = local_column_depth(terrain, i, j, nx, ny, ztop);
    double h_ip = local_column_depth(terrain, i + 1, j, nx, ny, ztop);
    double h_im = local_column_depth(terrain, i - 1, j, nx, ny, ztop);
    double h_jp = local_column_depth(terrain, i, j + 1, nx, ny, ztop);
    double h_jm = local_column_depth(terrain, i, j - 1, nx, ny, ztop);

    double flux_x = (h_ip * (double)u[idx3(i + 1, j, k, nx_h, ny_h)] -
                     h_im * (double)u[idx3(i - 1, j, k, nx_h, ny_h)]) / (2.0 * dx_eff);
    double flux_y = (h_jp * (double)v[idx3(i, j + 1, k, nx_h, ny_h)] -
                     h_jm * (double)v[idx3(i, j - 1, k, nx_h, ny_h)]) / (2.0 * dy_eff);
    return (flux_x + flux_y) / h_c;
}

__global__ void flow_control_metrics_kernel(
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ w,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ mapfac_m,
    double* __restrict__ metrics,
    int nx, int ny, int nz,
    double dx, double dy, double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k < 1 || k >= nz - 1) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    double mapfac = mapfac_m ? mapfac_m[j] : 1.0;
    double dx_eff = dx / mapfac;
    double dy_eff = dy / mapfac;
    double dz_half = local_centered_mass_dz(terrain, eta_m, i, j, k, nx, ztop);
    double hdiv = generalized_horizontal_divergence(
        u, v, terrain, i, j, k, nx, ny, nx_h, ny_h, dx_eff, dy_eff, ztop
    );
    double dwdz = ((double)w[idx3(i,j,k+1,nx_h,ny_h)] - (double)w[idx3(i,j,k-1,nx_h,ny_h)]) / (2.0 * dz_half);
    double div_u = hdiv + dwdz;

    double dudz = centered_vertical_derivative(u, i, j, k, nx_h, ny_h, dz_half);
    double dvdz = centered_vertical_derivative(v, i, j, k, nx_h, ny_h, dz_half);
    double w_ip = physical_vertical_velocity_from_contravariant(
        u, v, w, terrain, eta_m, i + 1, j, k, nx, ny, nx_h, ny_h, dx_eff, dy_eff, ztop
    );
    double w_im = physical_vertical_velocity_from_contravariant(
        u, v, w, terrain, eta_m, i - 1, j, k, nx, ny, nx_h, ny_h, dx_eff, dy_eff, ztop
    );
    double w_jp = physical_vertical_velocity_from_contravariant(
        u, v, w, terrain, eta_m, i, j + 1, k, nx, ny, nx_h, ny_h, dx_eff, dy_eff, ztop
    );
    double w_jm = physical_vertical_velocity_from_contravariant(
        u, v, w, terrain, eta_m, i, j - 1, k, nx, ny, nx_h, ny_h, dx_eff, dy_eff, ztop
    );
    double w_kp = physical_vertical_velocity_from_contravariant(
        u, v, w, terrain, eta_m, i, j, k + 1, nx, ny, nx_h, ny_h, dx_eff, dy_eff, ztop
    );
    double w_km = physical_vertical_velocity_from_contravariant(
        u, v, w, terrain, eta_m, i, j, k - 1, nx, ny, nx_h, ny_h, dx_eff, dy_eff, ztop
    );
    double dudy = ((double)u[idx3(i, j + 1, k, nx_h, ny_h)] - (double)u[idx3(i, j - 1, k, nx_h, ny_h)]) / (2.0 * dy_eff);
    double dvdx = ((double)v[idx3(i + 1, j, k, nx_h, ny_h)] - (double)v[idx3(i - 1, j, k, nx_h, ny_h)]) / (2.0 * dx_eff);
    double dwdx = (w_ip - w_im) / (2.0 * dx_eff);
    double dwdy = (w_jp - w_jm) / (2.0 * dy_eff);
    double dwdz_phys = (w_kp - w_km) / (2.0 * dz_half);

    double vort_x = dwdy - dvdz;
    double vort_y = dudz - dwdx;
    double vort_z = dvdx - dudy;
    double vort_mag = sqrt(vort_x * vort_x + vort_y * vort_y + vort_z * vort_z);
    double abs_div = fabs(div_u);
    double abs_hdiv = fabs(hdiv);
    double abs_dwdz = fabs(dwdz);
    (void)dwdz_phys;

    atomicAdd(&metrics[0], abs_div);
    atomicAdd(&metrics[1], vort_mag);
    atomicAdd(&metrics[2], vort_mag * vort_mag);
    atomicMax((unsigned long long*)&metrics[3], __double_as_longlong(abs_div));
    atomicMax((unsigned long long*)&metrics[4], __double_as_longlong(vort_mag));
    atomicAdd(&metrics[5], 1.0);
    atomicAdd(&metrics[6], div_u);
    atomicAdd(&metrics[7], hdiv);
    atomicAdd(&metrics[8], dwdz);
    atomicMax((unsigned long long*)&metrics[9], __double_as_longlong(abs_hdiv));
    atomicMax((unsigned long long*)&metrics[10], __double_as_longlong(abs_dwdz));
}

FlowControlMetrics compute_flow_control_metrics(const StateGPU& state, const GridConfig& grid) {
    FlowControlMetrics result;
    if (grid.nz < 3) return result;

    double zero[11] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double* d_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_metrics, sizeof(zero)));
    CUDA_CHECK(cudaMemcpy(d_metrics, zero, sizeof(zero), cudaMemcpyHostToDevice));

    dim3 block(8, 8, 4);
    dim3 grid3d((grid.nx + 7) / 8, (grid.ny + 7) / 8, (grid.nz + 3) / 4);
    flow_control_metrics_kernel<<<grid3d, block>>>(
        state.u, state.v, state.w, state.terrain, state.eta_m, grid.mapfac_m, d_metrics,
        grid.nx, grid.ny, grid.nz, grid.dx, grid.dy, grid.ztop
    );

    double host_metrics[11];
    CUDA_CHECK(cudaMemcpy(host_metrics, d_metrics, sizeof(host_metrics), cudaMemcpyDeviceToHost));
    cudaFree(d_metrics);

    double count = fmax(host_metrics[5], 1.0);
    result.mean_abs_div = host_metrics[0] / count;
    result.mean_abs_vort = host_metrics[1] / count;
    result.mean_vort2 = host_metrics[2] / count;
    result.max_abs_div = host_metrics[3];
    result.max_abs_vort = host_metrics[4];
    result.mean_div = host_metrics[6] / count;
    result.mean_hdiv = host_metrics[7] / count;
    result.mean_dwdz = host_metrics[8] / count;
    result.max_abs_hdiv = host_metrics[9];
    result.max_abs_dwdz = host_metrics[10];
    return result;
}

void reset_w_transport_diagnostics() {
    g_w_transport_tendency_calls = 0.0;
    if (g_w_transport_diag_device) {
        CUDA_CHECK(cudaMemset(g_w_transport_diag_device, 0, W_TRANSPORT_DIAG_COUNT * sizeof(double)));
    }
}

WTransportDiagnostics consume_w_transport_diagnostics() {
    WTransportDiagnostics result;
    result.tendency_calls = g_w_transport_tendency_calls;

    if (!g_w_transport_diag_device || g_w_transport_tendency_calls <= 0.0) {
        return result;
    }

    double host_stats[W_TRANSPORT_DIAG_COUNT];
    CUDA_CHECK(cudaMemcpy(host_stats, g_w_transport_diag_device,
                          sizeof(host_stats), cudaMemcpyDeviceToHost));

    double samples = fmax(host_stats[W_TRANSPORT_DIAG_SAMPLES], 0.0);
    result.samples = samples;
    if (samples > 0.0) {
        result.mean_abs_old_total = host_stats[W_TRANSPORT_DIAG_SUM_ABS_OLD] / samples;
        result.mean_abs_new_total = host_stats[W_TRANSPORT_DIAG_SUM_ABS_NEW] / samples;
        result.mean_abs_delta = host_stats[W_TRANSPORT_DIAG_SUM_ABS_DELTA] / samples;
        result.mean_delta = host_stats[W_TRANSPORT_DIAG_SUM_DELTA] / samples;
        result.mean_divergence = host_stats[W_TRANSPORT_DIAG_SUM_DIV] / samples;
        result.rms_delta = sqrt(host_stats[W_TRANSPORT_DIAG_SUM_DELTA2] / samples);
        result.rms_divergence = sqrt(host_stats[W_TRANSPORT_DIAG_SUM_DIV2] / samples);
        double denom = sqrt(host_stats[W_TRANSPORT_DIAG_SUM_DELTA2] *
                            host_stats[W_TRANSPORT_DIAG_SUM_DIV2]);
        if (denom > 1.0e-20) {
            result.delta_div_correlation = host_stats[W_TRANSPORT_DIAG_SUM_DELTA_DIV] / denom;
        }
    }

    reset_w_transport_diagnostics();
    return result;
}

// ----------------------------------------------------------
// Advection kernel for momentum
// ----------------------------------------------------------
__global__ void advection_momentum_kernel(
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ w,
    real_t* __restrict__ u_tend,
    real_t* __restrict__ v_tend,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ mapfac_m,
    int nx, int ny, int nz,
    double dx, double dy,
    double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k < 1 || k >= nz - 1) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);

    double u_c = (double)u[ijk], v_c = (double)v[ijk];
    double w_mass = interface_field_at_mass_level(w, i, j, k, nx_h, ny_h);
    double dz_upwind = local_upwind_dz(terrain, eta_m, i, j, k, nz, nx, ztop, w_mass);
    double mapfac = mapfac_m ? mapfac_m[j] : 1.0;
    double dx_eff = dx / mapfac;
    double dy_eff = dy / mapfac;

    #define F(field, di, dj, dk) (double)field[idx3(i+(di), j+(dj), k+(dk), nx_h, ny_h)]

    {
        double ax = advect_3rd(u_c, F(u,-2,0,0), F(u,-1,0,0), F(u,0,0,0), F(u,1,0,0), F(u,2,0,0), dx_eff);
        double ay = advect_3rd(v_c, F(u,0,-2,0), F(u,0,-1,0), F(u,0,0,0), F(u,0,1,0), F(u,0,2,0), dy_eff);
        double az = upwind_flux(w_mass, F(u,0,0,-1), F(u,0,0,0), F(u,0,0,1), dz_upwind);
        u_tend[ijk] = (real_t)((double)u_tend[ijk] - (ax + ay + az));
    }
    {
        double ax = advect_3rd(u_c, F(v,-2,0,0), F(v,-1,0,0), F(v,0,0,0), F(v,1,0,0), F(v,2,0,0), dx_eff);
        double ay = advect_3rd(v_c, F(v,0,-2,0), F(v,0,-1,0), F(v,0,0,0), F(v,0,1,0), F(v,0,2,0), dy_eff);
        double az = upwind_flux(w_mass, F(v,0,0,-1), F(v,0,0,0), F(v,0,0,1), dz_upwind);
        v_tend[ijk] = (real_t)((double)v_tend[ijk] - (ax + ay + az));
    }
    #undef F
}

__global__ void advection_w_interface_kernel(
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ w,
    real_t* __restrict__ w_tend,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_w,
    const double* __restrict__ mapfac_m,
    double w_transport_blend,
    double* __restrict__ transport_stats,
    int nx, int ny, int nz,
    double dx, double dy,
    double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k <= 0 || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk_w = idx3w(i, j, k, nx_h, ny_h);
    double mapfac = mapfac_m ? mapfac_m[j] : 1.0;
    double dx_eff = dx / mapfac;
    double dy_eff = dy / mapfac;
    double u_if = mass_field_at_interface(u, i, j, k, nz, nx_h, ny_h);
    double v_if = mass_field_at_interface(v, i, j, k, nz, nx_h, ny_h);
    double w_if = (double)w[ijk_w];
    double h_c = local_column_depth(terrain, i, j, nx, ny, ztop);
    double w_phys_c = physical_vertical_velocity_at_interface(
        u, v, w, terrain, eta_w, mapfac_m, i, j, k, nx, ny, nz, nx_h, ny_h, dx, dy, ztop
    );
    double h_im = local_column_depth(terrain, i - 1, j, nx, ny, ztop);
    double h_ip = local_column_depth(terrain, i + 1, j, nx, ny, ztop);
    double h_jm = local_column_depth(terrain, i, j - 1, nx, ny, ztop);
    double h_jp = local_column_depth(terrain, i, j + 1, nx, ny, ztop);
    double w_phys_im = physical_vertical_velocity_at_interface(
        u, v, w, terrain, eta_w, mapfac_m, i - 1, j, k, nx, ny, nz, nx_h, ny_h, dx, dy, ztop
    );
    double w_phys_ip = physical_vertical_velocity_at_interface(
        u, v, w, terrain, eta_w, mapfac_m, i + 1, j, k, nx, ny, nz, nx_h, ny_h, dx, dy, ztop
    );
    double w_phys_jm = physical_vertical_velocity_at_interface(
        u, v, w, terrain, eta_w, mapfac_m, i, j - 1, k, nx, ny, nz, nx_h, ny_h, dx, dy, ztop
    );
    double w_phys_jp = physical_vertical_velocity_at_interface(
        u, v, w, terrain, eta_w, mapfac_m, i, j + 1, k, nx, ny, nz, nx_h, ny_h, dx, dy, ztop
    );
    double hw_c = h_c * w_phys_c;
    double hw_im = h_im * w_phys_im;
    double hw_ip = h_ip * w_phys_ip;
    double hw_jm = h_jm * w_phys_jm;
    double hw_jp = h_jp * w_phys_jp;

    double u_face_hi = 0.5 * (u_if + mass_field_at_interface(u, i + 1, j, k, nz, nx_h, ny_h));
    double u_face_lo = 0.5 * (mass_field_at_interface(u, i - 1, j, k, nz, nx_h, ny_h) + u_if);
    double v_face_hi = 0.5 * (v_if + mass_field_at_interface(v, i, j + 1, k, nz, nx_h, ny_h));
    double v_face_lo = 0.5 * (mass_field_at_interface(v, i, j - 1, k, nz, nx_h, ny_h) + v_if);

    double fx_hi = upwind_face_flux(u_face_hi, hw_c, hw_ip);
    double fx_lo = upwind_face_flux(u_face_lo, hw_im, hw_c);
    double fy_hi = upwind_face_flux(v_face_hi, hw_c, hw_jp);
    double fy_lo = upwind_face_flux(v_face_lo, hw_jm, hw_c);
    double ax_new = (fx_hi - fx_lo) / (dx_eff * h_c);
    double ay_new = (fy_hi - fy_lo) / (dy_eff * h_c);

    double w_phys_km = physical_vertical_velocity_at_interface(
        u, v, w, terrain, eta_w, mapfac_m, i, j, k - 1, nx, ny, nz, nx_h, ny_h, dx, dy, ztop
    );
    double w_phys_kp = physical_vertical_velocity_at_interface(
        u, v, w, terrain, eta_w, mapfac_m, i, j, k + 1, nx, ny, nz, nx_h, ny_h, dx, dy, ztop
    );
    double omega_hi = 0.5 * (w_if + (double)w[idx3w(i, j, k + 1, nx_h, ny_h)]);
    double omega_lo = 0.5 * ((double)w[idx3w(i, j, k - 1, nx_h, ny_h)] + w_if);
    double fz_hi = upwind_face_flux(omega_hi, w_phys_c, w_phys_kp);
    double fz_lo = upwind_face_flux(omega_lo, w_phys_km, w_phys_c);
    double dz_center = local_centered_interface_dz(terrain, eta_w, i, j, k, nx, ztop);
    double az_new = (fz_hi - fz_lo) / dz_center;

    #define FW(di, dj, dk) (double)w[idx3w(i+(di), j+(dj), k+(dk), nx_h, ny_h)]
    double dz_upwind = local_upwind_dz_interface(terrain, eta_w, i, j, k, nz, nx, ztop, w_if);
    double ax_old = advect_3rd(u_if, FW(-2,0,0), FW(-1,0,0), FW(0,0,0), FW(1,0,0), FW(2,0,0), dx_eff);
    double ay_old = advect_3rd(v_if, FW(0,-2,0), FW(0,-1,0), FW(0,0,0), FW(0,1,0), FW(0,2,0), dy_eff);
    double az_old = upwind_flux(w_if, FW(0,0,-1), FW(0,0,0), FW(0,0,1), dz_upwind);

    double eta_if = eta_w[k];
    double zx_mm = local_metric_slope_x_at_eta(terrain, eta_if, i - 2, j, nx, ny, dx_eff, ztop);
    double zx_m = local_metric_slope_x_at_eta(terrain, eta_if, i - 1, j, nx, ny, dx_eff, ztop);
    double zx_c = local_metric_slope_x_at_eta(terrain, eta_if, i, j, nx, ny, dx_eff, ztop);
    double zx_p = local_metric_slope_x_at_eta(terrain, eta_if, i + 1, j, nx, ny, dx_eff, ztop);
    double zx_pp = local_metric_slope_x_at_eta(terrain, eta_if, i + 2, j, nx, ny, dx_eff, ztop);
    double zx_jmm = local_metric_slope_x_at_eta(terrain, eta_if, i, j - 2, nx, ny, dx_eff, ztop);
    double zx_jm = local_metric_slope_x_at_eta(terrain, eta_if, i, j - 1, nx, ny, dx_eff, ztop);
    double zx_jp = local_metric_slope_x_at_eta(terrain, eta_if, i, j + 1, nx, ny, dx_eff, ztop);
    double zx_jpp = local_metric_slope_x_at_eta(terrain, eta_if, i, j + 2, nx, ny, dx_eff, ztop);
    double zx_km = local_metric_slope_x_at_eta(terrain, eta_w[k - 1], i, j, nx, ny, dx_eff, ztop);
    double zx_kp = local_metric_slope_x_at_eta(terrain, eta_w[k + 1], i, j, nx, ny, dx_eff, ztop);
    double azx = advect_3rd(u_if, zx_mm, zx_m, zx_c, zx_p, zx_pp, dx_eff)
               + advect_3rd(v_if, zx_jmm, zx_jm, zx_c, zx_jp, zx_jpp, dy_eff)
               + upwind_flux(w_if, zx_km, zx_c, zx_kp, dz_upwind);

    double zy_mm = local_metric_slope_y_at_eta(terrain, eta_if, i - 2, j, nx, ny, dy_eff, ztop);
    double zy_m = local_metric_slope_y_at_eta(terrain, eta_if, i - 1, j, nx, ny, dy_eff, ztop);
    double zy_c = local_metric_slope_y_at_eta(terrain, eta_if, i, j, nx, ny, dy_eff, ztop);
    double zy_p = local_metric_slope_y_at_eta(terrain, eta_if, i + 1, j, nx, ny, dy_eff, ztop);
    double zy_pp = local_metric_slope_y_at_eta(terrain, eta_if, i + 2, j, nx, ny, dy_eff, ztop);
    double zy_jmm = local_metric_slope_y_at_eta(terrain, eta_if, i, j - 2, nx, ny, dy_eff, ztop);
    double zy_jm = local_metric_slope_y_at_eta(terrain, eta_if, i, j - 1, nx, ny, dy_eff, ztop);
    double zy_jp = local_metric_slope_y_at_eta(terrain, eta_if, i, j + 1, nx, ny, dy_eff, ztop);
    double zy_jpp = local_metric_slope_y_at_eta(terrain, eta_if, i, j + 2, nx, ny, dy_eff, ztop);
    double zy_km = local_metric_slope_y_at_eta(terrain, eta_w[k - 1], i, j, nx, ny, dy_eff, ztop);
    double zy_kp = local_metric_slope_y_at_eta(terrain, eta_w[k + 1], i, j, nx, ny, dy_eff, ztop);
    double azy = advect_3rd(u_if, zy_mm, zy_m, zy_c, zy_p, zy_pp, dx_eff)
               + advect_3rd(v_if, zy_jmm, zy_jm, zy_c, zy_jp, zy_jpp, dy_eff)
               + upwind_flux(w_if, zy_km, zy_c, zy_kp, dz_upwind);

    double old_total = ax_old + ay_old + az_old + u_if * azx + v_if * azy;
    double new_total = ax_new + ay_new + az_new;
    double total = (1.0 - w_transport_blend) * old_total
                 + w_transport_blend * new_total;

    if (transport_stats) {
        double dz_lo = local_mass_cell_thickness(terrain, eta_w, i, j, k - 1, nx, ztop);
        double dz_hi = local_mass_cell_thickness(terrain, eta_w, i, j, k, nx, ztop);
        double hdiv_lo = generalized_horizontal_divergence(
            u, v, terrain, i, j, k - 1, nx, ny, nx_h, ny_h, dx_eff, dy_eff, ztop
        );
        double hdiv_hi = generalized_horizontal_divergence(
            u, v, terrain, i, j, k, nx, ny, nx_h, ny_h, dx_eff, dy_eff, ztop
        );
        double div_lo = hdiv_lo +
            (((double)w[idx3w(i, j, k, nx_h, ny_h)] -
              (double)w[idx3w(i, j, k - 1, nx_h, ny_h)]) / dz_lo);
        double div_hi = hdiv_hi +
            (((double)w[idx3w(i, j, k + 1, nx_h, ny_h)] -
              (double)w[idx3w(i, j, k, nx_h, ny_h)]) / dz_hi);
        double div_if = 0.5 * (div_lo + div_hi);
        double delta = new_total - old_total;

        atomicAdd(&transport_stats[W_TRANSPORT_DIAG_SUM_ABS_OLD], fabs(old_total));
        atomicAdd(&transport_stats[W_TRANSPORT_DIAG_SUM_ABS_NEW], fabs(new_total));
        atomicAdd(&transport_stats[W_TRANSPORT_DIAG_SUM_ABS_DELTA], fabs(delta));
        atomicAdd(&transport_stats[W_TRANSPORT_DIAG_SUM_DELTA], delta);
        atomicAdd(&transport_stats[W_TRANSPORT_DIAG_SUM_DIV], div_if);
        atomicAdd(&transport_stats[W_TRANSPORT_DIAG_SUM_DELTA_DIV], delta * div_if);
        atomicAdd(&transport_stats[W_TRANSPORT_DIAG_SUM_DELTA2], delta * delta);
        atomicAdd(&transport_stats[W_TRANSPORT_DIAG_SUM_DIV2], div_if * div_if);
        atomicAdd(&transport_stats[W_TRANSPORT_DIAG_SAMPLES], 1.0);
    }

    w_tend[ijk_w] = (real_t)((double)w_tend[ijk_w] - total);
    #undef FW
}

// ----------------------------------------------------------
// Advection kernel for scalars
// ----------------------------------------------------------
__global__ void advection_scalar_kernel(
    const real_t* __restrict__ scalar,
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ w,
    real_t* __restrict__ scalar_tend,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ mapfac_m,
    int nx, int ny, int nz,
    double dx, double dy,
    double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k < 1 || k >= nz - 1) return;

    int nx_h = nx + 4, ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);
    double mapfac = mapfac_m[j];
    double dx_eff = dx / mapfac;
    double dy_eff = dy / mapfac;
    double u_c = (double)u[ijk];
    double v_c = (double)v[ijk];
    double w_c = interface_field_at_mass_level(w, i, j, k, nx_h, ny_h);

    // Issue 1 fix: direction-aware dz for vertical upwind
    double dz_upwind = local_upwind_dz(terrain, eta_m, i, j, k, nz, nx, ztop, w_c);

    double adv_x = advect_3rd(u_c,
        (double)scalar[idx3(i-2,j,k,nx_h,ny_h)], (double)scalar[idx3(i-1,j,k,nx_h,ny_h)],
        (double)scalar[ijk], (double)scalar[idx3(i+1,j,k,nx_h,ny_h)], (double)scalar[idx3(i+2,j,k,nx_h,ny_h)], dx_eff);
    double adv_y = advect_3rd(v_c,
        (double)scalar[idx3(i,j-2,k,nx_h,ny_h)], (double)scalar[idx3(i,j-1,k,nx_h,ny_h)],
        (double)scalar[ijk], (double)scalar[idx3(i,j+1,k,nx_h,ny_h)], (double)scalar[idx3(i,j+2,k,nx_h,ny_h)], dy_eff);
    double adv_z = upwind_flux(w_c,
        (double)scalar[idx3(i,j,k-1,nx_h,ny_h)], (double)scalar[ijk], (double)scalar[idx3(i,j,k+1,nx_h,ny_h)], dz_upwind);

    scalar_tend[ijk] = (real_t)((double)scalar_tend[ijk] - (adv_x + adv_y + adv_z));
}

// ----------------------------------------------------------
// Buoyancy
// ----------------------------------------------------------
// Buoyancy is evaluated at the interface between mass levels k-1 and k
// by averaging theta (and moisture) from the two adjacent mass levels.
// This is physically correct: w lives conceptually at interfaces even
// though it is stored on the same grid.  For the compressible system
// with prognostic p', the pressure-perturbation buoyancy contribution
// (-(cp*theta0/g) * dp'/dz) is already handled by the vertical
// acoustic pressure-gradient kernel, so no additional p' term is
// needed here.  The moisture coefficient uses the exact value
// (Rv/Rd - 1) = 0.6078 instead of the rounded 0.61.
// ----------------------------------------------------------
__global__ void buoyancy_kernel(
    const real_t* __restrict__ theta,
    const double* __restrict__ theta_base,
    const double* __restrict__ qv_base,
    const double* __restrict__ z_levels,
    const real_t* __restrict__ qv,
    const real_t* __restrict__ qc,
    const real_t* __restrict__ qr,
    real_t* __restrict__ w_tend,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int nx, int ny, int nz,
    double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k < 1 || k >= nz) return;

    int nx_h = nx + 4, ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);

    double th = (double)theta[ijk];

    double th0 = reference_profile_at_local_height(
        theta_base, z_levels, terrain, eta_m, i, j, k, nx, ny, nz, ztop
    );

    double qv_ref = reference_profile_at_local_height(
        qv_base, z_levels, terrain, eta_m, i, j, k, nx, ny, nz, ztop
    );

    double qv_val = fmax((double)qv[ijk], 0.0);
    double qc_val = fmax((double)qc[ijk], 0.0);
    double qr_val = fmax((double)qr[ijk], 0.0);

    // Exact coefficient: (Rv/Rd - 1) = (461.6/287.04 - 1) = 0.6078
    constexpr double RVRD_M1 = R_V / R_D - 1.0;

    double buoy = G * ((th - th0) / fmax(th0, 150.0)
                       + RVRD_M1 * (qv_val - fmax(qv_ref, 0.0))
                       - qc_val
                       - qr_val);

    w_tend[ijk] = (real_t)((double)w_tend[ijk] + buoy);
}

// ----------------------------------------------------------
// Coriolis
// ----------------------------------------------------------
// In a terrain-following coordinate system, the Coriolis force acts
// on the physical (Cartesian) velocity components.  The standard
// f-plane Coriolis terms are:
//   du/dt += f * v           (tends to accelerate u when v > 0)
//   dv/dt += -f * u          (tends to accelerate v when u < 0)
//
// The contravariant vertical velocity (w_contra = deta/dt) is NOT
// directly forced by Coriolis.  In WRF's terrain-following equations,
// the Coriolis contribution to the eta-dot equation arises indirectly
// through the coordinate metric when the horizontal momentum equations
// are transformed.  That metric coupling is already captured by the
// advection kernel's metric_source term (the u*azx + v*azy transport
// of the coordinate slopes).  Adding "-zx*u_cor - zy*v_cor" to w_tend
// here double-counts this effect, creating a spurious vertical forcing
// over sloped terrain.
//
// The cos(lat) Coriolis terms (2*Omega*cos(phi)*w in du/dt and
// -2*Omega*cos(phi)*u in dw/dt) are omitted.  These matter only
// near the equator for deep convective systems.  For mid-latitude
// CONUS domains this is standard practice (WRF also omits them by
// default).
// ----------------------------------------------------------
__global__ void coriolis_kernel(
    const real_t* __restrict__ u, const real_t* __restrict__ v,
    real_t* __restrict__ u_tend, real_t* __restrict__ v_tend,
    const double* __restrict__ coriolis_f,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int nx_h = nx + 4, ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);

    double f = coriolis_f[j];
    double u_val = (double)u[ijk];
    double v_val = (double)v[ijk];

    u_tend[ijk] = (real_t)((double)u_tend[ijk] + f * v_val);
    v_tend[ijk] = (real_t)((double)v_tend[ijk] - f * u_val);
}

// ----------------------------------------------------------
// Diffusion
// ----------------------------------------------------------
__global__ void diffusion_kernel(
    const real_t* __restrict__ field, real_t* __restrict__ tend,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ mapfac_m,
    int nx, int ny, int nz, double dx, double dy,
    double ztop, double kdiff_h, double kdiff_v
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int nx_h = nx + 4, ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);
    double f_c = (double)field[ijk];
    double f_ip = (double)field[idx3(i+1,j,k,nx_h,ny_h)], f_im = (double)field[idx3(i-1,j,k,nx_h,ny_h)];
    double f_jp = (double)field[idx3(i,j+1,k,nx_h,ny_h)], f_jm = (double)field[idx3(i,j-1,k,nx_h,ny_h)];
    double mapfac = mapfac_m ? mapfac_m[j] : 1.0;
    double dx_eff = dx / mapfac;
    double dy_eff = dy / mapfac;
    double result = (double)tend[ijk] + kdiff_h * ((f_ip - 2.0*f_c + f_im)/(dx_eff*dx_eff) + (f_jp - 2.0*f_c + f_jm)/(dy_eff*dy_eff));
    if (k > 0 && k < nz - 1) {
        double f_kp = (double)field[idx3(i,j,k+1,nx_h,ny_h)], f_km = (double)field[idx3(i,j,k-1,nx_h,ny_h)];
        double dz = local_centered_mass_dz(terrain, eta_m, i, j, k, nx, ztop);
        result += kdiff_v * (f_kp - 2.0*f_c + f_km) / (dz * dz);
    }
    tend[ijk] = (real_t)result;
}

__global__ void w_vertical_cfl_damping_kernel(
    const real_t* __restrict__ w,
    real_t* __restrict__ w_tend,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_w,
    int nx, int ny, int nz,
    double ztop, double dt,
    double w_alpha, double w_beta
) {
    // WRF-style vertical velocity damping: only act where the local
    // interface CFL exceeds the activation threshold.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k <= 0 || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk_w = idx3w(i, j, k, nx_h, ny_h);
    double w_val = (double)w[ijk_w];
    double dz = local_centered_interface_dz(terrain, eta_w, i, j, k, nx, ztop);
    double cfl = fabs(w_val) * dt / dz;

    if (cfl <= w_beta) return;

    double damping = copysign(w_alpha * (cfl - w_beta), w_val);
    w_tend[ijk_w] = (real_t)((double)w_tend[ijk_w] - damping);
}

// ----------------------------------------------------------
// Rayleigh damping near model top
// ----------------------------------------------------------
__global__ void rayleigh_damping_kernel(
    real_t* __restrict__ u, real_t* __restrict__ v, real_t* __restrict__ w,
    real_t* __restrict__ theta, real_t* __restrict__ p_pert,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ theta_base,
    const double* __restrict__ z_levels,
    int nx, int ny, int nz, double ztop, double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int nx_h = nx + 4, ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);
    double z = local_mass_level_height(terrain, eta_m, i, j, k, nx, ztop);
    double z_damp = ztop * 0.75;
    if (z <= z_damp) return;
    double frac = (z - z_damp) / (ztop - z_damp);
    double alpha = 0.2 * sin(frac * PI * 0.5) * sin(frac * PI * 0.5);
    double decay = 1.0 / (1.0 + alpha * dt);
    double theta_ref = reference_profile_at_local_height(
        theta_base, z_levels, terrain, eta_m, i, j, k, nx, ny, nz, ztop
    );
    w[ijk] = (real_t)((double)w[ijk] * decay);
    p_pert[ijk] = (real_t)((double)p_pert[ijk] * decay);
    theta[ijk] = (real_t)(theta_ref + ((double)theta[ijk] - theta_ref) * decay);
}

// ----------------------------------------------------------
// Pressure gradient force kernel (hydrostatic-subtraction form)
//
// Adds the horizontal -(1/rho0) * grad_z(p') term to slow momentum tendencies.
// Vertical pressure coupling is handled in a split-explicit acoustic loop.
//
// The terrain-following PG force is:
//   (dp'/dx)_z = (dp'/dx)_eta - zx * dp'/dz
//
// Over steep terrain the metric correction zx * dp'/dz is large (order
// rho*g * terrain_slope) and must nearly cancel with the along-eta derivative
// to leave the small true horizontal gradient. This large-cancellation
// problem is the classic source of spurious PG errors in sigma-coordinate
// models (Janjic 1977, Klemp 2011).
//
// To mitigate this, we use hydrostatic subtraction: note that in hydrostatic
// balance dp/dz = -rho*g, so dp'/dz ~ -rho_ref*g + residual. We rewrite:
//   zx * dp'/dz = zx * (dp'/dz + rho_ref*g) - zx * rho_ref * g
// The first term uses only the non-hydrostatic residual (much smaller),
// dramatically reducing cancellation error over steep terrain.
// ----------------------------------------------------------
__global__ void pressure_gradient_kernel(
    real_t* __restrict__ u_tend, real_t* __restrict__ v_tend, real_t* __restrict__ w_tend,
    const real_t* __restrict__ p_pert,
    const real_t* __restrict__ rho_ref,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ mapfac_m,
    int nx, int ny, int nz, double dx, double dy, double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int nx_h = nx + 4, ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);
    double rho_local = reference_density_from_field(rho_ref, i, j, k, nx_h, ny_h);
    double inv_rho = 1.0 / rho_local;
    double mapfac = mapfac_m[j];
    double dx_eff = dx / mapfac;
    double dy_eff = dy / mapfac;

    // --- Vertical derivative of p' ---
    double dpdz = vertical_derivative_mass_field(
        p_pert, terrain, eta_m, i, j, k, nx, nz, nx_h, ny_h, ztop
    );

    // --- Metric slopes: zx = (1-eta) * dh/dx ---
    double zx = local_metric_slope_x(terrain, eta_m, i, j, k, nx, ny, dx_eff, ztop);
    double zy = local_metric_slope_y(terrain, eta_m, i, j, k, nx, ny, dy_eff, ztop);

    // --- Along-eta horizontal derivatives of p' ---
    double dpdx_eta = ((double)p_pert[idx3(i+1,j,k,nx_h,ny_h)] -
                       (double)p_pert[idx3(i-1,j,k,nx_h,ny_h)]) / (2.0*dx_eff);
    double dpdy_eta = ((double)p_pert[idx3(i,j+1,k,nx_h,ny_h)] -
                       (double)p_pert[idx3(i,j-1,k,nx_h,ny_h)]) / (2.0*dy_eff);

    double dpdx = dpdx_eta - zx * dpdz;
    double dpdy = dpdy_eta - zy * dpdz;

    u_tend[ijk] = (real_t)((double)u_tend[ijk] - inv_rho * dpdx);
    v_tend[ijk] = (real_t)((double)v_tend[ijk] - inv_rho * dpdy);

    // Contravariant w tendency: -(zx * du/dt_PG + zy * dv/dt_PG)
    //   = inv_rho * (zx * dpdx + zy * dpdy)
    w_tend[ijk] = (real_t)((double)w_tend[ijk] + inv_rho * (zx * dpdx + zy * dpdy));
}

__global__ void acoustic_vertical_pg_kernel(
    real_t* __restrict__ w,
    const real_t* __restrict__ p_pert,
    const real_t* __restrict__ rho_ref,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int nx, int ny, int nz,
    double dt, double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k <= 0 || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk_w = idx3w(i, j, k, nx_h, ny_h);

    double dz = local_mass_center_spacing(terrain, eta_m, i, j, k, nx, ztop);
    double grad_p = ((double)p_pert[idx3(i, j, k, nx_h, ny_h)] -
                     (double)p_pert[idx3(i, j, k - 1, nx_h, ny_h)]) / dz;
    double rho_if = reference_density_at_interface_from_field(rho_ref, i, j, k, nz, nx_h, ny_h);

    double w_new = (double)w[ijk_w] - dt * grad_p / rho_if;
    w[ijk_w] = (real_t)w_new;
}

// ----------------------------------------------------------
// Pressure update from divergence (with reduced effective sound speed)
// ----------------------------------------------------------
__global__ void pressure_update_kernel(
    real_t* __restrict__ p_pert,
    const real_t* __restrict__ u, const real_t* __restrict__ v, const real_t* __restrict__ w,
    const real_t* __restrict__ rho_ref,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ eta_w,    // w-level eta values [nz+1]
    const double* __restrict__ mapfac_m,
    int nx, int ny, int nz, double dx, double dy, double dt,
    double cs_eff2,
    double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int nx_h = nx + 4, ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);
    double mapfac = mapfac_m ? mapfac_m[j] : 1.0;
    double dx_eff = dx / mapfac;
    double dy_eff = dy / mapfac;
    double hdiv = generalized_horizontal_divergence(
        u, v, terrain, i, j, k, nx, ny, nx_h, ny_h, dx_eff, dy_eff, ztop
    );

    double dz_cell = local_mass_cell_thickness(terrain, eta_w, i, j, k, nx, ztop);
    double dwdz = ((double)w[idx3w(i, j, k + 1, nx_h, ny_h)] -
                   (double)w[idx3w(i, j, k, nx_h, ny_h)]) / dz_cell;

    double div_u = hdiv + dwdz;
    double rho_ref_cell = reference_density_from_field(rho_ref, i, j, k, nx_h, ny_h);

    // pressure_retain has been removed. The divergence damping filter
    // (ACOUSTIC_SMDIV) is the correct mechanism for controlling acoustic
    // noise. Multiplying dp by a retain factor < 1 every substep was
    // eroding synoptic-scale pressure perturbations over time.
    double dp = -dt * rho_ref_cell * cs_eff2 * div_u;
    p_pert[ijk] = (real_t)((double)p_pert[ijk] + dp);
}

// ----------------------------------------------------------
// Standard kernels
// ----------------------------------------------------------
__global__ void rk3_update_kernel(
    real_t* __restrict__ field, const real_t* __restrict__ field_old,
    const real_t* __restrict__ tend, double dt, double rk_coeff, int n_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    field[idx] = (real_t)((double)field_old[idx] + dt * rk_coeff * (double)tend[idx]);
}

__global__ void acoustic_copy_field_kernel(
    real_t* __restrict__ dst,
    const real_t* __restrict__ src,
    int n_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    dst[idx] = src[idx];
}

__global__ void pressure_divergence_filter_kernel(
    real_t* __restrict__ p_pert,
    real_t* __restrict__ p_prev,
    double smdiv,
    int n_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    double p_now = (double)p_pert[idx];
    double p_old = (double)p_prev[idx];
    // Dampen the split-step pressure increment instead of amplifying it.
    double p_filt = p_now - smdiv * (p_now - p_old);
    p_pert[idx] = (real_t)p_filt;
    p_prev[idx] = (real_t)p_filt;
}

__global__ void sanitize_prognostic_state_kernel(
    real_t* __restrict__ u,
    real_t* __restrict__ v,
    real_t* __restrict__ w,
    real_t* __restrict__ theta,
    real_t* __restrict__ qv,
    real_t* __restrict__ qc,
    real_t* __restrict__ qr,
    real_t* __restrict__ p_pert,
    const double* __restrict__ theta_base,
    const double* __restrict__ p_base,
    const double* __restrict__ z_levels,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int nx, int ny, int nz,
    double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);

    // Sanitize w at mass level k
    double w_val = (double)w[ijk];
    if (!isfinite(w_val)) w_val = 0.0;
    w[ijk] = (real_t)w_val;

    {

        double theta_ref = reference_profile_at_local_height(
            theta_base, z_levels, terrain, eta_m, i, j, k, nx, ny, nz, ztop
        );
        double p_ref = reference_profile_at_local_height(
            p_base, z_levels, terrain, eta_m, i, j, k, nx, ny, nz, ztop
        );
        double p_floor = fmax(100.0, 0.05 * p_ref);

        double u_val = (double)u[ijk];
        double v_val = (double)v[ijk];
        double th_val = (double)theta[ijk];
        double qv_val = (double)qv[ijk];
        double qc_val = (double)qc[ijk];
        double qr_val = (double)qr[ijk];
        double p_val = (double)p_pert[ijk];

        if (!isfinite(u_val)) u_val = 0.0;
        if (!isfinite(v_val)) v_val = 0.0;
        if (!isfinite(th_val)) th_val = theta_ref;
        th_val = fmin(fmax(th_val, 150.0), 700.0);

        if (!isfinite(qv_val)) qv_val = 0.0;
        if (!isfinite(qc_val)) qc_val = 0.0;
        if (!isfinite(qr_val)) qr_val = 0.0;
        qv_val = fmin(fmax(qv_val, 0.0), 0.05);
        qc_val = fmin(fmax(qc_val, 0.0), 0.05);
        qr_val = fmin(fmax(qr_val, 0.0), 0.05);

        double p_full = p_ref + p_val;
        if (!isfinite(p_full) || p_full < p_floor) {
            p_val = p_floor - p_ref;
        }

        u[ijk] = (real_t)u_val;
        v[ijk] = (real_t)v_val;
        theta[ijk] = (real_t)th_val;
        qv[ijk] = (real_t)qv_val;
        qc[ijk] = (real_t)qc_val;
        qr[ijk] = (real_t)qr_val;
        p_pert[ijk] = (real_t)p_val;
    }
}

__global__ void zero_field_kernel(real_t* __restrict__ field, int n_total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    field[idx] = (real_t)0.0;
}

// ----------------------------------------------------------
// Boundary conditions
// ----------------------------------------------------------
__global__ void periodic_bc_x_kernel(real_t* __restrict__ field, int nx, int ny, int nz) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny || k >= nz) return;
    int nx_h = nx + 4, ny_h = ny + 4;
    field[idx3(-2,j,k,nx_h,ny_h)] = field[idx3(nx-2,j,k,nx_h,ny_h)];
    field[idx3(-1,j,k,nx_h,ny_h)] = field[idx3(nx-1,j,k,nx_h,ny_h)];
    field[idx3(nx,  j,k,nx_h,ny_h)] = field[idx3(0,j,k,nx_h,ny_h)];
    field[idx3(nx+1,j,k,nx_h,ny_h)] = field[idx3(1,j,k,nx_h,ny_h)];
}

__global__ void periodic_bc_y_kernel(real_t* __restrict__ field, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || k >= nz) return;
    int nx_h = nx + 4, ny_h = ny + 4;
    field[idx3(i,-2,k,nx_h,ny_h)] = field[idx3(i,ny-2,k,nx_h,ny_h)];
    field[idx3(i,-1,k,nx_h,ny_h)] = field[idx3(i,ny-1,k,nx_h,ny_h)];
    field[idx3(i,ny,  k,nx_h,ny_h)] = field[idx3(i,0,k,nx_h,ny_h)];
    field[idx3(i,ny+1,k,nx_h,ny_h)] = field[idx3(i,1,k,nx_h,ny_h)];
}

// Vertical boundary condition for contravariant w.
//
// In terrain-following coordinates, w stored in StateGPU is the
// CONTRAVARIANT vertical velocity  w_contra = deta/dt,  not the
// Cartesian vertical velocity.  The kinematic boundary condition
// is that no flow crosses coordinate surfaces:
//
//   Surface (eta=0):    deta/dt = 0   ->  w_contra = 0
//   Model top (eta=1):  deta/dt = 0   ->  w_contra = 0
//
// Therefore setting w=0 at k=0 and k=nz is physically correct.
//
// The extra parameters (u, v, terrain, eta_m, mapfac_m, dx, dy,
// ztop) are retained in the signature for ABI compatibility with
// callers.  They are unused because the contravariant BC is zero.
__global__ void bc_w_kernel(
    real_t* __restrict__ w,
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ mapfac_m,
    int nx, int ny, int nz,
    double dx, double dy, double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    // Unused -- contravariant w BC is zero regardless of terrain geometry.
    (void)u; (void)v; (void)terrain; (void)eta_m; (void)mapfac_m;
    (void)dx; (void)dy; (void)ztop;

    int nx_h = nx + 4, ny_h = ny + 4;
    w[idx3w(i, j, 0,  nx_h, ny_h)] = (real_t)0.0;  // surface
    w[idx3w(i, j, nz, nx_h, ny_h)] = (real_t)0.0;  // model top
}

__global__ void convert_w_to_contravariant_kernel(
    real_t* __restrict__ w,
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ mapfac_m,
    int nx, int ny, int nz,
    double dx, double dy, double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int ijk = idx3(i, j, k, nx_h, ny_h);
    double mapfac = mapfac_m ? mapfac_m[j] : 1.0;
    double dx_eff = dx / mapfac;
    double dy_eff = dy / mapfac;
    double zx = (1.0 - eta_m[k]) * local_terrain_slope_x(terrain, i, j, nx, ny, dx_eff, ztop);
    double zy = (1.0 - eta_m[k]) * local_terrain_slope_y(terrain, i, j, nx, ny, dy_eff, ztop);
    w[ijk] = (real_t)((double)w[ijk] - (double)u[ijk] * zx - (double)v[ijk] * zy);
}

__global__ void initialize_w_from_continuity_kernel(
    real_t* __restrict__ w,
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_w,
    const double* __restrict__ mapfac_m,
    double* __restrict__ metrics,
    int nx, int ny, int nz,
    double dx, double dy, double ztop
) {
    enum StartupBalanceMetric {
        METRIC_TOP_RAW_SUM = 0,
        METRIC_TOP_RAW_MAX = 1,
        METRIC_TOP_BAL_SUM = 2,
        METRIC_TOP_BAL_MAX = 3,
        METRIC_DIV_BAL_SUM = 4,
        METRIC_DIV_BAL_MAX = 5,
        METRIC_COLUMN_COUNT = 6,
        METRIC_CELL_COUNT = 7,
        METRIC_HDIV_SUM = 8
    };

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    double mapfac = mapfac_m ? mapfac_m[j] : 1.0;
    double dx_eff = dx / mapfac;
    double dy_eff = dy / mapfac;

    w[idx3w(i, j, 0, nx_h, ny_h)] = (real_t)0.0;

    double w_if = 0.0;
    for (int k = 0; k < nz; ++k) {
        double hdiv = generalized_horizontal_divergence(
            u, v, terrain, i, j, k, nx, ny, nx_h, ny_h, dx_eff, dy_eff, ztop
        );
        double dz_cell = local_mass_cell_thickness(terrain, eta_w, i, j, k, nx, ztop);
        w_if -= hdiv * dz_cell;
        w[idx3w(i, j, k + 1, nx_h, ny_h)] = (real_t)w_if;
    }

    double top_raw = w_if;
    double sum_abs_hdiv = 0.0;
    atomicAdd(&metrics[METRIC_TOP_RAW_SUM], fabs(top_raw));
    atomicMax((unsigned long long*)&metrics[METRIC_TOP_RAW_MAX], __double_as_longlong(fabs(top_raw)));
    atomicAdd(&metrics[METRIC_COLUMN_COUNT], 1.0);

    double eta_lo = eta_w[0];
    double eta_hi = eta_w[nz];
    double eta_span = fabs(eta_hi - eta_lo) > 1.0e-12 ? (eta_hi - eta_lo) : 1.0;
    for (int k_if = 1; k_if < nz; ++k_if) {
        double alpha = (eta_w[k_if] - eta_lo) / eta_span;
        int ijk_w = idx3w(i, j, k_if, nx_h, ny_h);
        w[ijk_w] = (real_t)((double)w[ijk_w] - alpha * top_raw);
    }

    w[idx3w(i, j, 0,  nx_h, ny_h)] = (real_t)0.0;
    w[idx3w(i, j, nz, nx_h, ny_h)] = (real_t)0.0;
    double top_balanced = (double)w[idx3w(i, j, nz, nx_h, ny_h)];
    atomicAdd(&metrics[METRIC_TOP_BAL_SUM], fabs(top_balanced));
    atomicMax((unsigned long long*)&metrics[METRIC_TOP_BAL_MAX], __double_as_longlong(fabs(top_balanced)));

    for (int k = 0; k < nz; ++k) {
        double hdiv = generalized_horizontal_divergence(
            u, v, terrain, i, j, k, nx, ny, nx_h, ny_h, dx_eff, dy_eff, ztop
        );
        sum_abs_hdiv += fabs(hdiv);
        double dz_cell = local_mass_cell_thickness(terrain, eta_w, i, j, k, nx, ztop);
        double dwdz = ((double)w[idx3w(i, j, k + 1, nx_h, ny_h)] -
                       (double)w[idx3w(i, j, k,     nx_h, ny_h)]) / dz_cell;
        double div = hdiv + dwdz;
        atomicAdd(&metrics[METRIC_DIV_BAL_SUM], fabs(div));
        atomicMax((unsigned long long*)&metrics[METRIC_DIV_BAL_MAX], __double_as_longlong(fabs(div)));
        atomicAdd(&metrics[METRIC_CELL_COUNT], 1.0);
    }
    atomicAdd(&metrics[METRIC_HDIV_SUM], sum_abs_hdiv);
}

__global__ void open_fast_bc_x_kernel(real_t* __restrict__ field, int nx, int ny, int nz) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    field[idx3(0, j, k, nx_h, ny_h)] = field[idx3(1, j, k, nx_h, ny_h)];
    field[idx3(-1, j, k, nx_h, ny_h)] = field[idx3(0, j, k, nx_h, ny_h)];
    field[idx3(-2, j, k, nx_h, ny_h)] = field[idx3(0, j, k, nx_h, ny_h)];
    field[idx3(nx - 1, j, k, nx_h, ny_h)] = field[idx3(nx - 2, j, k, nx_h, ny_h)];
    field[idx3(nx, j, k, nx_h, ny_h)] = field[idx3(nx - 1, j, k, nx_h, ny_h)];
    field[idx3(nx + 1, j, k, nx_h, ny_h)] = field[idx3(nx - 1, j, k, nx_h, ny_h)];
}

__global__ void open_fast_bc_y_kernel(real_t* __restrict__ field, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    field[idx3(i, 0, k, nx_h, ny_h)] = field[idx3(i, 1, k, nx_h, ny_h)];
    field[idx3(i, -1, k, nx_h, ny_h)] = field[idx3(i, 0, k, nx_h, ny_h)];
    field[idx3(i, -2, k, nx_h, ny_h)] = field[idx3(i, 0, k, nx_h, ny_h)];
    field[idx3(i, ny - 1, k, nx_h, ny_h)] = field[idx3(i, ny - 2, k, nx_h, ny_h)];
    field[idx3(i, ny, k, nx_h, ny_h)] = field[idx3(i, ny - 1, k, nx_h, ny_h)];
    field[idx3(i, ny + 1, k, nx_h, ny_h)] = field[idx3(i, ny - 1, k, nx_h, ny_h)];
}

// ----------------------------------------------------------
// Host driver
// ----------------------------------------------------------

void apply_boundary_conditions(StateGPU& state, const GridConfig& grid) {
    int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    dim3 block_jk(16,16), block_ik(16,16), block_ij(16,16);
    dim3 grid_jk((ny+15)/16,(nz+15)/16);
    dim3 grid_ik((nx+15)/16,(nz+15)/16);
    dim3 grid_jk_w((ny+15)/16,((nz + 1)+15)/16);
    dim3 grid_ik_w((nx+15)/16,((nz + 1)+15)/16);
    dim3 grid_ij((nx+15)/16,(ny+15)/16);

    real_t* mass_fields[] = {state.u, state.v, state.theta,
                             state.qv, state.qc, state.qr, state.p};
    for (auto* f : mass_fields) {
        periodic_bc_x_kernel<<<grid_jk, block_jk>>>(f, nx, ny, nz);
        periodic_bc_y_kernel<<<grid_ik, block_ik>>>(f, nx, ny, nz);
    }
    periodic_bc_x_kernel<<<grid_jk_w, block_jk>>>(state.w, nx, ny, nz + 1);
    periodic_bc_y_kernel<<<grid_ik_w, block_ik>>>(state.w, nx, ny, nz + 1);

    bc_w_kernel<<<grid_ij, block_ij>>>(
        state.w, state.u, state.v, state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop
    );
}

void apply_stage_boundaries(StateGPU& state, StateGPU& state_init,
                            const GridConfig& grid, bool use_open_bc,
                            int relax_width) {
    if (use_open_bc) {
        apply_open_boundaries(state, state_init, grid, relax_width);
    } else {
        apply_boundary_conditions(state, grid);
    }
}

void refresh_fast_field_boundaries(real_t* field, const GridConfig& grid, bool use_open_bc, int nz_field) {
    int nx = grid.nx;
    int ny = grid.ny;

    dim3 block_jk(16,16), block_ik(16,16);
    dim3 grid_jk((ny + 15) / 16, (nz_field + 15) / 16);
    dim3 grid_ik((nx + 15) / 16, (nz_field + 15) / 16);

    if (use_open_bc) {
        open_fast_bc_x_kernel<<<grid_jk, block_jk>>>(field, nx, ny, nz_field);
        open_fast_bc_y_kernel<<<grid_ik, block_ik>>>(field, nx, ny, nz_field);
    } else {
        periodic_bc_x_kernel<<<grid_jk, block_jk>>>(field, nx, ny, nz_field);
        periodic_bc_y_kernel<<<grid_ik, block_ik>>>(field, nx, ny, nz_field);
    }
}

void run_vertical_acoustic_substeps(
    StateGPU& state,
    const GridConfig& grid,
    double dt_rk,
    int acoustic_substeps,
    double cs_eff2,
    bool use_open_bc
) {
    int nx = grid.nx;
    int ny = grid.ny;
    int nz = grid.nz;
    int nx_h = nx + 4;
    int ny_h = ny + 4;
    int n_total = nx_h * ny_h * nz;

    double dt_ac = dt_rk / acoustic_substeps;

    dim3 block(8,8,4);
    dim3 grid3d((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
    dim3 grid3d_w((nx + 7) / 8, (ny + 7) / 8, ((nz + 1) + 3) / 4);
    dim3 block_ij(16,16);
    dim3 grid_ij((nx + 15) / 16, (ny + 15) / 16);
    int block1d = 256;
    int grid1d = (n_total + block1d - 1) / block1d;

    refresh_fast_field_boundaries(state.p, grid, use_open_bc, nz);
    refresh_fast_field_boundaries(state.w, grid, use_open_bc, nz + 1);
    acoustic_copy_field_kernel<<<grid1d, block1d>>>(state.phi, state.p, n_total);
    bc_w_kernel<<<grid_ij, block_ij>>>(
        state.w, state.u, state.v, state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop
    );

    for (int substep = 0; substep < acoustic_substeps; ++substep) {
        acoustic_vertical_pg_kernel<<<grid3d_w, block>>>(
            state.w, state.p, state.rho,
            state.terrain, state.eta_m,
            nx, ny, nz, 0.5 * dt_ac, grid.ztop
        );
        refresh_fast_field_boundaries(state.w, grid, use_open_bc, nz + 1);
        bc_w_kernel<<<grid_ij, block_ij>>>(
            state.w, state.u, state.v, state.terrain, state.eta_m, grid.mapfac_m,
            nx, ny, nz, grid.dx, grid.dy, grid.ztop
        );

        pressure_update_kernel<<<grid3d, block>>>(
            state.p,
            state.u, state.v, state.w,
            state.rho,
            state.terrain, state.eta_m, state.eta, grid.mapfac_m,
            nx, ny, nz, grid.dx, grid.dy, dt_ac,
            cs_eff2, grid.ztop
        );
        pressure_divergence_filter_kernel<<<grid1d, block1d>>>(
            state.p, state.phi, ACOUSTIC_SMDIV, n_total
        );
        refresh_fast_field_boundaries(state.p, grid, use_open_bc, nz);

        acoustic_vertical_pg_kernel<<<grid3d_w, block>>>(
            state.w, state.p, state.rho,
            state.terrain, state.eta_m,
            nx, ny, nz, 0.5 * dt_ac, grid.ztop
        );
        refresh_fast_field_boundaries(state.w, grid, use_open_bc, nz + 1);
        bc_w_kernel<<<grid_ij, block_ij>>>(
            state.w, state.u, state.v, state.terrain, state.eta_m, grid.mapfac_m,
            nx, ny, nz, grid.dx, grid.dy, grid.ztop
        );
    }
}

void sanitize_prognostic_state(StateGPU& state, const GridConfig& grid) {
    dim3 block(8, 8, 4);
    dim3 grid3d((grid.nx + 7) / 8, (grid.ny + 7) / 8, (grid.nz + 3) / 4);
    sanitize_prognostic_state_kernel<<<grid3d, block>>>(
        state.u, state.v, state.w,
        state.theta, state.qv, state.qc, state.qr, state.p,
        state.theta_base, state.p_base, state.z_levels, state.terrain, state.eta_m,
        grid.nx, grid.ny, grid.nz, grid.ztop
    );
}

void convert_w_to_contravariant(StateGPU& state, const GridConfig& grid) {
    int nx = grid.nx;
    int ny = grid.ny;
    int nz = grid.nz;

    dim3 block(8, 8, 4);
    dim3 grid3d((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
    dim3 block_ij(16, 16);
    dim3 grid_ij((nx + 15) / 16, (ny + 15) / 16);

    convert_w_to_contravariant_kernel<<<grid3d, block>>>(
        state.w, state.u, state.v, state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop
    );
    bc_w_kernel<<<grid_ij, block_ij>>>(
        state.w, state.u, state.v, state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop
    );
}

void initialize_w_from_continuity(StateGPU& state, const GridConfig& grid, const char* label) {
    enum StartupBalanceMetric {
        METRIC_TOP_RAW_SUM = 0,
        METRIC_TOP_RAW_MAX = 1,
        METRIC_TOP_BAL_SUM = 2,
        METRIC_TOP_BAL_MAX = 3,
        METRIC_DIV_BAL_SUM = 4,
        METRIC_DIV_BAL_MAX = 5,
        METRIC_COLUMN_COUNT = 6,
        METRIC_CELL_COUNT = 7,
        METRIC_HDIV_SUM = 8
    };

    int nx = grid.nx;
    int ny = grid.ny;
    int nz = grid.nz;

    dim3 block_ij(16, 16);
    dim3 grid_ij((nx + 15) / 16, (ny + 15) / 16);

    double* d_metrics = nullptr;
    double zero[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    CUDA_CHECK(cudaMalloc(&d_metrics, sizeof(zero)));
    CUDA_CHECK(cudaMemcpy(d_metrics, zero, sizeof(zero), cudaMemcpyHostToDevice));

    initialize_w_from_continuity_kernel<<<grid_ij, block_ij>>>(
        state.w, state.u, state.v, state.terrain, state.eta, grid.mapfac_m, d_metrics,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop
    );
    bc_w_kernel<<<grid_ij, block_ij>>>(
        state.w, state.u, state.v, state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop
    );

    double host_metrics[9];
    CUDA_CHECK(cudaMemcpy(host_metrics, d_metrics, sizeof(host_metrics), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_metrics));

    double column_count = fmax(host_metrics[METRIC_COLUMN_COUNT], 1.0);
    double cell_count = fmax(host_metrics[METRIC_CELL_COUNT], 1.0);
    printf("Startup w-balance [%s]: mean|w_top_raw|=%.4e m/s, mean|w_top_bal|=%.4e m/s, "
           "max|w_top_raw|=%.4e m/s, max|w_top_bal|=%.4e m/s, "
           "mean|hdiv|=%.4e s^-1, mean|div_bal|=%.4e s^-1, max|div_bal|=%.4e s^-1\n",
           label ? label : "state",
           host_metrics[METRIC_TOP_RAW_SUM] / column_count,
           host_metrics[METRIC_TOP_BAL_SUM] / column_count,
           host_metrics[METRIC_TOP_RAW_MAX],
           host_metrics[METRIC_TOP_BAL_MAX],
           host_metrics[METRIC_HDIV_SUM] / cell_count,
           host_metrics[METRIC_DIV_BAL_SUM] / cell_count,
           host_metrics[METRIC_DIV_BAL_MAX]);
}

// ----------------------------------------------------------
// Issue 4 fix: Positive-definite limiter for moisture tendencies.
//
// After computing the advection + diffusion tendencies for moisture fields
// (qv, qc, qr), this kernel limits the tendency so that applying it with
// the given dt cannot drive the field negative.  This is the simplest
// "positive-definite" or "mass-fixer" approach: if field + dt*tend < 0,
// scale tend back so that field + dt*tend = 0.
//
// This avoids the post-hoc clamp in sanitize_prognostic_state_kernel which
// creates artificial mass out of thin air (clamp to zero = mass source).
// The limiter also ensures that any over-depletion tendency is reduced,
// preventing the advection scheme from producing impossible states.
// ----------------------------------------------------------
__global__ void positive_definite_limiter_kernel(
    const real_t* __restrict__ field,
    real_t* __restrict__ tend,
    int n_total,
    double dt_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;

    double f_val = (double)field[idx];
    double t_val = (double)tend[idx];

    // Only limit negative tendencies that would make the field go negative
    if (t_val < 0.0 && f_val >= 0.0) {
        // Maximum allowable negative tendency: field + dt*tend >= 0
        // => tend >= -field/dt
        double tend_floor = -f_val / fmax(dt_max, 1.0e-10);
        if (t_val < tend_floor) {
            tend[idx] = (real_t)tend_floor;
        }
    }
    // If field is already negative (shouldn't happen), zero out further
    // negative tendency to avoid making it worse
    else if (f_val < 0.0 && t_val < 0.0) {
        tend[idx] = (real_t)0.0;
    }
}

void compute_tendencies(StateGPU& state, const GridConfig& grid,
                        double kdiff_h, double kdiff_v, double dt,
                        const StabilityControlConfig& stability_cfg) {
    int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    int nx_h = nx + 4, ny_h = ny + 4;
    int n_total = nx_h * ny_h * nz;
    int n_total_w = nx_h * ny_h * (nz + 1);

    dim3 block(8,8,4);
    dim3 grid3d((nx+7)/8,(ny+7)/8,(nz+3)/4);
    dim3 grid3d_w((nx + 7) / 8, (ny + 7) / 8, ((nz + 1) + 3) / 4);
    int block1d = 256, grid1d = (n_total+255)/256;
    int grid1d_w = (n_total_w + block1d - 1) / block1d;
    double* w_transport_diag = nullptr;
    if (stability_cfg.w_transport_diagnostics) {
        w_transport_diag = ensure_w_transport_diag_buffer();
        g_w_transport_tendency_calls += 1.0;
    }

    zero_field_kernel<<<grid1d, block1d>>>(state.u_tend, n_total);
    zero_field_kernel<<<grid1d, block1d>>>(state.v_tend, n_total);
    zero_field_kernel<<<grid1d_w, block1d>>>(state.w_tend, n_total_w);
    zero_field_kernel<<<grid1d, block1d>>>(state.theta_tend, n_total);
    zero_field_kernel<<<grid1d, block1d>>>(state.qv_tend, n_total);
    zero_field_kernel<<<grid1d, block1d>>>(state.qc_tend, n_total);
    zero_field_kernel<<<grid1d, block1d>>>(state.qr_tend, n_total);

    // Pressure gradient force (from current p')
    pressure_gradient_kernel<<<grid3d, block>>>(
        state.u_tend, state.v_tend, state.w_tend,
        state.p, state.rho,
        state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop);

    // Advection
    advection_momentum_kernel<<<grid3d, block>>>(
        state.u, state.v, state.w,
        state.u_tend, state.v_tend,
        state.terrain, state.eta_m, grid.mapfac_m, nx, ny, nz, grid.dx, grid.dy, grid.ztop);
    advection_w_interface_kernel<<<grid3d_w, block>>>(
        state.u, state.v, state.w, state.w_tend,
        state.terrain, state.eta, grid.mapfac_m,
        stability_cfg.w_transport_blend, w_transport_diag,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop);

    advection_scalar_kernel<<<grid3d, block>>>(
        state.theta, state.u, state.v, state.w, state.theta_tend,
        state.terrain, state.eta_m, grid.mapfac_m, nx, ny, nz, grid.dx, grid.dy, grid.ztop);
    advection_scalar_kernel<<<grid3d, block>>>(
        state.qv, state.u, state.v, state.w, state.qv_tend,
        state.terrain, state.eta_m, grid.mapfac_m, nx, ny, nz, grid.dx, grid.dy, grid.ztop);
    advection_scalar_kernel<<<grid3d, block>>>(
        state.qc, state.u, state.v, state.w, state.qc_tend,
        state.terrain, state.eta_m, grid.mapfac_m, nx, ny, nz, grid.dx, grid.dy, grid.ztop);
    advection_scalar_kernel<<<grid3d, block>>>(
        state.qr, state.u, state.v, state.w, state.qr_tend,
        state.terrain, state.eta_m, grid.mapfac_m, nx, ny, nz, grid.dx, grid.dy, grid.ztop);

    // Buoyancy
    buoyancy_kernel<<<grid3d, block>>>(
        state.theta, state.theta_base, state.qv_base, state.z_levels,
        state.qv, state.qc, state.qr,
        state.w_tend, state.terrain, state.eta_m, nx, ny, nz, grid.ztop);

    // Coriolis (horizontal components only; no direct w forcing)
    coriolis_kernel<<<grid3d, block>>>(
        state.u, state.v, state.u_tend, state.v_tend,
        grid.coriolis_f,
        nx, ny, nz);

    // Diffusion
    diffusion_kernel<<<grid3d, block>>>(state.u, state.u_tend, state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop, kdiff_h, kdiff_v);
    diffusion_kernel<<<grid3d, block>>>(state.v, state.v_tend, state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop, kdiff_h, kdiff_v);
    diffusion_kernel<<<grid3d, block>>>(state.w, state.w_tend, state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop, kdiff_h, kdiff_v * 0.5);
    diffusion_kernel<<<grid3d, block>>>(state.theta, state.theta_tend, state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop, kdiff_h * 0.1, 0.0);

    if (stability_cfg.w_cfl_damping) {
        w_vertical_cfl_damping_kernel<<<grid3d_w, block>>>(
            state.w, state.w_tend, state.terrain, state.eta,
            nx, ny, nz, grid.ztop, dt,
            stability_cfg.w_damping_alpha, stability_cfg.w_damping_beta
        );
    }

    // Issue 4 fix: apply positive-definite limiter to moisture tendencies.
    // This must come AFTER all tendency contributions (advection + diffusion)
    // have been accumulated, so the limiter sees the total tendency.
    positive_definite_limiter_kernel<<<grid1d, block1d>>>(state.qv, state.qv_tend, n_total, dt);
    positive_definite_limiter_kernel<<<grid1d, block1d>>>(state.qc, state.qc_tend, n_total, dt);
    positive_definite_limiter_kernel<<<grid1d, block1d>>>(state.qr, state.qr_tend, n_total, dt);
}

// ----------------------------------------------------------
// RK3 step with prognostic pressure
// ----------------------------------------------------------
void rk3_step(StateGPU& state, StateGPU& state_old, StateGPU& state_init,
              const GridConfig& grid, double dt, double kdiff, int rk_stage,
              bool use_open_bc, const StabilityControlConfig& stability_cfg,
              int relax_width) {
    double rk_coeffs[] = {1.0/3.0, 1.0/2.0, 1.0};
    double rk_coeff = rk_coeffs[rk_stage];
    double dt_rk = dt * rk_coeff;

    int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    int nx_h = nx + 4, ny_h = ny + 4;
    int n_total = nx_h * ny_h * nz;
    int n_total_w = nx_h * ny_h * (nz + 1);
    int block1d = 256, grid1d = (n_total+255)/256;
    int grid1d_w = (n_total_w + block1d - 1) / block1d;

    dim3 block(8,8,4);
    dim3 grid3d((nx+7)/8,(ny+7)/8,(nz+3)/4);

    // Ensure the current RK state has the correct halos before stencil work.
    apply_stage_boundaries(state, state_init, grid, use_open_bc, relax_width);

    AdaptiveStabilityState adaptive_state;
    if (stability_cfg.enabled) {
        FlowControlMetrics metrics = compute_flow_control_metrics(state, grid);
        adaptive_state = evaluate_adaptive_stability(stability_cfg, metrics, dt_rk);
    }

    // 1. Compute tendencies
    double kdiff_h_eff = kdiff * adaptive_state.kdiff_scale;
    double kdiff_v_eff = kdiff * 0.1 * sqrt(adaptive_state.kdiff_scale);
    compute_tendencies(state, grid, kdiff_h_eff, kdiff_v_eff, dt_rk, stability_cfg);

    // 2. RK3 update
    rk3_update_kernel<<<grid1d, block1d>>>(state.u, state_old.u, state.u_tend, dt, rk_coeff, n_total);
    rk3_update_kernel<<<grid1d, block1d>>>(state.v, state_old.v, state.v_tend, dt, rk_coeff, n_total);
    rk3_update_kernel<<<grid1d_w, block1d>>>(state.w, state_old.w, state.w_tend, dt, rk_coeff, n_total_w);
    rk3_update_kernel<<<grid1d, block1d>>>(state.theta, state_old.theta, state.theta_tend, dt, rk_coeff, n_total);
    rk3_update_kernel<<<grid1d, block1d>>>(state.qv, state_old.qv, state.qv_tend, dt, rk_coeff, n_total);
    rk3_update_kernel<<<grid1d, block1d>>>(state.qc, state_old.qc, state.qc_tend, dt, rk_coeff, n_total);
    rk3_update_kernel<<<grid1d, block1d>>>(state.qr, state_old.qr, state.qr_tend, dt, rk_coeff, n_total);
    sanitize_prognostic_state(state, grid);

    // Apply BCs
    apply_stage_boundaries(state, state_init, grid, use_open_bc, relax_width);

    // 3. Update pressure and vertical velocity in a split-explicit acoustic loop
    //
    // This model uses a REDUCED effective sound speed to filter acoustic
    // waves while retaining gravity waves and convective dynamics (the
    // "pseudo-incompressible" approach). The effective sound speed is set
    // to a fraction of the physical sound speed, and the number of acoustic
    // substeps ensures the CFL of this reduced speed is < 1.
    //
    // cs_eff ~ 50-80 m/s for 3-4km grids (vs physical cs ~ 340 m/s).
    // This damps acoustic modes rapidly while preserving meteorologically
    // relevant dynamics. The number of substeps is then small (1-3).
    //
    double dz_min = grid.ztop / (double)max(nz, 1);
    if (grid.eta) {
        for (int k = 0; k < nz; ++k) {
            dz_min = fmin(dz_min, (grid.eta[k + 1] - grid.eta[k]) * grid.ztop);
        }
    }
    dz_min = fmax(dz_min, 1.0);
    double ds_min = fmin(grid.dx, dz_min);
    int acoustic_substeps = (int)ceil(dt_rk / 4.0);
    if (acoustic_substeps < 1) acoustic_substeps = 1;
    double dt_ac = dt_rk / acoustic_substeps;
    double cs_eff = 0.4 * ds_min / dt_ac;
    double cs_eff2 = cs_eff * cs_eff;

    run_vertical_acoustic_substeps(
        state, grid, dt_rk, acoustic_substeps, cs_eff2,
        use_open_bc
    );
    sanitize_prognostic_state(state, grid);

    // 4. Rayleigh damping near model top (critical for stability)
    rayleigh_damping_kernel<<<grid3d, block>>>(
        state.u, state.v, state.w, state.theta,
        state.p, state.terrain, state.eta_m, state.theta_base, state.z_levels,
        nx, ny, nz, grid.ztop, dt_rk
    );

    // Final BCs
    apply_stage_boundaries(state, state_init, grid, use_open_bc, relax_width);
}

} // namespace gpuwm
