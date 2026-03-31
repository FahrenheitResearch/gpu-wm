// ============================================================
// GPU-WM: NetCDF Output
// Writes CF-compliant NetCDF files compatible with standard
// meteorological tools (ncview, GrADS, Python xarray, etc.)
// ============================================================

#include "../../include/constants.cuh"
#include "../../include/grid.cuh"
#include "../../include/projection.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <vector>

#ifdef HAS_NETCDF
#include <netcdf.h>
#define NC_CHECK(e) do { int _r = (e); if (_r != NC_NOERR) { \
    fprintf(stderr, "NetCDF error at %s:%d: %s\n", __FILE__, __LINE__, nc_strerror(_r)); \
    return; } } while(0)
#endif

namespace gpuwm {

static bool format_unix_time_utc(int64_t epoch_seconds,
                                 const char* fmt,
                                 char* out,
                                 size_t out_len) {
    if (!out || out_len == 0) return false;
    time_t tt = (time_t)epoch_seconds;
    struct tm utc_tm{};
    if (gmtime_r(&tt, &utc_tm) == nullptr) return false;
    return strftime(out, out_len, fmt, &utc_tm) > 0;
}

static void build_time_units_string(const GridConfig& grid,
                                    char* units_buf,
                                    size_t units_len) {
    if (!units_buf || units_len == 0) return;
    if (grid.init_valid_time_unix >= 0) {
        char base_buf[32];
        if (format_unix_time_utc(grid.init_valid_time_unix, "%Y-%m-%d %H:%M:%S",
                                 base_buf, sizeof(base_buf))) {
            snprintf(units_buf, units_len, "seconds since %s", base_buf);
            return;
        }
    }
    snprintf(units_buf, units_len, "seconds since init");
}

static void build_wrftime_string(const GridConfig& grid,
                                 double elapsed_seconds,
                                 char* out,
                                 size_t out_len) {
    if (!out || out_len == 0) return;
    if (grid.init_valid_time_unix >= 0) {
        int64_t valid_epoch = grid.init_valid_time_unix + (int64_t)llround(elapsed_seconds);
        if (format_unix_time_utc(valid_epoch, "%Y-%m-%d_%H:%M:%S", out, out_len)) {
            return;
        }
    }

    time_t sim_seconds = (time_t)llround(elapsed_seconds);
    struct tm sim_tm{};
    gmtime_r(&sim_seconds, &sim_tm);
    strftime(out, out_len, "%Y-%m-%d_%H:%M:%S", &sim_tm);
}

static void build_eta_coordinates(const GridConfig& grid,
                                  double*& eta_w_out,
                                  double*& eta_m_out) {
    eta_w_out = new double[grid.nz + 1];
    eta_m_out = new double[grid.nz];

    if (grid.eta && grid.eta_m) {
        memcpy(eta_w_out, grid.eta, (grid.nz + 1) * sizeof(double));
        memcpy(eta_m_out, grid.eta_m, grid.nz * sizeof(double));
        return;
    }

    for (int k = 0; k <= grid.nz; ++k) {
        eta_w_out[k] = (grid.nz > 0) ? (double)k / (double)grid.nz : 0.0;
    }
    for (int k = 0; k < grid.nz; ++k) {
        eta_m_out[k] = 0.5 * (eta_w_out[k] + eta_w_out[k + 1]);
    }
}

__device__ inline double clamped_column_terrain_netcdf(
    const real_t* __restrict__ terrain,
    int i, int j, int nx,
    double ztop
) {
    double terrain_val = (double)terrain[idx2(i, j, nx)];
    return fmin(terrain_val, ztop - 1.0);
}

__device__ inline double sample_terrain_clamped_netcdf(
    const real_t* __restrict__ terrain,
    int i, int j, int nx, int ny,
    double ztop
) {
    int ii = max(0, min(i, nx - 1));
    int jj = max(0, min(j, ny - 1));
    return clamped_column_terrain_netcdf(terrain, ii, jj, nx, ztop);
}

__device__ inline double local_metric_slope_x_netcdf(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k, int nx, int ny,
    double dx_eff, double ztop
) {
    int i_m = max(i - 1, 0);
    int i_p = min(i + 1, nx - 1);
    double h_m = sample_terrain_clamped_netcdf(terrain, i_m, j, nx, ny, ztop);
    double h_p = sample_terrain_clamped_netcdf(terrain, i_p, j, nx, ny, ztop);
    double ds = max((i_p - i_m) * dx_eff, 1.0);
    return (1.0 - eta_m[k]) * (h_p - h_m) / ds;
}

__device__ inline double local_metric_slope_y_netcdf(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k, int nx, int ny,
    double dy_eff, double ztop
) {
    int j_m = max(j - 1, 0);
    int j_p = min(j + 1, ny - 1);
    double h_m = sample_terrain_clamped_netcdf(terrain, i, j_m, nx, ny, ztop);
    double h_p = sample_terrain_clamped_netcdf(terrain, i, j_p, nx, ny, ztop);
    double ds = max((j_p - j_m) * dy_eff, 1.0);
    return (1.0 - eta_m[k]) * (h_p - h_m) / ds;
}

__device__ inline double physical_vertical_velocity_netcdf(
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ w_contra,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ mapfac_m,
    int i, int j, int k,
    int nx, int ny, int nx_h, int ny_h,
    double dx, double dy, double ztop
) {
    double mapfac = mapfac_m ? mapfac_m[j] : 1.0;
    double dx_eff = dx / mapfac;
    double dy_eff = dy / mapfac;
    double zx = local_metric_slope_x_netcdf(terrain, eta_m, i, j, k, nx, ny, dx_eff, ztop);
    double zy = local_metric_slope_y_netcdf(terrain, eta_m, i, j, k, nx, ny, dy_eff, ztop);
    int ijk = idx3(i, j, k, nx_h, ny_h);
    return (double)w_contra[ijk] + (double)u[ijk] * zx + (double)v[ijk] * zy;
}

__global__ void materialize_physical_w_netcdf_kernel(
    real_t* __restrict__ w_phys,
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ w_contra,
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
    w_phys[ijk] = (real_t)physical_vertical_velocity_netcdf(
        u, v, w_contra, terrain, eta_m, mapfac_m,
        i, j, k, nx, ny, nx_h, ny_h, dx, dy, ztop
    );
}

static void compute_terrain_slope_field(const real_t* terrain_host,
                                        int nx, int ny,
                                        double dx, double dy,
                                        float* slope_out) {
    if (!terrain_host || !slope_out || nx <= 0 || ny <= 0 || dx <= 0.0 || dy <= 0.0) {
        return;
    }

    auto terrain_at = [&](int i, int j) -> double {
        return (double)terrain_host[(size_t)j * (size_t)nx + (size_t)i];
    };

    for (int j = 0; j < ny; ++j) {
        int jm = (j > 0) ? (j - 1) : j;
        int jp = (j + 1 < ny) ? (j + 1) : j;
        for (int i = 0; i < nx; ++i) {
            int im = (i > 0) ? (i - 1) : i;
            int ip = (i + 1 < nx) ? (i + 1) : i;

            double dzdx = (terrain_at(ip, j) - terrain_at(im, j)) / ((ip == im) ? dx : ((ip - im) * dx));
            double dzdy = (terrain_at(i, jp) - terrain_at(i, jm)) / ((jp == jm) ? dy : ((jp - jm) * dy));
            slope_out[(size_t)j * (size_t)nx + (size_t)i] = (float)std::sqrt(dzdx * dzdx + dzdy * dzdy);
        }
    }
}

static void compute_terrain_slope_summary(const real_t* terrain_host,
                                          int nx, int ny,
                                          double dx, double dy,
                                          double& mean_slope,
                                          double& rms_slope,
                                          double& max_slope) {
    mean_slope = 0.0;
    rms_slope = 0.0;
    max_slope = 0.0;

    if (!terrain_host || nx <= 0 || ny <= 0 || dx <= 0.0 || dy <= 0.0) {
        return;
    }

    size_t count = 0;
    auto terrain_at = [&](int i, int j) -> double {
        return (double)terrain_host[(size_t)j * (size_t)nx + (size_t)i];
    };

    for (int j = 0; j < ny; ++j) {
        int jm = (j > 0) ? (j - 1) : j;
        int jp = (j + 1 < ny) ? (j + 1) : j;
        for (int i = 0; i < nx; ++i) {
            int im = (i > 0) ? (i - 1) : i;
            int ip = (i + 1 < nx) ? (i + 1) : i;

            double dzdx = (terrain_at(ip, j) - terrain_at(im, j)) / ((ip == im) ? dx : ((ip - im) * dx));
            double dzdy = (terrain_at(i, jp) - terrain_at(i, jm)) / ((jp == jm) ? dy : ((jp - jm) * dy));
            double slope = std::sqrt(dzdx * dzdx + dzdy * dzdy);

            mean_slope += slope;
            rms_slope += slope * slope;
            max_slope = std::fmax(max_slope, slope);
            ++count;
        }
    }

    if (count > 0) {
        mean_slope /= (double)count;
        rms_slope = std::sqrt(rms_slope / (double)count);
    }
}

static inline double clamp_terrain_host(double terrain, double ztop) {
    return fmin(terrain, ztop - 1.0);
}

static inline double reference_profile_at_local_height_host(
    const double* profile,
    const double* z_levels,
    int nz,
    double terrain,
    double eta,
    double ztop
) {
    if (nz <= 1) return profile[0];

    double z_local = terrain_following_height(clamp_terrain_host(terrain, ztop), eta, ztop);
    if (z_local <= z_levels[0]) {
        double dz = std::fmax(z_levels[1] - z_levels[0], 1.0);
        return profile[0] + (z_local - z_levels[0]) * (profile[1] - profile[0]) / dz;
    }
    if (z_local >= z_levels[nz - 1]) {
        double dz = std::fmax(z_levels[nz - 1] - z_levels[nz - 2], 1.0);
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

    double frac = (z_local - z_levels[lo]) / std::fmax(z_levels[hi] - z_levels[lo], 1.0);
    return profile[lo] + frac * (profile[hi] - profile[lo]);
}

static inline double saturation_vapor_pressure_liquid(double temperature_k) {
    double temp_c = temperature_k - 273.15;
    return 611.2 * exp(17.67 * temp_c / (temp_c + 243.5));
}

static inline double vapor_epsilon() {
    return R_D / R_V;
}

#ifdef HAS_NETCDF
void write_netcdf(const StateGPU& state, const GridConfig& grid,
                  const LambertConformal& proj,
                  double time, int output_num) {
    char filename[256];
    snprintf(filename, sizeof(filename), "output/gpuwm_%06d.nc", output_num);

    int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    int nx_h = nx + 4, ny_h = ny + 4;
    size_t n2d = (size_t)nx * ny;
    size_t n3d = (size_t)nx * ny * nz;
    size_t n3d_h = (size_t)nx_h * ny_h * nz;
    size_t n3d_u = (size_t)(nx + 1) * ny * nz;
    size_t n3d_v = (size_t)nx * (ny + 1) * nz;
    size_t n3d_w = (size_t)nx * ny * (nz + 1);

    // Create NetCDF file
    int ncid;
    NC_CHECK(nc_create(filename, NC_CLOBBER | NC_NETCDF4, &ncid));

    // Dimensions
    int dim_x, dim_y, dim_z, dim_t;
    int dim_eta, dim_eta_w;
    NC_CHECK(nc_def_dim(ncid, "x", nx, &dim_x));
    NC_CHECK(nc_def_dim(ncid, "y", ny, &dim_y));
    NC_CHECK(nc_def_dim(ncid, "z", nz, &dim_z));
    NC_CHECK(nc_def_dim(ncid, "eta", nz, &dim_eta));
    NC_CHECK(nc_def_dim(ncid, "eta_w", nz + 1, &dim_eta_w));
    NC_CHECK(nc_def_dim(ncid, "time", 1, &dim_t));

    int dim_time_wrf, dim_datestr;
    int dim_we, dim_sn, dim_bt, dim_we_stag, dim_sn_stag, dim_bt_stag;
    NC_CHECK(nc_def_dim(ncid, "Time", 1, &dim_time_wrf));
    NC_CHECK(nc_def_dim(ncid, "DateStrLen", 19, &dim_datestr));
    NC_CHECK(nc_def_dim(ncid, "west_east", nx, &dim_we));
    NC_CHECK(nc_def_dim(ncid, "south_north", ny, &dim_sn));
    NC_CHECK(nc_def_dim(ncid, "bottom_top", nz, &dim_bt));
    NC_CHECK(nc_def_dim(ncid, "west_east_stag", nx + 1, &dim_we_stag));
    NC_CHECK(nc_def_dim(ncid, "south_north_stag", ny + 1, &dim_sn_stag));
    NC_CHECK(nc_def_dim(ncid, "bottom_top_stag", nz + 1, &dim_bt_stag));

    // Coordinate variables
    int var_x, var_y, var_z, var_eta, var_eta_w, var_time, var_lat, var_lon, var_terrain, var_terrain_slope;
    NC_CHECK(nc_def_var(ncid, "x", NC_DOUBLE, 1, &dim_x, &var_x));
    NC_CHECK(nc_def_var(ncid, "y", NC_DOUBLE, 1, &dim_y, &var_y));
    NC_CHECK(nc_def_var(ncid, "z", NC_DOUBLE, 1, &dim_z, &var_z));
    NC_CHECK(nc_def_var(ncid, "eta", NC_DOUBLE, 1, &dim_eta, &var_eta));
    NC_CHECK(nc_def_var(ncid, "eta_w", NC_DOUBLE, 1, &dim_eta_w, &var_eta_w));
    NC_CHECK(nc_def_var(ncid, "time", NC_DOUBLE, 1, &dim_t, &var_time));

    int dims_2d[] = {dim_y, dim_x};
    NC_CHECK(nc_def_var(ncid, "lat", NC_DOUBLE, 2, dims_2d, &var_lat));
    NC_CHECK(nc_def_var(ncid, "lon", NC_DOUBLE, 2, dims_2d, &var_lon));
    NC_CHECK(nc_def_var(ncid, "TERRAIN", NC_FLOAT, 2, dims_2d, &var_terrain));
    NC_CHECK(nc_def_var(ncid, "TERRAIN_SLOPE", NC_FLOAT, 2, dims_2d, &var_terrain_slope));

    // Native mass-grid variables [z, y, x]
    int dims_3d[] = {dim_z, dim_y, dim_x};
    int var_u_mass, var_v_mass, var_w_mass, var_theta, var_qv, var_qc, var_qr;
    int var_pressure, var_temp, var_rho, var_rh;

    NC_CHECK(nc_def_var(ncid, "U_MASS", NC_FLOAT, 3, dims_3d, &var_u_mass));
    NC_CHECK(nc_def_var(ncid, "V_MASS", NC_FLOAT, 3, dims_3d, &var_v_mass));
    NC_CHECK(nc_def_var(ncid, "W_MASS", NC_FLOAT, 3, dims_3d, &var_w_mass));
    NC_CHECK(nc_def_var(ncid, "THETA", NC_FLOAT, 3, dims_3d, &var_theta));
    NC_CHECK(nc_def_var(ncid, "QV", NC_FLOAT, 3, dims_3d, &var_qv));
    NC_CHECK(nc_def_var(ncid, "QC", NC_FLOAT, 3, dims_3d, &var_qc));
    NC_CHECK(nc_def_var(ncid, "QR", NC_FLOAT, 3, dims_3d, &var_qr));
    NC_CHECK(nc_def_var(ncid, "PRESSURE", NC_FLOAT, 3, dims_3d, &var_pressure));
    NC_CHECK(nc_def_var(ncid, "TEMP", NC_FLOAT, 3, dims_3d, &var_temp));
    NC_CHECK(nc_def_var(ncid, "RHO", NC_FLOAT, 3, dims_3d, &var_rho));
    NC_CHECK(nc_def_var(ncid, "RH", NC_FLOAT, 3, dims_3d, &var_rh));

    // WRF-compatible fields
    int dims_time_char[] = {dim_time_wrf, dim_datestr};
    int dims_time_2d[] = {dim_time_wrf, dim_sn, dim_we};
    int dims_time_3d[] = {dim_time_wrf, dim_bt, dim_sn, dim_we};
    int dims_time_u[] = {dim_time_wrf, dim_bt, dim_sn, dim_we_stag};
    int dims_time_v[] = {dim_time_wrf, dim_bt, dim_sn_stag, dim_we};
    int dims_time_w[] = {dim_time_wrf, dim_bt_stag, dim_sn, dim_we};

    int var_times, var_xlat, var_xlong, var_hgt, var_mapfac_m, var_cosalpha, var_sinalpha;
    int var_psfc, var_t2, var_q2, var_u10, var_v10;
    int var_p, var_pb, var_ph, var_phb, var_t_wrf, var_qvapor, var_qcloud, var_qrain;
    int var_u_wrf, var_v_wrf, var_w_wrf;

    NC_CHECK(nc_def_var(ncid, "Times", NC_CHAR, 2, dims_time_char, &var_times));
    NC_CHECK(nc_def_var(ncid, "XLAT", NC_FLOAT, 3, dims_time_2d, &var_xlat));
    NC_CHECK(nc_def_var(ncid, "XLONG", NC_FLOAT, 3, dims_time_2d, &var_xlong));
    NC_CHECK(nc_def_var(ncid, "HGT", NC_FLOAT, 3, dims_time_2d, &var_hgt));
    NC_CHECK(nc_def_var(ncid, "MAPFAC_M", NC_FLOAT, 3, dims_time_2d, &var_mapfac_m));
    NC_CHECK(nc_def_var(ncid, "COSALPHA", NC_FLOAT, 3, dims_time_2d, &var_cosalpha));
    NC_CHECK(nc_def_var(ncid, "SINALPHA", NC_FLOAT, 3, dims_time_2d, &var_sinalpha));
    NC_CHECK(nc_def_var(ncid, "PSFC", NC_FLOAT, 3, dims_time_2d, &var_psfc));
    NC_CHECK(nc_def_var(ncid, "T2", NC_FLOAT, 3, dims_time_2d, &var_t2));
    NC_CHECK(nc_def_var(ncid, "Q2", NC_FLOAT, 3, dims_time_2d, &var_q2));
    NC_CHECK(nc_def_var(ncid, "U10", NC_FLOAT, 3, dims_time_2d, &var_u10));
    NC_CHECK(nc_def_var(ncid, "V10", NC_FLOAT, 3, dims_time_2d, &var_v10));
    NC_CHECK(nc_def_var(ncid, "P", NC_FLOAT, 4, dims_time_3d, &var_p));
    NC_CHECK(nc_def_var(ncid, "PB", NC_FLOAT, 4, dims_time_3d, &var_pb));
    NC_CHECK(nc_def_var(ncid, "PH", NC_FLOAT, 4, dims_time_w, &var_ph));
    NC_CHECK(nc_def_var(ncid, "PHB", NC_FLOAT, 4, dims_time_w, &var_phb));
    NC_CHECK(nc_def_var(ncid, "T", NC_FLOAT, 4, dims_time_3d, &var_t_wrf));
    NC_CHECK(nc_def_var(ncid, "QVAPOR", NC_FLOAT, 4, dims_time_3d, &var_qvapor));
    NC_CHECK(nc_def_var(ncid, "QCLOUD", NC_FLOAT, 4, dims_time_3d, &var_qcloud));
    NC_CHECK(nc_def_var(ncid, "QRAIN", NC_FLOAT, 4, dims_time_3d, &var_qrain));
    NC_CHECK(nc_def_var(ncid, "U", NC_FLOAT, 4, dims_time_u, &var_u_wrf));
    NC_CHECK(nc_def_var(ncid, "V", NC_FLOAT, 4, dims_time_v, &var_v_wrf));
    NC_CHECK(nc_def_var(ncid, "W", NC_FLOAT, 4, dims_time_w, &var_w_wrf));

    // Enable compression
    NC_CHECK(nc_def_var_deflate(ncid, var_u_mass, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_v_mass, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_w_mass, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_theta, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_qv, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_qc, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_qr, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_pressure, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_temp, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_rho, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_rh, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_p, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_pb, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_ph, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_phb, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_t_wrf, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_qvapor, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_qcloud, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_qrain, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_u_wrf, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_v_wrf, 0, 1, 4));
    NC_CHECK(nc_def_var_deflate(ncid, var_w_wrf, 0, 1, 4));

    // Attributes
    nc_put_att_text(ncid, NC_GLOBAL, "title", 26, "GPU-WM Weather Model Output");
    nc_put_att_text(ncid, NC_GLOBAL, "Conventions", 6, "CF-1.8");
    nc_put_att_text(ncid, NC_GLOBAL, "model", 6, "GPU-WM");
    nc_put_att_text(ncid, NC_GLOBAL, "vertical_coordinate",
                    strlen("terrain-following eta"), "terrain-following eta");
    nc_put_att_text(ncid, NC_GLOBAL, "vertical_coordinate_note",
                    strlen("z is retained as geometric height; eta and eta_w are auxiliary coordinates"),
                    "z is retained as geometric height; eta and eta_w are auxiliary coordinates");
    nc_put_att_text(ncid, NC_GLOBAL, "terrain_metric_note",
                    strlen("TERRAIN_SLOPE is a diagnostic derived from centered terrain height differences"),
                    "TERRAIN_SLOPE is a diagnostic derived from centered terrain height differences");
    nc_put_att_text(ncid, NC_GLOBAL, "wrf_compatibility_note",
                    strlen("WRF-compatible fields use synthesized staggering and lowest-model-level surface approximations"),
                    "WRF-compatible fields use synthesized staggering and lowest-model-level surface approximations");
    nc_put_att_text(ncid, NC_GLOBAL, "times_note",
                    strlen("Times stores UTC valid time from init metadata when available; otherwise it falls back to epoch plus elapsed seconds"),
                    "Times stores UTC valid time from init metadata when available; otherwise it falls back to epoch plus elapsed seconds");

    nc_put_att_text(ncid, var_u_mass, "units", 3, "m/s");
    nc_put_att_text(ncid, var_u_mass, "long_name", strlen("mass-grid x-wind component"),
                    "mass-grid x-wind component");
    nc_put_att_text(ncid, var_v_mass, "units", 3, "m/s");
    nc_put_att_text(ncid, var_v_mass, "long_name", strlen("mass-grid y-wind component"),
                    "mass-grid y-wind component");
    nc_put_att_text(ncid, var_w_mass, "units", 3, "m/s");
    nc_put_att_text(ncid, var_w_mass, "long_name", strlen("physical vertical velocity on mass levels"),
                    "physical vertical velocity on mass levels");
    nc_put_att_text(ncid, var_theta, "units", 1, "K");
    nc_put_att_text(ncid, var_theta, "long_name", 21, "potential temperature");
    nc_put_att_text(ncid, var_qv, "units", 5, "kg/kg");
    nc_put_att_text(ncid, var_qv, "long_name", 22, "water vapor mixing rat");
    nc_put_att_text(ncid, var_qc, "units", 5, "kg/kg");
    nc_put_att_text(ncid, var_qc, "long_name", 22, "cloud water mixing rat");
    nc_put_att_text(ncid, var_qr, "units", 5, "kg/kg");
    nc_put_att_text(ncid, var_qr, "long_name", 20, "rain water mixing rat");
    nc_put_att_text(ncid, var_pressure, "units", 2, "Pa");
    nc_put_att_text(ncid, var_pressure, "long_name", 13, "full pressure");
    nc_put_att_text(ncid, var_temp, "units", 1, "K");
    nc_put_att_text(ncid, var_temp, "long_name", 11, "temperature");
    nc_put_att_text(ncid, var_rho, "units", 6, "kg/m^3");
    nc_put_att_text(ncid, var_rho, "long_name", 7, "density");
    nc_put_att_text(ncid, var_rh, "units", 1, "%");
    nc_put_att_text(ncid, var_rh, "long_name", 17, "relative humidity");

    nc_put_att_text(ncid, var_z, "units", 1, "m");
    nc_put_att_text(ncid, var_z, "long_name", strlen("geometric height of mass levels"),
                    "geometric height of mass levels");
    nc_put_att_text(ncid, var_eta, "units", 1, "1");
    nc_put_att_text(ncid, var_eta, "long_name", strlen("terrain-following eta coordinate"),
                    "terrain-following eta coordinate");
    nc_put_att_text(ncid, var_eta_w, "units", 1, "1");
    nc_put_att_text(ncid, var_eta_w, "long_name",
                    strlen("terrain-following eta coordinate at w levels"),
                    "terrain-following eta coordinate at w levels");
    char time_units[64];
    build_time_units_string(grid, time_units, sizeof(time_units));
    nc_put_att_text(ncid, var_time, "units", strlen(time_units), time_units);
    nc_put_att_text(ncid, var_time, "calendar", 8, "standard");
    nc_put_att_text(ncid, NC_GLOBAL, "time_zone", 3, "UTC");

    char start_date[20];
    build_wrftime_string(grid, 0.0, start_date, sizeof(start_date));
    nc_put_att_text(ncid, NC_GLOBAL, "START_DATE", strlen(start_date), start_date);
    nc_put_att_text(ncid, NC_GLOBAL, "SIMULATION_START_DATE", strlen(start_date), start_date);
    if (grid.init_valid_time_unix >= 0) {
        nc_put_att_text(ncid, NC_GLOBAL, "source_valid_time", strlen(start_date), start_date);
    }
    if (grid.init_reference_time_unix >= 0) {
        char ref_date[20];
        if (format_unix_time_utc(grid.init_reference_time_unix, "%Y-%m-%d_%H:%M:%S",
                                 ref_date, sizeof(ref_date))) {
            nc_put_att_text(ncid, NC_GLOBAL, "source_reference_time", strlen(ref_date), ref_date);
        }
        nc_put_att_int(ncid, NC_GLOBAL, "source_forecast_hour", NC_INT, 1, &grid.init_forecast_hour);
    }
    nc_put_att_text(ncid, var_lat, "units", 12, "degrees_north");
    nc_put_att_text(ncid, var_lon, "units", 11, "degrees_east");
    nc_put_att_text(ncid, var_terrain, "units", 1, "m");
    nc_put_att_text(ncid, var_terrain, "long_name", 14, "terrain height");
    nc_put_att_text(ncid, var_terrain_slope, "units", 1, "1");
    nc_put_att_text(ncid, var_terrain_slope, "long_name", 21, "terrain slope magnitude");

    char proj_str[256];
    snprintf(proj_str, sizeof(proj_str),
             "lambert_conformal truelat1=%.1f truelat2=%.1f stand_lon=%.1f ref_lat=%.2f ref_lon=%.2f dx=%.0f dy=%.0f",
             proj.truelat1, proj.truelat2, proj.stand_lon, proj.ref_lat, proj.ref_lon, proj.dx, proj.dy);
    nc_put_att_text(ncid, NC_GLOBAL, "projection", strlen(proj_str), proj_str);

    double dx_att = grid.dx;
    double dy_att = grid.dy;
    double truelat1_att = proj.truelat1;
    double truelat2_att = proj.truelat2;
    double stand_lon_att = proj.stand_lon;
    double ref_lat_att = proj.ref_lat;
    double ref_lon_att = proj.ref_lon;
    int map_proj_att = 1;
    nc_put_att_double(ncid, NC_GLOBAL, "DX", NC_DOUBLE, 1, &dx_att);
    nc_put_att_double(ncid, NC_GLOBAL, "DY", NC_DOUBLE, 1, &dy_att);
    nc_put_att_double(ncid, NC_GLOBAL, "TRUELAT1", NC_DOUBLE, 1, &truelat1_att);
    nc_put_att_double(ncid, NC_GLOBAL, "TRUELAT2", NC_DOUBLE, 1, &truelat2_att);
    nc_put_att_double(ncid, NC_GLOBAL, "STAND_LON", NC_DOUBLE, 1, &stand_lon_att);
    nc_put_att_double(ncid, NC_GLOBAL, "REF_LAT", NC_DOUBLE, 1, &ref_lat_att);
    nc_put_att_double(ncid, NC_GLOBAL, "REF_LON", NC_DOUBLE, 1, &ref_lon_att);
    nc_put_att_int(ncid, NC_GLOBAL, "MAP_PROJ", NC_INT, 1, &map_proj_att);
    nc_put_att_double(ncid, NC_GLOBAL, "CEN_LAT", NC_DOUBLE, 1, &ref_lat_att);
    nc_put_att_double(ncid, NC_GLOBAL, "CEN_LON", NC_DOUBLE, 1, &ref_lon_att);

    NC_CHECK(nc_enddef(ncid));

    // Write coordinates
    double* x_vals = new double[nx];
    double* y_vals = new double[ny];
    for (int i = 0; i < nx; i++) x_vals[i] = (i - nx/2.0) * grid.dx / 1000.0;  // km
    for (int j = 0; j < ny; j++) y_vals[j] = (j - ny/2.0) * grid.dy / 1000.0;
    NC_CHECK(nc_put_var_double(ncid, var_x, x_vals));
    NC_CHECK(nc_put_var_double(ncid, var_y, y_vals));
    delete[] x_vals;
    delete[] y_vals;

    // Z levels
    double* z_host = new double[nz];
    CUDA_CHECK(cudaMemcpy(z_host, state.z_levels, nz * sizeof(double), cudaMemcpyDeviceToHost));
    NC_CHECK(nc_put_var_double(ncid, var_z, z_host));
    delete[] z_host;

    double* eta_w = nullptr;
    double* eta_m = nullptr;
    build_eta_coordinates(grid, eta_w, eta_m);
    NC_CHECK(nc_put_var_double(ncid, var_eta, eta_m));
    NC_CHECK(nc_put_var_double(ncid, var_eta_w, eta_w));
    delete[] eta_w;
    delete[] eta_m;

    // Time
    NC_CHECK(nc_put_var_double(ncid, var_time, &time));
    char times_buf[20];
    build_wrftime_string(grid, time, times_buf, sizeof(times_buf));
    NC_CHECK(nc_put_var_text(ncid, var_times, times_buf));

    // Lat/lon
    double* lat_vals = new double[n2d];
    double* lon_vals = new double[n2d];
    float* lat_vals_f = new float[n2d];
    float* lon_vals_f = new float[n2d];
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            size_t idx = (size_t)j * nx + i;
            proj.ij_to_latlon(i, j, lat_vals[idx], lon_vals[idx]);
            lat_vals_f[idx] = (float)lat_vals[idx];
            lon_vals_f[idx] = (float)lon_vals[idx];
        }
    }
    NC_CHECK(nc_put_var_double(ncid, var_lat, lat_vals));
    NC_CHECK(nc_put_var_double(ncid, var_lon, lon_vals));
    NC_CHECK(nc_put_var_float(ncid, var_xlat, lat_vals_f));
    NC_CHECK(nc_put_var_float(ncid, var_xlong, lon_vals_f));

    // Terrain
    real_t* terrain_host = new real_t[n2d];
    float* terrain_out = new float[n2d];
    CUDA_CHECK(cudaMemcpy(terrain_host, state.terrain, n2d * sizeof(real_t), cudaMemcpyDeviceToHost));
    for (size_t idx = 0; idx < n2d; idx++) {
        terrain_out[idx] = (float)terrain_host[idx];
    }
    NC_CHECK(nc_put_var_float(ncid, var_terrain, terrain_out));
    NC_CHECK(nc_put_var_float(ncid, var_hgt, terrain_out));

    float* terrain_slope = new float[n2d];
    compute_terrain_slope_field(terrain_host, nx, ny, grid.dx, grid.dy, terrain_slope);
    NC_CHECK(nc_put_var_float(ncid, var_terrain_slope, terrain_slope));

    double slope_mean = 0.0;
    double slope_rms = 0.0;
    double slope_max = 0.0;
    compute_terrain_slope_summary(terrain_host, nx, ny, grid.dx, grid.dy,
                                   slope_mean, slope_rms, slope_max);
    nc_put_att_double(ncid, NC_GLOBAL, "terrain_slope_mean", NC_DOUBLE, 1, &slope_mean);
    nc_put_att_double(ncid, NC_GLOBAL, "terrain_slope_rms", NC_DOUBLE, 1, &slope_rms);
    nc_put_att_double(ncid, NC_GLOBAL, "terrain_slope_max", NC_DOUBLE, 1, &slope_max);
    std::vector<double> p_base_h(nz);
    std::vector<double> z_levels_h(nz);
    CUDA_CHECK(cudaMemcpy(p_base_h.data(), state.p_base, nz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(z_levels_h.data(), state.z_levels, nz * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> mapfac_row(ny, 1.0);
    if (grid.mapfac_m) {
        CUDA_CHECK(cudaMemcpy(mapfac_row.data(), grid.mapfac_m, ny * sizeof(double), cudaMemcpyDeviceToHost));
    }
    std::vector<float> twod_buf(n2d, 0.0f);
    for (int j = 0; j < ny; ++j) {
        float mapfac = (float)mapfac_row[j];
        for (int i = 0; i < nx; ++i) {
            twod_buf[(size_t)j * nx + i] = mapfac;
        }
    }
    NC_CHECK(nc_put_var_float(ncid, var_mapfac_m, twod_buf.data()));
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            size_t idx = (size_t)j * nx + i;
            double dlon = lon_vals[idx] - proj.stand_lon;
            while (dlon > 180.0) dlon -= 360.0;
            while (dlon < -180.0) dlon += 360.0;
            double alpha = proj.n * dlon * PI / 180.0;
            twod_buf[idx] = (float)cos(alpha);
        }
    }
    NC_CHECK(nc_put_var_float(ncid, var_cosalpha, twod_buf.data()));
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            size_t idx = (size_t)j * nx + i;
            double dlon = lon_vals[idx] - proj.stand_lon;
            while (dlon > 180.0) dlon -= 360.0;
            while (dlon < -180.0) dlon += 360.0;
            double alpha = proj.n * dlon * PI / 180.0;
            twod_buf[idx] = (float)sin(alpha);
        }
    }
    NC_CHECK(nc_put_var_float(ncid, var_sinalpha, twod_buf.data()));
    delete[] lat_vals;
    delete[] lon_vals;
    delete[] lat_vals_f;
    delete[] lon_vals_f;

    // Copy 3D fields from GPU -> host -> netcdf
    real_t* host_buf = new real_t[n3d_h];
    std::vector<float> scratch_mass(n3d, 0.0f);
    std::vector<float> theta_mass(n3d, 0.0f);
    std::vector<float> qv_mass(n3d, 0.0f);
    std::vector<float> p_pert_mass(n3d, 0.0f);
    std::vector<float> pressure_mass(n3d, 0.0f);
    std::vector<float> u10(n2d, 0.0f);
    std::vector<float> v10(n2d, 0.0f);
    real_t* d_w_phys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_w_phys, n3d_h * sizeof(real_t)));

    auto copy_mass_field = [&](const real_t* d_field, std::vector<float>& out) {
        CUDA_CHECK(cudaMemcpy(host_buf, d_field, n3d_h * sizeof(real_t), cudaMemcpyDeviceToHost));
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int src = (k * ny_h + (j + 2)) * nx_h + (i + 2);
                    size_t dst = ((size_t)k * ny + j) * nx + i;
                    out[dst] = (float)host_buf[src];
                }
            }
        }
    };

    copy_mass_field(state.theta, theta_mass);
    NC_CHECK(nc_put_var_float(ncid, var_theta, theta_mass.data()));

    copy_mass_field(state.qv, qv_mass);
    NC_CHECK(nc_put_var_float(ncid, var_qv, qv_mass.data()));
    NC_CHECK(nc_put_var_float(ncid, var_qvapor, qv_mass.data()));

    copy_mass_field(state.p, p_pert_mass);
    NC_CHECK(nc_put_var_float(ncid, var_p, p_pert_mass.data()));

    copy_mass_field(state.qc, scratch_mass);
    NC_CHECK(nc_put_var_float(ncid, var_qc, scratch_mass.data()));
    NC_CHECK(nc_put_var_float(ncid, var_qcloud, scratch_mass.data()));

    copy_mass_field(state.qr, scratch_mass);
    NC_CHECK(nc_put_var_float(ncid, var_qr, scratch_mass.data()));
    NC_CHECK(nc_put_var_float(ncid, var_qrain, scratch_mass.data()));

    // Derived pressure/base-state and near-surface fields
    std::vector<float> psfc(n2d, 0.0f);
    std::vector<float> t2(n2d, 0.0f);
    std::vector<float> q2(n2d, 0.0f);
    for (int k = 0; k < nz; ++k) {
        double eta_k = grid.eta_m ? grid.eta_m[k] : ((k + 0.5) / nz);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = ((size_t)k * ny + j) * nx + i;
                double terrain = clamp_terrain_host((double)terrain_host[(size_t)j * nx + i], grid.ztop);
                double p_ref = reference_profile_at_local_height_host(
                    p_base_h.data(), z_levels_h.data(), nz, terrain, eta_k, grid.ztop
                );
                double p_full = p_ref + (double)p_pert_mass[idx];
                pressure_mass[idx] = (float)p_full;
                scratch_mass[idx] = (float)p_ref;
            }
        }
    }
    NC_CHECK(nc_put_var_float(ncid, var_pb, scratch_mass.data()));
    NC_CHECK(nc_put_var_float(ncid, var_pressure, pressure_mass.data()));

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = ((size_t)k * ny + j) * nx + i;
                double p_full = fmax((double)pressure_mass[idx], 1.0);
                double theta_full = (double)theta_mass[idx];
                double qv_val = fmax((double)qv_mass[idx], 0.0);
                double temp_k = theta_full * pow(p_full / P0, KAPPA);
                scratch_mass[idx] = (float)temp_k;

                double es = saturation_vapor_pressure_liquid(temp_k);
                double qvs = vapor_epsilon() * es / fmax(p_full - es, 1.0);
                double rh = 100.0 * qv_val / fmax(qvs, 1.0e-12);
                rh = fmin(100.0, fmax(0.0, rh));
                if (k == 0) {
                    size_t idx2d = (size_t)j * nx + i;
                    double terrain = clamp_terrain_host((double)terrain_host[idx2d], grid.ztop);
                    double z_mass0 = terrain_following_height(terrain, grid.eta_m ? grid.eta_m[0] : (0.5 / nz), grid.ztop);
                    double dz_agl = fmax(z_mass0 - terrain, 0.0);
                    double tv_low = temp_k * (1.0 + 0.61 * qv_val);
                    psfc[idx2d] = (float)(p_full * exp(G * dz_agl / (R_D * fmax(tv_low, 150.0))));
                    t2[idx2d] = (float)temp_k;
                    q2[idx2d] = (float)qv_val;
                }
            }
        }
    }
    NC_CHECK(nc_put_var_float(ncid, var_temp, scratch_mass.data()));
    for (size_t idx = 0; idx < n3d; ++idx) {
        scratch_mass[idx] = theta_mass[idx] - 300.0f;
    }
    NC_CHECK(nc_put_var_float(ncid, var_t_wrf, scratch_mass.data()));
    copy_mass_field(state.rho, scratch_mass);
    NC_CHECK(nc_put_var_float(ncid, var_rho, scratch_mass.data()));
    for (size_t idx = 0; idx < n3d; ++idx) {
        double p_full = fmax((double)pressure_mass[idx], 1.0);
        double theta_full = (double)theta_mass[idx];
        double qv_val = fmax((double)qv_mass[idx], 0.0);
        double temp_k = theta_full * pow(p_full / P0, KAPPA);
        double es = saturation_vapor_pressure_liquid(temp_k);
        double qvs = vapor_epsilon() * es / fmax(p_full - es, 1.0);
        double rh = 100.0 * qv_val / fmax(qvs, 1.0e-12);
        scratch_mass[idx] = (float)fmin(100.0, fmax(0.0, rh));
    }
    NC_CHECK(nc_put_var_float(ncid, var_rh, scratch_mass.data()));

    NC_CHECK(nc_put_var_float(ncid, var_psfc, psfc.data()));
    NC_CHECK(nc_put_var_float(ncid, var_t2, t2.data()));
    NC_CHECK(nc_put_var_float(ncid, var_q2, q2.data()));

    copy_mass_field(state.u, scratch_mass);
    NC_CHECK(nc_put_var_float(ncid, var_u_mass, scratch_mass.data()));
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            u10[(size_t)j * nx + i] = scratch_mass[(size_t)j * nx + i];
        }
    }
    std::vector<float> u_stag(n3d_u, 0.0f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                float val;
                if (i == 0) {
                    val = scratch_mass[((size_t)k * ny + j) * nx];
                } else if (i == nx) {
                    val = scratch_mass[((size_t)k * ny + j) * nx + (nx - 1)];
                } else {
                    size_t left = ((size_t)k * ny + j) * nx + (i - 1);
                    size_t right = left + 1;
                    val = 0.5f * (scratch_mass[left] + scratch_mass[right]);
                }
                u_stag[((size_t)k * ny + j) * (nx + 1) + i] = val;
            }
        }
    }
    NC_CHECK(nc_put_var_float(ncid, var_u_wrf, u_stag.data()));
    NC_CHECK(nc_put_var_float(ncid, var_u10, u10.data()));

    copy_mass_field(state.v, scratch_mass);
    NC_CHECK(nc_put_var_float(ncid, var_v_mass, scratch_mass.data()));
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            v10[(size_t)j * nx + i] = scratch_mass[(size_t)j * nx + i];
        }
    }
    std::vector<float> v_stag(n3d_v, 0.0f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                float val;
                if (j == 0) {
                    val = scratch_mass[((size_t)k * ny) * nx + i];
                } else if (j == ny) {
                    val = scratch_mass[((size_t)k * ny + (ny - 1)) * nx + i];
                } else {
                    size_t south = ((size_t)k * ny + (j - 1)) * nx + i;
                    size_t north = south + nx;
                    val = 0.5f * (scratch_mass[south] + scratch_mass[north]);
                }
                v_stag[((size_t)k * (ny + 1) + j) * nx + i] = val;
            }
        }
    }
    NC_CHECK(nc_put_var_float(ncid, var_v_wrf, v_stag.data()));
    NC_CHECK(nc_put_var_float(ncid, var_v10, v10.data()));

    dim3 block(8, 8, 4);
    dim3 grid3d((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
    materialize_physical_w_netcdf_kernel<<<grid3d, block>>>(
        d_w_phys, state.u, state.v, state.w, state.terrain, state.eta_m, grid.mapfac_m,
        nx, ny, nz, grid.dx, grid.dy, grid.ztop
    );
    copy_mass_field(d_w_phys, scratch_mass);
    NC_CHECK(nc_put_var_float(ncid, var_w_mass, scratch_mass.data()));
    std::vector<float> w_stag(n3d_w, 0.0f);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            size_t idx2d = (size_t)j * nx + i;
            float first = scratch_mass[idx2d];
            w_stag[idx2d] = first;
            for (int k = 1; k < nz; ++k) {
                size_t low = ((size_t)(k - 1) * ny + j) * nx + i;
                size_t high = ((size_t)k * ny + j) * nx + i;
                w_stag[((size_t)k * ny + j) * nx + i] = 0.5f * (scratch_mass[low] + scratch_mass[high]);
            }
            w_stag[((size_t)nz * ny + j) * nx + i] =
                scratch_mass[((size_t)(nz - 1) * ny + j) * nx + i];
        }
    }
    NC_CHECK(nc_put_var_float(ncid, var_w_wrf, w_stag.data()));

    std::vector<float> geopot_stag(n3d_w, 0.0f);
    for (int k = 0; k <= nz; ++k) {
        double eta_wk = grid.eta ? grid.eta[k] : ((double)k / nz);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double terrain = clamp_terrain_host((double)terrain_host[(size_t)j * nx + i], grid.ztop);
                double z_w = terrain_following_height(terrain, eta_wk, grid.ztop);
                geopot_stag[((size_t)k * ny + j) * nx + i] = (float)(G * z_w);
            }
        }
    }
    NC_CHECK(nc_put_var_float(ncid, var_phb, geopot_stag.data()));
    std::fill(geopot_stag.begin(), geopot_stag.end(), 0.0f);
    NC_CHECK(nc_put_var_float(ncid, var_ph, geopot_stag.data()));

    cudaFree(d_w_phys);
    delete[] host_buf;
    delete[] terrain_host;
    delete[] terrain_out;
    delete[] terrain_slope;

    NC_CHECK(nc_close(ncid));
    printf("  Output written: %s\n", filename);
}
#else
void write_netcdf(const StateGPU& state, const GridConfig& grid,
                  const LambertConformal& proj,
                  double time, int output_num) {
    fprintf(stderr, "NetCDF output disabled (libnetcdf not found)\n");
}
#endif

} // namespace gpuwm
