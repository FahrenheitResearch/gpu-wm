// ============================================================
// GPU-WM: Output Module
// Writes model state to binary files for visualization
// Simple custom binary format (easily readable by Python/etc)
// ============================================================

#include "../../include/constants.cuh"
#include "../../include/grid.cuh"
#include "../../include/stability_control.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>

namespace gpuwm {

static constexpr char GWM_TERRAIN_MAGIC[8] = {'G','W','M','T','E','R','R','1'};
static constexpr char GWM_ETA_MAGIC[8] = {'G','W','M','E','T','A','0','1'};
static constexpr char GWM_SLOPE_MAGIC[8] = {'G','W','M','S','L','P','0','1'};

struct IntegralBudgetMetrics {
    double kinetic_energy = 0.0;
    double vapor_mass = 0.0;
    double condensate_mass = 0.0;
};

struct TerrainSlopeSummary {
    int32_t nx = 0;
    int32_t ny = 0;
    double mean_slope = 0.0;
    double rms_slope = 0.0;
    double max_slope = 0.0;
};

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

__device__ inline double clamped_column_terrain_output(
    const real_t* __restrict__ terrain,
    int i, int j, int nx,
    double ztop
) {
    double terrain_val = (double)terrain[idx2(i, j, nx)];
    return fmin(terrain_val, ztop - 1.0);
}

__device__ inline double sample_terrain_clamped_output(
    const real_t* __restrict__ terrain,
    int i, int j, int nx, int ny,
    double ztop
) {
    int ii = max(0, min(i, nx - 1));
    int jj = max(0, min(j, ny - 1));
    return clamped_column_terrain_output(terrain, ii, jj, nx, ztop);
}

__device__ inline double local_metric_slope_x_output(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k, int nx, int ny,
    double dx_eff, double ztop
) {
    int i_m = max(i - 1, 0);
    int i_p = min(i + 1, nx - 1);
    double h_m = sample_terrain_clamped_output(terrain, i_m, j, nx, ny, ztop);
    double h_p = sample_terrain_clamped_output(terrain, i_p, j, nx, ny, ztop);
    double ds = max((i_p - i_m) * dx_eff, 1.0);
    return (1.0 - eta_m[k]) * (h_p - h_m) / ds;
}

__device__ inline double local_metric_slope_y_output(
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k, int nx, int ny,
    double dy_eff, double ztop
) {
    int j_m = max(j - 1, 0);
    int j_p = min(j + 1, ny - 1);
    double h_m = sample_terrain_clamped_output(terrain, i, j_m, nx, ny, ztop);
    double h_p = sample_terrain_clamped_output(terrain, i, j_p, nx, ny, ztop);
    double ds = max((j_p - j_m) * dy_eff, 1.0);
    return (1.0 - eta_m[k]) * (h_p - h_m) / ds;
}

__device__ inline double physical_vertical_velocity_output(
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
    double zx = local_metric_slope_x_output(terrain, eta_m, i, j, k, nx, ny, dx_eff, ztop);
    double zy = local_metric_slope_y_output(terrain, eta_m, i, j, k, nx, ny, dy_eff, ztop);
    int ijk = idx3(i, j, k, nx_h, ny_h);
    return (double)w_contra[ijk] + (double)u[ijk] * zx + (double)v[ijk] * zy;
}

__global__ void materialize_physical_w_kernel(
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
    w_phys[ijk] = (real_t)physical_vertical_velocity_output(
        u, v, w_contra, terrain, eta_m, mapfac_m,
        i, j, k, nx, ny, nx_h, ny_h, dx, dy, ztop
    );
}

__global__ void physical_w_stats_kernel(
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ w_contra,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    const double* __restrict__ mapfac_m,
    double* __restrict__ stats,
    int nx, int ny, int nz,
    double dx, double dy, double ztop
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int nx_h = nx + 4;
    int ny_h = ny + 4;
    double w_val = physical_vertical_velocity_output(
        u, v, w_contra, terrain, eta_m, mapfac_m,
        i, j, k, nx, ny, nx_h, ny_h, dx, dy, ztop
    );

    atomicAdd(&stats[0], w_val);
    atomicAdd(&stats[1], fabs(w_val));
    atomicMax((unsigned long long*)&stats[2], __double_as_longlong(fabs(w_val)));
}

static TerrainSlopeSummary compute_terrain_slope_summary(const real_t* terrain_host,
                                                        int nx, int ny,
                                                        double dx, double dy) {
    TerrainSlopeSummary summary;
    summary.nx = nx;
    summary.ny = ny;

    if (!terrain_host || nx <= 0 || ny <= 0 || dx <= 0.0 || dy <= 0.0) {
        return summary;
    }

    double slope_sum = 0.0;
    double slope_sq_sum = 0.0;
    double slope_max = 0.0;
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

            slope_sum += slope;
            slope_sq_sum += slope * slope;
            slope_max = std::fmax(slope_max, slope);
            ++count;
        }
    }

    if (count > 0) {
        summary.mean_slope = slope_sum / (double)count;
        summary.rms_slope = std::sqrt(slope_sq_sum / (double)count);
        summary.max_slope = slope_max;
    }

    return summary;
}

static void write_terrain_and_eta_trailers(FILE* fp,
                                           const StateGPU& state,
                                           const GridConfig& grid) {
    size_t n2d = (size_t)grid.nx * grid.ny;
    real_t* terrain_src = new real_t[n2d];
    double* terrain_host = new double[n2d];
    CUDA_CHECK(cudaMemcpy(terrain_src, state.terrain,
                          n2d * sizeof(real_t), cudaMemcpyDeviceToHost));
    for (size_t idx = 0; idx < n2d; ++idx) {
        terrain_host[idx] = (double)terrain_src[idx];
    }

    fwrite(GWM_TERRAIN_MAGIC, sizeof(GWM_TERRAIN_MAGIC), 1, fp);
    fwrite(terrain_host, sizeof(double), n2d, fp);

    double* eta_w = nullptr;
    double* eta_m = nullptr;
    build_eta_coordinates(grid, eta_w, eta_m);

    fwrite(GWM_ETA_MAGIC, sizeof(GWM_ETA_MAGIC), 1, fp);
    int32_t nz = (int32_t)grid.nz;
    fwrite(&nz, sizeof(int32_t), 1, fp);
    fwrite(eta_w, sizeof(double), (size_t)grid.nz + 1, fp);
    fwrite(eta_m, sizeof(double), (size_t)grid.nz, fp);

    TerrainSlopeSummary slope = compute_terrain_slope_summary(
        terrain_src, grid.nx, grid.ny, grid.dx, grid.dy);
    fwrite(GWM_SLOPE_MAGIC, sizeof(GWM_SLOPE_MAGIC), 1, fp);
    fwrite(&slope.nx, sizeof(int32_t), 1, fp);
    fwrite(&slope.ny, sizeof(int32_t), 1, fp);
    fwrite(&slope.mean_slope, sizeof(double), 1, fp);
    fwrite(&slope.rms_slope, sizeof(double), 1, fp);
    fwrite(&slope.max_slope, sizeof(double), 1, fp);

    delete[] terrain_src;
    delete[] terrain_host;
    delete[] eta_w;
    delete[] eta_m;
}

// ----------------------------------------------------------
// Diagnostic computation kernels
// ----------------------------------------------------------
__global__ void compute_temperature_kernel(
    const real_t* __restrict__ theta,
    const double* __restrict__ p_base,
    real_t* __restrict__ temperature,
    int n_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    // Simplified: use p_base from level 0 for now
    // Real version would interpolate p to the right level
}

// Reduction kernel for computing domain-wide statistics
__global__ void reduce_max_kernel(
    const real_t* __restrict__ field,
    double* __restrict__ result,
    int n_total
) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n_total) ? fabs((double)field[idx]) : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((unsigned long long*)result,
                  __double_as_longlong(sdata[0]));
    }
}

__global__ void reduce_sum_kernel(
    const real_t* __restrict__ field,
    double* __restrict__ result,
    int n_total
) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n_total) ? (double)field[idx] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

__global__ void reduce_abs_sum_kernel(
    const real_t* __restrict__ field,
    double* __restrict__ result,
    int n_total
) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n_total) ? fabs((double)field[idx]) : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

static double reduce_sum_host(const real_t* d_field, int n_total) {
    double zero = 0.0;
    double* d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_sum, &zero, sizeof(double), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (n_total + block - 1) / block;
    reduce_sum_kernel<<<grid, block, block * sizeof(double)>>>(d_field, d_sum, n_total);

    double sum = 0.0;
    CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_sum);
    return sum;
}

static double reduce_abs_sum_host(const real_t* d_field, int n_total) {
    double zero = 0.0;
    double* d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_sum, &zero, sizeof(double), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (n_total + block - 1) / block;
    reduce_abs_sum_kernel<<<grid, block, block * sizeof(double)>>>(d_field, d_sum, n_total);

    double sum = 0.0;
    CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_sum);
    return sum;
}

static double reduce_max_abs_host(const real_t* d_field, int n_total) {
    double zero = 0.0;
    double* d_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_max, &zero, sizeof(double), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (n_total + block - 1) / block;
    reduce_max_kernel<<<grid, block, block * sizeof(double)>>>(d_field, d_max, n_total);

    double max_val = 0.0;
    CUDA_CHECK(cudaMemcpy(&max_val, d_max, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_max);
    return max_val;
}

__global__ void budget_integrals_kernel(
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ w,
    const real_t* __restrict__ qv,
    const real_t* __restrict__ qc,
    const real_t* __restrict__ qr,
    const double* __restrict__ rho_base,
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta,
    const double* __restrict__ eta_m,
    const double* __restrict__ mapfac_m,
    double* __restrict__ integrals,
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

    double terrain_val = fmin((double)terrain[idx2(i, j, nx)], ztop - 1.0);
    double dz = fmax(
        terrain_following_layer_thickness(terrain_val, eta[k], eta[k + 1], ztop),
        1.0
    );

    double mapfac = mapfac_m ? mapfac_m[j] : 1.0;
    double cell_area = (dx / mapfac) * (dy / mapfac);
    double cell_mass = rho_base[k] * cell_area * dz;

    double u_val = (double)u[ijk];
    double v_val = (double)v[ijk];
    double w_val = physical_vertical_velocity_output(
        u, v, w, terrain, eta_m, mapfac_m,
        i, j, k, nx, ny, nx_h, ny_h, dx, dy, ztop
    );
    double qv_val = fmax((double)qv[ijk], 0.0);
    double qc_val = fmax((double)qc[ijk], 0.0);
    double qr_val = fmax((double)qr[ijk], 0.0);

    atomicAdd(&integrals[0], 0.5 * cell_mass * (u_val * u_val + v_val * v_val + w_val * w_val));
    atomicAdd(&integrals[1], cell_mass * qv_val);
    atomicAdd(&integrals[2], cell_mass * (qc_val + qr_val));
}

static IntegralBudgetMetrics compute_budget_integrals_host(
    const StateGPU& state, const GridConfig& grid
) {
    IntegralBudgetMetrics metrics;
    double zero[3] = {0.0, 0.0, 0.0};
    double* d_integrals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_integrals, sizeof(zero)));
    CUDA_CHECK(cudaMemcpy(d_integrals, zero, sizeof(zero), cudaMemcpyHostToDevice));

    dim3 block(8, 8, 4);
    dim3 grid3d((grid.nx + 7) / 8, (grid.ny + 7) / 8, (grid.nz + 3) / 4);
    budget_integrals_kernel<<<grid3d, block>>>(
        state.u, state.v, state.w, state.qv, state.qc, state.qr,
        state.rho_base, state.terrain, state.eta, state.eta_m, grid.mapfac_m, d_integrals,
        grid.nx, grid.ny, grid.nz, grid.dx, grid.dy, grid.ztop
    );

    double host_integrals[3];
    CUDA_CHECK(cudaMemcpy(host_integrals, d_integrals, sizeof(host_integrals), cudaMemcpyDeviceToHost));
    cudaFree(d_integrals);

    metrics.kinetic_energy = host_integrals[0];
    metrics.vapor_mass = host_integrals[1];
    metrics.condensate_mass = host_integrals[2];
    return metrics;
}

static void compute_physical_w_stats_host(
    const StateGPU& state, const GridConfig& grid,
    double& mean_w, double& mean_abs_w, double& max_abs_w
) {
    double zero[3] = {0.0, 0.0, 0.0};
    double* d_stats = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stats, sizeof(zero)));
    CUDA_CHECK(cudaMemcpy(d_stats, zero, sizeof(zero), cudaMemcpyHostToDevice));

    dim3 block(8, 8, 4);
    dim3 grid3d((grid.nx + 7) / 8, (grid.ny + 7) / 8, (grid.nz + 3) / 4);
    physical_w_stats_kernel<<<grid3d, block>>>(
        state.u, state.v, state.w, state.terrain, state.eta_m, grid.mapfac_m,
        d_stats, grid.nx, grid.ny, grid.nz, grid.dx, grid.dy, grid.ztop
    );

    double host_stats[3];
    CUDA_CHECK(cudaMemcpy(host_stats, d_stats, sizeof(host_stats), cudaMemcpyDeviceToHost));
    cudaFree(d_stats);

    double count = fmax((double)grid.nx * grid.ny * grid.nz, 1.0);
    mean_w = host_stats[0] / count;
    mean_abs_w = host_stats[1] / count;
    max_abs_w = host_stats[2];
}

// ----------------------------------------------------------
// Compute and print diagnostics
// ----------------------------------------------------------
void print_diagnostics(const StateGPU& state, const GridConfig& grid,
                       double time, int step,
                       const StabilityControlConfig& stability_cfg) {
    int nx_h = grid.nx + 4;
    int ny_h = grid.ny + 4;
    int n3d = nx_h * ny_h * grid.nz;

    // Copy a few values to host for diagnostics
    // Sample from domain center
    int ic = grid.nx / 2;
    int jc = grid.ny / 2;

    double u_sfc, v_sfc, theta_sfc, qv_sfc, qc_max, qr_max;

    int idx_sfc = idx3(ic, jc, 1, nx_h, ny_h);

    real_t tmp_val;
    CUDA_CHECK(cudaMemcpy(&tmp_val, state.u + idx_sfc, sizeof(real_t), cudaMemcpyDeviceToHost));
    u_sfc = (double)tmp_val;
    CUDA_CHECK(cudaMemcpy(&tmp_val, state.v + idx_sfc, sizeof(real_t), cudaMemcpyDeviceToHost));
    v_sfc = (double)tmp_val;
    CUDA_CHECK(cudaMemcpy(&tmp_val, state.theta + idx_sfc, sizeof(real_t), cudaMemcpyDeviceToHost));
    theta_sfc = (double)tmp_val;
    CUDA_CHECK(cudaMemcpy(&tmp_val, state.qv + idx_sfc, sizeof(real_t), cudaMemcpyDeviceToHost));
    qv_sfc = (double)tmp_val;

    double mean_u = reduce_sum_host(state.u, n3d) / n3d;
    double mean_v = reduce_sum_host(state.v, n3d) / n3d;
    double mean_w = 0.0;
    double mean_abs_w = 0.0;
    double w_max = 0.0;
    compute_physical_w_stats_host(state, grid, mean_w, mean_abs_w, w_max);
    double mean_theta = reduce_sum_host(state.theta, n3d) / n3d;
    double mean_qv = reduce_sum_host(state.qv, n3d) / n3d;
    double p_max = reduce_max_abs_host(state.p, n3d);
    IntegralBudgetMetrics budgets = compute_budget_integrals_host(state, grid);
    FlowControlMetrics control_metrics = compute_flow_control_metrics(state, grid);
    AdaptiveStabilityState adaptive_state =
        evaluate_adaptive_stability(stability_cfg, control_metrics, time > 0.0 ? time / fmax((double)step, 1.0) : 0.0);
    WTransportDiagnostics w_transport_diag;
    if (stability_cfg.w_transport_diagnostics) {
        w_transport_diag = consume_w_transport_diagnostics();
    }

    static bool baseline_set = false;
    static IntegralBudgetMetrics baseline_budgets;
    static double baseline_enstrophy = 0.0;
    if (!baseline_set || step == 0) {
        baseline_budgets = budgets;
        baseline_enstrophy = 0.5 * control_metrics.mean_vort2;
        baseline_set = true;
    }

    double total_water = budgets.vapor_mass + budgets.condensate_mass;
    double baseline_total_water = baseline_budgets.vapor_mass + baseline_budgets.condensate_mass;
    double enstrophy = 0.5 * control_metrics.mean_vort2;
    double ke_drift_pct = 100.0 * (budgets.kinetic_energy - baseline_budgets.kinetic_energy) /
        fmax(baseline_budgets.kinetic_energy, 1.0);
    double water_drift_pct = 100.0 * (total_water - baseline_total_water) /
        fmax(baseline_total_water, 1.0e-12);
    double enstrophy_drift_pct = 100.0 * (enstrophy - baseline_enstrophy) /
        fmax(baseline_enstrophy, 1.0e-12);

    // Find max qc and qr (simple host-side scan of a column)
    qc_max = 0.0;
    qr_max = 0.0;

    for (int k = 0; k < grid.nz; k++) {
        int idx_k = idx3(ic, jc, k, nx_h, ny_h);
        real_t qc_tmp, qr_tmp;
        CUDA_CHECK(cudaMemcpy(&qc_tmp, state.qc + idx_k, sizeof(real_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&qr_tmp, state.qr + idx_k, sizeof(real_t), cudaMemcpyDeviceToHost));
        double qc_val = (double)qc_tmp, qr_val = (double)qr_tmp;
        if (qc_val > qc_max) qc_max = qc_val;
        if (qr_val > qr_max) qr_max = qr_val;
    }

    printf(
        "Step %6d  t=%8.1fs | center u/v=%.2f/%.2f  th=%.2f  qv=%.5f | "
        "mean u/v/w=%.2f/%.2f/%.4f  mean|w|=%.4f  max|w|=%.3f  max|p'|=%.1f | "
        "mean th=%.2f  mean qv=%.5f | KE=%.3e (%+.2f%%)  qtot=%.3e (%+.2f%%)  Z=%.3e (%+.2f%%) | "
        "div=%.3e/%.3e (sgn=%+.3e h=%+.3e w=%+.3e)  vort=%.3e/%.3e | "
        "stab x%.2f  p=%.2f | qc_max=%.6f qr_max=%.6f\n",
        step, time, u_sfc, v_sfc, theta_sfc, qv_sfc,
        mean_u, mean_v, mean_w, mean_abs_w, w_max, p_max,
        mean_theta, mean_qv,
        budgets.kinetic_energy, ke_drift_pct,
        total_water, water_drift_pct,
        enstrophy, enstrophy_drift_pct,
        control_metrics.mean_abs_div, control_metrics.max_abs_div,
        control_metrics.mean_div, control_metrics.mean_hdiv, control_metrics.mean_dwdz,
        control_metrics.mean_abs_vort, control_metrics.max_abs_vort,
        adaptive_state.kdiff_scale, adaptive_state.pressure_retain,
        qc_max, qr_max
    );
    if (stability_cfg.w_transport_diagnostics && w_transport_diag.samples > 0.5) {
        printf(
            "                  w-transport: blend=%.2f  calls=%.0f  samples=%.0f | "
            "|old|=%.3e  |new|=%.3e  |delta|=%.3e  rms(delta)=%.3e  "
            "mean(div)=%.3e  rms(div)=%.3e  corr(delta,div)=%.3f\n",
            stability_cfg.w_transport_blend,
            w_transport_diag.tendency_calls,
            w_transport_diag.samples,
            w_transport_diag.mean_abs_old_total,
            w_transport_diag.mean_abs_new_total,
            w_transport_diag.mean_abs_delta,
            w_transport_diag.rms_delta,
            w_transport_diag.mean_divergence,
            w_transport_diag.rms_divergence,
            w_transport_diag.delta_div_correlation
        );
    }
}

// ----------------------------------------------------------
// Binary output format:
// Header: nx, ny, nz, time (doubles)
// Then each 3D field as flat array
// ----------------------------------------------------------
struct OutputHeader {
    int nx, ny, nz;
    double time;
    double dx, dy, ztop;
};

void write_output(const StateGPU& state, const GridConfig& grid,
                  double time, int output_num) {
    char filename[256];
    snprintf(filename, sizeof(filename), "output/gpuwm_%06d.bin", output_num);

    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s for writing\n", filename);
        return;
    }

    int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    int nx_h = nx + 4;
    int ny_h = ny + 4;
    size_t n3d_h = (size_t)nx_h * ny_h * nz;

    // Write header
    OutputHeader hdr;
    hdr.nx = nx; hdr.ny = ny; hdr.nz = nz;
    hdr.time = time;
    hdr.dx = grid.dx; hdr.dy = grid.dy; hdr.ztop = grid.ztop;
    fwrite(&hdr, sizeof(OutputHeader), 1, fp);

    // Write z_levels
    double* z_host = new double[nz];
    CUDA_CHECK(cudaMemcpy(z_host, state.z_levels, nz * sizeof(double), cudaMemcpyDeviceToHost));
    fwrite(z_host, sizeof(double), nz, fp);
    delete[] z_host;

    // Copy fields to host and write interior only (no halos)
    real_t* host_buf = new real_t[n3d_h];
    double* interior = new double[(size_t)nx * ny * nz];
    real_t* d_w_phys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_w_phys, n3d_h * sizeof(real_t)));

    auto write_field = [&](const real_t* d_field) {
        CUDA_CHECK(cudaMemcpy(host_buf, d_field, n3d_h * sizeof(real_t), cudaMemcpyDeviceToHost));
        // Extract interior and convert to double for output
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int src = (k * ny_h + (j + 2)) * nx_h + (i + 2);
                    int dst = (k * ny + j) * nx + i;
                    interior[dst] = (double)host_buf[src];
                }
            }
        }
        fwrite(interior, sizeof(double), (size_t)nx * ny * nz, fp);
    };

    auto write_physical_w_field = [&]() {
        dim3 block(8, 8, 4);
        dim3 grid3d((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
        materialize_physical_w_kernel<<<grid3d, block>>>(
            d_w_phys, state.u, state.v, state.w, state.terrain, state.eta_m, grid.mapfac_m,
            nx, ny, nz, grid.dx, grid.dy, grid.ztop
        );
        write_field(d_w_phys);
    };

    // Write all fields
    write_field(state.u);
    write_field(state.v);
    write_physical_w_field();
    write_field(state.theta);
    write_field(state.qv);
    write_field(state.qc);
    write_field(state.qr);
    write_field(state.p);
    write_field(state.rho);

    write_terrain_and_eta_trailers(fp, state, grid);

    cudaFree(d_w_phys);
    delete[] host_buf;
    delete[] interior;

    fclose(fp);
    printf("  Output written: %s\n", filename);
}

} // namespace gpuwm
