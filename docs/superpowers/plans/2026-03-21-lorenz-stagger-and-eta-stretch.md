# Lorenz Stagger (w→nz+1) + Stretched Eta + Reference Profile Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make w a true nz+1 interface field (Lorenz stagger), add surface-packed stretched eta levels, and fix `reference_profile_at_local_height` for non-uniform eta — completing the structural dycore work needed for HRRR-competitive upper-air dynamics.

**Architecture:** Three independent changes applied in order: (1) stretched eta with updated reference profile lookup, (2) w allocation and indexing from nz→nz+1 across all files, (3) build verification and HRRR test run. The stretched eta is done first because the w-stagger change will interact with eta-level spacing.

**Tech Stack:** CUDA C++, WSL build (`wsl -e bash -c "cd /mnt/c/Users/drew/gpu-wm && cmake --build build-wsl -j8"`), Python test runner (`tools/run_fast_case.py`), verification (`tools/verify_forecast.py`).

**Build command:** `wsl -e bash -c "cd /mnt/c/Users/drew/gpu-wm && cmake --build build-wsl -j8 2>&1 | tail -20"`

---

## Phase 1: Stretched Eta + Reference Profile Fix

### Task 1: Implement stretched eta coordinate

**Files:**
- Modify: `src/core/init.cu:95-121` (setup_vertical_levels)

- [ ] **Step 1: Replace uniform eta with surface-packed quadratic+linear stretching**

In `setup_vertical_levels` (line 95-121 of `src/core/init.cu`), replace the uniform `eta = k/nz` with a two-zone scheme:
- Bottom zone (k=0..n_sfc): quadratic packing in [0, eta_sfc]. First mass level ~10-20m AGL.
- Upper zone (k=n_sfc+1..nz): linear spacing in [eta_sfc, 1.0].

```cuda
void setup_vertical_levels(GridConfig& grid) {
    release_vertical_levels(grid);

    double* eta_h = new double[grid.nz + 1];
    double* eta_m_h = new double[grid.nz];

    // Two-zone stretching: pack levels near surface for PBL resolution.
    // Bottom zone: quadratic in [0, eta_sfc], giving ~15 levels below 1km.
    // Upper zone:  linear in [eta_sfc, 1.0].
    int n_sfc = 15;                       // levels in the surface zone
    if (n_sfc > grid.nz / 2) n_sfc = grid.nz / 2;
    double eta_sfc = 0.04;               // eta_sfc * ztop = top of surface zone
    // For ztop=25000, eta_sfc=0.04 => surface zone covers 0-1000m

    eta_h[0] = 0.0;
    for (int k = 1; k <= grid.nz; k++) {
        if (k <= n_sfc) {
            double f = (double)k / n_sfc;
            eta_h[k] = eta_sfc * f * f;   // quadratic: finer at bottom
        } else {
            double f = (double)(k - n_sfc) / (grid.nz - n_sfc);
            eta_h[k] = eta_sfc + (1.0 - eta_sfc) * f;
        }
    }
    eta_h[grid.nz] = 1.0;  // ensure exact top

    for (int k = 0; k < grid.nz; k++) {
        eta_m_h[k] = 0.5 * (eta_h[k] + eta_h[k + 1]);
    }

    grid.eta = new double[grid.nz + 1];
    grid.eta_m = new double[grid.nz];
    memcpy(grid.eta, eta_h, (grid.nz + 1) * sizeof(double));
    memcpy(grid.eta_m, eta_m_h, grid.nz * sizeof(double));

    delete[] eta_h;
    delete[] eta_m_h;
}
```

- [ ] **Step 2: Build and verify**

Run: `wsl -e bash -c "cd /mnt/c/Users/drew/gpu-wm && cmake --build build-wsl -j8 2>&1 | tail -5"`
Expected: `[100%] Built target gpu-wm`

---

### Task 2: Fix reference_profile_at_local_height for non-uniform eta

**Files:**
- Modify: `src/core/dynamics.cu:160-189` (reference_profile_at_local_height)

The current code assumes uniform spacing: `dz_ref = ztop / nz; ref_pos = z_local / dz_ref - 0.5`. With stretched eta, the base state levels are non-uniform and this lookup returns wrong values.

- [ ] **Step 1: Replace uniform-assumption lookup with binary search on z_levels**

The function needs the actual `z_levels` array (mass-level heights) to find the correct interpolation bracket. Add `z_levels` as a parameter:

```cuda
__device__ inline double reference_profile_at_local_height(
    const double* __restrict__ profile,
    const double* __restrict__ z_levels,   // actual mass-level heights [nz]
    const real_t* __restrict__ terrain,
    const double* __restrict__ eta_m,
    int i, int j, int k,
    int nx, int ny, int nz,
    double ztop
) {
    if (nz <= 1) return profile[0];

    double terrain_val = sample_terrain_clamped(terrain, i, j, nx, ny, ztop);
    double z_local = terrain_following_height(terrain_val, eta_m[k], ztop);

    // Binary search for the interval [z_levels[k0], z_levels[k0+1]] containing z_local
    if (z_local <= z_levels[0]) {
        // Extrapolate below
        double dz = fmax(z_levels[1] - z_levels[0], 1.0);
        return profile[0] + (z_local - z_levels[0]) * (profile[1] - profile[0]) / dz;
    }
    if (z_local >= z_levels[nz - 1]) {
        // Extrapolate above
        double dz = fmax(z_levels[nz-1] - z_levels[nz-2], 1.0);
        return profile[nz-1] + (z_local - z_levels[nz-1]) * (profile[nz-1] - profile[nz-2]) / dz;
    }

    // Binary search
    int lo = 0, hi = nz - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (z_levels[mid] <= z_local) lo = mid;
        else hi = mid;
    }

    double frac = (z_local - z_levels[lo]) / fmax(z_levels[hi] - z_levels[lo], 1.0);
    return profile[lo] + frac * (profile[hi] - profile[lo]);
}
```

- [ ] **Step 2: Update ALL call sites to pass `state.z_levels`**

Search for every call to `reference_profile_at_local_height` in dynamics.cu and add `z_levels` as the second argument. The callers are:
- `reference_density_at_local_height` (~line 200-213)
- `buoyancy_kernel` (~line 657-672)
- `pressure_gradient_kernel` (indirectly via `reference_density_from_field`)
- `rayleigh_damping_kernel` (~line 790)
- `sanitize_prognostic_state_kernel` (~line 1036)

For each, pass the `z_levels` field (already available as a kernel parameter or via `state.z_levels`). Kernels that don't currently have `z_levels` as a parameter need it added to their signatures and call sites.

- [ ] **Step 3: Also update sponge.cu's `sponge_theta_reference`**

This function in `src/core/sponge.cu` (lines 33-63) has its own copy of the uniform-spacing interpolation. Replace it with a call to the fixed version or duplicate the binary search logic. It needs `z_levels` passed through the sponge kernels.

- [ ] **Step 4: Build and verify**

Run: build command
Expected: clean build

---

## Phase 2: w → nz+1 Interface Field (Lorenz Stagger)

### Task 3: Allocation and indexing changes in grid.cuh

**Files:**
- Modify: `include/grid.cuh:106-186,219-221`

- [ ] **Step 1: Add idx3w helper and change w/w_tend allocation to nz+1**

```cuda
// After idx3 at line 221, add:
__host__ __device__ inline int idx3w(int i, int j, int k, int nx_h, int ny_h) {
    return (k * ny_h + (j + 2)) * nx_h + (i + 2);
}
```

In `allocate_state` (line 106):
- After `size_t n3d_h = ...` (line 112), add: `size_t n3d_w = (size_t)nx_h * ny_h * (grid.nz + 1);`
- Line 117: Change `cudaMalloc(&state.w, n3d_h * ...)` → `n3d_w`
- Line 130: Change `cudaMalloc(&state.w_tend, n3d_h * ...)` → `n3d_w`
- Line 170: Change `cudaMemset(state.w, 0, n3d_h * ...)` → `n3d_w`
- Line 180: Change `cudaMemset(state.w_tend, 0, n3d_h * ...)` → `n3d_w`

- [ ] **Step 2: Build and verify**

Expected: build succeeds (all kernels still use old nz guards, just more memory allocated)

---

### Task 4: Fix bc_w_kernel for nz+1

**Files:**
- Modify: `src/core/dynamics.cu:1126-1147` (bc_w_kernel)
- Modify: `src/core/boundaries.cu:233-256` (zero_w_vertical_bc_kernel)

- [ ] **Step 1: Update bc_w_kernel to set w[k=0]=0 and w[k=nz]=0 using idx3w**

```cuda
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
    (void)u; (void)v; (void)terrain; (void)eta_m; (void)mapfac_m;
    (void)dx; (void)dy; (void)ztop;
    int nx_h = nx + 4, ny_h = ny + 4;
    w[idx3w(i, j, 0,  nx_h, ny_h)] = (real_t)0.0;  // surface
    w[idx3w(i, j, nz, nx_h, ny_h)] = (real_t)0.0;  // model top
}
```

Apply the same change to `zero_w_vertical_bc_kernel` in boundaries.cu.

- [ ] **Step 2: Build and verify**

---

### Task 5: Fix lateral BC kernels for w with nz+1

**Files:**
- Modify: `src/core/dynamics.cu` (apply_boundary_conditions, refresh_fast_field_boundaries)
- Modify: `src/core/boundaries.cu` (apply_open_boundaries, refresh_open_halos)

- [ ] **Step 1: Pass nz+1 instead of nz for w in all lateral BC kernel launches**

In `apply_boundary_conditions` (~line 1209): separate w from other fields and call periodic BCs with `nz+1`:
```cuda
    real_t* mass_fields[] = {state.u, state.v, state.theta,
                             state.qv, state.qc, state.qr, state.p};
    for (auto* f : mass_fields) {
        periodic_bc_x_kernel<<<grid_jk, block_jk>>>(f, nx, ny, nz);
        periodic_bc_y_kernel<<<grid_ik, block_ik>>>(f, nx, ny, nz);
    }
    // w has nz+1 levels
    dim3 grid_jk_w((ny+15)/16, (nz+16)/16);
    dim3 grid_ik_w((nx+15)/16, (nz+16)/16);
    periodic_bc_x_kernel<<<grid_jk_w, block_jk>>>(state.w, nx, ny, nz+1);
    periodic_bc_y_kernel<<<grid_ik_w, block_ik>>>(state.w, nx, ny, nz+1);
```

Same pattern in `refresh_fast_field_boundaries` for the w field.

In `apply_open_boundaries` (boundaries.cu): w in `relax_fields[]` must use `nz+1` for all kernel calls (open_bc, relax_boundary, fill_halo). Same for `refresh_open_halos`.

- [ ] **Step 2: Build and verify**

---

### Task 6: Fix acoustic solver kernels for nz+1 w

**Files:**
- Modify: `src/core/dynamics.cu` (acoustic_vertical_pg_kernel, pressure_update_kernel, run_vertical_acoustic_substeps)

- [ ] **Step 1: Update acoustic_vertical_pg_kernel to use idx3w and loop k=1..nz-1**

The kernel already uses compact stencil `(p[k]-p[k-1])/dz`. Change indexing from `idx3` to `idx3w` for w reads/writes. The k-range stays `k=1..nz-1` (interior w-interfaces; k=0 and k=nz are BCs).

```cuda
    int ijk_w = idx3w(i, j, k, nx_h, ny_h);
    // ... (stencil unchanged) ...
    w[ijk_w] = (real_t)w_new;
```

- [ ] **Step 2: Update pressure_update_kernel to use idx3w for w reads**

The kernel reads `w[k]` and `w[k+1]`. Change to `idx3w`:
```cuda
    double w_top = (k < nz) ? (double)w[idx3w(i, j, k + 1, nx_h, ny_h)] : 0.0;
    double w_bot = (double)w[idx3w(i, j, k, nx_h, ny_h)];
```
Note: with nz+1 allocation, `w[idx3w(i,j,nz,...)]` is now valid (it's the top boundary = 0).

- [ ] **Step 3: Update run_vertical_acoustic_substeps for w refresh with nz+1**

In `refresh_fast_field_boundaries` calls for w, need to handle the extra level. Either modify `refresh_fast_field_boundaries` to accept a `nz_field` parameter, or call with `nz+1` directly.

- [ ] **Step 4: Build and verify**

---

### Task 7: Fix tendency kernels for nz+1 w_tend

**Files:**
- Modify: `src/core/dynamics.cu` (buoyancy_kernel, pressure_gradient_kernel, advection_momentum_kernel, diffusion_kernel, compute_tendencies, rk3_step)

- [ ] **Step 1: Update buoyancy_kernel to use idx3w for w_tend**

The kernel already averages theta from k and k-1. Change `w_tend[ijk]` to `w_tend[idx3w(i,j,k,...)]`. Loop bound stays `k=1..nz-1`.

- [ ] **Step 2: Update pressure_gradient_kernel w_tend to use idx3w**

The PG metric correction `w_tend += inv_rho*(zx*dpdx + zy*dpdy)` needs to target w_tend at w-levels. For now, keep computing at mass levels but write to the nearest w-level:
```cuda
    // Write PG contribution to w_tend at interface k (above this mass cell)
    // Average would be more correct but this is a reasonable approximation
    // for the slow-mode tendency
    w_tend[idx3w(i, j, k, nx_h, ny_h)] = (real_t)(
        (double)w_tend[idx3w(i, j, k, nx_h, ny_h)] + inv_rho * (zx * dpdx + zy * dpdy));
```

Note: This kernel's k-range is `k=0..nz-1`. The w_tend write for k=0 goes to the surface interface (which will be zeroed by BC anyway). This is safe.

- [ ] **Step 3: Update advection w section to use idx3w**

In `advection_momentum_kernel`, the w block (lines ~518-554) writes `w_tend[ijk]`. Change to `w_tend[idx3w(i,j,k,...)]`. Also change reads of `w` to use `idx3w`. The loop bound for w advection should be `k=1..nz-1` (interior w-levels).

- [ ] **Step 4: Update compute_tendencies**

- `zero_field_kernel` for w_tend: use `n_total_w = nx_h * ny_h * (nz+1)` and matching grid dimensions
- Diffusion kernel call for w: need `nz+1` vertical extent or a separate w-diffusion call
- Add `int n_total_w = nx_h * ny_h * (nz + 1); int grid1d_w = (n_total_w + 255) / 256;`

- [ ] **Step 5: Update rk3_step**

- `rk3_update_kernel` for w: use `n_total_w`
- Add the same `n_total_w` computation

- [ ] **Step 6: Build and verify**

---

### Task 8: Fix remaining kernels for nz+1 w

**Files:**
- Modify: `src/core/dynamics.cu` (sanitize, convert_w_to_contravariant, rayleigh_damping)
- Modify: `src/core/sponge.cu` (rayleigh_sponge_kernel, apply_sponge)
- Modify: `src/main.cu` (copy_state, blend_boundary_state)
- Modify: `src/core/init.cu` (init_fields_kernel, load_gfs_binary)

- [ ] **Step 1: sanitize_prognostic_state_kernel -- loop to nz+1 for w**

Guard mass-level fields with `if (k < nz)` and w with `if (k <= nz)`. Use `idx3w` for w reads/writes. Update the grid launch to cover nz+1 levels.

- [ ] **Step 2: convert_w_to_contravariant_kernel -- loop to nz+1, use idx3w, eta[k]**

Change guard from `k >= nz` to `k > nz`. Use `idx3w` and `eta[k]` (w-level) instead of `eta_m[k]` for metric slope:
```cuda
    double zx = (1.0 - eta[k]) * local_terrain_slope_x(...);
```
Update host driver `convert_w_to_contravariant` grid launch dimensions.

- [ ] **Step 3: rayleigh_damping_kernel (if still present) -- loop to nz+1, use idx3w, eta[k]**

Use `eta[k]` for height computation, `idx3w` for w access.

- [ ] **Step 4: sponge.cu -- rayleigh_sponge_kernel w access via idx3w**

Split w damping to use `idx3w` and loop to `k <= nz`. Pass `state.eta` for w-level heights. Update `apply_sponge` grid dimensions.

- [ ] **Step 5: main.cu -- copy_state and blend_boundary_state use n_total_w for w**

```cuda
int n_total_w = nx_h * ny_h * (grid.nz + 1);
int grid1d_w = (n_total_w + 255) / 256;
copy_field_kernel<<<grid1d_w, block>>>(dst.w, src.w, n_total_w);
// ... same for blend_field_kernel
```

- [ ] **Step 6: init.cu -- init_fields_kernel covers nz+1 for w, load_gfs_binary handles nz+1**

For `init_fields_kernel`: the w initialization (w_val=0 always) just needs the kernel to cover k=0..nz with `idx3w`. Since w is always 0 at init, this is straightforward.

For `load_gfs_binary`: w is typically loaded as nz values from the file. Initialize w as zero on all nz+1 levels, then load the nz file values into indices k=0..nz-1. Or simply zero-fill (w is near-zero at init anyway).

- [ ] **Step 7: Build and verify**

---

### Task 9: Fix output for nz+1 w

**Files:**
- Modify: `src/io/output.cu` (materialize_physical_w_kernel, physical_w_stats_kernel)
- Modify: `src/io/netcdf_output.cu` (materialize_physical_w_netcdf_kernel, write_netcdf)

- [ ] **Step 1: Update output kernels to read w via idx3w**

The output kernels convert contravariant w to physical w. They loop k=0..nz-1 and output nz mass-level values. With w on interfaces, interpolate to mass levels:
```cuda
double w_phys = 0.5 * ((double)w_contra[idx3w(i,j,k,...)] + (double)w_contra[idx3w(i,j,k+1,...)]);
// ... add metric correction to get physical w ...
```

- [ ] **Step 2: netcdf_output.cu already has WRF W stagger dimension (nz+1)**

The netcdf writer already defines `n3d_w = nx*ny*(nz+1)` and `dim_bt_stag = nz+1`. Update to write the actual nz+1 w-interface values for the staggered `W` variable, and interpolated mass-level values for `W_MASS`.

- [ ] **Step 3: Build and verify**

---

### Task 10: flow_control_metrics_kernel diagnostic fix

**Files:**
- Modify: `src/core/dynamics.cu` (~line 369)

- [ ] **Step 1: Interpolate w to mass levels in diagnostics**

Replace `w[idx3(i,j,k,...)]` with `0.5*(w[idx3w(i,j,k,...)] + w[idx3w(i,j,k+1,...)])` in the flow_control_metrics_kernel. This is diagnostic-only.

- [ ] **Step 2: Build and verify final clean build**

Run: build command
Expected: `[100%] Built target gpu-wm` with no errors

---

## Phase 3: HRRR Test Run

### Task 11: Run HRRR benchmark and verify improvements

- [ ] **Step 1: Run 15-minute HRRR test**

```bash
python3 /mnt/c/Users/drew/gpu-wm/tools/run_fast_case.py \
  --init /mnt/c/Users/drew/gpu-wm/data/hrrr_init_fast_20260321_t23z_retry_1350x795x50_dx4000_latp38.5_lonm097.5_terrain.bin \
  --regional-4km-large \
  --tend 900 \
  --output-interval 900 \
  --skip-init-plot \
  --tag hrrr_4km_15m_lorenz
```

- [ ] **Step 2: Verify against baseline**

```bash
python3 /mnt/c/Users/drew/gpu-wm/tools/verify_forecast.py \
  --reference /mnt/c/Users/drew/gpu-wm/data/hrrr_init_fast_20260321_t23z_retry_1350x795x50_dx4000_latp38.5_lonm097.5_terrain.bin \
  <output_nc_file>
```

Compare against old 15-min baseline:
- Old: U rmse=4.36, V rmse=4.74, THETA rmse=11.43
- Target: measurable improvement, especially in THETA and mean_w

- [ ] **Step 3: If stable, run full 1-hour test**

```bash
python3 /mnt/c/Users/drew/gpu-wm/tools/run_fast_case.py \
  --init /mnt/c/Users/drew/gpu-wm/data/hrrr_init_fast_20260321_t23z_retry_1350x795x50_dx4000_latp38.5_lonm097.5_terrain.bin \
  --regional-4km-large \
  --tend 3600 \
  --output-interval 1800 \
  --skip-init-plot \
  --tag hrrr_4km_1h_lorenz
```

- [ ] **Step 4: Render WRF-style 500mb products for visual comparison**

```bash
cmd.exe /c py -3 "C:\Users\drew\gpu-wm\tools\render_wrf_products.py" \
  --input <1h_output_nc> \
  --output-dir <run_dir>/wrf_products_1h
```
