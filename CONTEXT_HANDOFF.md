# GPU-WM Context Handoff

Last updated: 2026-03-31

## What This Repo Is

`gpu-wm` is an all-GPU regional weather-model prototype in CUDA, aimed at something in the HRRR/RRFS direction for limited-area forecasting on NVIDIA GPUs.

Current capabilities:

- all-GPU compressible nonhydrostatic core
- Lambert conformal regional geometry
- HRRR and GFS GRIB ingest via `tools/init_from_gfs.py`
- terrain-following init support
- NetCDF output with a WRF-compatible shim schema
- WRF-style plotting through `wrf-rust-plots` without modifying that repo
- Kessler warm-rain microphysics, optional Thompson
- PBL, sponge, open boundaries, adaptive stability control

This is still a prototype/research model, not an operational-quality weather model.

## Repository / Workspace State

- Workspace: `/mnt/c/Users/drew/gpu-wm`
- Validated build: `build-wsl/gpu-wm`
- Git worktree is very dirty and appears to have no committed `HEAD` yet
- Large generated artifacts exist in `data/`, `run-fast/`, and `run-cycles/`
- The previous `CONTEXT_HANDOFF.md` was stale; this file replaces it with the current branch state

## Most Important Current Reality

The model is no longer failing at the level of “can it run a CONUS-ish HRRR-sized domain at all.” It can.

The real remaining problem is now clear:

- the model can produce plausible-looking surface fields and somewhat weather-like upper-air plots
- but the upper-air solution still drifts too much with time
- the drift is partly physics coupling, but the main structural blocker is still the dycore pressure/mass closure over terrain

Short version:

- startup / ingest / plotting stack: much better than before
- `15-60 min` HRRR-initialized CONUS runs: meaningfully improved today
- still not operationally credible aloft
- next real step is structural dycore work, not more random tuning

## High-Level Architecture

Main files:

- `src/main.cu`: CLI, run loop, init loading, boundary interpolation, physics orchestration
- `src/core/dynamics.cu`: RK3 dynamics, pressure/acoustic step, terrain-metric operators
- `src/core/init.cu`: binary init load, terrain-aware hydrostatic reference reconstruction
- `src/core/boundaries.cu`: open-boundary / relaxation logic
- `src/core/sponge.cu`: upper/lateral sponge
- `src/physics/microphysics.cu`: Kessler warm-rain microphysics
- `src/physics/microphysics_thompson.cu`: Thompson mixed-phase microphysics
- `src/physics/pbl.cu`: PBL / vertical diffusion
- `src/io/netcdf_output.cu`: native + WRF-compatible NetCDF output
- `tools/init_from_gfs.py`: GRIB -> init binary builder; now supports native Lambert HRRR inputs
- `tools/rust-init-writer/`: Rust hot-path for fast large init generation
- `tools/run_fast_case.py`: main fast-loop runner
- `tools/verify_forecast.py`: forecast-vs-reference verifier
- `tools/render_wrf_products.py`: WRF-style plot frontend using `wrf-rust`
- `tools/plot_weather.py`: native plotting panels

## 2026-03-31 Update

### What landed

1. `w[nz+1]` runtime plumbing was finished in the places that were still half-converted.

- `src/main.cu` state copy/blend now copies the full `w` extent.
- `src/core/dynamics.cu` now uses `n_total_w` for `w_tend` zeroing and RK updates.
- `src/core/dynamics.cu` and `src/core/boundaries.cu` now refresh lateral halos for `w` with `nz+1` and enforce `w=0` at `k=0` and `k=nz`.

2. Reference-profile lookups were upgraded from uniform-height assumptions to actual `z_levels` interpolation.

- `src/core/dynamics.cu`
- `src/core/sponge.cu`
- `src/io/netcdf_output.cu`

3. The init pipeline now supports an experimental stretched-eta path.

- `tools/init_from_gfs.py` has a new opt-in `--stretched-eta`
- `tools/run_fast_case.py` passes that through

This is intentionally not the default.

### What was validated

Stable default real-data smoke test after the runtime/reference fixes:

- Run dir: `run-fast/eta_uniform_restore_20260331_024752`
- HRRR terrain-following init, `64x64x20`, `dx=3000 m`, `dt=10 s`, `t=120 s`
- verification at `120 s`
  - `mean_w = -0.7452`
  - `mean|w| = 1.3642`
  - `max|w| = 25.95`
  - `U rmse = 7.57`
  - `V rmse = 7.56`
  - `THETA rmse = 0.90`
  - `QV rmse = 0.00000`

Experimental stretched-init smoke test:

- Run dir: `run-fast/eta_stretch_hrrr_smoke2_20260331_024652`
- same HRRR `64x64x20` setup, but with stretched init levels
- verification at `120 s`
  - `mean_w = +20.7677`
  - `mean|w| = 21.7769`
  - `max|w| = 124.75`
  - `U rmse = 14.77`
  - `V rmse = 14.96`
  - `THETA rmse = 23.94`
  - `QV rmse = 0.00037`

Interpretation:

- the repo is now materially closer to a real Lorenz-stagger / nonuniform-eta core because the bookkeeping and reference-state infrastructure are no longer hard-coded to uniform `z`
- but the real-data dycore is still not ready to make stretched vertical spacing the default
- the remaining blocker is still the core vertical pressure/acoustic/continuity closure, not the GRIB ingest or the writers

### Adjacent tooling

- there is also a separate repo called `metrust[gpu]`, described by the maintainer as CUDA-accelerated Rust/MetPy tooling
- if preprocessing, interpolation, or diagnostics become the next bottleneck again, it is worth mining that repo and the related Rust tool repos before writing more Python hot paths here

## What Changed In The 2026-03-21 Session

### 1. Exploration-agent audit identified real structural issues

Three exploration agents were used. The most useful conclusions:

- **Dycore audit**: the current pressure/mass core is still fundamentally collocated and uses mass-level `w`; this is the main structural problem, not the late sanitizer.
- **Physics audit**: default Kessler microphysics was missing latent-heating feedback from saturation adjustment into `theta`, which can materially worsen upper-air/moist drift.
- **Output audit**: some plots were lying:
  - 500 mb WRF-style scalar products are the most trustworthy current upper-air diagnostics
  - 850 mb products are partly contaminated by below-ground interpolation over terrain
  - native `plot_weather.py` upper-air left panel was mislabeled and visually broken
  - `COSALPHA/SINALPHA` metadata were wrong, so WRF-style wind rotation was suspect

### 2. Real solver fixes landed

Two important model fixes were made:

- In `src/core/dynamics.cu`, the split-explicit pressure “filter” was corrected to **damp** substep pressure increments instead of amplifying them.
- In `src/physics/microphysics.cu`, Kessler saturation-adjustment latent heating now feeds back into `theta` consistently by converting final `T` back to `theta`.

These were not cosmetic. They improved the full-domain HRRR case at `15 min`, `30 min`, and `1 hr`.

### 3. Diagnostics / plotting fixes landed

- `src/io/netcdf_output.cu` now writes real Lambert `COSALPHA/SINALPHA` instead of `1/0` everywhere.
- `tools/plot_weather.py` no longer presents a mislabeled “500 hPa” streamplot panel; it now shows a geometric-height upper-level vector panel more honestly.
- `tools/render_wrf_products.py` already had the custom fallback for `T2/dp2m/rh2m`; that remains in place because native `wrf-rust` `field:T2` was blanking on our files.

## Current Best Reference Runs

### Main current branch run

Current best long HRRR-initialized full-domain run:

- Run dir: `run-fast/hrrr_4km_1h_pgaudit_20260321_185006`
- Init time: `2026-03-21 23:00:00 UTC`
- Outputs:
  - `output/gpuwm_000001.nc` at `2026-03-21 23:30:00 UTC`
  - `output/gpuwm_000002.nc` at `2026-03-22 00:00:00 UTC`

Verification against the HRRR-derived terrain-following init:

- `30 min` (`gpuwm_000001.nc`)
  - `mean_w = +0.7052`
  - `mean|w| = 2.1832`
  - `max|w| = 38.21`
  - `U rmse = 7.62`
  - `V rmse = 6.20`
  - `THETA rmse = 20.72`
  - `QV rmse = 0.00051`

- `1 hr` (`gpuwm_000002.nc`)
  - `mean_w = +0.0701`
  - `mean|w| = 2.5922`
  - `max|w| = 38.20`
  - `U rmse = 12.18`
  - `V rmse = 9.68`
  - `THETA rmse = 31.67`
  - `QV rmse = 0.00073`

Runtime:

- simulated `3600 s` in `868.04 s`
- about `4.1x` real-time on the RTX 5090

### Older baseline for comparison

Old worse hour run:

- Run dir: `run-fast/hrrr_4km_1h_30m_20260321_165723`

Old verification:

- `30 min`
  - `mean_w = +0.4509`
  - `mean|w| = 2.3432`
  - `U/V/THETA/QV rmse = 9.70 / 10.73 / 25.30 / 0.00076`

- `1 hr`
  - `mean_w = +0.0778`
  - `mean|w| = 2.7092`
  - `U/V/THETA/QV rmse = 12.71 / 12.75 / 33.80 / 0.00082`

Interpretation:

- the new branch is materially better at `30 min`
- still better at `1 hr`, though the gain is smaller there
- upper air is still not good enough, but it is no longer simply catastrophic by `30 min`

### Shorter HRRR run useful for quick comparison

Old 15-minute comparison target:

- Run dir: `run-fast/hrrr_4km_15m_fixed_20260321_163609`
- `15 min` verification:
  - `mean_w = +2.2415`
  - `mean|w| = 3.0361`
  - `U/V/THETA/QV rmse = 7.75 / 10.41 / 14.60 / 0.00073`

New 15-minute branch:

- Run dir: `run-fast/hrrr_4km_15m_pgaudit_20260321_184105`
- `15 min` verification:
  - `mean_w = +1.9994`
  - `mean|w| = 2.8888`
  - `U/V/THETA/QV rmse = 4.36 / 4.74 / 11.43 / 0.00034`

## Current Plot Artifacts To Look At

### New 30-minute plots

Run: `run-fast/hrrr_4km_1h_pgaudit_20260321_185006`

Upper air:

- `wrf_products_30m/500mb_temperature_height_winds.png`
- `wrf_products_30m/500mb_rh_height_winds.png`
- `wrf_products_30m/850mb_temperature_height_winds.png`
- `wrf_products_30m/850mb_rh_height_winds.png`
- `wrf_products_30m/field_slp.png`

Surface:

- `wrf_products_30m_surface/field_t2.png`
- `wrf_products_30m_surface/field_dp2m.png`
- `wrf_products_30m_surface/field_rh2m.png`

Native panels:

- `plots_30m/sfc_temp_t00001800.png`
- `plots_30m/analysis_t00001800.png`

### New 1-hour plots

Upper air:

- `wrf_products_1h/500mb_temperature_height_winds.png`
- `wrf_products_1h/500mb_rh_height_winds.png`
- `wrf_products_1h/850mb_temperature_height_winds.png`
- `wrf_products_1h/850mb_rh_height_winds.png`
- `wrf_products_1h/field_slp.png`

Surface:

- `wrf_products_1h_surface/field_t2.png`
- `wrf_products_1h_surface/field_dp2m.png`
- `wrf_products_1h_surface/field_rh2m.png`

Native panels:

- `plots_1h/sfc_temp_t00003600.png`
- `plots_1h/analysis_t00003600.png`

## Important Plotting / Diagnostic Caveats

### Trustworthy vs not

Most trustworthy current upper-air plots:

- WRF-style `500mb_temperature_height_winds`
- WRF-style `500mb_wind_speed_height`
- WRF-style `500mb_rh_height_winds`

Conditionally trustworthy:

- WRF-style `850 mb` products
  - over high terrain these are partly below-ground artifacts
  - over central/eastern lower terrain they are more useful

Not fully trustworthy / special notes:

- native `plot_weather.py` left upper-air panel is now more honest, but it is still a native diagnostic, not a true isobaric WRF-style field
- surface `T2/dp2m/rh2m` maps are **custom fallback renders**, not native `wrf-rust` products
  - this is because `wrf-rust` `field:T2` was producing blank/bad output on our current files
  - the field itself is fine; the renderer path was the issue
- WRF-style wind vectors should be more trustworthy now that `COSALPHA/SINALPHA` are fixed, but this is worth keeping an eye on

### Output schema notes

Current output already includes useful native fields:

- `PRESSURE`
- `TEMP`
- `RHO`
- `RH`
- WRF-compatible `P/PB/PH/PHB/T/QVAPOR/...`

The output audit still recommends eventually adding even more native diagnostic fields with explicit validity masks to reduce dependence on the WRF shim.

## Current Init / Data Situation

### HRRR ingest status

Native HRRR ingest is working now.

Relevant files:

- pressure GRIB: `data/hrrr/hrrr.t23z.wrfprsf00.grib2`
- surface GRIB: `data/hrrr/hrrr.t23z.wrfsfcf00.grib2`

Fast full-domain HRRR init used for current tests:

- `data/hrrr_init_fast_20260321_t23z_retry_1350x795x50_dx4000_latp38.5_lonm097.5_terrain.bin`

Important:

- the heavy vertical interpolation/write path is no longer Python-loop bound
- it was moved into `tools/rust-init-writer/`
- full `1350x795x50` terrain-following HRRR init generation is now practical

## What Is Actually Blocking Further Progress

### Primary blocker

The main remaining blocker is still the dycore pressure/mass closure.

The exploration audit’s most important conclusion:

- the solver is still effectively collocated for pressure/momentum
- `w` storage and boundary plumbing are now closer to interface-form, but the actual acoustic / continuity / PG operators still behave too much like a mass-level formulation
- the vertical acoustic / continuity / PG closure is therefore still wrong by construction for a terrain-following split-explicit core

This is the next real structural task.

### Secondary but real blockers

- physics still uses inconsistent vertical assumptions in places
  - PBL and microphysics still have flat-level / 1D reference-state assumptions in parts of the code
- 850 mb diagnostics remain terrain-contaminated because below-ground interpolation is not masked
- surface products are still proxy-style, not a full land-surface/surface-layer system

## Recommended Next Work

Order these roughly as follows:

1. **Vertical pressure/mass-core rewrite**
   - highest-value next step
   - make `w` a true `nz+1` interface field
   - use matched flux-form operators for continuity and acoustic PG
   - remove dependence on “pressure retain” / sanitizers as the main line of defense

2. **Terrain-aware physics consistency cleanup**
   - align Kessler/Thompson sedimentation and PBL with terrain-aware local `z`, `dz`, `p_ref`, `rho_ref`
   - right now dynamics is more terrain-aware than parts of physics

3. **Plot-quality cleanup**
   - keep `wrf-rust` for upper air
   - make the custom surface fallback renderer visually match `wrf-rust` better
   - ideally avoid touching the `wrf-rust-plots` repo itself

4. **Diagnostic QC**
   - add coarse-grid NaN/Inf/physical-range scanning to the writer and/or preflight
   - report bad lat/lon regions explicitly when a file is unhealthy

5. **Boundary-forcing progression**
   - once the core is steadier, rerun with real time-varying parent boundaries
   - for CONUS regional realism, one init and no evolving lateral boundaries is not the right long-term architecture

## Commands That Currently Matter

Build:

```bash
cmake --build /mnt/c/Users/drew/gpu-wm/build-wsl -j8
```

Run current full-domain HRRR 1-hour test:

```bash
python3 /mnt/c/Users/drew/gpu-wm/tools/run_fast_case.py \
  --init /mnt/c/Users/drew/gpu-wm/data/hrrr_init_fast_20260321_t23z_retry_1350x795x50_dx4000_latp38.5_lonm097.5_terrain.bin \
  --regional-4km-large \
  --tend 3600 \
  --output-interval 1800 \
  --skip-init-plot \
  --skip-verify \
  --tag hrrr_4km_1h_pgaudit
```

Verify outputs against the HRRR-derived init:

```bash
python3 /mnt/c/Users/drew/gpu-wm/tools/verify_forecast.py \
  --reference /mnt/c/Users/drew/gpu-wm/data/hrrr_init_fast_20260321_t23z_retry_1350x795x50_dx4000_latp38.5_lonm097.5_terrain.bin \
  /mnt/c/Users/drew/gpu-wm/run-fast/hrrr_4km_1h_pgaudit_20260321_185006/output/gpuwm_000001.nc \
  /mnt/c/Users/drew/gpu-wm/run-fast/hrrr_4km_1h_pgaudit_20260321_185006/output/gpuwm_000002.nc
```

Render WRF-style upper-air products:

```bash
cmd.exe /c py -3 "$(wslpath -w /mnt/c/Users/drew/gpu-wm/tools/render_wrf_products.py)" \
  --input "$(wslpath -w /mnt/c/Users/drew/gpu-wm/run-fast/hrrr_4km_1h_pgaudit_20260321_185006/output/gpuwm_000002.nc)" \
  --output-dir "$(wslpath -w /mnt/c/Users/drew/gpu-wm/run-fast/hrrr_4km_1h_pgaudit_20260321_185006/wrf_products_1h)"
```

Render custom surface fallback products:

```bash
cmd.exe /c py -3 "$(wslpath -w /mnt/c/Users/drew/gpu-wm/tools/render_wrf_products.py)" \
  --input "$(wslpath -w /mnt/c/Users/drew/gpu-wm/run-fast/hrrr_4km_1h_pgaudit_20260321_185006/output/gpuwm_000002.nc)" \
  --output-dir "$(wslpath -w /mnt/c/Users/drew/gpu-wm/run-fast/hrrr_4km_1h_pgaudit_20260321_185006/wrf_products_1h_surface)" \
  --products field:t2 field:dp2m field:rh2m
```

Call native plotting directly as a function if a custom output dir is needed:

```bash
python3 - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location('plot_weather_mod', '/mnt/c/Users/drew/gpu-wm/tools/plot_weather.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.plot_weather(
    '/mnt/c/Users/drew/gpu-wm/run-fast/hrrr_4km_1h_pgaudit_20260321_185006/output/gpuwm_000002.nc',
    '/mnt/c/Users/drew/gpu-wm/run-fast/hrrr_4km_1h_pgaudit_20260321_185006/plots_1h'
)
PY
```

## Practical Notes For Claude / Next Agent

- Do not assume the surface fields being “okay” means the model is okay. The upper-air drift is still the core issue.
- Do not spend another long cycle on tiny knobs first. The exploration audit already ruled that out as the main path.
- The next big pay-off is the vertical pressure/mass-core rewrite.
- Keep using the new HRRR init path and the current hour-run benchmark when testing major solver changes.
- If evaluating by eye, prefer the `500 mb` WRF-style products and the verified metrics over the `850 mb` plots in mountain regions.
- If touching plotting only, do not modify `wrf-rust-plots` unless truly necessary; keep compatibility work inside `gpu-wm`.
