# Contributing to GPU-WM

`gpu-wm` is a CUDA-first attempt at a WRF/HRRR-style limited-area numerical weather model built from scratch.

The project goal is not a toy visualizer. The target is a real regional forecast core with terrain-following coordinates, compressible dynamics, real-data initialization, NetCDF output, and a path toward broader accelerator backends later.

Coding agents and human contributors are welcome. The best contributions are the ones that reduce structural uncertainty in the solver, not cosmetic cleanup.

## Current Priority

The highest-priority work is the vertical `w` / pressure / continuity closure in the dycore.

What is already in place:

- GPU runtime and RK loop
- real-data HRRR/GFS init pipeline
- terrain-following init generation
- NetCDF output and verification
- fast vertical acoustic/pressure pair rewritten toward interface `w`
- stretched-eta init loading fixed to reconstruct `eta_w` consistently from loaded `z_levels`
- terrain-aware reference rebuild fixed to sample terrain-following columns using actual `eta_m`

What is still blocking a stable stretched terrain-following dycore:

- much of the slow `w` path still behaves like `w` is a mass-level field
- `w` initialization / conversion / forcing / damping remain only partially stagger-consistent
- stretched-eta runs still drift vertically much harder than uniform runs

## Best First Tasks

If you are picking up the repo cold, start here:

1. `src/core/dynamics.cu`
2. `src/core/init.cu`
3. `src/core/boundaries.cu`
4. `src/core/sponge.cu`
5. `tools/run_fast_case.py`
6. `tools/verify_forecast.py`

The smallest high-leverage rewrite sequence is:

1. convert `convert_w_to_contravariant_kernel()` to true interface semantics
2. convert `buoyancy_kernel()` to true interface semantics
3. fix `pressure_gradient_kernel()`'s `w_tend` metric coupling for interface `w`
4. convert `w` diffusion / Rayleigh damping / sanitize to interface semantics
5. only then convert the `w` branch of momentum advection

Avoid spending cycles on tuning coefficients before those structural issues are resolved.

## Fast Iteration Workflow

Build:

```bash
cmake -S . -B build-wsl
cmake --build build-wsl -j8
```

Uniform terrain-following smoke:

```bash
python3 tools/run_fast_case.py \
  --grib data/hrrr/hrrr.t23z.wrfprsf00.grib2 \
  --surface-grib data/hrrr/hrrr.t23z.wrfsfcf00.grib2 \
  --build build-wsl/gpu-wm \
  --nx 64 --ny 64 --nz 20 --dx 3000 \
  --dt 10 --tend 120 --output-interval 120 \
  --terrain-following-init
```

Stretched terrain-following smoke:

```bash
python3 tools/run_fast_case.py \
  --grib data/hrrr/hrrr.t23z.wrfprsf00.grib2 \
  --surface-grib data/hrrr/hrrr.t23z.wrfsfcf00.grib2 \
  --build build-wsl/gpu-wm \
  --nx 64 --ny 64 --nz 20 --dx 3000 \
  --dt 10 --tend 120 --output-interval 120 \
  --terrain-following-init --stretched-eta
```

Success is not "the plot looks okay." Track:

- `mean_w`
- `mean|w|`
- `max|w|`
- full-field `U/V/THETA/QV` RMSE

The stretched case should move toward the uniform case, not away from it.

Current branch-level gate runner:

```bash
python3 tools/run_gate_matrix.py --profile wdamp --include-6h
```

That runs the canonical 64x64x20 terrain-following cases and fails if the
current `w`/RMSE envelope regresses. Use `--no-enforce` for comparison-only runs.

## Precision Stance

Do not assume full `fp64` everywhere is required for the model to be useful.

Current recommended stance:

- `fp32` is acceptable for the main 3D prognostic state while the dycore is still being stabilized
- `fp64` should be used for geometry, hydrostatic/reference reconstruction, vertical-coordinate generation, reductions, and other numerically sensitive accumulations
- `fp16` is not an appropriate default precision for the core compressible terrain-following dycore

Practical implication:

- keep the solver numerically honest in mixed precision first
- preserve an optional `USE_DOUBLE=ON` path for verification and sensitivity checks
- do not optimize for tensor-core-style half precision until the staggered dycore is structurally correct

## Contribution Style

- Prefer small, testable structural changes over broad speculative rewrites
- Keep build and smoke-test commands runnable after your change
- If you change vertical staggering semantics, rerun both uniform and stretched 120 s fast cases
- If a change only helps the uniform case but not stretched terrain-following, assume the core issue is still unresolved

## Near-Term Goal

The immediate target is not "a perfect WRF clone."

The immediate target is:

- a genuinely stagger-consistent GPU dycore
- stable short real-data terrain-following runs on stretched vertical levels
- enough numerical integrity that bigger physics and backend questions are worth solving
