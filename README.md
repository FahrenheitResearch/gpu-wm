# GPU-WM

`gpu-wm` is an all-GPU regional weather-model prototype in CUDA aimed at a WRF/HRRR-class limited-area forecast core on NVIDIA hardware.

This repository is not an operational model. It is a research codebase with a working regional forecast pipeline, real-data HRRR/GFS initialization, NetCDF output, and a partially modernized dycore that still has important structural work left.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the current blocker chain, fast-case workflow, and contribution priorities.

## Current Status

- Real-data terrain-following initialization works for HRRR and GFS.
- The model runs fully on GPU and can produce plausible short-range fields.
- NetCDF output and WRF-style plotting/verification are in place.
- The main remaining blocker is the vertical pressure/mass/w closure in the dycore, especially over terrain.

## Most Important Files

- [src/main.cu](src/main.cu): run loop, init loading, boundary blending, physics orchestration
- [src/core/dynamics.cu](src/core/dynamics.cu): RK3 dycore, pressure/acoustic step, terrain metrics
- [src/core/init.cu](src/core/init.cu): base-state and binary init loading
- [tools/init_from_gfs.py](tools/init_from_gfs.py): GRIB to init-binary preprocessing
- [tools/run_fast_case.py](tools/run_fast_case.py): fast iteration harness
- [tools/verify_forecast.py](tools/verify_forecast.py): forecast-vs-reference verification

## Build

WSL/Linux build:

```bash
cmake -S . -B build-wsl
cmake --build build-wsl -j8
```

## Fast Smoke Test

```bash
python3 tools/run_fast_case.py \
  --grib data/hrrr/hrrr.t23z.wrfprsf00.grib2 \
  --surface-grib data/hrrr/hrrr.t23z.wrfsfcf00.grib2 \
  --build build-wsl/gpu-wm \
  --nx 64 --ny 64 --nz 20 --dx 3000 \
  --dt 10 --tend 120 --output-interval 120 \
  --terrain-following-init
```

Experimental stretched-init path:

```bash
python3 tools/run_fast_case.py ... --terrain-following-init --stretched-eta
```

That path is intentionally opt-in because the current dycore is not yet stable enough to make stretched vertical spacing the default on real-data runs.
