# GPU-WM Status

Last updated: 2026-04-01

## Current Position

`gpu-wm` is a real CUDA regional-model prototype. It is still below WRF-quality on forecast realism and robustness, but it has now crossed an important threshold on the active branch: the eastern Pennsylvania `768 x 640 x 50 @ 4 km` real-data case can survive to `+1 h` instead of catastrophically blowing up.

The strongest public baseline on `main` is:

- commit: `d92bef2`
- profile: `wdamp-erf-pure`
- key runtime knobs:
  - `--w-damp --w-damp-alpha 6.0 --w-damp-beta 0.0`
  - `--w-transport-blend 1.0`

## Latest External Review Summary

The latest GPT-5.4 Pro review on `main@d92bef2` concluded:

- the architecture is viable and does not need to be thrown away
- the main gap to WRF-like quality is still dycore correctness
- the worst remaining inconsistencies are mixed `w` semantics:
  - some operators treat `w` as interface `eta-dot`
  - others still behave like `w` is a mass-level field
- the second major robustness gap is contradictory open-boundary / sponge behavior
- initialization still needed a continuity-consistent startup `w`
- new physics should not be the next focus

## Active Experimental Branches

### `main`

- commit: `d92bef2`
- includes:
  - pure ERF-style interface `w` transport
  - promoted `wdamp-erf-pure` gate profile
  - free-stream-over-terrain regression case

### `exp/pfree-boundary-sponge`

- commit: `6f134e2`
- change:
  - stop relaxing `p'` toward parent state at open boundaries
  - stop laterally damping `p'` in `apply_sponge()`
- result:
  - canonical uniform/stretched terrain-following gates improved materially

### `exp/openbc-no-w-relax`

- commit: `f77a2b8`
- changes of interest:
  - stop relaxing open-boundary interface `w` toward the parent/boundary snapshot
  - startup-balanced interface `w` initialization is active for loaded real-data states
  - startup balance diagnostics now report raw vs balanced top-interface residuals and post-correction divergence
  - tracked cluster-worker bootstrap and queue docs are now in-repo
- current result:
  - this is the first branch to keep the eastern Pennsylvania `768 x 640 x 50 @ 4 km` case numerically healthy through `+1 h`

## Best Verified Canonical Gate Result So Far

The strongest canonical-gate breakthrough so far came from the pressure-free boundary / sponge cleanup.

Representative results on the 64x64x20 canonical gates:

- `uniform_120`: `U=0.28 V=0.26 TH=0.46 mean|w|=5.36 max|w|=42.35`
- `stretch_120`: `U=1.04 V=1.37 TH=8.09 mean|w|=4.46 max|w|=35.46`
- `stretch_900`: `U=2.76 V=4.11 TH=15.13 mean|w|=2.63 max|w|=19.78`
- `stretch_3600`: `U=4.82 V=5.38 TH=17.49 mean|w|=1.77 max|w|=17.25`
- `stretch_21600`: `U=5.33 V=5.00 TH=11.41 mean|w|=1.15 max|w|=11.63`

These matched on both:

- local RTX 5090
- remote H100 NVL

## Current Regional Reality

Medium real-data regional runs are now materially better on `exp/openbc-no-w-relax`.

The eastern Pennsylvania 4 km terrain-following regional setup:

- grid: `768 x 640 x 50`
- data: `2026-04-01 00z` GFS-derived init
- horizon tested: `+1 h`

Current behavior:

- the active branch survives to `+1 h` on both the local RTX 5090 and the remote H100
- representative `dt=8`, `alpha=6.0`, `blend=1.0` results at `+1 h`:
  - `mean_w = +0.506 m/s`
  - `mean|w| = 1.697 m/s`
  - `max|w| = 18.28 m/s`
  - `U/V/THETA rmse = 12.60 / 8.56 / 33.03`
- the local boundary-forced run and the remote static run matched closely at this horizon

Interpretation:

- freeing lateral `w` was a major regional-stability lever
- startup-balanced interface `w` removed the old catastrophic `+1 h` failure mode, but has looked mostly neutral on the surviving `+1 h` branch so far
- the project is now in a "survives but drifts" regime instead of an "instant blow-up" regime on this case
- the next problem is quality and longer-horizon robustness, not immediate `w/p` runaway at `+1 h`

Additional regional signal:

- the best regional control so far is the H100 static `dt=6`, `alpha=6.0`, `blend=1.0` run:
  - `mean_w = +0.240 m/s`
  - `mean|w| = 1.085 m/s`
  - `max|w| = 34.58 m/s`
  - `U/V/THETA rmse = 11.09 / 8.23 / 29.08`
- stronger damping at `alpha=8.0` did not clearly beat `alpha=6.0`
- static and boundary-forced `dt=8`, `alpha=6.0` runs matched closely at `+1 h`, which weakens the case that the first-hour failure is boundary-driven

Ops note:

- a real H100 idle gap happened once because the remote queue drained while the watchdog stayed alive
- the worker system now treats queue underflow as a fault and reseeds from tracked fallback queues instead of letting paid nodes sit idle
- serious regional runs can now auto-write `verify_all.json`, weather panels, and a collage via `tools/run_fast_case.py --postprocess-weather`
- `tools/run_fast_case.py` no longer hard-requires a GRIB file when an explicit existing `--init` is being reused without `--regen-init`; this matters for staging prebuilt regional binaries on rented workers and then launching runs directly

## Current Drift Diagnosis

The current branch is no longer dominated by immediate boundary blow-up. The remaining drift looks mostly interior and terrain-coupled.

Evidence from the eastern Pennsylvania `+1 h` case:

- static and boundary-forced runs are very close at `+1 h`
- errors are worse over high terrain than low terrain
- interior RMSE is worse than the outer relaxation band
- `mean_w` and `THETA` drift grow monotonically rather than exploding in a late burst
- domain `qtot` loss is large enough to suspect scalar/moisture transport drift, not just boundary export
- new regional band diagnostics in `tools/verify_forecast.py` show the current `dt=8`, `alpha=6.0` case loses much more moisture in the interior than in the outer 20-cell band:
  - `outer_20 qtot_d = -7.37%`
  - `interior qtot_d = -24.92%`
- new terrain-band diagnostics show the moisture loss is not concentrated on the highest terrain:
  - interior low-to-mid terrain quartiles (`zbar ~ 0 / 38 / 198 m`) lose `-28.53% / -35.88% / -30.29%` `qtot`
  - highest-terrain interior quartile (`zbar ~ 404 m`) has the worst `THETA` drift and `|w|`, but only `-0.84%` `qtot` drift
  - interpretation: thermal/w error is strongly terrain-coupled, but the moisture-loss mechanism looks more like interior scalar transport / tracer inconsistency than a simple steep-slope moisture leak
- startup-balanced `w` changes the initial state and startup diagnostics, but has not yet separated the `+1 h` metrics strongly on the tested `dt=8`, `alpha=8.0` case

Most likely current blocker order:

1. remaining slow-path `w` kernels that still behave like mass-level operators over terrain
2. scalar/moisture transport drift, especially domain `qtot` loss on regional runs; this now points first at `advection_scalar_kernel()` vertical/tracer consistency rather than another steep-terrain boundary tweak
3. contradictory boundary / sponge semantics, now secondary at `+1 h` but still wrong
4. startup imbalance, now lower priority than the interior drift

That means the next high-value implementation target is still dycore correctness, not new physics.

## Fresh Moonshot Directions

Three fresh moonshot reviews are now concrete enough to test, in this order:

1. conservative moisture transport only for `qv/qc/qr`
   - rationale: current regional `qtot` loss is much more consistent with a scalar transport/operator mismatch than a boundary leak
   - smallest branch: leave `theta` alone, add a moisture-only conservative transport kernel aligned with the transformed continuity skeleton already used by pressure
2. pressure-only radiative fast open boundary
   - rationale: if a boundary experiment is going to matter, the cleanest first try is to make `p` less reflective during the fast acoustic refresh, not to fake a lateral `p-w` characteristic pair
   - smallest branch: fast-step `p` strip-history / Orlanski-style radiation only; leave lateral `w` unchanged at first
3. columnwise semi-implicit vertical acoustic `p-w` corrector
   - rationale: the smallest serious numerics jump that could beat another damping sweep is a one-thread-per-column tridiagonal solve for the stiff vertical acoustic pair
   - this is larger than the first two ideas and should stay behind them unless the smaller transport/boundary experiments flatline

## Most Recent Prototype Result

The first moisture-only conservative transport prototype is now falsified on the canonical gates.

Branch:

- `exp/moisture-conservative-transport`

Result on H100 short gates:

- `uniform_120`: `U/V/TH = 0.14 / 0.15 / 0.46`, but `mean|w| = 8.04` failed the `6.5` gate
- `stretch_120`: `U/V/TH = 1.05 / 1.38 / 8.16`, but `mean_w = +6.90 m/s`, `mean|w| = 6.90`
- `stretch_900`: `U/V/TH = 2.68 / 3.97 / 15.21`, but `mean|w| = 4.96` failed the `4.0` gate
- worst signal: moisture drift flipped sign and exploded:
  - `stretch_900 outer_20 qtot_d = +29.92%`
  - `stretch_900 interior qtot_d = +88.95%`

Interpretation:

- the full transformed conservative moisture rewrite is not just neutral; it is wrong in its current form
- the failure is strong enough that it should not be queued behind more regional runs
- the right fallback is the narrower variant suggested by the moonshot review:
  - keep horizontal scalar advection unchanged
  - test only a vertical moisture-flux replacement using true interface `w` / `eta_w` divergence

## Immediate Next Work

The next implementation target is:

1. finish the `+1 h` confirmation matrix now in progress on both the local 5090 and the H100:
   - boundary-forced `dt=6`, `alpha=6.0`
   - boundary-forced `dt=8`, `alpha=6.0`
   - static `dt=6` follow-ups only if they buy real quality, not just margin
2. patch the remaining slow-path `w` kernels to true interface semantics, starting with:
   - `pressure_gradient_kernel()` `w_tend`
   - `buoyancy_kernel()`
   - dedicated interface-aware `w` diffusion / sanitize / Rayleigh
3. add one interior-vs-outer budget diagnostic for `qtot` and `THETA` so regional drift is measurable, not guessed
4. keep startup diagnostics in place, but do not treat startup balance alone as the main explanation anymore
5. after the slow `w` cleanup, revisit a dedicated interface-aware lateral `w` boundary operator instead of generic scalar semantics

## Experiment Discipline

Any change should be judged in this order:

1. free-stream-over-terrain regression
2. canonical 64x64x20 uniform/stretched gates
3. eastern Pennsylvania `+1 h` regional case
4. only then larger or longer real-data runs

The project is not blocked by missing sophisticated physics right now. It is blocked by making the solver numerically boring on real-data terrain-following regional domains and then making those surviving runs look meteorologically cleaner.
