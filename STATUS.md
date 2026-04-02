# GPU-WM Status

Last updated: 2026-04-01

## Current Position

`gpu-wm` is a real CUDA regional-model prototype. It is still below WRF-quality on forecast realism and robustness, but the active semi-implicit branch family has now crossed another threshold: the same solver seam that fixes the East-PA real-data `+1 h` case is also starting to line up with the stretched canonical gates instead of trading one benchmark off against the other.

The newest important change is the active `hdiv_half` refinement on top of `exp/semiimplicit-no-slow-metric@24d7eaf`:

- working branch: `exp/semiimplicit-hdiv-half` (local branch built from `24d7eaf`)
- change:
  - keep the coupled semi-implicit `p-w` correction from `24d7eaf`
  - snapshot `p` at the start of each acoustic substep
  - replace raw `generalized_horizontal_divergence(u,v,...)` in the column RHS
    with a half-step `hdiv_half` built from a pressure-gradient predictor off the
    old-pressure snapshot
- East-PA static `+1 h` on H100 NVL:
  - `U/V/THETA rmse = 2.65 / 3.59 / 7.18`
  - `mean|w| = 0.0857`
  - `max|w| = 4.13`
  - `outer_20 qtot_d = +0.76%`
  - `interior qtot_d = +0.25%`
- East-PA boundary-forced `+1 h` on H100 80GB:
  - `U/V/THETA rmse = 2.66 / 3.60 / 7.20`
  - `mean|w| = 0.0859`
  - `max|w| = 4.28`
  - `outer_20 qtot_d = +0.92%`
  - `interior qtot_d = +0.26%`
- stretched canonical gates now also look materially better on the same branch:
  - `stretch_900`: `U/V/THETA = 2.51 / 3.79 / 13.88`, `mean|w| = 3.93`, `max|w| = 19.05`
  - `stretch_3600`: `U/V/THETA = 4.33 / 4.83 / 15.13`, `mean|w| = 2.42`, `max|w| = 15.08`
  - `stretch_21600`: `U/V/THETA = 3.50 / 3.39 / 7.47`, `mean|w| = 0.72`, `max|w| = 8.02`
- interpretation:
  - this is the first semi-implicit refinement after `24d7eaf` that appears to
    preserve the strong East-PA regional result while also fixing the stretched
    canonical ladder
  - East-PA `+3 h` and the Panhandles HRRR rerun are now the decisive
    follow-on validations

The earlier important change was `exp/semiimplicit-pw-column@7c748cb`:

- the first columnwise semi-implicit fast `p-w` prototype is the strongest solver-side regional result yet
- on the eastern Pennsylvania static `+1 h` case at `dt=6`, `alpha=6.0`, `blend=1.0`, it improved materially versus the surviving explicit control:
  - `U/V/THETA rmse = 3.51 / 4.09 / 10.21`
  - `mean|w| = 0.158`
  - `max|w| = 6.28`
  - `outer_20 qtot_burden_d = -0.32%`
  - `interior qtot_burden_d = -4.93%`
- the same branch also now validates on the eastern Pennsylvania boundary-forced `+1 h` case:
  - `U/V/THETA rmse = 3.52 / 4.10 / 10.23`
  - `mean|w| = 0.158`
  - `max|w| = 6.43`
  - `outer_20 qtot_d = -0.10%`
  - `interior qtot_d = -4.27%`
- that is a non-incremental gain over the current explicit control family, which was still closer to:
  - `U/V/THETA ~ 10-11 / 8-9 / 29-30`
  - `mean|w| ~ 1.0`
  - `interior qtot_burden_d ~ -26%`
- the same first prototype is not yet a universal win:
  - the free-stream terrain regression stayed clean
  - but `stretch_900` regressed, with `mean|w| = 4.26` and strongly positive tracer drift
- interpretation:
  - the semi-implicit seam is now the leading solver path
  - but the first backward-Euler prototype still needs refinement before promotion

There is now also a second realism benchmark active on the same branch:

- Oklahoma / Texas Panhandles HRRR real-data case
- grid: `512 x 384 x 50 @ 4 km`
- init/boundary pair: `2026-04-01 19z` HRRR `f00` + `f03`
- local `+1 h` result from the first mirror run:
  - `U/V/THETA rmse = 2.64 / 2.95 / 7.81`
  - `mean|w| = 0.157`
  - `max|w| = 5.18`
  - `outer_20 qtot_burden_d = -0.15%`
  - `interior qtot_burden_d = -1.37%`
- interpretation:
  - the semi-implicit branch is not just surviving a terrain-stress case
  - it is now also producing a plausible short-range 4 km Plains benchmark from latest HRRR data

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

- commit: `ec6685f`
- changes of interest:
  - stop relaxing open-boundary interface `w` toward the parent/boundary snapshot
  - startup-balanced interface `w` initialization is active for loaded real-data states
  - startup balance diagnostics now report raw vs balanced top-interface residuals and post-correction divergence
  - tracked cluster-worker bootstrap and queue docs are now in-repo
- current result:
  - this is the first branch to keep the eastern Pennsylvania `768 x 640 x 50 @ 4 km` case numerically healthy through `+1 h`

### `exp/semiimplicit-pw-column`

- commit: `7c748cb`
- change:
  - add a runtime-gated columnwise semi-implicit replacement for the explicit fast vertical `p-w` trio inside `run_vertical_acoustic_substeps()`
  - keep the existing explicit path intact as the control path
- current result:
  - strongest East-PA static and boundary-forced `+1 h` results so far
  - not ready for promotion yet because the first prototype still regresses `stretch_900`
  - now also the active branch for the 4 km Panhandles HRRR realism benchmark

### `exp/semiimplicit-no-slow-metric`

- commit: `24d7eaf`
- change:
  - keep the semi-implicit column solve
  - move the acoustic pressure-divergence damping inside the implicit path as a
    coupled `p-w` correction
- current result:
  - best pre-`hdiv_half` East-PA static `+1 h` result
  - still left a stretched canonical failure mode, which motivated the next RHS
    time-level refinement

### `exp/semiimplicit-hdiv-half`

- branch base: `24d7eaf`
- change:
  - replace raw explicit `hdiv` in the implicit column RHS with a half-step
    pressure-predicted `hdiv_half`
- current result:
  - strongest cross-benchmark solver branch so far
  - preserves the East-PA static and boundary `+1 h` regional win
  - now also passes `stretch_900`, `stretch_3600`, and `stretch_21600`
  - current role: active best solver branch pending `+3 h` East-PA and
    Panhandles reruns

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
- a longer H100 boundary-forced `dt=6`, `alpha=6.0`, `blend=1.0` control has now been verified through `+1.5 h`:
  - `mean_w = +0.274 m/s`
  - `mean|w| = 1.391 m/s`
  - `max|w| = 32.59 m/s`
  - `U/V/THETA rmse = 14.12 / 11.26 / 37.91`
  - `outer_20 qtot_d = -7.98%`
  - `interior qtot_d = -27.04%`
  - interpretation: the branch is no longer failing by immediate runaway, but the same interior moisture/thermal drift pattern is still present beyond `+1 h`
- the semi-implicit prototype changes that picture sharply on the static `+1 h` regional case:
  - `mean_w = +0.0015 m/s`
  - `mean|w| = 0.1580 m/s`
  - `max|w| = 6.28 m/s`
  - `U/V/THETA rmse = 3.51 / 4.09 / 10.21`
  - `outer_20 qtot_burden_d = -0.32%`
  - `interior qtot_burden_d = -4.93%`
  - interpretation: this is the first branch that looks like a non-incremental solver gain on the real East-PA target, not just another neutral cleanup
- the same branch now matches that gain on the boundary-forced `+1 h` East-PA case:
  - `mean|w| = 0.1583 m/s`
  - `max|w| = 6.43 m/s`
  - `U/V/THETA rmse = 3.52 / 4.10 / 10.23`
  - `outer_20 qtot_d = -0.10%`
  - `interior qtot_d = -4.27%`
  - interpretation: this is no longer just a static-case win

Ops note:

- a real H100 idle gap happened once because the remote queue drained while the watchdog stayed alive
- the worker system now treats queue underflow as a fault and reseeds from tracked fallback queues instead of letting paid nodes sit idle
- the remote worker busy detector was widened to catch `gpu-wm` runs launched from sibling worktrees like `/root/gpu-wm-prad`; before that fix, a node could double-launch a second experiment on top of a manual worktree run
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
- the longer H100 `dt=6` control keeps the same shape of error through `+1.5 h`:
  - `outer_20 qtot_d = -7.98%`
  - `interior qtot_d = -27.04%`
  - interpretation: the branch is surviving longer, but the dominant moisture-loss mechanism is still interior and persistent rather than a short-lived startup transient
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

Updated interpretation after `exp/semiimplicit-pw-column@5126c38`:

1. the unresolved interior fast vertical coupling is now the most promising seam
2. the first semi-implicit prototype still needs refinement so it stops regressing stretched canonical terrain-following gates
3. scalar/moisture transport remains important, but no longer looks like the first lever on the East-PA target

## Rejected Follow-Up Branches

Two moisture-transport follow-ups have now been tested and rejected.

### `exp/moisture-conservative-transport`

- commit: `b07e55b`
- intent:
  - move `qv/qc/qr` to a transformed conservative transport path
- result:
  - failed short canonical gates
  - worst failure mode was a sign flip in regional/canonical moisture drift
  - representative `stretch_900` signal:
    - `outer_20 qtot_d = +29.92%`
    - `interior qtot_d = +88.95%`

### `exp/moisture-vertical-flux`

- commit: `bc76b9d`
- intent:
  - keep horizontal scalar transport unchanged
  - replace only the moisture vertical term with interface-flux divergence
- result:
  - also failed to cure the moisture-drift problem
  - representative `stretch_900` signal:
    - `outer_20 qtot_d = +30.85%`
    - `interior qtot_d = +89.00%`

Interpretation:

- moisture transport is still implicated in the regional drift story
- but these two direct conservative rewrites are not the next lowest-risk branch
- the next cleaner experimental lever is now the fast pressure radiative boundary branch, followed by the columnwise semi-implicit `p-w` corrector idea if that branch is flat or negative

## Fresh Moonshot Directions

Three fresh moonshot reviews are now concrete enough to test, in this order:

1. pressure-only radiative fast open boundary
   - rationale: if a boundary experiment is going to matter, the cleanest first try is to make `p` less reflective during the fast acoustic refresh, not to fake a lateral `p-w` characteristic pair
   - smallest branch: fast-step `p` strip-history / Orlanski-style radiation only; leave lateral `w` unchanged at first
2. columnwise semi-implicit vertical acoustic `p-w` corrector
   - rationale: the smallest serious numerics jump that could beat another damping sweep is a one-thread-per-column tridiagonal solve for the stiff vertical acoustic pair
   - this is larger than the first two ideas and should stay behind them unless the smaller transport/boundary experiments flatline
3. revisit moisture transport only after the failure mechanism is narrower
   - rationale: the direct conservative moisture branches already failed
   - the next moisture experiment should be driven by a sharper mechanism diagnosis, not another blind transport rewrite

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
