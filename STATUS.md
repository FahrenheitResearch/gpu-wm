# GPU-WM Status

Last updated: 2026-04-02

## Current Position

`gpu-wm` is a real CUDA regional-model prototype. It is still below WRF-quality on full forecast realism and physics breadth, but the current solver branch has now crossed the threshold where the drycore is no longer the main blocker.

The active baseline is `exp/semiimplicit-hdiv-half@df9586b`. That branch now:

- keeps the East-PA terrain-stress benchmark healthy through `+6 h`
- keeps both static and boundary-forced East-PA runs in the same clean regime
- fixes the stretched canonical ladder well enough that the old solver tradeoff is no longer dominant
- keeps the 4 km Panhandles HRRR realism benchmark plausible through `+3 h`

That means the project is now in solver-consolidation mode rather than solver-triage mode. The next major leverage point is surface realism, especially near-surface thermodynamics and proper surface-layer diagnostics, not another broad dycore rewrite unless the longer validations fail.

The first realism branch is now active:

- branch: `exp/tskin-slab@fa7ea3d`
- change:
  - add prognostic 2D `tskin`
  - update it once per physics step in `src/physics/surface_slab.cu`
  - feed PBL surface exchange with evolving `tskin`
  - write `TSK` to NetCDF
- early result:
  - the local Panhandles boundary `+3 h` rerun stayed effectively neutral versus
    the clean `hdiv_half` baseline on existing `T2/Q2/U10/V10` outputs
  - direct branch-vs-baseline deltas through `+3 h` are still tiny:
    - `T2 mean|delta| <= 0.0026 K`
    - `Q2 delta = 0` at float precision
    - `U10 mean|delta| <= 0.102 m/s`
    - `V10 mean|delta| <= 0.094 m/s`
  - but `TSK` itself is evolving, with domain-mean `TSK` rising from
    `297.40 K` to `297.85 K` by `+3 h`
- interpretation:
  - the slab branch is safe enough to keep pushing
  - but better near-surface diagnostics and/or stronger surface-energy coupling
    are still likely required before the Panhandles `T2/RH2` realism visibly
    changes

The most important new realism result is now diagnostic rather than solver-side:

- the branch now writes a true screen-level `T2/RH2` diagnostic alongside
  retained legacy `*_LML` proxies
- a clean Panhandles boundary `+3 h` A/B confirmed that the old output path was
  hiding most of the near-surface signal:
  - diagnosed `T2` changed by about `1.87 K` mean absolute difference over
    `+1/+2/+3 h`
  - diagnosed `RH2` changed by about `8.25` points mean absolute difference
  - retained `T2_LML`, `U10`, `V10`, and reflectivity-proxy fields stayed
    identical to the control run
- interpretation:
  - the first real breakthrough is that the benchmark was partly apples-to-
    oranges before
  - this does not solve surface physics by itself, but it removes a fake
    handicap and makes the next realism experiments honest

The realism dashboard is now being formalized instead of judged only from GIFs:

- new tool: `tools/verify_surface_realism.py`
- current coverage:
  - matched-product `T2/RH2/WIND10` screen-vs-`*_LML` skill
  - lead-time windows: `0-1 h`, `1-3 h`, `3-6 h`, `1-6 h`
  - terrain bins: `plains`, `moderate_relief`, `steep_high`
  - anomaly correlation and spread ratio
  - coarse solar-phase split
- workflow:
  - `tools/run_fast_case.py` now auto-writes
    `surface_realism.json` and `surface_realism.md`
    for HRRR-backed runs when matching `wrfsfcfXX` files are present
- first smoke result on Panhandles `+1/+2/+3 h`:
  - domain `T2 RMSE = 3.50` vs `4.99` for `T2_LML`
  - domain `RH2 RMSE = 11.48` vs `15.56` for `RH2_LML`
  - domain `10 m wind vector RMSE = 4.71` vs `9.81` for `U10_LML/V10_LML`
  - mountain wind remains the main weak spot:
    - `steep_high WIND10 RMSE = 5.84`
    - `steep_high anomaly corr = 0.70`

The next active realism push is now a tunable slab-coupling sweep:

- `surface_slab.cu` now accepts runtime controls for:
  - slab heat capacity
  - restore coefficient toward the prior surface state
  - anchor weight back to the diagnosed column surface state
- overnight queues are set to test stronger slab settings on Panhandles
  boundary/static follow-ons once the current `+12 h` screen-diagnostic runs
  finish

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
- East-PA static `+3 h` on H100 NVL:
  - `U/V/THETA rmse = 6.29 / 8.61 / 10.68`
  - `mean|w| = 0.1070`
  - `max|w| = 5.49`
  - `outer_20 qtot_d = -0.37%`
  - `interior qtot_d = -4.67%`
- East-PA boundary-forced `+3 h` on H100 80GB:
  - `U/V/THETA rmse = 6.24 / 8.81 / 10.79`
  - `mean|w| = 0.1072`
  - `max|w| = 5.01`
  - `outer_20 qtot_d = +0.18%`
  - `interior qtot_d = -4.48%`
- interpretation:
  - this is the first semi-implicit refinement after `24d7eaf` that preserves
    the strong East-PA regional result while also fixing the stretched
    canonical ladder
  - the branch now also holds that gain through `+3 h` on both static and
    boundary-forced East-PA runs

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
- `exp/semiimplicit-hdiv-half` `+1 h` on H100 80GB:
  - `U/V/THETA rmse = 2.25 / 2.67 / 5.58`
  - `mean|w| = 0.116`
  - `max|w| = 4.86`
  - `outer_20 qtot_d = +0.07%`
  - `interior qtot_d = -0.42%`
- `exp/semiimplicit-hdiv-half` `+2 h` on H100 80GB:
  - `U/V/THETA rmse = 3.45 / 4.96 / 7.03`
  - `mean|w| = 0.122`
  - `max|w| = 5.08`
  - `outer_20 qtot_d = +0.80%`
  - `interior qtot_d = +0.47%`
- `exp/semiimplicit-hdiv-half` `+3 h` on H100 80GB:
  - `U/V/THETA rmse = 4.71 / 7.32 / 8.55`
  - `mean|w| = 0.129`
  - `max|w| = 6.19`
  - `outer_20 qtot_d = +1.52%`
  - `interior qtot_d = +1.35%`
- interpretation:
  - the `hdiv_half` branch is not just surviving a terrain-stress case
  - it is also producing a plausible short-range 4 km Plains benchmark from
    latest HRRR data through `+3 h`

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
  - preserves the East-PA static and boundary `+1 h` regional win:
    - static `+1 h`: `U/V/THETA = 2.65 / 3.59 / 7.18`, `mean|w| = 0.0857`
    - boundary `+1 h`: `2.66 / 3.60 / 7.20`, `mean|w| = 0.0859`
  - now also carries the same East-PA setup through `+6 h`:
    - static `+6 h`: `U/V/THETA = 11.79 / 12.48 / 16.24`, `mean|w| = 0.1429`, `max|w| = 6.78`, `interior qtot_d = -8.14%`
    - boundary `+6 h`: `11.66 / 13.34 / 16.51`, `mean|w| = 0.1435`, `max|w| = 6.27`, `interior qtot_d = -7.56%`
  - still improves the stretched canonical ladder:
    - `stretch_900`: `2.51 / 3.79 / 13.88`, `mean|w| = 3.93`
    - `stretch_3600`: `4.33 / 4.83 / 15.13`, `mean|w| = 2.42`
    - `stretch_21600`: `3.50 / 3.39 / 7.47`, `mean|w| = 0.72`
  - also keeps the 4 km Panhandles HRRR realism benchmark clean:
    - static `+3 h`: `4.71 / 7.32 / 8.55`, `mean|w| = 0.1291`
    - boundary `+3 h`: `4.71 / 7.32 / 8.55`, `mean|w| = 0.1291`
  - current role: best solver baseline, with `+12 h` East-PA and Panhandles
    durability runs in progress

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

Real-data regional runs are now in a different regime than they were on
`exp/openbc-no-w-relax`.

The eastern Pennsylvania terrain-stress benchmark:

- grid: `768 x 640 x 50 @ 4 km`
- data: `2026-04-01 00z` GFS-derived init
- best current branch: `exp/semiimplicit-hdiv-half@df9586b`

Current behavior on that branch:

- static `+1 h`: `U/V/THETA = 2.65 / 3.59 / 7.18`, `mean|w| = 0.0857`
- boundary `+1 h`: `2.66 / 3.60 / 7.20`, `mean|w| = 0.0859`
- static `+3 h`: `6.29 / 8.61 / 10.68`, `mean|w| = 0.1070`
- boundary `+3 h`: `6.24 / 8.81 / 10.79`, `mean|w| = 0.1072`
- static `+6 h`: `11.79 / 12.48 / 16.24`, `mean|w| = 0.1429`, `max|w| = 6.78`
- boundary `+6 h`: `11.66 / 13.34 / 16.51`, `mean|w| = 0.1435`, `max|w| = 6.27`

Interpretation:

- the model is no longer in the old "survives but drifts badly by `+1 h`"
  regime
- the fast vertical pair is now settled enough that both static and
  boundary-forced East-PA runs stay in the same healthy regime through `+6 h`
- there is still forecast error growth, but not the old solver-style runaway

The Oklahoma / Texas Panhandles realism benchmark now matters as a second
validation target, not just a demo:

- grid: `512 x 384 x 50 @ 4 km`
- data: `2026-04-01 19z` HRRR
- `+1 h`: `2.25 / 2.67 / 5.58`, `mean|w| = 0.1158`
- `+2 h`: `3.45 / 4.96 / 7.03`, `mean|w| = 0.1223`
- `+3 h` static: `4.71 / 7.32 / 8.55`, `mean|w| = 0.1291`
- `+3 h` boundary: `4.71 / 7.32 / 8.55`, `mean|w| = 0.1291`

Interpretation:

- the new solver branch is not only holding on a terrain-stress case
- it is also producing plausible short-range Plains structure against HRRR
- the main visible weakness on this benchmark is now surface thermodynamics,
  not the core flow or reflectivity structure

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

The active solver moonshot has now been promoted. The next research-sidecar
work is a realism step, not another broad drycore rewrite.

1. minimal slab surface energy-balance model with prognostic `tskin`
   - rationale: the current solver branch is now good enough that the biggest
     visible weakness is stiff near-surface `T2/RH2` evolution against HRRR
   - smallest branch:
     - add a 2D `tskin`
     - update it with a one-layer surface heat-capacity equation
     - feed it into the existing PBL surface flux path
2. proper near-surface diagnostics
   - rationale: current `T2/Q2/U10/V10` are still lowest-model-level fields,
     not true surface-layer diagnostics
   - smallest branch:
     - add `TSK`
     - separate diagnostic `T2/Q2/U10/V10` from raw lowest-level output
3. wire radiation only after the slab branch exists
   - rationale: the existing radiation module by itself does not create a
     surface thermal memory; it only becomes a clean realism lever after the
     surface has its own state

## Immediate Next Work

The next implementation target is:

1. finish the in-flight `+12 h` durability checks on `exp/semiimplicit-hdiv-half`
   - East-PA static
   - Panhandles static
2. freeze `exp/semiimplicit-hdiv-half` as the working solver baseline unless
   those longer runs reveal a new fast-loop defect
3. start the first small realism branch:
   - add prognostic 2D `tskin`
   - feed it into the existing surface sensible-heat seam in `run_pbl()`
   - add `TSK` output
4. keep the corrected HRRR comparison path in place
   - the GRIB reader now validates cached message offsets so `f02` no longer
     misreads `2 m temperature`
5. only return to fast-loop `u/v` acoustic PG substepping if the `+12 h`
   durability runs expose a new solver-specific failure

## Experiment Discipline

Any change should be judged in this order:

1. free-stream-over-terrain regression
2. canonical 64x64x20 uniform/stretched gates
3. eastern Pennsylvania `+1 h` regional case
4. only then larger or longer real-data runs

The project is no longer blocked primarily by the drycore. The current blocker
is the first layer of realism work on top of the now-usable solver baseline.
