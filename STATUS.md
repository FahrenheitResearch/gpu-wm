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

- commit: `8f445ae`
- changes of interest:
  - stop relaxing open-boundary interface `w` toward the parent/boundary snapshot
  - startup-balanced interface `w` initialization is active for loaded real-data states
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
- startup-balanced interface `w` removed the old catastrophic `+1 h` failure mode
- the project is now in a "survives but drifts" regime instead of an "instant blow-up" regime on this case
- the next problem is quality and longer-horizon robustness, not immediate `w/p` runaway at `+1 h`

## Immediate Next Work

The next implementation target is:

1. finish the `+1 h` confirmation matrix now in progress:
   - H100 boundary-forced `dt=8`
   - local boundary-forced `dt=8`, stronger `w` damping
   - `dt=6` regional follow-ups only if they buy real quality, not just extra margin
2. promote the eastern Pennsylvania case from "survives `+1 h`" to a cleaner `+3 h` / `+6 h` signal
3. keep tightening startup and boundary diagnostics so drift sources are measurable
4. after that, try the smallest dedicated interface-aware lateral `w` boundary operator instead of generic scalar semantics

## Experiment Discipline

Any change should be judged in this order:

1. free-stream-over-terrain regression
2. canonical 64x64x20 uniform/stretched gates
3. eastern Pennsylvania `+1 h` regional case
4. only then larger or longer real-data runs

The project is not blocked by missing sophisticated physics right now. It is blocked by making the solver numerically boring on real-data terrain-following regional domains and then making those surviving runs look meteorologically cleaner.
