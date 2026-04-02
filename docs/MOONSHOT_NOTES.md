# GPU-WM Moonshot Notes

Last updated: 2026-04-01

## Purpose

This is the standing research-sidecar memo for `gpu-wm`.

The goal is not generic "better numerics" advice. The goal is to identify
high-upside ideas that are still close enough to the current codebase to test
in hours or days.

Current grounding:

- `exp/semiimplicit-hdiv-half` is now the best solver baseline.
- The semi-implicit `p-w` seam plus `hdiv_half` refinement carried:
  - East-PA static and boundary runs through `+6 h`
  - stretched canonical gates through `stretch_21600`
  - the 4 km Panhandles HRRR realism benchmark through `+3 h`
- That means the drycore is no longer the main blocker on the active branch.
- The most obvious realism weakness is now near-surface thermodynamics:
  `T2/RH2` evolve too stiffly relative to HRRR while wind and reflectivity look
  more alive.
- Two direct moisture-transport follow-ups already failed:
  - `exp/moisture-conservative-transport@b07e55b`
  - `exp/moisture-vertical-flux@bc76b9d`
  Both flipped the stretched `qtot` drift positive at `stretch_900`, so the
  next moonshot should not be another blind conservative-moisture rewrite.

This memo deliberately separates solver-correctness ideas from realism ideas
 and systems ideas.

## Active Next Moonshots

### 0. Minimal Slab Surface Energy-Balance Model

Category: realism physics

Why it might matter:

- The solver now looks good enough that the biggest visible weakness is stiff
  near-surface temperature and humidity evolution.
- Current `T2/Q2` are effectively lowest-model-level values, and the surface
  flux path in `pbl.cu` still leans on scalar priors instead of a prognostic
  surface thermal memory.
- A cheap slab `tskin` is the smallest way to give the model a real evolving
  surface state without dragging in a full land-surface system.

Files / functions:

- `include/grid.cuh`
- `src/physics/pbl.cu`
- `src/main.cu`
- new `src/physics/surface_slab.cu`
- `src/io/netcdf_output.cu`

Quick falsification:

- Re-run the Panhandles HRRR benchmark to `+6 h`.
- If `T2/RH2` still look just as stationary while wind/reflectivity stay the
  same, kill the idea quickly or shrink it to diagnostics only.

Expected compute impact:

- Small.
- One extra 2D state plus cheap per-column surface math.

### 1. Proper Near-Surface Diagnostics

Category: realism diagnostics

Why it might matter:

- Current `T2/Q2/U10/V10` outputs are still basically lowest-model-level
  fields.
- That is good enough for rough comparison, but not a fair apples-to-apples
  verification target against HRRR surface products.

Files / functions:

- `src/io/netcdf_output.cu`
- `src/physics/pbl.cu`

Quick falsification:

- Add `TSK` and separate 2 m / 10 m diagnostics.
- If the comparison story barely changes, keep the diagnostics but do not treat
  them as the main realism lever.

### 2. Fast Acoustic `u/v` Pressure-Gradient Substepping

Category: solver follow-up, only if needed

Why it might matter:

- If the in-flight `+12 h` durability runs expose a new fast-loop defect,
  the next smallest solver seam is to stop freezing fast horizontal PG effects
  on `u/v` outside the acoustic loop.
- GPT Pro already narrowed this as the next seam only if `hdiv_half` fails at
  longer horizons.

Files / functions:

- `src/core/dynamics.cu::run_vertical_acoustic_substeps`
- extracted `acoustic_horizontal_pg_kernel()`

Quick falsification:

- East-PA static `+1 h` and `+6 h` against the current `hdiv_half` baseline.
- If U/V do not recover further without harming stretched cases, drop it.

## Ranked Ideas (Historical Solver Archive)

The items below are retained for traceability. Most of them are no longer the
active research queue because the semi-implicit `hdiv_half` path has already
promoted the solver past the old failure regime.

### 0. Fast Pressure Radiative Boundary for the Acoustic Refresh

Category: solver correctness

Why it might matter:

- The active regional branch now survives to `+1 h`, which makes the remaining
  error more diagnostic than explosive.
- Boundary-forced and static `+1 h` runs are close, so the lateral boundary is
  no longer the obvious dominant failure, but the fast acoustic refresh is
  still the cleanest remaining boundary lever.
- A pressure-only radiative refresh is smaller and less dangerous than a fake
  characteristic `p-w` pair and can be tested quickly on top of
  `exp/openbc-no-w-relax`.

Files / functions:

- `src/core/dynamics.cu::{open_fast_bc_x_kernel, open_fast_bc_y_kernel}`
- `src/core/dynamics.cu::run_vertical_acoustic_substeps`

Quick falsification:

- Re-run the canonical short gates for regression safety.
- Then re-run the eastern Pennsylvania `+1 h` boundary-forced case.
- If regional `THETA` and `qtot` drift do not improve or the canonical
  `mean|w|` regresses badly, drop it and move on.

Expected compute impact:

- Roughly flat.
- This is a boundary refresh semantics change, not a cost-heavy method.

Skeptical note:

- If the current interior drift is mostly terrain-coupled source/transport
  error, this will be neutral.
- That is still acceptable if it lets us reject the boundary hypothesis
  cheaply.

### 1. Column Balance Startup Solve for `w` and `p'`

Category: solver correctness

Why it might matter:

- `tools/init_from_gfs.py` still writes `w`, `qc`, and `qr` as zeros.
- `src/core/init.cu::load_gfs_binary()` still zeroes interface `state.w` after
  reading the on-disk `w` field, then `src/main.cu` tries to recover a better
  startup state with `initialize_w_from_continuity()`.
- That continuity-only initializer is a good first step, but it does not also
  balance `p'` against the startup divergence and terrain metrics.
- A one-time column startup solve could eliminate the first-step acoustic burst
  more effectively than more damping.

Files / functions:

- `src/core/init.cu::load_gfs_binary`
- `src/core/dynamics.cu::initialize_w_from_continuity_kernel`
- `src/main.cu` startup path after `convert_w_to_contravariant()`
- `tools/run_fast_case.py` for explicit startup-balance toggles

Quick falsification:

- Add startup diagnostics for first 20 timesteps: `mean|w|`, `max|w|`,
  `max|p'|`, and column divergence norms.
- Re-run eastern Pennsylvania `768 x 640 x 50 @ 4 km` to `+1 h`.
- If the first-step burst barely changes and `+1 h` quality is unchanged, this
  idea is not the main lever.

Expected compute impact:

- Startup-only cost.
- Roughly flat runtime after initialization.

Skeptical note:

- High upside because startup is obviously imperfect.
- Could also be mostly neutral if the real problem is boundary reflection or
  slow `w` forcing later in the run.

### 2. Characteristic / Mode-Split Open Boundaries for `p` and `w`

Category: solver correctness

Why it might matter:

- `src/core/boundaries.cu::open_bc_x_kernel()` and `open_bc_y_kernel()` still
  apply the same generic extrapolation logic to almost everything.
- `state.p` and interface `state.w` are not ordinary passive scalars.
- The code is already close enough to a split fast/slow model that a dedicated
  acoustic/gravity-wave exit treatment could matter more than another damping
  coefficient sweep.
- Large regional runs are currently the exact place where bad `p/w` boundary
  semantics should show up first.

Files / functions:

- `src/core/boundaries.cu::{open_bc_x_kernel, open_bc_y_kernel, apply_open_boundaries, refresh_open_halos}`
- `src/core/dynamics.cu::run_vertical_acoustic_substeps`
- `src/main.cu::blend_boundary_state`

Quick falsification:

- Add a gravity-wave packet exit case and compare reflection amplitude near the
  edge for current BCs versus characteristic-style BCs.
- Re-run eastern Pennsylvania `+1 h` and inspect whether growth nucleates at
  the lateral boundaries or outer 10-20 cells.

Expected compute impact:

- Roughly `+5%` to `+15%`.
- Mostly extra logic, not a major memory event.

Skeptical note:

- This is one of the strongest candidates for the large-domain gap.
- If the interior still blows first over terrain gradients, boundary work alone
  will not save the model.

### 3. Columnwise Semi-Implicit `p'-w` Corrector

Category: solver correctness

Why it might matter:

- The current split-explicit fast pair is directionally right, but it still
  leans on damping for stretched stability.
- A per-column semi-implicit corrector for the coupled vertical `p'-w` system
  could let the model stop surviving only because `--w-damp` is strong enough.
- This is the smallest "serious numerical-method" jump that could produce a
  non-incremental gain without rewriting the whole architecture.

Files / functions:

- `src/core/dynamics.cu::{acoustic_vertical_pg_kernel, pressure_update_kernel, run_vertical_acoustic_substeps}`
- `include/stability_control.cuh`

Quick falsification:

- Build a branch that applies the semi-implicit corrector only in the split
  vertical acoustic loop.
- Compare against `wdamp-erf-pure` on `stretch_900`, `stretch_3600`, and the
  eastern Pennsylvania `+1 h` case with reduced `w_damp`.
- If stability does not improve when damping is backed off, the extra solver
  complexity is not earning its keep.

Expected compute impact:

- Roughly `+10%` to `+25%`.
- Small extra temporary storage per column.

Skeptical note:

- This is high upside and real numerics, not hand-tuning.
- It is also the easiest idea here to burn a week on if the algebra is not
  careful.

Current status:

- first prototype now exists on `exp/semiimplicit-pw-column@5126c38`
- it is already a real regional solver win on the East-PA static `+1 h` case:
  - `U/V/THETA rmse = 3.51 / 4.09 / 10.21`
  - `mean|w| = 0.158`
  - `outer_20 qtot_burden_d = -0.32%`
  - `interior qtot_burden_d = -4.93%`
- that same prototype still regresses `stretch_900`
- interpretation:
  - this idea is promoted from moonshot to active mainline solver path
  - the next refinement should target why the regional real-data case improves while the stretched canonical gate regresses

Newest narrowed refinement:

- the most likely remaining seam is the explicit terrain-pressure `w` metric kick
  still added by `pressure_gradient_kernel()` while the column `p-w` solve is on
- working hypothesis:
  - the implicit column solve already handles the stiff vertical `p-w` response
  - the leftover explicit pressure-to-`w` metric source then double-kicks or
    misphases `w` on stretched columns
- active next branch:
  - `exp/semiimplicit-no-slow-metric`
  - scope:
    - when `--semiimplicit-pw` is active, suppress the explicit pressure metric
      `w_tend` contribution from `pressure_gradient_kernel()`
    - keep the rest of the semi-implicit path unchanged
- decisive test:
  - `stretch_900`
  - East-PA static `+1 h`

Newest narrowed refinement:

- latest external review says the next remaining fast-pair defect is most likely
  explicit horizontal-divergence time level, not another filter or `dwdz`
  asymmetry
- active branch:
  - `exp/semiimplicit-hdiv-half`
- scope:
  - keep the coupled-filter `24d7eaf` semi-implicit core
  - snapshot old pressure each acoustic substep
  - replace raw `generalized_horizontal_divergence(u,v,...)` in the column RHS
    with a half-step `hdiv_half` built from a pressure-gradient predictor off
    that snapshot
- current result:
  - East-PA static `+1 h`: `U/V/THETA = 2.65 / 3.59 / 7.18`, `mean|w| = 0.0857`
  - East-PA boundary `+1 h`: `2.66 / 3.60 / 7.20`, `mean|w| = 0.0859`
  - `stretch_900`: `2.51 / 3.79 / 13.88`, `mean|w| = 3.93`
  - `stretch_3600`: `4.33 / 4.83 / 15.13`, `mean|w| = 2.42`
  - `stretch_21600`: `3.50 / 3.39 / 7.47`, `mean|w| = 0.72`
  - East-PA static `+3 h`: `6.29 / 8.61 / 10.68`, `mean|w| = 0.1070`
  - East-PA boundary `+3 h`: `6.24 / 8.81 / 10.79`, `mean|w| = 0.1072`
  - Panhandles HRRR `+3 h`: `4.71 / 7.32 / 8.55`, `mean|w| = 0.1291`
- interpretation:
  - this is the first post-`24d7eaf` refinement that appears to keep the strong
    regional result and fix the stretched canonical ladder at the same time
  - it now also holds the East-PA regional gain through `+3 h` and keeps the
    4 km Panhandles realism benchmark in a clean, plausible regime through `+3 h`

If `hdiv_half` fails at longer horizons:

- next GPT-5.4 Pro recommendation is no longer another RHS-only tweak
- next seam would be fast horizontal pressure-gradient substepping for `u/v`
  inside `run_vertical_acoustic_substeps()`
- smallest proposed change:
  - extract the `u/v` part of `pressure_gradient_kernel()` into an
    `acoustic_horizontal_pg_kernel()`
  - call it as a half-step before and after the semi-implicit column solve
  - suppress the slow `u/v` pressure-gradient contribution in the implicit mode
- current status:
  - not active yet
  - only promote this if the new `hdiv_half` branch fails at `+3 h` East-PA or
    the Panhandles HRRR realism rerun

### 4. Precomputed Face-Geometry Cache with Discrete Metric Identity Checks

Category: solver correctness

Why it might matter:

- The code repeatedly recomputes terrain slopes, local heights, and thicknesses
  in many kernels with slightly different helper paths.
- That is exactly how "same equation, different geometry" bugs survive.
- Precomputing mass-cell thickness, interface spacing, face slopes, and column
  depth once per grid could force all operators to use the same geometry and
  make free-stream-over-terrain errors measurable.

Files / functions:

- `include/grid.cuh`
- `src/core/init.cu` grid setup
- `src/core/dynamics.cu`
- `src/core/boundaries.cu`
- `src/core/sponge.cu`

Quick falsification:

- Add a geometry-cache branch and make the free-stream-over-terrain regression
  print correlation between spurious `w` and terrain slope.
- If mean `|w|`, max `|w|`, and slope correlation do not improve, this is not
  the next lever.

Expected compute impact:

- Runtime roughly flat to `+10%`.
- Memory up modestly for cached face geometry arrays.

Skeptical note:

- This is less flashy than a new solver, but it may be the cleanest route to
  free-stream preservation.

### 5. Dynamic `w`-Transport Blend Sensor Instead of One Global Blend Value

Category: solver correctness

Why it might matter:

- `src/core/dynamics.cu::advection_w_interface_kernel()` currently uses one
  global `w_transport_blend`.
- Flat terrain and steep terrain likely want different amounts of the old and
  new transport forms.
- A local sensor using interface CFL, discrete divergence, or terrain slope
  could keep the short uniform case clean without surrendering the stretched
  gains.

Files / functions:

- `src/core/dynamics.cu::advection_w_interface_kernel`
- `include/stability_control.cuh`
- `src/main.cu` CLI parsing

Quick falsification:

- Replace the scalar blend with a local sensor and compare against fixed
  `0.0`, `0.5`, and `1.0` on:
  - free-stream terrain
  - `uniform_120`
  - `stretch_900`
  - `stretch_3600`
- If it only emulates one fixed value, the extra complexity is not justified.

Expected compute impact:

- Roughly flat to `+5%`.

Skeptical note:

- High practical upside.
- Risk: it degenerates into a local tuning crutch instead of a cleaner
  transport operator.

### 6. Local Acoustic Subcycling Based on Column CFL Instead of One Global `dt`

Category: solver correctness / performance

Why it might matter:

- The eastern Pennsylvania case suggests a smaller `dt` may help materially.
- But paying the smaller `dt` everywhere is expensive.
- A per-column or per-tile acoustic subcycle count driven by local interface
  CFL and terrain steepness could attack the bad columns without slowing the
  whole domain equally.

Files / functions:

- `src/core/dynamics.cu::run_vertical_acoustic_substeps`
- `include/stability_control.cuh`
- `src/main.cu`

Quick falsification:

- Compare:
  - global `dt=8`
  - global `dt=6`
  - local-subcycle branch with mean cost near `dt=8`
- If local subcycling does not match or beat global `dt=6` on the same
  regional case, it is not worth the control complexity.

Expected compute impact:

- Anywhere from flat to `+20%`.
- Best case: cheaper than lowering the global timestep.

Skeptical note:

- Very plausible for this codebase.
- Harder to keep deterministic and easy to reason about.

## Fresh Agent Findings

### A. Conservative Moisture Transport First, Not Full Scalar Rewrite

Category: solver correctness

New recommendation:

- do not touch `theta` first
- add a moisture-only conservative transport experiment for `qv`, `qc`, and `qr`
- keep it aligned with the same transformed divergence skeleton already trusted by `pressure_update_kernel()`

Why this rose in priority:

- regional drift diagnostics now show large interior `qtot` loss while the outer relaxation band is much less severe
- that points more strongly at scalar/tracer inconsistency than at another boundary-only fix
- the existing scalar kernel still advects in a form that does not match the transformed continuity path the model now uses for air mass

Smallest branch:

- `exp/moisture-conservative-transport`
- add a `conservative_moisture_transport` switch
- only replace `qv/qc/qr` launches
- leave `theta` on the current kernel for the first pass

Fast falsification:

- `stretch_120`
- `stretch_900`
- eastern Pennsylvania static `+1 h`

Keep only if:

- interior `qtot_d` improves materially
- `QV` RMSE does not regress
- `THETA` and `mean|w|` stay neutral-to-better

### B. Do Not Fake Lateral `p-w` Characteristics; Radiate `p` First

Category: solver correctness

New recommendation:

- the smallest meaningful boundary experiment is **pressure-only** radiative treatment in the fast acoustic refresh
- do not start with a fake lateral `p-w` characteristic pair

Why:

- on lateral faces, the physically relevant acoustic pairing is `p` with the **boundary-normal horizontal velocity**, not contravariant `w`
- that makes a “pressure-only Orlanski-style fast refresh” the cleanest first branch

Smallest branch:

- `exp/fast-p-radbc`
- modify only fast `p` refresh in `run_vertical_acoustic_substeps()`
- keep lateral `w` unchanged for the first prototype

Expected signal:

- less outer-strip `p'` ringing
- smaller static vs boundary-forced gap on the eastern Pennsylvania `+1 h` case

### C. Columnwise Semi-Implicit Vertical Acoustic `p-w` Corrector

Category: solver correctness

New recommendation:

- the most credible “bigger than tuning” moonshot is now a one-thread-per-column semi-implicit solve for the vertical acoustic `p-w` pair
- structurally reuse the Thomas-solver pattern already present in the PBL

Why it remains attractive:

- it directly targets the stiffest remaining vertical coupling
- it is still local enough to prototype in days, not months
- if it works, it could reduce dependence on strong `w_damp`

Prototype shape:

- new flag: `--pw-column-corrector`
- branch inside `run_vertical_acoustic_substeps()`
- solve cell-centered `p` implicitly per column, then recover interface `w`
- keep the current divergence filter and current open-fast boundary refresh on the first prototype

Keep only if:

- canonical gates stay inside the current envelope
- eastern Pennsylvania `+1 h` materially improves
- runtime overhead stays modest

### 7. Diagnose Nonzero Startup `qc/qr` from Relative Humidity and Lift

Category: physics realism / initialization

Why it might matter:

- Real-data runs currently begin with `qc=0` and `qr=0`.
- That guarantees cloud spin-up shock and can feed back into buoyancy and
  precipitation timing.
- A cheap, diagnosed cloud-water startup could improve short-horizon realism
  before full data assimilation exists.

Files / functions:

- `tools/init_from_gfs.py`
- `src/core/init.cu::load_gfs_binary`
- `src/physics/microphysics_kessler.cu`
- `src/physics/microphysics_thompson.cu`

Quick falsification:

- Generate one init with diagnosed `qc/qr` and compare against the zero-cloud
  init on `1 h` to `6 h` regional runs.
- If precipitation timing, reflectivity structure, and low-level thermodynamics
  do not improve at all, drop it.

Expected compute impact:

- Very small runtime impact.
- Slightly more state variation at startup.

Skeptical note:

- This is realism, not the main stability fix.
- Do not confuse prettier clouds with a solved dycore.

### 8. Selective Precision Map for the `p'-w` Fast Path

Category: solver correctness / performance

Why it might matter:

- The repo already uses doubles for geometry/reference and floats for the main
  state.
- That is the right default, but the current split may still be too blunt.
- A targeted precision map could move only the most cancellation-prone `p'-w`
  kernels or accumulations to `fp64` or compensated math without doubling the
  full 3D state cost.

Files / functions:

- `src/core/dynamics.cu`
- `include/grid.cuh`
- `include/constants.cuh`
- build config around `USE_DOUBLE`

Quick falsification:

- Create a branch that keeps only the fast vertical `p'-w` path in higher
  precision or compensated accumulation.
- Re-run free-stream terrain, `stretch_3600`, and eastern Pennsylvania `+1 h`.
- If the metrics barely move, do not broaden precision work.

Expected compute impact:

- Roughly `+5%` to `+20%` depending on how surgical it is.

Skeptical note:

- Worth testing precisely because "full fp64 everywhere" is the wrong answer.
- Precision will not rescue a semantically wrong operator.

### 9. Cheap Slab Surface Energy Balance to Fix Diurnal Drift Early

Category: physics realism

Why it might matter:

- Once the model survives large regional runs, it will still not look WRF-like
  without a minimally credible surface response.
- The current physics stack has real PBL and moisture machinery, but no
  serious surface/soil state to anchor diurnal thermodynamics.
- A cheap slab surface model could produce a bigger realism jump per line of
  code than immediately chasing full land-surface complexity.

Files / functions:

- `src/main.cu` physics orchestration
- `src/physics/pbl.cu`
- new surface module under `src/physics/`
- `src/io/netcdf_output.cu` or related output wiring

Quick falsification:

- Run a `6 h` to `24 h` regional case and compare 2 m temperature trend,
  boundary-layer depth proxy, and surface flux signs against a trusted parent
  or analysis.
- If the surface signal barely changes, the minimal model is too cheap.

Expected compute impact:

- Roughly `+10%` to `+25%`.

Skeptical note:

- Realism upside is high.
- Not the next move until solver stability on large domains is less fragile.

### 10. Bandit-Style Experiment Controller for Many Rented Nodes

Category: systems / ops

Why it might matter:

- The repo already has the beginning of a worker model in
  `tools/ops/worker_tick.sh` and `docs/CLUSTER_EXPERIMENTS.md`.
- If many nodes are rented, the bottleneck will be experiment selection and
  artifact triage, not starting processes.
- A controller that uses gate outcomes to schedule the next highest-value
  hypothesis can increase discovery speed more than manual queue stuffing.

Files / functions:

- `tools/ops/worker_tick.sh`
- new controller script under `tools/ops/`
- `tools/run_gate_matrix.py`
- `STATUS.md` or a machine-readable result ledger

Quick falsification:

- Compare one week of manual experiment selection versus controller-guided
  selection on the same hardware budget.
- If the controller does not find better branches or better parameter regions
  faster, keep the system simple.

Expected compute impact:

- Negligible model runtime cost.
- Small ops overhead, potentially huge productivity gain.

Skeptical note:

- This will not invent new physics.
- It can absolutely increase the rate of useful failures and wins.

## Top 3 Next Experiment Branches

These are the next three branches worth implementing now that the solver is in
the usable zone.

### Branch 1: `exp/tskin-slab`

Goal:

- Add a minimal prognostic skin-temperature slab so the surface has real heat
  memory.

Minimal code delta:

- Add 2D `tskin` state.
- Update it once per physics step with a tiny slab heat-capacity equation.
- Feed it into the PBL sensible-heat seam.
- Output `TSK`.

Decision test:

- Panhandles HRRR `+6 h`
- compare `T2/RH2` trend and phase without letting wind/reflectivity regress

Stop condition:

- If `T2/RH2` remain just as stiff, stop growing this branch blindly.

### Branch 2: `exp/surface-diagnostics`

Goal:

- Separate real 2 m / 10 m diagnostics from lowest-model-level fields.

Minimal code delta:

- Add proper `T2/Q2/U10/V10` diagnostics through the surface-layer path.
- Keep the solver and physics otherwise unchanged.

Decision test:

- Panhandles HRRR comparison panels
- if the diagnostic story changes materially, keep it even if the slab branch
  is delayed

Stop condition:

- If diagnostics barely move the comparison, do not mistake them for a physics
  fix.

### Branch 3: `exp/acoustic-uv-pg`

Goal:

- Only if longer runs expose a new solver defect, move fast horizontal PG
  forcing for `u/v` into the acoustic loop.

Minimal code delta:

- Extract a small `acoustic_horizontal_pg_kernel()`.
- Apply half-step `u/v` PG kicks around the semi-implicit column solve.
- Suppress the duplicate slow `u/v` PG path in implicit mode.

Decision test:

- East-PA static `+1 h` and `+6 h`
- stretched canonical ladder

Stop condition:

- If it does not improve U/V without hurting the canonical ladder, stop.

## What Looks Exciting But Is Probably Premature

- Full new microphysics.
- Big land-surface ecosystems.
- Full-domain `fp64`.
- Multi-GPU scaling as a science fix.
- Huge-domain hero runs as the primary signal.

Those are all real future work. None of them are the shortest path to a
disproportionate breakthrough right now.

## Practical Interpretation

The fastest moonshot path is probably:

1. keep `exp/semiimplicit-hdiv-half` as the solver baseline unless the longer
   runs expose a new fast-loop defect
2. add the smallest plausible surface thermal memory (`tskin` slab)
3. improve near-surface diagnostics so HRRR comparisons are fair
4. only go back to acoustic `u/v` substepping if the longer-horizon solver
   checks actually justify it

That keeps the research queue aligned with the current code and the current
results instead of chasing older solver hypotheses that the branch has already
outgrown.
