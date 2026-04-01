# GPU-WM Moonshot Notes

Last updated: 2026-04-01

## Purpose

This is the standing research-sidecar memo for `gpu-wm`.

The goal is not generic "better numerics" advice. The goal is to identify
high-upside ideas that are still close enough to the current codebase to test
in hours or days.

Current grounding:

- `main@d92bef2` is a real CUDA regional-model prototype, but large real-data
  terrain-following runs are still below WRF-quality robustness.
- The repo and external reviews agree that the main blocker is still solver
  correctness, especially mixed `w` semantics across slow terms, boundaries,
  and sponge behavior.
- The current experimental family has already shown that `w` transport,
  `w` damping, and open-boundary `w` handling can move the model materially.

This memo deliberately separates solver-correctness ideas from realism ideas
 and systems ideas.

## Ranked Ideas

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

These are the next three branches worth implementing before another broad
solver review. They are ordered by upside-to-effort, not by conventionality.

### Branch 1: `exp/column-balance-startup`

Goal:

- Replace "zero `w` then continuity-only patch" with a one-time startup
  balance pass for interface `w` and `p'`.

Minimal code delta:

- Extend `initialize_w_from_continuity()` into a startup balance driver.
- Add one optional column pass that nudges `p'` to reduce initial divergence or
  hydrostatic residual after `w` is diagnosed.
- Log first-20-step startup metrics automatically for real-data runs.

Decision test:

- Eastern Pennsylvania `+1 h`
- first-step and first-20-step `mean|w|`, `max|w|`, `max|p'|`
- existing free-stream and canonical gates must stay green

Stop condition:

- If startup burst metrics do not improve materially, abandon quickly.

### Branch 2: `exp/characteristic-openbc`

Goal:

- Stop treating `p` and interface `w` like generic scalar fields at the open
  boundary.

Minimal code delta:

- Add dedicated `p/w` boundary routines in `src/core/boundaries.cu`.
- Keep existing mass-field relaxation for advected fields.
- Make `p/w` exit treatment mode-aware or at least one-way biased.

Decision test:

- new wave-exit or gravity-wave exit mini-case
- eastern Pennsylvania `+1 h`
- inspect outer-zone growth versus interior growth

Stop condition:

- If edge reflections do not drop and the interior still fails first, stop
  pushing boundary complexity.

### Branch 3: `exp/column-imex-pw`

Goal:

- Replace part of the "damp your way through it" strategy with a real vertical
  `p'-w` column corrector.

Minimal code delta:

- Keep the current architecture.
- Change only the split vertical acoustic step to a columnwise IMEX or
  tridiagonal corrector.
- Do not touch horizontal transport or physics in the same branch.

Decision test:

- `stretch_900`
- `stretch_3600`
- eastern Pennsylvania `+1 h`
- try reduced `--w-damp` alongside the baseline damping profile

Stop condition:

- If it only survives when damping stays just as strong, the complexity is not
  earning a place.

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

1. attack startup balance
2. attack `p/w` boundary semantics
3. if those still leave too much damping dependency, attack the vertical
   `p'-w` solve itself

That sequence stays aligned with the current code, the current failures, and
the current experiment ladder while still aiming above the obvious incremental
path.
