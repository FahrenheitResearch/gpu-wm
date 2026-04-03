# Moonshot Ideas

Working file for overnight physics ideation waves.

- Goal: collect plausible high-upside physics and diagnostics ideas without implementing them by default.
- Format: 50 waves total, 3 exploration agents per wave.
- Levels:
  - Level 1: realistic moonshots
  - Level 2: ambitious moonshots
  - Level 3: high-variance, never-been-done-if-possible moonshots
- Rule: check live node and experiment health before each new wave.
- Promotion rule: only mark an idea as experiment-worthy if it looks strong enough to justify spending 1-2 nodes.

## Counter

- Completed waves: 15 / 50

## Current Node Snapshot

- H100-2: clean East-PA static `+12 h` relaunch 4 is active and has written its first output after clearing a full-overlay blockage.
- A100-1: moderate boundary `+6 h` is active.
- A100-2: moderate static `+6 h` is active.
- A800: hot static `+6 h` is active.

## Master Ranked Table

Deduped master list through wave 13. Repeated nominations are kept once and ranked by my current estimate of near-term upside, honesty of the signal, and fit to the codebase.

| Rank | Idea | Level | Category |
| --- | --- | --- | --- |
| 1 | Flow-Heterogeneity Surface-Layer Gate | L1 | surface realism |
| 2 | Screen-Layer Representativeness Breakpoints | L1 | screen diagnostics |
| 3 | Soil Thermal Admittance Seams | L1 | surface thermodynamics |
| 4 | Interception-Lag Canopy Patches | L1 | canopy wetness |
| 5 | Dew-Adsorption Skin Nodes | L1 | surface moisture |
| 6 | Bowen-Ratio Dawn Hinges | L1 | surface energy partition |
| 7 | Roughness-Step Shear Hinges | L1 | dawn PBL / roughness |
| 8 | Invariant-Preserving Guardrail Projector | L3 | dycore guardrails |
| 9 | Acoustic Debt Ledger | L3 | dycore fast-mode memory |
| 10 | Thickness-Conjugate w Split Form | L3 | dycore split form |
| 11 | Metric-Commutator Split Form | L3 | dycore operator ordering |
| 12 | Dual-Normal Coframe Cache | L3 | dycore terrain geometry |
| 13 | Guardrail Echo Memory | L3 | dycore guardrails |
| 14 | Palindrome Defect Meter | L3 | dycore diagnostics |
| 15 | Halo-Lag Covenant Memory | L3 | dycore halo / boundary memory |
| 16 | Terrain-Geodesic Bernoulli Residual | L3 | dycore terrain invariant |
| 17 | Surface-Coupling Hysteresis State | L1 | PBL coupling memory |
| 18 | Subcanopy Screen Decoupling Cells | L1 | canopy microclimate |
| 19 | Shallow Water-Table Access Fraction | L1 | subsurface wetness access |
| 20 | Split-Schur Terrain Closure | L3 | dycore fast/slow coupling |
| 21 | Terrain Hodge Defect Split | L3 | dycore modal defects |
| 22 | Capillary-Recharge Dawn Rims | L1 | soil hydraulic moisture |
| 23 | Residue-Mulch Insulation Islands | L1 | surface insulation |
| 24 | Frost-Clearance Latent Brakes | L1 | frost / condensate memory |
| 25 | Canopy Sky-View Shelters | L1 | canopy radiation geometry |
| 26 | Precipitation-Wet Surface Memory | L1 | wet-surface memory |
| 27 | Dynamic Saturated Source-Area Fraction | L1 | wetness source-area dynamics |
| 28 | Orography-Anchored Recoupling Plumes | L1 | dawn terrain recoupling |
| 29 | Residual-Layer Return-Flow Shelf | L1 | residual-layer moisture |
| 30 | Drainage-Pool Reservoir | L1 | cold-air pooling |
| 31 | Valley-Gate Inflow Locks | L2 | terrain-organized convection |
| 32 | Oblique Surge Interference Seams | L2 | cold-pool interactions |
| 33 | Cold-Pool Age-Sorting Belts | L2 | cold-pool memory |
| 34 | Cold-Pool Depth-Step Mosaics | L2 | cold-pool structure |
| 35 | Pinch-Gate Inflow Filaments | L2 | inflow segmentation |
| 36 | Lee-Side Recapture Basins | L2 | terrain-organized convection |
| 37 | Contour-Rail Maintenance Bands | L2 | terrain-organized convection |
| 38 | Mesovortex Amalgamation Ratchets | L2 | QLCS mesovortices |
| 39 | Triple-Feed Vorticity Braid | L2 | QLCS vorticity organization |
| 40 | Thermal-Wave Scaffold | L2 | latent-heating organization |
| 41 | Stratiform Recharge Corridors | L2 | stratiform memory |
| 42 | Descending Stratiform Memory Sheets | L2 | stratiform memory |
| 43 | Stratiform Phase-Lag Ribs | L2 | stratiform memory |
| 44 | Subcloud Evaporation Debt Ledger | L2 | cold-pool memory |
| 45 | Subcloud Exposure Cooling Integral | L1 | subcloud evaporation physics |
| 46 | Latent-Acoustic Forcing Number | L2 | fast / moist coupling |
| 47 | Bore-Head Phase Locking | L2 | bore / wave timing |
| 48 | Saturation-Shock Substep Gate | L2 | stiff moist numerics |
| 49 | Split-Volume Afterimage | L3 | dycore split-memory |
| 50 | Corner-Twist Face Closure | L3 | terrain face geometry |
| 51 | Eta-Parity Braided Split | L3 | stagger aliasing |
| 52 | Plumb-Line / Sigma-Line Separation Ledger | L3 | terrain water bookkeeping |
| 53 | Sigma-Loop Holonomy Memory | L3 | terrain path-order memory |
| 54 | Moist-Virtual-Mass Centroid Precession | L3 | condensate / pressure geometry |
| 55 | Frozen-Density Inertia Shadow | L3 | fast reference-state error |
| 56 | Eta-Mass Water Residual Cube | L1 | water bookkeeping diagnostics |
| 57 | Latent-Stiffness Governor Field | L1 | moist-stiffness control |
| 58 | Canopy Stress Recovery Memory | L1 | canopy / transpiration memory |
| 59 | Riparian Heat-Capacity Filaments | L1 | riparian thermal heterogeneity |
| 60 | Terrain Longwave View-Factor Tile | L1 | terrain radiation geometry |
| 61 | Sunrise Horizon-Lag Facets | L1 | dawn solar geometry |
| 62 | Gap-Jet Moisture Injector | L1 | terrain moisture advection |
| 63 | Dripline Moisture Rings | L1 | canopy hydrology |
| 64 | Runon-Furrow Moisture Veins | L1 | lateral surface-water concentration |
| 65 | Moist-Variance CI Bridge | L1 | convective initiation |
| 66 | Prognostic EML Reservoir | L2 | cap / EML memory |
| 67 | Dryline Tail-PDF Trigger | L2 | dryline initiation |
| 68 | Bookend Vortex Jet Lens | L2 | QLCS end-vortex dynamics |
| 69 | Bow-Hinge Lattice | L2 | bowing organization |
| 70 | Rear-Edge Pressure Shelf | L2 | QLCS pressure organization |
| 71 | Stratiform Porosity Valves | L2 | stratiform ventilation |
| 72 | Precipitation Flux Confluence Arcs | L2 | precip-path organization |
| 73 | Condensate Relay Corridors | L2 | condensate pathway memory |
| 74 | Spur-Wake Relay Anchors | L2 | terrain-organized convection |
| 75 | Overhang-Curtain Intake Cells | L2 | inflow segmentation |
| 76 | Hydrometeor Hang-Time Echoes | L2 | hydrometeor memory |
| 77 | Gust-Front Skeleton Field | L3 | cold-pool line skeleton |
| 78 | Guardrail Jacobian Counterterm | L3 | dycore guardrails |
| 79 | 1.5-Moment PSD Memory With Radar Gate | L2 | microphysics memory |
| 80 | Melting-Layer Phase-Lag Clock | L2 | phase-change memory |
| 81 | Downshear Debris-Shadow Sheet | L2 | anvil / shading memory |
| 82 | Merger Shadow Notch | L2 | merger memory |
| 83 | Hydrometeor Transit-Time Memory | L3 | microphysics residence time |
| 84 | Downdraft Ancestry Stack | L3 | downdraft lineage |
| 85 | Advected Gravity-Wave Action Memory | L3 | gravity-wave memory |
| 86 | Inversion-Layer Bore Conjugacy | L3 | bore geometry |
| 87 | Boundary-Layer Roll Phasor | L3 | PBL roll memory |
| 88 | Shallow PV Ribbon Memory | L3 | PV ribbon memory |
| 89 | Moist-Isentrope / Eta Torsion Tensor | L3 | terrain / moist geometry |
| 90 | Moist-Entropy Defect Reservoir | L3 | entropy-defect memory |
| 91 | Lee-Wave Vapor Caustics | L3 | wave moisture focusing |
| 92 | Mixed-Layer Impedance Refraction | L2 | mixed-layer wave refraction |
| 93 | Hodograph-Twist Niches | L2 | mesoscale shear niches |
| 94 | Pre-Squall Mesolow Inflow Siphon | L2 | presquall inflow dynamics |

## Idea Log

### Wave 1

Node check before next wave:

- H100-2: still actively integrating East-PA boundary `+12 h`, GPU loaded, but the malformed `--semiimplicit-pw?` launch flag remains a trust issue.
- A100-2: solver finished; in `verify_forecast.py` for Panhandles screen-diagnostic boundary `+12 h`.
- A800: solver finished; in weather plotting/postprocess for Panhandles screen-diagnostic static `+12 h`.
- A100-1: idle and available.

Shortlisted ideas:

- `L1 | score 5 | Moist-Variance CI Bridge`
  Small subgrid RH-variance bridge between the current PBL and first condensate so CI is less binary and late at 4 km.

- `L1 | score 5 | Precipitation-Wet Surface Memory`
  Tiny 2-D wetness reservoir driven by rain and nighttime condensation to retain post-storm surface flux and thermal-memory footprints.

- `L2 | score 5 | Prognostic EML Reservoir`
  Per-column inversion-memory state for cap strength, entrainment, and mixed-layer properties to make dryline/cap evolution less instantaneously diffusive.

- `L2 | score 5 | 1.5-Moment PSD Memory With Radar Gate`
  One extra hydrometeor memory scalar to evolve fall speed, evaporation, and reflectivity texture without paying for full two-moment microphysics.

- `L3 | score 5 | Gust-Front Skeleton Field`
  Cheap 2-D cold-pool edge memory to preserve lifting lines, secondary initiation, and MCS organization after the resolved pool has numerically blurred.

- `L2 | score 4 | Dryline Tail-PDF Trigger`
  Subgrid buoyant-tail trigger tied to dryline gradients and shear so a parcel fraction can break CIN before the grid-mean parcel does.

### Wave 2

Node check before next wave:

- H100-2: East-PA boundary `+12 h` still active; output count increased to `4`, but the malformed `--semiimplicit-pw?` flag still makes this one provisional.
- A100-2: rolled from the finished screen-diagnostic `+12 h` into a tunable-slab boundary `+6 h` run and is back at full GPU load.
- A800: rolled from the finished screen-diagnostic `+12 h` into a tunable-slab static `+6 h` run and is back at full GPU load.
- A100-1: still idle at the time of the check and being prepared for a corrected tunable run.

Shortlisted ideas:

- `L3 | score 5 | Advected Gravity-Wave Action Memory`
  Mobile subgrid wave-action memory to carry bore / gravity-wave preconditioning into later CI and overnight organization without touching the drycore.

- `L1 | score 5 | Surface-Coupling Hysteresis State`
  A 2-D exchange-memory state to stop the screen layer from snapping instantly between decoupled and fully mixed regimes.

- `L1 | score 5 | Subcloud Exposure Cooling Integral`
  A sedimentation-path exposure integral to convert the same rain mass into different downdraft cooling depending on the depth and dryness of the subcloud layer.

- `L2 | score 5 | Subcloud Evaporation Debt Ledger`
  Column memory for remaining evaporation/cold-pool capacity so outflow strength is history-aware instead of purely local and instantaneous.

- `L2 | score 4 | Downshear Debris-Shadow Sheet`
  Thin sheared anvil-shadow memory field to alter surface heating and PBL growth after the parent storm core moves on.

- `L2 | score 4 | Melting-Layer Phase-Lag Clock`
  Simple residence-time memory through the melting layer to improve bright-band structure and downstream cold-pool realism.

### Wave 3

Node check before next wave:

- H100-2: East-PA boundary `+12 h` still active at `4` outputs, but still launched with malformed `--semiimplicit-pw?`, so this remains a watch item rather than a trusted baseline.
- A100-2: tunable-slab boundary `+6 h` active with `2` outputs; this is one of the accidentally ultra-hot restore-coefficient runs.
- A800: tunable-slab static `+6 h` active with `2` outputs; also an accidentally ultra-hot restore-coefficient run.
- A100-1: corrected moderate tunable-slab boundary `+6 h` launched cleanly and has written its first output.

Shortlisted ideas:

- `L3 | score 5 | Boundary-Layer Roll Phasor`
  Mixed-layer roll phase and orientation memory to restore coherent cloud-street / moisture-striping effects instead of diffusing them into noise.

- `L3 | score 5 | Shallow PV Ribbon Memory`
  Low-level PV-surrogate ribbon to preserve rotational preconditioning and line-end organization after reflectivity weakens.

- `L2 | score 5 | Latent-Acoustic Forcing Number`
  A physics-to-dycore coupling number that localizes where latent heating, loading, and flux impulses are injecting fast-mode trouble.

- `L2 | score 5 | Saturation-Shock Substep Gate`
  Selective microphysics substepping only in locally stiff columns instead of a global timestep reduction.

- `L1 | score 5 | Eta-Mass Water Residual Cube`
  Compressed online water-budget attribution cube by terrain and height band to catch which operator or regime is leaking moisture.

- `L1 | score 4 | Latent-Stiffness Governor Field`
  Per-column stiffness memory to track and tame thermodynamic overreaction in thin, terrain-elevated, low-density columns.

### Wave 4

Node check before next wave:

- H100-2: still advancing the East-PA boundary follow-on with the same malformed semi-implicit flag issue.
- A100-2: still healthy on the ultra-hot tunable boundary `+6 h` probe.
- A800: still healthy on the ultra-hot tunable static `+6 h` probe.
- A100-1: corrected moderate tunable boundary `+6 h` is active and has joined the overnight experiment set.

Shortlisted ideas:

- `L1 | score 5 | Drainage-Pool Reservoir`
  Terrain-trapped near-surface cold/moist pool memory so valley or rain-cooled air does not vanish as soon as instantaneous fluxes change sign.

- `L2 | score 5 | Bore-Head Phase Locking`
  Explicit focus on where elevated bores re-phase with the surface cold-pool head to create intermittent deep-lift hotspots for new cells.

- `L3 | score 5 | Moist-Isentrope / Eta Torsion Tensor`
  Dynamic measure of mismatch between moist-neutral displacement and terrain-following coordinate geometry to expose hidden phase-geometry errors.

- `L1 | score 4 | Canopy Stress Recovery Memory`
  Single transpiration-stress memory scalar so moisture release does not snap instantly with the current surface state.

- `L2 | score 4 | Mixed-Layer Impedance Refraction`
  Cold-pool refraction through ambient BL-depth and stability contrasts as a new daughter-cell focusing mechanism.

- `L3 | score 4 | Moist-Entropy Defect Reservoir`
  A prognostic store for split-step thermodynamic closure error so buoyancy and `w` ringing do not have to carry that memory implicitly.

### Wave 5

Node check before next wave:

- H100-2: still alive with `5` outputs, but currently the least trustworthy run because of the malformed `--semiimplicit-pw?` launch token.
- A100-2: ultra-hot tunable boundary `+6 h` still running at full GPU load with `2` outputs.
- A800: ultra-hot tunable static `+6 h` still running at full GPU load with `3` outputs.
- A100-1: corrected moderate tunable boundary `+6 h` still running at full GPU load with `2` outputs.

Shortlisted ideas:

- `L2 | score 5 | Stratiform Porosity Valves`
  Treat plan-view precipitation-shield permeability as an organizing control on rear-inflow descent, evaporation access, and bowing/notch evolution.

- `L2 | score 5 | Precipitation Flux Confluence Arcs`
  Horizontal convergence of falling hydrometeor flux as a way to focus loading, evaporation, and banded collector arcs between neighboring cells.

- `L1 | score 5 | Flow-Heterogeneity Surface-Layer Gate`
  Real-time detector for when classical similarity-based exchange is trustworthy versus when flow heterogeneity should push the surface layer toward a more sheltered branch.

- `L3 | score 5 | Inversion-Layer Bore Conjugacy`
  Nonclassical hydraulic-conjugacy framing for terrain-forced bores and lee jumps to catch physically wrong but numerically stable `w`/pressure responses.

- `L1 | score 4 | Terrain Longwave View-Factor Tile`
  Precomputed terrain longwave reciprocity and sky-view suppression to improve dawn inversions and valley-wall thermal trapping cheaply.

- `L2 | score 4 | Condensate Relay Corridors`
  Horizontal condensate handoff between mature and new cells as a back-building and line-filling organizer.

### Wave 6

Node check before next wave:

- H100-2: still alive with `5` outputs and back under load, but still tainted by the malformed semi-implicit launch token.
- A100-2: ultra-hot tunable boundary `+6 h` active and up to `4` outputs.
- A800: ultra-hot tunable static `+6 h` active and up to `5` outputs.
- A100-1: corrected moderate tunable boundary `+6 h` active and up to `4` outputs.

Shortlisted ideas:

- `L1 | score 5 | Shallow Water-Table Access Fraction`
  Geomorphically pinned shallow-groundwater access fraction to keep latent flux and near-surface humidity from collapsing too fast after rainfall.

- `L2 | score 5 | Bookend Vortex Jet Lens`
  Compact line-end vortex couplet as a predictor and amplifier of bowing upscale growth via focused rear-inflow and low-pressure lensing.

- `L3 | score 5 | Downdraft Ancestry Stack`
  Track descent-history source levels for surface-reaching downdrafts so cold-pool and near-surface anomalies inherit a physically meaningful ancestry.

- `L2 | score 4 | Pre-Squall Mesolow Inflow Siphon`
  Forward inflow-side mesolow as an organizer that can pull fragmented cells into a merger corridor before a clean gust front exists.

- `L3 | score 4 | Lee-Wave Vapor Caustics`
  Persistent trapped-wave moist/dry banding downstream of terrain as a hidden geometry-memory source for later clouding and transport errors.

- `L1 | score 4 | Dynamic Saturated Source-Area Fraction`
  Storage-controlled expansion and contraction of topographically favored wet patches without committing to a full lateral hydrology model.

### Wave 7

Node check before next wave:

- H100-2: East-PA boundary follow-on is still running and has reached `6` outputs, but remains provisionally tainted by the malformed semi-implicit flag.
- A100-2: ultra-hot tunable boundary `+6 h` is still actively integrating and has reached `6` outputs.
- A800: ultra-hot tunable static `+6 h` is still actively integrating and has reached `6` outputs.
- A100-1: corrected moderate tunable boundary `+6 h` is still actively integrating and has reached `4` outputs.

Shortlisted ideas:

- `L3 | score 5 | Sigma-Loop Holonomy Memory`
  Closed-loop path-ordering error on terrain-following coordinates as an accumulated source of `w`, `THETA`, and moisture drift.

- `L3 | score 5 | Plumb-Line / Sigma-Line Separation Ledger`
  Track the geometric mismatch between gravity-aligned fallout and sigma-column bookkeeping as a hidden water-loss mechanism over slopes.

- `L2 | score 5 | Bow-Hinge Lattice`
  Distributed along-line bowing hinges built from line-parallel shear and mesovortex chaining, not just one dominant apex.

- `L1 | score 5 | Residual-Layer Return-Flow Shelf`
  Elevation-specific nighttime moisture and exchange node where valley return flow rides above the surface drainage layer.

- `L2 | score 4 | Rear-Edge Pressure Shelf`
  Rear-stratiform overpressure shelf as an organization signal that may matter more than the leading gust-front pressure field in some lines.

- `L1 | score 4 | Gap-Jet Moisture Injector`
  Terrain-gap advective bypass for importing humid boundary-layer air into lee valleys and canyon mouths.

### Wave 8

Node check before next wave:

- H100-2: SSH timed out twice during this wave, so that node is now operationally flaky even though the run had reached at least `6` outputs before contact degraded.
- A100-2: ultra-hot tunable boundary `+6 h` active and has reached `6` outputs.
- A800: previous ultra-hot static run finished; node was immediately reused for a new hotter static probe, which is active but again launched with mangled slab parameters.
- A100-1: corrected moderate tunable boundary `+6 h` finished cleanly through `7` outputs and is now available for reuse.

Shortlisted ideas:

- `L3 | score 5 | Moist-Virtual-Mass Centroid Precession`
  Hidden offset between virtual-mass and condensable-water centroids as a terrain-following column memory feeding later `p'-w` adjustment.

- `L3 | score 5 | Plumb-Line / Sigma-Line Separation Ledger`
  Gravity-fall versus sigma-column bookkeeping mismatch as a hidden interior water-loss mechanism.

- `L1 | score 5 | Dew-Adsorption Skin Nodes`
  Thin non-rainfall moisture skin from dew and adsorption as a sunrise flux-partition lever before any full soil model exists.

- `L2 | score 5 | Mesovortex Amalgamation Ratchets`
  Merger and upscale consolidation of low-level mesovortices as a clean organization node for widening wind swaths and segment consolidation.

- `L2 | score 4 | Hodograph-Twist Niches`
  Local line-relative hodograph geometry as a selector for which segments of a long line harden into rotationally favored bowing/tornadic niches.

- `L3 | score 4 | Hydrometeor Transit-Time Memory`
  Delayed condensate-loading and evaporation response tied to residence time through stretched terrain-following layers.

### Wave 9

Node check before next wave:

- H100-2: still unreachable over SSH during this wave, so its run remains operationally uncertain.
- A100-2: ultra-hot tunable boundary `+6 h` remains active and past `6` outputs.
- A800: new hotter static probe is active and has written `2` outputs, but its parsed launch parameters were mangled again into another ultra-hot stress variant.
- A100-1: corrected moderate run finished cleanly; first attempt to launch a hotter follow-on failed because the launcher precheck matched its own shell and exited before starting the job.

Shortlisted ideas:

- `L3 | score 5 | Split-Volume Afterimage`
  Non-commuting fast/slow/scalar volume measures leaving a tiny delayed residual that re-enters later steps as drift instead of an immediate blow-up.

- `L2 | score 5 | Triple-Feed Vorticity Braid`
  Abruptly favored rotating QLCS segments when three different vorticity-bearing airstreams phase-lock into the same low-level updraft.

- `L1 | score 5 | Soil Thermal Admittance Seams`
  Dawn warm-up and RH-striping errors caused by patchy shallow-soil thermal admittance, not just surface moisture state.

- `L1 | score 4 | Sunrise Horizon-Lag Facets`
  Terrain skyline blocking delays first shortwave differently across nearby slopes, staggering inversion breakup and early surface recovery.

- `L2 | score 4 | Merger Shadow Notch`
  Post-supercell-merger thermodynamic scar where the line temporarily weakens its cold pool but preserves rotation potential.

- `L3 | score 4 | Frozen-Density Inertia Shadow`
  Long-memory pressure bias from fast kernels using effectively frozen reference density while thermodynamics evolve away from startup.

### Wave 10

Node check before next wave:

- H100-2: clean East-PA boundary `+12 h` was relaunched correctly and had written its first output when the new queue was attached.
- A100-2: corrected moderate static `+6 h` rerun was relaunched and had written its first output.
- A800: corrected hot static `+6 h` rerun was relaunched and had written its first output.
- A100-1: corrected hot boundary `+6 h` rerun was relaunched and had written its first output.

Shortlisted ideas:

- `L3 | score 5 | Acoustic Debt Ledger`
  Store unresolved fast-mode closure mismatch from acoustic substeps as repayable debt instead of recreating it as fresh burst-and-damp noise each pass.

- `L3 | score 5 | Thickness-Conjugate w Split Form`
  Replace the heuristic `w_transport_blend` compromise with a structure-preserving conjugate split form for `w` and depth-weighted physical flux.

- `L2 | score 5 | Thermal-Wave Scaffold`
  Treat latent-heating and low-level-cooling geometry as the organizer that scaffolds line-scale circulation, rear inflow, and stratiform asymmetry.

- `L2 | score 5 | Triple-Feed Vorticity Braid`
  Preferential rotating segments emerge when three distinct vorticity-bearing inflow streams phase-lock into one low-level updraft.

- `L1 | score 5 | Soil Thermal Admittance Seams`
  Patchy shallow-soil heat-storage release as a dawn warm-up and RH-striping lever even before a full soil model exists.

- `L1 | score 5 | Orography-Anchored Recoupling Plumes`
  Dawn inversion breakup begins at terrain-locked plume anchors rather than as a domain-wide smooth recoupling.

### Wave 11

Node check before next wave:

- H100-2: clean East-PA boundary `+12 h` relaunch remains healthy and has reached `5` outputs.
- A100-2: hot tunable Panhandles boundary `+6 h` is active and has written its first output.
- A800: moderate tunable Panhandles boundary `+6 h` is active and has written its first output.
- A100-1: hot tunable Panhandles static `+6 h` is active and has written its first output.

Shortlisted ideas:

- `L3 | score 5 | Metric-Commutator Split Form`
  Advance terrain-metric and transport operators in paired symmetric orderings, then track their antisymmetric commutator residue explicitly instead of letting it leak invisibly into the main state.

- `L3 | score 5 | Guardrail Echo Memory`
  Store the signed `w` / `p'` impulse removed by guardrails, sanitizers, or diffusion in a short-lived echo field and only release it when the same mode tries to re-form, turning hard clipping into hysteresis.

- `L3 | score 4 | Corner-Twist Face Closure`
  Treat each terrain-following face as a bilinear patch with a latent mixed-derivative twist term so `w` transport and pressure-gradient updates share the same discrete face normal.

- `L2 | score 5 | Oblique Surge Interference Seams`
  Track shallow-angle collisions between neighboring cold-pool surges as persistent along-line redevelopment and spin-up seams before any obvious vortex merger appears.

- `L2 | score 5 | Valley-Gate Inflow Locks`
  Use modest terrain gaps and valley axes as repeatable inlet stations that focus high-theta-e inflow into fixed along-line nodes, anchoring storm maintenance or breaks.

- `L2 | score 4 | Stratiform Recharge Corridors`
  Treat trailing-stratiform evaporative/melting cooling corridors as slow mesoscale recharge paths that repeatedly repair or re-bow the same line segments from behind.

- `L1 | score 5 | Roughness-Step Shear Hinges`
  Abrupt roughness jumps create pre-dawn drag and decoupling contrasts that turn into thin lines of wind acceleration and inversion breakup once first light mixes momentum down.

- `L1 | score 5 | Bowen-Ratio Dawn Hinges`
  Wet and dry patches split the first post-sunrise energy budget into different sensible-vs-latent pathways, creating short-distance dawn boundaries in T2, RH2, and shallow turbulence onset.

- `L1 | score 4 | Riparian Heat-Capacity Filaments`
  Rivers, reservoirs, and marshy bottoms lag the dawn warm-up enough to anchor narrow cool/moist shoreline or valley-floor filaments that perturb the first few hours of shallow wind and humidity structure.

### Wave 12

Node check before next wave:

- H100-2: clean East-PA boundary `+12 h` relaunch remains active and has reached `5` outputs.
- A100-1: hot tunable Panhandles static `+6 h` remains active and has reached `5` outputs.
- A100-2: hot tunable Panhandles boundary `+6 h` remains active and has reached `5` outputs.
- A800: moderate tunable Panhandles boundary `+6 h` remains active and has reached `5` outputs.

Shortlisted ideas:

- `L3 | score 5 | Dual-Normal Coframe Cache`
  Precompute matched primal and dual terrain metrics so divergence, pressure-gradient, and `w` conversion operators all use the same paired geometry objects instead of recomputing inconsistent normals and weights.

- `L3 | score 5 | Palindrome Defect Meter`
  Measure the fast-loop time-reversal defect on a scratch copy of each microcycle so hidden non-reversibility from filtering, halo refresh, or fast/slow leakage becomes an explicit field instead of buried drift.

- `L3 | score 5 | Invariant-Preserving Guardrail Projector`
  Split fast-state updates into invariant-compatible and invariant-violating components, then apply guardrails only to the violating piece so stabilization becomes a projection rather than a field clamp.

- `L2 | score 5 | Cold-Pool Age-Sorting Belts`
  Let new and old outflow self-sort into along-line density belts whose interfaces become repeat lift and shear strips for bowlets and mesovortices.

- `L2 | score 5 | Lee-Side Recapture Basins`
  Small lee-side recirculation pockets trap and remix prior storm air into terrain-anchored maintenance reservoirs that can repeatedly feed the same QLCS segment.

- `L2 | score 4 | Stratiform Phase-Lag Ribs`
  The trailing stratiform canopy lays down slowly drifting melting and evaporation ribs that later act as reactivation skeletons when intersected by inflow.

- `L1 | score 5 | Interception-Lag Canopy Patches`
  Canopy-held dew, fog, or trace rain spends early net radiation on evaporation instead of warming, delaying local dawn warming and shallow recoupling.

- `L1 | score 5 | Screen-Layer Representativeness Breakpoints`
  The true 2 m layer can live in a different dawn regime than the lowest model level, creating recurring screen-level bias transitions at vegetation and sheltering boundaries.

- `L1 | score 4 | Capillary-Recharge Dawn Rims`
  Fine-textured or shallow-water-table patches can recharge the top few millimeters overnight, creating narrow dawn humidity and inversion-retention ribbons without open water.

### Wave 13

Node check before next wave:

- H100-2: clean East-PA boundary `+12 h` relaunch remains active at `5` outputs.
- A100-1: hot static `+6 h` finished through `7` outputs and was manually rearmed into moderate boundary `+6 h`.
- A100-2: hot boundary `+6 h` finished through `7` outputs and was manually rearmed into moderate static `+6 h`.
- A800: moderate boundary `+6 h` finished through `7` outputs and was manually rearmed into hot static `+6 h`.

Shortlisted ideas:

- `L3 | score 5 | Halo-Lag Covenant Memory`
  Carry the mismatch between interior-predicted continuation and enforced halo or boundary refresh as a short-lived fast memory instead of letting repeated refreshes act like invisible impulses.

- `L3 | score 4 | Eta-Parity Braided Split`
  Treat odd-even eta staggering as an explicit alias mode and evolve or cancel it directly inside the fast `p-w` split instead of letting it leak into bulk drift.

- `L3 | score 5 | Terrain-Geodesic Bernoulli Residual`
  Define a terrain-following free-stream residual that only measures discrete Bernoulli violation, then damp or correct that residual rather than the raw fields.

- `L2 | score 5 | Pinch-Gate Inflow Filaments`
  Let neighboring cold-pool bulges and precipitation curtains squeeze ambient inflow into narrow gates that repeatedly feed the same line segments.

- `L2 | score 5 | Spur-Wake Relay Anchors`
  Use a chain of lee-side shelter pockets behind small ridges or spurs as a relay path that repeatedly supports storm maintenance along the same advancing line.

- `L2 | score 4 | Descending Stratiform Memory Sheets`
  Treat elevated moistening and loading laminae beneath the trailing stratiform canopy as slowly descending memory sheets that later precondition renewed ascent.

- `L1 | score 5 | Residue-Mulch Insulation Islands`
  Litter, duff, and crop residue act as a thin moist porous blanket that delays dawn warming and keeps near-surface air cooler and wetter than adjacent bare ground.

- `L1 | score 4 | Dripline Moisture Rings`
  Stemflow and throughfall concentrate intercepted canopy water into narrow ground rings and edge bands that preserve local dawn RH maxima and delayed warming.

- `L1 | score 5 | Subcanopy Screen Decoupling Cells`
  Dense canopy patches can leave the true 2 m air below the canopy displacement layer in a different dawn regime than the exchange layer above.

### Wave 14

Node check before next wave:

- H100-2: earlier East-PA boundary long run ended and the node had to be cleaned up because its overlay filesystem filled; clean static `+12 h` relaunch 4 is now back up and has written its first output.
- A100-1: moderate boundary `+6 h` is active after the prior hot static run completed.
- A100-2: moderate static `+6 h` is active after the prior hot boundary run completed.
- A800: hot static `+6 h` is active after the prior moderate boundary run completed.

Shortlisted ideas:

- `L3 | score 5 | Stage-Handshake Replay Memory`
  Store the mismatch between frozen-coefficient acoustic equilibrium and the post-slow/post-guardrail state at each RK-stage handoff, then replay it into the next stage instead of recreating the coupling error from zero.

- `L3 | score 5 | Metric-Curl Twin Closure`
  Replace separately sampled terrain slopes with a paired metric two-form whose face-flux and circulation identities close exactly on the staggered terrain-following grid, while exposing any leftover metric-curl defect explicitly.

- `L3 | score 4 | Operator-Braid Counterphase`
  Alternate two complementary whole-stage operator orderings so their leading split-order bias cancels over a macrostep, and treat the braid difference as a bounded diagnostic or correction mode.

- `L2 | score 5 | Cold-Pool Spillway Networks`
  Let dense outflow drain through a repeating terrain-routed network of low corridors and side valleys so storm maintenance depends on where the cold pool can escape, not only where inflow enters.

- `L2 | score 5 | Layer-Swap Inflow Terraces`
  Different line segments can tap different vertical slices of warm-sector inflow, creating step-like terraces of shallow versus deeper inflow that lock in segment spacing and maintenance.

- `L2 | score 4 | Hydrometeor Hang-Time Echoes`
  Hydrometeors can carry old convective structure aloft and later release it through delayed melting, evaporation, and drag, creating repeat secondary pressure or cold-pool pulses.

- `L1 | score 5 | Stomatal-Reopening Lag Patches`
  Vegetation does not restart transpiration uniformly at sunrise, so nearby grass, crop, and tree patches can flip from sensible-dominated to latent-dominated heating on different clocks.

- `L1 | score 5 | Hydraulic-Lift Moisture Halos`
  Deep-rooted vegetation can redistribute water upward overnight into near-surface soil, creating localized dawn moist halos that sustain cooler and more humid screen-layer air.

- `L1 | score 4 | Row-Shadow Retreat Strips`
  Shelterbelts, orchard rows, and row-crop geometry can leave narrow ground strips shaded after sunrise, delaying dew removal, sensible heating, and turbulence onset in moving bands.

### Wave 15

Node check before next wave:

- H100-2: the prior East-PA long run completed, the overlay filled with stale run output, and the node had to be cleaned before a clean static `+12 h` relaunch 4 was started.
- A100-1: moderate boundary `+6 h` is active and has progressed past early output.
- A100-2: moderate static `+6 h` is active and has progressed past early output.
- A800: hot static `+6 h` is active and has progressed past early output.

Shortlisted ideas:

- `L3 | score 5 | Split-Schur Terrain Closure`
  Approximate the omitted slow terrain-metric and guardrail response to each fast `p-w` increment and feed it back as a cheap Schur-complement counterterm inside the acoustic loop.

- `L3 | score 5 | Terrain Hodge Defect Split`
  Decompose terrain-following increments into exact, coexact, and harmonic components on the staggered grid so hidden harmonic drift modes stop masquerading as ordinary divergence noise.

- `L3 | score 5 | Guardrail Jacobian Counterterm`
  Linearize sanitize, Rayleigh, and diffusion around the current fast state and inject the implied counterterm inside the fast microsteps so the solver advances under an approximation of the guarded operator.

- `L2 | score 5 | Cold-Pool Depth-Step Mosaics`
  Let the cold pool partition into alternating deep and shallow slabs whose boundaries become repeat lift and shear strips for maintenance and mesovortex reuse.

- `L2 | score 5 | Contour-Rail Maintenance Bands`
  Allow dense outflow to turn and run along terrain contours, creating slope-parallel convergence rails that anchor redevelopment at fixed terrain-relative offsets.

- `L2 | score 4 | Overhang-Curtain Intake Cells`
  Stratiform overhang and virga can leave ahead-of-line cool/moist curtains that subdivide incoming warm-sector parcels before they reach the gust front.

- `L1 | score 5 | Frost-Clearance Latent Brakes`
  Uneven frost or frozen dew patches spend early net radiation on melting and sublimation, delaying the first burst of screen-layer warming and mixing.

- `L1 | score 5 | Canopy Sky-View Shelters`
  Vegetation geometry reduces local sky-view factor and longwave cooling, leaving pre-dawn warm-moist radiative shelters and sharp edge contrasts at true screen height.

- `L1 | score 4 | Runon-Furrow Moisture Veins`
  Shallow overland flow collects in furrows, wheel tracks, and runon bands, leaving narrow wet strips that survive overnight and create striped dawn cool-humid anomalies.
