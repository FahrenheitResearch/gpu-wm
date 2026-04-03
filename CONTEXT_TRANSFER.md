# GPU-WM Context Transfer

Date: 2026-04-02

## Read This First

This work is now in the `surface realism / screen diagnostic / slab physics` stage, not the `dycore survival` stage.

The most important strategic change is:

- The solver baseline is good enough that it is no longer the main blocker.
- The biggest confirmed breakthrough was not a new physics scheme, but an honest screen-level diagnostic path (`screen2`) that materially improved `T2/RH2/U10/V10` agreement with HRRR relative to lowest-model-level proxies.
- GPT-5.4 Pro has been the single most useful external tool. Narrow, concrete prompts have produced most of the actual breakthroughs and the best evaluation ideas. Broad "review the model" prompts were much less useful.

Immediate current state:

- the next branch is already live on 6 nodes
- it is a bounded `moisture-memory` sweep, not a fresh solver branch
- the aim is to stop surface humidity supply from reacting instantaneously to seam activation

## Repo / Branch / Current Code State

- Main repo: `C:\Users\drew\gpu-wm`
- Active worktree: `C:\Users\drew\gpu-wm-semiimplicit-hdiv`
- Branch: `exp/tskin-slab`
- Base commit: `fa7ea3d`

Current modified / added files in the active worktree at handoff:

- Modified:
  - `CMakeLists.txt`
  - `STATUS.md`
  - `docs/MOONSHOT_NOTES.md`
  - `include/config.cuh`
  - `src/io/netcdf_output.cu`
  - `src/main.cu`
  - `src/physics/pbl.cu`
  - `src/physics/surface_slab.cu`
  - `tools/run_fast_case.py`
- Added:
  - `include/surface_layer.cuh`
  - `moonshotideas.md`
  - `tools/verify_surface_realism.py`

Important local tools added in the main repo:

- `C:\Users\drew\gpu-wm\tools\verify_surface_realism.py`
- `C:\Users\drew\gpu-wm\tools\compare_surface_experiments.py`
- `C:\Users\drew\gpu-wm\tools\live_screen2_sync.py`
- `C:\Users\drew\gpu-wm\tools\render_screen2_gifs.py`
- `C:\Users\drew\gpu-wm\tools\ops\invoke_remote_python.ps1`

## What Is Proven vs Not Proven

### Proven

1. The settled semi-implicit / hdiv-half solver family is stable enough to move on from as the main bottleneck.
2. Honest screen diagnostics are a real win.
   - `screen2` diagnostics beat legacy `*_LML` proxies against HRRR for `T2`, `RH2`, and `10 m` wind.
   - This was the first major realism breakthrough.
3. The realism stage needs its own scoreboard.
   - A front-page `surface realism` dashboard now exists via `tools/verify_surface_realism.py`.
   - It follows GPT-5.4 Pro guidance: separate solver guardrails from realism metrics.

### Not Proven Yet

1. We do not yet have a clean surface-physics branch that improves `T2/RH2/W10` together.
2. Thermal-only slab variants can move `T2/TSK`, but they have not yet solved the moisture-side problem.
3. The best next win likely lives in bounded moisture-side coupling, not more dycore work.

What is live right now:

- `A100-1` `panhandles_a1001_moistmem_t600_static_3h`
- `A100-2` `panhandles_a1002_moistmem_t600_bdry_3h`
- `A800` `panhandles_a800_moistmem_t1800_static_3h`
- `A100-4` `panhandles_a1004_moistmem_t1800_bdry_3h`
- `A100-3` `panhandles_a1003_moistmem_t3600_static_3h`
- `A100-5` `panhandles_a1005_moistmem_t3600_bdry_3h`

All 6 were verified after launch with:

- GPU utilization `100%`
- `run_fast_case.py` active
- `gpu-wm` active
- fresh `run-fast/<tag>_<timestamp>` dirs
- boundary cases already writing `gpuwm_000000.nc`

## Best Confirmed Breakthrough So Far

### Screen Diagnostics (`screen2`)

This was the first real realism jump.

The new output path writes:

- `T2`, `Q2`, `RH2`, `U10`, `V10` from a coherent screen-level path
- legacy proxy fields alongside them:
  - `T2_LML`
  - `Q2_LML`
  - `RH2_LML`
  - `U10_LML`
  - `V10_LML`

This made the comparisons honest and showed that the old proxy path had been hiding useful surface signal.

One early smoke result from the realism dashboard:

- Domain `T2 RMSE`: `3.50` vs `4.99` for `T2_LML`
- Domain `RH2 RMSE`: `11.48` vs `15.56` for `RH2_LML`
- Domain `10 m` wind vector RMSE: `4.71` vs `9.81` for `U10_LML/V10_LML`

## Experiment History After `screen2`

### 1. `windclamp-v1`

Intent:

- Terrain-aware clamp to stop `10 m` wind from looking too intense over the Rockies.

Result:

- Rejected.
- It over-damped wind broadly instead of trimming the terrain artifact intelligently.
- It produced strongly negative `10 m` wind speed bias and worse overall wind skill.

Where scores live:

- `C:\Users\drew\gpu-wm\tmp\windclamp_scores`

### 2. `flowgate-v1`

Intent:

- First pass at `Flow-Heterogeneity Surface-Layer Gate`

Result:

- Rejected.
- It helped wind somewhat, but badly regressed `RH2`.
- It was not a net realism win.

Where scores live:

- `C:\Users\drew\gpu-wm\tmp\flowgate_scores`

### 3. `admitseam-v1`

Intent:

- `Soil Thermal Admittance Seams`
- make slab response spatially heterogeneous so `TSK` can separate more meaningfully

Result:

- Partial signal, but not a real win.
- It produced a small `T2` improvement in the active seam mask.
- It still regressed `RH2` badly.

Where scores live:

- `C:\Users\drew\gpu-wm\tmp\admitseam_scores`
- paired experiment-vs-control analysis:
  - `C:\Users\drew\gpu-wm\tmp\admitseam_pair`

Most important paired admitseam result:

- In the active seam mask, `1-3 h`:
  - `T2 DeltaMAE` was slightly negative (small improvement)
  - `RH2 DeltaMAE` was strongly positive (clear regression)

Direct active-mask realized experiment-minus-control means showed the problem clearly:

- `dT2 ~= +0.013 K`
- `dTSK ~= +0.080 K`
- `dQ2 ~= +0.00213 kg/kg`
- `dRH2 ~= +21.3 points`

Interpretation:

- Thermal-only seam physics can create the right thermal signal.
- The moisture-side coupling is wrong.
- This is why the next branch moved to a bounded moisture gate.

### 4. `qvgate-v1` (Current Branch at Handoff)

Intent:

- Use the same heterogeneity / seam activation to reduce effective surface moisture availability in active cells.
- Apply it in:
  - the slab latent-flux path
  - the screen `Q2/RH2` diagnostic path

New runtime control:

- `--tskin-moisture-gate-strength`

New NetCDF field:

- `MOISTGATE`

Current state:

- The `qvgate` batch finished on the Panhandles nodes.
- Those runs have not yet been fully scored in this thread after the restart.
- This is the next thing the new thread should do.

### 5. `moistmem-v1` (Current Live Batch)

Intent:

- keep the same bounded seam activation and moisture gate strength
- add a lagged surface moisture-availability state so the moisture supply responds over time instead of instantly

New runtime control:

- `--tskin-moisture-memory-timescale`

New NetCDF field:

- `MOISTMEM`

Current live sweep:

- static / boundary pairs at `600 s`, `1800 s`, and `3600 s`
- this is the active experiment to judge next, not `qvgate-v1`

Live qvgate local GIF dirs:

- `C:\Users\drew\gpu-wm\node_plots\live_screen2\panhandles_a1001_qvgate_mod_static_3h_20260403_015025`
- `C:\Users\drew\gpu-wm\node_plots\live_screen2\panhandles_a1002_qvgate_mod_bdry_3h_20260403_015026`
- `C:\Users\drew\gpu-wm\node_plots\live_screen2\panhandles_a800_qvgate_hot_static_3h_20260403_015024`
- `C:\Users\drew\gpu-wm\node_plots\live_screen2\panhandles_a1004_qvgate_hot_bdry_3h_20260403_015026`

## Current Biggest Issue

The current bottleneck is not the dycore. It is this:

- Can we improve near-surface realism without regressing moisture?

More specifically:

- thermal-side changes can slightly improve `T2` and `TSK`
- but moisture-side coupling has repeatedly wrecked `RH2`
- the next real win probably comes from the smallest bounded moisture-side mechanism that preserves the thermal gain

## Node State At Handoff

### Finished / Idle

- `A100-1` (`ssh -p 31155 root@211.14.147.110`)
  - latest run: `panhandles_a1001_qvgate_mod_static_3h_20260403_015025`
  - run summary showed `Outputs: 4 files`
  - GPU idle at last check
- `A100-2` (`ssh -p 31697 root@157.90.56.162`)
  - latest run: `panhandles_a1002_qvgate_mod_bdry_3h_20260403_015026`
  - run summary showed normal completion metadata
  - GPU idle at last check
- `A800` (`ssh -p 40034 root@173.207.82.240`)
  - latest run: `panhandles_a800_qvgate_hot_static_3h_20260403_015024`
  - run summary showed `Outputs: 4 files`
  - GPU idle at last check
- `A100-4` (`ssh -p 30791 root@94.253.163.12`)
  - latest run: `panhandles_a1004_qvgate_hot_bdry_3h_20260403_015026`
  - run summary showed normal completion metadata
  - GPU idle at last check
- `A100-3` (`ssh -p 12472 root@192.165.134.28`)
  - latest Panhandles run: `panhandles_a1003_qvgate_xhot_bdry_3h_20260403_015307`
  - earlier also carried `eastpa_a1003_clean_bdry_12h`
  - at handoff, GPU idle; likely fully finished and ready for reuse

### Spare / Fully Staged

- `A100-5` (`ssh -p 12856 root@192.165.134.28`)
  - clean spare node
  - repo staged at `/root/gpu-wm-semiimplicit-hdiv`
  - Panhandles `f00` and `f03` binaries copied
  - NetCDF + eccodes installed
  - `sm_80` build completed at:
    - `/root/gpu-wm-semiimplicit-hdiv/build-screen2fix/gpu-wm`

## GPT-5.4 Pro Has Been Critical

This should be explicit for the next thread:

- GPT-5.4 Pro has been the highest-value external advisor in this session.
- The most useful pattern was:
  - ask one narrow question
  - give measured evidence
  - ask for the smallest next change

Broad prompts were much less helpful.

### High-value GPT Pro contributions already made

1. It correctly identified that the old `T2/RH2/U10/V10` path was too dumb and that better screen diagnostics were the next win.
2. It gave the right shape for the realism dashboard.
3. It gave the paired active-mask falsifier for `admitseam-v1`, which exposed that the branch was mostly cosmetic plus moisture-regressive.

### Best Next GPT Pro Question

Ask this or something very close:

```text
You are advising on the next smallest physics change for a CUDA regional weather model.

Context:
- Solver baseline is stable enough and is not the current bottleneck.
- Honest screen diagnostics (`screen2`) already beat lowest-model-level proxies for T2/RH2/U10/V10 against HRRR.
- A broad terrain wind clamp was rejected because it over-damped 10 m wind.
- A first flow-heterogeneity surface-layer gate was rejected because it hurt RH2 badly.
- A first thermal-only slab experiment (`admittance seams`) produced a small real T2 gain in the active heterogeneity mask, but clearly regressed RH2.

Important measured result from the paired falsifier:
- In the active seam mask (top 10% of ADMITSEAM), 1-3 h paired scores vs matched `screen2` control showed:
  - T2 DeltaMAE slightly negative (small improvement)
  - RH2 DeltaMAE strongly positive (clear regression)
- The realized active-mask experiment-minus-control mean differences were approximately:
  - dT2 ~= +0.013 K
  - dTSK ~= +0.080 K
  - dQ2 ~= +0.00213 kg/kg
  - dRH2 ~= +21.3 points
- Interpretation: the thermal signal is there, but the moisture-side coupling is wrong.

Current branch just launched:
- We added a bounded "moisture gate" tied to the same admittance/heterogeneity activation, reducing effective surface moisture availability in active cells.
- This acts in both the slab latent-flux path and the screen Q2/RH2 diagnostic path.
- It is intended to preserve the small T2 gain while preventing Q2/RH2 blow-up.

Question:
What is the single best moisture-side mechanism to try next if this simple moisture gate still fails?

I do NOT want a broad land-surface rewrite.
I want the smallest bounded change most likely to improve RH2 in the active mask without giving back T2/W10 gains and without muddying solver validation.

Please answer exactly in this format:
1. One-sentence recommendation
2. Why this is the best next move physically
3. Whether the change should act on:
   - surface moisture availability (`qv_sfc`)
   - latent heat / transfer coefficient (`Ch` or equivalent)
   - diagnostic Q2 only
   - a short moisture-memory state
   - something else
4. Minimal implementation plan
5. What success should look like in the active-mask paired 1-3 h metrics
6. What failure / fake win should look like
7. The single most likely reason this whole moisture-side path could fail even if the idea is sound
```

## Realism Dashboard

The new realism scoreboard exists in:

- `C:\Users\drew\gpu-wm\tools\verify_surface_realism.py`
- mirrored copy in the active worktree:
  - `C:\Users\drew\gpu-wm-semiimplicit-hdiv\tools\verify_surface_realism.py`

It separates solver guardrails from realism scoring and now follows the tighter second GPT-5.4 Pro answer:

- `T2` and `RH2`: `ME + MAE`
- `10 m` wind: vector RMSE + speed bias
- fixed windows like `0-3 h`, `3-12 h`
- screen-vs-`LML` head-to-head
- coarse terrain split
- stress panel for the tricky subsets

Important: keep solver guardrails separate.

Do not blend realism metrics with:

- `mean|w|`
- `max|w|`
- volume `U/V/THETA/QV` RMSE

Those remain pass/fail guardrails, not the realism score.

## WSL / Vast / Windows Quirks

These matter a lot. The next thread should know them immediately.

1. PowerShell heredocs frequently mangle remote Python / shell snippets.
   - Do not keep rediscovering this.
   - Use:
     - `C:\Users\drew\gpu-wm\tools\ops\invoke_remote_python.ps1`

2. Vast banner text pollutes parsing.
   - Every remote command may emit:
     - `Welcome to vast.ai...`
   - Do not parse command output naively without accounting for that banner.

3. Fresh A100 nodes need explicit dependency setup.
   - Install:
     - `libnetcdf-dev`
     - `libeccodes-dev`
   - Configure with:
     - `-DCMAKE_CUDA_ARCHITECTURES=80`

4. `run_fast_case.py` takes `--build`, not `--binary`.
   - The first post-restart moisture-memory launch failed on all nodes because `--binary` was passed.
   - The corrected live batch uses:
     - `python3 tools/run_fast_case.py --build build-screen2fix/gpu-wm ... -- <gpu-wm args>`

5. Windows `scp` / `ssh` can waste time if left unpoliced.
   - Stale `scp` and `ssh` processes happened repeatedly.
   - If transfer/file counts are not moving, kill and restart.
   - Do not assume "a running command" means "progress."

6. Windows path handling with `scp` is fragile.
   - Prefer simple explicit paths.
   - If copying into local Windows destinations, avoid opaque destination syntax and verify file size growth.

7. Local WSL build is the reliable path for the worktree.
   - Build the active worktree from WSL when using `build-wsl`.
   - The Windows-side build can trip over the WSL CMake cache.

8. Some nodes share the same host IP with different ports.
   - Example:
     - `192.165.134.28:12472`
     - `192.165.134.28:12856`
   - Treat them as different nodes/containers.

## GIFs / Plot Locations

Live experimental GIFs are under:

- `C:\Users\drew\gpu-wm\node_plots\live_screen2`

Current qvgate dirs:

- `panhandles_a1001_qvgate_mod_static_3h_20260403_015025`
- `panhandles_a1002_qvgate_mod_bdry_3h_20260403_015026`
- `panhandles_a800_qvgate_hot_static_3h_20260403_015024`
- `panhandles_a1004_qvgate_hot_bdry_3h_20260403_015026`

The GIF renderer now supports:

- `t2`
- `rh2`
- `wind10m`
- `wind10m_minus_lml`
- `t2_minus_lml`
- `rh2_minus_lml`
- `refc`
- `tsk`
- `admitseam`
- `moistgate`
- `moistmem`

## Recommended Immediate Next Steps For A New Thread

1. Monitor the 6 live `moistmem` runs and make sure none go idle before outputs land.
2. Pull the first completed outputs locally as soon as they appear.
3. Render the first `MOISTMEM` and standard `screen2` GIFs locally.
4. Score the 6-run `moistmem` sweep against the matched `screen2` controls.
5. Use the same paired active-mask logic that falsified `admitseam-v1`.
6. If one timescale stands out, promote it immediately.
7. If none do, ask GPT-5.4 Pro the moisture-memory follow-up question.

## Short Blunt Summary

The project is no longer trapped in solver triage. The main value now is coming from:

- honest diagnostics
- targeted surface / slab / moisture experiments
- narrow GPT-5.4 Pro prompts
- a realism dashboard that keeps solver guardrails separate

The current frontier is simple:

- `screen2` was a real win
- `windclamp-v1` was rejected
- `flowgate-v1` was rejected
- `admitseam-v1` was instructive but failed RH2
- `qvgate-v1` motivated the next branch but is no longer the live focus
- the live focus is a 6-node `moistmem-v1` timescale sweep
- `A100-5` is no longer spare; it is part of that live sweep
