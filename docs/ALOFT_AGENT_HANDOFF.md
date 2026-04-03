# Aloft Agent Handoff

## Mission

Own the non-surface realism lane on 3 nodes:

- `A100-4`
- `A100-3`
- `A100-5`

## Baseline / Experiment Split

- baseline/control: `A100-5`
- experiments: `A100-4`, `A100-3`

## Scientific Focus

- reflectivity evolution
- mesoscale line organization
- upper-level wind speed + vectors
- cold-pool / structure realism
- non-surface diagnostics and plots

## Primary Benchmark Direction

Use the St. Louis convective-line quick-turn benchmark after the current shared `moistmem` batch finishes on these nodes.

## Current Live Work

During handoff, all 3 owned nodes are temporarily participating in the shared 6-node `moistmem` sweep:

- `A100-4` `panhandles_a1004_moistmem_t1800_bdry_3h`
- `A100-3` `panhandles_a1003_moistmem_t3600_static_3h`
- `A100-5` `panhandles_a1005_moistmem_t3600_bdry_3h`

Do not interrupt them mid-run just to start the split.

## What To Do Next

1. As soon as the live batch finishes, stop participating in surface sweeps.
2. Set up the St. Louis convective-line domain cleanly.
3. Keep one baseline/control run alive on `A100-5`.
4. Use `A100-4` and `A100-3` for mesoscale/non-surface experiments only.
5. Produce direct GIFs for:
   - reflectivity
   - upper-level wind speed
   - upper-level vectors
   - any cold-pool / organization proxy that becomes available

## Files To Read

- `CONTEXT_TRANSFER.md`
- `docs/AGENT_SPLIT.md`
- local plot tools in `C:\\Users\\drew\\gpu-wm\\tools`
