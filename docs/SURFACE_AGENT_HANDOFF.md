# Surface Agent Handoff

## Mission

Own the near-surface realism lane on 3 nodes:

- `A100-1`
- `A100-2`
- `A800`

## Baseline / Experiment Split

- baseline/control: `A100-2`
- experiments: `A100-1`, `A800`

## Scientific Focus

- `T2`, `RH2`, `Q2`, `TSK`, `U10`, `V10`
- slab thermodynamics
- moisture supply / humidity availability
- screen diagnostics

## Current Live Work

During handoff, all 3 owned nodes are temporarily participating in the shared 6-node `moistmem` sweep:

- `A100-1` `panhandles_a1001_moistmem_t600_static_3h`
- `A100-2` `panhandles_a1002_moistmem_t600_bdry_3h`
- `A800` `panhandles_a800_moistmem_t1800_static_3h`

Do not interrupt them. Score them as soon as they finish.

## What To Do Next

1. Score the `moistmem` runs vs matched `screen2` controls.
2. Use active-mask paired `DeltaMAE_T2` and `DeltaMAE_RH2`.
3. Promote the best moisture-memory timescale only if it improves `T2` without another major `RH2` regression.
4. Keep one clean boundary baseline lane alive on `A100-2`.

## Files To Read

- `CONTEXT_TRANSFER.md`
- `docs/AGENT_SPLIT.md`
- `tools/verify_surface_realism.py`
- `tools/compare_surface_experiments.py`

