# GPT-5.4 Pro Handoff Prompt

Use this repo as the working source of truth for the current state of `gpu-wm`.

Start by reading:

1. `README.md`
2. `CONTEXT_HANDOFF.md`
3. `src/main.cu`
4. `src/core/dynamics.cu`
5. `src/core/init.cu`
6. `src/core/boundaries.cu`
7. `src/core/sponge.cu`
8. `tools/init_from_gfs.py`
9. `tools/run_fast_case.py`
10. `tools/verify_forecast.py`

Important context:

- This is a CUDA regional weather-model prototype aimed in the WRF/HRRR direction.
- The current repo already has a working real-data pipeline, GPU forecast loop, NetCDF output, and verification harness.
- The biggest remaining blocker is not ingest or plotting. It is the dycore vertical pressure/mass/w closure over terrain.
- Runtime plumbing for `w[nz+1]` and nonuniform-reference lookups has been improved, but the actual acoustic/continuity/PG operators still need a more correct interface-field formulation.
- An experimental stretched-eta init path exists via `tools/init_from_gfs.py --stretched-eta`, but it is intentionally opt-in because the current dycore destabilizes quickly on that path.

What I want from you:

1. Audit the current vertical/acoustic/continuity closure in detail.
2. Identify the highest-leverage structural rewrite needed to move this closer to a true GPU-only WRF-class core.
3. Prefer dycore work over tuning.
4. Use the existing fast-case harness and verification tooling instead of proposing a totally new workflow.
5. Treat the latest uniform HRRR smoke test as the stable baseline and the stretched-eta smoke test as evidence that the bookkeeping layer is ahead of the dycore.

Concrete expectations:

- Be explicit about which current kernels are still effectively mass-level even though storage/BC plumbing has been partly modernized.
- Distinguish between fixes that are merely stabilizing and fixes that are structurally correct.
- Recommend the next coding step in terms of actual files and operator changes, not general advice.

If you propose a rewrite plan, anchor it to the current files in this repo rather than describing an abstract ideal model.
