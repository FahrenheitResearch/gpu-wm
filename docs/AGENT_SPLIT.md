# Two-Agent Node Ownership Split

Date: 2026-04-02

This file defines the operating split for two concurrent Codex agents so node control does not overlap.

## Purpose

Use two owner agents with disjoint node sets and disjoint scientific scope:

- `surface-agent`: near-surface physics and diagnostics
- `aloft-agent`: non-surface / mesoscale / convective structure and upper-level realism

This split is better than `static vs boundary` because the questions and metrics are different.

## Global Rules

1. Each agent owns exactly 3 nodes.
2. Each agent keeps 1 baseline/control lane and 2 experiment lanes.
3. Agents must not launch on each other's nodes.
4. Agents must not change the other agent's repo, run dirs, or remote logs.
5. Surface and aloft lanes may use different benchmark cases.
6. Solver guardrails remain global pass/fail checks for both lanes.

## Current Live Batch

At the moment this file was written, all 6 nodes are running the surface `moistmem` sweep:

- `A100-1` `panhandles_a1001_moistmem_t600_static_3h`
- `A100-2` `panhandles_a1002_moistmem_t600_bdry_3h`
- `A800` `panhandles_a800_moistmem_t1800_static_3h`
- `A100-4` `panhandles_a1004_moistmem_t1800_bdry_3h`
- `A100-3` `panhandles_a1003_moistmem_t3600_static_3h`
- `A100-5` `panhandles_a1005_moistmem_t3600_bdry_3h`

Do not kill or steal these runs mid-flight just to satisfy the split. Let them finish, score them, then reassign by the ownership below.

## Surface Agent Ownership

Scope:

- `T2`
- `RH2`
- `Q2`
- `TSK`
- `U10/V10`
- surface moisture supply
- slab / screen-diagnostic / PBL-adjacent near-surface realism

Primary case:

- Panhandles HRRR benchmark

Owned nodes:

- `A100-1` `ssh -p 31155 root@211.14.147.110`
- `A100-2` `ssh -p 31697 root@157.90.56.162`
- `A800` `ssh -p 40034 root@173.207.82.240`

Roles after current live batch finishes:

- baseline/control: `A100-2`
- experiment lane 1: `A100-1`
- experiment lane 2: `A800`

Why:

- `A100-2` is the cleanest boundary-forced realism lane for comparison against HRRR.
- `A100-1` and `A800` are good experiment lanes for static or alternate surface physics variants.

## Aloft Agent Ownership

Scope:

- reflectivity / convective line morphology
- cold-pool / mesoscale organization
- upper-level winds and vectors
- non-surface realism plots and diagnostics
- optional St. Louis convective-line quick-turn benchmark

Primary case after current surface batch:

- St. Louis convective-line domain / latest interesting weather case

Owned nodes:

- `A100-4` `ssh -p 30791 root@94.253.163.12`
- `A100-3` `ssh -p 12472 root@192.165.134.28`
- `A100-5` `ssh -p 12856 root@192.165.134.28`

Roles after current live batch finishes:

- baseline/control: `A100-5`
- experiment lane 1: `A100-4`
- experiment lane 2: `A100-3`

Why:

- this gives the aloft lane 3 clean A100-class nodes without touching the surface lane
- `A100-5` is the newest clean setup, so it is a good baseline/control lane

## Per-Agent Outputs

Each agent should maintain its own:

- remote run tags prefixed with its lane intent
- local score directory
- local GIF directory
- short markdown status file

Recommended naming:

- surface:
  - tags start with `surf_`
  - scores under `tmp/surface_lane_*`
  - GIFs under `node_plots/surface_lane_*`
- aloft:
  - tags start with `aloft_`
  - scores under `tmp/aloft_lane_*`
  - GIFs under `node_plots/aloft_lane_*`

## Near-Term Transition Plan

1. Let the current 6-node `moistmem` surface batch finish.
2. Surface agent scores the full batch and decides whether one moisture-memory timescale is worth promoting.
3. Aloft agent uses its 3 owned nodes only after the live batch ends on those nodes.
4. Aloft agent should then set up the St. Louis convective-line benchmark and keep one clean baseline plus two experiment lanes.

## Current Spawned Owner Agents

At the time this file was updated, the active Codex worker agents were:

- surface owner agent: `Dalton` id `019d5145-2b45-7dd0-9648-7b65487221c2`
- aloft owner agent: `Sagan` id `019d5145-4b59-77b0-9a1a-c2fcb070e601`

If a future thread resumes control, reuse these agents if possible instead of spawning duplicate owners.

## Non-Negotiable Anti-Confusion Rules

1. No node borrowing across agents without an explicit ownership change written here first.
2. No silent relaunches on another agent's node.
3. No shared baseline node.
4. Keep the scientific question separate:
   - surface agent answers near-surface realism questions
   - aloft agent answers convective / upper-air realism questions
