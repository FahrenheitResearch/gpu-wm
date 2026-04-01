# Cluster Experiment Operations

Last updated: 2026-04-01

## Goal

`gpu-wm` should scale from one local workstation to many rented GPU workers without changing the experiment discipline.

The design target is:

- one controller that decides what to try next
- one permanent moonshot research sidecar agent that keeps searching for high-upside ideas
- many workers that:
  - keep the repo up to date
  - watch a queue file
  - launch the next job when idle
  - log enough state to recover what happened
  - do all of that without manual babysitting

## Worker Model

Each worker owns a local repo checkout and runs a one-shot `10 minute` tick.

The tick does four things:

1. inspect GPU utilization and active `gpu-wm` processes
2. if the node is already busy, log status and exit
3. if the node is idle, pop the next command from a queue file
4. launch the job in a detached `tmux` session with its own log file

This is intentionally simple. The worker should not try to be smart about science. The controller decides what to enqueue. The worker only guarantees:

- jobs keep starting
- logs keep being written
- idle nodes do not stay idle for long

The permanent moonshot sidecar agent has a different job:

- keep scanning for high-upside algorithmic or physics ideas
- turn those ideas into bounded experiments
- never directly mutate the live worker queues without controller review

During active pushes, the worker tick should stay installed on every active node. If a machine should pause, empty its queue instead of killing the tick.

## Standard Layout

On each Linux worker:

- repo root: for example `/root/gpu-wm`
- build dir: `/root/gpu-wm/build-remote`
- Python venv: `/root/gpuwm-venv`
- queue file: `/root/gpu-wm/output/watchdog/worker_queue.txt`
- worker log: `/root/gpu-wm/output/watchdog/worker_tick.log`
- per-job logs: `/root/gpu-wm/output/watchdog/remote_jobs/`

On the local Windows workstation:

- repo root: for example `C:\Users\drew\gpu-wm`
- tracked tick script: `tools/ops/local_worker_tick.ps1`
- tracked installer: `tools/ops/install_local_worker_task.ps1`
- queue file: `output/watchdog/local_queue.txt`
- worker log: `output/watchdog/local_tick.log`
- per-job logs: `output/watchdog/local_jobs\`

## Install Standard

For rented Linux GPU nodes, use:

```bash
cd /root
git clone https://github.com/FahrenheitResearch/gpu-wm.git
cd gpu-wm
bash tools/ops/bootstrap_linux_worker.sh --branch main
```

That script is the standard install path for:

- `A100`
- `H100`
- `H200`

The script assumes:

- NVIDIA driver is already present
- `nvidia-smi` works
- CUDA toolkit / `nvcc` is available on the node image

The bootstrap script also installs the native development packages needed for:

- NetCDF output
- ecCodes / GRIB support

If a provider image does not include CUDA build tooling, fix the image first. Do not solve that ad hoc per experiment.

For the local Windows workstation, install the `10 minute` worker tick with:

```powershell
powershell -ExecutionPolicy Bypass -File tools\ops\install_local_worker_task.ps1
```

That makes the workstation follow the same queue discipline as rented nodes instead of relying on hidden ad hoc scripts.

## Queue Discipline

Queue files should contain exactly one shell command per line.

Good queue items:

- one branch
- one hypothesis
- one run tag
- one clear log artifact
- for serious regional runs, include postprocess so the job yields plots as well as NetCDF

Examples:

```bash
python tools/run_gate_matrix.py --profile wdamp-erf-pure --cases uniform_120 stretch_120
python tools/run_fast_case.py --postprocess-weather --init data/gfs_init_eastpa_20260401_t00z_768x640x50_dx4000_terrain.bin --nx 768 --ny 640 --nz 50 --dx 4000 --dt 8 --ref-lat 40.5 --ref-lon -76.5 --tend 3600 --output-interval 900 --tag eastpa_static_dt8 -- --w-damp --w-damp-alpha 6.0 --w-damp-beta 0.0 --w-transport-blend 1.0
```

Bad queue items:

- giant shell pipelines with side effects
- mixed code edits + model runs in one line
- “run whatever looks useful”
- commands that download large datasets every time

## Bandwidth Rules

Bandwidth is a real cost center. Treat it as one.

Rules:

- reuse init binaries and boundary binaries already on the node
- do not keep re-downloading GRIBs
- do not copy large NetCDF outputs back unless they are worth keeping
- move small logs, metrics, and selected PNGs first
- only sync a full run directory when the run is a real keeper

The fastest path is usually:

- build once
- copy one init binary once
- copy one boundary binary once
- run many ablations locally on the node
- collect only compact summaries

## What Workers Should Run First

Workers should always prefer the benchmark ladder:

1. free-stream-over-terrain
2. canonical 64x64x20 uniform/stretched gates
3. medium real-data `+1 h`
4. large real-data `+1 h`
5. longer real-data runs only after the shorter case is clean

Do not fill expensive nodes with long hero runs while the `+1 h` regional failure is still unresolved.

## Controller Rules

The controller should:

- keep the local workstation and remote nodes busy unless they are intentionally paused
- treat queue underflow on paid nodes as a fault, not a benign state
- prefer one hypothesis per branch and one run per queue item
- commit status updates whenever a result materially changes the diagnosis
- converge workers onto the tracked `tools/ops/*` scripts instead of stale ad hoc watchdogs
- let the moonshot sidecar generate ideas, but only promote ideas that survive the benchmark ladder

## No-Idle Rule

The repo now carries tracked fallback queues:

- `tools/ops/default_local_queue.txt`
- `tools/ops/default_remote_queue.txt`

If a live queue drains while a worker is otherwise healthy, the worker tick should reseed from the fallback queue and immediately launch the next job. On rented nodes, `queue empty` should only happen when both the live queue and fallback queue are intentionally empty.

## Current Scientific Priority

The current priority is not new physics. It is solver correctness:

- continuity-consistent startup `w`
- interface-consistent slow `w` terms
- open-boundary / sponge cleanup
- large-domain real-data stability

The worker system should serve that loop, not distract from it.
