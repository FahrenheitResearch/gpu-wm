#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${GPUWM_REPO_ROOT:-$(pwd)}"
WATCHDIR="${GPUWM_WATCHDIR:-$REPO_ROOT/output/watchdog}"
QUEUE_FILE="${GPUWM_QUEUE_FILE:-$WATCHDIR/worker_queue.txt}"
DEFAULT_QUEUE_FILE="${GPUWM_DEFAULT_QUEUE_FILE:-$REPO_ROOT/tools/ops/default_remote_queue.txt}"
LOG_FILE="${GPUWM_LOG_FILE:-$WATCHDIR/worker_tick.log}"
JOB_LOG_DIR="${GPUWM_JOB_LOG_DIR:-$WATCHDIR/remote_jobs}"
ACTIVE_REGEX="${GPUWM_ACTIVE_REGEX:-(/gpu-wm/build|python3? tools/run_fast_case.py|python3? tools/run_gate_matrix.py|python3? tools/run_freestream_terrain.py)}"

mkdir -p "$WATCHDIR" "$JOB_LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"$LOG_FILE"
}

gpu_snapshot() {
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | tr '\n' ';' || true
}

gpu_busy() {
  python3 - <<'PY'
import subprocess, sys
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip().splitlines()
except Exception:
    print("0")
    raise SystemExit

busy = False
for line in out:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2:
        continue
    util = int(parts[0])
    mem = int(parts[1])
    if util >= 20 or mem >= 10000:
        busy = True
        break
print("1" if busy else "0")
PY
}

has_active_run() {
  ps -eo pid,args | grep -E "$ACTIVE_REGEX" | grep -vE 'grep|worker_tick|tmux' >/dev/null 2>&1
}

pop_queue_line() {
  python3 - "$QUEUE_FILE" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.exists():
    sys.exit(0)
lines = path.read_text().splitlines()
cmd = None
rest = []
for line in lines:
    if cmd is None and line.strip() and not line.lstrip().startswith("#"):
        cmd = line.strip()
    else:
        rest.append(line)
if cmd is None:
    sys.exit(0)
path.write_text("\n".join(rest) + ("\n" if rest else ""))
print(cmd)
PY
}

refill_queue_from_default() {
  python3 - "$QUEUE_FILE" "$DEFAULT_QUEUE_FILE" <<'PY'
import pathlib, sys
queue_path = pathlib.Path(sys.argv[1])
default_path = pathlib.Path(sys.argv[2])
if not default_path.exists():
    print("0")
    raise SystemExit
default_lines = [
    line.rstrip("\n")
    for line in default_path.read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if not default_lines:
    print("0")
    raise SystemExit
existing = queue_path.read_text().splitlines() if queue_path.exists() else []
queue_path.write_text("\n".join(existing + default_lines) + "\n")
print("1")
PY
}

start_next_job() {
  local cmd="$1"
  local stamp session joblog
  stamp="$(date '+%Y%m%d_%H%M%S')"
  session="gpuwm-job-$stamp"
  joblog="$JOB_LOG_DIR/$stamp.log"
  tmux new-session -d -s "$session" "cd '$REPO_ROOT' && (export PYTHONUNBUFFERED=1; source /root/gpuwm-venv/bin/activate 2>/dev/null || true; $cmd) >> '$joblog' 2>&1"
  log "started session=$session log=$joblog cmd=$cmd"
}

gpu="$(gpu_snapshot)"
if has_active_run || [[ "$(gpu_busy)" == "1" ]]; then
  runs="$(ps -eo pid,args | grep -E "$ACTIVE_REGEX" | grep -vE 'grep|worker_tick|tmux' | tr '\n' ';' || true)"
  log "busy gpu=[$gpu] runs=[$runs]"
  exit 0
fi

log "idle gpu=[$gpu]"
next="$(pop_queue_line || true)"
if [[ -z "${next:-}" ]]; then
  if [[ "$(refill_queue_from_default)" == "1" ]]; then
    log "queue refilled from default template"
    next="$(pop_queue_line || true)"
  fi
fi

if [[ -z "${next:-}" ]]; then
  log "queue empty"
  exit 0
fi

start_next_job "$next"
