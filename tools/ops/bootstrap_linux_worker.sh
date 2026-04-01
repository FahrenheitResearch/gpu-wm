#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/FahrenheitResearch/gpu-wm.git"
BRANCH="main"
REPO_ROOT="/root/gpu-wm"
VENV_ROOT="/root/gpuwm-venv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --venv-root)
      VENV_ROOT="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; use a proper NVIDIA GPU image" >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; install a CUDA-capable build image before bootstrapping" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  libeccodes-dev \
  libnetcdf-c++4-dev \
  libnetcdf-dev \
  ninja-build \
  python3-venv \
  python3-pip \
  tmux \
  rsync \
  curl

if [[ ! -d "$REPO_ROOT/.git" ]]; then
  git clone "$REPO_URL" "$REPO_ROOT"
fi

cd "$REPO_ROOT"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

python3 -m venv "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip
python -m pip install numpy netCDF4 matplotlib eccodes

cmake -S . -B build-remote -DCMAKE_BUILD_TYPE=Release
cmake --build build-remote -j"$(nproc)"
ln -sfn build-remote build-wsl

mkdir -p output/watchdog output/watchdog/remote_jobs
touch output/watchdog/worker_queue.txt
chmod +x tools/ops/worker_tick.sh

(crontab -l 2>/dev/null | grep -v 'tools/ops/worker_tick.sh'; echo "*/10 * * * * cd $REPO_ROOT && /bin/bash tools/ops/worker_tick.sh >> output/watchdog/worker_cron.log 2>&1") | crontab -

echo "bootstrapped $REPO_ROOT on branch $BRANCH"
