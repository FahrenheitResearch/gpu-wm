#!/usr/bin/env python3
"""Run the idealized free-stream-over-terrain regression case."""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="build-wsl/gpu-wm", help="Path to gpu-wm executable")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nz", type=int, default=30)
    parser.add_argument("--dx", type=float, default=1000.0)
    parser.add_argument("--ztop", type=float, default=15000.0)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--tend", type=float, default=1800.0)
    parser.add_argument("--output-interval", type=float, default=None)
    parser.add_argument("--diag-interval", type=int, default=60)
    parser.add_argument("--tag", default="freestream")
    parser.add_argument("extra_model_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_state(path: Path) -> dict[str, np.ndarray]:
    with nc.Dataset(path) as ds:
        return {
            "time": np.array(ds.variables["time"][:], dtype=np.float64).ravel()[0],
            "U": np.array(ds.variables["U"][:], dtype=np.float64),
            "V": np.array(ds.variables["V"][:], dtype=np.float64),
            "W": np.array(ds.variables["W"][:], dtype=np.float64),
            "THETA": np.array(ds.variables["THETA"][:], dtype=np.float64),
            "TERRAIN": np.array(ds.variables["TERRAIN"][:], dtype=np.float64),
        }


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return math.sqrt(float(np.mean((a - b) ** 2)))


def main() -> int:
    args = parse_args()
    output_interval = args.output_interval if args.output_interval is not None else args.tend
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "run-fast" / f"{args.tag}_{timestamp}"
    (run_dir / "output").mkdir(parents=True, exist_ok=True)

    build_path = Path(args.build)
    if not build_path.is_absolute():
        build_path = REPO_ROOT / build_path

    extra_args = list(args.extra_model_args)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]

    cmd = [
        str(build_path),
        "--test", "4",
        "--nx", str(args.nx),
        "--ny", str(args.ny),
        "--nz", str(args.nz),
        "--dx", str(args.dx),
        "--ztop", str(args.ztop),
        "--dt", str(args.dt),
        "--tend", str(args.tend),
        "--output-interval", str(output_interval),
        "--diag-interval", str(args.diag_interval),
        "--netcdf",
    ]
    cmd.extend(extra_args)

    print(f"$ (cd {run_dir} && {' '.join(shlex.quote(part) for part in cmd)})")
    subprocess.run(cmd, cwd=run_dir, check=True)

    outputs = sorted((run_dir / "output").glob("gpuwm_*.nc"))
    if len(outputs) < 2:
        raise RuntimeError("expected initial and final NetCDF outputs")

    initial = load_state(outputs[0])
    final = load_state(outputs[-1])

    u_rmse = rmse(final["U"], initial["U"])
    v_rmse = rmse(final["V"], initial["V"])
    theta_rmse = rmse(final["THETA"], initial["THETA"])
    mean_w = float(np.mean(final["W"]))
    mean_abs_w = float(np.mean(np.abs(final["W"])))
    max_abs_w = float(np.max(np.abs(final["W"])))
    terrain_max = float(np.max(final["TERRAIN"]))

    print("\nFree-stream Terrain Summary")
    print(f"run_dir:    {run_dir}")
    print(f"terrain:    max={terrain_max:.1f} m")
    print(f"time:       {final['time']:.1f} s")
    print(f"U/V/TH rmse {u_rmse:.3f} / {v_rmse:.3f} / {theta_rmse:.3f}")
    print(f"w stats:    mean={mean_w:+.5f}  mean|w|={mean_abs_w:.5f}  max|w|={max_abs_w:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
