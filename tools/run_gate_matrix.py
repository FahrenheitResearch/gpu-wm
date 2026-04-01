#!/usr/bin/env python3
"""Run GPU-WM benchmark gates and enforce simple stability thresholds."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_FAST_CASE = REPO_ROOT / "tools" / "run_fast_case.py"

PROFILE_ARGS: Dict[str, List[str]] = {
    "baseline": [],
    # Strongest known w-damp setting that still passes the stretched 15 min,
    # 1 h, and 6 h gate set as of 2026-03-31.
    "wdamp": ["--w-damp", "--w-damp-alpha", "6.0", "--w-damp-beta", "0.0"],
    "wdamp-legacy": ["--w-damp", "--w-damp-alpha", "3.0", "--w-damp-beta", "0.0"],
    "wdamp-moderate": ["--w-damp", "--w-damp-alpha", "2.0", "--w-damp-beta", "0.2"],
}


@dataclass(frozen=True)
class GateThresholds:
    u_rmse_max: float
    v_rmse_max: float
    theta_rmse_max: float
    mean_abs_w_max: float
    max_abs_w_max: float


@dataclass(frozen=True)
class GateCase:
    name: str
    init_rel: str
    tend: int
    output_interval: int
    thresholds: GateThresholds


CASES: List[GateCase] = [
    GateCase(
        "uniform_120",
        "data/hrrr_init_fast_64x64x20_dx3000_latp38.5_lonm097.5_terrain.bin",
        120,
        120,
        GateThresholds(8.0, 8.0, 1.0, 6.5, 50.0),
    ),
    GateCase(
        "stretch_120",
        "data/ifw_stretch_v4.bin",
        120,
        120,
        GateThresholds(11.5, 11.5, 11.0, 8.0, 45.0),
    ),
    GateCase(
        "stretch_900",
        "data/ifw_stretch_v4.bin",
        900,
        900,
        GateThresholds(17.0, 17.5, 17.0, 4.0, 35.0),
    ),
    GateCase(
        "stretch_3600",
        "data/ifw_stretch_v4.bin",
        3600,
        900,
        GateThresholds(19.0, 19.5, 19.0, 3.5, 31.0),
    ),
    GateCase(
        "stretch_21600",
        "data/ifw_stretch_v4.bin",
        21600,
        3600,
        GateThresholds(19.5, 20.0, 19.0, 3.5, 31.0),
    ),
]

HEALTH_RE = re.compile(
    r"health:\s+mean_u=.*?mean_v=.*?mean_w=([+\-0-9.eE]+)\s+"
    r"mean\|w\|=([+\-0-9.eE]+)\s+max\|w\|=([+\-0-9.eE]+)"
)
FULL_RE = re.compile(
    r"full:\s+U rmse=([+\-0-9.eE]+)\s+V rmse=([+\-0-9.eE]+)\s+"
    r"THETA rmse=([+\-0-9.eE]+)\s+QV rmse=([+\-0-9.eE]+)"
)
FLAGS_RE = re.compile(r"flags:\s+(.*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_ARGS),
        default="wdamp",
        help="Model-argument profile to apply to each gate",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Subset of case names to run (default: all except stretch_21600)",
    )
    parser.add_argument(
        "--include-6h",
        action="store_true",
        help="Include the 6-hour stretched gate",
    )
    parser.add_argument(
        "--no-enforce",
        action="store_true",
        help="Report metrics without failing on threshold violations",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for JSON summary output",
    )
    parser.add_argument(
        "--tag-prefix",
        default="gate",
        help="Prefix for run_fast_case tags",
    )
    parser.add_argument(
        "extra_model_args",
        nargs=argparse.REMAINDER,
        help="Additional model args after '--'",
    )
    return parser.parse_args()


def selected_cases(args: argparse.Namespace) -> List[GateCase]:
    include = set(args.cases) if args.cases else None
    result: List[GateCase] = []
    for case in CASES:
        if case.name == "stretch_21600" and not args.include_6h:
            continue
        if include is not None and case.name not in include:
            continue
        result.append(case)
    if not result:
        raise SystemExit("No benchmark cases selected")
    return result


def parse_summary(stdout: str) -> Dict[str, object]:
    health_matches = HEALTH_RE.findall(stdout)
    full_matches = FULL_RE.findall(stdout)
    flags_matches = FLAGS_RE.findall(stdout)
    if not health_matches or not full_matches:
        raise RuntimeError("Could not parse verification summary from run_fast_case output")

    mean_w, mean_abs_w, max_abs_w = [float(v) for v in health_matches[-1]]
    u_rmse, v_rmse, theta_rmse, qv_rmse = [float(v) for v in full_matches[-1]]
    flags = flags_matches[-1].strip() if flags_matches else ""
    return {
        "mean_w": mean_w,
        "mean_abs_w": mean_abs_w,
        "max_abs_w": max_abs_w,
        "u_rmse": u_rmse,
        "v_rmse": v_rmse,
        "theta_rmse": theta_rmse,
        "qv_rmse": qv_rmse,
        "flags": flags,
    }


def evaluate_case(summary: Dict[str, object], thresholds: GateThresholds) -> List[str]:
    failures: List[str] = []
    if summary["u_rmse"] > thresholds.u_rmse_max:
        failures.append(f"U>{thresholds.u_rmse_max}")
    if summary["v_rmse"] > thresholds.v_rmse_max:
        failures.append(f"V>{thresholds.v_rmse_max}")
    if summary["theta_rmse"] > thresholds.theta_rmse_max:
        failures.append(f"TH>{thresholds.theta_rmse_max}")
    if summary["mean_abs_w"] > thresholds.mean_abs_w_max:
        failures.append(f"|w|>{thresholds.mean_abs_w_max}")
    if summary["max_abs_w"] > thresholds.max_abs_w_max:
        failures.append(f"max|w|>{thresholds.max_abs_w_max}")
    return failures


def run_case(case: GateCase, args: argparse.Namespace) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(RUN_FAST_CASE),
        "--init",
        str(REPO_ROOT / case.init_rel),
        "--nx",
        "64",
        "--ny",
        "64",
        "--nz",
        "20",
        "--dx",
        "3000",
        "--ztop",
        "25000",
        "--tend",
        str(case.tend),
        "--output-interval",
        str(case.output_interval),
        "--tag",
        f"{args.tag_prefix}_{case.name}",
        "--skip-init-plot",
    ]
    extra_args = PROFILE_ARGS[args.profile] + args.extra_model_args
    if extra_args:
        cmd.append("--")
        cmd.extend(extra_args)

    print(f"[gate] {case.name}: {' '.join(extra_args) if extra_args else '(no extra model args)'}")
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{case.name} failed with exit code {proc.returncode}")

    summary = parse_summary(proc.stdout)
    summary["case"] = case.name
    summary["tend_s"] = case.tend
    summary["threshold_failures"] = evaluate_case(summary, case.thresholds)
    summary["passed"] = not summary["threshold_failures"]
    return summary


def main() -> int:
    args = parse_args()
    cases = selected_cases(args)
    results = [run_case(case, args) for case in cases]

    print("\nGate Summary")
    print("case               U_rmse  V_rmse  TH_rmse  mean|w|  max|w|  status")
    for result in results:
        status = "PASS" if result["passed"] else "FAIL:" + ",".join(result["threshold_failures"])
        print(
            f"{result['case']:<18} "
            f"{result['u_rmse']:>6.2f}  {result['v_rmse']:>6.2f}  "
            f"{result['theta_rmse']:>7.2f}  {result['mean_abs_w']:>7.2f}  "
            f"{result['max_abs_w']:>7.2f}  {status}"
        )

    payload = {
        "cwd": str(REPO_ROOT),
        "profile": args.profile,
        "cases": results,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.no_enforce:
        return 0
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
