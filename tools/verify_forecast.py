#!/usr/bin/env python3
"""GPU-WM forecast verification and drift diagnostics.

Compares GPU-WM NetCDF outputs against a reference state. The reference can be:
  - the binary initialization file produced by tools/init_from_gfs.py
  - another GPU-WM NetCDF output file

This is intentionally verification-focused, not visualization-focused:
it reports bias/RMSE/correlation plus dynamical health metrics like
domain-mean vertical velocity drift.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct
import sys
from typing import Dict, List, Tuple

import numpy as np

try:
    from netCDF4 import Dataset
except ImportError:
    print("ERROR: netCDF4 required. Install with: pip install netCDF4")
    sys.exit(1)


TARGET_LEVELS_M = [
    ("sfc", 250.0),
    ("850ish", 1500.0),
    ("500ish", 5500.0),
]

FIELD_MAP = [
    ("U", "u"),
    ("V", "v"),
    ("THETA", "theta"),
    ("QV", "qv"),
]

PROJ_MAGIC = b"GWMPRJ1\x00"
PRES_MAGIC = b"GWMPRES1"
TERRAIN_MAGIC = b"GWMTERR1"
INIT_MODE_MAGIC = b"GWMINIT1"
TIME_MAGIC = b"GWMTIME1"


def load_netcdf_state(path: str) -> Dict[str, np.ndarray]:
    ds = Dataset(path, "r")

    def read_first_available(*names: str) -> np.ndarray:
        for name in names:
            if name in ds.variables:
                return np.array(ds.variables[name][:], dtype=np.float64)
        raise KeyError(f"Missing variables {names} in {path}")

    data = {
        "path": path,
        "format": "netcdf",
        "nx": len(ds.dimensions["x"]),
        "ny": len(ds.dimensions["y"]),
        "nz": len(ds.dimensions["z"]),
        "time": float(np.array(ds.variables["time"][:]).ravel()[0]),
        "z": np.array(ds.variables["z"][:], dtype=np.float64),
        "U": read_first_available("U_MASS", "U"),
        "V": read_first_available("V_MASS", "V"),
        "W": read_first_available("W_MASS", "W"),
        "THETA": np.array(ds.variables["THETA"][:], dtype=np.float64),
        "QV": np.array(ds.variables["QV"][:], dtype=np.float64),
        "QC": np.array(ds.variables["QC"][:], dtype=np.float64),
        "QR": np.array(ds.variables["QR"][:], dtype=np.float64),
    }
    ds.close()
    return data


def load_init_binary_state(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        header = f.read(3 * 4 + 3 * 8)
        if len(header) != 36:
            raise RuntimeError("Failed to read binary init header")
        nx, ny, nz = struct.unpack("iii", header[:12])
        dx, dy, ztop = struct.unpack("ddd", header[12:])

        z = np.frombuffer(f.read(nz * 8), dtype=np.float64).copy()
        n3d = nx * ny * nz

        def read_field(name: str) -> np.ndarray:
            raw = f.read(n3d * 8)
            if len(raw) != n3d * 8:
                raise RuntimeError(f"Failed to read field {name} from {path}")
            return np.frombuffer(raw, dtype=np.float64).reshape(nz, ny, nx).copy()

        data = {
            "path": path,
            "format": "init_binary",
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "time": 0.0,
            "dx": dx,
            "dy": dy,
            "ztop": ztop,
            "z": z,
            "U": read_field("u"),
            "V": read_field("v"),
            "W": read_field("w"),
            "THETA": read_field("theta"),
            "QV": read_field("qv"),
            "QC": read_field("qc"),
            "QR": read_field("qr"),
        }

        while True:
            magic = f.read(8)
            if not magic:
                break
            if len(magic) != 8:
                raise RuntimeError("Binary init file has truncated trailer magic")
            if magic == PROJ_MAGIC:
                raw = f.read(5 * 8)
                if len(raw) != 5 * 8:
                    raise RuntimeError("Binary init file has truncated projection trailer")
                proj_vals = struct.unpack("ddddd", raw)
                data["projection"] = {
                    "truelat1": proj_vals[0],
                    "truelat2": proj_vals[1],
                    "stand_lon": proj_vals[2],
                    "ref_lat": proj_vals[3],
                    "ref_lon": proj_vals[4],
                }
            elif magic == PRES_MAGIC:
                raw = f.read(n3d * 8)
                if len(raw) != n3d * 8:
                    raise RuntimeError("Binary init file has truncated pressure trailer")
                pressure = np.frombuffer(raw, dtype=np.float64)
                data["pressure_stats"] = {
                    "min": float(np.min(pressure)),
                    "max": float(np.max(pressure)),
                    "mean": float(np.mean(pressure)),
                }
            elif magic == TERRAIN_MAGIC:
                raw = f.read(nx * ny * 8)
                if len(raw) != nx * ny * 8:
                    raise RuntimeError("Binary init file has truncated terrain trailer")
                # Terrain is optional metadata; keep lightweight stats instead of duplicating the array.
                terrain = np.frombuffer(raw, dtype=np.float64)
                data["terrain_stats"] = {
                    "min": float(np.min(terrain)),
                    "max": float(np.max(terrain)),
                    "mean": float(np.mean(terrain)),
                }
            elif magic == INIT_MODE_MAGIC:
                raw = f.read(2 * 4)
                if len(raw) != 2 * 4:
                    raise RuntimeError("Binary init file has truncated init-mode trailer")
                terrain_following_init, reserved = struct.unpack("ii", raw)
                data["init_mode"] = {
                    "terrain_following_init": bool(terrain_following_init),
                    "reserved": int(reserved),
                }
            elif magic == TIME_MAGIC:
                raw = f.read(2 * 8 + 2 * 4)
                if len(raw) != 2 * 8 + 2 * 4:
                    raise RuntimeError("Binary init file has truncated time trailer")
                valid_unix, reference_unix, forecast_hour, reserved = struct.unpack("qqii", raw)
                data["time_metadata"] = {
                    "valid_unix": int(valid_unix),
                    "reference_unix": int(reference_unix),
                    "forecast_hour": int(forecast_hour),
                    "reserved": int(reserved),
                }
            else:
                raise RuntimeError(
                    "Binary init file has unsupported trailing bytes; "
                    "expected known GPU-WM trailers"
                )

    return data


def load_state(path: str) -> Dict[str, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".nc":
        return load_netcdf_state(path)
    if ext == ".bin":
        return load_init_binary_state(path)
    raise RuntimeError(f"Unsupported reference format: {path}")


def validate_compatibility(ref: Dict[str, np.ndarray], fcst: Dict[str, np.ndarray]) -> None:
    for key in ("nx", "ny", "nz"):
        if ref[key] != fcst[key]:
            raise RuntimeError(
                f"Grid mismatch for {fcst['path']}: "
                f"forecast={fcst['nx']}x{fcst['ny']}x{fcst['nz']} "
                f"reference={ref['nx']}x{ref['ny']}x{ref['nz']}"
            )


def nearest_level(z_levels: np.ndarray, target_m: float) -> int:
    return int(np.argmin(np.abs(z_levels - target_m)))


def scalar_metrics(fcst: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    diff = fcst - ref
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    ref_mean = float(np.mean(ref))
    fcst_mean = float(np.mean(fcst))

    fcst_flat = fcst.ravel()
    ref_flat = ref.ravel()
    fcst_std = float(np.std(fcst_flat))
    ref_std = float(np.std(ref_flat))
    if fcst_std < 1.0e-12 or ref_std < 1.0e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(fcst_flat, ref_flat)[0, 1])

    return {
        "bias": bias,
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "forecast_mean": fcst_mean,
        "reference_mean": ref_mean,
    }


def compute_health_metrics(state: Dict[str, np.ndarray]) -> Dict[str, float]:
    w = state["W"]
    u = state["U"]
    v = state["V"]
    theta = state["THETA"]
    qv = state["QV"]

    return {
        "mean_u": float(np.mean(u)),
        "mean_v": float(np.mean(v)),
        "mean_w": float(np.mean(w)),
        "mean_abs_w": float(np.mean(np.abs(w))),
        "max_abs_w": float(np.max(np.abs(w))),
        "mean_theta": float(np.mean(theta)),
        "mean_qv": float(np.mean(qv)),
        "max_qc": float(np.max(state["QC"])),
        "max_qr": float(np.max(state["QR"])),
    }


def compute_regional_band_metrics(
    fcst: Dict[str, np.ndarray], ref: Dict[str, np.ndarray], outer_band: int
) -> Dict[str, Dict[str, float]]:
    nx = int(fcst["nx"])
    ny = int(fcst["ny"])
    if outer_band <= 0 or outer_band * 2 >= nx or outer_band * 2 >= ny:
        return {}

    mask2d_outer = np.zeros((ny, nx), dtype=bool)
    mask2d_outer[:outer_band, :] = True
    mask2d_outer[-outer_band:, :] = True
    mask2d_outer[:, :outer_band] = True
    mask2d_outer[:, -outer_band:] = True
    mask2d_inner = ~mask2d_outer

    qtot_fcst = fcst["QV"] + fcst["QC"] + fcst["QR"]
    qtot_ref = ref["QV"] + ref["QC"] + ref["QR"]
    eps = 1.0e-12

    def band_metrics(mask2d: np.ndarray) -> Dict[str, float]:
        mask3d = np.broadcast_to(mask2d[np.newaxis, :, :], fcst["THETA"].shape)
        fcst_u = fcst["U"][mask3d]
        ref_u = ref["U"][mask3d]
        fcst_v = fcst["V"][mask3d]
        ref_v = ref["V"][mask3d]
        fcst_th = fcst["THETA"][mask3d]
        ref_th = ref["THETA"][mask3d]
        fcst_qv = fcst["QV"][mask3d]
        ref_qv = ref["QV"][mask3d]
        fcst_w = fcst["W"][mask3d]
        qtot_f = qtot_fcst[mask3d]
        qtot_r = qtot_ref[mask3d]

        qtot_ref_sum = float(np.sum(qtot_r))
        qtot_fcst_sum = float(np.sum(qtot_f))
        return {
            "cells": int(mask3d.sum()),
            "u_rmse": float(np.sqrt(np.mean((fcst_u - ref_u) ** 2))),
            "v_rmse": float(np.sqrt(np.mean((fcst_v - ref_v) ** 2))),
            "theta_rmse": float(np.sqrt(np.mean((fcst_th - ref_th) ** 2))),
            "qv_rmse": float(np.sqrt(np.mean((fcst_qv - ref_qv) ** 2))),
            "theta_bias": float(np.mean(fcst_th - ref_th)),
            "mean_w": float(np.mean(fcst_w)),
            "mean_abs_w": float(np.mean(np.abs(fcst_w))),
            "max_abs_w": float(np.max(np.abs(fcst_w))),
            "qtot_reference_sum": qtot_ref_sum,
            "qtot_forecast_sum": qtot_fcst_sum,
            "qtot_rel_change_pct": float(100.0 * (qtot_fcst_sum - qtot_ref_sum) / max(abs(qtot_ref_sum), eps)),
        }

    return {
        f"outer_{outer_band}": band_metrics(mask2d_outer),
        "interior": band_metrics(mask2d_inner),
    }


def evaluate_flags(
    health: Dict[str, float], ref_health: Dict[str, float]
) -> List[str]:
    flags: List[str] = []

    if abs(health["mean_w"]) > 0.05:
        flags.append(f"mean_w drift {health['mean_w']:+.3f} m/s")
    if health["mean_abs_w"] > 2.0:
        flags.append(f"mean|w| high {health['mean_abs_w']:.2f} m/s")
    if health["max_abs_w"] > 30.0:
        flags.append(f"max|w| high {health['max_abs_w']:.1f} m/s")

    du = health["mean_u"] - ref_health["mean_u"]
    dv = health["mean_v"] - ref_health["mean_v"]
    if abs(du) > 2.0:
        flags.append(f"mean_u drift {du:+.2f} m/s")
    if abs(dv) > 2.0:
        flags.append(f"mean_v drift {dv:+.2f} m/s")

    if health["max_qc"] > 0.005:
        flags.append(f"qc peak high {health['max_qc']:.4f} kg/kg")
    if health["max_qr"] > 0.005:
        flags.append(f"qr peak high {health['max_qr']:.4f} kg/kg")

    return flags


def summarize_case(
    fcst: Dict[str, np.ndarray], ref: Dict[str, np.ndarray], outer_band: int = 20
) -> Dict[str, object]:
    validate_compatibility(ref, fcst)

    summary: Dict[str, object] = {
        "path": fcst["path"],
        "time_seconds": float(fcst["time"]),
        "time_hours": float(fcst["time"]) / 3600.0,
        "health": compute_health_metrics(fcst),
        "levels": {},
        "full_volume": {},
    }

    ref_health = compute_health_metrics(ref)
    summary["reference_health"] = ref_health

    for label, target_m in TARGET_LEVELS_M:
        k_fcst = nearest_level(fcst["z"], target_m)
        k_ref = nearest_level(ref["z"], target_m)
        level_info: Dict[str, object] = {
            "forecast_level_index": k_fcst,
            "forecast_level_m": float(fcst["z"][k_fcst]),
            "reference_level_index": k_ref,
            "reference_level_m": float(ref["z"][k_ref]),
            "fields": {},
        }
        for fcst_name, ref_name in FIELD_MAP:
            metrics = scalar_metrics(fcst[fcst_name][k_fcst], ref[fcst_name.upper()][k_ref])
            level_info["fields"][fcst_name] = metrics
        summary["levels"][label] = level_info

    for fcst_name, _ in FIELD_MAP:
        summary["full_volume"][fcst_name] = scalar_metrics(fcst[fcst_name], ref[fcst_name])

    band_metrics = compute_regional_band_metrics(fcst, ref, outer_band)
    if band_metrics:
        summary["regional_bands"] = band_metrics

    summary["flags"] = evaluate_flags(summary["health"], ref_health)
    return summary


def format_value(value: float, fmt: str) -> str:
    if math.isnan(value):
        return "nan"
    return format(value, fmt)


def print_summary(summary: Dict[str, object]) -> None:
    print(f"{summary['path']}  t={summary['time_hours']:.2f} h")

    health = summary["health"]
    print(
        "  health: "
        f"mean_u={health['mean_u']:+.2f}  "
        f"mean_v={health['mean_v']:+.2f}  "
        f"mean_w={health['mean_w']:+.4f}  "
        f"mean|w|={health['mean_abs_w']:.4f}  "
        f"max|w|={health['max_abs_w']:.2f}"
    )

    full_u = summary["full_volume"]["U"]
    full_v = summary["full_volume"]["V"]
    full_th = summary["full_volume"]["THETA"]
    full_qv = summary["full_volume"]["QV"]
    print(
        "  full:   "
        f"U rmse={full_u['rmse']:.2f}  "
        f"V rmse={full_v['rmse']:.2f}  "
        f"THETA rmse={full_th['rmse']:.2f}  "
        f"QV rmse={full_qv['rmse']:.5f}"
    )

    for level_name in ("sfc", "850ish", "500ish"):
        level = summary["levels"][level_name]
        u = level["fields"]["U"]
        v = level["fields"]["V"]
        th = level["fields"]["THETA"]
        qv = level["fields"]["QV"]
        print(
            f"  {level_name:6s} z={level['forecast_level_m']:.0f} m: "
            f"U rmse={u['rmse']:.2f} bias={u['bias']:+.2f} corr={format_value(u['corr'], '.2f')} | "
            f"V rmse={v['rmse']:.2f} bias={v['bias']:+.2f} corr={format_value(v['corr'], '.2f')} | "
            f"TH rmse={th['rmse']:.2f} | "
            f"QV rmse={qv['rmse']:.5f}"
        )

    regional_bands = summary.get("regional_bands")
    if regional_bands:
        ordered_band_names = sorted(
            regional_bands.keys(),
            key=lambda name: (0 if name.startswith("outer_") else 1, name),
        )
        for band_name in ordered_band_names:
            band = regional_bands[band_name]
            print(
                f"  {band_name:8s} "
                f"U rmse={band['u_rmse']:.2f}  "
                f"V rmse={band['v_rmse']:.2f}  "
                f"TH rmse={band['theta_rmse']:.2f}  "
                f"mean|w|={band['mean_abs_w']:.3f}  "
                f"qtot_d={band['qtot_rel_change_pct']:+.2f}%"
            )

    flags = summary["flags"]
    if flags:
        print("  flags:  " + "; ".join(flags))
    else:
        print("  flags:  none")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify GPU-WM forecast outputs against a reference state"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Forecast NetCDF files to verify (default: output/gpuwm_*.nc)",
    )
    parser.add_argument(
        "--reference",
        help=(
            "Reference state (.bin from init_from_gfs.py or .nc). "
            "Default: first forecast file."
        ),
    )
    parser.add_argument(
        "--json-out",
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--outer-band",
        type=int,
        default=20,
        help="Outer horizontal band width for regional drift diagnostics",
    )
    args = parser.parse_args()

    if args.files:
        forecast_files = args.files
    else:
        forecast_files = sorted(glob.glob("output/gpuwm_*.nc"))

    if not forecast_files:
        raise RuntimeError("No forecast NetCDF files found")

    if args.reference:
        ref = load_state(args.reference)
    else:
        ref = load_netcdf_state(forecast_files[0])

    results = []
    for path in forecast_files:
        fcst = load_netcdf_state(path)
        results.append(summarize_case(fcst, ref, outer_band=args.outer_band))

    print("=" * 88)
    print(f"GPU-WM verification  reference={ref['path']}")
    print("=" * 88)
    for result in results:
        print_summary(result)
        print("-" * 88)

    if args.json_out:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "reference": ref["path"],
                        "results": results,
                    },
                    f,
                    indent=2,
                )
            print(f"Wrote JSON summary: {args.json_out}")
        except OSError as exc:
            print(f"WARNING: failed to write JSON summary {args.json_out}: {exc}")


if __name__ == "__main__":
    main()
