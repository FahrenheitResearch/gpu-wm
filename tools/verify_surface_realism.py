#!/usr/bin/env python3
"""Score GPU-WM screen-level realism against HRRR surface fields.

This script is intentionally separate from the solver-centric verifier:
it compares screen diagnostics (T2/RH2/U10/V10) and legacy lowest-model-level
proxies against HRRR surface products, then aggregates the skill into a small
set of interpretable windows and terrain classes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from netCDF4 import Dataset

from init_from_gfs import bilinear_interp_field, build_grib_index, read_grib_field

EPSILON = 0.622


@dataclass(frozen=True)
class FieldDef:
    key: str
    units: str
    hrrr_loader: Callable[[dict], dict[str, np.ndarray]]
    screen_loader: Callable[[dict], dict[str, np.ndarray] | None]
    lml_loader: Callable[[dict], dict[str, np.ndarray] | None]
    rmse_label: str


@dataclass(frozen=True)
class WindowDef:
    key: str
    label: str
    start_h: float
    end_h: float


SUMMARY_WINDOWS = [
    WindowDef("startup_0_3h", "0-3 h", 0.0, 3.0),
    WindowDef("mature_3_12h", "3-12 h", 3.0, 12.0),
    WindowDef("late_12_24h", "12-24 h", 12.0, 24.0),
]

DETAIL_WINDOWS = [
    WindowDef("all_1_6h", "1-6 h", 1.0, 6.0),
    WindowDef("spinup_0_1h", "0-1 h", 0.0, 1.0),
    WindowDef("settling_1_3h", "1-3 h", 1.0, 3.0),
    WindowDef("persistent_3_6h", "3-6 h", 3.0, 6.0),
]

SOLAR_PHASES = ("night", "dawn_dusk", "day")
STRESS_PHASES = ("night", "dawn_dusk")
SUMMARY_MASKS = ("domain", "rest", "complex_high")
DETAIL_MASKS = ("domain", "plains", "moderate_relief", "steep_high")
HIGH_WIND_COMPLEX_THRESHOLD_MS = 12.0


def saturation_vapor_pressure_pa(temp_k: np.ndarray) -> np.ndarray:
    temp_c = temp_k - 273.15
    return 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))


def mixing_ratio_to_rh_pct(w: np.ndarray, p_pa: np.ndarray, temp_k: np.ndarray) -> np.ndarray:
    vapor_pressure = p_pa * w / np.maximum(EPSILON + w, 1.0e-12)
    sat = saturation_vapor_pressure_pa(temp_k)
    rh = 100.0 * vapor_pressure / np.maximum(sat, 1.0)
    return np.clip(rh, 0.0, 100.0)


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u * u + v * v)


def read_var_2d(ds: Dataset, *names: str) -> np.ndarray | None:
    for name in names:
        if name in ds.variables:
            return np.array(ds.variables[name][:], dtype=np.float64).squeeze()
    return None


def load_model_state(path: Path) -> dict:
    with Dataset(path, "r") as ds:
        state = {
            "path": str(path),
            "time_h": float(np.array(ds.variables["time"][:], dtype=np.float64).ravel()[0]) / 3600.0,
            "lat": np.array(ds.variables["lat"][:], dtype=np.float64).squeeze(),
            "lon": np.array(ds.variables["lon"][:], dtype=np.float64).squeeze(),
            "terrain": read_var_2d(ds, "TERRAIN"),
            "terrain_slope": read_var_2d(ds, "TERRAIN_SLOPE"),
            "t2": read_var_2d(ds, "T2"),
            "rh2": read_var_2d(ds, "RH2"),
            "q2": read_var_2d(ds, "Q2"),
            "psfc": read_var_2d(ds, "PSFC"),
            "u10": read_var_2d(ds, "U10"),
            "v10": read_var_2d(ds, "V10"),
            "t2_lml": read_var_2d(ds, "T2_LML"),
            "rh2_lml": read_var_2d(ds, "RH2_LML"),
            "u10_lml": read_var_2d(ds, "U10_LML"),
            "v10_lml": read_var_2d(ds, "V10_LML"),
        }
        if state["rh2"] is None and state["q2"] is not None and state["psfc"] is not None and state["t2"] is not None:
            state["rh2"] = mixing_ratio_to_rh_pct(state["q2"], state["psfc"], state["t2"])
        if state["rh2_lml"] is None and state["t2_lml"] is not None and state["q2"] is not None and state["psfc"] is not None:
            state["rh2_lml"] = mixing_ratio_to_rh_pct(state["q2"], state["psfc"], state["t2_lml"])
        state["screen_diag_revision"] = ds.getncattr("screen_diag_revision") if "screen_diag_revision" in ds.ncattrs() else None
    return state


def hrrr_interp(cache: dict, surface_grib: Path, model_lat: np.ndarray, model_lon: np.ndarray, short_name: str, level: int, type_of_level: str) -> np.ndarray:
    if short_name not in cache:
        field, grid = read_grib_field(str(surface_grib), short_name, level, type_of_level, cache["_index"])
        cache[short_name] = bilinear_interp_field(field, grid, model_lat, model_lon)
    return cache[short_name]


def load_hrrr_state(surface_grib: Path, model_lat: np.ndarray, model_lon: np.ndarray) -> dict:
    cache: dict[str, np.ndarray] = {"_index": build_grib_index(str(surface_grib))}
    t2 = hrrr_interp(cache, surface_grib, model_lat, model_lon, "2t", 2, "heightAboveGround")
    return {
        "surface_grib": str(surface_grib),
        "t2": t2,
        "rh2": np.clip(hrrr_interp(cache, surface_grib, model_lat, model_lon, "2r", 2, "heightAboveGround"), 0.0, 100.0),
        "u10": hrrr_interp(cache, surface_grib, model_lat, model_lon, "10u", 10, "heightAboveGround"),
        "v10": hrrr_interp(cache, surface_grib, model_lat, model_lon, "10v", 10, "heightAboveGround"),
        "valid_hour_utc": parse_hrrr_valid_hour_utc(surface_grib),
    }


def compute_slope(terrain: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if terrain is None:
        raise ValueError("terrain required for terrain classes")
    dy_m = 111_000.0 * max(abs(float(np.nanmean(np.diff(lat[:, 0])))), 1.0e-6)
    mean_lat = float(np.nanmean(lat))
    dx_m = 111_000.0 * math.cos(math.radians(mean_lat)) * max(abs(float(np.nanmean(np.diff(lon[0, :])))), 1.0e-6)
    gy, gx = np.gradient(terrain, dy_m, dx_m)
    return np.sqrt(gx * gx + gy * gy)


def build_terrain_masks(model: dict) -> dict[str, np.ndarray]:
    terrain = model["terrain"]
    if terrain is None:
        ones = np.ones_like(model["t2"], dtype=bool)
        return {"domain": ones}

    slope = model["terrain_slope"]
    if slope is None:
        slope = compute_slope(terrain, model["lat"], model["lon"])

    plains = (terrain < 1200.0) & (slope < 0.03)
    steep_high = (terrain >= 1800.0) | (slope >= 0.08)
    moderate_relief = ~(plains | steep_high)
    domain = np.isfinite(terrain)

    masks = {
        "domain": domain,
        "rest": domain & ~steep_high,
        "complex_high": steep_high & domain,
        "plains": plains & domain,
        "moderate_relief": moderate_relief & domain,
        "steep_high": steep_high & domain,
    }
    return {name: mask for name, mask in masks.items() if np.any(mask)}


def parse_hrrr_valid_hour_utc(surface_grib: Path) -> int | None:
    match = re.search(r"\.t(\d{2})z\..*f(\d{2})\.grib2$", surface_grib.name)
    if not match:
        return None
    cycle_hour = int(match.group(1))
    forecast_hour = int(match.group(2))
    return (cycle_hour + forecast_hour) % 24


def classify_solar_phase(valid_hour_utc: int | None, mean_lon_deg: float) -> str | None:
    if valid_hour_utc is None or math.isnan(mean_lon_deg):
        return None
    local_solar_hour = (valid_hour_utc + mean_lon_deg / 15.0) % 24.0
    if 5.0 <= local_solar_hour < 7.0 or 17.0 <= local_solar_hour < 19.0:
        return "dawn_dusk"
    if 7.0 <= local_solar_hour < 17.0:
        return "day"
    return "night"


def anomaly_corr(model_vals: np.ndarray, ref_vals: np.ndarray) -> float:
    if model_vals.size < 2:
        return float("nan")
    ma = model_vals - float(np.mean(model_vals))
    ra = ref_vals - float(np.mean(ref_vals))
    ms = float(np.std(ma))
    rs = float(np.std(ra))
    if ms < 1.0e-12 or rs < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(ma, ra)[0, 1])


def aggregate_scalar_metrics(model_vals: list[np.ndarray], ref_vals: list[np.ndarray]) -> dict[str, float]:
    model_flat = np.concatenate([arr.ravel() for arr in model_vals])
    ref_flat = np.concatenate([arr.ravel() for arr in ref_vals])
    diff = model_flat - ref_flat
    return {
        "n": int(model_flat.size),
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "anomaly_corr": anomaly_corr(model_flat, ref_flat),
        "spread_ratio": float(np.std(model_flat) / max(np.std(ref_flat), 1.0e-12)),
    }


def aggregate_wind_metrics(model_u: list[np.ndarray], model_v: list[np.ndarray], ref_u: list[np.ndarray], ref_v: list[np.ndarray]) -> dict[str, float]:
    mu = np.concatenate([arr.ravel() for arr in model_u])
    mv = np.concatenate([arr.ravel() for arr in model_v])
    ru = np.concatenate([arr.ravel() for arr in ref_u])
    rv = np.concatenate([arr.ravel() for arr in ref_v])
    mspd = wind_speed(mu, mv)
    rspd = wind_speed(ru, rv)
    vec_rmse = float(np.sqrt(np.mean((mu - ru) ** 2 + (mv - rv) ** 2)))
    vec_mae = float(np.mean(np.sqrt((mu - ru) ** 2 + (mv - rv) ** 2)))
    speed_bias = float(np.mean(mspd - rspd))
    speed_mae = float(np.mean(np.abs(mspd - rspd)))
    return {
        "n": int(mu.size),
        "bias": speed_bias,
        "speed_bias": speed_bias,
        "speed_mae": speed_mae,
        "mae": vec_mae,
        "vector_mae": vec_mae,
        "rmse": vec_rmse,
        "vector_rmse": vec_rmse,
        "anomaly_corr": anomaly_corr(mspd, rspd),
        "spread_ratio": float(np.std(mspd) / max(np.std(rspd), 1.0e-12)),
    }


FIELDS = [
    FieldDef(
        key="T2",
        units="K",
        hrrr_loader=lambda ref: {"value": ref["t2"]},
        screen_loader=lambda model: {"value": model["t2"]} if model["t2"] is not None else None,
        lml_loader=lambda model: {"value": model["t2_lml"]} if model["t2_lml"] is not None else None,
        rmse_label="rmse",
    ),
    FieldDef(
        key="RH2",
        units="%",
        hrrr_loader=lambda ref: {"value": ref["rh2"]},
        screen_loader=lambda model: {"value": model["rh2"]} if model["rh2"] is not None else None,
        lml_loader=lambda model: {"value": model["rh2_lml"]} if model["rh2_lml"] is not None else None,
        rmse_label="rmse",
    ),
    FieldDef(
        key="WIND10",
        units="m/s",
        hrrr_loader=lambda ref: {"u": ref["u10"], "v": ref["v10"]},
        screen_loader=lambda model: {"u": model["u10"], "v": model["v10"]} if model["u10"] is not None and model["v10"] is not None else None,
        lml_loader=lambda model: {"u": model["u10_lml"], "v": model["v10_lml"]} if model["u10_lml"] is not None and model["v10_lml"] is not None else None,
        rmse_label="vector_rmse",
    ),
]


def in_window(time_h: float, window: WindowDef) -> bool:
    if math.isclose(window.start_h, 0.0):
        return window.start_h <= time_h <= window.end_h + 1.0e-9
    return window.start_h < time_h <= window.end_h + 1.0e-9


def select_mask(sample: dict, mask_name: str, field_key: str, high_wind_threshold_ms: float | None = None) -> np.ndarray | None:
    if mask_name == "high_wind_complex":
        if field_key != "WIND10":
            return None
        base_mask = sample["terrain_masks"].get("complex_high")
        if base_mask is None or not np.any(base_mask):
            return None
        hrrr_speed = wind_speed(sample["hrrr"]["u10"], sample["hrrr"]["v10"])
        mask = base_mask & np.isfinite(hrrr_speed)
        if high_wind_threshold_ms is not None:
            mask &= hrrr_speed >= high_wind_threshold_ms
        return mask if np.any(mask) else None
    mask = sample["terrain_masks"].get(mask_name)
    if mask is None or not np.any(mask):
        return None
    return mask


def score_field_rows(
    samples: list[dict],
    mask_name: str,
    fields: tuple[str, ...] | None = None,
    high_wind_threshold_ms: float | None = None,
) -> dict[str, object]:
    field_rows: dict[str, object] = {}
    field_allow = set(fields) if fields is not None else None
    for field in FIELDS:
        if field_allow is not None and field.key not in field_allow:
            continue
        screen_model: list[np.ndarray] = []
        lml_model: list[np.ndarray] = []
        ref_values: list[np.ndarray] = []
        screen_u: list[np.ndarray] = []
        screen_v: list[np.ndarray] = []
        lml_u: list[np.ndarray] = []
        lml_v: list[np.ndarray] = []
        ref_u: list[np.ndarray] = []
        ref_v: list[np.ndarray] = []

        for sample in samples:
            mask = select_mask(sample, mask_name, field.key, high_wind_threshold_ms=high_wind_threshold_ms)
            if mask is None:
                continue
            ref_pack = field.hrrr_loader(sample["hrrr"])
            screen_pack = field.screen_loader(sample["model"])
            lml_pack = field.lml_loader(sample["model"])
            if screen_pack is None:
                continue
            if field.key == "WIND10":
                screen_u.append(screen_pack["u"][mask])
                screen_v.append(screen_pack["v"][mask])
                ref_u.append(ref_pack["u"][mask])
                ref_v.append(ref_pack["v"][mask])
                if lml_pack is not None:
                    lml_u.append(lml_pack["u"][mask])
                    lml_v.append(lml_pack["v"][mask])
            else:
                screen_model.append(screen_pack["value"][mask])
                ref_values.append(ref_pack["value"][mask])
                if lml_pack is not None:
                    lml_model.append(lml_pack["value"][mask])

        if field.key == "WIND10":
            if not screen_u:
                continue
            screen_metrics = aggregate_wind_metrics(screen_u, screen_v, ref_u, ref_v)
            row: dict[str, object] = {"screen": screen_metrics}
            if lml_u:
                lml_metrics = aggregate_wind_metrics(lml_u, lml_v, ref_u, ref_v)
                row["lml"] = lml_metrics
                row["delta"] = {
                    "bias": float(screen_metrics["speed_bias"] - lml_metrics["speed_bias"]),
                    "vector_rmse": float(screen_metrics["vector_rmse"] - lml_metrics["vector_rmse"]),
                    "vector_mae": float(screen_metrics["vector_mae"] - lml_metrics["vector_mae"]),
                }
                row["screen_gain"] = {
                    "vector_rmse": float(lml_metrics["vector_rmse"] - screen_metrics["vector_rmse"]),
                    "vector_mae": float(lml_metrics["vector_mae"] - screen_metrics["vector_mae"]),
                }
            field_rows[field.key] = row
        else:
            if not screen_model:
                continue
            screen_metrics = aggregate_scalar_metrics(screen_model, ref_values)
            row = {"screen": screen_metrics}
            if lml_model:
                lml_metrics = aggregate_scalar_metrics(lml_model, ref_values)
                row["lml"] = lml_metrics
                row["delta"] = {
                    "bias": float(screen_metrics["bias"] - lml_metrics["bias"]),
                    "mae": float(screen_metrics["mae"] - lml_metrics["mae"]),
                    "rmse": float(screen_metrics["rmse"] - lml_metrics["rmse"]),
                }
                row["screen_gain"] = {
                    "mae": float(lml_metrics["mae"] - screen_metrics["mae"]),
                    "rmse": float(lml_metrics["rmse"] - screen_metrics["rmse"]),
                }
            field_rows[field.key] = row
    return field_rows


def pair_surface_grib(model_path: Path, time_h: float, args: argparse.Namespace) -> Path:
    if args.surface_grib:
        raise RuntimeError("Explicit --surface-grib pairing not implemented for batch mode")
    hour = int(round(time_h))
    surface_path = Path(args.surface_grib_dir) / args.surface_grib_pattern.format(hour=hour)
    return surface_path


def build_samples(args: argparse.Namespace) -> list[dict]:
    samples = []
    skipped: list[dict[str, object]] = []
    for model_path_str in args.files:
        model_path = Path(model_path_str)
        model = load_model_state(model_path)
        surface_grib = pair_surface_grib(model_path, model["time_h"], args)
        if not surface_grib.exists():
            skipped.append({"model": str(model_path), "time_h": model["time_h"], "missing_surface_grib": str(surface_grib)})
            continue
        hrrr = load_hrrr_state(surface_grib, model["lat"], model["lon"])
        samples.append(
            {
                "model": model,
                "hrrr": hrrr,
                "terrain_masks": build_terrain_masks(model),
                "solar_phase": classify_solar_phase(hrrr["valid_hour_utc"], float(np.nanmean(model["lon"]))),
            }
        )
    if not samples:
        missing_list = "\n".join(item["missing_surface_grib"] for item in skipped)
        raise FileNotFoundError(f"No matched HRRR surface files were available.\n{missing_list}")
    return samples, skipped


def score_samples(samples: list[dict]) -> dict:
    results: dict[str, object] = {"summary": {}, "windows": {}, "solar_phase": {}, "stress_tests": {}, "metadata": {}}
    if not samples:
        return results

    results["metadata"] = {
        "screen_diag_revision": samples[-1]["model"].get("screen_diag_revision"),
        "terrain_classes": ["domain", "rest", "complex_high", "plains", "moderate_relief", "steep_high"],
        "summary_windows": [{"key": w.key, "label": w.label} for w in SUMMARY_WINDOWS],
        "detail_windows": [{"key": w.key, "label": w.label} for w in DETAIL_WINDOWS],
        "solar_phases": list(SOLAR_PHASES),
        "high_wind_complex_threshold_ms": HIGH_WIND_COMPLEX_THRESHOLD_MS,
    }

    for window in SUMMARY_WINDOWS:
        window_samples = [sample for sample in samples if in_window(sample["model"]["time_h"], window)]
        if not window_samples:
            continue

        window_result: dict[str, object] = {}
        for mask_name in SUMMARY_MASKS:
            field_rows = score_field_rows(window_samples, mask_name)
            if field_rows:
                window_result[mask_name] = field_rows

        if window_result:
            results["summary"][window.key] = window_result

    for window in DETAIL_WINDOWS:
        window_samples = [sample for sample in samples if in_window(sample["model"]["time_h"], window)]
        if not window_samples:
            continue

        window_result: dict[str, object] = {}
        for mask_name in DETAIL_MASKS:
            field_rows = score_field_rows(window_samples, mask_name)
            if field_rows:
                window_result[mask_name] = field_rows

        if window_result:
            results["windows"][window.key] = window_result

    for phase in SOLAR_PHASES:
        phase_samples = [sample for sample in samples if sample.get("solar_phase") == phase]
        if not phase_samples:
            continue
        phase_result: dict[str, object] = {}
        for mask_name in DETAIL_MASKS:
            field_rows = score_field_rows(phase_samples, mask_name)
            if field_rows:
                phase_result[mask_name] = field_rows
        if phase_result:
            results["solar_phase"][phase] = phase_result

    for phase in STRESS_PHASES:
        phase_samples = [sample for sample in samples if sample.get("solar_phase") == phase]
        if not phase_samples:
            continue
        field_rows = score_field_rows(phase_samples, "complex_high", fields=("T2", "RH2"))
        if field_rows:
            results["stress_tests"][f"{phase}_complex_high"] = {
                "label": f"{phase} / complex terrain",
                "rows": field_rows,
            }

    mature_samples = [sample for sample in samples if in_window(sample["model"]["time_h"], SUMMARY_WINDOWS[1])]
    if mature_samples:
        field_rows = score_field_rows(
            mature_samples,
            "high_wind_complex",
            fields=("WIND10",),
            high_wind_threshold_ms=HIGH_WIND_COMPLEX_THRESHOLD_MS,
        )
        if field_rows:
            results["stress_tests"]["high_wind_complex"] = {
                "label": f"high-wind / complex terrain ({HIGH_WIND_COMPLEX_THRESHOLD_MS:.0f}+ m/s HRRR)",
                "rows": field_rows,
            }
    return results


def format_metric(value: float, fmt: str) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return format(value, fmt)


def build_markdown(scorecard: dict) -> str:
    lines: list[str] = []
    lines.append("# Surface Realism Dashboard")
    lines.append("")
    lines.append(f"- screen_diag_revision: `{scorecard.get('metadata', {}).get('screen_diag_revision') or 'unknown'}`")
    lines.append("")
    lines.append("## Front Page")
    lines.append("")
    lines.append("| Window | Mask | T2 ME / MAE | RH2 ME / MAE | W10 vecRMSE / speed bias | Screen gain vs LML (T2 / RH2 / W10) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for window in SUMMARY_WINDOWS:
        window_data = scorecard.get("summary", {}).get(window.key)
        if not window_data:
            continue
        for mask_name in SUMMARY_MASKS:
            mask_data = window_data.get(mask_name)
            if not mask_data:
                continue
            t2 = mask_data.get("T2", {})
            rh2 = mask_data.get("RH2", {})
            w10 = mask_data.get("WIND10", {})
            t2_screen = t2.get("screen", {})
            rh2_screen = rh2.get("screen", {})
            w10_screen = w10.get("screen", {})
            t2_gain = t2.get("screen_gain", {})
            rh2_gain = rh2.get("screen_gain", {})
            w10_gain = w10.get("screen_gain", {})
            lines.append(
                "| "
                f"{window.label} | {mask_name} | "
                f"{format_metric(t2_screen.get('bias', float('nan')), '.2f')} / {format_metric(t2_screen.get('mae', float('nan')), '.2f')} | "
                f"{format_metric(rh2_screen.get('bias', float('nan')), '.2f')} / {format_metric(rh2_screen.get('mae', float('nan')), '.2f')} | "
                f"{format_metric(w10_screen.get('vector_rmse', float('nan')), '.2f')} / {format_metric(w10_screen.get('speed_bias', float('nan')), '.2f')} | "
                f"{format_metric(t2_gain.get('mae', float('nan')), '.2f')} / {format_metric(rh2_gain.get('mae', float('nan')), '.2f')} / {format_metric(w10_gain.get('vector_rmse', float('nan')), '.2f')} |"
            )
    lines.append("")
    if scorecard.get("stress_tests"):
        lines.append("## Stress Tests")
        lines.append("")
        lines.append("| Case | T2 ME / MAE | RH2 ME / MAE | W10 vecRMSE / speed bias | Screen gain vs LML |")
        lines.append("|---|---:|---:|---:|---:|")
        for key in ("night_complex_high", "dawn_dusk_complex_high", "high_wind_complex"):
            item = scorecard["stress_tests"].get(key)
            if not item:
                continue
            rows = item["rows"]
            t2 = rows.get("T2", {})
            rh2 = rows.get("RH2", {})
            w10 = rows.get("WIND10", {})
            lines.append(
                "| "
                f"{item['label']} | "
                f"{format_metric(t2.get('screen', {}).get('bias', float('nan')), '.2f')} / {format_metric(t2.get('screen', {}).get('mae', float('nan')), '.2f')} | "
                f"{format_metric(rh2.get('screen', {}).get('bias', float('nan')), '.2f')} / {format_metric(rh2.get('screen', {}).get('mae', float('nan')), '.2f')} | "
                f"{format_metric(w10.get('screen', {}).get('vector_rmse', float('nan')), '.2f')} / {format_metric(w10.get('screen', {}).get('speed_bias', float('nan')), '.2f')} | "
                f"{format_metric(t2.get('screen_gain', {}).get('mae', float('nan')), '.2f')} / {format_metric(rh2.get('screen_gain', {}).get('mae', float('nan')), '.2f')} / {format_metric(w10.get('screen_gain', {}).get('vector_rmse', float('nan')), '.2f')} |"
            )
        lines.append("")
    lines.append("## Detailed Debug Windows")
    lines.append("")
    for window in DETAIL_WINDOWS:
        window_data = scorecard.get("windows", {}).get(window.key)
        if not window_data:
            continue
        lines.append(f"## {window.label}")
        lines.append("")
        lines.append("| Field | Mask | Bias | MAE | RMSE | Anom Corr | Spread | Bias_LML | MAE_LML | RMSE_LML | ScreenGain |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for mask_name in DETAIL_MASKS:
            mask_data = window_data.get(mask_name)
            if not mask_data:
                continue
            for field in ("T2", "RH2", "WIND10"):
                row = mask_data.get(field)
                if not row:
                    continue
                screen = row["screen"]
                lml = row.get("lml", {})
                delta = row.get("delta", {})
                lines.append(
                    "| "
                    f"{field} | {mask_name} | "
                    f"{format_metric(screen['bias'], '.2f')} | "
                    f"{format_metric(screen.get('mae', float('nan')), '.2f')} | "
                    f"{format_metric(screen['rmse'], '.2f')} | "
                    f"{format_metric(screen['anomaly_corr'], '.2f')} | "
                    f"{format_metric(screen['spread_ratio'], '.2f')} | "
                    f"{format_metric(lml.get('bias', float('nan')), '.2f')} | "
                    f"{format_metric(lml.get('mae', float('nan')), '.2f')} | "
                    f"{format_metric(lml.get('rmse', float('nan')), '.2f')} | "
                    f"{format_metric(row.get('screen_gain', {}).get('mae', row.get('screen_gain', {}).get('vector_rmse', float('nan'))), '.2f')} |"
                )
        lines.append("")
    for phase in SOLAR_PHASES:
        phase_data = scorecard.get("solar_phase", {}).get(phase)
        if not phase_data:
            continue
        lines.append(f"## solar phase: {phase}")
        lines.append("")
        lines.append("| Field | Mask | Bias | MAE | RMSE | Anom Corr | Spread | Bias_LML | MAE_LML | RMSE_LML | ScreenGain |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for mask_name in DETAIL_MASKS:
            mask_data = phase_data.get(mask_name)
            if not mask_data:
                continue
            for field in ("T2", "RH2", "WIND10"):
                row = mask_data.get(field)
                if not row:
                    continue
                screen = row["screen"]
                lml = row.get("lml", {})
                delta = row.get("delta", {})
                lines.append(
                    "| "
                    f"{field} | {mask_name} | "
                    f"{format_metric(screen['bias'], '.2f')} | "
                    f"{format_metric(screen.get('mae', float('nan')), '.2f')} | "
                    f"{format_metric(screen['rmse'], '.2f')} | "
                    f"{format_metric(screen['anomaly_corr'], '.2f')} | "
                    f"{format_metric(screen['spread_ratio'], '.2f')} | "
                    f"{format_metric(lml.get('bias', float('nan')), '.2f')} | "
                    f"{format_metric(lml.get('mae', float('nan')), '.2f')} | "
                    f"{format_metric(lml.get('rmse', float('nan')), '.2f')} | "
                    f"{format_metric(row.get('screen_gain', {}).get('mae', row.get('screen_gain', {}).get('vector_rmse', float('nan'))), '.2f')} |"
                )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="GPU-WM NetCDF outputs")
    parser.add_argument("--surface-grib-dir", required=True, help="Directory containing HRRR wrfsfc files")
    parser.add_argument(
        "--surface-grib-pattern",
        default="hrrr.t19z.wrfsfcf{hour:02d}.grib2",
        help="Filename pattern inside --surface-grib-dir using {hour:02d}",
    )
    parser.add_argument("--surface-grib", nargs="*", help="Explicit HRRR wrfsfc files (not yet supported in batch mode)")
    parser.add_argument("--json-out", help="Write JSON scorecard")
    parser.add_argument("--markdown-out", help="Write markdown summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples, skipped = build_samples(args)
    scorecard = score_samples(samples)
    scorecard["metadata"]["skipped_missing_surface_grib"] = skipped

    markdown = build_markdown(scorecard)
    print(markdown)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {out}")

    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote markdown: {out}")


if __name__ == "__main__":
    main()
