#!/usr/bin/env python3
"""Render GPU-WM vs HRRR comparison panels on the model grid."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from netCDF4 import Dataset

from init_from_gfs import build_grib_index, read_grib_field, bilinear_interp_field
from plot_weather import nws_reflectivity_cmap, temperature_cmap

EPSILON = 0.622


def load_model_state(path: str) -> dict:
    ds = Dataset(path, "r")
    state = {
        "time": float(np.array(ds.variables["time"][:]).ravel()[0]),
        "lat": np.array(ds.variables["lat"][:], dtype=np.float64),
        "lon": np.array(ds.variables["lon"][:], dtype=np.float64),
        "t2": np.array(ds.variables["T2"][:], dtype=np.float64).squeeze(),
        "q2": np.array(ds.variables["Q2"][:], dtype=np.float64).squeeze(),
        "psfc": np.array(ds.variables["PSFC"][:], dtype=np.float64).squeeze(),
        "u10": np.array(ds.variables["U10"][:], dtype=np.float64).squeeze(),
        "v10": np.array(ds.variables["V10"][:], dtype=np.float64).squeeze(),
        "qc": np.array(ds.variables["QC"][:], dtype=np.float64),
        "qr": np.array(ds.variables["QR"][:], dtype=np.float64),
    }
    ds.close()
    return state


def saturation_vapor_pressure_pa(temp_k: np.ndarray) -> np.ndarray:
    temp_c = temp_k - 273.15
    return 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))


def mixing_ratio_to_rh_pct(w: np.ndarray, p_pa: np.ndarray, temp_k: np.ndarray) -> np.ndarray:
    vapor_pressure = p_pa * w / np.maximum(EPSILON + w, 1.0e-12)
    sat = saturation_vapor_pressure_pa(temp_k)
    rh = 100.0 * vapor_pressure / np.maximum(sat, 1.0)
    return np.clip(rh, 0.0, 100.0)


def model_reflectivity_dbz(qc: np.ndarray, qr: np.ndarray) -> np.ndarray:
    qr_col = np.max(qr, axis=0) * 1000.0
    qc_col = np.max(qc, axis=0) * 1000.0
    return np.where(
        qr_col > 0.01,
        10.0 * np.log10(300.0 * qr_col ** 1.4 + 0.1),
        np.where(qc_col > 0.01, 10.0 * np.log10(50.0 * qc_col + 0.1), -10.0),
    )


def load_hrrr_fields(surface_grib: str, model_lat: np.ndarray, model_lon: np.ndarray) -> dict:
    cache = build_grib_index(surface_grib)

    def interp(short_name: str, level: int, type_of_level: str) -> np.ndarray:
        field, grid = read_grib_field(surface_grib, short_name, level, type_of_level, cache)
        return bilinear_interp_field(field, grid, model_lat, model_lon)

    return {
        "t2": interp("2t", 2, "heightAboveGround"),
        "rh2": interp("2r", 2, "heightAboveGround"),
        "u10": interp("10u", 10, "heightAboveGround"),
        "v10": interp("10v", 10, "heightAboveGround"),
        "refc": interp("refc", 0, "atmosphere"),
    }


def plot_field_row(axs, lon, lat, model_field, ref_field, title, units,
                   cmap, diff_cmap, vmin=None, vmax=None, diff_abs=None,
                   u_model=None, v_model=None, u_ref=None, v_ref=None) -> None:
    if vmin is None:
        vmin = float(np.nanpercentile(np.concatenate([model_field.ravel(), ref_field.ravel()]), 2))
    if vmax is None:
        vmax = float(np.nanpercentile(np.concatenate([model_field.ravel(), ref_field.ravel()]), 98))
    diff = model_field - ref_field
    if diff_abs is None:
        diff_abs = float(np.nanpercentile(np.abs(diff), 98))
    diff_abs = max(diff_abs, 1.0e-6)

    c0 = axs[0].pcolormesh(lon, lat, model_field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    c1 = axs[1].pcolormesh(lon, lat, ref_field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    c2 = axs[2].pcolormesh(lon, lat, diff, cmap=diff_cmap, vmin=-diff_abs, vmax=diff_abs, shading="auto")

    if u_model is not None and v_model is not None and u_ref is not None and v_ref is not None:
        ny, nx = model_field.shape
        skip = max(1, min(nx, ny) // 22)
        axs[0].barbs(lon[::skip, ::skip], lat[::skip, ::skip],
                     u_model[::skip, ::skip] * 1.944, v_model[::skip, ::skip] * 1.944,
                     length=4.5, linewidth=0.35, color="black")
        axs[1].barbs(lon[::skip, ::skip], lat[::skip, ::skip],
                     u_ref[::skip, ::skip] * 1.944, v_ref[::skip, ::skip] * 1.944,
                     length=4.5, linewidth=0.35, color="black")

    axs[0].set_title(f"GPU-WM {title}")
    axs[1].set_title(f"HRRR {title}")
    axs[2].set_title(f"GPU-WM - HRRR {title}")

    for ax in axs:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.colorbar(c0, ax=axs[:2], shrink=0.84, pad=0.02, label=units)
    plt.colorbar(c2, ax=axs[2], shrink=0.84, pad=0.02, label=units)


def make_figure(model_nc: str, surface_grib: str, output_path: str) -> str:
    model = load_model_state(model_nc)
    hrrr = load_hrrr_fields(surface_grib, model["lat"], model["lon"])

    model_t2_c = model["t2"] - 273.15
    hrrr_t2_c = hrrr["t2"] - 273.15
    model_rh2 = mixing_ratio_to_rh_pct(model["q2"], model["psfc"], model["t2"])
    hrrr_rh2 = np.clip(hrrr["rh2"], 0.0, 100.0)
    model_wspd = np.sqrt(model["u10"] ** 2 + model["v10"] ** 2)
    hrrr_wspd = np.sqrt(hrrr["u10"] ** 2 + hrrr["v10"] ** 2)
    model_refc = model_reflectivity_dbz(model["qc"], model["qr"])
    hrrr_refc = hrrr["refc"]

    fig, axes = plt.subplots(4, 3, figsize=(18, 20), constrained_layout=True)

    plot_field_row(
        axes[0], model["lon"], model["lat"],
        model_t2_c, hrrr_t2_c,
        "2 m Temperature", "C",
        temperature_cmap(), "RdBu_r",
        diff_abs=8.0,
    )
    plot_field_row(
        axes[1], model["lon"], model["lat"],
        model_rh2, hrrr_rh2,
        "2 m Relative Humidity", "%",
        "YlGnBu", "BrBG",
        vmin=0.0, vmax=100.0, diff_abs=30.0,
    )
    plot_field_row(
        axes[2], model["lon"], model["lat"],
        model_wspd, hrrr_wspd,
        "10 m Wind Speed", "m/s",
        "viridis", "PuOr",
        vmin=0.0,
        vmax=max(10.0, float(np.nanpercentile(np.concatenate([model_wspd.ravel(), hrrr_wspd.ravel()]), 99))),
        diff_abs=12.0,
        u_model=model["u10"], v_model=model["v10"], u_ref=hrrr["u10"], v_ref=hrrr["v10"],
    )
    plot_field_row(
        axes[3], model["lon"], model["lat"],
        model_refc, hrrr_refc,
        "Composite Reflectivity", "dBZ",
        nws_reflectivity_cmap(), "RdBu_r",
        vmin=-10.0, vmax=75.0, diff_abs=25.0,
    )

    valid_hour = model["time"] / 3600.0
    fig.suptitle(
        f"GPU-WM vs HRRR comparison  t={valid_hour:.1f} h\n"
        f"model={Path(model_nc).name}  hrrr={Path(surface_grib).name}",
        fontsize=16,
        fontweight="bold",
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GPU-WM vs HRRR surface comparison panels")
    parser.add_argument("model_nc", help="GPU-WM NetCDF output file")
    parser.add_argument("--surface-grib", required=True, help="HRRR wrfsfc GRIB2 file at the same valid time")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    out = make_figure(args.model_nc, args.surface_grib, args.output)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
