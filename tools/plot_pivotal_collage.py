#!/usr/bin/env python3
"""Generate a multi-panel forecast collage for a single GPU-WM NetCDF file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colors import LightSource

try:
    from netCDF4 import Dataset
except ImportError:
    print("ERROR: netCDF4 required. Install with: pip install netCDF4")
    raise SystemExit(1)

from plot_weather import nws_reflectivity_cmap, temperature_cmap

P0 = 100000.0
KAPPA = 0.286


def load_model_nc(path: str) -> dict:
    ds = Dataset(path, "r")

    def read_first_available(*names: str) -> np.ndarray:
        for name in names:
            if name in ds.variables:
                return np.array(ds.variables[name][:], dtype=np.float64)
        raise KeyError(f"Missing variables {names} in {path}")

    data = {
        "path": path,
        "nx": len(ds.dimensions["x"]),
        "ny": len(ds.dimensions["y"]),
        "nz": len(ds.dimensions["z"]),
        "time": float(np.array(ds.variables["time"][:]).ravel()[0]),
        "x": np.array(ds.variables["x"][:], dtype=np.float64),
        "y": np.array(ds.variables["y"][:], dtype=np.float64),
        "z": np.array(ds.variables["z"][:], dtype=np.float64),
        "u": read_first_available("U_MASS", "U"),
        "v": read_first_available("V_MASS", "V"),
        "w": read_first_available("W_MASS", "W"),
        "theta": np.array(ds.variables["THETA"][:], dtype=np.float64),
        "qv": np.array(ds.variables["QV"][:], dtype=np.float64),
        "qc": np.array(ds.variables["QC"][:], dtype=np.float64),
        "qr": np.array(ds.variables["QR"][:], dtype=np.float64),
    }
    if "lat" in ds.variables:
        data["lat"] = np.array(ds.variables["lat"][:], dtype=np.float64)
        data["lon"] = np.array(ds.variables["lon"][:], dtype=np.float64)
    if "TERRAIN" in ds.variables:
        data["terrain"] = np.array(ds.variables["TERRAIN"][:], dtype=np.float64)
    ds.close()
    return data


def nearest_level(z: np.ndarray, target_m: float) -> int:
    return int(np.argmin(np.abs(z - target_m)))


def format_flags(metrics: dict | None) -> str:
    if not metrics:
        return ""
    flags = metrics.get("flags") or []
    if not flags:
        return "flags: none"
    return "flags: " + "; ".join(flags)


def get_coords(data: dict) -> tuple[np.ndarray, np.ndarray, str, str]:
    if "lat" in data and "lon" in data:
        return data["lon"], data["lat"], "Longitude", "Latitude"
    xcoord = np.broadcast_to(data["x"][np.newaxis, :], (data["ny"], data["nx"]))
    ycoord = np.broadcast_to(data["y"][:, np.newaxis], (data["ny"], data["nx"]))
    return xcoord, ycoord, "x (km)", "y (km)"


def pivotal_terrain_cmap() -> LinearSegmentedColormap:
    colors = [
        (0.00, (0.00, 0.25, 0.50)),
        (0.01, (0.00, 0.40, 0.20)),
        (0.05, (0.13, 0.55, 0.13)),
        (0.10, (0.33, 0.65, 0.20)),
        (0.15, (0.56, 0.74, 0.22)),
        (0.25, (0.76, 0.70, 0.30)),
        (0.35, (0.65, 0.50, 0.24)),
        (0.50, (0.55, 0.40, 0.30)),
        (0.65, (0.60, 0.55, 0.50)),
        (0.80, (0.75, 0.72, 0.70)),
        (0.90, (0.88, 0.87, 0.86)),
        (1.00, (1.00, 1.00, 1.00)),
    ]
    return LinearSegmentedColormap.from_list("terrain_topo", colors, N=512)


def plot_collage(ncfile: str, output_dir: str, metrics: dict | None = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    data = load_model_nc(ncfile)

    nx = data["nx"]
    ny = data["ny"]
    z = data["z"]
    t = data["time"]
    hours = t / 3600.0
    mins = t / 60.0

    xcoord, ycoord, xlabel, ylabel = get_coords(data)
    terrain = data.get("terrain", np.zeros((ny, nx), dtype=np.float64))
    terrain_max = max(float(np.max(terrain)), 1.0)

    k_sfc = 1
    k_850 = nearest_level(z, 1500.0)
    k_500 = nearest_level(z, 5500.0)
    k_mid = nearest_level(z, 3000.0)

    u_sfc = data["u"][k_sfc]
    v_sfc = data["v"][k_sfc]
    wspd_sfc = np.sqrt(u_sfc ** 2 + v_sfc ** 2)

    z_sfc = z[k_sfc]
    p_sfc = P0 * np.exp(-z_sfc / 8500.0)
    exner = (p_sfc / P0) ** KAPPA
    t_sfc_c = data["theta"][k_sfc] * exner - 273.15

    qr_col = np.max(data["qr"], axis=0) * 1000.0
    qc_col = np.max(data["qc"], axis=0) * 1000.0
    dbz = np.where(
        qr_col > 0.01,
        10.0 * np.log10(300.0 * qr_col ** 1.4 + 0.1),
        np.where(qc_col > 0.01, 10.0 * np.log10(50.0 * qc_col + 0.1), -999.0),
    )

    cloud_top_t = np.full((ny, nx), np.nan, dtype=np.float64)
    for k in range(data["nz"] - 1, -1, -1):
        cloud = (data["qc"][k] + data["qr"][k]) > 1.0e-5
        p_k = P0 * np.exp(-z[k] / 8500.0)
        t_k = data["theta"][k] * (p_k / P0) ** KAPPA - 273.15
        cloud_top_t = np.where(cloud & np.isnan(cloud_top_t), t_k, cloud_top_t)
    display_t = np.where(np.isnan(cloud_top_t), t_sfc_c, cloud_top_t)

    qv_sfc = data["qv"][k_sfc] * 1000.0
    theta_sfc = data["theta"][k_sfc]
    w_mid = data["w"][k_mid]
    wspd_500 = np.sqrt(data["u"][k_500] ** 2 + data["v"][k_500] ** 2)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    ls = LightSource(azdeg=315, altdeg=35)

    ax = axes[0, 0]
    hill = ls.hillshade(terrain, vert_exag=3.0, dx=1, dy=1, fraction=1.2)
    ax.pcolormesh(xcoord, ycoord, hill, cmap="gray", shading="auto", alpha=0.30, zorder=1)
    temp_vmin = max(-30.0, float(np.percentile(t_sfc_c, 2)))
    temp_vmax = min(45.0, float(np.percentile(t_sfc_c, 98)))
    c = ax.pcolormesh(xcoord, ycoord, t_sfc_c, cmap=temperature_cmap(),
                      vmin=temp_vmin, vmax=temp_vmax, shading="auto", alpha=0.90, zorder=2)
    skip = max(1, nx // 25)
    ax.barbs(xcoord[::skip, ::skip], ycoord[::skip, ::skip],
             u_sfc[::skip, ::skip] * 1.944, v_sfc[::skip, ::skip] * 1.944,
             length=5, linewidth=0.4, color="black", zorder=3)
    plt.colorbar(c, ax=ax, shrink=0.82, pad=0.02, label="C")
    ax.set_title("Surface Temperature + Winds")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax = axes[0, 1]
    cmap_r = nws_reflectivity_cmap()
    cmap_r.set_under("black")
    c = ax.pcolormesh(xcoord, ycoord, dbz, cmap=cmap_r,
                      vmin=-10, vmax=75, shading="auto")
    plt.colorbar(c, ax=ax, shrink=0.82, pad=0.02, label="dBZ")
    ax.set_title("Composite Reflectivity")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_facecolor("black")

    ax = axes[1, 0]
    c = ax.pcolormesh(xcoord, ycoord, display_t, cmap=plt.cm.gray_r,
                      vmin=-70, vmax=30, shading="auto")
    plt.colorbar(c, ax=ax, shrink=0.82, pad=0.02, label="C")
    ax.set_title("IR Satellite Proxy")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax = axes[1, 1]
    c = ax.pcolormesh(xcoord, ycoord, wspd_500, cmap="hot_r",
                      vmin=0, vmax=max(30.0, float(np.percentile(wspd_500, 99))), shading="auto")
    skip500 = max(1, nx // 22)
    ax.barbs(xcoord[::skip500, ::skip500], ycoord[::skip500, ::skip500],
             data["u"][k_500, ::skip500, ::skip500] * 1.944,
             data["v"][k_500, ::skip500, ::skip500] * 1.944,
             length=5, linewidth=0.35, color="white")
    plt.colorbar(c, ax=ax, shrink=0.82, pad=0.02, label="m/s")
    ax.set_title(f"500 mb Wind Speed  z~{z[k_500]/1000:.1f} km")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax = axes[2, 0]
    c = ax.pcolormesh(xcoord, ycoord, qv_sfc, cmap="YlGnBu",
                      vmin=0, vmax=max(5.0, float(np.percentile(qv_sfc, 98))), shading="auto")
    levels = np.arange(np.floor(np.min(theta_sfc)), np.ceil(np.max(theta_sfc)), 2.0)
    if len(levels) > 2:
        ax.contour(xcoord, ycoord, theta_sfc, levels=levels,
                   colors="red", linewidths=0.45, alpha=0.7)
    plt.colorbar(c, ax=ax, shrink=0.82, pad=0.02, label="g/kg")
    ax.set_title("Surface Moisture + Theta Contours")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax = axes[2, 1]
    rgb = ls.shade(terrain, cmap=pivotal_terrain_cmap(), vert_exag=2.0,
                   blend_mode="soft", vmin=0, vmax=terrain_max)
    ax.imshow(rgb, origin="lower", aspect="auto",
              extent=[xcoord.min(), xcoord.max(), ycoord.min(), ycoord.max()], zorder=1)
    c = ax.pcolormesh(xcoord, ycoord, w_mid, cmap="RdBu_r",
                      vmin=-max(1.0, float(np.percentile(np.abs(w_mid), 99))),
                      vmax=max(1.0, float(np.percentile(np.abs(w_mid), 99))),
                      shading="auto", alpha=0.60, zorder=2)
    plt.colorbar(c, ax=ax, shrink=0.82, pad=0.02, label="m/s")
    ax.set_title(f"Mid-Level Vertical Velocity  z~{z[k_mid]/1000:.1f} km")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    base = Path(ncfile).stem
    summary_bits = [
        f"lead={hours:.2f}h ({mins:.0f} min)",
    ]
    if metrics:
        health = metrics.get("health", {})
        full = metrics.get("full_volume", {})
        if health:
            summary_bits.append(
                f"mean_w={health.get('mean_w', float('nan')):+.3f} "
                f"mean|w|={health.get('mean_abs_w', float('nan')):.3f} "
                f"max|w|={health.get('max_abs_w', float('nan')):.2f}"
            )
        if full:
            summary_bits.append(
                f"rmse U/V/TH/QV="
                f"{full.get('U', {}).get('rmse', float('nan')):.2f}/"
                f"{full.get('V', {}).get('rmse', float('nan')):.2f}/"
                f"{full.get('THETA', {}).get('rmse', float('nan')):.2f}/"
                f"{full.get('QV', {}).get('rmse', float('nan')):.5f}"
            )
    fig.suptitle("GPU-WM Forecast Collage\n" + " | ".join(summary_bits),
                 fontsize=16, fontweight="bold", y=0.98)
    flag_text = format_flags(metrics)
    if flag_text:
        fig.text(0.5, 0.01, flag_text, ha="center", va="bottom", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.965])

    output_path = os.path.join(output_dir, f"{base}_collage.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def build_overview(image_paths: list[str], output_path: str) -> str:
    if not image_paths:
        return output_path
    fig, axes = plt.subplots(len(image_paths), 1, figsize=(16, 6 * len(image_paths)))
    if len(image_paths) == 1:
        axes = [axes]
    for ax, path in zip(axes, image_paths):
        ax.imshow(mpimg.imread(path))
        ax.set_title(Path(path).name, fontsize=12, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a forecast collage from GPU-WM NetCDF output")
    parser.add_argument("files", nargs="+", help="NetCDF files to plot")
    parser.add_argument("--output-dir", default="plots", help="Output directory")
    args = parser.parse_args()

    for path in args.files:
        plot_collage(path, args.output_dir)


if __name__ == "__main__":
    main()
