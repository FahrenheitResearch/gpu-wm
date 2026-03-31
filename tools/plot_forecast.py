#!/usr/bin/env python3
"""
GPU-WM: Combined forecast plot with terrain.

Overlays model output (surface temperature, wind barbs) on shaded terrain
relief. Uses the same Lambert Conformal projection as the model.

Generates plots for t=0 (initial) and the last available output file.

Usage:
    python tools/plot_forecast.py
    python tools/plot_forecast.py --terrain data/terrain.bin
    python tools/plot_forecast.py output/gpuwm_000003.nc
"""

import argparse
import os
import sys
import struct
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, LightSource

try:
    from netCDF4 import Dataset
except ImportError:
    print("ERROR: netCDF4 required. Install with: pip install netCDF4")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RE = 6.371e6
KAPPA = 0.286
P0 = 100000.0


# ===================================================================
# Read terrain binary file
# ===================================================================
def read_terrain(filepath):
    """Read terrain from binary file written by extract_terrain.py.

    Format: nx(i32) + ny(i32) + terrain[ny,nx](f64)
    """
    with open(filepath, 'rb') as f:
        nx, ny = struct.unpack('ii', f.read(8))
        terrain = np.frombuffer(f.read(ny * nx * 8), dtype=np.float64).reshape(ny, nx)
    return terrain, nx, ny


# ===================================================================
# Read model output (NetCDF)
# ===================================================================
def read_model_nc(filepath):
    """Read a GPU-WM NetCDF output file."""
    ds = Dataset(filepath, 'r')

    data = {}
    data['nx'] = len(ds.dimensions['x'])
    data['ny'] = len(ds.dimensions['y'])
    data['nz'] = len(ds.dimensions['z'])

    time_val = ds.variables['time'][:]
    data['time'] = float(time_val) if np.ndim(time_val) == 0 else float(time_val[0])

    data['x'] = np.array(ds.variables['x'][:])
    data['y'] = np.array(ds.variables['y'][:])
    data['z'] = np.array(ds.variables['z'][:])

    data['u'] = np.array(ds.variables['U'][:])
    data['v'] = np.array(ds.variables['V'][:])
    data['theta'] = np.array(ds.variables['THETA'][:])
    data['qv'] = np.array(ds.variables['QV'][:])
    data['qc'] = np.array(ds.variables['QC'][:])
    data['qr'] = np.array(ds.variables['QR'][:])
    if 'TERRAIN' in ds.variables:
        data['terrain'] = np.array(ds.variables['TERRAIN'][:])

    if 'lat' in ds.variables:
        data['lat'] = np.array(ds.variables['lat'][:])
        data['lon'] = np.array(ds.variables['lon'][:])

    ds.close()
    return data


# ===================================================================
# Temperature colormap (weather-forecast style)
# ===================================================================
def temperature_cmap():
    """Weather forecast temperature colormap (F or C)."""
    colors = [
        '#4b0082', '#0000cd', '#0040ff', '#0080ff', '#00bfff',
        '#00e5ff', '#40ffbf', '#80ff80', '#bfff40', '#e5ff00',
        '#ffff00', '#ffd700', '#ffaa00', '#ff7f00', '#ff4500',
        '#ff0000', '#dc143c', '#b22222', '#8b0000', '#4a0000'
    ]
    return LinearSegmentedColormap.from_list('temp', colors, N=256)


# ===================================================================
# Plot a single forecast frame with terrain
# ===================================================================
def plot_forecast_frame(data, terrain, output_path):
    """Create a combined surface temperature + terrain + wind barb plot.

    Parameters
    ----------
    data : dict      Model output from read_model_nc
    terrain : ndarray Terrain height (ny, nx) in meters
    output_path : str Output PNG path
    """
    nx, ny = data['nx'], data['ny']
    time = data['time']
    hours = time / 3600.0
    mins = time / 60.0

    k_sfc = 1  # near-surface level

    # Approximate surface temperature from potential temperature
    # T = theta * (p/p0)^kappa, approximate p ~ p0 * exp(-z/H)
    # For surface, use a simple Exner-function approach
    z_sfc = data['z'][k_sfc]
    p_sfc = P0 * np.exp(-z_sfc / 8500.0)
    exner = (p_sfc / P0) ** KAPPA
    T_sfc_C = data['theta'][k_sfc] * exner - 273.15  # Celsius

    u_sfc = data['u'][k_sfc]
    v_sfc = data['v'][k_sfc]

    # Use lat/lon coordinates if available
    if 'lat' in data:
        lat = data['lat']
        lon = data['lon']
    else:
        # Fallback: use x,y in km
        lat = np.broadcast_to(data['y'][:, np.newaxis], (ny, nx))
        lon = np.broadcast_to(data['x'][np.newaxis, :], (ny, nx))

    # ===== Create figure with 2 panels =====
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # --- Panel 1: Surface Temperature draped over terrain ---
    ax = axes[0]

    # Terrain hillshade
    ls = LightSource(azdeg=315, altdeg=35)
    terrain_max = max(terrain.max(), 1.0)

    # Temperature field
    cmap_t = temperature_cmap()
    vmin_t = max(-30, np.percentile(T_sfc_C, 1))
    vmax_t = min(50, np.percentile(T_sfc_C, 99))
    if vmax_t - vmin_t < 5:
        mid = np.mean(T_sfc_C)
        vmin_t = mid - 10
        vmax_t = mid + 10

    # Plot temperature as colored fill
    c = ax.pcolormesh(lon, lat, T_sfc_C, cmap=cmap_t,
                      vmin=vmin_t, vmax=vmax_t, shading='auto',
                      alpha=0.85, zorder=2)

    # Overlay terrain hillshade (semi-transparent gray)
    terrain_gray = terrain / terrain_max
    hill = ls.hillshade(terrain, vert_exag=3.0, dx=1, dy=1, fraction=1.2)
    ax.pcolormesh(lon, lat, hill, cmap='gray', shading='auto',
                  alpha=0.25, zorder=3)

    # Terrain contour lines
    contour_levels = [200, 500, 1000, 1500, 2000, 2500, 3000]
    contour_levels = [lev for lev in contour_levels if lev < terrain.max()]
    if contour_levels:
        cs = ax.contour(lon, lat, terrain, levels=contour_levels,
                        colors='k', linewidths=0.4, alpha=0.5, zorder=4)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%d m')

    # Wind barbs
    skip = max(1, nx // 25)
    ax.barbs(lon[::skip, ::skip], lat[::skip, ::skip],
             u_sfc[::skip, ::skip] * 1.944,
             v_sfc[::skip, ::skip] * 1.944,
             length=5, linewidth=0.5, color='black', zorder=5)

    cb = plt.colorbar(c, ax=ax, label='Temperature (\u00b0C)', shrink=0.85, pad=0.02)

    if hours >= 1.0:
        time_str = '%.1f hr' % hours
    else:
        time_str = '%.0f min' % mins

    ax.set_title(
        'GPU-WM Surface Temperature + Terrain + Winds\n'
        'Forecast t = %s (%.0f s)' % (time_str, time),
        fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # --- Panel 2: Terrain with wind speed overlay ---
    ax = axes[1]

    # Terrain shaded relief as base
    terrain_colors = [
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
    cmap_terr = LinearSegmentedColormap.from_list('terrain_topo', terrain_colors, N=512)

    rgb = ls.shade(terrain, cmap=cmap_terr, vert_exag=2.0,
                   blend_mode='soft', vmin=0, vmax=terrain_max)

    ax.imshow(rgb, origin='lower', aspect='equal',
              extent=[lon.min(), lon.max(), lat.min(), lat.max()],
              zorder=1)

    # Overlay wind speed as semi-transparent layer
    wspd = np.sqrt(u_sfc ** 2 + v_sfc ** 2)
    c2 = ax.pcolormesh(lon, lat, wspd, cmap='YlOrRd', shading='auto',
                       vmin=0, vmax=max(20, np.percentile(wspd, 99)),
                       alpha=0.55, zorder=2)
    plt.colorbar(c2, ax=ax, label='Wind Speed (m/s)', shrink=0.85, pad=0.02)

    # Wind barbs
    ax.barbs(lon[::skip, ::skip], lat[::skip, ::skip],
             u_sfc[::skip, ::skip] * 1.944,
             v_sfc[::skip, ::skip] * 1.944,
             length=5, linewidth=0.5, color='black', zorder=5)

    # Temperature contours
    temp_levels = np.arange(np.floor(vmin_t / 5) * 5,
                            np.ceil(vmax_t / 5) * 5 + 1, 5)
    if len(temp_levels) > 2:
        cs_t = ax.contour(lon, lat, T_sfc_C, levels=temp_levels,
                          colors='blue', linewidths=0.6, alpha=0.7, zorder=4)
        ax.clabel(cs_t, inline=True, fontsize=7, fmt='%.0f\u00b0C')

    ax.set_title(
        'GPU-WM Wind Speed + Terrain Relief + Temperature Contours\n'
        'Forecast t = %s' % time_str,
        fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: %s" % output_path)


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description='GPU-WM: Forecast plots with terrain overlay')
    parser.add_argument('files', nargs='*',
                        help='NetCDF output files to plot (default: auto-detect)')
    parser.add_argument('--terrain', default='data/terrain.bin',
                        help='Terrain binary file')
    parser.add_argument('--output-dir', default='plots',
                        help='Output directory for plots')
    args = parser.parse_args()

    print("=" * 60)
    print("  GPU-WM: Forecast Visualization with Terrain")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Find output files
    # ------------------------------------------------------------------
    if args.files:
        nc_files = args.files
    else:
        nc_files = sorted(glob.glob('output/gpuwm_*.nc'))
        if not nc_files:
            print("ERROR: No output files found in output/")
            print("  Run the GPU-WM model first.")
            sys.exit(1)

    terrain = None
    tnx = tny = None
    if os.path.exists(args.terrain):
        print("Loading terrain: %s" % args.terrain)
        terrain, tnx, tny = read_terrain(args.terrain)
        print("  Terrain grid: %d x %d" % (tnx, tny))
        print("  Elevation range: %.0f - %.0f m" % (terrain.min(), terrain.max()))
        print()
    else:
        print("Terrain sidecar not found: %s" % args.terrain)
        print("  Will use TERRAIN from NetCDF when available.")
        print()

    # Select t=0 (first) and last available
    files_to_plot = []
    files_to_plot.append(nc_files[0])
    if len(nc_files) > 1:
        files_to_plot.append(nc_files[-1])

    print("Plotting %d file(s):" % len(files_to_plot))
    for f in files_to_plot:
        print("  %s" % f)
    print()

    # ------------------------------------------------------------------
    # 2. Generate plots
    # ------------------------------------------------------------------
    for ncfile in files_to_plot:
        print("Processing: %s" % ncfile)
        data = read_model_nc(ncfile)

        terrain_for_plot = None
        if 'terrain' in data:
            terrain_for_plot = data['terrain']
        elif terrain is not None and data['nx'] == tnx and data['ny'] == tny:
            terrain_for_plot = terrain
        elif terrain is not None:
            print("  WARNING: Grid size mismatch! Model=%dx%d, Terrain=%dx%d" %
                  (data['nx'], data['ny'], tnx, tny))
            print("  Skipping this file.")
            continue
        else:
            print("  WARNING: No terrain available in NetCDF or sidecar.")
            print("  Skipping this file.")
            continue

        # Build output filename
        base = os.path.splitext(os.path.basename(ncfile))[0]
        out_path = os.path.join(args.output_dir, '%s_forecast.png' % base)
        plot_forecast_frame(data, terrain_for_plot, out_path)
        print()

    print("=" * 60)
    print("  Forecast visualization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
