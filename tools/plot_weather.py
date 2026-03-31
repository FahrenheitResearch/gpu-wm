#!/usr/bin/env python3
"""GPU-WM: Production weather visualization
Generates plots that look like actual weather maps/radar/satellite"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import sys, os, glob

try:
    from netCDF4 import Dataset
except ImportError:
    print("pip install netCDF4"); sys.exit(1)


def nws_reflectivity_cmap():
    """NWS-style radar reflectivity colormap"""
    colors = [
        (0.00, (0.0, 0.0, 0.0, 0.0)),    # transparent below 5
        (0.07, (0.0, 0.93, 0.93, 1.0)),   # 5 dBZ - light cyan
        (0.13, (0.0, 0.63, 0.96, 1.0)),   # 10 - blue
        (0.20, (0.0, 0.0, 0.96, 1.0)),    # 15 - dark blue
        (0.27, (0.0, 1.0, 0.0, 1.0)),     # 20 - green
        (0.33, (0.0, 0.78, 0.0, 1.0)),    # 25 - darker green
        (0.40, (0.0, 0.56, 0.0, 1.0)),    # 30 - dark green
        (0.47, (1.0, 1.0, 0.0, 1.0)),     # 35 - yellow
        (0.53, (0.91, 0.75, 0.0, 1.0)),   # 40 - gold
        (0.60, (1.0, 0.56, 0.0, 1.0)),    # 45 - orange
        (0.67, (1.0, 0.0, 0.0, 1.0)),     # 50 - red
        (0.73, (0.84, 0.0, 0.0, 1.0)),    # 55 - dark red
        (0.80, (0.75, 0.0, 0.0, 1.0)),    # 60 - maroon
        (0.87, (1.0, 0.0, 1.0, 1.0)),     # 65 - magenta
        (0.93, (0.6, 0.33, 0.79, 1.0)),   # 70 - purple
        (1.00, (1.0, 1.0, 1.0, 1.0)),     # 75 - white
    ]
    return LinearSegmentedColormap.from_list('nws_refl', colors, N=256)


def temperature_cmap():
    """Weather forecast temperature colormap"""
    colors = [
        '#4b0082', '#0000cd', '#0040ff', '#0080ff', '#00bfff',
        '#00e5ff', '#40ffbf', '#80ff80', '#bfff40', '#e5ff00',
        '#ffff00', '#ffd700', '#ffaa00', '#ff7f00', '#ff4500',
        '#ff0000', '#dc143c', '#b22222', '#8b0000', '#4a0000'
    ]
    return LinearSegmentedColormap.from_list('temp', colors, N=256)


def plot_weather(ncfile, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    ds = Dataset(ncfile, 'r')

    def read_first_available(*names):
        for name in names:
            if name in ds.variables:
                return np.array(ds.variables[name][:])
        raise KeyError(f"Missing variables {names} in {ncfile}")

    nx = len(ds.dimensions['x'])
    ny = len(ds.dimensions['y'])
    nz = len(ds.dimensions['z'])
    time = ds.variables['time'][:]
    t = float(time) if np.ndim(time) == 0 else float(time[0])
    z = ds.variables['z'][:]
    x = ds.variables['x'][:]
    y = ds.variables['y'][:]

    u = read_first_available('U_MASS', 'U')
    v = read_first_available('V_MASS', 'V')
    w = read_first_available('W_MASS', 'W')
    theta = np.array(ds.variables['THETA'][:])
    qv = np.array(ds.variables['QV'][:])
    qc = np.array(ds.variables['QC'][:])
    qr = np.array(ds.variables['QR'][:])

    has_latlon = 'lat' in ds.variables
    if has_latlon:
        lat = np.array(ds.variables['lat'][:])
        lon = np.array(ds.variables['lon'][:])
    ds.close()

    # Determine coordinate system
    if has_latlon:
        xcoord, ycoord = lon, lat
        xlabel, ylabel = 'Longitude', 'Latitude'
        aspect = 1.3
    else:
        xcoord = np.broadcast_to(x[np.newaxis, :], (ny, nx))
        ycoord = np.broadcast_to(y[:, np.newaxis], (ny, nx))
        xlabel, ylabel = 'x (km)', 'y (km)'
        aspect = 'equal'

    k_sfc = 1  # surface level
    hours = t / 3600.0
    mins = t / 60.0

    # ================================================================
    # FIGURE 1: Simulated Radar
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 10))

    # Composite reflectivity from qr + qc
    qr_col = np.max(qr, axis=0) * 1000  # g/kg
    qc_col = np.max(qc, axis=0) * 1000

    # Marshall-Palmer: Z = 200 * (rho*qr)^1.6, simplified
    dbz = np.where(qr_col > 0.01,
                   10 * np.log10(300.0 * qr_col**1.4 + 0.1),
                   np.where(qc_col > 0.01,
                            10 * np.log10(50.0 * qc_col**1.0 + 0.1),
                            -999))

    cmap_r = nws_reflectivity_cmap()
    cmap_r.set_under('black')

    c = ax.pcolormesh(xcoord, ycoord, dbz, cmap=cmap_r,
                      vmin=-10, vmax=75, shading='auto')
    cb = plt.colorbar(c, ax=ax, label='dBZ', shrink=0.85,
                      extend='both', ticks=np.arange(-10, 80, 10))

    ax.set_title(f'GPU-WM Simulated Radar Reflectivity\nt = {hours:.1f} hr ({mins:.0f} min)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if aspect != 'equal':
        ax.set_aspect(aspect)
    else:
        ax.set_aspect('equal')
    ax.set_facecolor('black')

    outf = os.path.join(output_dir, f'radar_t{t:08.0f}.png')
    plt.savefig(outf, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f'Saved: {outf}')

    # ================================================================
    # FIGURE 2: Surface Temperature Map
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 10))

    # Convert theta to temperature: T = theta * (p/p0)^kappa
    # Approximate surface pressure from base state
    p_sfc = 100000.0  # Pa (approximate)
    kappa = 0.286
    exner = (p_sfc / 100000.0)**kappa
    T_sfc = theta[k_sfc] * exner - 273.15  # Celsius

    cmap_t = temperature_cmap()
    vmin_t = max(-30, np.percentile(T_sfc, 2))
    vmax_t = min(45, np.percentile(T_sfc, 98))
    if vmax_t - vmin_t < 5:
        vmin_t = np.mean(T_sfc) - 10
        vmax_t = np.mean(T_sfc) + 10

    c = ax.pcolormesh(xcoord, ycoord, T_sfc, cmap=cmap_t,
                      vmin=vmin_t, vmax=vmax_t, shading='auto')
    plt.colorbar(c, ax=ax, label='Temperature (°C)', shrink=0.85)

    # Wind barbs
    skip = max(1, nx // 25)
    u_sfc = u[k_sfc]
    v_sfc = v[k_sfc]
    # Convert to knots for barbs
    ax.barbs(xcoord[::skip, ::skip], ycoord[::skip, ::skip],
             u_sfc[::skip, ::skip] * 1.944, v_sfc[::skip, ::skip] * 1.944,
             length=5, linewidth=0.5, color='black', zorder=5)

    ax.set_title(f'GPU-WM Surface Temperature & Winds\nt = {hours:.1f} hr ({mins:.0f} min)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if aspect != 'equal':
        ax.set_aspect(aspect)
    else:
        ax.set_aspect('equal')

    outf = os.path.join(output_dir, f'sfc_temp_t{t:08.0f}.png')
    plt.savefig(outf, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outf}')

    # ================================================================
    # FIGURE 3: Simulated Satellite (IR-style cloud tops)
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 10))

    # Cloud top temperature: find highest level with qc > threshold
    cloud_top_T = np.full((ny, nx), np.nan)
    total_cwp = np.zeros((ny, nx))

    for k in range(nz - 1, -1, -1):
        cloud = (qc[k] + qr[k]) > 1e-5
        # Cloud top = coldest cloud
        p_k = 100000.0 * np.exp(-z[k] / 8500.0)  # approximate pressure
        T_k = theta[k] * (p_k / 100000.0)**kappa - 273.15
        cloud_top_T = np.where(cloud & np.isnan(cloud_top_T), T_k, cloud_top_T)
        total_cwp += (qc[k] + qr[k]) * 500  # crude integration

    # Where no cloud, use surface T
    clear_sky = np.isnan(cloud_top_T)
    p_sfc_field = 100000.0
    T_surface = theta[k_sfc] * (p_sfc_field / 100000.0)**kappa - 273.15
    display_T = np.where(clear_sky, T_surface, cloud_top_T)

    # IR satellite: cold = bright (white), warm = dark (gray/black)
    cmap_ir = plt.cm.gray_r
    c = ax.pcolormesh(xcoord, ycoord, display_T, cmap=cmap_ir,
                      vmin=-70, vmax=30, shading='auto')
    plt.colorbar(c, ax=ax, label='Brightness Temperature (°C)', shrink=0.85)

    ax.set_title(f'GPU-WM Simulated IR Satellite\nt = {hours:.1f} hr ({mins:.0f} min)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if aspect != 'equal':
        ax.set_aspect(aspect)
    else:
        ax.set_aspect('equal')

    outf = os.path.join(output_dir, f'satellite_t{t:08.0f}.png')
    plt.savefig(outf, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outf}')

    # ================================================================
    # FIGURE 4: Upper-level jet + surface fronts
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: upper-level wind on a geometric-height slice
    ax = axes[0]
    k_jet = np.argmin(np.abs(z - 5500))
    wspd_jet = np.sqrt(u[k_jet]**2 + v[k_jet]**2)

    c = ax.pcolormesh(xcoord, ycoord, wspd_jet, cmap='hot_r',
                      vmin=0, vmax=max(30, np.nanpercentile(wspd_jet, 99)),
                      shading='auto')
    plt.colorbar(c, ax=ax, label='Wind Speed (m/s)', shrink=0.85)

    step = max(1, min(nx, ny) // 28)
    ax.quiver(
        xcoord[::step, ::step],
        ycoord[::step, ::step],
        u[k_jet, ::step, ::step],
        v[k_jet, ::step, ::step],
        color='white',
        pivot='mid',
        scale=700,
        width=0.0018,
        alpha=0.8,
    )

    ax.set_title(f'Upper-Level Wind Speed + Vectors\nz ≈ {z[k_jet]/1000:.1f} km',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_aspect('equal')

    # Right: Low-level moisture convergence
    ax = axes[1]
    k_low = min(3, nz - 1)  # ~1.5 km
    # Moisture flux convergence proxy: qv * w (updraft + moisture = storms)
    moist_conv = qv[k_low] * 1000 * np.clip(w[k_low], 0, None)

    qv_plot = qv[k_sfc] * 1000  # g/kg
    c = ax.pcolormesh(xcoord, ycoord, qv_plot, cmap='YlGnBu',
                      vmin=0, vmax=max(5, np.nanpercentile(qv_plot, 98)),
                      shading='auto')
    plt.colorbar(c, ax=ax, label='Mixing Ratio (g/kg)', shrink=0.85)

    # Overlay theta contours (fronts show up as tight gradients)
    theta_sfc = theta[k_sfc]
    theta_finite = theta_sfc[np.isfinite(theta_sfc)]
    levels = np.array([])
    if theta_finite.size:
        levels = np.arange(round(np.nanmin(theta_finite)), round(np.nanmax(theta_finite)), 2)
    if len(levels) > 2:
        ax.contour(xcoord, ycoord, theta_sfc, levels=levels,
                   colors='red', linewidths=0.5, alpha=0.7)

    ax.set_title(f'Surface Moisture + θ Contours (fronts)\nt = {hours:.1f} hr',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_aspect('equal')

    plt.suptitle(f'GPU-WM  t = {hours:.1f} hr ({mins:.0f} min)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    outf = os.path.join(output_dir, f'analysis_t{t:08.0f}.png')
    plt.savefig(outf, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outf}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        files = sorted(glob.glob('output/gpuwm_*.nc'))
        if not files:
            print('No .nc files found'); sys.exit(1)
    else:
        files = sys.argv[1:]

    for f in files:
        print(f'Processing {f}...')
        plot_weather(f)
