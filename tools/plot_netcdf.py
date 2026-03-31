#!/usr/bin/env python3
"""Plot GPU-WM NetCDF output - weather-style maps"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys, os, glob

try:
    from netCDF4 import Dataset
except ImportError:
    print("pip install netCDF4")
    sys.exit(1)


def plot_weather_map(ncfile, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    ds = Dataset(ncfile, 'r')

    nx = len(ds.dimensions['x'])
    ny = len(ds.dimensions['y'])
    nz = len(ds.dimensions['z'])
    time = ds.variables['time'][:]
    t = float(time) if np.ndim(time) == 0 else float(time[0])
    z = ds.variables['z'][:]
    x = ds.variables['x'][:]
    y = ds.variables['y'][:]

    u = ds.variables['U'][:]
    v = ds.variables['V'][:]
    w = ds.variables['W'][:]
    theta = ds.variables['THETA'][:]
    qv = ds.variables['QV'][:]
    qc = ds.variables['QC'][:]
    qr = ds.variables['QR'][:]

    # Try lat/lon
    has_latlon = 'lat' in ds.variables
    if has_latlon:
        lat = ds.variables['lat'][:]
        lon = ds.variables['lon'][:]

    ds.close()

    # --- Plan view at multiple levels ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(f'GPU-WM   t = {t:.0f} s  ({t/60:.0f} min  /  {t/3600:.1f} hr)', fontsize=16, fontweight='bold')

    # Surface-level fields (k=1)
    k_sfc = 1
    k_mid = nz // 4   # ~2-3 km
    k_cld = nz // 3   # ~5 km

    # 1) 10m wind speed + barbs
    ax = axes[0, 0]
    wspd = np.sqrt(u[k_sfc]**2 + v[k_sfc]**2)
    c = ax.pcolormesh(x, y, wspd, cmap='YlOrRd', shading='auto', vmin=0, vmax=max(25, np.percentile(wspd, 99)))
    plt.colorbar(c, ax=ax, label='m/s', shrink=0.8)
    # Wind barbs (subsample)
    skip = max(1, nx // 30)
    ax.barbs(x[::skip], y[::skip], u[k_sfc, ::skip, ::skip], v[k_sfc, ::skip, ::skip],
             length=5, linewidth=0.4, color='k')
    ax.set_title(f'Wind Speed & Barbs  z={z[k_sfc]/1000:.1f}km')
    ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)')
    ax.set_aspect('equal')

    # 2) Potential temperature perturbation
    ax = axes[0, 1]
    th_mean = np.mean(theta[k_sfc])
    th_pert = theta[k_sfc] - th_mean
    vmax = max(3, np.percentile(np.abs(th_pert), 99))
    c = ax.pcolormesh(x, y, th_pert, cmap='RdBu_r', shading='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar(c, ax=ax, label='K', shrink=0.8)
    ax.set_title(f'θ\' (perturbation)  z={z[k_sfc]/1000:.1f}km')
    ax.set_xlabel('x (km)')
    ax.set_aspect('equal')

    # 3) Vertical velocity at mid-level
    ax = axes[0, 2]
    w_mid = w[k_mid]
    vmax_w = max(1, np.percentile(np.abs(w_mid), 99.5))
    c = ax.pcolormesh(x, y, w_mid, cmap='RdBu_r', shading='auto', vmin=-vmax_w, vmax=vmax_w)
    plt.colorbar(c, ax=ax, label='m/s', shrink=0.8)
    ax.set_title(f'Vertical Velocity (w)  z={z[k_mid]/1000:.1f}km')
    ax.set_xlabel('x (km)')
    ax.set_aspect('equal')

    # 4) Composite reflectivity (derived from qr)
    ax = axes[1, 0]
    # Z = 200 * (rho * qr)^1.6, simplified dBZ
    # Sum column-max of qr as proxy for composite reflectivity
    qr_col_max = np.max(qr, axis=0) * 1000.0  # g/kg
    # Convert to pseudo-dBZ: Z ~ 300 * qr^1.4 (Marshall-Palmer approx)
    dbz = np.where(qr_col_max > 0.001,
                   10 * np.log10(300.0 * qr_col_max**1.4 + 1e-10),
                   -20)
    dbz = np.clip(dbz, -10, 75)

    # NWS reflectivity colormap
    cmap_dbz = mcolors.ListedColormap([
        '#00ecec', '#01a0f6', '#0000f6', '#00ff00',
        '#00c800', '#009000', '#ffff00', '#e7c000',
        '#ff9000', '#ff0000', '#d60000', '#c00000',
        '#ff00ff', '#9955c9', '#808080'
    ])
    bounds = [-10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75]
    norm = mcolors.BoundaryNorm(bounds, cmap_dbz.N)
    c = ax.pcolormesh(x, y, dbz, cmap=cmap_dbz, norm=norm, shading='auto')
    plt.colorbar(c, ax=ax, label='dBZ', shrink=0.8, ticks=bounds[::2])
    ax.set_title('Composite Reflectivity')
    ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)')
    ax.set_aspect('equal')

    # 5) Cloud water column
    ax = axes[1, 1]
    # Vertically integrated cloud water (cloud optical depth proxy)
    dz_avg = z[-1] / nz  # approximate
    cwp = np.sum(qc, axis=0) * dz_avg * 1000  # g/m^2 (crude)
    c = ax.pcolormesh(x, y, cwp, cmap='Greys_r', shading='auto',
                      vmin=0, vmax=max(1, np.percentile(cwp[cwp > 0], 99) if np.any(cwp > 0) else 1))
    plt.colorbar(c, ax=ax, label='g/m²', shrink=0.8)
    ax.set_title('Cloud Water Path (satellite view)')
    ax.set_xlabel('x (km)')
    ax.set_aspect('equal')

    # 6) Water vapor at surface
    ax = axes[1, 2]
    c = ax.pcolormesh(x, y, qv[k_sfc] * 1000, cmap='YlGnBu', shading='auto')
    plt.colorbar(c, ax=ax, label='g/kg', shrink=0.8)
    ax.set_title(f'Water Vapor  z={z[k_sfc]/1000:.1f}km')
    ax.set_xlabel('x (km)')
    ax.set_aspect('equal')

    plt.tight_layout()
    base = os.path.splitext(os.path.basename(ncfile))[0]
    outfile = os.path.join(output_dir, f'{base}_weather.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')

    # --- Cross section through domain center ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    jc = ny // 2
    fig.suptitle(f'GPU-WM Cross Section (y={y[jc]:.0f}km)  t={t/60:.0f}min', fontsize=14)

    ax = axes[0]
    th_xs = theta[:, jc, :] - np.mean(theta[:, jc, :], axis=1, keepdims=True)
    vmax = max(0.5, np.percentile(np.abs(th_xs), 99))
    c = ax.pcolormesh(x, z/1000, th_xs, cmap='RdBu_r', shading='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar(c, ax=ax, label='K')
    ax.set_title('θ\' perturbation'); ax.set_ylabel('Height (km)'); ax.set_xlabel('x (km)')

    ax = axes[1]
    c = ax.pcolormesh(x, z/1000, w[:, jc, :], cmap='RdBu_r', shading='auto',
                      vmin=-max(1, np.percentile(np.abs(w[:, jc, :]), 99)),
                      vmax=max(1, np.percentile(np.abs(w[:, jc, :]), 99)))
    plt.colorbar(c, ax=ax, label='m/s')
    ax.set_title('Vertical velocity'); ax.set_xlabel('x (km)')

    ax = axes[2]
    qc_xs = qc[:, jc, :] * 1000
    qr_xs = qr[:, jc, :] * 1000
    c = ax.pcolormesh(x, z/1000, qc_xs, cmap='Blues', shading='auto',
                      vmin=0, vmax=max(0.1, np.max(qc_xs)))
    ax.contour(x, z/1000, qr_xs, levels=[0.1, 0.5, 1.0, 2.0], colors='green', linewidths=0.8)
    plt.colorbar(c, ax=ax, label='g/kg')
    ax.set_title('Cloud (blue) + Rain (green contours)'); ax.set_xlabel('x (km)')

    plt.tight_layout()
    outfile2 = os.path.join(output_dir, f'{base}_xsect.png')
    plt.savefig(outfile2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile2}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        files = sorted(glob.glob('output/gpuwm_*.nc'))
        if not files:
            print('No .nc files found in output/')
            sys.exit(1)
    else:
        files = sys.argv[1:]

    for f in files:
        print(f'Processing {f}...')
        plot_weather_map(f)
