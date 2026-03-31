#!/usr/bin/env python3
"""
GPU-WM: Extract terrain (orography) from GFS GRIB2 and interpolate to model grid.

Reads the GFS 0.25-degree orography field, interpolates it onto the Lambert
Conformal CONUS model grid (256x256, dx=3000m), saves a binary terrain file,
and generates a terrain visualization.

Usage:
    python tools/extract_terrain.py
    python tools/extract_terrain.py --gfs data/gfs_latest.grib2
    python tools/extract_terrain.py --nx 512 --ny 512 --dx 3000
"""

import argparse
import os
import sys
import struct
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RE = 6.371e6  # Earth radius (m)


# ===================================================================
# Lambert Conformal Conic projection (matches include/projection.cuh)
# ===================================================================
class LambertConformal:
    """Lambert Conformal Conic projection identical to the GPU model."""

    def __init__(self, truelat1=38.5, truelat2=38.5, stand_lon=-97.5,
                 ref_lat=38.5, ref_lon=-97.5, nx=256, ny=256, dx=3000.0):
        self.truelat1 = truelat1
        self.truelat2 = truelat2
        self.stand_lon = stand_lon
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx

        phi1 = np.radians(truelat1)
        phi2 = np.radians(truelat2)

        if abs(truelat1 - truelat2) < 1e-6:
            self.n = np.sin(phi1)
        else:
            self.n = (np.log(np.cos(phi1) / np.cos(phi2)) /
                      np.log(np.tan(np.pi / 4 + phi2 / 2) /
                             np.tan(np.pi / 4 + phi1 / 2)))

        self.F = np.cos(phi1) * np.tan(np.pi / 4 + phi1 / 2) ** self.n / self.n
        phi0 = np.radians(ref_lat)
        lam0 = np.radians(ref_lon - stand_lon)
        self.rho0 = RE * self.F / np.tan(np.pi / 4 + phi0 / 2) ** self.n
        theta0 = self.n * lam0
        self.x0 = self.rho0 * np.sin(theta0)
        self.y0 = self.rho0 * np.cos(theta0)

    def ij_to_latlon(self, i, j):
        """Convert grid (i,j) arrays to (lat, lon) arrays."""
        x = (i - self.nx / 2.0) * self.dx
        y = (j - self.ny / 2.0) * self.dy

        x_shift = x + self.x0
        y_shift = self.y0 - y
        rho = np.sign(self.n) * np.sqrt(x_shift ** 2 + y_shift ** 2)
        theta = np.arctan2(x_shift, y_shift)

        lat = np.degrees(2.0 * np.arctan((RE * self.F / rho) ** (1.0 / self.n))
                         - np.pi / 2.0)
        lon = self.stand_lon + np.degrees(theta / self.n)
        return lat, lon


# ===================================================================
# Bilinear interpolation from GFS lat/lon grid to model grid
# ===================================================================
def bilinear_interp_field(gfs_data, grid_info, model_lats, model_lons):
    """Interpolate a 2D GFS field to model grid points."""
    ni = grid_info['ni']
    nj = grid_info['nj']
    lat0 = grid_info['lat_first']   # 90.0 (north)
    lon0 = grid_info['lon_first']   # 0.0
    dlat = grid_info['dj']          # 0.25
    dlon = grid_info['di']          # 0.25

    # Convert model longitudes to 0-360 range
    lon360 = model_lons.copy()
    lon360[lon360 < 0] += 360.0

    # Fractional grid indices
    fi = lon360 / dlon
    fj = (lat0 - model_lats) / dlat

    # Integer indices and weights
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)

    di = fi - i0
    dj = fj - j0

    # Wrap longitude, clamp latitude
    i0 = np.mod(i0, ni)
    i1 = np.mod(i0 + 1, ni)
    j0 = np.clip(j0, 0, nj - 1)
    j1 = np.clip(j0 + 1, 0, nj - 1)

    # Bilinear interpolation
    result = ((1 - di) * (1 - dj) * gfs_data[j0, i0] +
              di       * (1 - dj) * gfs_data[j0, i1] +
              (1 - di) * dj       * gfs_data[j1, i0] +
              di       * dj       * gfs_data[j1, i1])

    return result


# ===================================================================
# GRIB2 reader
# ===================================================================
def read_grib_orog(filepath):
    """Read the orography field from a GFS GRIB2 file.

    Returns
    -------
    data : ndarray (nj, ni)
    grid_info : dict
    """
    import eccodes

    with open(filepath, 'rb') as f:
        while True:
            msgid = eccodes.codes_grib_new_from_file(f)
            if msgid is None:
                break
            name = eccodes.codes_get(msgid, 'shortName')
            tol = eccodes.codes_get(msgid, 'typeOfLevel')
            if name == 'orog' and tol == 'surface':
                vals = eccodes.codes_get_double_array(msgid, 'values')
                ni = eccodes.codes_get(msgid, 'Ni')
                nj = eccodes.codes_get(msgid, 'Nj')
                lat1 = eccodes.codes_get(msgid, 'latitudeOfFirstGridPointInDegrees')
                lon1 = eccodes.codes_get(msgid, 'longitudeOfFirstGridPointInDegrees')
                dj = eccodes.codes_get(msgid, 'jDirectionIncrementInDegrees')
                di = eccodes.codes_get(msgid, 'iDirectionIncrementInDegrees')
                eccodes.codes_release(msgid)
                grid_info = dict(ni=ni, nj=nj, lat_first=lat1, lon_first=lon1,
                                 di=di, dj=dj)
                return vals.reshape(nj, ni), grid_info
            eccodes.codes_release(msgid)

    raise RuntimeError("Orography field (orog, surface) not found in %s" % filepath)


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Extract terrain from GFS GRIB2 for GPU-WM')
    parser.add_argument('--gfs', default='data/gfs_latest.grib2',
                        help='Path to GFS GRIB2 file')
    parser.add_argument('--output', default='data/terrain.bin',
                        help='Output binary terrain file')
    parser.add_argument('--plot', default='plots/terrain_conus.png',
                        help='Output terrain plot')
    parser.add_argument('--nx', type=int, default=256,
                        help='Grid points in x')
    parser.add_argument('--ny', type=int, default=256,
                        help='Grid points in y')
    parser.add_argument('--dx', type=float, default=3000.0,
                        help='Grid spacing (m)')
    args = parser.parse_args()

    nx, ny, dx = args.nx, args.ny, args.dx
    gfs_file = args.gfs

    print("=" * 60)
    print("  GPU-WM: Terrain Extraction")
    print("=" * 60)
    print("  GFS file : %s" % gfs_file)
    print("  Grid     : %d x %d, dx=%.0f m" % (nx, ny, dx))
    print()

    if not os.path.exists(gfs_file):
        print("ERROR: GFS file not found: %s" % gfs_file)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Setup Lambert Conformal projection
    # ------------------------------------------------------------------
    print("Setting up Lambert Conformal projection...")
    proj = LambertConformal(nx=nx, ny=ny, dx=dx)

    ii, jj = np.meshgrid(np.arange(nx, dtype=float),
                          np.arange(ny, dtype=float))
    model_lats, model_lons = proj.ij_to_latlon(ii, jj)

    print("  Domain lat range: %.2f to %.2f" % (model_lats.min(), model_lats.max()))
    print("  Domain lon range: %.2f to %.2f" % (model_lons.min(), model_lons.max()))
    print()

    # ------------------------------------------------------------------
    # 2. Read orography from GFS
    # ------------------------------------------------------------------
    print("Reading orography from GFS GRIB2...")
    orog_gfs, grid_info = read_grib_orog(gfs_file)
    print("  GFS orography: %d x %d" % (grid_info['nj'], grid_info['ni']))
    print("  GFS value range: %.1f to %.1f m" % (orog_gfs.min(), orog_gfs.max()))
    print()

    # ------------------------------------------------------------------
    # 3. Interpolate to model grid
    # ------------------------------------------------------------------
    print("Interpolating to model grid...")
    terrain = bilinear_interp_field(orog_gfs, grid_info, model_lats, model_lons)

    # Clamp negative terrain (ocean) to zero
    terrain = np.maximum(terrain, 0.0)

    print("  Model terrain range: %.1f to %.1f m" % (terrain.min(), terrain.max()))
    print("  Mean elevation: %.1f m" % terrain.mean())
    print()

    # ------------------------------------------------------------------
    # 4. Write binary file: nx(i32), ny(i32), terrain[ny,nx](f64)
    # ------------------------------------------------------------------
    output_file = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    print("Writing binary terrain file: %s" % output_file)
    with open(output_file, 'wb') as f:
        f.write(struct.pack('ii', nx, ny))
        terrain.astype(np.float64).tofile(f)

    file_size = os.path.getsize(output_file)
    expected = 2 * 4 + ny * nx * 8
    print("  File size: %.1f KB (expected: %.1f KB)" %
          (file_size / 1e3, expected / 1e3))
    print("  Format: nx(i32) + ny(i32) + terrain[%d,%d](f64)" % (ny, nx))
    print()

    # ------------------------------------------------------------------
    # 5. Generate terrain visualization
    # ------------------------------------------------------------------
    print("Generating terrain plot...")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LightSource

    plot_dir = os.path.dirname(args.plot)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    # --- Custom terrain colormap: ocean blue -> green lowlands ->
    #     brown foothills -> gray rock -> white peaks
    terrain_colors = [
        (0.00, (0.00, 0.25, 0.50)),   # deep blue (ocean/sea level)
        (0.01, (0.00, 0.40, 0.20)),   # dark green (near sea level)
        (0.05, (0.13, 0.55, 0.13)),   # forest green
        (0.10, (0.33, 0.65, 0.20)),   # green
        (0.15, (0.56, 0.74, 0.22)),   # yellow-green
        (0.25, (0.76, 0.70, 0.30)),   # tan
        (0.35, (0.65, 0.50, 0.24)),   # brown
        (0.50, (0.55, 0.40, 0.30)),   # dark brown
        (0.65, (0.60, 0.55, 0.50)),   # gray-brown
        (0.80, (0.75, 0.72, 0.70)),   # light gray
        (0.90, (0.88, 0.87, 0.86)),   # near-white
        (1.00, (1.00, 1.00, 1.00)),   # white (peaks)
    ]
    cmap_terrain = LinearSegmentedColormap.from_list('terrain_topo', terrain_colors, N=512)

    fig, ax = plt.subplots(figsize=(14, 12))

    # Hillshade for 3D relief effect
    ls = LightSource(azdeg=315, altdeg=35)
    terrain_max = max(terrain.max(), 1.0)

    # Shade the terrain with the hillshade
    rgb = ls.shade(terrain, cmap=cmap_terrain, vert_exag=2.0,
                   blend_mode='soft', vmin=0, vmax=terrain_max)

    ax.imshow(rgb, origin='lower', aspect='equal',
              extent=[model_lons.min(), model_lons.max(),
                      model_lats.min(), model_lats.max()])

    # Contour lines for major elevations
    contour_levels = [100, 300, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    contour_levels = [lev for lev in contour_levels if lev < terrain.max()]
    if contour_levels:
        cs = ax.contour(model_lons, model_lats, terrain,
                        levels=contour_levels,
                        colors='k', linewidths=0.3, alpha=0.4)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%d m')

    # Lat/lon grid lines
    lat_ticks = np.arange(np.floor(model_lats.min()), np.ceil(model_lats.max()) + 1, 2)
    lon_ticks = np.arange(np.floor(model_lons.min()), np.ceil(model_lons.max()) + 1, 2)
    for lat_line in lat_ticks:
        ax.axhline(lat_line, color='gray', linewidth=0.3, alpha=0.5)
    for lon_line in lon_ticks:
        ax.axvline(lon_line, color='gray', linewidth=0.3, alpha=0.5)

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(
        'GPU-WM Terrain Height (GFS Orography)\n'
        '%d x %d grid, dx=%.0f m\n'
        'Lat: %.1f to %.1f  Lon: %.1f to %.1f' %
        (nx, ny, dx,
         model_lats.min(), model_lats.max(),
         model_lons.min(), model_lons.max()),
        fontsize=14, fontweight='bold')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_terrain,
                               norm=plt.Normalize(vmin=0, vmax=terrain_max))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, label='Elevation (m)', shrink=0.8, pad=0.02)

    # Add peak elevation annotation
    peak_idx = np.unravel_index(np.argmax(terrain), terrain.shape)
    peak_lat = model_lats[peak_idx]
    peak_lon = model_lons[peak_idx]
    ax.plot(peak_lon, peak_lat, 'r^', markersize=8, markeredgecolor='k',
            markeredgewidth=0.5)
    ax.annotate('%.0f m' % terrain.max(),
                xy=(peak_lon, peak_lat),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(args.plot, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: %s" % args.plot)
    print()

    print("=" * 60)
    print("  Terrain extraction complete!")
    print("  Binary file : %s (%.1f KB)" % (output_file, file_size / 1e3))
    print("  Plot        : %s" % args.plot)
    print("  Elevation   : %.0f - %.0f m" % (terrain.min(), terrain.max()))
    print("=" * 60)


if __name__ == '__main__':
    main()
