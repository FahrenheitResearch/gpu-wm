#!/usr/bin/env python3
"""
GPU-WM: Initialize model from GRIB2 data.

Reads pressure-level and surface GRIB2 analyses, interpolates them onto a
Lambert Conformal regional grid, vertically interpolates to model height
levels using the hypsometric equation, and writes a binary initialization
file that the GPU model can load.

This remains backward-compatible with the original single-file GFS workflow,
but it can also ingest split pressure/surface products such as native HRRR
`wrfprsf` + `wrfsfcf`.

Usage:
    python tools/init_from_gfs.py --nx 256 --ny 256 --dx 3000
    python tools/init_from_gfs.py --gfs data/gfs_latest.grib2 --output data/gfs_init.bin
    python tools/init_from_gfs.py --pressure-grib data/hrrr.t18z.wrfprsf00.grib2 \\
        --surface-grib data/hrrr.t18z.wrfsfcf00.grib2 --output data/hrrr_init.bin
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time as time_mod
import struct
from datetime import datetime, timezone
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (must match include/constants.cuh)
# ---------------------------------------------------------------------------
R_D = 287.04        # Gas constant dry air (J/kg/K)
CP_D = 1004.5       # Specific heat at const pressure (J/kg/K)
KAPPA = R_D / CP_D  # ~0.2857
G = 9.81            # Gravitational acceleration (m/s^2)
P0 = 100000.0       # Reference pressure (Pa)
RE = 6.371e6        # Earth radius (m)
PROJ_MAGIC = b'GWMPRJ1\x00'
PRES_MAGIC = b'GWMPRES1'
TERRAIN_MAGIC = b'GWMTERR1'
INIT_MODE_MAGIC = b'GWMINIT1'
TIME_MAGIC = b'GWMTIME1'


def build_stretched_vertical_levels(nz, ztop):
    """Return stretched reference z and eta levels matching the CUDA core."""
    if nz <= 0:
        raise ValueError("nz must be positive")

    n_sfc = nz // 5
    n_sfc = min(max(n_sfc, 3), 8)
    if nz > 1:
        n_sfc = min(n_sfc, nz - 1)
    else:
        n_sfc = 1

    eta_sfc = 1000.0 / max(ztop, 1000.0)
    eta_sfc = min(max(eta_sfc, 0.02), 0.08)
    sfc_power = 1.4

    eta_w = np.zeros(nz + 1, dtype=np.float64)
    eta_w[0] = 0.0
    for k in range(1, nz + 1):
        if k <= n_sfc:
            frac = k / n_sfc
            eta_w[k] = eta_sfc * frac ** sfc_power
        else:
            frac = (k - n_sfc) / (nz - n_sfc)
            eta_w[k] = eta_sfc + (1.0 - eta_sfc) * frac
    eta_w[-1] = 1.0

    eta_m = 0.5 * (eta_w[:-1] + eta_w[1:])
    z_levels = eta_m * ztop
    return z_levels, eta_m, eta_w


# ===================================================================
# Lambert Conformal Conic projection
# (matches include/projection.cuh exactly)
# ===================================================================
class LambertConformal:
    """Lambert Conformal Conic projection identical to the GPU model."""

    def __init__(self, truelat1=38.5, truelat2=38.5, stand_lon=-97.5,
                 ref_lat=38.5, ref_lon=-97.5, nx=512, ny=512, dx=3000.0):
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


class LambertSourceGrid:
    """Lambert source grid defined by GRIB metadata and the first grid point."""

    def __init__(
        self,
        *,
        truelat1,
        truelat2,
        stand_lon,
        lat_first,
        lon_first,
        ni,
        nj,
        dx,
        dy,
        i_scans_negatively=False,
        j_scans_positively=True,
    ):
        self.truelat1 = float(truelat1)
        self.truelat2 = float(truelat2)
        self.stand_lon = normalize_longitude(float(stand_lon))
        self.lat_first = float(lat_first)
        self.lon_first = normalize_longitude(float(lon_first))
        self.ni = int(ni)
        self.nj = int(nj)
        self.dx = float(dx)
        self.dy = float(dy)
        self.i_sign = -1.0 if i_scans_negatively else 1.0
        # GRIB Lambert grids report the first point with increasing j toward
        # larger latitude, but the projection-space y metric decreases in that
        # direction for this formulation, so the sign is opposite of i_sign.
        self.j_sign = -1.0 if j_scans_positively else 1.0
        self.i_anchor = float(self.ni - 1 if i_scans_negatively else 0)
        self.j_anchor = float(0 if j_scans_positively else self.nj - 1)

        phi1 = np.radians(self.truelat1)
        phi2 = np.radians(self.truelat2)
        if abs(self.truelat1 - self.truelat2) < 1e-6:
            self.n = np.sin(phi1)
        else:
            self.n = (np.log(np.cos(phi1) / np.cos(phi2)) /
                      np.log(np.tan(np.pi / 4 + phi2 / 2) /
                             np.tan(np.pi / 4 + phi1 / 2)))

        self.F = np.cos(phi1) * np.tan(np.pi / 4 + phi1 / 2) ** self.n / self.n
        self.x_anchor, self.y_anchor = self._latlon_to_xy(self.lat_first, self.lon_first)

    def _latlon_to_xy(self, lat, lon):
        phi = np.radians(lat)
        lam = np.radians(normalize_longitude(lon) - self.stand_lon)
        rho = RE * self.F / np.tan(np.pi / 4 + phi / 2) ** self.n
        theta = self.n * lam
        x = rho * np.sin(theta)
        y = rho * np.cos(theta)
        return x, y

    def latlon_to_ij(self, lat, lon):
        x, y = self._latlon_to_xy(lat, lon)
        i = self.i_anchor + self.i_sign * (x - self.x_anchor) / self.dx
        j = self.j_anchor + self.j_sign * (y - self.y_anchor) / self.dy
        return i, j


def normalize_longitude(lon):
    lon_arr = np.asarray(lon, dtype=np.float64)
    return np.mod(lon_arr + 180.0, 360.0) - 180.0


def _safe_codes_get(msgid, key, default=None):
    import eccodes

    try:
        return eccodes.codes_get(msgid, key)
    except eccodes.KeyValueNotFoundError:
        return default


def _extract_grid_info(msgid):
    import eccodes

    grid_type = eccodes.codes_get(msgid, 'gridType')
    ni = int(_safe_codes_get(msgid, 'Ni', _safe_codes_get(msgid, 'Nx')))
    nj = int(_safe_codes_get(msgid, 'Nj', _safe_codes_get(msgid, 'Ny')))
    i_scans_negatively = int(_safe_codes_get(msgid, 'iScansNegatively', 0))
    j_scans_positively = int(_safe_codes_get(msgid, 'jScansPositively', 0))

    grid_info = dict(
        grid_type=grid_type,
        ni=ni,
        nj=nj,
        i_scans_negatively=i_scans_negatively,
        j_scans_positively=j_scans_positively,
    )

    if grid_type == 'regular_ll':
        grid_info.update(
            lat_first=float(eccodes.codes_get(msgid, 'latitudeOfFirstGridPointInDegrees')),
            lon_first=float(eccodes.codes_get(msgid, 'longitudeOfFirstGridPointInDegrees')),
            di=float(eccodes.codes_get(msgid, 'iDirectionIncrementInDegrees')),
            dj=float(eccodes.codes_get(msgid, 'jDirectionIncrementInDegrees')),
        )
        return grid_info

    if grid_type == 'lambert':
        stand_lon = _safe_codes_get(msgid, 'LoVInDegrees')
        if stand_lon is None:
            stand_lon = _safe_codes_get(msgid, 'orientationOfTheGridInDegrees')
        if stand_lon is None:
            raise RuntimeError("Lambert GRIB field is missing LoV/orientation metadata")

        grid_info.update(
            lat_first=float(eccodes.codes_get(msgid, 'latitudeOfFirstGridPointInDegrees')),
            lon_first=float(eccodes.codes_get(msgid, 'longitudeOfFirstGridPointInDegrees')),
            truelat1=float(eccodes.codes_get(msgid, 'Latin1InDegrees')),
            truelat2=float(eccodes.codes_get(msgid, 'Latin2InDegrees')),
            stand_lon=float(stand_lon),
            dx=float(eccodes.codes_get(msgid, 'DxInMetres')),
            dy=float(eccodes.codes_get(msgid, 'DyInMetres')),
        )
        return grid_info

    raise RuntimeError(f"Unsupported GRIB grid type: {grid_type}")


# ===================================================================
# GRIB2 reader (eccodes)
# ===================================================================
def read_grib_field(filepath, short_name, level, type_of_level, cache=None):
    """Read a single GRIB2 field.

    Parameters
    ----------
    filepath : str
    short_name : str   e.g. 'u', 'v', 't', 'q', 'gh', 'sp', 'orog'
    level : int         pressure level in hPa, or 0 for surface
    type_of_level : str e.g. 'isobaricInhPa', 'surface'
    cache : dict or None  pre-indexed message cache for fast access

    Returns
    -------
    data : ndarray (nj, ni)   values on the native GRIB grid
    grid_info : dict          source grid metadata
    """
    import eccodes

    if cache is not None:
        key = (short_name, type_of_level, level)
        if key in cache:
            offset = cache[key]
            with open(filepath, 'rb') as f:
                f.seek(offset)
                msgid = eccodes.codes_grib_new_from_file(f)
                if msgid is None:
                    raise RuntimeError("Failed to read cached message at offset %d" % offset)
                name = eccodes.codes_get(msgid, 'shortName')
                tol = eccodes.codes_get(msgid, 'typeOfLevel')
                lev = eccodes.codes_get(msgid, 'level')
                if name != short_name or tol != type_of_level or lev != level:
                    eccodes.codes_release(msgid)
                    msgid = None
                else:
                    vals = eccodes.codes_get_double_array(msgid, 'values')
                    grid_info = _extract_grid_info(msgid)
                    eccodes.codes_release(msgid)
                    return vals.reshape(grid_info['nj'], grid_info['ni']), grid_info

    # Sequential scan (fallback)
    with open(filepath, 'rb') as f:
        while True:
            msgid = eccodes.codes_grib_new_from_file(f)
            if msgid is None:
                break
            name = eccodes.codes_get(msgid, 'shortName')
            tol = eccodes.codes_get(msgid, 'typeOfLevel')
            lev = eccodes.codes_get(msgid, 'level')
            if name == short_name and tol == type_of_level and lev == level:
                vals = eccodes.codes_get_double_array(msgid, 'values')
                grid_info = _extract_grid_info(msgid)
                eccodes.codes_release(msgid)
                return vals.reshape(grid_info['nj'], grid_info['ni']), grid_info
            eccodes.codes_release(msgid)

    raise RuntimeError("Field not found: %s level=%d type=%s" %
                       (short_name, level, type_of_level))


def build_grib_index(filepath):
    """Build an offset index for every message in the GRIB file.

    Returns a dict mapping (shortName, typeOfLevel, level) -> file offset.
    """
    import eccodes

    index = {}
    with open(filepath, 'rb') as f:
        while True:
            msgid = eccodes.codes_grib_new_from_file(f)
            if msgid is None:
                break
            offset = eccodes.codes_get_message_offset(msgid)
            name = eccodes.codes_get(msgid, 'shortName')
            tol = eccodes.codes_get(msgid, 'typeOfLevel')
            lev = eccodes.codes_get(msgid, 'level')
            index[(name, tol, lev)] = offset
            eccodes.codes_release(msgid)
    return index


def _yyyymmdd_hhmm_to_unix(date_int, time_int):
    year = int(date_int) // 10000
    month = (int(date_int) // 100) % 100
    day = int(date_int) % 100
    hour = int(time_int) // 100
    minute = int(time_int) % 100
    dt = datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)
    return int(dt.timestamp())


def read_grib_time_metadata(filepath, cache=None):
    """Read valid/reference time metadata from the first available GRIB message."""
    import eccodes

    offset = None
    if cache:
        for preferred in [
            ('t', 'isobaricInhPa', 1000),
            ('gh', 'isobaricInhPa', 500),
            ('sp', 'surface', 0),
        ]:
            if preferred in cache:
                offset = cache[preferred]
                break

    with open(filepath, 'rb') as f:
        if offset is not None:
            f.seek(offset)
        msgid = eccodes.codes_grib_new_from_file(f)
        if msgid is None:
            raise RuntimeError(f"Failed to read GRIB message from {filepath}")

        try:
            data_date = int(eccodes.codes_get(msgid, 'dataDate'))
            data_time = int(eccodes.codes_get(msgid, 'dataTime'))
            validity_date = int(eccodes.codes_get(msgid, 'validityDate'))
            validity_time = int(eccodes.codes_get(msgid, 'validityTime'))
            forecast_time = int(eccodes.codes_get(msgid, 'forecastTime'))
        finally:
            eccodes.codes_release(msgid)

    return {
        "reference_date": data_date,
        "reference_time": data_time,
        "validity_date": validity_date,
        "validity_time": validity_time,
        "forecast_time": forecast_time,
        "reference_unix": _yyyymmdd_hhmm_to_unix(data_date, data_time),
        "validity_unix": _yyyymmdd_hhmm_to_unix(validity_date, validity_time),
    }


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _wsl_to_windows_path(path):
    return subprocess.check_output(
        ["wslpath", "-w", str(path)],
        text=True,
    ).strip()


def _windows_cmd(command):
    subprocess.run(["cmd.exe", "/c", command], check=True)


def _rust_writer_paths(repo_root):
    crate_dir = repo_root / "tools" / "rust-init-writer"
    exe_path = crate_dir / "target" / "release" / "gpuwm-init-writer.exe"
    return crate_dir, exe_path


def ensure_rust_init_writer(repo_root):
    crate_dir, exe_path = _rust_writer_paths(repo_root)
    source_mtime = max(
        path.stat().st_mtime
        for path in [crate_dir / "Cargo.toml", *sorted((crate_dir / "src").glob("*.rs"))]
    )

    if exe_path.exists() and exe_path.stat().st_mtime >= source_mtime:
        return exe_path

    crate_dir_win = _wsl_to_windows_path(crate_dir)
    print("Building Rust init writer...")
    _windows_cmd(f'cd /d {crate_dir_win} && cargo build --release')
    if not exe_path.exists():
        raise RuntimeError(f"Rust init writer build did not produce {exe_path}")
    return exe_path


def run_rust_init_writer(
    repo_root,
    output_file,
    nx,
    ny,
    nz,
    dx,
    ztop,
    truelat1,
    truelat2,
    stand_lon,
    ref_lat,
    ref_lon,
    terrain_following_init,
    time_meta,
    z_levels,
    p_levels_pa,
    u_plev,
    v_plev,
    t_plev,
    q_plev,
    gh_plev,
    orog_model,
):
    exe_path = ensure_rust_init_writer(repo_root)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="rust-init-", dir=repo_root / "data") as scratch_dir_str:
        scratch_dir = Path(scratch_dir_str)

        def write_array(name, array):
            path = scratch_dir / name
            np.ascontiguousarray(array, dtype=np.float64).tofile(path)
            return path

        z_levels_path = write_array("z_levels.bin", z_levels)
        p_levels_path = write_array("p_levels_pa.bin", p_levels_pa)
        u_path = write_array("u_plev.bin", u_plev)
        v_path = write_array("v_plev.bin", v_plev)
        t_path = write_array("t_plev.bin", t_plev)
        q_path = write_array("q_plev.bin", q_plev)
        gh_path = write_array("gh_plev.bin", gh_plev)
        orog_path = write_array("orog_model.bin", orog_model)

        manifest = {
            "nx": int(nx),
            "ny": int(ny),
            "nz": int(nz),
            "n_plev": int(len(p_levels_pa)),
            "dx": float(dx),
            "dy": float(dx),
            "ztop": float(ztop),
            "truelat1": float(truelat1),
            "truelat2": float(truelat2),
            "stand_lon": float(stand_lon),
            "ref_lat": float(ref_lat),
            "ref_lon": float(ref_lon),
            "terrain_following_init": bool(terrain_following_init),
            "validity_unix": int(time_meta["validity_unix"]),
            "reference_unix": int(time_meta["reference_unix"]),
            "forecast_hour": int(time_meta["forecast_time"]),
            "output": _wsl_to_windows_path(output_path),
            "scratch_dir": _wsl_to_windows_path(scratch_dir),
            "z_levels": _wsl_to_windows_path(z_levels_path),
            "p_levels_pa": _wsl_to_windows_path(p_levels_path),
            "u_plev": _wsl_to_windows_path(u_path),
            "v_plev": _wsl_to_windows_path(v_path),
            "t_plev": _wsl_to_windows_path(t_path),
            "q_plev": _wsl_to_windows_path(q_path),
            "gh_plev": _wsl_to_windows_path(gh_path),
            "orog_model": _wsl_to_windows_path(orog_path),
        }

        manifest_path = scratch_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        repo_root_win = _wsl_to_windows_path(repo_root)
        exe_win = _wsl_to_windows_path(exe_path)
        manifest_win = _wsl_to_windows_path(manifest_path)
        _windows_cmd(f'cd /d {repo_root_win} && {exe_win} --manifest {manifest_win}')


# ===================================================================
# Bilinear interpolation from source GRIB grid to model grid
# ===================================================================
def bilinear_interp_field(gfs_data, grid_info, model_lats, model_lons):
    """Interpolate a 2D GRIB field to model grid points.

    Parameters
    ----------
    gfs_data : ndarray (nj, ni)  field on the source GRIB grid
    grid_info : dict              grid metadata from read_grib_field
    model_lats : ndarray (ny, nx)
    model_lons : ndarray (ny, nx)

    Returns
    -------
    interp : ndarray (ny, nx)
    """
    ni = grid_info['ni']
    nj = grid_info['nj']
    grid_type = grid_info['grid_type']

    if grid_type == 'regular_ll':
        lat0 = grid_info['lat_first']
        lon0 = grid_info['lon_first']
        dlat = grid_info['dj']
        dlon = grid_info['di']

        lon360 = np.mod(model_lons, 360.0)
        lon0_360 = lon0 % 360.0

        if grid_info['i_scans_negatively']:
            fi = (lon0_360 - lon360) / dlon
        else:
            fi = (lon360 - lon0_360) / dlon

        if grid_info['j_scans_positively']:
            fj = (model_lats - lat0) / dlat
        else:
            fj = (lat0 - model_lats) / dlat

        i0 = np.floor(fi).astype(int)
        j0 = np.floor(fj).astype(int)
        di = fi - i0
        dj = fj - j0

        i0 = np.mod(i0, ni)
        i1 = np.mod(i0 + 1, ni)
        j0 = np.clip(j0, 0, nj - 1)
        j1 = np.clip(j0 + 1, 0, nj - 1)
    elif grid_type == 'lambert':
        projector = LambertSourceGrid(
            truelat1=grid_info['truelat1'],
            truelat2=grid_info['truelat2'],
            stand_lon=grid_info['stand_lon'],
            lat_first=grid_info['lat_first'],
            lon_first=grid_info['lon_first'],
            ni=ni,
            nj=nj,
            dx=grid_info['dx'],
            dy=grid_info['dy'],
            i_scans_negatively=bool(grid_info['i_scans_negatively']),
            j_scans_positively=bool(grid_info['j_scans_positively']),
        )
        fi, fj = projector.latlon_to_ij(model_lats, normalize_longitude(model_lons))
        i0 = np.floor(fi).astype(int)
        j0 = np.floor(fj).astype(int)
        di = fi - i0
        dj = fj - j0
        i0 = np.clip(i0, 0, ni - 1)
        i1 = np.clip(i0 + 1, 0, ni - 1)
        j0 = np.clip(j0, 0, nj - 1)
        j1 = np.clip(j0 + 1, 0, nj - 1)
    else:
        raise RuntimeError(f"Unsupported source grid type: {grid_type}")

    return ((1 - di) * (1 - dj) * gfs_data[j0, i0] +
            di       * (1 - dj) * gfs_data[j0, i1] +
            (1 - di) * dj       * gfs_data[j1, i0] +
            di       * dj       * gfs_data[j1, i1])


# ===================================================================
# Main processing
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Initialize GPU-WM from GRIB2 data (GFS or HRRR)')
    parser.add_argument('--gfs', '--pressure-grib', dest='pressure_grib',
                        default='data/gfs_latest.grib2',
                        help='Primary GRIB2 file for pressure-level fields')
    parser.add_argument('--surface-grib',
                        help='Optional secondary GRIB2 file for surface fields (e.g. HRRR wrfsfcf00)')
    parser.add_argument('--output', default='data/gfs_init.bin',
                        help='Output binary file')
    parser.add_argument('--nx', type=int, default=512,
                        help='Grid points in x')
    parser.add_argument('--ny', type=int, default=512,
                        help='Grid points in y')
    parser.add_argument('--nz', type=int, default=50,
                        help='Vertical levels')
    parser.add_argument('--dx', type=float, default=3000.0,
                        help='Grid spacing (m)')
    parser.add_argument('--truelat1', type=float, default=38.5,
                        help='Lambert true latitude 1')
    parser.add_argument('--truelat2', type=float, default=38.5,
                        help='Lambert true latitude 2')
    parser.add_argument('--stand-lon', type=float, default=-97.5,
                        help='Lambert standard longitude')
    parser.add_argument('--ref-lat', type=float, default=38.5,
                        help='Domain center latitude')
    parser.add_argument('--ref-lon', type=float, default=-97.5,
                        help='Domain center longitude')
    parser.add_argument('--ztop', type=float, default=25000.0,
                        help='Model top (m)')
    parser.add_argument('--plot', default='plots/gfs_init_conus.png',
                        help='Path for CONUS plot')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating the initialization PNG')
    parser.add_argument('--python-vertical', action='store_true',
                        help='Force the legacy Python vertical interpolation path instead of the Rust helper')
    parser.add_argument('--terrain-following-init', action='store_true',
                        help='Interpolate columns directly onto terrain-following model levels')
    parser.add_argument('--stretched-eta', action='store_true',
                        help='Use experimental stretched reference model levels instead of uniform spacing')
    args = parser.parse_args()

    nx, ny, nz = args.nx, args.ny, args.nz
    dx = args.dx
    ztop = args.ztop
    pressure_file = args.pressure_grib
    surface_file = args.surface_grib or pressure_file
    source_label = "HRRR-like split GRIB" if args.surface_grib else "single-file GRIB"

    print("=" * 60)
    print("  GPU-WM: GRIB Initialization")
    print("=" * 60)
    print("  Source   : %s" % source_label)
    print("  Pressure : %s" % pressure_file)
    if surface_file != pressure_file:
        print("  Surface  : %s" % surface_file)
    print("  Grid     : %d x %d x %d" % (nx, ny, nz))
    print("  dx       : %.0f m" % dx)
    print("  ztop     : %.0f m" % ztop)
    print("  Lambert  : truelat1=%.2f truelat2=%.2f stand_lon=%.2f ref=(%.2f, %.2f)" %
          (args.truelat1, args.truelat2, args.stand_lon, args.ref_lat, args.ref_lon))
    print()

    if not os.path.exists(pressure_file):
        print("ERROR: pressure GRIB file not found: %s" % pressure_file)
        sys.exit(1)
    if not os.path.exists(surface_file):
        print("ERROR: surface GRIB file not found: %s" % surface_file)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Setup Lambert Conformal projection and compute lat/lon grid
    # ------------------------------------------------------------------
    print("Setting up Lambert Conformal projection...")
    proj = LambertConformal(
        truelat1=args.truelat1,
        truelat2=args.truelat2,
        stand_lon=args.stand_lon,
        ref_lat=args.ref_lat,
        ref_lon=args.ref_lon,
        nx=nx,
        ny=ny,
        dx=dx,
    )

    ii, jj = np.meshgrid(np.arange(nx, dtype=float),
                          np.arange(ny, dtype=float))
    model_lats, model_lons = proj.ij_to_latlon(ii, jj)

    print("  Domain lat range: %.2f to %.2f" % (model_lats.min(), model_lats.max()))
    print("  Domain lon range: %.2f to %.2f" % (model_lons.min(), model_lons.max()))
    print()

    # ------------------------------------------------------------------
    # 2. Build GRIB indexes for fast field access
    # ------------------------------------------------------------------
    t0 = time_mod.time()
    print("Indexing pressure GRIB2 file...")
    pressure_cache = build_grib_index(pressure_file)
    print("  Indexed %d messages in %.1f s" % (len(pressure_cache), time_mod.time() - t0))
    surface_cache = pressure_cache
    if surface_file != pressure_file:
        t_surface = time_mod.time()
        print("Indexing surface GRIB2 file...")
        surface_cache = build_grib_index(surface_file)
        print("  Indexed %d messages in %.1f s" % (len(surface_cache), time_mod.time() - t_surface))

    time_meta = read_grib_time_metadata(pressure_file, pressure_cache)
    if surface_file != pressure_file:
        surface_time_meta = read_grib_time_metadata(surface_file, surface_cache)
        if (surface_time_meta["reference_unix"] != time_meta["reference_unix"] or
                surface_time_meta["validity_unix"] != time_meta["validity_unix"]):
            raise RuntimeError(
                "Pressure and surface GRIB files do not share the same reference/valid time"
            )
    ref_dt = datetime.fromtimestamp(time_meta["reference_unix"], tz=timezone.utc)
    valid_dt = datetime.fromtimestamp(time_meta["validity_unix"], tz=timezone.utc)
    print("  Reference time: %s" % ref_dt.strftime("%Y-%m-%d %H:%M UTC"))
    print("  Valid time    : %s" % valid_dt.strftime("%Y-%m-%d %H:%M UTC"))
    print("  Forecast hour : %d" % time_meta["forecast_time"])
    print()

    # ------------------------------------------------------------------
    # 3. Pressure levels (hPa) -- use only those available
    # ------------------------------------------------------------------
    all_plevels = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650,
                   600, 550, 500, 450, 400, 350, 300, 250, 200, 150,
                   100, 70, 50, 40, 30, 20, 15, 10, 7, 5, 3, 2, 1]

    # Filter to levels that actually exist for all needed fields
    plevels = []
    for p in all_plevels:
        if (('gh', 'isobaricInhPa', p) in pressure_cache and
            ('t', 'isobaricInhPa', p) in pressure_cache and
            ('u', 'isobaricInhPa', p) in pressure_cache and
            ('v', 'isobaricInhPa', p) in pressure_cache):
            plevels.append(p)

    n_plev = len(plevels)
    print("Using %d pressure levels:" % n_plev)
    print("  ", plevels)
    print()

    # Pressure in Pa (descending: 1000 hPa first, 1 hPa last)
    p_levels_pa = np.array(plevels, dtype=np.float64) * 100.0

    # ------------------------------------------------------------------
    # 4. Read and horizontally interpolate all source GRIB fields
    # ------------------------------------------------------------------
    print("Reading and interpolating GRIB fields...")

    # Surface fields
    print("  Surface pressure (sp)...")
    sp_gfs, gi = read_grib_field(surface_file, 'sp', 0, 'surface', surface_cache)
    sp_model = bilinear_interp_field(sp_gfs, gi, model_lats, model_lons)

    print("  Orography (orog)...")
    orog_gfs, gi = read_grib_field(surface_file, 'orog', 0, 'surface', surface_cache)
    orog_model = bilinear_interp_field(orog_gfs, gi, model_lats, model_lons)

    # 3D fields on pressure levels
    u_plev = np.zeros((n_plev, ny, nx), dtype=np.float64)
    v_plev = np.zeros((n_plev, ny, nx), dtype=np.float64)
    t_plev = np.zeros((n_plev, ny, nx), dtype=np.float64)
    q_plev = np.zeros((n_plev, ny, nx), dtype=np.float64)
    gh_plev = np.zeros((n_plev, ny, nx), dtype=np.float64)

    for ip, plev in enumerate(plevels):
        if (ip + 1) % 5 == 0 or ip == 0:
            print("  Pressure level %d hPa (%d/%d)..." % (plev, ip + 1, n_plev))

        data, gi = read_grib_field(pressure_file, 'u', plev, 'isobaricInhPa', pressure_cache)
        u_plev[ip] = bilinear_interp_field(data, gi, model_lats, model_lons)

        data, gi = read_grib_field(pressure_file, 'v', plev, 'isobaricInhPa', pressure_cache)
        v_plev[ip] = bilinear_interp_field(data, gi, model_lats, model_lons)

        data, gi = read_grib_field(pressure_file, 't', plev, 'isobaricInhPa', pressure_cache)
        t_plev[ip] = bilinear_interp_field(data, gi, model_lats, model_lons)

        # Specific humidity may not be available at highest levels -- use 0
        if ('q', 'isobaricInhPa', plev) in pressure_cache:
            data, gi = read_grib_field(pressure_file, 'q', plev, 'isobaricInhPa', pressure_cache)
            q_plev[ip] = bilinear_interp_field(data, gi, model_lats, model_lons)
        else:
            q_plev[ip] = 0.0

        data, gi = read_grib_field(pressure_file, 'gh', plev, 'isobaricInhPa', pressure_cache)
        gh_plev[ip] = bilinear_interp_field(data, gi, model_lats, model_lons)

    print()

    # ------------------------------------------------------------------
    # 5. Define model eta levels.
    # ------------------------------------------------------------------
    if args.stretched_eta:
        # Experimental stretched coordinate for dycore work. Keep this opt-in
        # until the real-data path is fully stable with the new spacing.
        z_levels, eta_m, eta_w = build_stretched_vertical_levels(nz, ztop)
        dz_levels = np.diff(eta_w) * ztop
        level_mode = "stretched"
    else:
        dz = ztop / nz
        z_levels = np.array([(k + 0.5) * dz for k in range(nz)], dtype=np.float64)
        eta_m = z_levels / ztop
        dz_levels = np.full(nz, dz, dtype=np.float64)
        level_mode = "uniform"

    print("Model height levels:")
    print("  mode   = %s" % level_mode)
    print("  dz_min = %.1f m, dz_max = %.1f m" % (dz_levels.min(), dz_levels.max()))
    print("  z[0] = %.1f m, z[-1] = %.1f m" % (z_levels[0], z_levels[-1]))
    print()

    # ------------------------------------------------------------------
    # 6. Vertical interpolation: pressure levels -> terrain-following model
    #    levels. The target geometric height varies by column:
    #      z(x,y,eta) = h(x,y) + eta * (ztop - h(x,y))
    # ------------------------------------------------------------------
    interp_mode = "terrain-following" if args.terrain_following_init else "flat-height"
    output_file = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    use_rust_vertical = args.no_plot and not args.python_vertical
    rust_written = False
    u_model = None
    v_model = None
    theta_model = None
    qv_model = None
    p_model = None

    if use_rust_vertical:
        print(f"Vertical interpolation ({interp_mode}, hypsometric, Rust helper)...")
        try:
            run_rust_init_writer(
                repo_root=_repo_root(),
                output_file=output_file,
                nx=nx,
                ny=ny,
                nz=nz,
                dx=dx,
                ztop=ztop,
                truelat1=args.truelat1,
                truelat2=args.truelat2,
                stand_lon=args.stand_lon,
                ref_lat=args.ref_lat,
                ref_lon=args.ref_lon,
                terrain_following_init=args.terrain_following_init,
                time_meta=time_meta,
                z_levels=z_levels,
                p_levels_pa=p_levels_pa,
                u_plev=u_plev,
                v_plev=v_plev,
                t_plev=t_plev,
                q_plev=q_plev,
                gh_plev=gh_plev,
                orog_model=orog_model,
            )
            print("  sp    : min=%.0f  max=%.0f Pa" % (sp_model.min(), sp_model.max()))
            print()
            rust_written = True
        except Exception as exc:
            print(f"  Rust helper failed: {exc}")
            print("  Falling back to Python vertical interpolation.")
            print()

    if not rust_written:
        print(f"Vertical interpolation ({interp_mode}, hypsometric, log-p)...")

        # Convert source geopotential height from geopotential meters to
        # geometric height (they are nearly the same, but gh is in gpm)
        # gh_plev is already in meters (geopotential height ~ geometric height
        # for our purposes)

        u_model = np.zeros((nz, ny, nx), dtype=np.float64)
        v_model = np.zeros((nz, ny, nx), dtype=np.float64)
        theta_model = np.zeros((nz, ny, nx), dtype=np.float64)
        qv_model = np.zeros((nz, ny, nx), dtype=np.float64)
        p_model = np.zeros((nz, ny, nx), dtype=np.float64)

        # Convert T -> potential temperature on pressure levels
        # theta = T * (P0/p)^kappa
        theta_plev = np.zeros_like(t_plev)
        for ip in range(n_plev):
            theta_plev[ip] = t_plev[ip] * (P0 / p_levels_pa[ip]) ** KAPPA

        # Convert specific humidity to mixing ratio on pressure levels
        # qv = q / (1 - q)
        qv_plev = q_plev / np.maximum(1.0 - q_plev, 1e-12)
        qv_plev = np.maximum(qv_plev, 0.0)

        # For each model grid column, interpolate vertically
        # gh_plev gives height at each pressure level for each column
        # Pressure levels are ordered: index 0 = 1000 hPa (low), index -1 = 1 hPa (high)
        # gh_plev[ip] increases with ip (height increases as pressure decreases)
        log_p = np.log(p_levels_pa)

        for j in range(ny):
            for i in range(nx):
                h_col = gh_plev[:, j, i]

                for k in range(nz):
                    if args.terrain_following_init:
                        terrain = min(orog_model[j, i], ztop - 1.0)
                        column_depth = max(ztop - terrain, 1.0)
                        z_target = terrain + eta_m[k] * column_depth
                    else:
                        z_target = z_levels[k]

                    if z_target <= h_col[0]:
                        idx_lo, idx_hi = 0, 1
                    elif z_target >= h_col[-1]:
                        idx_lo, idx_hi = n_plev - 2, n_plev - 1
                    else:
                        idx_lo = 0
                        for ip in range(n_plev - 1):
                            if h_col[ip] <= z_target <= h_col[ip + 1]:
                                idx_lo = ip
                                break
                        idx_hi = idx_lo + 1

                    h_lo = h_col[idx_lo]
                    h_hi = h_col[idx_hi]
                    if abs(h_hi - h_lo) < 1e-3:
                        w_hi = 0.5
                    else:
                        w_hi = (z_target - h_lo) / (h_hi - h_lo)
                    w_hi = np.clip(w_hi, 0.0, 1.0)
                    w_lo = 1.0 - w_hi

                    log_p_target = w_lo * log_p[idx_lo] + w_hi * log_p[idx_hi]
                    p_model[k, j, i] = np.exp(log_p_target)
                    u_model[k, j, i] = w_lo * u_plev[idx_lo, j, i] + w_hi * u_plev[idx_hi, j, i]
                    v_model[k, j, i] = w_lo * v_plev[idx_lo, j, i] + w_hi * v_plev[idx_hi, j, i]
                    theta_model[k, j, i] = (w_lo * theta_plev[idx_lo, j, i] +
                                            w_hi * theta_plev[idx_hi, j, i])
                    qv_model[k, j, i] = max(
                        0.0,
                        w_lo * qv_plev[idx_lo, j, i] + w_hi * qv_plev[idx_hi, j, i],
                    )

        print("  Done.")
        print()

        print("Field statistics after interpolation:")
        print("  u     : min=%.2f  max=%.2f  mean=%.2f m/s" %
              (u_model.min(), u_model.max(), u_model.mean()))
        print("  v     : min=%.2f  max=%.2f  mean=%.2f m/s" %
              (v_model.min(), v_model.max(), v_model.mean()))
        print("  theta : min=%.2f  max=%.2f  mean=%.2f K" %
              (theta_model.min(), theta_model.max(), theta_model.mean()))
        print("  qv    : min=%.6f  max=%.6f  mean=%.6f kg/kg" %
              (qv_model.min(), qv_model.max(), qv_model.mean()))
        print("  p     : min=%.0f  max=%.0f  mean=%.0f Pa" %
              (p_model.min(), p_model.max(), p_model.mean()))
        print("  sp    : min=%.0f  max=%.0f Pa" % (sp_model.min(), sp_model.max()))
        print("  orog  : min=%.0f  max=%.0f m" % (orog_model.min(), orog_model.max()))
        print()

        # ------------------------------------------------------------------
        # 7. Write binary output file
        # ------------------------------------------------------------------
        print("Writing binary file: %s" % output_file)

        w_model = np.zeros((nz, ny, nx), dtype=np.float64)
        qc_model = np.zeros((nz, ny, nx), dtype=np.float64)
        qr_model = np.zeros((nz, ny, nx), dtype=np.float64)

        with open(output_file, 'wb') as f:
            f.write(struct.pack('iii', nx, ny, nz))
            f.write(struct.pack('ddd', dx, dx, ztop))
            z_levels.tofile(f)
            u_model.tofile(f)
            v_model.tofile(f)
            w_model.tofile(f)
            theta_model.tofile(f)
            qv_model.tofile(f)
            qc_model.tofile(f)
            qr_model.tofile(f)
            f.write(PROJ_MAGIC)
            f.write(struct.pack('ddddd',
                                args.truelat1, args.truelat2,
                                args.stand_lon, args.ref_lat, args.ref_lon))
            f.write(PRES_MAGIC)
            p_model.tofile(f)
            f.write(TERRAIN_MAGIC)
            orog_model.astype(np.float64).tofile(f)
            f.write(INIT_MODE_MAGIC)
            f.write(struct.pack('ii', 1 if args.terrain_following_init else 0, 0))
            f.write(TIME_MAGIC)
            f.write(struct.pack('qqii',
                                time_meta["validity_unix"],
                                time_meta["reference_unix"],
                                time_meta["forecast_time"],
                                0))

    file_size = os.path.getsize(output_file)
    expected = (
        3 * 4 + 3 * 8 + nz * 8 + 7 * nz * ny * nx * 8 +
        len(PROJ_MAGIC) + 5 * 8 +
        len(PRES_MAGIC) + nz * ny * nx * 8 +
        len(TERRAIN_MAGIC) + ny * nx * 8 +
        len(INIT_MODE_MAGIC) + 2 * 4 +
        len(TIME_MAGIC) + 2 * 8 + 2 * 4
    )
    print("  File size: %.1f MB (expected: %.1f MB)" %
          (file_size / 1e6, expected / 1e6))
    print("  Format: header(3xi32 + 3xf64) + z_levels(%dxf64) + 7 x [%d,%d,%d] f64 + proj trailer + pressure trailer + terrain trailer + init-mode trailer + time trailer" %
          (nz, nz, ny, nx))
    print()

    if not args.no_plot:
        # ------------------------------------------------------------------
        # 8. Generate initialization visualization
        # ------------------------------------------------------------------
        print("Generating initialization visualization...")

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        plot_dir = os.path.dirname(args.plot)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)

        # Find the model level closest to 850 hPa
        if 850 in plevels:
            ip_850 = plevels.index(850)
            z_850_mean = np.mean(gh_plev[ip_850])
        else:
            z_850_mean = 1500.0
        k_850 = np.argmin(np.abs(z_levels - z_850_mean))
        print("  850 hPa approx at model level k=%d (z=%.0f m, target=%.0f m)" %
              (k_850, z_levels[k_850], z_850_mean))

        if 500 in plevels:
            ip_500 = plevels.index(500)
            z_500_mean = np.mean(gh_plev[ip_500])
        else:
            z_500_mean = 5500.0
        k_500 = np.argmin(np.abs(z_levels - z_500_mean))

        k_sfc = 1

        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle('GPU-WM: GRIB Initialization on Lambert Conformal Regional Grid\n'
                     '%d x %d x %d  dx=%.0f m' % (nx, ny, nz, dx),
                     fontsize=15, fontweight='bold')

        x_km = np.arange(nx) * dx / 1000.0
        y_km = np.arange(ny) * dx / 1000.0

        ax = axes[0, 0]
        wspd = np.sqrt(u_model[k_850] ** 2 + v_model[k_850] ** 2)
        c = ax.pcolormesh(x_km, y_km, wspd, cmap='YlOrRd', shading='auto',
                          vmin=0, vmax=max(30, np.percentile(wspd, 99)))
        plt.colorbar(c, ax=ax, label='m/s', shrink=0.8)
        skip = max(1, nx // 25)
        ax.barbs(x_km[::skip], y_km[::skip],
                 u_model[k_850, ::skip, ::skip],
                 v_model[k_850, ::skip, ::skip],
                 length=5, linewidth=0.4, color='k')
        ax.set_title('850 hPa Wind Speed & Barbs (z=%.0f m)' % z_levels[k_850])
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_aspect('equal')

        ax = axes[0, 1]
        c = ax.pcolormesh(x_km, y_km, theta_model[k_850], cmap='RdYlBu_r',
                          shading='auto')
        plt.colorbar(c, ax=ax, label='K', shrink=0.8)
        ax.set_title('850 hPa Potential Temperature (z=%.0f m)' % z_levels[k_850])
        ax.set_xlabel('x (km)')
        ax.set_aspect('equal')

        ax = axes[0, 2]
        c = ax.pcolormesh(x_km, y_km, qv_model[k_850] * 1000, cmap='YlGnBu',
                          shading='auto')
        plt.colorbar(c, ax=ax, label='g/kg', shrink=0.8)
        ax.set_title('850 hPa Mixing Ratio (z=%.0f m)' % z_levels[k_850])
        ax.set_xlabel('x (km)')
        ax.set_aspect('equal')

        ax = axes[1, 0]
        c = ax.pcolormesh(x_km, y_km, sp_model / 100.0, cmap='viridis',
                          shading='auto')
        plt.colorbar(c, ax=ax, label='hPa', shrink=0.8)
        ax.set_title('Surface Pressure')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_aspect('equal')

        ax = axes[1, 1]
        wspd_500 = np.sqrt(u_model[k_500] ** 2 + v_model[k_500] ** 2)
        c = ax.pcolormesh(x_km, y_km, wspd_500, cmap='cool', shading='auto')
        plt.colorbar(c, ax=ax, label='m/s', shrink=0.8)
        skip500 = max(1, nx // 20)
        ax.barbs(x_km[::skip500], y_km[::skip500],
                 u_model[k_500, ::skip500, ::skip500],
                 v_model[k_500, ::skip500, ::skip500],
                 length=5, linewidth=0.4, color='k')
        ax.set_title('500 hPa Wind Speed (z=%.0f m)' % z_levels[k_500])
        ax.set_xlabel('x (km)')
        ax.set_aspect('equal')

        ax = axes[1, 2]
        jc = ny // 2
        theta_xs = theta_model[:, jc, :]
        c = ax.pcolormesh(x_km, z_levels / 1000.0, theta_xs,
                          cmap='RdYlBu_r', shading='auto')
        plt.colorbar(c, ax=ax, label='K', shrink=0.8)
        wspd_xs = np.sqrt(u_model[:, jc, :] ** 2 + v_model[:, jc, :] ** 2)
        cs = ax.contour(x_km, z_levels / 1000.0, wspd_xs,
                        levels=[10, 20, 30, 40, 50], colors='k', linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')
        ax.set_title('Cross Section (y=%d): theta + wind contours' % jc)
        ax.set_xlabel('x (km)')
        ax.set_ylabel('Height (km)')
        ax.set_ylim(0, ztop / 1000.0)

        plt.tight_layout()
        plt.savefig(args.plot, dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: %s" % args.plot)
        print()

    print("=" * 60)
    print("  Initialization complete!")
    print("  Binary file: %s (%.1f MB)" % (output_file, file_size / 1e6))
    print("  Plot:        %s" % ("skipped" if args.no_plot else args.plot))
    print("=" * 60)


if __name__ == '__main__':
    main()
