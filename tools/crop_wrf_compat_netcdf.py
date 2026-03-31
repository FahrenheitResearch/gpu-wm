#!/usr/bin/env python3
"""Crop a GPU-WM/WRF-compatible NetCDF file to a smaller x/y window.

Preserves both the native GPU-WM dimensions (`x`, `y`) and the WRF-style
dimensions (`west_east`, `south_north`, and staggered variants).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from netCDF4 import Dataset


X_DIMS = {"x", "west_east"}
Y_DIMS = {"y", "south_north"}
X_STAG_DIMS = {"west_east_stag"}
Y_STAG_DIMS = {"south_north_stag"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop a GPU-WM/WRF-compatible NetCDF spatially")
    parser.add_argument("--input", required=True, help="Input NetCDF path")
    parser.add_argument("--output", required=True, help="Output cropped NetCDF path")
    parser.add_argument("--i0", type=int, help="Starting x index (inclusive)")
    parser.add_argument("--i1", type=int, help="Ending x index (exclusive)")
    parser.add_argument("--j0", type=int, help="Starting y index (inclusive)")
    parser.add_argument("--j1", type=int, help="Ending y index (exclusive)")
    parser.add_argument("--crop-nx", type=int, help="Centered crop width")
    parser.add_argument("--crop-ny", type=int, help="Centered crop height")
    return parser.parse_args()


def resolve_window(nx: int, ny: int, args: argparse.Namespace) -> tuple[int, int, int, int]:
    if args.crop_nx is not None or args.crop_ny is not None:
        crop_nx = args.crop_nx or nx
        crop_ny = args.crop_ny or ny
        i0 = max((nx - crop_nx) // 2, 0)
        j0 = max((ny - crop_ny) // 2, 0)
        i1 = min(i0 + crop_nx, nx)
        j1 = min(j0 + crop_ny, ny)
        return i0, i1, j0, j1

    if None in (args.i0, args.i1, args.j0, args.j1):
        raise SystemExit("Specify either --crop-nx/--crop-ny or explicit --i0/--i1/--j0/--j1")
    return args.i0, args.i1, args.j0, args.j1


def build_dim_sizes(src: Dataset, i0: int, i1: int, j0: int, j1: int) -> Dict[str, int]:
    dim_sizes: Dict[str, int] = {}
    for name, dim in src.dimensions.items():
        if name in X_DIMS:
            dim_sizes[name] = i1 - i0
        elif name in Y_DIMS:
            dim_sizes[name] = j1 - j0
        elif name in X_STAG_DIMS:
            dim_sizes[name] = (i1 - i0) + 1
        elif name in Y_STAG_DIMS:
            dim_sizes[name] = (j1 - j0) + 1
        else:
            dim_sizes[name] = len(dim)
    return dim_sizes


def dim_slice(name: str, i0: int, i1: int, j0: int, j1: int):
    if name in X_DIMS:
        return slice(i0, i1)
    if name in Y_DIMS:
        return slice(j0, j1)
    if name in X_STAG_DIMS:
        return slice(i0, i1 + 1)
    if name in Y_STAG_DIMS:
        return slice(j0, j1 + 1)
    return slice(None)


def main() -> int:
    args = parse_args()
    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(in_path, "r") as src, Dataset(out_path, "w", format="NETCDF4") as dst:
        nx = len(src.dimensions["x"])
        ny = len(src.dimensions["y"])
        i0, i1, j0, j1 = resolve_window(nx, ny, args)
        if not (0 <= i0 < i1 <= nx and 0 <= j0 < j1 <= ny):
            raise SystemExit(f"Invalid crop window: x=[{i0},{i1}) y=[{j0},{j1}) for grid {nx}x{ny}")

        dim_sizes = build_dim_sizes(src, i0, i1, j0, j1)
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else dim_sizes[name])

        for attr in src.ncattrs():
            dst.setncattr(attr, src.getncattr(attr))

        for name, var in src.variables.items():
            fill_value = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
            out_var = dst.createVariable(name, var.datatype, var.dimensions, zlib=True, complevel=4, fill_value=fill_value)
            for attr in var.ncattrs():
                if attr == "_FillValue":
                    continue
                out_var.setncattr(attr, var.getncattr(attr))

            slices = tuple(dim_slice(dim_name, i0, i1, j0, j1) for dim_name in var.dimensions)
            out_var[:] = var[slices]

    print(f"Cropped {in_path} -> {out_path}")
    print(f"Window: x=[{i0},{i1}) y=[{j0},{j1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
