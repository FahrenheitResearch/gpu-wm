#!/usr/bin/env python3
"""Render WRF-style native products from a GPU-WM NetCDF file.

Intended to be run with the Windows Python environment that already has
`wrf-rust` installed, e.g.:

    py -3.13 tools\\render_wrf_products.py --input output\\gpuwm_000001.nc
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np


DEFAULT_PRODUCTS = [
    "field:slp",
    "850mb_temperature_height_winds",
    "500mb_temperature_height_winds",
    "850mb_wind_speed_height",
    "500mb_wind_speed_height",
    "850mb_rh_height_winds",
    "500mb_rh_height_winds",
]

CUSTOM_SURFACE_FIELDS = {"t2", "dp2m", "rh2m"}


def _bad_sample(value: float, fill_value: float | None) -> bool:
    if value != value:
        return True
    if fill_value is not None and value == fill_value:
        return True
    return abs(value) > 1.0e20


def _normalize_sample(sample) -> float | None:
    if np.ma.is_masked(sample):
        return None
    try:
        return float(sample)
    except (TypeError, ValueError):
        return None


def _required_vars_for_products(products: list[str]) -> dict[str, tuple[int, ...]]:
    checks: dict[str, tuple[int, ...]] = {
        "XLAT": (),
        "XLONG": (),
    }
    for product in products:
        if product.startswith("field:"):
            field_name = product.split(":", 1)[1].lower()
            if field_name == "t2":
                checks["T2"] = ()
                continue
            if field_name in {"dp2m", "rh2m"}:
                checks["T2"] = ()
                checks["Q2"] = ()
                checks["PSFC"] = ()
                continue
            if field_name == "slp":
                checks["P"] = (0, 0)
                checks["PB"] = (0, 0)
                checks["T"] = (0, 0)
                checks["PSFC"] = ()
                continue

        checks["P"] = (0, 0)
        checks["PB"] = (0, 0)
        checks["T"] = (0, 0)
        checks["PSFC"] = ()
    return checks


def preflight_netcdf(nc_path: Path, products: list[str]) -> None:
    """Reject obviously incomplete files before handing them to wrf-rust.

    A killed write can leave required variables present but filled entirely with
    `_FillValue`, which causes downstream plots/stats to return nonsense rather
    than failing clearly.
    """

    try:
        from netCDF4 import Dataset
    except ImportError:
        return

    checks = _required_vars_for_products(products)
    with Dataset(nc_path, "r") as ds:
        missing = [name for name in checks if name not in ds.variables]
        if missing:
            raise RuntimeError(f"missing required variables: {', '.join(missing)}")

        nx = len(ds.dimensions["west_east"])
        ny = len(ds.dimensions["south_north"])
        sample_i = sorted({0, max(nx // 2, 0), max(nx - 1, 0)})
        sample_j = sorted({0, max(ny // 2, 0), max(ny - 1, 0)})

        for name, leading_index in checks.items():
            var = ds.variables[name]
            fill_value = getattr(var, "_FillValue", None)
            samples = []
            for j in sample_j:
                for i in sample_i:
                    if var.ndim == 4:
                        index = tuple(leading_index) + (j, i)
                    elif var.ndim == 3:
                        index = (0, j, i)
                    elif var.ndim == 2:
                        index = (j, i)
                    else:
                        raise RuntimeError(f"unsupported dimensionality for {name}: {var.ndim}")
                    samples.append(_normalize_sample(var[index]))
            if all(value is None or _bad_sample(value, fill_value) for value in samples):
                raise RuntimeError(
                    f"{name} appears uninitialized or filled with missing values; "
                    "the NetCDF write may be incomplete"
                )


def _calc_surface_dewpoint_c(t_k: np.ndarray, qv: np.ndarray, psfc: np.ndarray) -> np.ndarray:
    eps = 0.622
    vapor_pressure = np.clip(psfc * qv / np.maximum(eps + qv, 1.0e-12), 1.0, None)
    ln_ratio = np.log(vapor_pressure / 611.2)
    td_c = 243.5 * ln_ratio / np.maximum(17.67 - ln_ratio, 1.0e-6)
    return td_c


def _calc_surface_rh_pct(t_k: np.ndarray, qv: np.ndarray, psfc: np.ndarray) -> np.ndarray:
    t_c = t_k - 273.15
    es = 611.2 * np.exp((17.67 * t_c) / np.maximum(t_c + 243.5, 1.0e-6))
    qvs = 0.622 * es / np.maximum(psfc - es, 1.0)
    rh = 100.0 * qv / np.maximum(qvs, 1.0e-12)
    return np.clip(rh, 0.0, 100.0)


def _render_custom_surface_field(
    nc_path: Path,
    field_name: str,
    out_path: Path,
    width: int,
    height: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from netCDF4 import Dataset

    with Dataset(nc_path, "r") as ds:
        lats = np.array(ds.variables["XLAT"][0], dtype=np.float32)
        lons = np.array(ds.variables["XLONG"][0], dtype=np.float32)
        t2 = np.array(ds.variables["T2"][0], dtype=np.float32)

        if field_name == "t2":
            data = t2 - 273.15
            cmap = "turbo"
            label = "2-m temperature (C)"
            title = "2-m temperature (C)"
            vmin = float(np.nanpercentile(data, 2))
            vmax = float(np.nanpercentile(data, 98))
        else:
            q2 = np.array(ds.variables["Q2"][0], dtype=np.float32)
            psfc = np.array(ds.variables["PSFC"][0], dtype=np.float32)
            if field_name == "dp2m":
                data = _calc_surface_dewpoint_c(t2, q2, psfc)
                cmap = "terrain"
                label = "2-m dewpoint (C)"
                title = "2-m dewpoint temperature (degC)"
                vmin = float(np.nanpercentile(data, 2))
                vmax = float(np.nanpercentile(data, 98))
            elif field_name == "rh2m":
                data = _calc_surface_rh_pct(t2, q2, psfc)
                cmap = "YlGnBu"
                label = "2-m relative humidity (%)"
                title = "2-m relative humidity (%)"
                vmin = 0.0
                vmax = 100.0
            else:
                raise ValueError(f"unsupported custom surface field: {field_name}")

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise RuntimeError(f"{field_name} has no finite values to render")
    if abs(vmax - vmin) < 1.0e-6:
        center = float(np.nanmean(finite))
        vmin = center - 1.0
        vmax = center + 1.0

    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    mesh = ax.pcolormesh(lons, lats, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.88)
    cbar.set_label(label)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.nanmin(lons)), float(np.nanmax(lons)))
    ax.set_ylim(float(np.nanmin(lats)), float(np.nanmax(lats)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _product_uses_wrf(product: str) -> bool:
    if not product.startswith("field:"):
        return True
    field_name = product.split(":", 1)[1].lower()
    return field_name not in CUSTOM_SURFACE_FIELDS


def main() -> int:
    parser = argparse.ArgumentParser(description="Render native wrf-rust products from a GPU-WM NetCDF file")
    parser.add_argument("--input", required=True, help="Path to GPU-WM NetCDF file")
    parser.add_argument("--output-dir", help="Directory for output PNGs")
    parser.add_argument("--width", type=int, default=1400, help="PNG width")
    parser.add_argument("--height", type=int, default=1000, help="PNG height")
    parser.add_argument("--no-borders", action="store_true", help="Disable map border overlays")
    parser.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS, help="Product names to render")
    args = parser.parse_args()

    need_wrf = any(_product_uses_wrf(product) for product in args.products)
    wrf = None
    if need_wrf:
        try:
            import wrf as wrf_mod
            wrf = wrf_mod
        except ImportError:
            print("ERROR: wrf-rust Python package is not installed in this interpreter", file=sys.stderr)
            return 1

    nc_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else nc_path.parent / "wrf_products"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        preflight_netcdf(nc_path, args.products)
    except Exception as exc:
        print(f"ERROR: NetCDF preflight failed for {nc_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Rendering {len(args.products)} products from {nc_path}")
    if need_wrf and wrf is not None:
        wf_info = wrf.WrfFile(str(nc_path))
        print(f"Grid: {wf_info.nx} x {wf_info.ny} x {wf_info.nz}")
        print(f"Times: {wf_info.times()}")
        del wf_info
        gc.collect()

    written: list[Path] = []
    failed = False

    for product in args.products:
        slug = product.lower().replace(":", "_").replace(" ", "_").replace("-", "_")
        out_path = out_dir / f"{slug}.png"
        try:
            if product.startswith("field:"):
                field_name = product.split(":", 1)[1]
                if field_name.lower() in CUSTOM_SURFACE_FIELDS:
                    _render_custom_surface_field(
                        nc_path,
                        field_name.lower(),
                        out_path,
                        width=args.width,
                        height=args.height,
                    )
                else:
                    wf = wrf.WrfFile(str(nc_path))
                    out_path.write_bytes(
                        wrf.render_field(
                            wf,
                            field_name,
                            width=args.width,
                            height=args.height,
                            borders=not args.no_borders,
                        )
                    )
                    del wf
                    gc.collect()
            else:
                wf = wrf.WrfFile(str(nc_path))
                wrf.save_native_plot(
                    wf,
                    product,
                    output_path=out_path,
                    width=args.width,
                    height=args.height,
                    borders=not args.no_borders,
                )
                del wf
                gc.collect()
            written.append(out_path)
            print(f"  wrote {out_path}")
        except Exception as exc:  # pragma: no cover - diagnostic tool
            failed = True
            print(f"  FAILED {product}: {exc}", file=sys.stderr)

    manifest = out_dir / "manifest.txt"
    manifest.write_text(
        "\n".join(str(path) for path in written) + ("\n" if written else ""),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
