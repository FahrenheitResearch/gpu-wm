#!/usr/bin/env python3
"""Run a reduced-domain GPU-WM case for fast solver iteration.

This script keeps the iteration loop tight:
1. regenerate a small GFS-derived init if needed
2. launch an isolated forecast run
3. verify the produced NetCDF outputs against the init state
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"$ (cd {cwd} && {' '.join(shlex.quote(part) for part in cmd)})")
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_command_spec(spec: str) -> list[str]:
    parts = shlex.split(spec)
    if not parts:
        raise ValueError("command spec must not be empty")
    return parts


def default_wrf_render_cmd() -> str:
    if shutil.which("cmd.exe"):
        probe = subprocess.run(
            ["cmd.exe", "/c", "py", "-3", "--version"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return "cmd.exe /c py -3"
    return sys.executable


def wsl_to_windows_path(path: Path) -> str:
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def runner_uses_windows_paths(runner_cmd: list[str]) -> bool:
    if not runner_cmd:
        return False
    first = runner_cmd[0].lower()
    return first in {"cmd.exe", "py", "powershell.exe", "pwsh.exe"}


def estimate_mem_gb(nx: int, ny: int, nz: int, extra_states: int = 0) -> float:
    n3d = (nx + 4) * (ny + 4) * nz
    state_count = 3 + extra_states
    mem_3d_fields = n3d * 17 * 4 * state_count
    mem_base = nz * 5 * 8 * state_count
    return (mem_3d_fields + mem_base) / 1e9


def _normalize_boundary_spec(spec: str) -> str:
    token = spec.strip()
    if not token:
        raise ValueError("Boundary state spec must not be empty")
    if "@" not in token:
        return token

    path_token, maybe_time = token.rsplit("@", 1)
    try:
        boundary_time = float(maybe_time)
    except ValueError:
        return token
    return f"{path_token}@{boundary_time:g}"


def resolve_boundary_specs(args: argparse.Namespace, repo_root: Path) -> list[str]:
    specs: list[str] = []

    for item in args.boundary_state or []:
        normalized = _normalize_boundary_spec(item)
        if "@" in normalized:
            path_token, time_token = normalized.rsplit("@", 1)
            path_obj = Path(path_token)
            if not path_obj.is_absolute():
                path_obj = repo_root / path_obj
            specs.append(f"{path_obj}@{time_token}")
        else:
            path_obj = Path(normalized)
            if not path_obj.is_absolute():
                path_obj = repo_root / path_obj
            specs.append(str(path_obj))

    for pattern in args.boundary_glob or []:
        matches = sorted(repo_root.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No boundary states matched glob: {pattern}")
        specs.extend(str(path) for path in matches)

    if args.boundary_next:
        next_path = Path(args.boundary_next)
        if not next_path.is_absolute():
            next_path = repo_root / next_path
        next_spec = str(next_path)
        if args.boundary_interval is not None:
            next_spec = f"{next_spec}@{args.boundary_interval:g}"
        specs.append(next_spec)

    return specs


def ensure_init(
    repo_root: Path,
    python_exe: str,
    grib_path: Path,
    surface_grib_path: Path | None,
    init_path: Path,
    plot_path: Path,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    truelat1: float,
    truelat2: float,
    stand_lon: float,
    ref_lat: float,
    ref_lon: float,
    ztop: float,
    regen: bool,
    skip_init_plot: bool,
    terrain_following_init: bool,
    stretched_eta: bool,
) -> None:
    if init_path.exists() and not regen:
        print(f"Reusing init: {init_path}")
        return

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        "tools/init_from_gfs.py",
        "--pressure-grib",
        str(grib_path),
        "--output",
        str(init_path),
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--nz",
        str(nz),
        "--dx",
        f"{dx}",
        "--truelat1",
        f"{truelat1}",
        "--truelat2",
        f"{truelat2}",
        "--stand-lon",
        f"{stand_lon}",
        "--ref-lat",
        f"{ref_lat}",
        "--ref-lon",
        f"{ref_lon}",
        "--ztop",
        f"{ztop}",
        "--plot",
        str(plot_path),
    ]
    if surface_grib_path is not None:
        cmd.extend(["--surface-grib", str(surface_grib_path)])
    if skip_init_plot:
        cmd.append("--no-plot")
    if terrain_following_init:
        cmd.append("--terrain-following-init")
    if stretched_eta:
        cmd.append("--stretched-eta")
    run_command(cmd, repo_root)


def collect_outputs(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "output").glob("gpuwm_*.nc"))


def infer_surface_grib_pattern(surface_grib_path: Path) -> tuple[Path, str] | None:
    match = re.match(r"^(.*wrfsfcf)(\d{2})(\.grib2)$", surface_grib_path.name)
    if not match:
        return None
    prefix, _, suffix = match.groups()
    return surface_grib_path.parent, f"{prefix}{{hour:02d}}{suffix}"


def maybe_run_surface_realism(
    repo_root: Path,
    python_exe: str,
    outputs: list[Path],
    run_dir: Path,
    surface_grib_path: Path | None,
) -> tuple[Path, Path] | None:
    if surface_grib_path is None:
        return None
    pattern_info = infer_surface_grib_pattern(surface_grib_path)
    if pattern_info is None:
        print(f"WARNING: could not infer HRRR surface pattern from {surface_grib_path}; skipping surface realism")
        return None
    surface_grib_dir, surface_grib_pattern = pattern_info
    realism_json = run_dir / "surface_realism.json"
    realism_md = run_dir / "surface_realism.md"
    try:
        run_command(
            [
                python_exe,
                "tools/verify_surface_realism.py",
                *[str(path) for path in outputs],
                "--surface-grib-dir",
                str(surface_grib_dir),
                "--surface-grib-pattern",
                surface_grib_pattern,
                "--json-out",
                str(realism_json),
                "--markdown-out",
                str(realism_md),
            ],
            repo_root,
        )
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: surface realism verification failed: {exc}")
        return None
    return realism_json, realism_md


def render_wrf_products(
    repo_root: Path,
    runner_cmd: list[str],
    outputs: list[Path],
    render_all: bool,
) -> list[Path]:
    use_windows_paths = runner_uses_windows_paths(runner_cmd)
    render_targets = outputs if render_all else outputs[-1:]
    rendered_dirs: list[Path] = []
    for nc_path in render_targets:
        if render_all:
            out_dir = nc_path.parent.parent / f"wrf_products_{nc_path.stem}"
        else:
            out_dir = nc_path.parent.parent / "wrf_products"
        cmd = [
            *runner_cmd,
            wsl_to_windows_path(repo_root / "tools" / "render_wrf_products.py")
            if use_windows_paths else str(repo_root / "tools" / "render_wrf_products.py"),
            "--input",
            wsl_to_windows_path(nc_path) if use_windows_paths else str(nc_path),
            "--output-dir",
            wsl_to_windows_path(out_dir) if use_windows_paths else str(out_dir),
        ]
        run_command(cmd, repo_root)
        rendered_dirs.append(out_dir)
    return rendered_dirs


def load_verify_results(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("results", []))


def postprocess_outputs(
    repo_root: Path,
    python_exe: str,
    init_path: Path,
    run_dir: Path,
    outputs: list[Path],
    surface_grib_path: Path | None,
) -> None:
    verify_json = run_dir / "verify_all.json"
    weather_dir = run_dir / "plots_weather"
    collage_dir = run_dir / "plots_collage"

    run_command(
        [
            python_exe,
            "tools/verify_forecast.py",
            "--reference",
            str(init_path),
            "--json-out",
            str(verify_json),
            *[str(path) for path in outputs],
        ],
        repo_root,
    )

    maybe_run_surface_realism(
        repo_root=repo_root,
        python_exe=python_exe,
        outputs=outputs,
        run_dir=run_dir,
        surface_grib_path=surface_grib_path,
    )

    run_command(
        [
            python_exe,
            "tools/plot_weather.py",
            "--output-dir",
            str(weather_dir),
            *[str(path) for path in outputs],
        ],
        repo_root,
    )

    verify_results = load_verify_results(verify_json)
    if verify_results:
        metrics_path = collage_dir / "last_metrics.json"
        collage_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(verify_results[-1], indent=2), encoding="utf-8")
    run_command(
        [
            python_exe,
            "tools/plot_pivotal_collage.py",
            "--output-dir",
            str(collage_dir),
            str(outputs[-1]),
        ],
        repo_root,
    )


def main() -> int:
    argv_tokens = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Run a small GPU-WM case for quick iteration",
    )
    parser.add_argument(
        "--grib",
        default="data/gfs_latest.grib2",
        help="Primary source GRIB2 file (single-file GFS or pressure-level HRRR)",
    )
    parser.add_argument(
        "--surface-grib",
        help="Optional secondary source GRIB2 file for surface fields, e.g. HRRR wrfsfcf00",
    )
    parser.add_argument("--init", help="Reduced-domain init binary to use or create")
    parser.add_argument("--build", default="build-wsl/gpu-wm", help="Path to gpu-wm executable")
    parser.add_argument("--regional-1km", action="store_true", help="Preset for a 1 km regional prototype")
    parser.add_argument("--regional-3km-large", action="store_true", help="Preset for a large 3 km regional domain")
    parser.add_argument("--regional-4km-large", action="store_true", help="Preset for a large 4 km regional/near-CONUS domain")
    parser.add_argument("--print-estimate", action="store_true", help="Print memory estimate and exit")
    parser.add_argument("--nx", type=int, default=256, help="Fast-loop domain size in x")
    parser.add_argument("--ny", type=int, default=256, help="Fast-loop domain size in y")
    parser.add_argument("--nz", type=int, default=50, help="Vertical levels")
    parser.add_argument("--dx", type=float, default=3000.0, help="Grid spacing in meters")
    parser.add_argument("--ztop", type=float, default=25000.0, help="Model top in meters")
    parser.add_argument("--dt", type=float, help="Model time step in seconds")
    parser.add_argument("--truelat1", type=float, default=38.5, help="Lambert true latitude 1")
    parser.add_argument("--truelat2", type=float, default=38.5, help="Lambert true latitude 2")
    parser.add_argument("--stand-lon", type=float, default=-97.5, help="Lambert standard longitude")
    parser.add_argument("--ref-lat", type=float, default=38.5, help="Regional domain center latitude")
    parser.add_argument("--ref-lon", type=float, default=-97.5, help="Regional domain center longitude")
    parser.add_argument("--tend", type=float, default=900.0, help="Forecast length in seconds")
    parser.add_argument(
        "--output-interval",
        type=float,
        help="Output interval in seconds (defaults to 900 s for longer runs)",
    )
    parser.add_argument("--tag", default="fast", help="Run tag for the output directory")
    parser.add_argument("--regen-init", action="store_true", help="Force regeneration of the reduced init")
    parser.add_argument("--skip-init-plot", action="store_true", help="Skip PNG generation during init")
    parser.add_argument(
        "--terrain-following-init",
        action="store_true",
        help="Generate a terrain-following init instead of the default flat-height column sampling",
    )
    parser.add_argument(
        "--stretched-eta",
        action="store_true",
        help="Use experimental stretched reference levels when generating a new init",
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip post-run verification")
    parser.add_argument(
        "--render-wrf-products",
        action="store_true",
        help="Render WRF-style products after the run using tools/render_wrf_products.py",
    )
    parser.add_argument(
        "--render-wrf-products-all",
        action="store_true",
        help="Render WRF-style products for every NetCDF output instead of only the final one",
    )
    parser.add_argument(
        "--wrf-render-cmd",
        default=default_wrf_render_cmd(),
        help="Interpreter/command used to run render_wrf_products.py, e.g. 'py -3.13'",
    )
    parser.add_argument(
        "--boundary-state",
        action="append",
        default=[],
        help="Repeatable parent-boundary snapshot, optionally PATH@SECONDS",
    )
    parser.add_argument(
        "--boundary-glob",
        action="append",
        default=[],
        help="Repeatable glob for parent-boundary snapshots (resolved relative to repo root)",
    )
    parser.add_argument("--boundary-next", help="Optional next parent init binary for time-varying boundaries")
    parser.add_argument("--boundary-interval", type=float, help="Seconds spanned by start->next boundary interpolation")
    parser.add_argument(
        "--no-adaptive-stab",
        action="store_true",
        help="Pass --no-adaptive-stab to gpu-wm",
    )
    parser.add_argument(
        "--postprocess-weather",
        action="store_true",
        help="Write verify JSON plus weather plots/collage into the run directory after completion",
    )
    parser.add_argument(
        "model_args",
        nargs=argparse.REMAINDER,
        help="Extra gpu-wm args to append after --",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable

    explicit_nx = "--nx" in argv_tokens
    explicit_ny = "--ny" in argv_tokens
    explicit_nz = "--nz" in argv_tokens
    explicit_dx = "--dx" in argv_tokens
    explicit_dt = "--dt" in argv_tokens

    if args.regional_1km:
        if not explicit_nx:
            args.nx = 1024
        if not explicit_ny:
            args.ny = 1024
        if not explicit_nz:
            args.nz = 75
        if not explicit_dx:
            args.dx = 1000.0

    if args.dt is None and not explicit_dt:
        args.dt = min(10.0, max(1.0, 10.0 * args.dx / 3000.0))
        if args.regional_1km:
            args.dt = min(args.dt, 3.0)

    if args.regional_3km_large:
        if not explicit_nx:
            args.nx = 1536
        if not explicit_ny:
            args.ny = 1024
        if not explicit_nz:
            args.nz = 50
        if not explicit_dx:
            args.dx = 3000.0
        if args.dt is None or not explicit_dt:
            args.dt = 10.0

    if args.regional_4km_large:
        if not explicit_nx:
            args.nx = 1350
        if not explicit_ny:
            args.ny = 795
        if not explicit_nz:
            args.nz = 50
        if not explicit_dx:
            args.dx = 4000.0
        if args.dt is None or not explicit_dt:
            args.dt = 12.0

    boundary_specs = resolve_boundary_specs(args, repo_root)
    extra_boundary_states = len(boundary_specs) + (1 if boundary_specs else 0)
    mem_est_gb = estimate_mem_gb(args.nx, args.ny, args.nz, extra_states=extra_boundary_states)
    if args.print_estimate:
        print(f"Grid:      {args.nx} x {args.ny} x {args.nz} @ {args.dx:.0f} m")
        print(f"Projection: truelat1={args.truelat1:.2f} truelat2={args.truelat2:.2f} "
              f"stand_lon={args.stand_lon:.2f} ref=({args.ref_lat:.2f}, {args.ref_lon:.2f})")
        print(f"dt:        {args.dt:.2f} s")
        print(f"Est. VRAM: {mem_est_gb:.2f} GB")
        return 0

    region_tag = (
        f"{args.nx}x{args.ny}x{args.nz}_dx{int(round(args.dx))}"
        f"_lat{args.ref_lat:+05.1f}_lon{args.ref_lon:+06.1f}"
    ).replace("+", "p").replace("-", "m")
    source_hint = f"{args.grib} {args.surface_grib or ''}".lower()
    source_tag = "hrrr" if "hrrr" in source_hint else "gfs" if "gfs" in source_hint else "grib"
    init_suffix_parts: list[str] = []
    if args.terrain_following_init and not args.init:
        init_suffix_parts.append("terrain")
    if args.stretched_eta and not args.init:
        init_suffix_parts.append("stretch")
    init_suffix = f"_{'_'.join(init_suffix_parts)}" if init_suffix_parts else ""
    init_name = f"{source_tag}_init_fast_{region_tag}{init_suffix}.bin"
    plot_name = f"{source_tag}_init_fast_{region_tag}{init_suffix}.png"
    init_path = (repo_root / args.init) if args.init else (repo_root / "data" / init_name)
    plot_path = repo_root / "plots" / plot_name
    grib_path = repo_root / args.grib
    surface_grib_path = (repo_root / args.surface_grib) if args.surface_grib else None
    build_path = repo_root / args.build

    need_grib_inputs = args.regen_init or (not args.init and not init_path.exists())
    if need_grib_inputs and not grib_path.exists():
        raise FileNotFoundError(f"Primary GRIB file not found: {grib_path}")
    if need_grib_inputs and surface_grib_path is not None and not surface_grib_path.exists():
        raise FileNotFoundError(f"Surface GRIB file not found: {surface_grib_path}")
    if not build_path.exists():
        raise FileNotFoundError(f"gpu-wm executable not found: {build_path}")

    output_interval = args.output_interval if args.output_interval is not None else min(args.tend, 900.0)

    t0 = time.perf_counter()
    ensure_init(
        repo_root=repo_root,
        python_exe=python_exe,
        grib_path=grib_path,
        surface_grib_path=surface_grib_path,
        init_path=init_path,
        plot_path=plot_path,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        dx=args.dx,
        truelat1=args.truelat1,
        truelat2=args.truelat2,
        stand_lon=args.stand_lon,
        ref_lat=args.ref_lat,
        ref_lon=args.ref_lon,
        ztop=args.ztop,
        regen=args.regen_init,
        skip_init_plot=args.skip_init_plot,
        terrain_following_init=args.terrain_following_init,
        stretched_eta=args.stretched_eta,
    )
    init_elapsed = time.perf_counter() - t0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / "run-fast" / f"{args.tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(build_path),
        "--gfs",
        str(init_path),
        "--netcdf",
        "--dt",
        f"{args.dt}",
        "--tend",
        f"{args.tend}",
        "--output-interval",
        f"{output_interval}",
    ]
    if args.no_adaptive_stab:
        cmd.append("--no-adaptive-stab")
    for spec in boundary_specs:
        cmd.extend(["--boundary-state", spec])

    extra_args = args.model_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)

    t1 = time.perf_counter()
    run_command(cmd, run_dir)
    run_elapsed = time.perf_counter() - t1

    outputs = collect_outputs(run_dir)
    if not outputs:
        raise RuntimeError(f"No NetCDF outputs found in {run_dir / 'output'}")

    if not args.skip_verify:
        verify_cmd = [
            python_exe,
            "tools/verify_forecast.py",
            "--reference",
            str(init_path),
            *[str(path) for path in outputs],
        ]
        run_command(verify_cmd, repo_root)
        if not args.postprocess_weather:
            maybe_run_surface_realism(
                repo_root=repo_root,
                python_exe=python_exe,
                outputs=outputs,
                run_dir=run_dir,
                surface_grib_path=surface_grib_path,
            )

    if args.postprocess_weather:
        postprocess_outputs(
            repo_root=repo_root,
            python_exe=python_exe,
            init_path=init_path,
            run_dir=run_dir,
            outputs=outputs,
            surface_grib_path=surface_grib_path,
        )

    rendered_dirs: list[Path] = []
    if args.render_wrf_products:
        rendered_dirs = render_wrf_products(
            repo_root=repo_root,
            runner_cmd=parse_command_spec(args.wrf_render_cmd),
            outputs=outputs,
            render_all=args.render_wrf_products_all,
        )

    print()
    print("Fast-loop run complete")
    print(f"  Init:      {init_path}")
    print(f"  Run dir:   {run_dir}")
    if need_grib_inputs:
        print(f"  Source:    {grib_path}")
    else:
        print("  Source:    existing --init")
    if surface_grib_path is not None:
        print(f"  Surface:   {surface_grib_path}")
    print(f"  Grid:      {args.nx} x {args.ny} x {args.nz} @ {args.dx:.0f} m")
    print(f"  Projection: truelat1={args.truelat1:.2f} truelat2={args.truelat2:.2f} "
          f"stand_lon={args.stand_lon:.2f} ref=({args.ref_lat:.2f}, {args.ref_lon:.2f})")
    print(f"  dt:        {args.dt:.2f} s")
    print(f"  Est. VRAM: {mem_est_gb:.2f} GB")
    print(f"  Outputs:   {len(outputs)} files")
    if boundary_specs:
        print(f"  Boundary states: {len(boundary_specs)}")
    if rendered_dirs:
        print(f"  WRF plots: {len(rendered_dirs)} directories")
    if surface_grib_path is not None:
        realism_json = run_dir / "surface_realism.json"
        realism_md = run_dir / "surface_realism.md"
        if realism_json.exists():
            print(f"  Realism:   {realism_json}")
        if realism_md.exists():
            print(f"  RealismMD: {realism_md}")
    print(f"  Init time: {init_elapsed:.1f} s")
    print(f"  Run time:  {run_elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
