#!/usr/bin/env python3
"""Run a repeatable evaluation cycle for GPU-WM model iteration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import run_fast_case as rfc
import plot_pivotal_collage as collage


PRESETS = {
    "4km-large": {"nx": 1350, "ny": 795, "nz": 50, "dx": 4000.0, "dt": 12.0},
    "3km-large": {"nx": 1536, "ny": 1024, "nz": 50, "dx": 3000.0, "dt": 10.0},
    "1km-regional": {"nx": 1024, "ny": 1024, "nz": 75, "dx": 1000.0, "dt": 3.0},
}


def run_subprocess(cmd: list[str], cwd: Path) -> None:
    print(f"$ (cd {cwd} && {' '.join(cmd)})")
    subprocess.run(cmd, cwd=cwd, check=True)


def pick_metric_entry(results: list[dict], path: Path) -> dict | None:
    for item in results:
        if Path(item["path"]).resolve() == path.resolve():
            return item
    return None


def expected_forecast_outputs(dt: float, t_end: float, output_interval: float) -> int:
    total_steps = int(t_end / dt)
    next_output = output_interval
    count = 0
    for step in range(1, total_steps + 1):
        sim_time = step * dt
        if sim_time >= next_output - 0.5 * dt:
            count += 1
            next_output += output_interval
    return count


def write_summary(
    output_path: Path,
    preset: str,
    run_dir: Path,
    init_path: Path,
    verify_json: Path,
    collage_paths: list[Path],
    wrf_product_dirs: list[Path],
    results: list[dict],
    boundary_specs: list[str],
) -> None:
    lines = [
        f"# GPU-WM Iteration Cycle",
        "",
        f"- Preset: `{preset}`",
        f"- Init: `{init_path}`",
        f"- Run dir: `{run_dir}`",
        f"- Verification JSON: `{verify_json}`",
        f"- WRF product dirs: `{len(wrf_product_dirs)}`",
        "",
        "## Forecast Steps",
        "",
        "| Lead (h) | Mean w | MeanAbsW | MaxAbsW | U RMSE | V RMSE | THETA RMSE | QV RMSE | Flags | Collage |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    collage_by_stem = {path.stem.replace("_collage", ""): path for path in collage_paths}
    for result in results:
        time_h = result["time_hours"]
        health = result["health"]
        full = result["full_volume"]
        stem = Path(result["path"]).stem
        collage_path = collage_by_stem.get(stem)
        flags = "; ".join(result.get("flags", [])) or "none"
        collage_ref = collage_path.name if collage_path else "-"
        lines.append(
            f"| {time_h:.2f} | {health['mean_w']:+.3f} | {health['mean_abs_w']:.3f} | "
            f"{health['max_abs_w']:.2f} | {full['U']['rmse']:.2f} | {full['V']['rmse']:.2f} | "
            f"{full['THETA']['rmse']:.2f} | {full['QV']['rmse']:.5f} | {flags} | {collage_ref} |"
        )

    lines.extend(["", "## Notes", ""])
    if boundary_specs:
        lines.append(f"- Boundary schedule uses {len(boundary_specs)} parent snapshots.")
    max_mean_w = max(abs(item["health"]["mean_w"]) for item in results)
    max_uv_rmse = max(max(item["full_volume"]["U"]["rmse"], item["full_volume"]["V"]["rmse"]) for item in results)
    if max_mean_w > 0.05:
        lines.append("- Broad ascent bias is still present. The next solver focus should stay on pressure/mass closure and lower-boundary realism.")
    if max_uv_rmse > 2.0:
        lines.append("- Wind drift is noticeable over the cycle. Boundary forcing and pressure-gradient consistency are still likely limiting factors.")
    if max_mean_w <= 0.05 and max_uv_rmse <= 2.0:
        lines.append("- Short-range stability is acceptable on this cycle; move to a longer forecast or a larger domain before changing physics.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a model-improvement evaluation cycle")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="4km-large")
    parser.add_argument("--timesteps", type=int, default=3, help="Number of forecast timesteps to keep")
    parser.add_argument("--step-seconds", type=float, default=1200.0, help="Seconds between forecast outputs")
    parser.add_argument("--tag", default="cycle", help="Run tag")
    parser.add_argument("--grib", default="data/gfs_latest.grib2", help="Source GFS GRIB2 file")
    parser.add_argument("--build", default="build-wsl/gpu-wm", help="Path to gpu-wm executable")
    parser.add_argument("--ref-lat", type=float, default=38.5)
    parser.add_argument("--ref-lon", type=float, default=-97.5)
    parser.add_argument("--truelat1", type=float, default=38.5)
    parser.add_argument("--truelat2", type=float, default=38.5)
    parser.add_argument("--stand-lon", type=float, default=-97.5)
    parser.add_argument("--regen-init", action="store_true")
    parser.add_argument("--skip-init-plot", action="store_true")
    parser.add_argument(
        "--terrain-following-init",
        action="store_true",
        help="Regenerate and use a terrain-following init instead of the default flat-height init",
    )
    parser.add_argument("--adaptive-stab", action="store_true", help="Keep adaptive stability enabled")
    parser.add_argument("--thompson", action="store_true", help="Use Thompson microphysics")
    parser.add_argument(
        "--render-wrf-products",
        action="store_true",
        help="Render WRF-style products after the run using tools/render_wrf_products.py",
    )
    parser.add_argument(
        "--render-wrf-products-all",
        action="store_true",
        help="Render WRF-style products for every forecast output instead of only the final one",
    )
    parser.add_argument(
        "--wrf-render-cmd",
        default=sys.executable,
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
    parser.add_argument("--output-root", default="run-cycles", help="Root directory for cycle outputs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable
    preset = PRESETS[args.preset].copy()

    nx = preset["nx"]
    ny = preset["ny"]
    nz = preset["nz"]
    dx = preset["dx"]
    dt = preset["dt"]
    ztop = 25000.0
    t_end = args.timesteps * args.step_seconds
    output_interval = args.step_seconds

    boundary_specs = rfc.resolve_boundary_specs(args, repo_root)
    extra_boundary_states = len(boundary_specs) + (1 if boundary_specs else 0)
    mem_gb = rfc.estimate_mem_gb(nx, ny, nz, extra_states=extra_boundary_states)
    print(f"Cycle preset: {args.preset}")
    print(f"Grid: {nx} x {ny} x {nz} @ {dx:.0f} m  dt={dt:.1f} s  est_vram={mem_gb:.2f} GB")

    region_tag = (
        f"{nx}x{ny}x{nz}_dx{int(round(dx))}"
        f"_lat{args.ref_lat:+05.1f}_lon{args.ref_lon:+06.1f}"
    ).replace("+", "p").replace("-", "m")
    init_suffix = "_terrain" if args.terrain_following_init else ""
    init_path = repo_root / "data" / f"gfs_init_fast_{region_tag}{init_suffix}.bin"
    plot_path = repo_root / "plots" / f"gfs_init_fast_{region_tag}{init_suffix}.png"

    init_start = time.perf_counter()
    rfc.ensure_init(
        repo_root=repo_root,
        python_exe=python_exe,
        grib_path=repo_root / args.grib,
        init_path=init_path,
        plot_path=plot_path,
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        truelat1=args.truelat1,
        truelat2=args.truelat2,
        stand_lon=args.stand_lon,
        ref_lat=args.ref_lat,
        ref_lon=args.ref_lon,
        ztop=ztop,
        regen=args.regen_init,
        skip_init_plot=args.skip_init_plot,
        terrain_following_init=args.terrain_following_init,
    )
    init_elapsed = time.perf_counter() - init_start

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / args.output_root / f"{args.preset}_{args.tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_cmd = [
        str(repo_root / args.build),
        "--gfs", str(init_path),
        "--netcdf",
        "--dt", f"{dt}",
        "--tend", f"{t_end}",
        "--output-interval", f"{output_interval}",
    ]
    if not args.adaptive_stab:
        model_cmd.append("--no-adaptive-stab")
    if args.thompson:
        model_cmd.append("--thompson")
    for spec in boundary_specs:
        model_cmd.extend(["--boundary-state", spec])

    run_start = time.perf_counter()
    rfc.run_command(model_cmd, run_dir)
    run_elapsed = time.perf_counter() - run_start

    outputs = rfc.collect_outputs(run_dir)
    expected_forecasts = expected_forecast_outputs(dt, t_end, output_interval)
    expected_total = expected_forecasts + 1
    if len(outputs) < expected_total:
        raise RuntimeError(f"Expected at least {expected_total} outputs, found {len(outputs)}")

    verify_json = run_dir / "verification_summary.json"
    verify_cmd = [
        python_exe,
        "tools/verify_forecast.py",
        "--reference", str(init_path),
        "--json-out", str(verify_json),
        *[str(path) for path in outputs],
    ]
    run_subprocess(verify_cmd, repo_root)

    verify_payload = json.loads(verify_json.read_text(encoding="utf-8"))
    results = verify_payload["results"]

    collage_dir = run_dir / "collages"
    collage_dir.mkdir(exist_ok=True)
    forecast_outputs = outputs[1:expected_total]
    collage_paths: list[Path] = []
    for out_path in forecast_outputs:
        metrics = pick_metric_entry(results, out_path)
        collage_path = Path(collage.plot_collage(str(out_path), str(collage_dir), metrics=metrics))
        collage_paths.append(collage_path)

    wrf_product_dirs: list[Path] = []
    if args.render_wrf_products:
        wrf_product_dirs = rfc.render_wrf_products(
            repo_root=repo_root,
            runner_cmd=rfc.parse_command_spec(args.wrf_render_cmd),
            outputs=forecast_outputs,
            render_all=args.render_wrf_products_all,
        )

    overview_path = run_dir / "cycle_overview.png"
    collage.build_overview([str(path) for path in collage_paths], str(overview_path))

    summary_path = run_dir / "SUMMARY.md"
    summary_results = [pick_metric_entry(results, path) for path in forecast_outputs]
    write_summary(
        output_path=summary_path,
        preset=args.preset,
        run_dir=run_dir,
        init_path=init_path,
        verify_json=verify_json,
        collage_paths=collage_paths,
        wrf_product_dirs=wrf_product_dirs,
        results=[item for item in summary_results if item is not None],
        boundary_specs=boundary_specs,
    )

    manifest = {
        "preset": args.preset,
        "run_dir": str(run_dir),
        "init_path": str(init_path),
        "boundary_states": boundary_specs,
        "boundary_next": args.boundary_next,
        "boundary_interval": args.boundary_interval,
        "verification_json": str(verify_json),
        "summary": str(summary_path),
        "overview": str(overview_path),
        "collages": [str(path) for path in collage_paths],
        "wrf_product_dirs": [str(path) for path in wrf_product_dirs],
        "init_time_seconds": init_elapsed,
        "run_time_seconds": run_elapsed,
        "model_command": model_cmd,
    }
    manifest_path = run_dir / "cycle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print("Iteration cycle complete")
    print(f"  Run dir:   {run_dir}")
    print(f"  Summary:   {summary_path}")
    print(f"  Overview:  {overview_path}")
    print(f"  Collages:  {len(collage_paths)}")
    if wrf_product_dirs:
        print(f"  WRF plots: {len(wrf_product_dirs)}")
    if boundary_specs:
        print(f"  Boundary states: {len(boundary_specs)}")
    print(f"  Init time: {init_elapsed:.1f} s")
    print(f"  Run time:  {run_elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
