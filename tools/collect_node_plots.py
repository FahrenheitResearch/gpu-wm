#!/usr/bin/env python3
"""Collect the most useful weather plots into a single node_plots folder."""

from __future__ import annotations

import shutil
from pathlib import Path


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def latest_run_with_outputs(repo_root: Path, prefix: str) -> Path | None:
    run_root = repo_root / "run-fast"
    candidates = sorted(
        (
            path for path in run_root.glob(f"{prefix}*")
            if path.is_dir()
            and (path / "plots_weather").exists()
            and (path / "plots_collage").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def latest_file(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def copy_latest_weather_set(run_dir: Path, out_root: Path, label: str) -> None:
    weather_dir = run_dir / "plots_weather"
    collage_dir = run_dir / "plots_collage"
    latest_analysis = latest_file(weather_dir, "analysis_t*.png")
    latest_radar = latest_file(weather_dir, "radar_t*.png")
    latest_satellite = latest_file(weather_dir, "satellite_t*.png")
    latest_sfc = latest_file(weather_dir, "sfc_temp_t*.png")
    latest_collage = latest_file(collage_dir, "gpuwm_*_collage.png")

    if latest_analysis:
        copy_if_exists(latest_analysis, out_root / label / latest_analysis.name)
    if latest_radar:
        copy_if_exists(latest_radar, out_root / label / latest_radar.name)
    if latest_satellite:
        copy_if_exists(latest_satellite, out_root / label / latest_satellite.name)
    if latest_sfc:
        copy_if_exists(latest_sfc, out_root / label / latest_sfc.name)
    if latest_collage:
        copy_if_exists(latest_collage, out_root / label / latest_collage.name)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_root = repo_root / "node_plots"
    out_root.mkdir(exist_ok=True)

    latest_local = latest_run_with_outputs(repo_root, "auto_local_")
    if latest_local is None:
        latest_local = latest_run_with_outputs(repo_root, "manual_local_")
    if latest_local is not None:
        copy_latest_weather_set(latest_local, out_root, "local_latest")

    latest_remote = latest_run_with_outputs(repo_root, "auto_remote_")
    if latest_remote is not None:
        copy_latest_weather_set(latest_remote, out_root, "remote_latest")

    latest_remote2 = latest_run_with_outputs(repo_root, "auto_remote2_")
    if latest_remote2 is not None:
        copy_latest_weather_set(latest_remote2, out_root, "remote2_latest")

    copy_if_exists(
        repo_root / "plots" / "erfpure_6h_collage" / "gpuwm_000006_collage.png",
        out_root / "six_hour_reference" / "gpuwm_000006_collage.png",
    )
    copy_if_exists(
        repo_root / "plots" / "erfpure_6h_weather" / "radar_t00021600.png",
        out_root / "six_hour_reference" / "radar_t00021600.png",
    )
    copy_if_exists(
        repo_root / "plots" / "erfpure_6h_weather" / "sfc_temp_t00021600.png",
        out_root / "six_hour_reference" / "sfc_temp_t00021600.png",
    )
    copy_if_exists(
        repo_root / "plots" / "erfpure_6h_weather" / "satellite_t00021600.png",
        out_root / "six_hour_reference" / "satellite_t00021600.png",
    )
    copy_if_exists(
        repo_root / "plots" / "erfpure_6h_weather" / "analysis_t00021600.png",
        out_root / "six_hour_reference" / "analysis_t00021600.png",
    )

    print(f"Collected plots into {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
