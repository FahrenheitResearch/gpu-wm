#!/usr/bin/env python3
"""
Download latest GFS 0.25-degree analysis data from NOAA NOMADS.

Automatically detects the latest available GFS cycle and downloads
the f000 (analysis) GRIB2 file.

Usage:
    python tools/download_gfs.py                          # auto-detect latest
    python tools/download_gfs.py --date 20260320 --cycle 18
    python tools/download_gfs.py --output data/my_gfs.grib2
    python tools/download_gfs.py --forecast 006           # 6-hour forecast
"""

import argparse
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta


NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
CYCLES = ["18", "12", "06", "00"]
DEFAULT_OUTPUT = "data/gfs_latest.grib2"


def check_url_exists(url):
    """Check if a URL exists without downloading."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        size = resp.headers.get("Content-Length")
        return True, int(size) if size else None
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return False, None


def build_url(date_str, cycle, forecast="000"):
    """Build the NOMADS URL for a GFS file."""
    return (
        f"{NOMADS_BASE}/gfs.{date_str}/{cycle}/atmos/"
        f"gfs.t{cycle}z.pgrb2.0p25.f{forecast}"
    )


def find_latest_cycle(max_lookback_days=3):
    """Find the most recent available GFS cycle on NOMADS.

    Searches today first (most recent cycle first), then goes back
    up to max_lookback_days.

    Returns:
        Tuple of (date_str, cycle) or (None, None) if nothing found.
    """
    now = datetime.utcnow()
    for day_offset in range(max_lookback_days + 1):
        dt = now - timedelta(days=day_offset)
        date_str = dt.strftime("%Y%m%d")
        for cycle in CYCLES:
            url = build_url(date_str, cycle)
            print(f"  Checking {date_str}/{cycle}z ... ", end="", flush=True)
            exists, size = check_url_exists(url)
            if exists:
                size_mb = f" ({size / 1e6:.0f} MB)" if size else ""
                print(f"AVAILABLE{size_mb}")
                return date_str, cycle
            else:
                print("not yet")
    return None, None


def download_file(url, output_path, chunk_size=1024 * 1024):
    """Download a file with progress reporting.

    Args:
        url: URL to download.
        output_path: Local path to save to.
        chunk_size: Download chunk size (default 1 MB).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"  URL: {url}")
    print(f"  Saving to: {output_path}")

    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=300)

    total = resp.headers.get("Content-Length")
    total = int(total) if total else None
    total_mb = f"{total / 1e6:.0f} MB" if total else "unknown size"
    print(f"  File size: {total_mb}")

    downloaded = 0
    t0 = time.time()
    last_report = t0

    with open(output_path, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            # Progress every 5 seconds or on completion
            now = time.time()
            if now - last_report >= 5.0 or not chunk:
                elapsed = now - t0
                speed = downloaded / elapsed if elapsed > 0 else 0
                pct = f" ({100 * downloaded / total:.1f}%)" if total else ""
                print(
                    f"  Downloaded {downloaded / 1e6:.1f} MB{pct}"
                    f"  [{speed / 1e6:.1f} MB/s]",
                    flush=True,
                )
                last_report = now

    elapsed = time.time() - t0
    print(
        f"  Complete: {downloaded / 1e6:.1f} MB in {elapsed:.1f}s"
        f" ({downloaded / elapsed / 1e6:.1f} MB/s)"
    )
    return output_path


def verify_grib2(filepath):
    """Verify the downloaded GRIB2 file and list available fields.

    Uses eccodes if available, falls back to checking the GRIB magic number.
    """
    # Check file exists and has reasonable size
    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        return False

    size = os.path.getsize(filepath)
    if size < 1_000_000:
        print(f"  ERROR: File too small ({size} bytes), likely corrupt")
        return False

    print(f"  File size: {size / 1e6:.1f} MB")

    # Check GRIB magic number
    with open(filepath, "rb") as f:
        magic = f.read(4)
    if magic != b"GRIB":
        print(f"  ERROR: Not a valid GRIB file (magic: {magic!r})")
        return False
    print("  GRIB magic number: OK")

    # Try eccodes for detailed inspection
    try:
        import eccodes

        field_summary = {}
        with open(filepath, "rb") as f:
            count = 0
            while True:
                msgid = eccodes.codes_grib_new_from_file(f)
                if msgid is None:
                    break
                try:
                    name = eccodes.codes_get(msgid, "shortName")
                    level_type = eccodes.codes_get(msgid, "typeOfLevel")
                    level = eccodes.codes_get(msgid, "level")
                    key = (name, level_type)
                    if key not in field_summary:
                        field_summary[key] = []
                    field_summary[key].append(level)
                    count += 1
                finally:
                    eccodes.codes_release(msgid)

        print(f"  Total GRIB messages: {count}")
        print()
        print("  Available fields:")
        print(f"  {'Short Name':<12} {'Level Type':<25} {'Levels'}")
        print(f"  {'-'*12} {'-'*25} {'-'*40}")

        for (name, ltype), levels in sorted(field_summary.items()):
            levels_sorted = sorted(set(levels))
            if len(levels_sorted) > 8:
                lvl_str = (
                    f"{levels_sorted[0]}, {levels_sorted[1]}, ... "
                    f"{levels_sorted[-1]}  ({len(levels_sorted)} levels)"
                )
            else:
                lvl_str = ", ".join(str(l) for l in levels_sorted)
            print(f"  {name:<12} {ltype:<25} {lvl_str}")

        return True

    except ImportError:
        print("  eccodes not available -- skipping detailed field listing")
        print("  Install with: pip install eccodes")
        return True
    except Exception as e:
        print(f"  WARNING: eccodes verification error: {e}")
        print("  File may still be valid -- GRIB header is correct")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Download GFS 0.25-degree analysis data from NOAA NOMADS"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date in YYYYMMDD format (default: auto-detect latest)",
    )
    parser.add_argument(
        "--cycle",
        type=str,
        default=None,
        choices=["00", "06", "12", "18"],
        help="GFS cycle hour (default: auto-detect latest)",
    )
    parser.add_argument(
        "--forecast",
        type=str,
        default="000",
        help="Forecast hour as 3-digit string, e.g. 000, 006, 012 (default: 000 = analysis)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing file, do not download",
    )
    args = parser.parse_args()

    # Verify-only mode
    if args.verify_only:
        print(f"Verifying: {args.output}")
        ok = verify_grib2(args.output)
        sys.exit(0 if ok else 1)

    # Determine date and cycle
    if args.date and args.cycle:
        date_str, cycle = args.date, args.cycle
        print(f"Using specified cycle: {date_str}/{cycle}z")
    elif args.date and not args.cycle:
        # Date given, find latest cycle for that date
        print(f"Finding latest cycle for {args.date}...")
        date_str = args.date
        cycle = None
        for c in CYCLES:
            url = build_url(date_str, c)
            exists, _ = check_url_exists(url)
            if exists:
                cycle = c
                break
        if cycle is None:
            print(f"ERROR: No GFS data found for {date_str}")
            sys.exit(1)
        print(f"Latest cycle for {date_str}: {cycle}z")
    else:
        # Auto-detect
        print("Finding latest available GFS cycle...")
        date_str, cycle = find_latest_cycle()
        if date_str is None:
            print("ERROR: Could not find any available GFS data on NOMADS")
            sys.exit(1)
        print(f"Latest available: {date_str}/{cycle}z")

    # Build URL and download
    url = build_url(date_str, cycle, args.forecast)
    print()
    print(f"Downloading GFS {date_str}/{cycle}z f{args.forecast}:")

    try:
        download_file(url, args.output)
    except urllib.error.HTTPError as e:
        print(f"ERROR: HTTP {e.code} -- {url}")
        if e.code == 404:
            print("  File not found. The cycle may not have this forecast hour yet.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        sys.exit(1)

    # Verify
    print()
    print("Verifying download:")
    ok = verify_grib2(args.output)
    if not ok:
        sys.exit(1)

    print()
    print(f"GFS data ready: {args.output}")


if __name__ == "__main__":
    main()
