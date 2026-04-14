"""
Download ERA5 hourly 10m wind data for the southern hemisphere (0°→−90°, global longitude),
one file per day.  Skips days that are already on disk.

Usage:
    python scripts/download_era5_hourly_so.py [--data-dir data/era5]

Target dates (hardcoded — edit MONTHS below to extend):
    December 2022, January 2023, February 2023

Output files:
    <data_dir>/era5_1h_so_YYYY_MM_DD.nc   (~46 MB each, 24 timesteps)

Requires ~/.cdsapirc with valid Copernicus CDS credentials.
"""

import argparse
import calendar
import sys
from pathlib import Path

import cdsapi

# ── Target months ──────────────────────────────────────────────────────────────
# Each entry: (year, month)
MONTHS = [
    (2022, 12),
    (2023,  1),
    (2023,  2),
]

AREA = [0, -180, -90, 180]   # north, west, south, east — matches existing files

HOURS = [f"{h:02d}:00" for h in range(24)]

DATASET = "reanalysis-era5-single-levels"

REQUEST_BASE = {
    "product_type": "reanalysis",
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],
    "time": HOURS,
    "area": AREA,
    "data_format": "netcdf",
    "download_format": "unarchived",
}


def iter_days():
    """Yield (year, month, day) for every day in MONTHS."""
    for year, month in MONTHS:
        n_days = calendar.monthrange(year, month)[1]
        for day in range(1, n_days + 1):
            yield year, month, day


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    days = list(iter_days())
    total = len(days)

    skipped = 0
    downloaded = 0

    for i, (year, month, day) in enumerate(days, 1):
        fname = f"era5_1h_so_{year}_{month:02d}_{day:02d}.nc"
        out   = data_dir / fname

        if out.exists():
            print(f"[{i:3d}/{total}] Already exists — skipping: {fname}")
            skipped += 1
            continue

        print(f"[{i:3d}/{total}] Requesting {year}-{month:02d}-{day:02d} …", flush=True)

        request = {
            **REQUEST_BASE,
            "year":  str(year),
            "month": f"{month:02d}",
            "day":   f"{day:02d}",
        }

        try:
            client.retrieve(DATASET, request, str(out))
            size_mb = out.stat().st_size / 1e6
            print(f"[{i:3d}/{total}] Saved → {fname}  ({size_mb:.1f} MB)", flush=True)
            downloaded += 1
        except Exception as exc:
            print(f"[{i:3d}/{total}] FAILED {fname}: {exc}", file=sys.stderr, flush=True)

    print(f"\nDone.  Downloaded {downloaded}, skipped {skipped} (already on disk), "
          f"total {total} days.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/era5",
                        help="Directory to write NetCDF files (default: data/era5)")
    args = parser.parse_args()
    download(Path(args.data_dir))


if __name__ == "__main__":
    main()
