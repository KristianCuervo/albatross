"""
Download ERA5 hourly 10m wind data for Hamiltonian migration IVP simulation.

Dataset  : ERA5 reanalysis single-levels
Variables: u10, v10 (10m eastward and northward wind components)
Grid     : 0.25° × 0.25°  (Southern Ocean half-globe)
Period   : N days from start date (default: 7 days from 2022-12-01)
Timesteps: 00:00, 01:00, ..., 23:00 UTC (24 per day)

Output files (in CG2/data/), one per day:
    era5_1h_so_2022_12_01.nc   — 2022-12-01 (24 hourly steps, ~80 MB)
    era5_1h_so_2022_12_02.nc   — 2022-12-02
    …
    era5_1h_so_2022_12_07.nc   — 2022-12-07

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CDS API SETUP (one-time) — same as download_era5_6h.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Requires ~/.cdsapirc with your Personal Access Token from:
    https://cds.climate.copernicus.eu

Usage:
    python CG2/download_era5_1h.py                         # 7 days from 2022-12-01
    python CG2/download_era5_1h.py --start 2023-01-15 --days 7
"""

import argparse
import cdsapi
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 24 hourly timesteps per day
TIMES = [f"{h:02d}:00" for h in range(24)]

# Southern Ocean half-globe: [North, West, South, East]
AREA_SO = [0, -180, -90, 180]

DEFAULT_START = "2022-12-01"
DEFAULT_DAYS  = 7


def _day_filename(date: datetime) -> str:
    return f"era5_1h_so_{date.strftime('%Y_%m_%d')}.nc"


def download_day(date: datetime, client: cdsapi.Client) -> None:
    output = DATA_DIR / _day_filename(date)

    if output.exists():
        print(f"  Already exists: {output.name}  — delete to re-fetch.")
        return

    print(
        f"  Requesting {date.strftime('%Y-%m-%d')}  "
        f"(24 hourly timesteps, 0.25° grid, Southern Ocean half-globe)"
    )
    print(f"  Submitting to CDS… (may queue for several minutes)")

    request = {
        "product_type": "reanalysis",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year":  date.strftime("%Y"),
        "month": date.strftime("%m"),
        "day":   date.strftime("%d"),
        "time":  TIMES,
        "area":  AREA_SO,
        "grid":  [0.25, 0.25],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    client.retrieve("reanalysis-era5-single-levels", request, str(output))
    size_mb = output.stat().st_size / 1e6
    print(f"  Saved → {output.name}  ({size_mb:.1f} MB)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ERA5 hourly wind — Southern Ocean, N days from start date"
    )
    parser.add_argument(
        "--start", default=DEFAULT_START, metavar="YYYY-MM-DD",
        help=f"First day to download (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS, metavar="N",
        help=f"Number of days to download (default: {DEFAULT_DAYS})",
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    dates = [start_date + timedelta(days=i) for i in range(args.days)]

    print(f"Downloading {len(dates)} day(s) starting {args.start}:\n")
    client = cdsapi.Client()
    for date in dates:
        download_day(date, client)
    print("Done.")
