"""
Download ERA5 6-hourly 10m wind data for dynamic soaring opportunity analysis.

Dataset  : ERA5 reanalysis single-levels (not monthly means)
Variables: u10, v10 (10m eastward and northward wind components)
Grid     : 1.0° × 1.0°  (global, 360 × 181 grid)
Period   : Full year — Dec 2022 through Nov 2023 (4 austral seasons)
Timesteps: 00:00, 06:00, 12:00, 18:00 UTC (4 per day)

Output files (in CG2/data/):
    era5_6h_global_2022_12.nc   — Dec 2022
    era5_6h_global_2023_01.nc   — Jan 2023
    era5_6h_global_2023_02.nc   — Feb 2023
    era5_6h_global_2023_03.nc   — Mar 2023
    era5_6h_global_2023_04.nc   — Apr 2023
    era5_6h_global_2023_05.nc   — May 2023
    era5_6h_global_2023_06.nc   — Jun 2023
    era5_6h_global_2023_07.nc   — Jul 2023
    era5_6h_global_2023_08.nc   — Aug 2023
    era5_6h_global_2023_09.nc   — Sep 2023
    era5_6h_global_2023_10.nc   — Oct 2023
    era5_6h_global_2023_11.nc   — Nov 2023

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CDS API SETUP (one-time) — same as download_era5.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Requires ~/.cdsapirc with your Personal Access Token from:
    https://cds.climate.copernicus.eu

Run:
    python CG2/download_era5_6h.py              # download all 12 months
    python CG2/download_era5_6h.py --season djf # single season
    python CG2/download_era5_6h.py --month 2023-07  # single month
"""

import argparse

import calendar
import cdsapi
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 4 synoptic times per day
TIMES = ["00:00", "06:00", "12:00", "18:00"]

# Global bounding box: [North, West, South, East]
AREA_GLOBAL = [90, -180, -90, 180]

# All 12 months in order (one CDS request each)
MONTHS = [
    {"year": "2022", "month": "12", "file": "era5_6h_global_2022_12.nc"},
    {"year": "2023", "month": "01", "file": "era5_6h_global_2023_01.nc"},
    {"year": "2023", "month": "02", "file": "era5_6h_global_2023_02.nc"},
    {"year": "2023", "month": "03", "file": "era5_6h_global_2023_03.nc"},
    {"year": "2023", "month": "04", "file": "era5_6h_global_2023_04.nc"},
    {"year": "2023", "month": "05", "file": "era5_6h_global_2023_05.nc"},
    {"year": "2023", "month": "06", "file": "era5_6h_global_2023_06.nc"},
    {"year": "2023", "month": "07", "file": "era5_6h_global_2023_07.nc"},
    {"year": "2023", "month": "08", "file": "era5_6h_global_2023_08.nc"},
    {"year": "2023", "month": "09", "file": "era5_6h_global_2023_09.nc"},
    {"year": "2023", "month": "10", "file": "era5_6h_global_2023_10.nc"},
    {"year": "2023", "month": "11", "file": "era5_6h_global_2023_11.nc"},
]

# Austral season groupings (keys match ds_opportunity_map.py SEASONS)
SEASON_MONTHS = {
    "djf": {"2022_12", "2023_01", "2023_02"},
    "mam": {"2023_03", "2023_04", "2023_05"},
    "jja": {"2023_06", "2023_07", "2023_08"},
    "son": {"2023_09", "2023_10", "2023_11"},
}


def _all_days(year: int, month: int) -> list[str]:
    """Return zero-padded day strings for every day in the given month."""
    n_days = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, n_days + 1)]


def download_month(cfg: dict, client: cdsapi.Client) -> None:
    output = DATA_DIR / cfg["file"]

    if output.exists():
        print(f"  Already exists: {output.name}  — delete to re-fetch.")
        return

    year  = int(cfg["year"])
    month = int(cfg["month"])
    days  = _all_days(year, month)
    n_ts  = len(days) * len(TIMES)

    print(f"  Requesting {cfg['year']}-{cfg['month']}  ({n_ts} timesteps, {len(days)} days × 4 times)")
    print(f"  Submitting to CDS… (may queue for several minutes)")

    request = {
        "product_type": "reanalysis",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year":  cfg["year"],
        "month": cfg["month"],
        "day":   days,
        "time":  TIMES,
        "area":  AREA_GLOBAL,
        "grid":  [1.0, 1.0],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    client.retrieve("reanalysis-era5-single-levels", request, str(output))
    size_mb = output.stat().st_size / 1e6
    print(f"  Saved → {output.name}  ({size_mb:.1f} MB)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ERA5 6-hourly wind — 4 austral seasons"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--season", choices=list(SEASON_MONTHS.keys()), metavar="SEASON",
        help=f"Download only one season's months. Choices: {', '.join(SEASON_MONTHS)}",
    )
    grp.add_argument(
        "--month", metavar="YYYY-MM",
        help="Download a single month, e.g. 2023-07",
    )
    args = parser.parse_args()

    if args.season:
        keys = SEASON_MONTHS[args.season]
        to_download = [m for m in MONTHS if f"{m['year']}_{m['month']}" in keys]
    elif args.month:
        y, mo = args.month.split("-")
        to_download = [m for m in MONTHS if m["year"] == y and m["month"] == mo]
        if not to_download:
            raise SystemExit(f"Month {args.month} not in download list.")
    else:
        to_download = MONTHS

    print(f"Downloading {len(to_download)} month(s):\n")
    client = cdsapi.Client()
    for cfg in to_download:
        download_month(cfg, client)
    print("Done.")
