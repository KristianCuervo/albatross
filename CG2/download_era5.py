"""
Download ERA5 monthly mean 10m wind data for albatross study regions.

Dataset  : ERA5 monthly averaged single-levels
Variables: u10, v10 (10m eastward and northward wind components)
Grid     : 0.25° × 0.25°  (~28 km)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CDS API SETUP (one-time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Register a free account at:
       https://cds.climate.copernicus.eu

2. Accept the ERA5 Terms of Use (required before any download):
       https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
       → scroll to "Terms of use" → tick the box → submit

3. Get your Personal Access Token:
       Log in → click your profile icon (top-right) → "Personal Access Token"
       → copy the token (long UUID string)

4. Create the credentials file ~/.cdsapirc:

       # Linux / macOS — run these two lines in your terminal:
       echo "url: https://cds.climate.copernicus.eu/api" > ~/.cdsapirc
       echo "key: 6ee4b527-cd06-48aa-b41d-9e7fc6497f31" >> ~/.cdsapirc
       chmod 600 ~/.cdsapirc

       # Windows — create %USERPROFILE%\.cdsapirc with contents:
       #   url: https://cds.climate.copernicus.eu/api
       #   key: YOUR_TOKEN_HERE

5. Install Python dependencies:
       pip install -r requirements.txt

6. Run this script:
       python download_era5.py                  # download all regions
       python download_era5.py --region soatl   # single region by key

Expected download sizes: 2–10 MB per region
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import cdsapi
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─── Region definitions ───────────────────────────────────────────────────────
# area format: [North, West, South, East]  (degrees, negative = S or W)
#
# Wandering albatross (Diomedea exulans) key regions:
#   Colonies: South Georgia (54°S 36°W), Crozet (46°S 52°E),
#             Kerguelen (49°S 70°E), Heard (53°S 73°E),
#             Macquarie (54°S 159°E), Antipodes/Campbell NZ (50–52°S 169–178°E)
#   Foraging: circumpolar Southern Ocean, roughly 30°S–65°S
#
# Month convention:
#   Southern Hemisphere winter = July (peak westerlies, strongest wind shear)
#   Northern Hemisphere summer = July (existing equatorial download)

DATASETS = {
    # ── existing equatorial Atlantic (SH summer winds, kept for comparison) ──
    "eq_atlantic": {
        "desc":  "Equatorial Atlantic — July 2023",
        "area":  [30, -80, -30, 20],
        "year":  "2023",
        "month": "07",
        "file":  "era5_wind_eq_atlantic_2023_07.nc",
    },
    # ── Southern Ocean full circuit (SH winter = peak albatross season) ────
    # Covers Chile → South Atlantic → South Africa → Indian Ocean →
    # Australia → New Zealand, including all major colony islands.
    "southern_ocean": {
        "desc":  "Southern Ocean full circuit — July 2023 (SH winter)",
        "area":  [-30, -180, -65, 180],
        "year":  "2023",
        "month": "07",
        "file":  "era5_wind_southern_ocean_2023_07.nc",
    },
    # ── South Atlantic (colony sector: South Georgia, Tristan da Cunha) ────
    "soatl": {
        "desc":  "South Atlantic colony sector — July 2023",
        "area":  [-30, -70, -65, 20],
        "year":  "2023",
        "month": "07",
        "file":  "era5_wind_soatl_2023_07.nc",
    },
    # ── North Atlantic (vagrant range, for comparison) ─────────────────────
    "n_atlantic": {
        "desc":  "North Atlantic — July 2023",
        "area":  [70, -80, 30, 20],
        "year":  "2023",
        "month": "07",
        "file":  "era5_wind_n_atlantic_2023_07.nc",
    },
}

BASE_REQUEST = {
    "product_type": "monthly_averaged_reanalysis",
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],
    "time": "00:00",
    "grid": [0.25, 0.25],
    "data_format": "netcdf",
    "download_format": "unarchived",
}


def download(region_key: str, client: cdsapi.Client) -> None:
    cfg    = DATASETS[region_key]
    output = DATA_DIR / cfg["file"]

    if output.exists():
        print(f"  [{region_key}] Already exists: {output.name}  — delete to re-fetch.")
        return

    request = {
        **BASE_REQUEST,
        "year":  cfg["year"],
        "month": cfg["month"],
        "area":  cfg["area"],
    }
    N, W, S, E = cfg["area"]
    print(f"  [{region_key}] {cfg['desc']}")
    print(f"             area = {N}°N, {W}°E, {S}°N, {E}°E")
    print(f"             Submitting to CDS… (may queue for a few minutes)")
    client.retrieve("reanalysis-era5-single-levels-monthly-means", request, str(output))
    print(f"             Saved → {output.name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ERA5 monthly mean wind data for albatross regions"
    )
    parser.add_argument(
        "--region", choices=list(DATASETS.keys()), default=None,
        metavar="KEY",
        help=(
            "Download a single region by key. "
            f"Available: {', '.join(DATASETS.keys())}. "
            "Default: download all."
        ),
    )
    args = parser.parse_args()

    keys = [args.region] if args.region else list(DATASETS.keys())

    print(f"Downloading {len(keys)} region(s):\n")
    client = cdsapi.Client()
    for key in keys:
        download(key, client)

    print("Done.")
