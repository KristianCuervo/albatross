"""
wind_explorer.py — Interactive ERA5 wind speed explorer with time slider.

Displays hourly 10 m wind speed over the Southern Ocean centred on Bird Island.
Winds below the DS threshold (8.6 m/s) are masked in grey.

Usage
-----
    python scripts/wind_explorer.py [--days N] [--start-date YYYY-MM-DD]

Defaults to the first 14 days of December 2022 (matching the migration notebook).
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # change to Qt5Agg if TkAgg is unavailable on your system
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realWind import ERA5Interpolator

# ── Constants ──────────────────────────────────────────────────────────────────
BIRD_LAT  = -54.0
BIRD_LON  = -38.05
DS_THRESH = 8.6          # m/s — minimum wind for dynamic soaring
ERA5_DIR  = ROOT / "data" / "era5"

# Map extent [lon_min, lon_max, lat_min, lat_max]
EXTENT = [-80.0, 10.0, -75.0, -30.0]


def parse_args():
    p = argparse.ArgumentParser(description="Interactive ERA5 wind speed explorer.")
    p.add_argument("--days",       type=int, default=31,
                   help="Number of days to load (default: 31)")
    p.add_argument("--start-date", type=str, default="2023-01-01",
                   help="Start date YYYY-MM-DD (default: 2023-01-01)")
    return p.parse_args()


def load_era5(start_date: str, n_days: int) -> tuple[ERA5Interpolator, float]:
    nc_files = sorted(ERA5_DIR.glob("era5_1h_so_*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No ERA5 files found in {ERA5_DIR}")

    t0 = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp()
    t1 = t0 + n_days * 86400 + 3600

    print(f"Loading ERA5: {start_date}  +{n_days} days …")
    era5 = ERA5Interpolator(
        nc_files,
        lat_range  = (EXTENT[3], EXTENT[2]),   # (north, south)
        time_range = (t0, t1),
    )
    print(f"  Loaded {era5._u10.shape}  ({era5._u10.nbytes / 1e6:.0f} MB)")
    return era5, t0


def compute_speed(era5: ERA5Interpolator, step: int) -> np.ndarray:
    """Return 2-D wind speed array for the given integer hour step index."""
    u = era5._u10[step].astype(np.float64)
    v = era5._v10[step].astype(np.float64)
    return np.sqrt(u**2 + v**2)


def build_figure(era5: ERA5Interpolator, t0: float):
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(13, 8))
    ax_map = fig.add_axes([0.05, 0.12, 0.88, 0.82], projection=proj)
    ax_sl  = fig.add_axes([0.10, 0.04, 0.78, 0.03])

    # ── Static map features ───────────────────────────────────────────────────
    ax_map.set_extent(EXTENT, crs=proj)
    ax_map.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5", zorder=0)
    ax_map.add_feature(cfeature.LAND,      facecolor="#c8c8c8", zorder=3)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#444444", zorder=4)
    ax_map.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "antarctic_ice_shelves_polys", "50m",
            facecolor="white", edgecolor="none"),
        zorder=2,
    )
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, linestyle="--",
                          alpha=0.5, color="#666666")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    ax_map.scatter([BIRD_LON], [BIRD_LAT],
                   marker="*", s=240, c="red", edgecolors="white", linewidths=0.6,
                   zorder=8, transform=proj, label="Bird Island")
    ax_map.legend(loc="upper right", fontsize=10, framealpha=0.85)

    # ── Colourmap: grey for below-threshold, YlGnBu above ────────────────────
    base_cmap = plt.cm.YlGnBu.copy()
    base_cmap.set_under("#b8b8b8")   # grey for speeds < DS_THRESH
    norm = mcolors.Normalize(vmin=DS_THRESH, vmax=30.0)

    # ── Initial wind field via pcolormesh ─────────────────────────────────────
    lat = era5._lat
    lon = era5._lon
    LON, LAT = np.meshgrid(lon, lat)

    speed0 = compute_speed(era5, 0)

    pcm = ax_map.pcolormesh(
        LON, LAT, speed0,
        cmap=base_cmap, norm=norm,
        transform=proj,
        shading="auto",
        zorder=1,
    )

    # Colourbar
    cbar = fig.colorbar(pcm, ax=ax_map, shrink=0.75, pad=0.02, aspect=30)
    cbar.set_label("10 m wind speed  [m/s]", fontsize=11)
    cbar.ax.axhline(y=0.0, color="#cc0000", linewidth=1.2, linestyle="--",
                    label=f"{DS_THRESH} m/s threshold")
    # Annotate threshold on colorbar
    cbar.ax.text(2.6, 0.01, f"{DS_THRESH} m/s", va="bottom", ha="left",
                 fontsize=8, color="#cc0000",
                 transform=cbar.ax.transAxes)

    # ── Title ─────────────────────────────────────────────────────────────────
    dt0 = datetime.utcfromtimestamp(float(era5._unix_times[0]))
    title = ax_map.set_title(
        f"ERA5 10 m wind speed — {dt0.strftime('%Y-%m-%d  %H:%M UTC')}  "
        f"|  grey = < {DS_THRESH} m/s (no DS)",
        fontsize=11, pad=6,
    )

    # ── Slider ────────────────────────────────────────────────────────────────
    n_steps = len(era5._unix_times) - 1
    slider = Slider(ax_sl, "Hour", 0, n_steps, valinit=0, valstep=1,
                    color="#4a90d9")

    # ── Callback ──────────────────────────────────────────────────────────────
    def update(val):
        step   = int(slider.val)
        speed  = compute_speed(era5, step)
        unix_t = float(era5._unix_times[step])

        # Update pcolormesh data in-place — no remove/redraw of collections
        pcm.set_array(speed)

        dt = datetime.utcfromtimestamp(unix_t)
        title.set_text(
            f"ERA5 10 m wind speed — {dt.strftime('%Y-%m-%d  %H:%M UTC')}  "
            f"|  grey = < {DS_THRESH} m/s (no DS)"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._slider = slider   # prevent garbage collection

    return fig


def main():
    args = parse_args()
    era5, t0 = load_era5(args.start_date, args.days)
    fig = build_figure(era5, t0)
    plt.show()


if __name__ == "__main__":
    main()
