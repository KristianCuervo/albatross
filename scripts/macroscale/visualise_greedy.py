"""
visualise_greedy.py — Interactive migration trajectory + wind explorer.

Loads a precomputed greedy IVP NPZ (from migration.ipynb) and displays
trajectories over a live ERA5 wind speed background with a time slider.

- Background: ERA5 10 m wind speed; grey = below DS threshold (8.6 m/s)
- Coloured lines: trajectory paths from start to current step
- Coloured dots: current position of each trajectory
- Colour: HSV mapped to initial costate direction (full circle)

Usage
-----
    python scripts/migration_explorer.py
    python scripts/migration_explorer.py --npz data/migration_jan2023_snippet.npz
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from albatross.macroscale import ERA5Interpolator

# ── Constants ──────────────────────────────────────────────────────────────────
DS_THRESH = 8.6
ERA5_DIR  = ROOT / "data" / "era5"
EXTENT    = [-80.0, 10.0, -75.0, -30.0]   # [lon_min, lon_max, lat_min, lat_max]


def parse_args():
    p = argparse.ArgumentParser(description="Interactive migration trajectory explorer.")
    p.add_argument(
        "--npz", type=Path,
        default=ROOT / "data" / "macroscale" / "migration_jan2023_snippet.npz",
        help="Path to migration NPZ produced by migration.ipynb",
    )
    return p.parse_args()


def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run migration.ipynb first (set RECOMPUTE=True)."
        )
    d = np.load(path)
    # positions : (n_steps+1, n_dirs, 2)  columns = (lat, lon)
    # times     : (n_steps+1,)            unix timestamps [s]
    return d["positions"], d["times"]


def load_era5(t0: float, t1: float) -> ERA5Interpolator:
    nc_files = sorted(ERA5_DIR.glob("era5_1h_so_*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No ERA5 files in {ERA5_DIR}")
    print("Loading ERA5 …")
    era5 = ERA5Interpolator(
        nc_files,
        lat_range  = (EXTENT[3], EXTENT[2]),
        time_range = (t0, t1 + 3600),
    )
    print(f"  Loaded {era5._u10.shape}  ({era5._u10.nbytes / 1e6:.0f} MB)")
    return era5


def compute_speed_at(era5: ERA5Interpolator, unix_t: float) -> np.ndarray:
    lo, hi, wt = era5._time_weights(float(unix_t))
    u = (1 - wt) * era5._u10[lo].astype(np.float64) + wt * era5._u10[hi].astype(np.float64)
    v = (1 - wt) * era5._v10[lo].astype(np.float64) + wt * era5._v10[hi].astype(np.float64)
    return np.sqrt(u**2 + v**2)


def build_figure(positions: np.ndarray, times: np.ndarray, era5: ERA5Interpolator):
    n_steps, n_dirs, _ = positions.shape
    n_steps -= 1

    # ── Colours: HSV over initial costate direction ───────────────────────────
    theta_dirs  = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    traj_colors = plt.cm.hsv(theta_dirs / (2 * np.pi))

    # ── Wind colourmap ────────────────────────────────────────────────────────
    wind_cmap = plt.cm.YlGnBu.copy()
    wind_cmap.set_under("#b8b8b8")
    wind_norm = mcolors.Normalize(vmin=DS_THRESH, vmax=30.0)

    # ── Figure layout ─────────────────────────────────────────────────────────
    proj = ccrs.PlateCarree()
    fig  = plt.figure(figsize=(13, 8))
    ax   = fig.add_axes([0.05, 0.12, 0.88, 0.82], projection=proj)
    ax_sl = fig.add_axes([0.10, 0.04, 0.78, 0.03])

    # Static map features
    ax.set_extent(EXTENT, crs=proj)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5", zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor="#c8c8c8", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#444444", zorder=4)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "antarctic_ice_shelves_polys", "50m",
            facecolor="white", edgecolor="none"),
        zorder=2,
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--",
                      alpha=0.5, color="#666666")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    # Bird Island start marker
    start_lon = float(positions[0, 0, 1])
    start_lat = float(positions[0, 0, 0])
    ax.scatter([start_lon], [start_lat],
               marker="*", s=260, c="black", edgecolors="white", linewidths=0.7,
               zorder=9, transform=proj, label="Bird Island")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)

    # ── Wind background ───────────────────────────────────────────────────────
    lat = era5._lat
    lon = era5._lon
    LON_G, LAT_G = np.meshgrid(lon, lat)

    speed0 = compute_speed_at(era5, times[0])
    pcm = ax.pcolormesh(
        LON_G, LAT_G, speed0,
        cmap=wind_cmap, norm=wind_norm,
        transform=proj, shading="auto",
        zorder=1, alpha=0.55,
    )
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02, aspect=30)
    cbar.set_label("10 m wind speed  [m/s]", fontsize=10)

    # ── Trajectory lines (one per direction, initially empty) ─────────────────
    lines = []
    for i in range(n_dirs):
        line, = ax.plot([], [], color=traj_colors[i], lw=0.8, alpha=0.75,
                        transform=proj, zorder=5)
        lines.append(line)

    # ── Current-position dots ─────────────────────────────────────────────────
    sc = ax.scatter(
        np.full(n_dirs, start_lon),
        np.full(n_dirs, start_lat),
        c=traj_colors, s=14, zorder=7,
        transform=proj, edgecolors="none",
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    dt0 = datetime.utcfromtimestamp(float(times[0]))
    title = ax.set_title(
        f"Greedy migration — {dt0.strftime('%Y-%m-%d  %H:%M UTC')}  "
        f"(step 0 / {n_steps})  |  grey = < {DS_THRESH} m/s",
        fontsize=11, pad=6,
    )

    # ── Slider ────────────────────────────────────────────────────────────────
    slider = Slider(ax_sl, "Step", 0, n_steps, valinit=0, valstep=1, color="#4a90d9")

    # ── Callback ──────────────────────────────────────────────────────────────
    def update(val):
        step   = int(slider.val)
        unix_t = times[step]

        pcm.set_array(compute_speed_at(era5, unix_t))

        for i, line in enumerate(lines):
            line.set_data(positions[:step + 1, i, 1],   # lons
                          positions[:step + 1, i, 0])   # lats

        sc.set_offsets(positions[step, :, ::-1])         # (lon, lat)

        dt = datetime.utcfromtimestamp(float(unix_t))
        title.set_text(
            f"Greedy migration — {dt.strftime('%Y-%m-%d  %H:%M UTC')}  "
            f"(step {step} / {n_steps})  |  grey = < {DS_THRESH} m/s"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._slider = slider

    return fig


def main():
    args = parse_args()
    positions, times = load_npz(args.npz)
    print(f"Loaded {args.npz.name}: {positions.shape}  ({len(times)} timesteps)")

    era5 = load_era5(float(times[0]), float(times[-1]))
    fig  = build_figure(positions, times, era5)
    plt.show()


if __name__ == "__main__":
    main()
