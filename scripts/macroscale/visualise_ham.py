"""
visualise_ham.py — Interactive Hamiltonian IVP trajectory + wind explorer.

Loads a precomputed Hamiltonian fan NPZ (from migration.ipynb) and displays
trajectories over a live ERA5 wind speed background with a time slider.

Each trajectory is integrated with adaptive RK45, so time arrays are
variable-length (padded with NaN in the NPZ).  Positions are interpolated
at the slider's query time for smooth animation.

- Background: ERA5 10 m wind speed; grey = below DS threshold (8.6 m/s)
- Coloured lines: trajectory paths from start to current time
- Coloured dots: interpolated current position of each trajectory
- Colour: HSV mapped to initial costate angle φ ∈ [0, 2π)

Usage
-----
    python scripts/ham_explorer.py
    python scripts/ham_explorer.py --npz data/hamiltonian_jan2023_snippet.npz
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
    p = argparse.ArgumentParser(description="Interactive Hamiltonian migration explorer.")
    p.add_argument(
        "--npz", type=Path,
        default=ROOT / "data" / "macroscale" / "hamiltonian_jan2023_snippet.npz",
        help="Path to Hamiltonian NPZ produced by migration.ipynb",
    )
    return p.parse_args()


def load_npz(path: Path):
    """
    Load padded Hamiltonian fan NPZ.

    Returns
    -------
    valid_ts, valid_xs, valid_ys : lists of 1-D arrays (NaN padding stripped)
    phis                         : (n_dirs,) initial costate angles [rad]
    t0, t_end                    : float, float  simulation time bounds [unix s]
    start_lat, start_lon         : float, float  start position [deg]
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run migration.ipynb first (set RECOMPUTE_HAM=True)."
        )
    d    = np.load(path)
    ys   = d["ys"]    # (n_dirs, max_nt)
    xs   = d["xs"]
    ts   = d["ts"]
    phis = d["phis"]

    n_dirs = len(phis)
    valid_ts, valid_xs, valid_ys = [], [], []
    for i in range(n_dirs):
        mask = ~np.isnan(ts[i])
        valid_ts.append(ts[i][mask])
        valid_xs.append(xs[i][mask])
        valid_ys.append(ys[i][mask])

    t0      = float(min(v[0]  for v in valid_ts if len(v) > 0))
    t_end   = float(max(v[-1] for v in valid_ts if len(v) > 0))
    start_lat = float(valid_ys[0][0])
    start_lon = float(valid_xs[0][0])

    return valid_ts, valid_xs, valid_ys, phis, t0, t_end, start_lat, start_lon


def load_era5(t0: float, t_end: float) -> ERA5Interpolator:
    nc_files = sorted(ERA5_DIR.glob("era5_1h_so_*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No ERA5 files in {ERA5_DIR}")
    print("Loading ERA5 …")
    era5 = ERA5Interpolator(
        nc_files,
        lat_range  = (EXTENT[3], EXTENT[2]),
        time_range = (t0, t_end + 3600),
    )
    print(f"  Loaded {era5._u10.shape}  ({era5._u10.nbytes / 1e6:.0f} MB)")
    return era5


def compute_speed_at(era5: ERA5Interpolator, unix_t: float) -> np.ndarray:
    lo, hi, wt = era5._time_weights(float(unix_t))
    u = (1 - wt) * era5._u10[lo].astype(np.float64) + wt * era5._u10[hi].astype(np.float64)
    v = (1 - wt) * era5._v10[lo].astype(np.float64) + wt * era5._v10[hi].astype(np.float64)
    return np.sqrt(u**2 + v**2)


def build_figure(valid_ts, valid_xs, valid_ys, phis, t0, t_end, start_lat, start_lon, era5):
    n_dirs   = len(phis)
    T_total  = t_end - t0
    n_hours  = int(round(T_total / 3600))

    # ── Colours: HSV over initial costate angle φ ─────────────────────────────
    traj_colors = plt.cm.hsv(phis / (2 * np.pi))

    # ── Wind colourmap ────────────────────────────────────────────────────────
    wind_cmap = plt.cm.YlGnBu.copy()
    wind_cmap.set_under("#b8b8b8")
    wind_norm = mcolors.Normalize(vmin=DS_THRESH, vmax=30.0)

    # ── Figure layout ─────────────────────────────────────────────────────────
    proj  = ccrs.PlateCarree()
    fig   = plt.figure(figsize=(13, 8))
    ax    = fig.add_axes([0.05, 0.12, 0.88, 0.82], projection=proj)
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

    ax.scatter([start_lon], [start_lat],
               marker="*", s=260, c="black", edgecolors="white", linewidths=0.7,
               zorder=9, transform=proj, label="Bird Island")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)

    # ── Wind background ───────────────────────────────────────────────────────
    lat = era5._lat
    lon = era5._lon
    LON_G, LAT_G = np.meshgrid(lon, lat)

    speed0 = compute_speed_at(era5, t0)
    pcm = ax.pcolormesh(
        LON_G, LAT_G, speed0,
        cmap=wind_cmap, norm=wind_norm,
        transform=proj, shading="auto",
        zorder=1, alpha=0.55,
    )
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02, aspect=30)
    cbar.set_label("10 m wind speed  [m/s]", fontsize=10)

    # ── Trajectory lines ──────────────────────────────────────────────────────
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
    dt0 = datetime.utcfromtimestamp(t0)
    title = ax.set_title(
        f"Hamiltonian IVP migration — {dt0.strftime('%Y-%m-%d  %H:%M UTC')}  "
        f"(+0 h / {n_hours} h)  |  grey = < {DS_THRESH} m/s",
        fontsize=11, pad=6,
    )

    # ── Slider (hourly steps matching ERA5 resolution) ────────────────────────
    slider = Slider(ax_sl, "Hour", 0, n_hours, valinit=0, valstep=1, color="#4a90d9")

    # ── Callback ──────────────────────────────────────────────────────────────
    def update(val):
        h      = int(slider.val)
        unix_t = t0 + h * 3600.0

        # Wind background
        pcm.set_array(compute_speed_at(era5, unix_t))

        dot_lons = np.full(n_dirs, start_lon)
        dot_lats = np.full(n_dirs, start_lat)

        for i, line in enumerate(lines):
            ts_i = valid_ts[i]
            xs_i = valid_xs[i]
            ys_i = valid_ys[i]

            if len(ts_i) == 0 or unix_t < ts_i[0]:
                line.set_data([], [])
                continue

            if unix_t >= ts_i[-1]:
                # Trajectory has finished — show full path, dot at endpoint
                line.set_data(xs_i, ys_i)
                dot_lons[i] = xs_i[-1]
                dot_lats[i] = ys_i[-1]
            else:
                # Interpolate current position, show path up to here
                cur_lon = float(np.interp(unix_t, ts_i, xs_i))
                cur_lat = float(np.interp(unix_t, ts_i, ys_i))
                mask    = ts_i <= unix_t
                path_xs = np.append(xs_i[mask], cur_lon)
                path_ys = np.append(ys_i[mask], cur_lat)
                line.set_data(path_xs, path_ys)
                dot_lons[i] = cur_lon
                dot_lats[i] = cur_lat

        sc.set_offsets(np.column_stack([dot_lons, dot_lats]))

        dt = datetime.utcfromtimestamp(unix_t)
        title.set_text(
            f"Hamiltonian IVP migration — {dt.strftime('%Y-%m-%d  %H:%M UTC')}  "
            f"(+{h} h / {n_hours} h)  |  grey = < {DS_THRESH} m/s"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._slider = slider

    return fig


def main():
    args = parse_args()
    valid_ts, valid_xs, valid_ys, phis, t0, t_end, start_lat, start_lon = load_npz(args.npz)
    print(f"Loaded {args.npz.name}: {len(phis)} trajectories over "
          f"{(t_end - t0) / 3600:.1f} h")

    era5 = load_era5(t0, t_end)
    fig  = build_figure(valid_ts, valid_xs, valid_ys, phis, t0, t_end,
                        start_lat, start_lon, era5)
    plt.show()


if __name__ == "__main__":
    main()
