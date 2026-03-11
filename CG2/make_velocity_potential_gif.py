"""
make_velocity_potential_gif.py
==============================
Animated GIF of DS velocity potential over JJA 2023.

At each 6-hourly ERA5 timestep and grid point:
  - Velocity potential = 0                   where |wind| < 9 m/s (DS threshold)
  - Velocity potential = max hull radius      where |wind| >= 9 m/s

"Max hull radius" is the maximum ground speed achievable via DS in any
direction at the local wind speed V_ref, read directly from the pre-computed
velocity hull table (velocity_hulls.npz).

Output
------
CG2/figures/velocity_potential_jja.gif

Usage
-----
    python CG2/make_velocity_potential_gif.py
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

HERE        = Path(__file__).parent
DATA_DIR    = HERE / "data"
FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DS_THRESHOLD = 9.0   # m/s

JJA_FILES = [
    "era5_6h_global_2023_06.nc",
    "era5_6h_global_2023_07.nc",
    "era5_6h_global_2023_08.nc",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_var(ds, *candidates):
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(f"None of {candidates} found in {list(ds.data_vars)}")


def load_hull():
    hull_path = DATA_DIR / "velocity_hulls.npz"
    if not hull_path.exists():
        sys.exit(f"ERROR: Hull table not found: {hull_path}\n"
                 "Run: python CG2/build_velocity_hull.py --hull-only")
    raw = np.load(hull_path)
    return {k: raw[k] for k in raw.files}


def build_max_speed_table(hull_data):
    """1-D lookup: V_ref → max achievable DS ground speed (m/s)."""
    v_ref_levels = hull_data["v_ref_levels"]      # (n_v,)
    hull_radii   = hull_data["hull_radii"]         # (n_v, n_rays)  — NaN for unreachable rays
    max_speeds   = np.nanmax(hull_radii, axis=1)   # (n_v,)
    return v_ref_levels, max_speeds


def compute_potential(u10, v10, v_ref_levels, max_speeds):
    """
    DS velocity potential at every grid point, in m/s.
    Zero where wind speed is below the DS threshold.
    """
    v_ref     = np.hypot(u10, v10)
    v_clipped = np.clip(v_ref, v_ref_levels.min(), v_ref_levels.max())
    potential = np.interp(v_clipped, v_ref_levels, max_speeds).astype(np.float32)
    potential[v_ref < DS_THRESHOLD] = 0.0
    return potential


# ─── Load and precompute all frames ───────────────────────────────────────────

def load_all_frames(v_ref_levels, max_speeds):
    """
    Load JJA ERA5 files and precompute velocity potential for every timestep.

    Returns
    -------
    lat        : (n_lat,)
    lon        : (n_lon,)
    potentials : (n_frames, n_lat, n_lon) float32
    labels     : list of timestamp strings, length n_frames
    """
    potentials, labels = [], []
    lat = lon = None

    for fname in JJA_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            sys.exit(f"ERROR: Missing file {path}")
        print(f"  {fname}…", flush=True)

        ds         = xr.open_dataset(path)
        u          = _get_var(ds, "u10", "u_10m", "eastward_wind")
        v          = _get_var(ds, "v10", "v_10m", "northward_wind")
        time_coord = "valid_time" if "valid_time" in ds.coords else "time"
        times      = ds[time_coord].values   # numpy datetime64

        if lat is None:
            lat = u.latitude.values.astype(np.float64)
            lon = u.longitude.values.astype(np.float64)

        u_arr = u.values.astype(np.float32)   # (T, lat, lon)
        v_arr = v.values.astype(np.float32)
        ds.close()

        epoch = np.datetime64(0, "s")
        one_s = np.timedelta64(1, "s")

        for t_idx in range(u_arr.shape[0]):
            pot = compute_potential(u_arr[t_idx], v_arr[t_idx], v_ref_levels, max_speeds)
            potentials.append(pot)

            unix_t = float((times[t_idx] - epoch) / one_s)
            labels.append(datetime.utcfromtimestamp(unix_t).strftime("%Y-%m-%d %H:%M UTC"))

        print(f"    {u_arr.shape[0]} frames", flush=True)

    return lat, lon, np.stack(potentials, axis=0), labels


# ─── Animation ────────────────────────────────────────────────────────────────

def make_gif(lat, lon, potentials, labels):
    n_frames = potentials.shape[0]
    vmax     = float(np.nanmax(potentials))   # ~56 m/s from hull

    print(f"  Frames : {n_frames}", flush=True)
    print(f"  vmax   : {vmax:.1f} m/s", flush=True)

    LON, LAT = np.meshgrid(lon, lat)

    fig, ax = plt.subplots(
        figsize=(14, 7),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    ax.set_global()

    coll = ax.pcolormesh(
        LON, LAT, potentials[0],
        cmap="inferno",
        vmin=0, vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=2,
        rasterized=True,
    )
    cbar = plt.colorbar(
        coll, ax=ax, shrink=0.6, pad=0.03,
        label="Max DS ground speed (m/s)",
        orientation="vertical",
    )
    cbar.set_ticks(np.arange(0, vmax + 1, 10))

    ax.add_feature(cfeature.LAND,      facecolor="#3a3a3a", zorder=5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#888888", zorder=6)

    title = ax.set_title("", fontsize=11)

    def update(i):
        coll.set_array(potentials[i].ravel())
        title.set_text(
            f"DS Velocity Potential — {labels[i]}\n"
            f"ERA5 6-hourly  •  max hull ground speed  •  threshold {DS_THRESHOLD} m/s"
        )
        return coll, title

    anim = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=125,    # ms between frames (8 fps)
        blit=False,
    )

    out = FIGURES_DIR / "velocity_potential_jja.gif"
    print(f"  Saving GIF ({n_frames} frames at 8 fps)…", flush=True)
    anim.save(str(out), writer="pillow", fps=8, dpi=90)
    plt.close(fig)
    print(f"  Saved → {out}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading hull table…")
    hull_data              = load_hull()
    v_ref_levels, max_speeds = build_max_speed_table(hull_data)
    print(f"  V_ref range : {v_ref_levels.min():.0f}–{v_ref_levels.max():.0f} m/s")
    print(f"  Max DS speed: {max_speeds.max():.1f} m/s  (at V_ref = {v_ref_levels.max():.0f} m/s)")

    print("\nLoading ERA5 and computing velocity potentials…")
    lat, lon, potentials, labels = load_all_frames(v_ref_levels, max_speeds)
    print(f"  Grid   : {lat.shape[0]} lat × {lon.shape[0]} lon")
    print(f"  Frames : {len(labels)}")

    print("\nBuilding GIF…")
    make_gif(lat, lon, potentials, labels)


if __name__ == "__main__":
    main()
