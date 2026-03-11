"""
plot_isocurves.py
=================
Plot Hamiltonian migration iso-curves over DJF DS opportunity density background.

Layers
------
1. DJF dynamic soaring opportunity density (plasma contourf background)
2. Iso-curves at configurable hour intervals, colour-coded by elapsed time (viridis)
3. Crozet start point (white star)
4. Cartopy land mask + coastlines
5. Colourbar for elapsed time

Envelope methods
----------------
angular (default)  — connect endpoints in migration-heading order.
                     Fast, but spiky when wind field is non-uniform because
                     adjacent headings can lead to very different positions.
convex             — convex hull of the endpoint cloud at each timestep.
                     Always smooth; correctly represents the outer reachability
                     boundary; slight over-estimate of extent where concavities exist.

Output
------
    CG2/figures/migration_isocurves_crozet_dec2022.png  (unless --interactive)

Usage
-----
    python CG2/plot_isocurves.py                              # every 6 h, convex hull
    python CG2/plot_isocurves.py --iso-step 24                # one curve per day
    python CG2/plot_isocurves.py --t-max 24                   # show only first day
    python CG2/plot_isocurves.py --t-max 48 --iso-step 6      # first 2 days, every 6 h
    python CG2/plot_isocurves.py --envelope angular           # raw angle-sorted curves
    python CG2/plot_isocurves.py --interactive                 # zoom/pan window
    python CG2/plot_isocurves.py --iso-step 12 --interactive
"""

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from scipy.spatial import ConvexHull

HERE        = Path(__file__).parent
DATA_DIR    = HERE / "data"
FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DEFAULT_NPZ      = DATA_DIR / "migration_isocurves.npz"
OUT_FIGURE       = FIGURES_DIR / "migration_isocurves_crozet_dec2022.png"
DEFAULT_ISO_STEP = 6    # hours between consecutive iso-curves
DEFAULT_ENVELOPE = "convex"   # "convex" | "angular"

DS_THRESHOLD = 9.0

# DJF 6-hourly ERA5 files for background density
DJF_FILES = [
    "era5_6h_global_2022_12.nc",
    "era5_6h_global_2023_01.nc",
    "era5_6h_global_2023_02.nc",
]


# ─── Density background ───────────────────────────────────────────────────────

def _get_var(ds: xr.Dataset, *candidates: str) -> xr.DataArray:
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(f"None of {candidates!r} found. Available: {list(ds.data_vars)}")


def compute_djf_density() -> xr.DataArray:
    density_sum = None
    n_timesteps = 0
    lat_arr = lon_arr = None

    for fname in DJF_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            sys.exit(
                f"ERROR: Missing DJF file: {path}\n"
                "Run: python CG2/download_era5_6h.py --season djf"
            )
        print(f"    {fname}…", end=" ", flush=True)
        ds = xr.open_dataset(path)
        u  = _get_var(ds, "u10", "u_10m", "eastward_wind")
        v  = _get_var(ds, "v10", "v_10m", "northward_wind")

        time_dim = "valid_time" if "valid_time" in u.dims else "time"
        n = u.sizes[time_dim]

        speed = np.sqrt(u.values ** 2 + v.values ** 2)
        above = (speed > DS_THRESHOLD).sum(axis=0).astype(np.float64)

        if density_sum is None:
            density_sum = above
            lat_arr     = u.latitude.values
            lon_arr     = u.longitude.values
        else:
            density_sum += above

        n_timesteps += n
        ds.close()
        print(f"{n} timesteps")

    return xr.DataArray(
        density_sum / n_timesteps,
        dims=["latitude", "longitude"],
        coords={"latitude": lat_arr, "longitude": lon_arr},
    )


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _convex_hull_curve(lat_pts: np.ndarray, lon_pts: np.ndarray):
    """
    Return (lon, lat) of the convex hull boundary of a point cloud.
    Handles longitude wrap-around by shifting to a centred coordinate frame.
    """
    # Shift longitudes so the cloud is centred near 0° (avoids wrap-around splits)
    lon_centre = float(np.arctan2(
        np.mean(np.sin(np.deg2rad(lon_pts))),
        np.mean(np.cos(np.deg2rad(lon_pts))),
    ) * 180 / np.pi)
    lon_shifted = ((lon_pts - lon_centre + 180) % 360) - 180

    pts_2d = np.column_stack([lon_shifted, lat_pts])
    if len(pts_2d) < 3:
        return lon_pts, lat_pts

    try:
        hull = ConvexHull(pts_2d)
    except Exception:
        return lon_pts, lat_pts

    verts = hull.vertices
    # Order vertices counter-clockwise and close the loop
    hull_lon = lon_shifted[verts]
    hull_lat = lat_pts[verts]
    hull_lon = np.append(hull_lon, hull_lon[0])
    hull_lat = np.append(hull_lat, hull_lat[0])

    # Shift back to original longitude frame
    hull_lon = (hull_lon + lon_centre + 180) % 360 - 180
    return hull_lon, hull_lat


def plot_isocurves(
    isocurve_npz: Path,
    iso_step_h: int = DEFAULT_ISO_STEP,
    t_max_h: int | None = None,
    envelope: str = DEFAULT_ENVELOPE,
    interactive: bool = False,
) -> None:
    # ── Load iso-curve data ──────────────────────────────────────────────────
    if not isocurve_npz.exists():
        sys.exit(
            f"ERROR: Iso-curve data not found: {isocurve_npz}\n"
            "Run: python CG2/migration_hamiltonian.py"
        )
    print(f"Loading iso-curve data from {isocurve_npz.name}…")
    raw_data   = np.load(isocurve_npz)
    positions  = raw_data["positions"]    # (n_steps+1, N_dirs, 2)
    directions = raw_data["directions"]   # (N_dirs, 2)
    times_unix = raw_data["times"]        # (n_steps+1,)
    start_pos  = raw_data["start"]        # (2,) [lat, lon]

    n_steps, n_dirs, _ = positions.shape
    dt_s  = float(times_unix[1] - times_unix[0]) if len(times_unix) > 1 else 3600.0
    t_max = int(round((n_steps - 1) * dt_s / 3600))   # total hours covered

    # Build list of iso-curve hours, optionally capped by --t-max
    cap = min(t_max_h, t_max) if t_max_h is not None else t_max
    iso_hours = list(range(iso_step_h, cap + 1, iso_step_h))
    if not iso_hours:
        sys.exit(
            f"ERROR: --iso-step {iso_step_h} h exceeds requested range "
            f"(--t-max {t_max_h} h, simulation length {t_max} h)."
        )
    n_iso = len(iso_hours)
    print(f"  {n_iso} iso-curves at every {iso_step_h} h  "
          f"(up to {iso_hours[-1]} h)  envelope={envelope}")

    # ── Compute DJF density background ──────────────────────────────────────
    print("Computing DJF DS opportunity density…")
    density = compute_djf_density()
    bg_lat  = density.latitude.values
    bg_lon  = density.longitude.values
    bg_data = density.values

    # ── Direction sort order for closed-loop curves ──────────────────────────
    dir_angles = np.arctan2(directions[:, 1], directions[:, 0])
    sort_order = np.argsort(dir_angles)

    # ── Colourmap for iso-curves (viridis, mapped to elapsed hours) ──────────
    iso_cmap = plt.get_cmap("viridis")
    iso_norm = mcolors.Normalize(vmin=iso_hours[0], vmax=iso_hours[-1])

    # ── Figure ───────────────────────────────────────────────────────────────
    if interactive:
        plt.ion()

    fig, ax = plt.subplots(
        figsize=(16, 9),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_extent([-180, 180, -80, 0], crs=ccrs.PlateCarree())

    # Layer 1: DJF density background
    cf = ax.contourf(
        bg_lon, bg_lat, bg_data,
        levels=np.linspace(0, 1, 51),
        cmap="plasma",
        transform=ccrs.PlateCarree(),
        extend="neither",
    )
    cbar_bg = plt.colorbar(
        cf, ax=ax, shrink=0.55, pad=0.01, orientation="vertical",
        label="DJF DS opportunity (fraction of time > 9 m/s)",
    )
    cbar_bg.set_ticks(np.arange(0, 1.1, 0.1))

    # Layer 2: Iso-curves
    for k, h in enumerate(iso_hours):
        step = int(round(h * 3600.0 / dt_s))
        if step >= n_steps:
            print(f"  Warning: hour {h} exceeds available steps; skipping.")
            continue

        lat_pts = positions[step, :, 0]
        lon_pts = positions[step, :, 1]

        if envelope == "convex":
            lon_c, lat_c = _convex_hull_curve(lat_pts, lon_pts)
        else:
            # angular: connect endpoints in migration-heading order
            pts_sorted = positions[step, sort_order, :]
            lat_c = np.append(pts_sorted[:, 0], pts_sorted[0, 0])
            lon_c = np.append(pts_sorted[:, 1], pts_sorted[0, 1])

        color = iso_cmap(iso_norm(h))
        lw = 2.2 if h % 24 == 0 else 1.0
        ax.plot(
            lon_c, lat_c,
            color=color, linewidth=lw,
            transform=ccrs.PlateCarree(),
            zorder=10 + k,
        )

    # Iso-curve elapsed-time colourbar
    iso_sm = plt.cm.ScalarMappable(cmap=iso_cmap, norm=iso_norm)
    iso_sm.set_array([])
    cbar_iso = plt.colorbar(
        iso_sm, ax=ax, shrink=0.45, pad=0.07, orientation="vertical",
        label="Elapsed time (h)",
    )
    # Tick at every 24 h (day boundary); fall back to every iso_step if < 24 h span
    day_ticks = [h for h in iso_hours if h % 24 == 0]
    if day_ticks:
        cbar_iso.set_ticks(day_ticks)
        cbar_iso.set_ticklabels([f"{h} h\n(day {h//24})" for h in day_ticks])
    else:
        cbar_iso.set_ticks(iso_hours)
        cbar_iso.set_ticklabels([f"{h} h" for h in iso_hours])

    # Layer 3: Start point
    ax.plot(
        float(start_pos[1]), float(start_pos[0]),
        marker="*", markersize=14, color="white",
        markeredgecolor="black", markeredgewidth=0.7,
        transform=ccrs.PlateCarree(), zorder=20,
        label="Crozet (start)",
    )
    ax.annotate(
        "Crozet",
        xy=(float(start_pos[1]), float(start_pos[0])),
        xytext=(float(start_pos[1]) + 3, float(start_pos[0]) + 2),
        color="white", fontsize=9,
        transform=ccrs.PlateCarree(), zorder=21,
    )

    # Layer 4: Land + coastlines
    ax.add_feature(cfeature.LAND, facecolor="#404040", zorder=15)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="white", zorder=16)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--",
                      alpha=0.5, color="gray")
    gl.top_labels   = False
    gl.right_labels = False

    ax.set_title(
        "Hamiltonian Migration Iso-Curves — Crozet Island\n"
        f"ERA5 hourly 10m wind, 0.25°  •  DS hull support  •  "
        f"{n_dirs} directions  •  iso-step {iso_step_h} h  •  envelope: {envelope}",
        fontsize=12,
    )

    ax.legend(loc="lower left", fontsize=9, framealpha=0.6)

    plt.tight_layout()

    if interactive:
        plt.show(block=True)
    else:
        plt.savefig(OUT_FIGURE, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved → {OUT_FIGURE}")

    # ── Quick stats ──────────────────────────────────────────────────────────
    for h in [h for h in [24, 72, 168] if h in iso_hours]:
        step = int(round(h * 3600.0 / dt_s))
        if step >= n_steps:
            continue
        pts = positions[step]
        print(f"  h={h:3d} ({h//24}d): "
              f"centroid ({pts[:,0].mean():.1f}°N, {pts[:,1].mean():.1f}°E)  "
              f"Δlat={pts[:,0].max()-pts[:,0].min():.1f}°  "
              f"Δlon={pts[:,1].max()-pts[:,1].min():.1f}°")


# ─── Entry point ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Plot Hamiltonian migration iso-curves",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--isocurve-npz", type=Path, default=DEFAULT_NPZ,
        metavar="PATH", help="Path to migration_isocurves.npz",
    )
    p.add_argument(
        "--iso-step", type=int, default=DEFAULT_ISO_STEP,
        metavar="H", help="Hours between consecutive iso-curves",
    )
    p.add_argument(
        "--t-max", type=int, default=None,
        metavar="H",
        help="Stop iso-curves at this elapsed hour (default: full simulation length). "
             "E.g. --t-max 24 shows only the first day.",
    )
    p.add_argument(
        "--envelope", choices=["convex", "angular"], default=DEFAULT_ENVELOPE,
        help="How to draw each iso-curve: 'convex' = convex hull of endpoints (smooth); "
             "'angular' = connect in migration-heading order (can be spiky).",
    )
    p.add_argument(
        "--interactive", action="store_true",
        help="Show interactive zoomable window instead of saving PNG",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_isocurves(
        isocurve_npz = args.isocurve_npz,
        iso_step_h   = args.iso_step,
        t_max_h      = args.t_max,
        envelope     = args.envelope,
        interactive  = args.interactive,
    )
