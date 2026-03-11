"""
make_figures.py
===============
Generate four publication figures from existing ERA5 and simulation data.

Figures
-------
1.  global_wind_jja.png            — Global mean wind speed + direction arrows, JJA
2.  global_ds_potential_jja.png    — Global DS opportunity density, JJA (plasma)
3.  so_isocurves_7day.png          — Southern Ocean iso-curves, 7 days, white background
4.  crozet_isocurves_3day.png      — Crozet close-up, first 3 days every 6 h

Usage
-----
    python CG2/make_figures.py

All files read from CG2/data/, all figures saved to CG2/figures/.
"""

from pathlib import Path
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from scipy.spatial import ConvexHull

HERE        = Path(__file__).parent
DATA_DIR    = HERE / "data"
FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DS_THRESHOLD = 9.0  # m/s

JJA_FILES = [
    "era5_6h_global_2023_06.nc",
    "era5_6h_global_2023_07.nc",
    "era5_6h_global_2023_08.nc",
]

SEASONS = {
    "DJF": ["era5_6h_global_2022_12.nc", "era5_6h_global_2023_01.nc", "era5_6h_global_2023_02.nc"],
    "MAM": ["era5_6h_global_2023_03.nc", "era5_6h_global_2023_04.nc", "era5_6h_global_2023_05.nc"],
    "JJA": ["era5_6h_global_2023_06.nc", "era5_6h_global_2023_07.nc", "era5_6h_global_2023_08.nc"],
    "SON": ["era5_6h_global_2023_09.nc", "era5_6h_global_2023_10.nc", "era5_6h_global_2023_11.nc"],
}

SEASON_LABELS = {
    "DJF": "Dec–Feb 2022/23",
    "MAM": "Mar–May 2023",
    "JJA": "Jun–Aug 2023",
    "SON": "Sep–Nov 2023",
}

ISOCURVE_NPZ = DATA_DIR / "migration_isocurves.npz"


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _get_var(ds, *candidates):
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(f"None of {candidates} found in {list(ds.data_vars)}")


def _load_seasonal_mean(fnames: list[str]):
    """
    Compute time-mean u10, v10 over a set of 6-hourly ERA5 monthly files.
    Returns (lat, lon, u_mean, v_mean) as numpy arrays.
    """
    u_sum = v_sum = None
    n_total = 0
    lat_arr = lon_arr = None

    for fname in fnames:
        path = DATA_DIR / fname
        if not path.exists():
            sys.exit(f"ERROR: Missing ERA5 file: {path}")
        ds = xr.open_dataset(path)
        u  = _get_var(ds, "u10", "u_10m", "eastward_wind")
        v  = _get_var(ds, "v10", "v_10m", "northward_wind")
        n  = u.shape[0]
        if u_sum is None:
            u_sum   = u.values.sum(axis=0)
            v_sum   = v.values.sum(axis=0)
            lat_arr = u.latitude.values.astype(np.float64)
            lon_arr = u.longitude.values.astype(np.float64)
        else:
            u_sum += u.values.sum(axis=0)
            v_sum += v.values.sum(axis=0)
        n_total += n
        ds.close()

    return lat_arr, lon_arr, u_sum / n_total, v_sum / n_total


def _load_ds_density(fnames: list[str]):
    """
    Compute DS-opportunity density (fraction of timesteps with |wind| > 9 m/s).
    Returns (lat, lon, density) as numpy arrays.
    """
    density_sum = None
    n_total = 0
    lat_arr = lon_arr = None

    for fname in fnames:
        path = DATA_DIR / fname
        if not path.exists():
            sys.exit(f"ERROR: Missing ERA5 file: {path}")
        ds  = xr.open_dataset(path)
        u   = _get_var(ds, "u10", "u_10m", "eastward_wind")
        v   = _get_var(ds, "v10", "v_10m", "northward_wind")
        spd = np.hypot(u.values, v.values)
        above = (spd > DS_THRESHOLD).sum(axis=0).astype(np.float64)
        if density_sum is None:
            density_sum = above
            lat_arr     = u.latitude.values.astype(np.float64)
            lon_arr     = u.longitude.values.astype(np.float64)
        else:
            density_sum += above
        n_total += spd.shape[0]
        ds.close()

    return lat_arr, lon_arr, density_sum / n_total


def _convex_hull_curve(lat_pts, lon_pts):
    """
    Convex hull of an endpoint cloud; handles longitude wrap-around.
    Returns (lon_hull, lat_hull) arrays for plotting.
    """
    lon_c = float(np.arctan2(
        np.mean(np.sin(np.deg2rad(lon_pts))),
        np.mean(np.cos(np.deg2rad(lon_pts))),
    ) * 180 / np.pi)
    lon_s = ((lon_pts - lon_c + 180) % 360) - 180
    pts   = np.column_stack([lon_s, lat_pts])
    if len(pts) < 3:
        return lon_pts, lat_pts
    try:
        hull = ConvexHull(pts)
    except Exception:
        return lon_pts, lat_pts
    v = hull.vertices
    lons_h = np.append(lon_s[v], lon_s[v[0]])
    lats_h = np.append(lat_pts[v], lat_pts[v[0]])
    lons_h = (lons_h + lon_c + 180) % 360 - 180
    return lons_h, lats_h


def _angular_curve(positions_step, directions):
    """
    Connect endpoints in migration-heading order (angular sweep).
    Returns (lon_c, lat_c) closed arrays.
    """
    dir_angles = np.arctan2(directions[:, 1], directions[:, 0])
    order      = np.argsort(dir_angles)
    pts        = positions_step[order, :]
    lat_c = np.append(pts[:, 0], pts[0, 0])
    lon_c = np.append(pts[:, 1], pts[0, 1])
    return lon_c, lat_c


def _land_mask(lon_1d, lat_1d):
    """
    Return a 2-D boolean array (len(lat_1d), len(lon_1d)) that is True
    where the grid point falls on land.  Uses cartopy's 110 m Natural Earth
    land polygons via shapely for efficiency.
    """
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import MultiPolygon, Point
    from shapely.ops import unary_union

    land_shp = shpreader.natural_earth(resolution="110m", category="physical", name="land")
    reader   = shpreader.Reader(land_shp)
    land_geom = unary_union([g for g in reader.geometries()])

    mask = np.zeros((len(lat_1d), len(lon_1d)), dtype=bool)
    for j, la in enumerate(lat_1d):
        for i, lo in enumerate(lon_1d):
            mask[j, i] = land_geom.contains(Point(lo, la))
    return mask


def _add_land_coast(ax, land_color="#cccccc"):
    ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#555555", zorder=6)


def _gridlines(ax):
    gl = ax.gridlines(draw_labels=True, linewidth=0.35, linestyle="--",
                      alpha=0.6, color="gray")
    gl.top_labels   = False
    gl.right_labels = False
    return gl


# ─── Figure 1: Global JJA mean wind ──────────────────────────────────────────

def fig1_global_wind():
    print("Figure 1: Global JJA wind map…")
    lat, lon, u_mean, v_mean = _load_seasonal_mean(JJA_FILES)
    speed = np.hypot(u_mean, v_mean)

    # Subsample for arrows: every 5° on the 1° ERA5 grid
    step = 5
    lat_s  = lat[::step]
    lon_s  = lon[::step]
    u_s    = u_mean[::step, ::step]
    v_s    = v_mean[::step, ::step]
    spd_s  = speed[::step, ::step]

    # Normalise to unit length so all arrows show direction only
    mag    = np.hypot(u_s, v_s)
    u_n    = u_s / np.where(mag > 0.1, mag, 1.0)
    v_n    = v_s / np.where(mag > 0.1, mag, 1.0)

    fig, ax = plt.subplots(
        figsize=(18, 9),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    ax.set_global()

    # Background: wind speed contourf
    cf = ax.contourf(
        lon, lat, speed,
        levels=np.linspace(0, 20, 41),
        cmap="viridis",
        transform=ccrs.PlateCarree(),
        extend="max",
    )
    cbar = plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.03,
                        label="Mean wind speed (m/s)", orientation="vertical")
    cbar.set_ticks(np.arange(0, 22, 2))

    # DS-threshold contour at 9 m/s
    ax.contour(
        lon, lat, speed,
        levels=[DS_THRESHOLD],
        colors=["white"], linewidths=[1.0], linestyles=["--"],
        transform=ccrs.PlateCarree(), zorder=4,
    )

    # Direction arrows — unit-length, white, semi-transparent
    q = ax.quiver(
        lon_s, lat_s, u_n, v_n,
        transform=ccrs.PlateCarree(),
        color="white", alpha=0.65,
        scale=35, width=0.0018,
        headwidth=4, headlength=4, headaxislength=3.5,
        zorder=7,
    )

    _add_land_coast(ax, land_color="#3a3a3a")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#777777", zorder=6)

    ax.set_title(
        "ERA5 JJA Mean 10 m Wind (Jun–Aug 2023)\n"
        "Background: speed  •  Arrows: direction (unit-length, every 5°)  •  "
        "Dashed white: 9 m/s DS threshold",
        fontsize=12,
    )

    out = FIGURES_DIR / "global_wind_jja.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─── Seasonal wind maps ───────────────────────────────────────────────────────

def fig_seasonal_wind_maps():
    """Generate global mean wind maps for DJF, MAM, JJA, SON."""
    for season, fnames in SEASONS.items():
        print(f"Seasonal wind map: {season}…")
        lat, lon, u_mean, v_mean = _load_seasonal_mean(fnames)
        speed = np.hypot(u_mean, v_mean)

        # Subsample for arrows: every 5° on the 1° ERA5 grid
        step = 5
        lat_s = lat[::step]
        lon_s = lon[::step]
        u_s   = u_mean[::step, ::step]
        v_s   = v_mean[::step, ::step]

        # Normalise to unit length so arrows show direction only
        mag = np.hypot(u_s, v_s)
        u_n = u_s / np.where(mag > 0.1, mag, 1.0)
        v_n = v_s / np.where(mag > 0.1, mag, 1.0)

        # Mask land points
        on_land = _land_mask(lon_s, lat_s)
        u_n = np.where(on_land, np.nan, u_n)
        v_n = np.where(on_land, np.nan, v_n)

        fig, ax = plt.subplots(
            figsize=(18, 9),
            subplot_kw={"projection": ccrs.Robinson()},
        )
        ax.set_global()

        cf = ax.contourf(
            lon, lat, speed,
            levels=np.linspace(0, 20, 41),
            cmap="viridis",
            transform=ccrs.PlateCarree(),
            extend="max",
        )
        cbar = plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.03,
                            label="Mean wind speed (m/s)", orientation="vertical")
        cbar.set_ticks(np.arange(0, 22, 2))

        # Half-size arrows: scale doubled, width/head halved
        ax.quiver(
            lon_s, lat_s, u_n, v_n,
            transform=ccrs.PlateCarree(),
            color="white", alpha=0.65,
            scale=120, width=0.0006,
            headwidth=1.5, headlength=1.5, headaxislength=1.25,
            zorder=7,
        )

        _add_land_coast(ax, land_color="#3a3a3a")
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#777777", zorder=6)

        label = SEASON_LABELS[season]
        ax.set_title(
            f"ERA5 {season} Mean 10 m Wind ({label})\n"
            "Background: speed  •  Arrows: direction (unit-length, every 5°)",
            fontsize=12,
        )

        out = FIGURES_DIR / f"global_wind_{season.lower()}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")


# ─── Figure 2: Global DS opportunity density (JJA) ───────────────────────────

def fig2_global_ds_potential():
    print("Figure 2: Global JJA DS potential map…")
    lat, lon, density = _load_ds_density(JJA_FILES)

    fig, ax = plt.subplots(
        figsize=(18, 9),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    ax.set_global()

    cf = ax.contourf(
        lon, lat, density,
        levels=np.linspace(0, 1, 51),
        cmap="plasma",
        transform=ccrs.PlateCarree(),
        extend="neither",
    )
    cbar = plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.03,
                        label="Fraction of time with wind > 9 m/s",
                        orientation="vertical")
    cbar.set_ticks(np.arange(0, 1.1, 0.1))

    _add_land_coast(ax, land_color="#404040")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#777777", zorder=6)

    # Annotate SO mean vs tropics
    so_mask = (lat >= -60) & (lat <= -40)
    tr_mask = (lat >= -15) & (lat <= 15)
    so_mean = float(np.nanmean(density[so_mask, :]))
    tr_mean = float(np.nanmean(density[tr_mask, :]))

    ax.set_title(
        "ERA5 JJA Dynamic Soaring Opportunity (Jun–Aug 2023)\n"
        f"Southern Ocean 40–60°S: {so_mean:.2f}  •  Tropics 15°S–15°N: {tr_mean:.2f}",
        fontsize=12,
    )

    out = FIGURES_DIR / "global_ds_potential_jja.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─── Figure 3: Southern Ocean iso-curves (7 days, white background) ──────────

def fig3_so_isocurves_7day(envelope: str = "convex"):
    label = "Convex-hull" if envelope == "convex" else "Angular"
    suffix = "" if envelope == "convex" else "_angular"
    print(f"Figure 3 ({envelope}): Southern Ocean 7-day iso-curves…")
    if not ISOCURVE_NPZ.exists():
        sys.exit(f"ERROR: {ISOCURVE_NPZ} not found. Run migration_hamiltonian.py first.")

    raw        = np.load(ISOCURVE_NPZ)
    positions  = raw["positions"]   # (n_steps+1, N_dirs, 2)
    directions = raw["directions"]  # (N_dirs, 2)
    times      = raw["times"]
    start_pos  = raw["start"]

    n_steps, n_dirs, _ = positions.shape
    dt_s = float(times[1] - times[0])

    iso_hours = list(range(24, 168 + 1, 24))

    iso_cmap = plt.get_cmap("viridis")
    iso_norm = mcolors.Normalize(vmin=iso_hours[0], vmax=iso_hours[-1])

    fig, ax = plt.subplots(
        figsize=(16, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_facecolor("white")
    ax.set_extent([0, 180, -80, -20], crs=ccrs.PlateCarree())

    for k, h in enumerate(iso_hours):
        step = int(round(h * 3600.0 / dt_s))
        if step >= n_steps:
            continue
        lat_pts = positions[step, :, 0]
        lon_pts = positions[step, :, 1]

        if envelope == "convex":
            lon_c, lat_c = _convex_hull_curve(lat_pts, lon_pts)
        else:
            lon_c, lat_c = _angular_curve(positions[step], directions)

        color = iso_cmap(iso_norm(h))
        lw    = 2.5 if h % 24 == 0 else 1.2
        ax.plot(lon_c, lat_c, color=color, linewidth=lw,
                transform=ccrs.PlateCarree(), zorder=10 + k,
                label=f"Day {h // 24}")

    ax.plot(float(start_pos[1]), float(start_pos[0]),
            marker="*", markersize=16, color="red",
            markeredgecolor="black", markeredgewidth=0.8,
            transform=ccrs.PlateCarree(), zorder=25, label="Crozet")

    _add_land_coast(ax, land_color="#cccccc")

    gl = _gridlines(ax)
    gl.xlocator = mticker.MultipleLocator(20)
    gl.ylocator = mticker.MultipleLocator(10)

    iso_sm = plt.cm.ScalarMappable(cmap=iso_cmap, norm=iso_norm)
    iso_sm.set_array([])
    cbar = plt.colorbar(iso_sm, ax=ax, shrink=0.7, pad=0.02,
                        label="Elapsed time (h)", orientation="vertical")
    day_ticks = [h for h in iso_hours if h % 24 == 0]
    cbar.set_ticks(day_ticks)
    cbar.set_ticklabels([f"{h} h  (day {h//24})" for h in day_ticks])

    ax.legend(loc="lower right", fontsize=9, framealpha=0.85,
              title=f"Iso-curve ({label})")

    n_dirs_str = f"{n_dirs:,}"
    ax.set_title(
        "Hamiltonian DS Migration — Southern Ocean Reachability  (Jan 2023, Crozet)\n"
        f"ERA5 1-h 0.25° wind  •  {n_dirs_str} directions  •  "
        f"{label} envelope  •  one iso-curve per day",
        fontsize=11,
    )

    out = FIGURES_DIR / f"so_isocurves_7day{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─── Figure 4: Crozet close-up (first 3 days, every 6 h) ────────────────────

def fig4_crozet_closeup(envelope: str = "convex"):
    label  = "Convex-hull" if envelope == "convex" else "Angular"
    suffix = "" if envelope == "convex" else "_angular"
    print(f"Figure 4 ({envelope}): Crozet close-up, 3 days every 6 h…")
    if not ISOCURVE_NPZ.exists():
        sys.exit(f"ERROR: {ISOCURVE_NPZ} not found. Run migration_hamiltonian.py first.")

    raw        = np.load(ISOCURVE_NPZ)
    positions  = raw["positions"]
    directions = raw["directions"]
    times      = raw["times"]
    start_pos  = raw["start"]

    n_steps, n_dirs, _ = positions.shape
    dt_s  = float(times[1] - times[0])
    t_max = min(72, int(round((n_steps - 1) * dt_s / 3600)))

    iso_hours = list(range(6, t_max + 1, 6))

    iso_cmap = plt.get_cmap("plasma")
    iso_norm = mcolors.Normalize(vmin=0, vmax=t_max)

    fig, ax = plt.subplots(
        figsize=(14, 9),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_facecolor("white")

    # Extent from day-3 cloud + margin
    step3 = int(round(t_max * 3600.0 / dt_s))
    if step3 >= n_steps:
        step3 = n_steps - 1
    p3      = positions[step3]
    lat_min = max(-80, p3[:, 0].min() - 5)
    lat_max = min(-15, p3[:, 0].max() + 5)
    lon_min = p3[:, 1].min() - 8
    lon_max = p3[:, 1].max() + 8
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    for k, h in enumerate(iso_hours):
        step = int(round(h * 3600.0 / dt_s))
        if step >= n_steps:
            continue
        lat_pts = positions[step, :, 0]
        lon_pts = positions[step, :, 1]

        if envelope == "convex":
            lon_c, lat_c = _convex_hull_curve(lat_pts, lon_pts)
        else:
            lon_c, lat_c = _angular_curve(positions[step], directions)

        color = iso_cmap(iso_norm(h))
        lw    = 2.8 if h % 24 == 0 else 1.0
        alpha = 1.0 if h % 24 == 0 else 0.75
        ax.plot(lon_c, lat_c, color=color, linewidth=lw, alpha=alpha,
                transform=ccrs.PlateCarree(), zorder=10 + k)

        if h % 24 == 0:
            ax.text(float(np.median(lon_c)) + 1, float(np.median(lat_c)) + 0.5,
                    f"Day {h//24}", fontsize=9, color=color, fontweight="bold",
                    transform=ccrs.PlateCarree(), zorder=30, ha="left")

    ax.plot(float(start_pos[1]), float(start_pos[0]),
            marker="*", markersize=18, color="gold",
            markeredgecolor="black", markeredgewidth=1.0,
            transform=ccrs.PlateCarree(), zorder=25, label="Crozet")

    _add_land_coast(ax, land_color="#cccccc")

    gl = _gridlines(ax)
    gl.xlocator = mticker.MultipleLocator(10)
    gl.ylocator = mticker.MultipleLocator(5)

    iso_sm = plt.cm.ScalarMappable(cmap=iso_cmap, norm=iso_norm)
    iso_sm.set_array([])
    cbar = plt.colorbar(iso_sm, ax=ax, shrink=0.75, pad=0.02,
                        label="Elapsed time (h)", orientation="vertical")
    cbar.set_ticks([h for h in iso_hours if h % 12 == 0])
    cbar.set_ticklabels([f"{h} h" for h in iso_hours if h % 12 == 0])

    ax.legend(loc="upper left", fontsize=10, framealpha=0.85)

    n_dirs_str = f"{n_dirs:,}"
    ax.set_title(
        f"Crozet Close-up — First {t_max} h, every 6 h  (Jan 2023)\n"
        f"ERA5 1-h 0.25°  •  {n_dirs_str} directions  •  {label} envelope",
        fontsize=11,
    )

    out = FIGURES_DIR / f"crozet_isocurves_3day{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig_seasonal_wind_maps()
    fig1_global_wind()
    fig2_global_ds_potential()
    for env in ("convex", "angular"):
        fig3_so_isocurves_7day(envelope=env)
        fig4_crozet_closeup(envelope=env)
    print("\nAll figures saved to", FIGURES_DIR)
