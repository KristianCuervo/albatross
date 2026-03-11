"""
Dynamic Soaring Opportunity Density Map — four austral seasons.

Loads ERA5 6-hourly wind data (downloaded by download_era5_6h.py), computes
the fraction of timesteps where 10m wind speed exceeds the 9 m/s dynamic
soaring threshold, and saves one density map per season.

The 9 m/s threshold matches the minimum V_ref used in the tacking diagram
(refactor/tacking_diagram.py, range 9–25 m/s).

Usage:
    python CG2/ds_opportunity_map.py                       # all four seasons
    python CG2/ds_opportunity_map.py --season djf          # single season
    python CG2/ds_opportunity_map.py --season djf --monthly # one figure per month

Output (CG2/figures/):
    Seasonal:
        ds_opportunity_density_djf.png   — Dec 2022 – Feb 2023 (austral summer)
        ds_opportunity_density_mam.png   — Mar – May 2023       (austral autumn)
        ds_opportunity_density_jja.png   — Jun – Aug 2023       (austral winter)
        ds_opportunity_density_son.png   — Sep – Nov 2023       (austral spring)
    Monthly (--monthly, default season djf):
        ds_opportunity_density_2022_12.png
        ds_opportunity_density_2023_01.png
        ds_opportunity_density_2023_02.png
        … etc.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

DATA_DIR    = Path(__file__).parent / "data"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DS_THRESHOLD = 9.0  # m/s — minimum wind speed for dynamic soaring candidacy

# Austral seasons: key → (label for title, list of data files)
SEASONS = {
    "djf": {
        "label": "Dec 2022 – Feb 2023 (Austral Summer)",
        "files": [
            "era5_6h_global_2022_12.nc",
            "era5_6h_global_2023_01.nc",
            "era5_6h_global_2023_02.nc",
        ],
    },
    "mam": {
        "label": "Mar – May 2023 (Austral Autumn)",
        "files": [
            "era5_6h_global_2023_03.nc",
            "era5_6h_global_2023_04.nc",
            "era5_6h_global_2023_05.nc",
        ],
    },
    "jja": {
        "label": "Jun – Aug 2023 (Austral Winter)",
        "files": [
            "era5_6h_global_2023_06.nc",
            "era5_6h_global_2023_07.nc",
            "era5_6h_global_2023_08.nc",
        ],
    },
    "son": {
        "label": "Sep – Nov 2023 (Austral Spring)",
        "files": [
            "era5_6h_global_2023_09.nc",
            "era5_6h_global_2023_10.nc",
            "era5_6h_global_2023_11.nc",
        ],
    },
}


def _get_var(ds: xr.Dataset, *candidates: str) -> xr.DataArray:
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(
        f"None of {candidates} found. Available: {list(ds.data_vars)}"
    )


# Human-readable labels for individual month files
MONTH_LABELS = {
    "era5_6h_global_2022_12.nc": "December 2022",
    "era5_6h_global_2023_01.nc": "January 2023",
    "era5_6h_global_2023_02.nc": "February 2023",
    "era5_6h_global_2023_03.nc": "March 2023",
    "era5_6h_global_2023_04.nc": "April 2023",
    "era5_6h_global_2023_05.nc": "May 2023",
    "era5_6h_global_2023_06.nc": "June 2023",
    "era5_6h_global_2023_07.nc": "July 2023",
    "era5_6h_global_2023_08.nc": "August 2023",
    "era5_6h_global_2023_09.nc": "September 2023",
    "era5_6h_global_2023_10.nc": "October 2023",
    "era5_6h_global_2023_11.nc": "November 2023",
}


def compute_density_from_file(path: Path) -> xr.DataArray:
    """Compute DS-opportunity density for a single monthly NetCDF file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\nRun download_era5_6h.py first."
        )
    print(f"    {path.name}…", end=" ", flush=True)
    ds = xr.open_dataset(path)
    u = _get_var(ds, "u10", "u_10m", "eastward_wind")
    v = _get_var(ds, "v10", "v_10m", "northward_wind")

    time_dim = "valid_time" if "valid_time" in u.dims else "time"
    n = u.sizes[time_dim]

    speed = np.sqrt(u.values**2 + v.values**2)
    density_np = (speed > DS_THRESHOLD).mean(axis=0).astype(np.float64)
    lat_arr = u.latitude.values
    lon_arr = u.longitude.values
    ds.close()
    print(f"{n} timesteps")

    return xr.DataArray(
        density_np,
        dims=["latitude", "longitude"],
        coords={"latitude": lat_arr, "longitude": lon_arr},
    )


def compute_density(season_key: str) -> xr.DataArray:
    """
    Compute DS-opportunity density for one season.

    Processes monthly files one at a time (no dask) to keep memory manageable.

    Returns
    -------
    xr.DataArray, shape (lat, lon), values in [0, 1]
    """
    cfg = SEASONS[season_key]
    paths = [DATA_DIR / f for f in cfg["files"]]

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files for season '{season_key}'. "
            "Run download_era5_6h.py first.\n"
            + "\n".join(f"  {p}" for p in missing)
        )

    density_sum = None
    n_timesteps = 0
    lat_arr = lon_arr = None

    for path in paths:
        print(f"    {path.name}…", end=" ", flush=True)
        ds = xr.open_dataset(path)
        u = _get_var(ds, "u10", "u_10m", "eastward_wind")
        v = _get_var(ds, "v10", "v_10m", "northward_wind")

        time_dim = "valid_time" if "valid_time" in u.dims else "time"
        n = u.sizes[time_dim]

        speed = np.sqrt(u.values**2 + v.values**2)   # (time, lat, lon)
        above = (speed > DS_THRESHOLD).sum(axis=0)    # (lat, lon)

        if density_sum is None:
            density_sum = above.astype(np.float64)
            lat_arr = u.latitude.values
            lon_arr = u.longitude.values
        else:
            density_sum += above

        n_timesteps += n
        ds.close()
        print(f"{n} timesteps")

    density_np = density_sum / n_timesteps
    return xr.DataArray(
        density_np,
        dims=["latitude", "longitude"],
        coords={"latitude": lat_arr, "longitude": lon_arr},
    )


def plot_density(density: xr.DataArray, season_key: str) -> None:
    label = SEASONS[season_key]["label"]
    lat   = density.latitude.values
    lon   = density.longitude.values
    data  = density.values

    fig, ax = plt.subplots(
        figsize=(18, 9),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    ax.set_global()

    cf = ax.contourf(
        lon, lat, data,
        levels=np.linspace(0, 1, 51),
        cmap="plasma",
        transform=ccrs.PlateCarree(),
        extend="neither",
    )
    cbar = plt.colorbar(
        cf, ax=ax,
        label="Fraction of time wind > 9 m/s (DS opportunity)",
        shrink=0.6, pad=0.03, orientation="vertical",
    )
    cbar.set_ticks(np.arange(0, 1.1, 0.1))

    ax.add_feature(cfeature.LAND, facecolor="#404040", zorder=5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="white", zorder=6)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--",
                      alpha=0.5, color="gray")
    gl.top_labels   = False
    gl.right_labels = False

    ax.set_title(
        f"Dynamic Soaring Opportunity — {label}\n"
        "ERA5 6-hourly 10m wind, 1° grid  •  threshold: 9 m/s",
        fontsize=13,
    )

    plt.tight_layout()
    output = FIGURES_DIR / f"ds_opportunity_density_{season_key}.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output}")

    # Sanity-check statistics
    so_mask = (lat >= -60) & (lat <= -40)
    tr_mask = (lat >= -15) & (lat <= 15)
    so_mean = float(np.nanmean(data[so_mask, :]))
    tr_mean = float(np.nanmean(data[tr_mask, :]))
    print(f"  Southern Ocean (40–60°S): {so_mean:.3f}  |  Tropics (15°S–15°N): {tr_mean:.3f}")


def plot_density_monthly(density: xr.DataArray, filename: str) -> None:
    """Plot and save a density map for a single month."""
    label = MONTH_LABELS.get(filename, filename)
    # Derive a short file stem for the output name, e.g. "2022_12"
    stem = filename.replace("era5_6h_global_", "").replace(".nc", "")

    lat  = density.latitude.values
    lon  = density.longitude.values
    data = density.values

    fig, ax = plt.subplots(
        figsize=(18, 9),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    ax.set_global()

    cf = ax.contourf(
        lon, lat, data,
        levels=np.linspace(0, 1, 51),
        cmap="plasma",
        transform=ccrs.PlateCarree(),
        extend="neither",
    )
    cbar = plt.colorbar(
        cf, ax=ax,
        label="Fraction of time wind > 9 m/s (DS opportunity)",
        shrink=0.6, pad=0.03, orientation="vertical",
    )
    cbar.set_ticks(np.arange(0, 1.1, 0.1))

    ax.add_feature(cfeature.LAND, facecolor="#404040", zorder=5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="white", zorder=6)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--",
                      alpha=0.5, color="gray")
    gl.top_labels   = False
    gl.right_labels = False

    ax.set_title(
        f"Dynamic Soaring Opportunity — {label}\n"
        "ERA5 6-hourly 10m wind, 1° grid  •  threshold: 9 m/s",
        fontsize=13,
    )

    plt.tight_layout()
    output = FIGURES_DIR / f"ds_opportunity_density_{stem}.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output}")

    so_mask = (lat >= -60) & (lat <= -40)
    tr_mask = (lat >= -15) & (lat <= 15)
    so_mean = float(np.nanmean(data[so_mask, :]))
    tr_mean = float(np.nanmean(data[tr_mask, :]))
    print(f"  Southern Ocean (40–60°S): {so_mean:.3f}  |  Tropics (15°S–15°N): {tr_mean:.3f}")


def plot_monthly_mean_threshold(path: Path) -> None:
    """
    Plot wind speed from a monthly-mean ERA5 file with the DS threshold contour.

    Since there is only one timestep (monthly average), there is no time
    fraction to compute. Instead the map shows actual wind speed shaded, with
    a bold contour at DS_THRESHOLD marking where DS is likely.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ds = xr.open_dataset(path)
    u = _get_var(ds, "u10", "u_10m", "eastward_wind").squeeze()
    v = _get_var(ds, "v10", "v_10m", "northward_wind").squeeze()

    lat  = u.latitude.values
    lon  = u.longitude.values
    speed = np.sqrt(u.values**2 + v.values**2)
    ds.close()

    fig, ax = plt.subplots(
        figsize=(14, 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    lat_pad = (lat.max() - lat.min()) * 0.03
    lon_pad = (lon.max() - lon.min()) * 0.02
    ax.set_extent(
        [lon.min() - lon_pad, lon.max() + lon_pad,
         lat.min() - lat_pad, lat.max() + lat_pad],
        crs=ccrs.PlateCarree(),
    )

    # Wind speed background
    cf = ax.contourf(
        lon, lat, speed,
        levels=np.linspace(0, max(speed.max(), DS_THRESHOLD + 2), 30),
        cmap="YlOrRd",
        transform=ccrs.PlateCarree(),
        extend="max",
    )
    plt.colorbar(cf, ax=ax, label="Wind speed (m/s)", shrink=0.75, pad=0.02)

    # DS threshold contour — bold white line
    ax.contour(
        lon, lat, speed,
        levels=[DS_THRESHOLD],
        colors=["white"],
        linewidths=[2.0],
        transform=ccrs.PlateCarree(),
    )
    # Invisible proxy for legend
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[Line2D([0], [0], color="white", lw=2)],
        labels=[f"DS threshold ({DS_THRESHOLD} m/s)"],
        loc="lower left", framealpha=0.6, fontsize=9,
    )

    ax.add_feature(cfeature.LAND, facecolor="#c8c8c8", zorder=5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=6)
    ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5, color="gray")

    ax.set_title(
        f"ERA5 10m Wind Speed — {path.stem}\n"
        f"Monthly mean  •  White contour = {DS_THRESHOLD} m/s DS threshold",
        fontsize=12,
    )

    plt.tight_layout()
    stem   = path.stem
    output = FIGURES_DIR / f"ds_opportunity_density_{stem}.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output}")

    pct_above = float((speed > DS_THRESHOLD).mean() * 100)
    print(f"  {pct_above:.1f}% of grid points have monthly-mean wind > {DS_THRESHOLD} m/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and plot DS opportunity density by austral season"
    )
    parser.add_argument(
        "--season", choices=list(SEASONS.keys()), default=None, metavar="SEASON",
        help=f"Process a single season. Choices: {', '.join(SEASONS)}. Default: all.",
    )
    parser.add_argument(
        "--monthly", action="store_true",
        help="Save one figure per month instead of one per season. "
             "Use with --season to limit which months are processed.",
    )
    parser.add_argument(
        "--file", metavar="PATH",
        help="Plot DS threshold map from a single monthly-mean ERA5 NetCDF file "
             "(e.g. data/era5_wind_southern_ocean_2023_07.nc).",
    )
    args = parser.parse_args()

    if args.file:
        plot_monthly_mean_threshold(Path(args.file))
    elif args.monthly:
        season_keys = [args.season] if args.season else list(SEASONS.keys())
        for skey in season_keys:
            print(f"\n[{skey.upper()}] {SEASONS[skey]['label']} — monthly breakdown")
            for fname in SEASONS[skey]["files"]:
                density = compute_density_from_file(DATA_DIR / fname)
                plot_density_monthly(density, fname)
    else:
        season_keys = [args.season] if args.season else list(SEASONS.keys())
        for key in season_keys:
            print(f"\n[{key.upper()}] {SEASONS[key]['label']}")
            density = compute_density(key)
            plot_density(density, key)

    print("\nDone.")
