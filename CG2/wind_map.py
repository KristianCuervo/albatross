"""
ERA5 equatorial Atlantic wind map — vector field plotting and pointwise query.

Usage
-----
  Vector field plot (interactive):
      python wind_map.py

  Save plot to file:
      python wind_map.py --save figures/july2023.png

  Adjust arrow density (default: every 4th grid point):
      python wind_map.py --stride 6

  Pointwise wind query at a single coordinate:
      python wind_map.py --point --lat -15 --lon -30

Importable API (for albatross analysis scripts):
    from wind_map import load_wind, query_point
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

DATA_FILE = Path(__file__).parent / "data" / "era5_wind_eq_atlantic_2023_07.nc"


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_wind(path=DATA_FILE):
    """
    Load u10, v10 DataArrays from ERA5 NetCDF file.

    Returns
    -------
    ds : xr.Dataset
    u  : xr.DataArray  — 10m eastward wind, shape (lat, lon)
    v  : xr.DataArray  — 10m northward wind, shape (lat, lon)
    """
    ds = xr.open_dataset(path)

    def _get(ds, *candidates):
        for name in candidates:
            if name in ds:
                return ds[name]
        raise KeyError(
            f"None of {candidates} found. Available variables: {list(ds.data_vars)}"
        )

    u = _get(ds, "u10", "u_10m", "eastward_wind").squeeze()
    v = _get(ds, "v10", "v_10m", "northward_wind").squeeze()
    return ds, u, v


# ─── Vector field plot ────────────────────────────────────────────────────────

def plot_vector_field(stride=4, output=None, path=DATA_FILE):
    """
    Plot 10m wind as a quiver (arrow) field with wind speed as background shading.

    Parameters
    ----------
    stride : int
        Plot an arrow every Nth grid point in each direction.
        stride=1 → all 0.25° points (very dense); stride=4 → every ~1°.
    output : str or Path, optional
        If given, save to this file path instead of showing interactively.
    path : Path
        ERA5 NetCDF file to load.
    """
    _, u, v = load_wind(path)

    lat = u.latitude.values
    lon = u.longitude.values
    u_vals = u.values
    v_vals = v.values
    speed = np.sqrt(u_vals**2 + v_vals**2)

    fig, ax = plt.subplots(
        figsize=(14, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    lat_pad = (lat.max() - lat.min()) * 0.05
    lon_pad = (lon.max() - lon.min()) * 0.05
    ax.set_extent(
        [lon.min() - lon_pad, lon.max() + lon_pad,
         lat.min() - lat_pad, lat.max() + lat_pad],
        crs=ccrs.PlateCarree(),
    )

    # Background: wind speed as filled contours
    cf = ax.contourf(
        lon, lat, speed,
        levels=np.linspace(0, min(speed.max(), 12), 25),
        cmap="YlOrRd",
        transform=ccrs.PlateCarree(),
        extend="max",
    )
    plt.colorbar(cf, ax=ax, label="Wind speed (m/s)", shrink=0.75, pad=0.02)

    # Arrows: subsampled quiver — scale adapts to median wind speed
    q_scale = float(np.nanmedian(speed)) * 25
    lo_s = lon[::stride]
    la_s = lat[::stride]
    LO, LA = np.meshgrid(lo_s, la_s)
    Q = ax.quiver(
        LO, LA,
        u_vals[::stride, ::stride],
        v_vals[::stride, ::stride],
        transform=ccrs.PlateCarree(),
        color="white",
        scale=q_scale,
        width=0.0018,
        headwidth=4,
        alpha=0.85,
    )
    ax.quiverkey(
        Q, 0.92, 1.03, 10, r"$10\ \mathrm{m/s}$",
        labelpos="E", coordinates="axes", fontproperties={"size": 9},
    )

    ax.add_feature(cfeature.LAND, facecolor="#c8c8c8", zorder=5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.4, zorder=6)
    ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5, color="gray")

    ax.set_title(
        f"ERA5 10m Wind — {Path(path).stem}\n"
        f"0.25° monthly mean  •  {u_vals.shape[1]} × {u_vals.shape[0]} grid points",
        fontsize=12,
    )

    plt.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved → {output}")
    else:
        plt.show()


# ─── Pointwise query ──────────────────────────────────────────────────────────

def query_point(lat, lon, verbose=True, path=DATA_FILE):
    """
    Return wind components at the ERA5 grid point nearest to (lat, lon).

    Parameters
    ----------
    lat : float
        Latitude in degrees North (negative = south).
    lon : float
        Longitude in degrees East (negative = west).
    verbose : bool
        Print a formatted summary to stdout.

    Returns
    -------
    dict with keys:
        lat           — snapped grid latitude (°N)
        lon           — snapped grid longitude (°E)
        u10           — eastward wind component (m/s)
        v10           — northward wind component (m/s)
        speed         — wind speed magnitude (m/s)
        bearing_from  — meteorological wind direction: direction wind is coming
                        FROM, in degrees clockwise from north (0–360)
    """
    _, u, v = load_wind(path)

    u_pt = float(u.sel(latitude=lat, longitude=lon, method="nearest"))
    v_pt = float(v.sel(latitude=lat, longitude=lon, method="nearest"))
    speed = float(np.sqrt(u_pt**2 + v_pt**2))

    # Meteorological convention: bearing wind is coming FROM
    bearing_from = float((270.0 - np.degrees(np.arctan2(v_pt, u_pt))) % 360.0)

    actual_lat = float(u.latitude.sel(latitude=lat, method="nearest"))
    actual_lon = float(u.longitude.sel(longitude=lon, method="nearest"))

    result = {
        "lat": actual_lat,
        "lon": actual_lon,
        "u10": u_pt,
        "v10": v_pt,
        "speed": speed,
        "bearing_from": bearing_from,
    }

    if verbose:
        print(f"Nearest grid point : ({actual_lat:.2f}°N, {actual_lon:.2f}°E)")
        print(f"  u10          = {u_pt:+.3f} m/s  (eastward)")
        print(f"  v10          = {v_pt:+.3f} m/s  (northward)")
        print(f"  speed        = {speed:.3f} m/s")
        print(f"  bearing_from = {bearing_from:.1f}°  (wind coming from, met. convention)")

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ERA5 wind map — plot or point query"
    )
    parser.add_argument(
        "--era5", type=str, default=None, metavar="PATH",
        help="Path to ERA5 NetCDF file (default: data/era5_wind_eq_atlantic_2023_07.nc)",
    )
    parser.add_argument(
        "--point", action="store_true",
        help="Query nearest grid point instead of plotting",
    )
    parser.add_argument("--lat", type=float, default=-15.0, metavar="DEG",
                        help="Latitude in °N for --point query (default: -15)")
    parser.add_argument("--lon", type=float, default=-30.0, metavar="DEG",
                        help="Longitude in °E for --point query (default: -30)")
    parser.add_argument(
        "--stride", type=int, default=4, metavar="N",
        help="Arrow density: plot every Nth grid point (default: 4 ≈ 1° spacing)",
    )
    parser.add_argument(
        "--save", type=str, default=None, metavar="PATH",
        help="Save plot to file instead of showing (e.g. figures/july2023.png)",
    )
    args = parser.parse_args()

    era5_path = Path(args.era5) if args.era5 else DATA_FILE

    if args.point:
        query_point(args.lat, args.lon, path=era5_path)
    else:
        plot_vector_field(stride=args.stride, output=args.save, path=era5_path)
