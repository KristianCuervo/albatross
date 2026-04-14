"""
Produce seasonal-mean wind maps (DJF / MAM / JJA / SON) from the ERA5 Southern
Ocean files in data/era5/.

Each season's files are loaded, the u10/v10 fields are averaged over all
timesteps, then plotted on a South Polar Stereographic projection with:
  - pcolormesh wind-speed background (YlOrRd, DS-threshold grey)
  - quiver arrows for direction
  - circular map boundary matching run_ivp_3month.py style

Outputs:
    figures/seasonal_wind_djf.png
    figures/seasonal_wind_mam.png
    figures/seasonal_wind_jja.png
    figures/seasonal_wind_son.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Config ────────────────────────────────────────────────────────────────────
ERA5_DIR = Path(__file__).resolve().parents[2] / "data" / "era5"
FIG_DIR  = Path(__file__).resolve().parents[2] / "figures" / "macroscale"
FIG_DIR.mkdir(exist_ok=True)

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

SEASON_LABELS = {
    "DJF": "December – February",
    "MAM": "March – May",
    "JJA": "June – August",
    "SON": "September – November",
}

POLAR_LAT_CUTOFF = -20.0   # northernmost latitude shown
QUIVER_STRIDE    = 8        # arrow every N grid points (0.25° grid → 8 ≈ 2°)
DS_THRESHOLD     = 9.0      # m/s — grey below this (matches simulation)
WIND_VMAX        = 25.0     # m/s — colour-scale upper bound


# ── Helpers ───────────────────────────────────────────────────────────────────

def _season_files(months: list[int]) -> list[Path]:
    """Return all ERA5 files whose month matches any in `months`."""
    files = []
    for f in sorted(ERA5_DIR.glob("era5_1h_so_*.nc")):
        # filename: era5_1h_so_YYYY_MM_DD.nc
        parts = f.stem.split("_")   # ['era5','1h','so','YYYY','MM','DD']
        try:
            month = int(parts[4])
        except (IndexError, ValueError):
            continue
        if month in months:
            files.append(f)
    return files


def _seasonal_mean(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Open all files one-by-one and accumulate a running mean of u10/v10.

    Returns (lat, lon, u_mean, v_mean) as numpy arrays.
    """
    print(f"  Opening {len(files)} files …")
    u_acc = None
    v_acc = None
    n_steps = 0
    lat = lon = None

    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        u = ds["u10"].values   # (T, lat, lon) float32
        v = ds["v10"].values
        if u_acc is None:
            lat   = ds["latitude"].values
            lon   = ds["longitude"].values
            u_acc = u.sum(axis=0).astype(np.float64)
            v_acc = v.sum(axis=0).astype(np.float64)
        else:
            u_acc += u.sum(axis=0)
            v_acc += v.sum(axis=0)
        n_steps += u.shape[0]
        ds.close()
        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"    {i+1}/{len(files)} files processed")

    return lat, lon, u_acc / n_steps, v_acc / n_steps


def _cmap_wind():
    cm = plt.cm.YlOrRd.copy()
    cm.set_under("#d0d0d0")   # grey for sub-threshold speeds
    return cm


def _make_polar_ax(fig, pos):
    """Add a circular South Polar Stereographic axes at subplot position `pos`."""
    proj = ccrs.SouthPolarStereo()
    ax   = fig.add_subplot(*pos, projection=proj)
    ax.set_extent([-180, 180, -90, POLAR_LAT_CUTOFF], crs=ccrs.PlateCarree())
    theta  = np.linspace(0, 2 * np.pi, 200)
    circle = mpath.Path(np.column_stack([np.sin(theta), np.cos(theta)]) * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND,      facecolor="#c8c8c8", zorder=4)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5,        zorder=5)
    ax.gridlines(linewidth=0.35, linestyle="--", alpha=0.45)
    return ax


# ── Per-season plot ────────────────────────────────────────────────────────────

def plot_season_preloaded(season: str, files: list[Path], ax) -> None:
    lat, lon, u, v = _seasonal_mean(files)
    speed = np.sqrt(u**2 + v**2)

    LON, LAT = np.meshgrid(lon, lat)

    # ── Wind speed background ────────────────────────────────────────────────
    mesh = ax.pcolormesh(
        LON, LAT, speed,
        transform=ccrs.PlateCarree(),
        cmap=_cmap_wind(),
        vmin=DS_THRESHOLD,
        vmax=WIND_VMAX,
        shading="nearest",
        alpha=0.85,
        zorder=1,
    )
    plt.colorbar(mesh, ax=ax, label="Wind speed (m/s)", shrink=0.75,
                 pad=0.05, extend="both")

    # ── Quiver arrows ────────────────────────────────────────────────────────
    sl = slice(None, None, QUIVER_STRIDE)
    LO_s = LON[sl, sl]
    LA_s = LAT[sl, sl]
    U_s  = u[sl, sl]
    V_s  = v[sl, sl]

    q_scale = float(np.nanmedian(speed[speed > DS_THRESHOLD])) * 20
    Q = ax.quiver(
        LO_s, LA_s, U_s, V_s,
        transform=ccrs.PlateCarree(),
        color="white",
        scale=q_scale,
        width=0.0025,
        headwidth=3.5,
        alpha=0.9,
        zorder=3,
    )
    ax.quiverkey(Q, 0.88, 1.02, 10, r"$10\ \mathrm{m/s}$",
                 labelpos="E", coordinates="axes",
                 fontproperties={"size": 8})

    ax.set_title(f"{season}  ({SEASON_LABELS[season]})", fontsize=11)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Seasonal ERA5 wind maps")
    print(f"ERA5 dir : {ERA5_DIR}")
    print(f"Output   : {FIG_DIR}\n")

    # Determine which seasons have data
    available = {
        s: _season_files(months)
        for s, months in SEASONS.items()
    }
    available = {s: files for s, files in available.items() if files}

    if not available:
        print("No ERA5 files found. Check ERA5_DIR.")
        return

    n = len(available)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(10 * ncols, 10 * nrows))

    for i, (season, files) in enumerate(available.items()):
        row, col = divmod(i, ncols)
        pos = (nrows, ncols, i + 1)
        print(f"Season: {season}  ({len(files)} files)")
        ax = _make_polar_ax(fig, pos)
        plot_season_preloaded(season, files, ax)

    seasons_str = " / ".join(available.keys())
    fig.suptitle(
        f"ERA5 10 m seasonal-mean wind speed & direction\n"
        f"Southern Ocean  [{seasons_str}]",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    tag = "_".join(available.keys()).lower()
    out = FIG_DIR / f"seasonal_wind_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
