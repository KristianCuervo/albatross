"""
albatross.macroscale.wind — ERA5 wind data: download, interpolation, and plotting.

Merges functionality from:
  CG2/download_era5.py      (CDS API download)
  CG2/era5_wind_interp.py   (spatiotemporal bilinear interpolation)
  CG2/wind_map.py           (Cartopy vector-field visualisation)
"""

import warnings
from pathlib import Path

import numpy as np

# ─── Predefined download regions ──────────────────────────────────────────────

DATASETS = {
    "eq_atlantic": {
        "desc":  "Equatorial Atlantic — July 2023",
        "area":  [30, -80, -30, 20],
        "year":  "2023",
        "month": "07",
        "file":  "era5_wind_eq_atlantic_2023_07.nc",
    },
    "southern_ocean": {
        "desc":  "Southern Ocean full circuit — July 2023 (SH winter)",
        "area":  [-30, -180, -65, 180],
        "year":  "2023",
        "month": "07",
        "file":  "era5_wind_southern_ocean_2023_07.nc",
    },
    "soatl": {
        "desc":  "South Atlantic colony sector — July 2023",
        "area":  [-30, -70, -65, 20],
        "year":  "2023",
        "month": "07",
        "file":  "era5_wind_soatl_2023_07.nc",
    },
    "n_atlantic": {
        "desc":  "North Atlantic — July 2023",
        "area":  [70, -80, 30, 20],
        "year":  "2023",
        "month": "07",
        "file":  "era5_wind_n_atlantic_2023_07.nc",
    },
}

_BASE_REQUEST = {
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


def download_era5(
    region: str,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Download ERA5 monthly mean 10m wind data for a predefined region.

    Requires ~/.cdsapirc with valid Copernicus CDS credentials.

    Parameters
    ----------
    region     : str   Key from DATASETS (eq_atlantic, southern_ocean, soatl, n_atlantic).
    output_dir : Path  Directory where the NetCDF file is saved.
                       Defaults to data/era5/ relative to the repo root.

    Returns
    -------
    Path to the downloaded file.
    """
    import cdsapi

    if region not in DATASETS:
        raise ValueError(f"Unknown region {region!r}. Choose from: {list(DATASETS)}")

    cfg = DATASETS[region]

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "era5"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / cfg["file"]
    if output.exists():
        print(f"  [{region}] Already exists: {output}  — delete to re-fetch.")
        return output

    request = {
        **_BASE_REQUEST,
        "year":  cfg["year"],
        "month": cfg["month"],
        "area":  cfg["area"],
    }
    print(f"  [{region}] {cfg['desc']}")
    print(f"  Submitting to CDS… (may queue for a few minutes)")
    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-single-levels-monthly-means", request, str(output))
    print(f"  Saved → {output}")
    return output


# ─── Spatiotemporal interpolator ──────────────────────────────────────────────

class ERA5Interpolator:
    """
    Spatiotemporal bilinear interpolator for ERA5 hourly 10m wind.

    Loads NetCDF files sequentially (no dask) and provides fast bilinear
    spatial + linear temporal interpolation of u10, v10.

    Supports scalar or array lat/lon inputs at a single unix_t per call.

    Parameters
    ----------
    nc_paths : list[Path]  ERA5 hourly NetCDF files to load.
    """

    def __init__(
        self,
        nc_paths: list[Path],
        lat_range: tuple[float, float] | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        nc_paths   : ERA5 hourly NetCDF files to load.
        lat_range  : Optional (lat_north, lat_south) pair to spatially subset the
                     data on load, e.g. (-20.0, -80.0) for the southern ocean.
                     Reduces memory when only a latitude band is needed.
        time_range : Optional (t_start, t_end) pair of Unix timestamps [s].
                     Files whose entire time range lies outside this window are
                     skipped entirely.  Use to avoid loading 90 daily files when
                     only a 10-day simulation window is needed.
        """
        import xarray as xr

        if not nc_paths:
            raise ValueError("No ERA5 files provided to ERA5Interpolator.")

        epoch = np.datetime64(0, "s")
        one_s = np.timedelta64(1, "s")

        t_start = float(time_range[0]) if time_range is not None else -np.inf
        t_end   = float(time_range[1]) if time_range is not None else  np.inf

        def _get(ds, *candidates):
            for name in candidates:
                if name in ds:
                    return ds[name]
            raise KeyError(f"None of {candidates!r} found. Available: {list(ds.data_vars)}")

        def _lat_name(ds):
            return "latitude" if "latitude" in ds.coords else "lat"

        def _lon_name(ds):
            return "longitude" if "longitude" in ds.coords else "lon"

        def _time_name(ds):
            return "valid_time" if "valid_time" in ds.coords else "time"

        # ── Pass 1: scan time coords, filter files, read grid shape ──────────
        # Only materialises tiny coordinate arrays (not u10/v10), so it is fast
        # even for 90 files.  This gives us the total timestep count needed to
        # pre-allocate the destination arrays in one shot, avoiding the peak RAM
        # spike caused by building an intermediate list and then concatenating.
        valid_paths: list[Path] = []
        file_nt:    list[int]   = []
        lat_set = False

        for path in sorted(nc_paths):
            ds = xr.open_dataset(path)
            times_raw = ds[_time_name(ds)].values
            ts_unix   = ((times_raw - epoch) / one_s).astype(np.float64)

            # Skip files entirely outside the requested time window
            if ts_unix[-1] < t_start or ts_unix[0] > t_end:
                ds.close()
                continue

            valid_paths.append(path)
            file_nt.append(len(ts_unix))

            # Read grid coordinates from the first valid file
            if not lat_set:
                if lat_range is not None:
                    lat_n, lat_s = max(lat_range), min(lat_range)
                    ds_sel = ds.sel({_lat_name(ds): slice(lat_n, lat_s)})
                else:
                    ds_sel = ds
                self._lat = ds_sel[_lat_name(ds_sel)].values.astype(np.float64)
                self._lon = ds_sel[_lon_name(ds_sel)].values.astype(np.float64)
                lat_set = True

            ds.close()

        if not valid_paths:
            raise ValueError(
                "No ERA5 files overlap the requested time_range "
                f"({t_start:.0f} – {t_end:.0f})."
            )

        # ── Pre-allocate destination arrays (single allocation, no peak spike) ─
        n_t_total = sum(file_nt)
        n_lat     = len(self._lat)
        n_lon     = len(self._lon)

        self._u10       = np.empty((n_t_total, n_lat, n_lon), dtype=np.float32)
        self._v10       = np.empty((n_t_total, n_lat, n_lon), dtype=np.float32)
        self._unix_times = np.empty(n_t_total, dtype=np.float64)

        # ── Pass 2: load u10/v10 into pre-allocated slices ───────────────────
        cursor = 0
        for path, nt in zip(valid_paths, file_nt):
            ds = xr.open_dataset(path)
            if lat_range is not None:
                lat_n, lat_s = max(lat_range), min(lat_range)
                ds = ds.sel({_lat_name(ds): slice(lat_n, lat_s)})

            u         = _get(ds, "u10", "u_10m", "eastward_wind")
            v         = _get(ds, "v10", "v_10m", "northward_wind")
            times_raw = ds[_time_name(ds)].values

            self._u10[cursor:cursor + nt]        = u.values.astype(np.float32)
            self._v10[cursor:cursor + nt]        = v.values.astype(np.float32)
            self._unix_times[cursor:cursor + nt] = (
                (times_raw - epoch) / one_s
            ).astype(np.float64)

            cursor += nt
            ds.close()

        self._n_lat = n_lat
        self._n_lon = n_lon
        self._dlat  = float(self._lat[1] - self._lat[0])
        self._dlon  = float(self._lon[1] - self._lon[0])

        # Time-interpolation cache — avoids redundant searchsorted when adjacent
        # RK4/RK45 stages query the same timestamp (e.g. k2 and k3 in RK4).
        self._cached_t:       float | None               = None
        self._cached_weights: tuple[int, int, float] | None = None

    def _time_weights(self, unix_t: float):
        # LRU(1) cache: adjacent RK4/RK45 stages that share the same timestamp
        # (k2 and k3 in RK4, or the merged 5-point Hamiltonian query) hit this
        # and skip the searchsorted + slab lookup entirely.
        if self._cached_t is not None and unix_t == self._cached_t:
            return self._cached_weights  # type: ignore[return-value]

        times = self._unix_times
        if unix_t < times[0]:
            warnings.warn(
                f"ERA5 query is before the loaded data (data starts {times[0]:.0f}). "
                "Clamping to first timestep.",
                stacklevel=3,
            )
            result = (0, 0, 0.0)
            self._cached_t, self._cached_weights = unix_t, result
            return result
        if unix_t >= times[-1]:
            warnings.warn(
                f"ERA5 query is after the loaded data (data ends {times[-1]:.0f}). "
                "Clamping to last timestep.",
                stacklevel=3,
            )
            n = len(times) - 1
            result = (n, n, 1.0)
            self._cached_t, self._cached_weights = unix_t, result
            return result

        lo = int(np.searchsorted(times, unix_t) - 1)
        lo = max(0, min(lo, len(times) - 2))
        hi = lo + 1
        dt = times[hi] - times[lo]
        wt = (unix_t - times[lo]) / dt if dt > 0.0 else 0.0
        result = (lo, hi, wt)
        self._cached_t, self._cached_weights = unix_t, result
        return result

    def _bilinear(self, field_lo, field_hi, wt, lat, lon):
        scalar_input = np.isscalar(lat) and np.isscalar(lon)
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))

        lon_wrapped = (lon - self._lon[0]) % 360.0 + self._lon[0]
        lat_f = (lat - self._lat[0]) / self._dlat
        lon_f = (lon_wrapped - self._lon[0]) / self._dlon

        i0 = np.clip(np.floor(lat_f).astype(int), 0, self._n_lat - 2)
        j0 = np.floor(lon_f).astype(int) % self._n_lon
        i1 = i0 + 1
        j1 = (j0 + 1) % self._n_lon

        fi = np.clip(lat_f - np.floor(lat_f), 0.0, 1.0)
        fj = np.clip(lon_f - np.floor(lon_f), 0.0, 1.0)

        # Gather only the 4 corner values per query point — avoids materialising
        # the full interpolated (n_lat × n_lon) field for just N query points.
        c00 = (1.0 - wt) * field_lo[i0, j0] + wt * field_hi[i0, j0]
        c01 = (1.0 - wt) * field_lo[i0, j1] + wt * field_hi[i0, j1]
        c10 = (1.0 - wt) * field_lo[i1, j0] + wt * field_hi[i1, j0]
        c11 = (1.0 - wt) * field_lo[i1, j1] + wt * field_hi[i1, j1]

        result = (
            (1 - fi) * (1 - fj) * c00
            + (1 - fi) * fj     * c01
            + fi * (1 - fj)     * c10
            + fi * fj           * c11
        )

        return float(result[0]) if scalar_input else result

    def query(self, lat, lon, unix_t: float):
        """
        Return interpolated (u10, v10) at given position(s) and time.

        Parameters
        ----------
        lat    : float or ndarray  latitude  [°N]
        lon    : float or ndarray  longitude [°E]
        unix_t : float             Unix timestamp [s]

        Returns
        -------
        (u10, v10) : each is float (scalar) or ndarray (array input) [m/s]
        """
        lo, hi, wt = self._time_weights(float(unix_t))
        u10 = self._bilinear(self._u10[lo], self._u10[hi], wt, lat, lon)
        v10 = self._bilinear(self._v10[lo], self._v10[hi], wt, lat, lon)
        return u10, v10


# ─── Wind map plotting ────────────────────────────────────────────────────────

def _load_wind(path: Path):
    """Load u10, v10 DataArrays from ERA5 NetCDF file."""
    import xarray as xr

    ds = xr.open_dataset(path)

    def _get(ds, *candidates):
        for name in candidates:
            if name in ds:
                return ds[name]
        raise KeyError(f"None of {candidates} found. Available: {list(ds.data_vars)}")

    u = _get(ds, "u10", "u_10m", "eastward_wind").squeeze()
    v = _get(ds, "v10", "v_10m", "northward_wind").squeeze()
    return ds, u, v


def plot_wind_map(
    path: str | Path,
    stride: int = 4,
    output: str | Path | None = None,
) -> None:
    """
    Plot ERA5 10m wind as a quiver (arrow) field with speed background shading.

    Requires cartopy.

    Parameters
    ----------
    path   : Path to ERA5 NetCDF file.
    stride : Plot an arrow every Nth grid point (default 4 ≈ 1° spacing).
    output : If given, save to this file instead of showing interactively.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    _, u, v = _load_wind(Path(path))

    lat    = u.latitude.values
    lon    = u.longitude.values
    u_vals = u.values
    v_vals = v.values
    # Collapse any remaining leading dimensions (e.g. valid_time with T > 1)
    # to get a 2D (lat, lon) field — take the mean over time.
    while u_vals.ndim > 2:
        u_vals = u_vals.mean(axis=0)
    while v_vals.ndim > 2:
        v_vals = v_vals.mean(axis=0)
    speed  = np.sqrt(u_vals**2 + v_vals**2)

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

    cf = ax.contourf(
        lon, lat, speed,
        levels=np.linspace(0, min(speed.max(), 12), 25),
        cmap="YlOrRd",
        transform=ccrs.PlateCarree(),
        extend="max",
    )
    plt.colorbar(cf, ax=ax, label="Wind speed (m/s)", shrink=0.75, pad=0.02)

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
    ax.quiverkey(Q, 0.92, 1.03, 10, r"$10\ \mathrm{m/s}$",
                 labelpos="E", coordinates="axes", fontproperties={"size": 9})

    ax.add_feature(cfeature.LAND, facecolor="#c8c8c8", zorder=5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=6)
    ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5)

    ax.set_title(f"ERA5 10m Wind — {Path(path).stem}", fontsize=12)
    plt.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved → {output}")
    else:
        plt.show()
