"""
realWind.py — ERA5-backed wind field conforming to the local Wind interface.

Position coordinates follow the artisinal/src convention:
    x[0] = East  [m]
    x[1] = North [m]
measured from a fixed geographic origin (origin_lat, origin_lon).

Time is managed via set_time(t), where t is seconds elapsed from the
start of the simulation (matching state.t in the integrator).  Trajectory.
simulate() calls set_time(state.t) at each step, so RealWind stays in sync.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from wind import Wind

# Default ERA5 data directory relative to this file
_ERA5_DIR = Path(__file__).parent / 'data' / 'era5'

# Degrees per metre (latitude direction)
_DEG_PER_M = 1.0 / 111_320.0

# ERA5 datasets and their spatial resolutions
# --------------------------------------------
# era5_1h_so_YYYY_MM_DD.nc  —  0.25° × 0.25°, 1-hourly, Southern Ocean
#     90 daily files, Dec 2022 – Feb 2023.  Spatial extent: 0° to −90°
#     latitude, full longitude.  ~46 MB per file, 24 timesteps each.
#
#     Preferred for Southern Ocean DJF migration runs:
#     - Temporal resolution (1 h) matches typical simulation step sizes and
#       produces no temporal aliasing even for long trajectories.
#     - 0.25° spatial resolution (~28 km at the equator, ~20 km at 54°S)
#       resolves mesoscale wind structure that matters for the costate ODE.
#
# era5_6h_global_YYYY_MM.nc  —  1.0° × 1.0°, 6-hourly, global
#     12 monthly files, Dec 2022 – Nov 2023.  ~36 MB per file, 120 timesteps.
#
#     Use as a fallback when the 1h SO files do not cover the required
#     region or time range:
#     - 1° spatial grid (~111 km) smooths mesoscale wind features.  For
#       position integration (greedy migration) this is acceptable, but
#       the Hamiltonian costate is sensitive to wind_grad — use the SO
#       files wherever possible.
#     - 6-hour temporal resolution introduces ±3 h interpolation error
#       relative to a 1-hourly integration step.
#
# era5_wind_{region}_YYYY_MM.nc  —  0.25° × 0.25°, single timestep
#     Regional snapshots for diagnostic wind maps only.  Not suitable for
#     time-evolving simulation.


class ERA5Interpolator:
    """
    Spatiotemporal bilinear interpolator for ERA5 hourly 10m wind.

    Loads NetCDF files sequentially (no dask) and provides fast bilinear
    spatial + linear temporal interpolation of u10, v10.

    Supports scalar or array lat/lon inputs at a single unix_t per call.
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
        time_range : Optional (t_start, t_end) pair of Unix timestamps [s].
                     Files outside this window are skipped entirely.
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

        valid_paths: list[Path] = []
        file_nt:    list[int]   = []
        lat_set = False

        for path in sorted(nc_paths):
            ds = xr.open_dataset(path)
            times_raw = ds[_time_name(ds)].values
            ts_unix   = ((times_raw - epoch) / one_s).astype(np.float64)

            if ts_unix[-1] < t_start or ts_unix[0] > t_end:
                ds.close()
                continue

            valid_paths.append(path)
            file_nt.append(len(ts_unix))

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

        n_t_total = sum(file_nt)
        n_lat     = len(self._lat)
        n_lon     = len(self._lon)

        self._u10        = np.empty((n_t_total, n_lat, n_lon), dtype=np.float32)
        self._v10        = np.empty((n_t_total, n_lat, n_lon), dtype=np.float32)
        self._unix_times = np.empty(n_t_total, dtype=np.float64)

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

        self._cached_t:       float | None               = None
        self._cached_weights: tuple[int, int, float] | None = None

    def _time_weights(self, unix_t: float):
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


class RealWind(Wind):
    """
    ERA5-backed wind field conforming to the artisinal Wind interface.

    Parameters
    ----------
    interpolator : ERA5Interpolator
        Pre-loaded ERA5 data.
    origin_lat   : float  geographic latitude  of local origin x=(0,0) [°N]
    origin_lon   : float  geographic longitude of local origin x=(0,0) [°E]
    t0_unix      : float  Unix timestamp [s] corresponding to state.t = 0
    h_default    : float  default gradient finite-difference step [m].
                   Should be ≥ one ERA5 grid cell to avoid bilinear noise:
                   ~25 000 m for 0.25° SO files, ~100 000 m for 1° global.
    """

    def __init__(
        self,
        interpolator,
        origin_lat: float,
        origin_lon: float,
        t0_unix: float,
        h_default: float = 25_000.0,
    ) -> None:
        self._interp      = interpolator
        self._origin_lat  = float(origin_lat)
        self._origin_lon  = float(origin_lon)
        self._t0_unix     = float(t0_unix)
        self._t           = 0.0          # seconds from simulation start
        self._h_default   = h_default

    # ─── Time interface ───────────────────────────────────────────────────────

    def set_time(self, t: float) -> None:
        """Advance wind clock to simulation time t [s from start]."""
        self._t = float(t)

    # ─── Position conversion ──────────────────────────────────────────────────

    def _to_geo(self, x: np.ndarray) -> tuple[float, float]:
        """Convert local Cartesian [m] → (lat, lon) [degrees]."""
        cos_lat = np.cos(np.deg2rad(self._origin_lat))
        lat = self._origin_lat + x[1] * _DEG_PER_M
        lon = self._origin_lon + x[0] * _DEG_PER_M / max(cos_lat, 1e-6)
        return float(lat), float(lon)

    def _unix_t(self) -> float:
        return self._t0_unix + self._t

    # ─── Wind interface ───────────────────────────────────────────────────────

    def velocity(self, x: np.ndarray) -> np.ndarray:
        """
        10 m wind vector [u_East, v_North] in m/s at local position x [m].

        Converts the local Cartesian position to geographic coordinates,
        queries ERA5, and returns (u10, v10) in the artisinal East/North
        frame.  system.py uses _rotation(-alpha) which correctly handles the
        clockwise-vs-CCW convention mismatch.
        """
        lat, lon = self._to_geo(x)
        u10, v10 = self._interp.query(lat, lon, self._unix_t())
        return np.array([float(u10), float(v10)])

    def gradient(self, x: np.ndarray, h: float | None = None) -> np.ndarray:
        """
        Wind Jacobian ∂(u10, v10)/∂(x_East, x_North) [m/s per m].

        Delegates to the base-class 4-point central finite-difference scheme.
        h defaults to self._h_default (set by the factory to one ERA5 grid
        cell width) rather than the base-class default of 1 km, which is
        smaller than the ERA5 grid and gives only the within-cell bilinear
        slope.
        """
        if h is None:
            h = self._h_default
        return super().gradient(x, h=h)

    # ─── Factory class methods ────────────────────────────────────────────────

    @classmethod
    def from_1h_southern_ocean(
        cls,
        origin_lat: float,
        origin_lon: float,
        t0_unix: float,
        time_range: tuple[float, float] | None = None,
        lat_range: tuple[float, float] | None = None,
        era5_dir: str | Path | None = None,
    ) -> 'RealWind':
        """
        Load era5_1h_so_*.nc files (Southern Ocean DJF suite, Dec 2022 – Feb 2023).

        Sets h_default = 25 000 m ≈ one 0.25° grid cell.  Preferred for
        Southern Ocean migration simulations.

        Parameters
        ----------
        origin_lat  : geographic latitude  of the simulation origin [°N]
        origin_lon  : geographic longitude of the simulation origin [°E]
        t0_unix     : Unix timestamp [s] at state.t = 0
        time_range  : optional (t_start, t_end) Unix timestamps to subset files
        lat_range   : optional (lat_north, lat_south) degrees to subset spatially
        era5_dir    : directory containing the NetCDF files; defaults to data/era5/
        """
        d = Path(era5_dir) if era5_dir else _ERA5_DIR
        paths = sorted(d.glob('era5_1h_so_*.nc'))
        if not paths:
            raise FileNotFoundError(f'No era5_1h_so_*.nc files found in {d}')
        interp = ERA5Interpolator(paths, lat_range=lat_range, time_range=time_range)
        return cls(interp, origin_lat, origin_lon, t0_unix, h_default=25_000.0)

    @classmethod
    def from_6h_global(
        cls,
        origin_lat: float,
        origin_lon: float,
        t0_unix: float,
        time_range: tuple[float, float] | None = None,
        lat_range: tuple[float, float] | None = None,
        era5_dir: str | Path | None = None,
    ) -> 'RealWind':
        """
        Load era5_6h_global_*.nc files (global 6-hourly suite, Dec 2022 – Nov 2023).

        Sets h_default = 100 000 m ≈ one 1° grid cell.  Use as a fallback
        when the 1h SO files do not cover the required region or time range.

        Parameters
        ----------
        origin_lat  : geographic latitude  of the simulation origin [°N]
        origin_lon  : geographic longitude of the simulation origin [°E]
        t0_unix     : Unix timestamp [s] at state.t = 0
        time_range  : optional (t_start, t_end) Unix timestamps to subset files
        lat_range   : optional (lat_north, lat_south) degrees to subset spatially
        era5_dir    : directory containing the NetCDF files; defaults to data/era5/
        """
        d = Path(era5_dir) if era5_dir else _ERA5_DIR
        paths = sorted(d.glob('era5_6h_global_*.nc'))
        if not paths:
            raise FileNotFoundError(f'No era5_6h_global_*.nc files found in {d}')
        interp = ERA5Interpolator(paths, lat_range=lat_range, time_range=time_range)
        return cls(interp, origin_lat, origin_lon, t0_unix, h_default=100_000.0)


# ─── Test / demo functions ────────────────────────────────────────────────────

_BIRD_LAT  = -54.0          # °N  (South Georgia archipelago)
_BIRD_LON  = -32.05         # °E
_T0_UNIX   = 1_672_531_200 + 86400 # 2023-01-01 00:00 UTC
_DS_THRESH = 8.6            # m/s — minimum wind speed for dynamic soaring


def test_wind_animation() -> None:
    """
    Smooth FuncAnimation of ERA5 wind speed centred on Bird Island.

    Loops through 3 days of 1-hourly Southern Ocean ERA5 data, updating
    the pcolormesh at each frame.  Same visual style as wind_explorer.py
    (Cartopy map, YlGnBu colormap, grey below DS threshold), but driven
    by FuncAnimation rather than a slider.
    """
    import datetime
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.animation as animation
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Focused extent around Bird Island [lon_min, lon_max, lat_min, lat_max]
    extent = [-60.0, -20.0, -65.0, -45.0]
    time_range = (_T0_UNIX, _T0_UNIX + 3 * 86400)

    print('Loading ERA5 (Southern Ocean, 3 days around Bird Island) …')
    wind = RealWind.from_1h_southern_ocean(
        origin_lat=_BIRD_LAT,
        origin_lon=_BIRD_LON,
        t0_unix=_T0_UNIX,
        time_range=time_range,
        lat_range=(-45.0, -65.0),
    )
    era5 = wind._interp
    print(f'  Loaded {era5._u10.shape}  ({era5._u10.nbytes / 1e6:.0f} MB)')

    proj    = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection': proj})

    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.OCEAN,     facecolor='#d0e8f5', zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor='#c8c8c8', zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='#444444', zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle='--',
                      alpha=0.5, color='#666666')
    gl.top_labels   = False
    gl.right_labels = False

    ax.scatter([_BIRD_LON], [_BIRD_LAT], marker='*', s=240, c='red',
               edgecolors='white', linewidths=0.6, zorder=8,
               transform=proj, label='Bird Island')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.85)

    base_cmap = plt.cm.YlGnBu.copy()
    base_cmap.set_under('#b8b8b8')
    norm = mcolors.Normalize(vmin=_DS_THRESH, vmax=30.0)

    lat, lon = era5._lat, era5._lon
    LON, LAT = np.meshgrid(lon, lat)

    def _speed(step: int) -> np.ndarray:
        return np.hypot(era5._u10[step].astype(np.float64),
                        era5._v10[step].astype(np.float64))

    pcm = ax.pcolormesh(LON, LAT, _speed(0), cmap=base_cmap, norm=norm,
                        transform=proj, shading='auto', zorder=1)

    cbar = fig.colorbar(pcm, ax=ax, shrink=0.75, pad=0.02, aspect=30)
    cbar.set_label('10 m wind speed  [m/s]', fontsize=11)
    cbar.ax.axhline(y=0.0, color='#cc0000', linewidth=1.2, linestyle='--')
    cbar.ax.text(2.6, 0.01, f'{_DS_THRESH} m/s', va='bottom', ha='left',
                 fontsize=8, color='#cc0000', transform=cbar.ax.transAxes)

    def _dt_str(step: int) -> str:
        return datetime.datetime.utcfromtimestamp(
            float(era5._unix_times[step])
        ).strftime('%Y-%m-%d  %H:%M UTC')

    title = ax.set_title(
        f'ERA5 10 m wind speed — {_dt_str(0)}  |  grey = < {_DS_THRESH} m/s',
        fontsize=11, pad=6,
    )

    def update(frame: int) -> None:
        pcm.set_array(_speed(frame))
        title.set_text(
            f'ERA5 10 m wind speed — {_dt_str(frame)}  |  grey = < {_DS_THRESH} m/s'
        )

    anim = animation.FuncAnimation(   # noqa: F841  (keep reference to prevent GC)
        fig, update,
        frames=len(era5._unix_times),
        interval=100,    # ms per frame — 72 frames ≈ 7 s per loop at 10 fps
        repeat=True,
        blit=False,      # blit=True conflicts with cartopy axes redraws
    )
    plt.tight_layout()
    plt.show()


def test_shooting() -> None:
    """
    Fan of 8 Hamiltonian trajectories over 1 day, starting at Bird Island.

    Mirrors test_shooting() in shooter.py with RealWind (ERA5), displayed on
    a Cartopy geographic map so South Georgia and surrounding islands are
    visible.  The ERA5 wind background uses direct slab access (array index
    only, no per-point queries) so each frame renders in < 10 ms.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.animation as animation
    from matplotlib.colors import Normalize
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from macroscale.hull import Hull
    from system import System
    from shooter import Shooter
    from integrator import Leapfrog

    dt = 900.0        # s — 15-min steps
    T  = 3 * 86400.0  # s — 1-day run

    time_range = (_T0_UNIX, _T0_UNIX + 5 * 86400)
    print('Loading ERA5 (Southern Ocean, 2 days around Bird Island) …')
    wind = RealWind.from_1h_southern_ocean(
        origin_lat=_BIRD_LAT,
        origin_lon=_BIRD_LON,
        t0_unix=_T0_UNIX,
        time_range=time_range,
        lat_range=(-41.0, -63.0),
    )

    hull       = Hull()
    system     = System(hull, wind)
    integrator = Leapfrog()
    shooter    = Shooter(system=system, integrator=integrator,
                         x0=np.array([0., 0.]),
                         dt=dt, T=T)

    print('Shooting 8 trajectories …')
    trajectories = shooter.shoot(n=36)

    era5  = wind._interp
    clat  = max(float(np.cos(np.deg2rad(wind._origin_lat))), 1e-6)

    # Convert trajectory states from local Cartesian [m] to geographic [deg]
    traj_lats, traj_lons = [], []
    for traj in trajectories:
        traj_lats.append([wind._origin_lat + s.x[1] * _DEG_PER_M        for s in traj.states])
        traj_lons.append([wind._origin_lon + s.x[0] * _DEG_PER_M / clat for s in traj.states])

    # Geographic extent fitted to trajectory bounding box
    all_lats = np.concatenate(traj_lats)
    all_lons = np.concatenate(traj_lons)
    extent = [float(all_lons.min()) - 3.0, float(all_lons.max()) + 3.0,
              float(all_lats.min()) - 2.0, float(all_lats.max()) + 2.0]

    proj    = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': proj})

    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.OCEAN,     facecolor='#d0e8f5', zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor='#c8c8c8', zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='#444444', zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle='--',
                      alpha=0.5, color='#666666')
    gl.top_labels   = False
    gl.right_labels = False
    ax.scatter([_BIRD_LON], [_BIRD_LAT], marker='*', s=200, c='red',
               edgecolors='white', linewidths=0.6, zorder=8,
               transform=proj, label='Bird Island')

    # ERA5 pcolormesh — direct slab access, no bilinear queries
    base_cmap = plt.cm.YlGnBu.copy()
    base_cmap.set_under('#b8b8b8')
    norm_wind = mcolors.Normalize(vmin=_DS_THRESH, vmax=30.0)
    LON, LAT  = np.meshgrid(era5._lon, era5._lat)

    def _slab_speed(frame: int) -> np.ndarray:
        unix_t = wind._t0_unix + frame * dt
        idx    = int(np.clip(np.searchsorted(era5._unix_times, unix_t) - 1,
                             0, len(era5._unix_times) - 1))
        return np.hypot(era5._u10[idx].astype(np.float64),
                        era5._v10[idx].astype(np.float64))

    pcm = ax.pcolormesh(LON, LAT, _slab_speed(0), cmap=base_cmap, norm=norm_wind,
                        transform=proj, shading='auto', zorder=1)
    fig.colorbar(pcm, ax=ax, shrink=0.6, pad=0.02, label='Wind speed [m/s]')

    # Trajectory lines coloured by co-state magnitude
    colors   = plt.cm.tab10(np.linspace(0, 0.9, len(trajectories)))
    lines    = [ax.plot([], [], '-', color=c, lw=1.5, alpha=0.9,
                        transform=proj, zorder=6)[0] for c in colors]
    dots     = [ax.plot([], [], 'o', color=c, ms=6, zorder=7,
                        transform=proj)[0] for c in colors]

    lam_vals = np.concatenate([[np.linalg.norm(s.lam) for s in t.states]
                               for t in trajectories])
    lam_norm = Normalize(vmin=float(lam_vals.min()),
                         vmax=float(max(lam_vals.max(), 1e-9)))
    lam_cmap = plt.cm.viridis
    lam_sm   = plt.cm.ScalarMappable(norm=lam_norm, cmap=lam_cmap)
    lam_sm.set_array([])
    fig.colorbar(lam_sm, ax=ax, label='Co-state magnitude',
                 location='left', pad=0.08, shrink=0.6)

    nframes = max(len(t.states) for t in trajectories)
    title   = ax.set_title('ERA5 Southern Ocean  —  t = 0.0 h', fontsize=11)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85)

    def update(frame: int) -> None:
        pcm.set_array(_slab_speed(frame))
        for i, (line, dot) in enumerate(zip(lines, dots)):
            end = min(frame + 1, len(traj_lats[i]))
            line.set_data(traj_lons[i][:end], traj_lats[i][:end])
            dot.set_data([traj_lons[i][end - 1]], [traj_lats[i][end - 1]])
            lam_mag = float(np.linalg.norm(trajectories[i].states[end - 1].lam))
            col = lam_cmap(lam_norm(lam_mag))
            line.set_color(col)
            dot.set_color(col)
        title.set_text(f'ERA5 Southern Ocean  —  t = {frame * dt / 3600:.1f} h')

    anim = animation.FuncAnimation(   # noqa: F841
        fig, update,
        frames=nframes,
        interval=30,
        repeat=False,
        blit=False,
    )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #test_wind_animation()
    test_shooting()
