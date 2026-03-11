"""
ERA5 hourly wind spatiotemporal interpolator for migration IVP simulation.

Loads pre-downloaded hourly ERA5 files (era5_1h_so_*.nc) and provides
bilinear spatial + linear temporal interpolation of u10, v10 at arbitrary
(lat, lon, unix_time) query points.

Coordinate conventions
----------------------
ERA5 latitude   : descending (0° → −90°) for Southern Ocean files
ERA5 longitude  : ascending  (−180° → +180°) — wrap-around handled
Time axis       : sorted ascending Unix seconds

Usage
-----
    from pathlib import Path
    from CG2.era5_wind_interp import ERA5WindInterpolator

    nc_paths = sorted(Path("CG2/data").glob("era5_1h_so_*.nc"))
    interp = ERA5WindInterpolator(nc_paths)

    # Scalar query — Crozet, 2022-12-01 00:00 UTC
    u10, v10 = interp.query(-46.4, 52.0, 1669852800)

    # Array query — 360 positions at once
    import numpy as np
    lats = np.full(360, -46.4)
    lons = np.linspace(0, 360, 360, endpoint=False)
    u10_arr, v10_arr = interp.query(lats, lons, 1669852800)
"""

from pathlib import Path

import numpy as np
import xarray as xr


class ERA5WindInterpolator:
    """
    Spatiotemporal interpolator for ERA5 hourly 10m wind.

    Loads nc_paths sequentially with xarray (no dask), preloads u10 and v10
    as float32 numpy arrays for fast bilinear + linear interpolation.

    Supports scalar or array lat/lon inputs at a single unix_t per call.
    """

    def __init__(self, nc_paths: list[Path]) -> None:
        if not nc_paths:
            raise ValueError("No ERA5 files provided to ERA5WindInterpolator.")

        def _get(ds, *candidates):
            """Try variable name candidates; raise KeyError if none found."""
            for name in candidates:
                if name in ds:
                    return ds[name]
            raise KeyError(
                f"None of {candidates!r} found. Available: {list(ds.data_vars)}"
            )

        all_u, all_v, all_times = [], [], []
        lat_set = lon_set = False

        for path in sorted(nc_paths):
            ds = xr.open_dataset(path)
            u = _get(ds, "u10", "u_10m", "eastward_wind")
            v = _get(ds, "v10", "v_10m", "northward_wind")

            # Time coordinate (valid_time or time, as ERA5 uses both)
            time_coord = "valid_time" if "valid_time" in ds.coords else "time"
            times_raw = ds[time_coord].values  # numpy datetime64

            all_u.append(u.values.astype(np.float32))
            all_v.append(v.values.astype(np.float32))
            all_times.append(times_raw)

            if not lat_set:
                lat_name = "latitude" if "latitude" in ds.coords else "lat"
                lon_name = "longitude" if "longitude" in ds.coords else "lon"
                self._lat = ds[lat_name].values.astype(np.float64)
                self._lon = ds[lon_name].values.astype(np.float64)
                lat_set = lon_set = True

            ds.close()

        # Stack all files along time axis → (T, lat, lon)
        self._u10 = np.concatenate(all_u, axis=0)
        self._v10 = np.concatenate(all_v, axis=0)

        # Convert datetime64 → Unix float seconds
        epoch = np.datetime64(0, "s")
        one_s = np.timedelta64(1, "s")
        self._unix_times = (
            (np.concatenate(all_times, axis=0) - epoch) / one_s
        ).astype(np.float64)

        # Grid metadata for bilinear lookup
        self._n_lat = len(self._lat)
        self._n_lon = len(self._lon)
        self._dlat  = float(self._lat[1] - self._lat[0])  # negative for ERA5 N→S
        self._dlon  = float(self._lon[1] - self._lon[0])

    # ─── Internal helpers ──────────────────────────────────────────────────────

    def _time_weights(self, unix_t: float):
        """Return (lo_index, hi_index, hi_weight) for linear time interp."""
        times = self._unix_times
        if unix_t <= times[0]:
            return 0, 0, 0.0
        if unix_t >= times[-1]:
            n = len(times) - 1
            return n, n, 1.0
        lo = int(np.searchsorted(times, unix_t) - 1)
        lo = max(0, min(lo, len(times) - 2))
        hi = lo + 1
        dt = times[hi] - times[lo]
        wt = (unix_t - times[lo]) / dt if dt > 0.0 else 0.0
        return lo, hi, wt

    def _bilinear(
        self,
        field_lo: np.ndarray,
        field_hi: np.ndarray,
        wt: float,
        lat,
        lon,
    ):
        """
        Bilinear spatial interpolation of temporally blended field.

        Parameters
        ----------
        field_lo, field_hi : (n_lat, n_lon) numpy arrays
        wt                 : linear weight for field_hi in [0, 1]
        lat, lon           : scalar or ndarray of query positions

        Returns
        -------
        Scalar float if lat/lon are scalar, else ndarray of same shape.
        """
        scalar_input = np.isscalar(lat) and np.isscalar(lon)
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))

        # Temporal blend (done once, avoids two separate bilinear calls)
        field = (1.0 - wt) * field_lo + wt * field_hi  # (n_lat, n_lon)

        # Wrap longitude into the grid's first-column domain
        lon_wrapped = (lon - self._lon[0]) % 360.0 + self._lon[0]

        # Fractional indices (dlat may be negative → ascending index toward S)
        lat_f = (lat - self._lat[0]) / self._dlat
        lon_f = (lon_wrapped - self._lon[0]) / self._dlon

        # Integer floor indices
        i0 = np.clip(np.floor(lat_f).astype(int), 0, self._n_lat - 2)
        j0 = np.floor(lon_f).astype(int) % self._n_lon

        i1 = i0 + 1
        j1 = (j0 + 1) % self._n_lon

        # Sub-cell weights
        fi = np.clip(lat_f - np.floor(lat_f), 0.0, 1.0)
        fj = np.clip(lon_f - np.floor(lon_f), 0.0, 1.0)

        # Four-point bilinear combination
        result = (
            (1 - fi) * (1 - fj) * field[i0, j0]
            + (1 - fi) * fj       * field[i0, j1]
            + fi       * (1 - fj) * field[i1, j0]
            + fi       * fj       * field[i1, j1]
        )

        return float(result[0]) if scalar_input else result

    # ─── Public interface ──────────────────────────────────────────────────────

    def query(self, lat, lon, unix_t: float):
        """
        Return interpolated (u10, v10) at given position(s) and time.

        Parameters
        ----------
        lat      : float or ndarray   geodetic latitude  [°N]
        lon      : float or ndarray   longitude          [°E]
        unix_t   : float              Unix timestamp     [s]

        Returns
        -------
        (u10, v10) : tuple
            Each element is a float (scalar input) or ndarray (array input).
            u10 = eastward  10m wind component [m/s]
            v10 = northward 10m wind component [m/s]
        """
        lo, hi, wt = self._time_weights(float(unix_t))
        u10 = self._bilinear(self._u10[lo], self._u10[hi], wt, lat, lon)
        v10 = self._bilinear(self._v10[lo], self._v10[hi], wt, lat, lon)
        return u10, v10
