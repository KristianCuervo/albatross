"""
albatross.macroscale.migration — greedy Hamiltonian IVP migration simulation.

The DS cycle is energy-conserving; the macroscale dynamics are formulated as
an optimal-control problem whose Pontryagin Hamiltonian is:

    H = p_x · ẋ + p_y · ẏ

where costate p = (p_x, p_y) = (c_x, c_y) is a fixed unit migration direction.

For each direction d swept over the full unit circle, the bird greedily
maximises ground speed projected onto d at every time step — this is the
support function of the local DS velocity hull.  Integrating for N_dirs
directions simultaneously (vectorised) yields the reachability envelope;
endpoints at t = 1 … 7 days form "iso-curves" of migration distance.

Coordinate conventions
-----------------------
Wind frame:
  u — crosswind (m/s, +ve = right when facing upwind)
  v — upwind    (m/s, +ve = into the wind)

Geographic frame (ERA5):
  x = eastward  (u10 axis)
  y = northward  (v10 axis)

α = arctan2(−v10, −u10)   direction FROM which wind blows, CCW from East

Rotation wind-frame → geographic:
  vx_geo = v_wind · cos(α) − u_wind · sin(α)
  vy_geo = v_wind · sin(α) + u_wind · cos(α)

No-DS condition
---------------
If V_ref = |wind| < DS_THRESHOLD (9 m/s), velocity is set to zero.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

DS_THRESHOLD = 9.0   # m/s — minimum V_ref for dynamic soaring


class GreedyMigration:
    """
    Vectorised RK4 greedy IVP migration isocurve simulator.

    Parameters
    ----------
    hull  : VelocityHull  (or dict with keys v_ref_levels, hull_radii, hull_angles)
    era5  : ERA5Interpolator
    """

    def __init__(self, hull, era5):
        # Accept either VelocityHull instance or plain dict
        if hasattr(hull, 'as_dict'):
            self._hull = hull.as_dict()
        else:
            self._hull = hull
        self._era5 = era5

        ha = self._hull["hull_angles"]
        self._sin_angles = np.sin(ha)
        self._cos_angles = np.cos(ha)

    # ------------------------------------------------------------------
    # Core RHS (vectorised over all n_dirs directions simultaneously)
    # ------------------------------------------------------------------

    def _interpolate_hull_vec(self, v_ref_arr: np.ndarray) -> np.ndarray:
        """(N,) → (N, n_rays)  hull radii, NaN replaced with 0."""
        levels = self._hull["v_ref_levels"]
        radii  = self._hull["hull_radii"]
        v      = np.clip(v_ref_arr, levels.min(), levels.max())
        idx_f  = np.interp(v, levels, np.arange(len(levels)))
        lo     = np.floor(idx_f).astype(int)
        hi     = np.minimum(lo + 1, len(levels) - 1)
        frac   = idx_f - lo
        result = (1.0 - frac[:, None]) * radii[lo] + frac[:, None] * radii[hi]
        return np.nan_to_num(result, nan=0.0)

    def _rhs(
        self,
        positions: np.ndarray,
        t: float,
        directions: np.ndarray,
        n_dirs: int,
    ) -> np.ndarray:
        """
        d(lat, lon)/dt [degrees/second] for all n_dirs trajectories.

        Returns
        -------
        deriv : (n_dirs, 2)  columns are (dlat/dt, dlon/dt) in degrees/second
        """
        lat_q = np.clip(positions[:, 0], -90.0, 90.0)
        lon_q = (positions[:, 1] + 180.0) % 360.0 - 180.0

        u10_arr, v10_arr = self._era5.query(lat_q, lon_q, t)

        alpha_arr = np.arctan2(-v10_arr, -u10_arr)
        V_ref_arr = np.hypot(u10_arr, v10_arr)
        ds_mask   = V_ref_arr >= DS_THRESHOLD

        V_ref_clipped = np.where(ds_mask, V_ref_arr, DS_THRESHOLD)
        radii_arr     = self._interpolate_hull_vec(V_ref_clipped)   # (N, n_rays)

        d_x      = directions[:, 0]
        d_y      = directions[:, 1]
        d_v_wind =  np.cos(alpha_arr) * d_x + np.sin(alpha_arr) * d_y
        d_u_wind = -np.sin(alpha_arr) * d_x + np.cos(alpha_arr) * d_y

        dot = radii_arr * (
            d_u_wind[:, None] * self._sin_angles[None, :]
            + d_v_wind[:, None] * self._cos_angles[None, :]
        )

        best_idx  = np.argmax(dot, axis=1)
        r_opt     = radii_arr[np.arange(n_dirs), best_idx]
        theta_opt = self._hull["hull_angles"][best_idx]

        u_opt = r_opt * np.sin(theta_opt)
        v_opt = r_opt * np.cos(theta_opt)

        vx_geo = v_opt * np.cos(alpha_arr) - u_opt * np.sin(alpha_arr)
        vy_geo = v_opt * np.sin(alpha_arr) + u_opt * np.cos(alpha_arr)

        vx_geo = np.where(ds_mask, vx_geo, 0.0)
        vy_geo = np.where(ds_mask, vy_geo, 0.0)

        cos_lat = np.cos(np.deg2rad(lat_q))
        dlat_dt = vy_geo / 111320.0
        dlon_dt = vx_geo / (111320.0 * np.maximum(cos_lat, 1e-6))

        return np.column_stack([dlat_dt, dlon_dt])   # (N, 2) deg/s

    # ------------------------------------------------------------------
    # RK4 integrator
    # ------------------------------------------------------------------

    def run(
        self,
        start: tuple[float, float],
        start_time: float | datetime,
        n_steps: int = 168,
        dt: float = 3600.0,
        n_dirs: int = 360,
    ) -> np.ndarray:
        """
        Integrate migration IVP for all n_dirs directions with RK4.

        Parameters
        ----------
        start      : (lat, lon) start position [°N, °E]
        start_time : Unix timestamp [s] or datetime (UTC)
        n_steps    : number of integration steps (168 = 7 days at dt=3600 s)
        dt         : time step [s]
        n_dirs     : number of migration directions swept over [0, 2π)

        Returns
        -------
        positions : (n_steps+1, n_dirs, 2)  (lat, lon) for every step × direction
        """
        if isinstance(start_time, datetime):
            start_unix = start_time.replace(tzinfo=timezone.utc).timestamp()
        else:
            start_unix = float(start_time)

        theta_dirs = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
        directions = np.column_stack([np.cos(theta_dirs), np.sin(theta_dirs)])

        positions = np.tile(np.array(start, dtype=np.float64), (n_dirs, 1))

        out     = np.empty((n_steps + 1, n_dirs, 2), dtype=np.float64)
        out[0]  = positions

        rhs_kw = dict(directions=directions, n_dirs=n_dirs)

        for step in range(n_steps):
            t = start_unix + step * dt

            k1 = self._rhs(positions,               t,          **rhs_kw)
            k2 = self._rhs(positions + 0.5*dt * k1, t + 0.5*dt, **rhs_kw)
            k3 = self._rhs(positions + 0.5*dt * k2, t + 0.5*dt, **rhs_kw)
            k4 = self._rhs(positions +     dt * k3, t +     dt, **rhs_kw)

            positions = positions + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            positions[:, 0] = np.clip(positions[:, 0], -90.0, 90.0)
            positions[:, 1] = (positions[:, 1] + 180.0) % 360.0 - 180.0

            out[step + 1] = positions

            elapsed_h = (step + 1) * dt / 3600.0
            total_h   = n_steps * dt / 3600.0
            if abs(elapsed_h % 24) < dt / 3600.0 * 0.5:
                print(f"  Day {elapsed_h/24:.1f} / {total_h/24:.1f}  complete")

        self._last_positions  = out
        self._last_directions = directions
        self._last_start      = start
        self._last_start_unix = start_unix
        self._last_dt         = dt
        self._last_n_steps    = n_steps

        return out

    def save(self, path: str | Path) -> None:
        """Save isocurve data from the most recent run()."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        times_unix = np.array([
            self._last_start_unix + i * self._last_dt
            for i in range(self._last_n_steps + 1)
        ])
        np.savez_compressed(
            path,
            positions  = self._last_positions,
            directions = self._last_directions,
            times      = times_unix,
            start      = np.array(self._last_start),
        )
        print(f"Saved migration isocurves → {path}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_isocurves(
        self,
        positions: np.ndarray | None = None,
        days: list[int] | None = None,
        ax=None,
        save: bool = False,
        output: str | Path | None = None,
    ):
        """
        Plot migration reachability isocurves.

        Parameters
        ----------
        positions : (n_steps+1, n_dirs, 2)  from run() or loaded NPZ.
                    If None, uses the last run() result.
        days      : which day-indices to plot (default: [1, 2, 3, 5, 7]).
        """
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            HAS_CARTOPY = True
        except ImportError:
            HAS_CARTOPY = False

        import matplotlib.pyplot as plt

        if positions is None:
            positions = self._last_positions
        if days is None:
            dt       = getattr(self, '_last_dt', 3600.0)
            steps_per_day = max(1, int(round(86400.0 / dt)))
            n_steps  = positions.shape[0] - 1
            days     = [d for d in [1, 2, 3, 5, 7]
                        if d * steps_per_day <= n_steps]

        cmap   = plt.cm.viridis
        colors = {d: cmap(i / max(len(days) - 1, 1)) for i, d in enumerate(days)}

        proj = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=proj)
        else:
            fig = ax.get_figure()

        dt             = getattr(self, '_last_dt', 3600.0)
        steps_per_day  = max(1, int(round(86400.0 / dt)))
        trans          = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}

        for d in days:
            idx = d * steps_per_day
            pts = positions[idx]
            lons = pts[:, 1]
            lats = pts[:, 0]
            # Close the isocurve
            lons = np.append(lons, lons[0])
            lats = np.append(lats, lats[0])
            ax.plot(lons, lats, color=colors[d], lw=1.5, label=f'Day {d}', **trans)

        # Mark start
        start = getattr(self, '_last_start', None)
        if start is not None:
            ax.scatter([start[1]], [start[0]], c='red', s=60, zorder=5,
                       label='Start', **trans)

        if HAS_CARTOPY:
            ax.add_feature(cfeature.LAND, facecolor='#d0d0d0', zorder=3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=4)
            ax.gridlines(draw_labels=True, linewidth=0.4, linestyle='--', alpha=0.5)

        ax.legend(loc='upper right')
        ax.set_title("Migration reachability isocurves")

        if output or save:
            out = Path(output) if output else Path("figures/migration_isocurves.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f"Saved → {out}")

        return fig, ax
