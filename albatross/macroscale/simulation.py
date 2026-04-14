"""
albatross.macroscale.simulation — adapter between VelocityHull/ERA5 and the IVP shooter.

Configure once with configure(), then call wind_interp() and velocity_hull()
from ivp_shooter.py.
"""

from __future__ import annotations

import numpy as np

DS_THRESHOLD = 9.0   # m/s — matches migration.py

_hull = None   # VelocityHull instance
_era5 = None   # ERA5Interpolator instance


def configure(hull, era5) -> None:
    """Set module-level hull and era5. Must be called before wind_interp / velocity_hull."""
    global _hull, _era5
    _hull = hull
    _era5 = era5


def wind_interp(x: np.ndarray, t: float) -> np.ndarray:
    """
    x=[lat, lon] deg, t=unix s → [u10, v10] m/s, shape (2,)
    """
    u10, v10 = _era5.query(float(x[0]), float(x[1]), t)
    return np.array([float(u10), float(v10)])


def velocity_hull(u_angle: float, W: np.ndarray) -> np.ndarray:
    """
    u_angle : heading in WIND frame [rad]  (0 = upwind, π/2 = crosswind)
    W       : [u10, v10] m/s, shape (2,)
    Returns : geographic velocity [vx_East, vy_North] m/s, shape (2,)

    Returns zeros if |W| < DS_THRESHOLD.

    Rotation (wind frame → geographic):
      α = arctan2(−v10, −u10)          # direction wind blows FROM, CCW from East
      u_wind = r · sin(u_angle)         # crosswind
      v_wind = r · cos(u_angle)         # upwind
      vx_East  = v_wind · cos(α) − u_wind · sin(α)
      vy_North = v_wind · sin(α) + u_wind · cos(α)
    """
    V_ref = float(np.hypot(W[0], W[1]))
    if V_ref < DS_THRESHOLD:
        return np.zeros(2)

    r = _hull.query(V_ref, u_angle)
    alpha = np.arctan2(-W[1], -W[0])

    u_wind = r * np.sin(u_angle)
    v_wind = r * np.cos(u_angle)

    vx_East  = v_wind * np.cos(alpha) - u_wind * np.sin(alpha)
    vy_North = v_wind * np.sin(alpha) + u_wind * np.cos(alpha)

    return np.array([vx_East, vy_North])
