"""
hjb_ivp_shooter.py
==================
Pontryagin Maximum Principle (PMP) IVP shooter for optimal wandering-albatross
dynamic-soaring migration.

Theory
------
The Hamiltonian for displacement in direction e_hat is:

    H(q, λ, u) = λ_lat · ẏ(u, w(q,t)) / 111320
               + λ_lon · ẋ(u, w(q,t)) / (111320 · cos(lat))

where q = (lat, lon) [deg] is the state, λ = (λ_lat, λ_lon) is the costate,
ẋ = vx_geo [m/s] (eastward), ẏ = vy_geo [m/s] (northward), and u is the
wind-frame heading (argmax over the DS hull boundary).

The co-state evolves as λ̇ = −∂H*/∂q, with spatial derivatives evaluated via
central finite differences (envelope theorem — same heading at perturbed winds).

Sweeping the initial costate angle φ ∈ [0, 2π) traces the full reachable-set
boundary (Pontryagin fan).

Coordinate conventions
-----------------------
  Wind frame (build_velocity_hull.py convention):
    u : crosswind (m/s, +ve right when facing upwind)
    v : upwind    (m/s, +ve into the wind)
    θ = 0 → headwind, θ = π/2 → crosswind, θ = π → tailwind
    u_wind = r(θ) · sin(θ),  v_wind = r(θ) · cos(θ)

  Geographic frame (ERA5):
    vx : eastward  (u10 axis)
    vy : northward (v10 axis)
    α = arctan2(−v10, −u10)         (wind-from direction, CCW from East)
    vx_geo = v_wind · cos(α) − u_wind · sin(α)
    vy_geo = v_wind · sin(α) + u_wind · cos(α)

  State in degree-space:
    dlat/dt = vy_geo / 111320                      [°/s]
    dlon/dt = vx_geo / (111320 · cos(lat_rad))    [°/s]

Imports
-------
    from CG2.hjb_ivp_shooter import shoot_fan, plot_fan
    from CG2.build_velocity_hull import load_hull_table
    from CG2.era5_wind_interp import ERA5WindInterpolator

Usage
-----
    python CG2/hjb_ivp_shooter.py
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

# ── Import from sibling modules ───────────────────────────────────────────────
sys.path.insert(0, str(HERE.parent))
from CG2.migration_hamiltonian import _interpolate_hull_vec, DS_THRESHOLD, run_shooter
from CG2.build_velocity_hull import load_hull_table
from CG2.era5_wind_interp import ERA5WindInterpolator

# ── Constants ─────────────────────────────────────────────────────────────────
_M_PER_DEG_LAT = 111_320.0   # metres per degree latitude (constant)
_EPS_S = 1e-3                 # spatial perturbation [deg] for costate FD
_EPS_W = 1e-2                 # wind perturbation [m/s]   (not currently used)


# ─── Low-level helpers ────────────────────────────────────────────────────────

def velocity_hull_array(
    u10: float,
    v10: float,
    hull_data: dict,
    n_angles: int = 360,
) -> np.ndarray:
    """
    Sample the DS velocity hull at given wind conditions.

    Interpolates hull radii at V_ref = hypot(u10, v10), then converts
    n_angles evenly-spaced wind-frame headings into geographic velocities.

    Parameters
    ----------
    u10, v10  : float   ERA5 10m wind components [m/s]
    hull_data : dict    from load_hull_table()
    n_angles  : int     number of heading samples (≥ 2)

    Returns
    -------
    v_geo : (n_angles, 2)
        Geographic velocities [m/s]; columns are (vx_east, vy_north).
        Zero array if V_ref < DS_THRESHOLD.
    """
    V_ref = float(np.hypot(u10, v10))
    if V_ref < DS_THRESHOLD:
        return np.zeros((n_angles, 2))

    alpha = float(np.arctan2(-v10, -u10))

    hull_angles_720 = hull_data["hull_angles"]          # (720,)
    radii_720 = _interpolate_hull_vec(np.array([V_ref]), hull_data)[0]  # (720,)

    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # Circular interpolation — append one extra period to wrap correctly
    angles_ext = np.concatenate([hull_angles_720, hull_angles_720 + 2 * np.pi])
    radii_ext  = np.concatenate([radii_720, radii_720])
    r = np.interp(thetas, angles_ext, radii_ext)        # (n_angles,)

    u_w = r * np.sin(thetas)   # crosswind component
    v_w = r * np.cos(thetas)   # upwind component

    vx_geo = v_w * np.cos(alpha) - u_w * np.sin(alpha)
    vy_geo = v_w * np.sin(alpha) + u_w * np.cos(alpha)

    return np.column_stack([vx_geo, vy_geo])            # (n_angles, 2)


def _vxy_at_theta(
    u10: float,
    v10: float,
    theta: float,
    hull_data: dict,
) -> tuple[float, float]:
    """
    Geographic velocity (vx, vy) [m/s] at wind (u10, v10) and wind-frame
    heading theta [rad].

    Returns (0, 0) if V_ref < DS_THRESHOLD.
    Used for the envelope-theorem spatial-derivative finite differences.
    """
    V_ref = float(np.hypot(u10, v10))
    if V_ref < DS_THRESHOLD:
        return 0.0, 0.0

    alpha = float(np.arctan2(-v10, -u10))
    hull_angles = hull_data["hull_angles"]                      # (720,)
    radii       = _interpolate_hull_vec(np.array([V_ref]), hull_data)[0]  # (720,)

    # Circular interpolation at the fixed heading
    angles_ext = np.concatenate([hull_angles, hull_angles + 2 * np.pi])
    radii_ext  = np.concatenate([radii, radii])
    r = float(np.interp(theta % (2 * np.pi), angles_ext, radii_ext))

    u_w = r * np.sin(theta)
    v_w = r * np.cos(theta)

    vx = float(v_w * np.cos(alpha) - u_w * np.sin(alpha))
    vy = float(v_w * np.sin(alpha) + u_w * np.cos(alpha))
    return vx, vy


def _optimal_velocity(
    lat: float,
    lon: float,
    t: float,
    lam: np.ndarray,
    era5: ERA5WindInterpolator,
    hull_data: dict,
    n_angles: int = 360,
) -> tuple[float, float, float]:
    """
    Find the Pontryagin-optimal geographic velocity at (lat, lon, t) for
    costate λ.

    Returns
    -------
    vx_star, vy_star : float  optimal geographic velocity [m/s]
    theta_val        : float  optimal wind-frame heading [rad] (NaN if no DS)
    """
    u10, v10 = era5.query(float(lat), float(lon), float(t))
    V_ref = float(np.hypot(u10, v10))

    if V_ref < DS_THRESHOLD:
        return 0.0, 0.0, float("nan")

    alpha = float(np.arctan2(-v10, -u10))
    hull_angles_720 = hull_data["hull_angles"]
    radii_720 = _interpolate_hull_vec(np.array([V_ref]), hull_data)[0]

    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    angles_ext = np.concatenate([hull_angles_720, hull_angles_720 + 2 * np.pi])
    radii_ext  = np.concatenate([radii_720, radii_720])
    r = np.interp(thetas, angles_ext, radii_ext)

    u_w = r * np.sin(thetas)
    v_w = r * np.cos(thetas)
    vx_geo = v_w * np.cos(alpha) - u_w * np.sin(alpha)
    vy_geo = v_w * np.sin(alpha) + u_w * np.cos(alpha)

    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat = max(float(cos_lat), 1e-6)

    proj = lam[0] * vy_geo / _M_PER_DEG_LAT + lam[1] * vx_geo / (_M_PER_DEG_LAT * cos_lat)
    best = int(np.argmax(proj))

    return float(vx_geo[best]), float(vy_geo[best]), float(thetas[best])


# ─── PMP right-hand side ──────────────────────────────────────────────────────

def _rhs(
    t: float,
    z: list | np.ndarray,
    era5: ERA5WindInterpolator,
    hull_data: dict,
    n_angles: int = 360,
) -> list[float]:
    """
    ODE right-hand side for a single PMP trajectory.

    State vector z = [lat, lon, λ_lat, λ_lon].

    Costate equation (envelope theorem — same heading θ* at perturbed winds):

        dλ_lat/dt = −[λ_lat · dvy*/dlat / M  +  λ_lon · dvx*/dlat / (M · cos_lat)
                      +  λ_lon · vx* · sin(lat_rad) · (π/180) / (M · cos²_lat)]

        dλ_lon/dt = −[λ_lat · dvy*/dlon / M  +  λ_lon · dvx*/dlon / (M · cos_lat)]

    where M = 111320 m/° and spatial derivatives are central-difference with
    eps = 1e-3 degrees.

    Parameters
    ----------
    t         : float  current Unix timestamp [s]
    z         : [lat, lon, lam0, lam1]
    era5      : ERA5WindInterpolator
    hull_data : dict
    n_angles  : int

    Returns
    -------
    [dlat_dt, dlon_dt, dlam0_dt, dlam1_dt]
    """
    lat, lon, lam0, lam1 = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    lam = np.array([lam0, lam1])

    vx_s, vy_s, theta_star = _optimal_velocity(lat, lon, t, lam, era5, hull_data, n_angles)

    cos_lat = max(np.cos(np.deg2rad(lat)), 1e-6)
    M = _M_PER_DEG_LAT

    dlat_dt = vy_s / M
    dlon_dt = vx_s / (M * cos_lat)

    if np.isnan(theta_star):
        # No DS at this position — costate frozen (no spatial gradient)
        return [dlat_dt, dlon_dt, 0.0, 0.0]

    # Spatial derivatives of optimal velocity via central differences
    # (envelope theorem: same theta* when wind changes slightly with position)

    # -- lat derivatives --
    u_p, v_p = era5.query(lat + _EPS_S, lon, t)
    u_m, v_m = era5.query(lat - _EPS_S, lon, t)
    vx_p_lat, vy_p_lat = _vxy_at_theta(u_p, v_p, theta_star, hull_data)
    vx_m_lat, vy_m_lat = _vxy_at_theta(u_m, v_m, theta_star, hull_data)
    dvy_dlat = (vy_p_lat - vy_m_lat) / (2.0 * _EPS_S)
    dvx_dlat = (vx_p_lat - vx_m_lat) / (2.0 * _EPS_S)

    # -- lon derivatives --
    u_p, v_p = era5.query(lat, lon + _EPS_S, t)
    u_m, v_m = era5.query(lat, lon - _EPS_S, t)
    vx_p_lon, vy_p_lon = _vxy_at_theta(u_p, v_p, theta_star, hull_data)
    vx_m_lon, vy_m_lon = _vxy_at_theta(u_m, v_m, theta_star, hull_data)
    dvy_dlon = (vy_p_lon - vy_m_lon) / (2.0 * _EPS_S)
    dvx_dlon = (vx_p_lon - vx_m_lon) / (2.0 * _EPS_S)

    # Metric correction: ∂/∂lat [1/cos(lat)] = sin(lat_rad)·(π/180) / cos²(lat)
    sin_lat = np.sin(np.deg2rad(lat))
    metric_corr = lam1 * vx_s * sin_lat * (np.pi / 180.0) / (M * cos_lat ** 2)

    dlam0_dt = -(lam0 * dvy_dlat / M
                 + lam1 * dvx_dlat / (M * cos_lat)
                 + metric_corr)
    dlam1_dt = -(lam0 * dvy_dlon / M
                 + lam1 * dvx_dlon / (M * cos_lat))

    return [dlat_dt, dlon_dt, dlam0_dt, dlam1_dt]


# ─── Single-trajectory shooter ────────────────────────────────────────────────

class _ShootResult:
    """Lightweight container mimicking scipy OdeResult for shoot() output."""
    __slots__ = ("t", "y", "success")

    def __init__(self, t: np.ndarray, y: np.ndarray, success: bool = True):
        self.t = t        # (N_steps+1,)
        self.y = y        # (4, N_steps+1)  rows: lat, lon, lam0, lam1
        self.success = success


def shoot(
    x0: tuple[float, float],
    phi: float,
    start_unix: float,
    T: float,
    era5: ERA5WindInterpolator,
    hull_data: dict,
    n_angles: int = 360,
    dt: float = 3600.0,
) -> _ShootResult:
    """
    Integrate a single PMP trajectory with a fixed-step RK4.

    A fixed step (default = 3600 s, matching ERA5 hourly resolution) is used
    instead of an adaptive solver.  The RHS has a hard discontinuity whenever
    wind speed crosses DS_THRESHOLD; adaptive methods stall trying to resolve
    that edge — fixed-step RK4 passes through it cleanly in O(T/dt) steps.

    Parameters
    ----------
    x0         : (lat0, lon0) starting position [°]
    phi        : float  initial costate angle [rad]; λ₀ = (cos φ, sin φ)
    start_unix : float  Unix timestamp at t = 0 [s]
    T          : float  total integration duration [s]
    era5       : ERA5WindInterpolator
    hull_data  : dict   from load_hull_table()
    n_angles   : int    heading candidates per RHS evaluation
    dt         : float  fixed RK4 step size [s]  (default 3600 = 1 h)

    Returns
    -------
    result : _ShootResult
        .t    — Unix timestamps  (N_steps+1,)
        .y    — (4, N_steps+1)  rows: lat, lon, λ_lat, λ_lon
        .success — True always (no adaptive failure mode)
    """
    n_steps = max(1, round(T / dt))
    dt_actual = T / n_steps          # exact even division

    z = np.array([float(x0[0]), float(x0[1]), np.cos(phi), np.sin(phi)])
    t = float(start_unix)

    ts = np.empty(n_steps + 1)
    ys = np.empty((4, n_steps + 1))
    ts[0] = t
    ys[:, 0] = z

    for i in range(n_steps):
        dz = np.asarray(_rhs(t, z, era5, hull_data, n_angles))
        k1 = dz

        z2 = z + 0.5 * dt_actual * k1
        dz2 = np.asarray(_rhs(t + 0.5 * dt_actual, z2, era5, hull_data, n_angles))
        k2 = dz2

        z3 = z + 0.5 * dt_actual * k2
        dz3 = np.asarray(_rhs(t + 0.5 * dt_actual, z3, era5, hull_data, n_angles))
        k3 = dz3

        z4 = z + dt_actual * k3
        dz4 = np.asarray(_rhs(t + dt_actual, z4, era5, hull_data, n_angles))
        k4 = dz4

        z = z + (dt_actual / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt_actual

        ts[i + 1] = t
        ys[:, i + 1] = z

    return _ShootResult(ts, ys)


# ─── Multi-trajectory fan (parallel) ──────────────────────────────────────────

def _shoot_worker(args: tuple):
    """Module-level worker for ProcessPoolExecutor picklability."""
    x0, phi, start_unix, T, era5, hull_data, n_angles, dt = args
    return shoot(x0, phi, start_unix, T, era5, hull_data, n_angles, dt)


def shoot_fan(
    x0: tuple[float, float],
    start_unix: float,
    T: float,
    era5: ERA5WindInterpolator,
    hull_data: dict,
    n_phi: int = 36,
    n_angles: int = 360,
    dt: float = 3600.0,
    n_workers: int | None = None,
) -> tuple[list, np.ndarray]:
    """
    Shoot n_phi PMP trajectories with evenly-spaced initial costate angles.

    Parameters
    ----------
    x0         : (lat0, lon0) starting position [°]
    start_unix : float  Unix timestamp at t = 0 [s]
    T          : float  total integration duration [s]
    era5       : ERA5WindInterpolator
    hull_data  : dict   from load_hull_table()
    n_phi      : int    number of initial costate angles (fan density)
    n_angles   : int    heading candidates per RHS evaluation
    dt         : float  fixed RK4 step [s]
    n_workers  : int | None  ProcessPoolExecutor workers (None = CPU count)

    Returns
    -------
    results : list of n_phi OdeResult objects (same order as phis)
    phis    : (n_phi,) initial costate angles [rad]
    """
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    args_list = [
        (x0, phi, start_unix, T, era5, hull_data, n_angles, dt)
        for phi in phis
    ]

    results: list = [None] * n_phi
    t0 = time.perf_counter()

    print(f"shoot_fan: launching {n_phi} trajectories (n_workers={n_workers}) …")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(_shoot_worker, a): i for i, a in enumerate(args_list)
        }
        completed = 0
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                print(f"  trajectory {i} (φ={np.rad2deg(phis[i]):.1f}°) failed: {exc}")
                results[i] = None
            completed += 1
            if completed % max(1, n_phi // 8) == 0 or completed == n_phi:
                elapsed = time.perf_counter() - t0
                print(
                    f"  {completed}/{n_phi} done  ({elapsed:.1f} s)",
                    flush=True,
                )

    n_ok = sum(r is not None and r.success for r in results)
    print(f"shoot_fan: {n_ok}/{n_phi} trajectories converged.")
    return results, phis


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_fan(
    fan_results: list,
    phis: np.ndarray,
    x0: tuple[float, float],
    title: str = "PMP Trajectory Fan",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a Pontryagin trajectory fan on a map.

    Trajectories are colour-coded by initial costate angle φ.
    Requires matplotlib; cartopy is used if available.

    Parameters
    ----------
    fan_results : list of OdeResult (from shoot_fan)
    phis        : (n_phi,) initial costate angles [rad]
    x0          : (lat0, lon0) start position [°]
    title       : str
    save_path   : path to save figure (None = show only)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        _USE_CARTOPY = True
    except ImportError:
        _USE_CARTOPY = False

    n_phi = len(phis)
    cmap = cm.hsv

    if _USE_CARTOPY:
        lons_all = [
            sol.y[1]
            for sol in fan_results
            if sol is not None and sol.success and sol.y.shape[1] > 1
        ]
        lats_all = [
            sol.y[0]
            for sol in fan_results
            if sol is not None and sol.success and sol.y.shape[1] > 1
        ]
        if lons_all:
            lon_c = float(np.median(np.concatenate(lons_all)))
        else:
            lon_c = float(x0[1])

        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(
            1, 1, 1,
            projection=ccrs.PlateCarree(central_longitude=lon_c),
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.6)
        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(figsize=(14, 9))
        transform = None

    for i, (sol, phi) in enumerate(zip(fan_results, phis)):
        if sol is None or not sol.success or sol.y.shape[1] < 2:
            continue
        lats = sol.y[0]
        lons = sol.y[1]
        color = cmap(i / n_phi)
        kw = dict(color=color, linewidth=0.9, alpha=0.8)
        if _USE_CARTOPY:
            ax.plot(lons, lats, transform=transform, **kw)
        else:
            ax.plot(lons, lats, **kw)

    # Mark start
    kw_start = dict(marker="o", color="black", markersize=7, zorder=5, linestyle="none")
    if _USE_CARTOPY:
        ax.plot([x0[1]], [x0[0]], transform=transform, **kw_start)
    else:
        ax.plot([x0[1]], [x0[0]], **kw_start)
        ax.set_xlabel("Longitude [°E]")
        ax.set_ylabel("Latitude [°N]")

    ax.set_title(title)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    plt.show()


# ─── Reachable-set envelope ───────────────────────────────────────────────────

def reachable_envelope(
    fan_results: list,
    phis: np.ndarray,
    t_query: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract endpoint positions from a fan at a given time (or final time).

    Parameters
    ----------
    fan_results : list of OdeResult
    phis        : (n_phi,) initial costate angles [rad]
    t_query     : Unix timestamp to evaluate; None → final time of each sol

    Returns
    -------
    end_lats, end_lons : (n_phi,) arrays (NaN where trajectory failed)
    """
    end_lats = np.full(len(phis), np.nan)
    end_lons = np.full(len(phis), np.nan)

    for i, sol in enumerate(fan_results):
        if sol is None or not sol.success or sol.y.shape[1] < 1:
            continue
        if t_query is None:
            end_lats[i] = float(sol.y[0, -1])
            end_lons[i] = float(sol.y[1, -1])
        else:
            # Find closest recorded time step
            idx = int(np.argmin(np.abs(sol.t - t_query)))
            end_lats[i] = float(sol.y[0, idx])
            end_lons[i] = float(sol.y[1, idx])

    return end_lats, end_lons


# ─── Iso-curve comparison plot ────────────────────────────────────────────────

def plot_isocurves(
    pmp_results: list,
    greedy_positions: np.ndarray,
    x0: tuple[float, float],
    day_milestones: list[int] | None = None,
    dt: float = 3600.0,
    title: str = "Reachability Iso-curves: PMP vs Greedy",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot iso-curves (reachable-set boundaries) at daily milestones for both
    the PMP (costate-evolving) fan and the greedy (fixed-costate) fan.

    Each iso-curve is formed by extracting all trajectory positions at a given
    time, sorting by bearing angle from x0, and connecting them into a closed
    curve.

    Parameters
    ----------
    pmp_results      : list of _ShootResult from shoot_fan
    greedy_positions : (n_steps+1, n_dirs, 2) from run_shooter; col 0=lat, 1=lon
    x0               : (lat0, lon0) start position [°]
    day_milestones   : list of integer days to plot (default [1..7])
    dt               : fixed step size used in both runs [s]
    title            : figure title
    save_path        : path to save figure; None = show only
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.lines as mlines

    if day_milestones is None:
        day_milestones = list(range(1, 8))

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        _USE_CARTOPY = True
    except ImportError:
        _USE_CARTOPY = False

    # ── Collect all points to determine map extent ────────────────────────────
    all_lats, all_lons = [x0[0]], [x0[1]]
    for sol in pmp_results:
        if sol is not None and sol.success:
            all_lats.append(float(sol.y[0, -1]))
            all_lons.append(float(sol.y[1, -1]))
    if greedy_positions is not None:
        all_lats.extend(greedy_positions[-1, :, 0].tolist())
        all_lons.extend(greedy_positions[-1, :, 1].tolist())
    lon_c = float(np.median(all_lons))

    # ── Axes setup ────────────────────────────────────────────────────────────
    if _USE_CARTOPY:
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(
            1, 1, 1,
            projection=ccrs.PlateCarree(central_longitude=lon_c),
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5, zorder=2)
        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(figsize=(14, 9))
        ax.set_xlabel("Longitude [°E]")
        ax.set_ylabel("Latitude [°N]")
        transform = None

    # ── Color map: one colour per day milestone ───────────────────────────────
    n_days = len(day_milestones)
    palette = cm.plasma(np.linspace(0.15, 0.95, n_days))

    def _sorted_closed_curve(lats, lons, origin):
        """Sort (lat, lon) points by bearing from origin; return closed arrays."""
        bearing = np.arctan2(lons - origin[1], lats - origin[0])
        order = np.argsort(bearing)
        lats_s = np.append(lats[order], lats[order[0]])
        lons_s = np.append(lons[order], lons[order[0]])
        return lats_s, lons_s

    def _plot_curve(lats_c, lons_c, color, lw, ls, zorder, label=None):
        kw = dict(color=color, linewidth=lw, linestyle=ls, zorder=zorder)
        if label:
            kw["label"] = label
        if _USE_CARTOPY:
            ax.plot(lons_c, lats_c, transform=transform, **kw)
        else:
            ax.plot(lons_c, lats_c, **kw)

    # ── Draw iso-curves for each milestone ───────────────────────────────────
    steps_per_day = int(86400.0 / dt)

    for i, day in enumerate(day_milestones):
        step_idx = day * steps_per_day
        color = palette[i]
        label_day = f"Day {day}"

        # PMP curve
        pmp_lats, pmp_lons = [], []
        for sol in pmp_results:
            if sol is None or not sol.success:
                continue
            if sol.y.shape[1] > step_idx:
                pmp_lats.append(float(sol.y[0, step_idx]))
                pmp_lons.append(float(sol.y[1, step_idx]))

        if len(pmp_lats) >= 3:
            lats_c, lons_c = _sorted_closed_curve(
                np.array(pmp_lats), np.array(pmp_lons), x0
            )
            _plot_curve(lats_c, lons_c, color, lw=2.2, ls="-",
                        zorder=5, label=label_day if greedy_positions is None else None)

        # Greedy curve
        if greedy_positions is not None and greedy_positions.shape[0] > step_idx:
            g_pts = greedy_positions[step_idx]          # (n_dirs, 2)
            lats_c, lons_c = _sorted_closed_curve(
                g_pts[:, 0], g_pts[:, 1], x0
            )
            _plot_curve(lats_c, lons_c, color, lw=1.5, ls="--", zorder=4)

        # Day label near the PMP curve's northernmost point (if available)
        if len(pmp_lats) >= 1:
            north_idx = int(np.argmax(pmp_lats))
            lbl_lat = pmp_lats[north_idx]
            lbl_lon = pmp_lons[north_idx]
            kw_txt = dict(fontsize=7, color=color, ha="center", va="bottom",
                          fontweight="bold", zorder=6)
            if _USE_CARTOPY:
                ax.text(lbl_lon, lbl_lat + 0.5, label_day,
                        transform=transform, **kw_txt)
            else:
                ax.text(lbl_lon, lbl_lat + 0.5, label_day, **kw_txt)

    # ── Start marker ──────────────────────────────────────────────────────────
    kw_start = dict(marker="*", color="black", markersize=12,
                    zorder=10, linestyle="none")
    if _USE_CARTOPY:
        ax.plot([x0[1]], [x0[0]], transform=transform, **kw_start)
    else:
        ax.plot([x0[1]], [x0[0]], **kw_start)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mlines.Line2D([], [], color="gray", lw=2.2, ls="-",  label="PMP (costate evolution)"),
        mlines.Line2D([], [], color="gray", lw=1.5, ls="--", label="Greedy (fixed costate)"),
    ]
    # Colour patches for days
    for i, day in enumerate(day_milestones):
        legend_handles.append(
            mlines.Line2D([], [], color=palette[i], lw=2, label=f"Day {day}")
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.8, ncol=2)

    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    plt.show()


# ─── __main__ demo ────────────────────────────────────────────────────────────

def main() -> None:
    """
    Compare PMP (costate-evolving) vs Greedy (fixed-costate) iso-curves.

    Crozet Island (−46.4°, 52.0°), 2023-01-15T00:00:00 UTC, 7 days.
    Loads era5_1h_so_2023_01_15…21.nc (7 daily files, 1-h resolution).
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive for headless runs; remove for GUI

    print("─" * 60)
    print("PMP Fan vs Greedy — Iso-curve Comparison")
    print("─" * 60)

    # 1. Load hull
    hull_path = DATA_DIR / "velocity_hulls.npz"
    if not hull_path.exists():
        sys.exit(f"ERROR: hull file not found: {hull_path}")
    hull_data = load_hull_table(hull_path)
    print(f"Hull: V_max = {np.nanmax(hull_data['hull_radii']):.1f} m/s, "
          f"{hull_data['hull_radii'].shape[1]} rays, "
          f"{len(hull_data['v_ref_levels'])} levels")

    # 2. Load ERA5
    nc_paths = sorted(DATA_DIR.glob("era5_1h_so_2023_01_*.nc"))
    if not nc_paths:
        sys.exit(f"ERROR: no ERA5 files found in {DATA_DIR}")
    print(f"ERA5: {len(nc_paths)} × daily 1-h files")
    era5 = ERA5WindInterpolator(nc_paths)

    # 3. Simulation parameters
    start_unix = datetime(2023, 1, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    T_s   = 7 * 24 * 3600.0
    x0    = (-46.4, 52.0)           # Crozet Island
    n_phi = 36
    dt    = 3600.0                  # 1 h step (matches ERA5 temporal resolution)
    n_steps = int(T_s / dt)         # 168

    print(f"x0 = {x0}  (Crozet Island),  T = {T_s/86400:.0f} days,  n_φ = {n_phi}")

    # 4a. PMP fan (parallel, costate evolves)
    print("\n── PMP fan ──")
    t0 = time.perf_counter()
    pmp_results, phis = shoot_fan(
        x0, start_unix, T_s, era5, hull_data,
        n_phi=n_phi, dt=dt,
    )
    print(f"PMP done in {time.perf_counter() - t0:.1f} s")

    # 4b. Greedy fan (vectorised, fixed costate throughout)
    print("\n── Greedy fan ──")
    t0 = time.perf_counter()
    greedy_pos = run_shooter(
        x0, start_unix, n_steps, dt, n_phi, hull_data, era5,
    )   # → (n_steps+1, n_phi, 2)  last dim = (lat, lon)
    print(f"Greedy done in {time.perf_counter() - t0:.1f} s")

    # 5. Displacement comparison at 7 days
    pmp_end_lats, pmp_end_lons = reachable_envelope(pmp_results, phis)
    ok = ~np.isnan(pmp_end_lats)
    if ok.any():
        pmp_disp = np.hypot(
            (pmp_end_lats[ok] - x0[0]) * _M_PER_DEG_LAT,
            (pmp_end_lons[ok] - x0[1]) * _M_PER_DEG_LAT,
        ) / 1e3
        greedy_disp = np.hypot(
            (greedy_pos[-1, :, 0] - x0[0]) * _M_PER_DEG_LAT,
            (greedy_pos[-1, :, 1] - x0[1]) * _M_PER_DEG_LAT,
        ) / 1e3
        print(f"\nDisplacement at 7 days:")
        print(f"  PMP    max = {pmp_disp.max():.0f} km   mean = {pmp_disp.mean():.0f} km")
        print(f"  Greedy max = {greedy_disp.max():.0f} km   mean = {greedy_disp.mean():.0f} km")
        pct = 100.0 * (pmp_disp.max() - greedy_disp.max()) / greedy_disp.max()
        print(f"  PMP gain (max displacement): {pct:+.1f}%")

    # 6. Iso-curve comparison plot
    fig_path = DATA_DIR / "pmp_vs_greedy_isocurves.png"
    plot_isocurves(
        pmp_results, greedy_pos, x0,
        day_milestones=list(range(1, 8)),
        dt=dt,
        title=f"Reachability Iso-curves: PMP vs Greedy — Crozet 2023-01-15, {n_phi} directions",
        save_path=fig_path,
    )


if __name__ == "__main__":
    main()
