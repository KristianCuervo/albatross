"""
migration_hamiltonian.py
========================
Hamiltonian IVP migration iso-curve simulation for wandering albatross.

Theory
------
The DS cycle is energy-conserving (same altitude/speed at start and end).
The macro-scale dynamics are formulated as an optimal-control problem whose
Pontryagin Hamiltonian is:

    H = p_x * x_dot + p_y * y_dot

where costate p = (p_x, p_y) = (c_x, c_y) is a fixed unit migration direction.

For each direction d swept over the full unit circle, the bird greedily
maximises ground speed projected onto d at every time step — this is the
support function of the local DS velocity hull.  Integrating the IVP for
N_dirs = 360 directions simultaneously (vectorised) yields the reachability
envelope; endpoints at t = 1 … 7 days form "iso-curves" of migration distance.

Coordinate conventions (consistent with build_velocity_hull.py)
---------------------------------------------------------------
Wind frame:
  u  — crosswind (m/s, +ve = right when facing upwind)
  v  — upwind    (m/s, +ve = into the wind)

Geographic frame (ERA5):
  x  = eastward   (u10 axis)
  y  = northward  (v10 axis)

α = arctan2(−v10, −u10)     direction FROM which wind blows, CCW from East

Rotation wind-frame → geographic:
  vx_geo = v_wind * cos(α) − u_wind * sin(α)
  vy_geo = v_wind * sin(α) + u_wind * cos(α)

Inverse — geographic direction d → wind-frame components:
  d_v_wind = cos(α) * d_x + sin(α) * d_y     (upwind projection)
  d_u_wind = −sin(α) * d_x + cos(α) * d_y    (crosswind projection)

Hull boundary parameterisation (from build_velocity_hull.py, N_RAYS = 720):
  At angle θ: u_hull = r(θ) * sin(θ),  v_hull = r(θ) * cos(θ)
  (θ = 0 → headwind,  θ = π/2 → crosswind,  θ = π → tailwind)

No-DS condition
---------------
If V_ref = |wind| < 9 m/s the bird cannot DS; velocity is set to zero and
the trajectory stalls at its current position for that hour.

Outputs
-------
CG2/data/migration_isocurves.npz
    positions   (169, 360, 2)  — [time, direction, (lat, lon)]
    directions  (360, 2)       — (c_x, c_y) unit vectors swept over unit circle
    times       (169,)         — Unix timestamps for each time step
    start       (2,)           — (lat, lon) of start position

Usage
-----
    python CG2/migration_hamiltonian.py
    python CG2/migration_hamiltonian.py --start-lat -46.4 --start-lon 52.0 \\
        --start-time 2022-12-01T00:00:00 --n-steps 168 --n-dirs 360
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE     = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

OUT_NPZ  = DATA_DIR / "migration_isocurves.npz"

# Simulation defaults
DEFAULT_LAT        = -46.4          # Crozet Island [°N]
DEFAULT_LON        =  52.0          # Crozet Island [°E]
DEFAULT_START_ISO  = "2022-12-01T00:00:00"
DEFAULT_N_STEPS    = 168            # 7 days × 24 h
DEFAULT_N_DIRS     = 720
DEFAULT_DT         = 3600.0         # 1 hour [s]
#DEFAULT_DT         = 600.0         # 1 hour [s]
DS_THRESHOLD       = 9.0            # m/s — minimum V_ref for DS


# ─── Hull helpers ─────────────────────────────────────────────────────────────

def _interpolate_hull_vec(v_ref_arr: np.ndarray, hull_data: dict) -> np.ndarray:
    """
    Vectorised hull-radii interpolation for N query wind speeds.

    Parameters
    ----------
    v_ref_arr : (N,)   query V_ref values [m/s]
    hull_data : dict   from load_hull_table() in build_velocity_hull.py

    Returns
    -------
    radii : (N, n_rays)  hull boundary radii in wind frame [m/s]
    """
    levels = hull_data["v_ref_levels"]   # (n_v,)
    radii  = hull_data["hull_radii"]     # (n_v, n_rays)

    v = np.clip(v_ref_arr, levels.min(), levels.max())
    idx_f = np.interp(v, levels, np.arange(len(levels)))  # (N,)
    lo    = np.floor(idx_f).astype(int)
    hi    = np.minimum(lo + 1, len(levels) - 1)
    frac  = idx_f - lo                                     # (N,)

    result = (1.0 - frac[:, None]) * radii[lo] + frac[:, None] * radii[hi]  # (N, n_rays)
    # NaN rays indicate directions physically unreachable at this V_ref
    # (e.g. upwind at low V_ref). Replace with 0 so argmax ignores them.
    return np.nan_to_num(result, nan=0.0)


def hull_support(
    d_geo: np.ndarray,
    alpha: float,
    V_ref: float,
    hull_data: dict,
) -> tuple[float, float]:
    """
    Support function of the DS velocity hull: find the hull boundary point that
    maximises d_geo · v_geo.

    Parameters
    ----------
    d_geo     : (2,)  migration direction unit vector in geographic frame (East, North)
    alpha     : float  wind-from direction [rad], CCW from East  = arctan2(−v10, −u10)
    V_ref     : float  local wind speed = sqrt(u10² + v10²) [m/s]
    hull_data : dict   from load_hull_table()

    Returns
    -------
    (vx_geo, vy_geo) : tuple[float, float]  optimal geographic velocity [m/s]
                        (0, 0) if V_ref < DS_THRESHOLD
    """
    if V_ref < DS_THRESHOLD:
        return 0.0, 0.0

    hull_angles = hull_data["hull_angles"]   # (720,)

    # Interpolate hull radii at this V_ref
    radii = _interpolate_hull_vec(np.array([V_ref]), hull_data)[0]  # (720,)

    # Project migration direction into wind frame
    d_x, d_y = float(d_geo[0]), float(d_geo[1])
    d_v_wind =  np.cos(alpha) * d_x + np.sin(alpha) * d_y   # upwind projection
    d_u_wind = -np.sin(alpha) * d_x + np.cos(alpha) * d_y   # crosswind projection

    # Dot product with each hull boundary point: r(θ) * (d_u·sin θ + d_v·cos θ)
    sin_a = np.sin(hull_angles)
    cos_a = np.cos(hull_angles)
    dot   = radii * (d_u_wind * sin_a + d_v_wind * cos_a)

    # dot may still be NaN where radii were originally NaN but got zeroed;
    # nan_to_num above already handled that, so argmax is safe here.
    best      = int(np.argmax(dot))
    r_opt     = float(radii[best])
    theta_opt = float(hull_angles[best])

    u_opt = r_opt * sin_a[best]   # crosswind
    v_opt = r_opt * cos_a[best]   # upwind

    # Rotate wind frame → geographic
    vx_geo = v_opt * np.cos(alpha) - u_opt * np.sin(alpha)
    vy_geo = v_opt * np.sin(alpha) + u_opt * np.cos(alpha)
    return float(vx_geo), float(vy_geo)


# ─── ODE right-hand side ──────────────────────────────────────────────────────

def _rhs(
    positions: np.ndarray,
    t: float,
    hull_data: dict,
    era5,
    directions: np.ndarray,
    sin_angles: np.ndarray,
    cos_angles: np.ndarray,
    n_dirs: int,
) -> np.ndarray:
    """
    Evaluate d(lat, lon)/dt [degrees/second] for all n_dirs trajectories.

    Lat/lon are clamped to valid ERA5 query bounds before each lookup so that
    intermediate RK4 stages that transiently exceed [-90, 90] or [-180, 180)
    do not cause out-of-bounds interpolation errors.

    Returns
    -------
    deriv : (n_dirs, 2)  columns are (dlat/dt, dlon/dt) in degrees/second
    """
    # Clamp/wrap for ERA5 query safety at RK4 intermediate stages
    lat_q = np.clip(positions[:, 0], -90.0, 90.0)
    lon_q = (positions[:, 1] + 180.0) % 360.0 - 180.0

    u10_arr, v10_arr = era5.query(lat_q, lon_q, t)

    alpha_arr = np.arctan2(-v10_arr, -u10_arr)           # FROM direction (N,)
    V_ref_arr = np.hypot(u10_arr, v10_arr)               # (N,)
    ds_mask   = V_ref_arr >= DS_THRESHOLD                # (N,)

    V_ref_clipped = np.where(ds_mask, V_ref_arr, DS_THRESHOLD)
    radii_arr     = _interpolate_hull_vec(V_ref_clipped, hull_data)  # (N, 720)

    d_x      = directions[:, 0]   # East  (N,)
    d_y      = directions[:, 1]   # North (N,)
    d_v_wind =  np.cos(alpha_arr) * d_x + np.sin(alpha_arr) * d_y   # upwind    (N,)
    d_u_wind = -np.sin(alpha_arr) * d_x + np.cos(alpha_arr) * d_y   # crosswind (N,)

    # dot = r(θ) * (d_u * sin θ + d_v * cos θ)   shape (N, 720)
    dot = radii_arr * (
        d_u_wind[:, None] * sin_angles[None, :]
        + d_v_wind[:, None] * cos_angles[None, :]
    )

    best_idx  = np.argmax(dot, axis=1)                          # (N,)
    r_opt     = radii_arr[np.arange(n_dirs), best_idx]          # (N,)
    theta_opt = hull_data["hull_angles"][best_idx]               # (N,)

    u_opt = r_opt * np.sin(theta_opt)   # crosswind [m/s]
    v_opt = r_opt * np.cos(theta_opt)   # upwind    [m/s]

    vx_geo = v_opt * np.cos(alpha_arr) - u_opt * np.sin(alpha_arr)   # East  [m/s]
    vy_geo = v_opt * np.sin(alpha_arr) + u_opt * np.cos(alpha_arr)   # North [m/s]

    vx_geo = np.where(ds_mask, vx_geo, 0.0)
    vy_geo = np.where(ds_mask, vy_geo, 0.0)

    cos_lat = np.cos(np.deg2rad(lat_q))
    dlat_dt = vy_geo / 111320.0
    dlon_dt = vx_geo / (111320.0 * np.maximum(cos_lat, 1e-6))

    return np.column_stack([dlat_dt, dlon_dt])   # (N, 2) deg/s


# ─── Vectorised IVP shooter ───────────────────────────────────────────────────

def run_shooter(
    start: tuple[float, float],
    start_unix: float,
    n_steps: int,
    dt: float,
    n_dirs: int,
    hull_data: dict,
    era5,
) -> np.ndarray:
    """
    Integrate the migration IVP for all n_dirs directions simultaneously
    using a fourth-order Runge-Kutta (RK4) scheme.

    RK4 requires four ERA5 + hull evaluations per step (at t, t+dt/2, t+dt/2,
    t+dt), compared to one for Forward Euler, but reduces the global integration
    error from O(dt) to O(dt⁴).  ERA5WindInterpolator interpolates linearly in
    time between hourly snapshots, so intermediate-time queries at t+dt/2 are
    well-defined.

    Parameters
    ----------
    start      : (lat, lon) start position [°N, °E]
    start_unix : float      Unix timestamp of t=0 [s]
    n_steps    : int        number of integration steps (168 = 7 days at dt=3600 s)
    dt         : float      time step [s]  (3600 for hourly)
    n_dirs     : int        number of migration directions to sweep
    hull_data  : dict       from build_velocity_hull.load_hull_table()
    era5       : ERA5WindInterpolator instance

    Returns
    -------
    positions : (n_steps+1, n_dirs, 2)  — (lat, lon) for every step × direction
    """
    # Migration direction unit vectors sweeping [0, 2π) in East-North frame
    theta_dirs = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    directions = np.column_stack([np.cos(theta_dirs), np.sin(theta_dirs)])  # (N, 2)

    # Pre-compute hull angle sines/cosines once (720 rays, fixed)
    hull_angles = hull_data["hull_angles"]   # (720,)
    sin_angles  = np.sin(hull_angles)
    cos_angles  = np.cos(hull_angles)

    # All trajectories begin at the same start position
    positions = np.tile(np.array(start, dtype=np.float64), (n_dirs, 1))  # (N, 2)

    out = np.empty((n_steps + 1, n_dirs, 2), dtype=np.float64)
    out[0] = positions

    rhs_args = (hull_data, era5, directions, sin_angles, cos_angles, n_dirs)

    for step in range(n_steps):
        t = start_unix + step * dt

        # RK4: four evaluations of the velocity field
        k1 = _rhs(positions,               t,          *rhs_args)
        k2 = _rhs(positions + 0.5*dt * k1, t + 0.5*dt, *rhs_args)
        k3 = _rhs(positions + 0.5*dt * k2, t + 0.5*dt, *rhs_args)
        k4 = _rhs(positions +     dt * k3, t +     dt, *rhs_args)

        positions = positions + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

        # Clamp latitude to [-90, 90] — bird cannot cross poles
        positions[:, 0] = np.clip(positions[:, 0], -90.0, 90.0)
        # Wrap longitude to [-180, 180)
        positions[:, 1] = (positions[:, 1] + 180.0) % 360.0 - 180.0

        out[step + 1] = positions

        elapsed_h = (step + 1) * dt / 3600.0
        total_h   = n_steps * dt / 3600.0
        if abs(elapsed_h % 24) < dt / 3600.0 * 0.5:
            print(f"  Day {elapsed_h/24:.1f} / {total_h/24:.1f}  complete")

    return out


# ─── Entry point ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Hamiltonian IVP migration iso-curve simulation"
    )
    p.add_argument("--start-lat",  type=float, default=DEFAULT_LAT,
                   metavar="LAT", help=f"Start latitude [°N]  (default: {DEFAULT_LAT})")
    p.add_argument("--start-lon",  type=float, default=DEFAULT_LON,
                   metavar="LON", help=f"Start longitude [°E] (default: {DEFAULT_LON})")
    p.add_argument("--start-time", default=DEFAULT_START_ISO,
                   metavar="ISO8601", help=f"Start time UTC (default: {DEFAULT_START_ISO})")
    p.add_argument("--n-steps",    type=int,   default=DEFAULT_N_STEPS,
                   metavar="N",
                   help=f"Number of integration steps (default: {DEFAULT_N_STEPS}). "
                        "Total duration = n_steps × dt seconds.")
    p.add_argument("--n-dirs",     type=int,   default=DEFAULT_N_DIRS,
                   metavar="N", help=f"Number of directions  (default: {DEFAULT_N_DIRS})")
    p.add_argument("--dt",         type=float, default=DEFAULT_DT,
                   metavar="S",
                   help=f"Integration time step [s] (default: {DEFAULT_DT:.0f} = 1 h). "
                        "Finer dt refines trajectory integration but adds no new wind "
                        "information below the ERA5 1-hour temporal resolution. "
                        "Accuracy gain is negligible below ~900 s. "
                        "n_steps × dt must cover the desired duration "
                        "(e.g. 7 days = 604800 s → --n-steps 2016 --dt 300).")
    return p.parse_args()


def main():
    args = _parse_args()

    # ── Load hull data ────────────────────────────────────────────────────────
    hull_path = DATA_DIR / "velocity_hulls.npz"
    if not hull_path.exists():
        sys.exit(f"ERROR: Hull table not found at {hull_path}\n"
                 "Run: python CG2/build_velocity_hull.py --hull-only")
    print(f"Loading hull table from {hull_path.name}…")
    raw = np.load(hull_path)
    hull_data = {k: raw[k] for k in raw.files}
    print(f"  v_ref_levels: {hull_data['v_ref_levels']}")
    print(f"  hull_radii shape: {hull_data['hull_radii'].shape}")

    # ── Load ERA5 interpolator ────────────────────────────────────────────────
    nc_files = sorted(DATA_DIR.glob("era5_1h_so_*.nc"))
    if not nc_files:
        sys.exit(
            "ERROR: No hourly ERA5 files found in CG2/data/.\n"
            "Run: python CG2/download_era5_1h.py"
        )
    print(f"\nLoading {len(nc_files)} ERA5 file(s)…")
    for f in nc_files:
        print(f"  {f.name}")

    from era5_wind_interp import ERA5WindInterpolator
    era5 = ERA5WindInterpolator(nc_files)
    print(f"  ERA5 grid: {era5._n_lat} lat × {era5._n_lon} lon, "
          f"{len(era5._unix_times)} timesteps total")

    # ── Parse start time → Unix ───────────────────────────────────────────────
    start_dt   = datetime.fromisoformat(args.start_time).replace(tzinfo=timezone.utc)
    start_unix = start_dt.timestamp()

    # Warn and auto-adjust if start time is outside the ERA5 coverage window
    era5_t0 = float(era5._unix_times[0])
    era5_t1 = float(era5._unix_times[-1])
    era5_dt0 = datetime.utcfromtimestamp(era5_t0).isoformat()
    era5_dt1 = datetime.utcfromtimestamp(era5_t1).isoformat()

    if start_unix < era5_t0 or start_unix > era5_t1:
        print(f"\n  WARNING: Requested start time {start_dt.isoformat()} is outside "
              f"ERA5 coverage [{era5_dt0} … {era5_dt1}].")
        start_unix = era5_t0
        start_dt   = datetime.utcfromtimestamp(era5_t0)
        print(f"  Auto-adjusting to ERA5 start: {start_dt.isoformat()} UTC")
        print(f"  (Re-run with --start-time {start_dt.strftime('%Y-%m-%dT%H:%M:%S')} to suppress this.)")

    start_pos  = (args.start_lat, args.start_lon)

    print(f"\nSimulation parameters:")
    print(f"  Start position : {args.start_lat}°N, {args.start_lon}°E  (Crozet)")
    print(f"  Start time     : {start_dt.isoformat()} UTC")
    dt         = args.dt
    duration_h = args.n_steps * dt / 3600.0
    print(f"  Steps          : {args.n_steps}  ({duration_h:.1f} h = {duration_h/24:.2f} days)")
    print(f"  Directions     : {args.n_dirs}")
    print(f"  dt             : {dt:.0f} s")
    print()

    # ── Run shooter ───────────────────────────────────────────────────────────
    print("Running IVP shooter…")
    positions = run_shooter(
        start      = start_pos,
        start_unix = start_unix,
        n_steps    = args.n_steps,
        dt         = dt,
        n_dirs     = args.n_dirs,
        hull_data  = hull_data,
        era5       = era5,
    )
    print(f"  positions shape: {positions.shape}  (time, direction, lat/lon)")

    # ── Compute output arrays ─────────────────────────────────────────────────
    theta_dirs = np.linspace(0.0, 2.0 * np.pi, args.n_dirs, endpoint=False)
    directions = np.column_stack([np.cos(theta_dirs), np.sin(theta_dirs)])

    times_unix = np.array([
        start_unix + i * dt for i in range(args.n_steps + 1)
    ])

    # ── Save output ───────────────────────────────────────────────────────────
    np.savez_compressed(
        OUT_NPZ,
        positions  = positions,                          # (169, 360, 2)
        directions = directions,                         # (360, 2)
        times      = times_unix,                         # (169,)
        start      = np.array([args.start_lat, args.start_lon]),
    )
    print(f"\nSaved → {OUT_NPZ}")
    print(f"  positions  : {positions.shape}")
    print(f"  directions : {directions.shape}")
    print(f"  times      : {times_unix.shape}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    steps_per_day = int(round(86400.0 / dt))
    for day in [1, 3, 7]:
        idx = day * steps_per_day
        if idx >= positions.shape[0]:
            continue
        pts        = positions[idx]
        lat_c      = pts[:, 0].mean()
        lon_c      = pts[:, 1].mean()
        lat_spread = pts[:, 0].max() - pts[:, 0].min()
        lon_spread = pts[:, 1].max() - pts[:, 1].min()
        print(f"  Day {day:2d}: centroid ({lat_c:.1f}°N, {lon_c:.1f}°E)  "
              f"spread Δlat={lat_spread:.1f}°  Δlon={lon_spread:.1f}°")


if __name__ == "__main__":
    main()
