"""
albatross.macroscale.ivp_shooter — Mayer-form Hamiltonian IVP shooter.

Integrates the full adjoint ODE so the costate λ evolves with position,
revealing the true optimal direction at the terminal time ("e_hat").

Fan-shooting over initial costate angles φ ∈ [0, 2π) gives a reachability
envelope whose support function is the migration potential.
"""

from __future__ import annotations

import multiprocessing
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull

import albatross.macroscale.simulation as _sim
from .simulation import wind_interp, velocity_hull, DS_THRESHOLD


# ─── Fork-based parallel worker ───────────────────────────────────────────────
# Each trajectory in a fan is fully independent.  On Linux (fork context) the
# worker inherits the parent's memory (ERA5 arrays, hull) via copy-on-write so
# no large data needs to be pickled across processes.
#
# _FORKED_SHOOTER is set in the parent immediately before the Pool is created.
# Workers inherit the reference via fork and call .shoot() read-only.

_FORKED_SHOOTER = None   # set by shoot_fan() before fork


def _shoot_worker(args: tuple) -> dict | None:
    """Top-level function required for multiprocessing picklability."""
    phi, x0, t0, T, rtol, atol = args
    try:
        return _FORKED_SHOOTER.shoot(x0, phi, t0, T, rtol=rtol, atol=atol)
    except Exception:
        return None


# ─── Finite-difference helpers ───────────────────────────────────────────────

def wind_jacobian(x: np.ndarray, t: float, dx: float = 1e-3) -> np.ndarray:
    """
    Central finite differences of wind_interp w.r.t. position.

    Parameters
    ----------
    x  : (2,)  [lat, lon] degrees
    t  : float  unix timestamp
    dx : float  step in degrees (1e-3 deg ≈ 111 m)

    Returns
    -------
    grad_W : (2, 2)
        [[∂u10/∂lat, ∂u10/∂lon],
         [∂v10/∂lat, ∂v10/∂lon]]   [m/s per deg]
    """
    grad_W = np.zeros((2, 2))
    for j in range(2):
        xp = x.copy(); xp[j] += dx
        xm = x.copy(); xm[j] -= dx
        Wp = wind_interp(xp, t)
        Wm = wind_interp(xm, t)
        grad_W[:, j] = (Wp - Wm) / (2.0 * dx)
    return grad_W


def hull_jacobian(u_star: float, W: np.ndarray, dw: float = 0.5) -> np.ndarray:
    """
    Central finite differences of velocity_hull w.r.t. wind components.

    Parameters
    ----------
    u_star : float  optimal heading in wind frame [rad] (held fixed, envelope theorem)
    W      : (2,)   [u10, v10] m/s
    dw     : float  step in m/s

    Returns
    -------
    dV_dW : (2, 2)
        [[∂vx_East/∂u10, ∂vx_East/∂v10],
         [∂vy_North/∂u10, ∂vy_North/∂v10]]   dimensionless
    """
    dV_dW = np.zeros((2, 2))
    for j in range(2):
        Wp = W.copy(); Wp[j] += dw
        Wm = W.copy(); Wm[j] -= dw
        Vp = velocity_hull(u_star, Wp)
        Vm = velocity_hull(u_star, Wm)
        dV_dW[:, j] = (Vp - Vm) / (2.0 * dw)
    return dV_dW


# ─── HamiltonianShooter ───────────────────────────────────────────────────────

class HamiltonianShooter:
    """
    Pontryagin/Mayer-form Hamiltonian IVP shooter with full adjoint ODE.

    Parameters
    ----------
    hull       : VelocityHull
    era5       : ERA5Interpolator
    n_headings : int  angular resolution for Hamiltonian argmax (default 360)
    """

    def __init__(self, hull, era5, n_headings: int = 360):
        _sim.configure(hull, era5)
        self._hull       = hull
        self._era5       = era5
        self._n_headings = n_headings
        self._angles     = np.linspace(0, 2 * np.pi, n_headings, endpoint=False)

        # Precompute nearest hull-ray index for each test heading — constant,
        # so no need to call searchsorted inside the hot rhs() loop.
        _idx = np.searchsorted(hull.hull_angles, self._angles % (2 * np.pi))
        self._idx_nearest = np.clip(_idx, 0, len(hull.hull_angles) - 1)

    # ------------------------------------------------------------------
    # Core RHS
    # ------------------------------------------------------------------

    def rhs(self, t: float, z: np.ndarray) -> np.ndarray:
        """
        State: z = [lat, lon, λ1, λ2]  (deg, deg, dimensionless, dimensionless)

        Returns dz/dt = [dlat/dt, dlon/dt, dλ1/dt, dλ2/dt]

        Performance notes
        -----------------
        All ERA5 queries are batched into a single array call (5 points for
        wind + spatial gradient).  Hull radii are interpolated once per step
        (not once per test heading).  idx_nearest is precomputed in __init__.
        """
        x   = z[:2]
        lam = z[2:]

        # ── Steps 1+2: centre-point wind + spatial gradient in one query ────────
        # All 5 points share the same unix timestamp t, so a single batched call
        # hits the time-weight cache once instead of twice and reads the bilinear
        # slab pair exactly once.
        _dx = 1e-3   # degrees ≈ 111 m
        _lats5 = np.array([x[0],     x[0]+_dx, x[0]-_dx, x[0],     x[0]    ])
        _lons5 = np.array([x[1],     x[1],     x[1],     x[1]+_dx, x[1]-_dx])
        _u5, _v5 = self._era5.query(_lats5, _lons5, t)   # each (5,)

        _u0, _v0 = float(_u5[0]), float(_v5[0])
        _u,  _v  = _u5[1:], _v5[1:]   # gradient perturbation points (4,)

        V_ref = float(np.hypot(_u0, _v0))
        if V_ref < DS_THRESHOLD:
            return np.zeros(4)

        W      = np.array([_u0, _v0])
        grad_W = np.array([
            [(_u[0] - _u[1]) / (2*_dx),  (_u[2] - _u[3]) / (2*_dx)],
            [(_v[0] - _v[1]) / (2*_dx),  (_v[2] - _v[3]) / (2*_dx)],
        ])  # (2, 2): rows = [u, v], cols = [∂/∂lat, ∂/∂lon]

        alpha = np.arctan2(-W[1], -W[0])
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)

        # ── Step 3: argmax — one _interpolate_radii call for this V_ref ──────
        # All n_headings test angles share the same V_ref, so one call suffices.
        radii_1d = self._hull._interpolate_radii(np.array([V_ref]))[0]  # (n_rays,)
        r_test   = radii_1d[self._idx_nearest]                          # (n_headings,)

        angles     = self._angles
        u_wind_all = r_test * np.sin(angles)
        v_wind_all = r_test * np.cos(angles)
        vx_all     = v_wind_all * cos_a - u_wind_all * sin_a
        vy_all     = v_wind_all * sin_a + u_wind_all * cos_a

        lat_rad = np.deg2rad(float(x[0]))
        cos_lat = np.cos(lat_rad)
        dlat_all = vy_all / 111320.0
        dlon_all = vx_all / (111320.0 * max(abs(cos_lat), 1e-6))

        dot    = lam[0] * dlat_all + lam[1] * dlon_all
        u_star = angles[int(np.argmax(dot))]

        # ── Step 4: optimal velocity (reuse radii_1d) ────────────────────────
        sin_u  = np.sin(u_star)
        cos_u  = np.cos(u_star)
        r_star = float(np.interp(u_star % (2*np.pi), self._hull.hull_angles, radii_1d))
        vx_star = (r_star * cos_u) * cos_a - (r_star * sin_u) * sin_a
        vy_star = (r_star * cos_u) * sin_a + (r_star * sin_u) * cos_a

        f = np.array([
            vy_star / 111320.0,
            vx_star / (111320.0 * max(abs(cos_lat), 1e-6)),
        ])  # [dlat/dt, dlon/dt] deg/s

        # ── Step 5: hull Jacobian — one batched _interpolate_radii call ──────
        # Four perturbed wind vectors: ±dw in u10, ±dw in v10.
        _dw = 0.5
        _u4 = np.array([W[0]+_dw, W[0]-_dw, W[0],     W[0]    ])
        _v4 = np.array([W[1],     W[1],     W[1]+_dw, W[1]-_dw])
        _Vr4    = np.hypot(_u4, _v4)                               # (4,)
        _alp4   = np.arctan2(-_v4, -_u4)                          # (4,)
        _rad4   = self._hull._interpolate_radii(_Vr4)             # (4, n_rays)
        _r4     = np.array([
            float(np.interp(u_star % (2*np.pi), self._hull.hull_angles, _rad4[k]))
            for k in range(4)
        ])

        def _geo_vel(k):
            ca = np.cos(_alp4[k]); sa = np.sin(_alp4[k])
            uw = _r4[k] * sin_u;   vw = _r4[k] * cos_u
            return np.array([vw*ca - uw*sa, vw*sa + uw*ca])

        Vp0, Vm0 = _geo_vel(0), _geo_vel(1)   # W[0] ± dw
        Vp1, Vm1 = _geo_vel(2), _geo_vel(3)   # W[1] ± dw

        dV_dW = np.array([
            [(Vp0[0]-Vm0[0])/(2*_dw), (Vp1[0]-Vm1[0])/(2*_dw)],
            [(Vp0[1]-Vm0[1])/(2*_dw), (Vp1[1]-Vm1[1])/(2*_dw)],
        ])  # rows = [vx_East, vy_North], cols = [∂/∂u10, ∂/∂v10]

        # ── Step 6: costate ODE  λ̇ = −(λ⊺ dF/dW) · ∂W/∂x ─────────────────
        dF_dW = np.array([
            [dV_dW[1, 0] / 111320.0,           dV_dW[1, 1] / 111320.0],
            [dV_dW[0, 0] / (111320.0 * max(abs(cos_lat), 1e-6)),
             dV_dW[0, 1] / (111320.0 * max(abs(cos_lat), 1e-6))],
        ])

        lam_dot = -(lam @ dF_dW) @ grad_W

        return np.array([f[0], f[1], lam_dot[0], lam_dot[1]])

    # ------------------------------------------------------------------
    # Single trajectory
    # ------------------------------------------------------------------

    def shoot(
        self,
        x0: tuple[float, float],
        phi: float,
        t0: float,
        T: float,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> dict:
        """
        Integrate one Hamiltonian trajectory.

        Parameters
        ----------
        x0  : (lat, lon) start position [deg]
        phi : initial costate angle [rad]
        t0  : start time (unix timestamp)
        T   : duration [s]
        rtol, atol : solver tolerances.  Default 1e-3 / 1e-5 balances speed and
                     accuracy: endpoint error ~10 km vs 12 km physical uncertainty
                     from bilinear-interpolated wind.  Tighten to 1e-6/1e-8 only
                     if Hamiltonian conservation is a concern.

        Returns
        -------
        dict with keys: xs, ys, lam1, lam2, ts, endpoint, e_hat, H_values, phi
        """
        z0 = np.array([x0[0], x0[1], np.cos(phi), np.sin(phi)])

        sol = solve_ivp(
            self.rhs,
            [t0, t0 + T],
            z0,
            method='RK45',
            dense_output=False,
            rtol=rtol,
            atol=atol,
        )

        lam_T = sol.y[2:, -1]
        lam_T_norm = np.linalg.norm(lam_T)
        e_hat = lam_T / lam_T_norm if lam_T_norm > 1e-12 else lam_T

        # Hamiltonian: H ≈ λ · (Δx/Δt) estimated from stored trajectory
        # (avoids redundant rhs calls; exact H would require re-running the argmax)
        lam_traj = sol.y[2:, :]   # (2, n_t)
        x_traj   = sol.y[:2, :]   # (2, n_t)  [lat, lon]
        dt_arr   = np.diff(sol.t)  # (n_t - 1,)
        dx_arr   = np.diff(x_traj, axis=1)   # (2, n_t - 1)  in degrees
        # midpoint costate
        lam_mid  = 0.5 * (lam_traj[:, :-1] + lam_traj[:, 1:])   # (2, n_t-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            f_mid = dx_arr / np.where(dt_arr > 0, dt_arr, np.inf)  # deg/s
        H_mid    = np.einsum('ij,ij->j', lam_mid, f_mid)           # (n_t-1,)
        # Pad to full length (first value mirrored from second)
        H_values = np.concatenate([[H_mid[0]] if len(H_mid) else [0.0], H_mid])

        return {
            'xs':       sol.y[1, :],          # lon trajectory
            'ys':       sol.y[0, :],          # lat trajectory
            'lam1':     sol.y[2, :],
            'lam2':     sol.y[3, :],
            'ts':       sol.t,
            'endpoint': sol.y[:2, -1],        # (lat, lon)
            'e_hat':    e_hat,
            'H_values': H_values,
            'phi':      phi,
        }

    # ------------------------------------------------------------------
    # Fan shooting
    # ------------------------------------------------------------------

    def shoot_fan(
        self,
        x0: tuple[float, float],
        t0: float,
        T: float,
        n_angles: int = 360,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        n_workers: int | None = None,
    ) -> list[dict]:
        """
        Shoot fan of trajectories over all initial costate angles.

        Parameters
        ----------
        x0        : (lat, lon) start position [deg]
        t0        : start time (unix timestamp)
        T         : duration [s]
        n_angles  : number of initial costate angles
        rtol, atol : solver tolerances
        n_workers : number of parallel worker processes.
                    Defaults to os.cpu_count().  Set to 1 to disable parallelism.
                    Parallelism uses fork (Linux/macOS) so the large ERA5 arrays
                    are shared read-only via copy-on-write without pickling.

        Returns
        -------
        list of shoot() dicts (failed integrations silently skipped),
        in the same order as the initial phi angles.
        """
        import warnings as _warnings

        global _FORKED_SHOOTER

        phis = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        worker_args = [(float(phi), x0, t0, T, rtol, atol) for phi in phis]

        # Determine effective worker count
        n_cpu = os.cpu_count() or 1
        n_w   = n_workers if n_workers is not None else n_cpu
        n_w   = max(1, min(n_w, n_angles))

        print(f"Shooting fan of {n_angles} trajectories …")

        # ── Serial fallback ───────────────────────────────────────────────────
        if n_w <= 1:
            results = []
            for i, phi in enumerate(phis):
                try:
                    r = self.shoot(x0, phi, t0, T, rtol=rtol, atol=atol)
                    results.append(r)
                except Exception as exc:
                    print(f"  φ={phi:.3f} failed: {exc}")
                if (i + 1) % 45 == 0 or (i + 1) == n_angles:
                    print(f"  {i + 1}/{n_angles} trajectories complete")
            return results

        # ── Fork-based parallel execution ─────────────────────────────────────
        # Set the module-level global BEFORE creating the Pool so that the forked
        # workers inherit the reference to this shooter (including its ERA5 arrays)
        # via copy-on-write — no pickling of large data required.
        try:
            ctx = multiprocessing.get_context('fork')
        except ValueError:
            # 'fork' unavailable (Windows) — fall back to serial
            _warnings.warn(
                "multiprocessing 'fork' context unavailable on this platform. "
                "Falling back to serial fan shooting.",
                stacklevel=2,
            )
            return self.shoot_fan(
                x0, t0, T, n_angles=n_angles,
                rtol=rtol, atol=atol, n_workers=1,
            )

        _FORKED_SHOOTER = self   # inherited by workers after fork

        results_with_none: list[dict | None]
        with ctx.Pool(processes=n_w) as pool:
            results_with_none = list(
                pool.imap(_shoot_worker, worker_args, chunksize=max(1, n_angles // (n_w * 4)))
            )

        # Filter failures, preserve order
        results = [r for r in results_with_none if r is not None]
        n_failed = results_with_none.count(None)
        if n_failed:
            print(f"  {n_failed} trajectory/trajectories failed (skipped)")
        print(f"  {len(results)}/{n_angles} trajectories complete")
        return results

    # ------------------------------------------------------------------
    # Migration potential (support function)
    # ------------------------------------------------------------------

    @staticmethod
    def migration_potential(fan_results: list[dict], n_query: int = 360) -> dict:
        """
        Support function of the Hamiltonian endpoint cloud (lat/lon, degrees).

        For each query direction d, returns max_{endpoint} d · endpoint.
        This is the boundary of the reachable set in position space.

        Returns dict with keys: angles, potential, endpoints, e_hats
        """
        endpoints = np.array([r['endpoint'] for r in fan_results])   # (N, 2)
        e_hats    = np.array([r['e_hat']    for r in fan_results])    # (N, 2)

        query_angles = np.linspace(0, 2 * np.pi, n_query, endpoint=False)
        query_dirs   = np.column_stack([
            np.cos(query_angles),
            np.sin(query_angles),
        ])  # (n_query, 2)

        potential_mat = query_dirs @ endpoints.T    # (n_query, N)
        potential     = potential_mat.max(axis=1)   # (n_query,)

        return {
            'angles':    query_angles,
            'potential': potential,
            'endpoints': endpoints,
            'e_hats':    e_hats,
        }

    @staticmethod
    def velocity_hull_migration_potential(
        hull,
        W: np.ndarray,
        T: float,
        n_query: int = 360,
    ) -> dict:
        """
        Support function of the velocity hull (from tacking_diagram / velocity_hulls.npz)
        in geographic frame, scaled by duration T.

        For each geographic direction d, computes:
            h(d) = max_{θ ∈ hull} r(θ, V_ref) · (d rotated to wind frame)
        and returns h(d) · T / 111320 in degrees (same units as lat/lon endpoints).

        Parameters
        ----------
        hull    : VelocityHull  (from VelocityHull.from_npz('velocity_hulls.npz'))
        W       : (2,) [u10, v10] m/s — wind at start position
        T       : float — duration [s]
        n_query : int — angular resolution

        Returns
        -------
        dict with keys:
            angles       : (n_query,) geographic query angles [rad]
            potential_deg: (n_query,) support function in degrees (lat/lon)
            potential_km : (n_query,) support function in km
            V_ref        : float — local wind speed used
            alpha        : float — wind-from angle [rad]
        """
        V_ref = float(np.hypot(W[0], W[1]))
        query_angles = np.linspace(0, 2 * np.pi, n_query, endpoint=False)

        if V_ref < DS_THRESHOLD:
            return {
                'angles':        query_angles,
                'potential_deg': np.zeros(n_query),
                'potential_km':  np.zeros(n_query),
                'V_ref':         V_ref,
                'alpha':         0.0,
            }

        # Wind geometry
        alpha = np.arctan2(-W[1], -W[0])   # direction wind blows FROM, CCW from East
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)

        # Hull radii at this V_ref
        hull_angles = hull.hull_angles                                   # (n_rays,)
        radii_1d    = hull._interpolate_radii(np.array([V_ref]))[0]     # (n_rays,)
        sin_h = np.sin(hull_angles)   # (n_rays,)
        cos_h = np.cos(hull_angles)   # (n_rays,)

        # Geographic query directions
        d_geo = np.column_stack([
            np.cos(query_angles),   # East component
            np.sin(query_angles),   # North component
        ])  # (n_query, 2)

        # Rotate each geographic direction into wind frame
        # d_v_wind (upwind):   cos(α)·d_x + sin(α)·d_y
        # d_u_wind (crosswind): -sin(α)·d_x + cos(α)·d_y
        d_v_wind = cos_a * d_geo[:, 0] + sin_a * d_geo[:, 1]   # (n_query,)
        d_u_wind = -sin_a * d_geo[:, 0] + cos_a * d_geo[:, 1]  # (n_query,)

        # Support function: max_θ r(θ) · (d_u·sin(θ) + d_v·cos(θ))
        # shape (n_query, n_rays)
        scores = radii_1d[None, :] * (
            d_u_wind[:, None] * sin_h[None, :]
            + d_v_wind[:, None] * cos_h[None, :]
        )
        h_speed = np.maximum(scores.max(axis=1), 0.0)  # (n_query,) m/s

        potential_m   = h_speed * T
        potential_km  = potential_m / 1e3
        potential_deg = potential_m / 111320.0

        return {
            'angles':        query_angles,
            'potential_deg': potential_deg,
            'potential_km':  potential_km,
            'V_ref':         V_ref,
            'alpha':         alpha,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate(fan_results: list[dict]) -> dict:
        """
        Run three sanity checks on a completed fan.

        Returns
        -------
        dict with keys: h_conservation, e_hat_coverage, speed_sanity, passed
        """
        # 1. Hamiltonian conservation
        h_flags = []
        h_ratios = []
        for r in fan_results:
            H = r['H_values']
            mean_H = float(np.mean(H))
            if abs(mean_H) > 1e-12:
                ratio = float(np.std(H) / abs(mean_H))
            else:
                ratio = 0.0
            h_ratios.append(ratio)
            h_flags.append(ratio > 1e-2)

        h_conservation = {
            'ratios':      h_ratios,
            'n_flagged':   int(sum(h_flags)),
            'max_ratio':   float(max(h_ratios)) if h_ratios else float('nan'),
            'passed':      not any(h_flags),
        }

        # 2. e_hat coverage — max angular gap
        e_hats = np.array([r['e_hat'] for r in fan_results])
        angles_revealed = np.arctan2(e_hats[:, 1], e_hats[:, 0])
        angles_sorted   = np.sort(angles_revealed % (2 * np.pi))
        if len(angles_sorted) > 1:
            gaps = np.diff(angles_sorted)
            wrap_gap = (2 * np.pi - angles_sorted[-1]) + angles_sorted[0]
            max_gap_deg = float(np.degrees(max(gaps.max(), wrap_gap)))
        else:
            max_gap_deg = 360.0

        e_hat_coverage = {
            'max_gap_deg': max_gap_deg,
            'passed':      max_gap_deg <= 15.0,
        }

        # 3. Speed sanity
        max_speed = 0.0
        for r in fan_results:
            xs = r['xs']  # lon
            ys = r['ys']  # lat
            ts = r['ts']
            if len(ts) > 1:
                dlat = np.diff(ys) * 111320.0
                dlon = np.diff(xs) * 111320.0 * np.cos(np.deg2rad(np.mean(ys)))
                dt   = np.diff(ts)
                with np.errstate(divide='ignore', invalid='ignore'):
                    speeds = np.hypot(dlat, dlon) / np.where(dt > 0, dt, np.inf)
                max_speed = max(max_speed, float(np.nanmax(speeds)))

        speed_sanity = {
            'max_speed_ms': max_speed,
            'passed':       max_speed <= 50.0,
        }

        passed = (
            h_conservation['passed']
            and e_hat_coverage['passed']
            and speed_sanity['passed']
        )

        return {
            'h_conservation': h_conservation,
            'e_hat_coverage': e_hat_coverage,
            'speed_sanity':   speed_sanity,
            'passed':         passed,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def plot_isocurves(
        fan_results: list[dict],
        days: list[int] | None = None,
        ax=None,
        output: str | Path | None = None,
    ):
        """
        Plot Hamiltonian fan reachability isocurves.

        For each day in `days`, interpolates every fan trajectory at
        t0 + day * 86400 and draws the resulting closed isocurve —
        matching the style of GreedyMigration.plot_isocurves().

        Parameters
        ----------
        fan_results : list of shoot() dicts (each with xs, ys, ts)
        days        : day indices to plot (default [1, 2, 3, 5, 7])
        ax          : existing axes (created if None)
        output      : if given, save figure to this path

        Returns
        -------
        (fig, ax)
        """
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            HAS_CARTOPY = True
        except ImportError:
            HAS_CARTOPY = False

        import matplotlib.pyplot as plt

        if not fan_results:
            raise ValueError("fan_results is empty")

        t0 = float(fan_results[0]['ts'][0])
        T  = float(fan_results[0]['ts'][-1]) - t0

        if days is None:
            days = [d for d in [1, 2, 3, 5, 7] if d * 86400 <= T]

        cmap   = plt.cm.viridis
        colors = {d: cmap(i / max(len(days) - 1, 1)) for i, d in enumerate(days)}

        proj = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=proj)
        else:
            fig = ax.get_figure()

        trans = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}

        for d in days:
            t_query = t0 + d * 86400.0
            lats, lons = [], []
            for r in fan_results:
                ts = r['ts']
                if t_query > ts[-1] + 1:
                    continue
                lats.append(float(np.interp(t_query, ts, r['ys'])))
                lons.append(float(np.interp(t_query, ts, r['xs'])))
            if len(lats) < 2:
                continue

            lats = np.array(lats)
            lons = np.array(lons)

            # Sort points by angle around their centroid so the closed curve
            # has no crossings (φ order ≠ endpoint angular order).
            clat, clon = lats.mean(), lons.mean()
            order = np.argsort(np.arctan2(lats - clat, lons - clon))
            lats, lons = lats[order], lons[order]

            lats_c = np.append(lats, lats[0])
            lons_c = np.append(lons, lons[0])
            ax.plot(lons_c, lats_c, color=colors[d], lw=1.5,
                    label=f'Day {d}', **trans)

        # Mark start position
        lat0 = float(fan_results[0]['ys'][0])
        lon0 = float(fan_results[0]['xs'][0])
        ax.scatter([lon0], [lat0], c='red', s=60, zorder=5, label='Start', **trans)

        if HAS_CARTOPY:
            ax.add_feature(cfeature.LAND, facecolor='#d0d0d0', zorder=3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=4)
            ax.gridlines(draw_labels=True, linewidth=0.4, linestyle='--', alpha=0.5)

        ax.legend(loc='upper right')
        ax.set_title("Hamiltonian reachability isocurves")

        if output:
            from pathlib import Path as _Path
            out = _Path(output)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f"Saved → {out}")

        return fig, ax

    def plot_fan(
        self,
        fan_results: list[dict],
        migration_pot: dict,
        hull_pot: dict | None = None,
    ):
        """
        Two-panel figure: trajectory map + migration potential polar plot.

        Parameters
        ----------
        fan_results   : list of shoot() dicts
        migration_pot : output of migration_potential() — endpoint cloud support fn
        hull_pot      : output of velocity_hull_migration_potential() — velocity hull
                        support fn in geographic frame.  If provided, the polar panel
                        shows the hull-based potential (physics-grounded) and overlays
                        the endpoint support function as a dashed comparison curve.

        Returns
        -------
        (fig, axes) : matplotlib figure and (ax_map, ax_polar)
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        fig = plt.figure(figsize=(14, 6))
        ax_map   = fig.add_subplot(1, 2, 1)
        ax_polar = fig.add_subplot(1, 2, 2, projection='polar')

        # ── Left panel: trajectories coloured by e_hat angle ──────────
        e_hats   = np.array([r['e_hat'] for r in fan_results])
        e_angles = np.arctan2(e_hats[:, 1], e_hats[:, 0]) % (2 * np.pi)

        cmap = plt.cm.viridis
        norm = Normalize(vmin=0, vmax=2 * np.pi)

        for i, r in enumerate(fan_results):
            color = cmap(norm(e_angles[i]))
            ax_map.plot(r['xs'], r['ys'], color=color, lw=0.4, alpha=0.5)

        # Mark start
        if fan_results:
            lat0, lon0 = fan_results[0]['ys'][0], fan_results[0]['xs'][0]
            ax_map.scatter([lon0], [lat0], c='red', s=60, zorder=5, label='Start')

        # Convex hull of endpoints
        endpoints = migration_pot['endpoints']
        if len(endpoints) >= 3:
            try:
                ch = ConvexHull(endpoints[:, ::-1])   # (lon, lat) for 2-D hull
                verts_lon = endpoints[:, 1][ch.vertices]
                verts_lat = endpoints[:, 0][ch.vertices]
                ax_map.plot(
                    np.append(verts_lon, verts_lon[0]),
                    np.append(verts_lat, verts_lat[0]),
                    'k--', lw=1.2, label='Endpoint hull',
                )
            except Exception:
                pass

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_map, label='e_hat angle [rad]', shrink=0.75)

        ax_map.set_xlabel('Longitude [°E]')
        ax_map.set_ylabel('Latitude [°N]')
        ax_map.set_title(
            'Hamiltonian trajectories\n(coloured by terminal costate direction)'
        )
        ax_map.legend(loc='upper right', fontsize=8)

        # ── Right panel: migration potential polar ─────────────────────
        if hull_pot is not None:
            # Primary curve: velocity hull support function (physics-grounded,
            # units: degrees of lat/lon ≈ distance)
            angles_h  = hull_pot['angles']
            pot_h     = hull_pot['potential_deg']
            V_ref_lbl = hull_pot['V_ref']

            ax_polar.plot(angles_h, pot_h, lw=2.0, color='steelblue',
                          label=f'Hull support (V_ref={V_ref_lbl:.1f} m/s)')
            ax_polar.fill(angles_h, pot_h, alpha=0.20, color='steelblue')

            # Overlay: endpoint cloud support function as relative displacement (dashed)
            # h_rel(d) = max_endpoint d·(endpoint − x0) gives displacement in degrees
            angles_e = migration_pot['angles']
            if fan_results:
                x0 = np.array([fan_results[0]['ys'][0], fan_results[0]['xs'][0]])
                endpoints_rel = migration_pot['endpoints'] - x0[None, :]  # (N, 2)
                query_dirs_e  = np.column_stack([
                    np.cos(angles_e), np.sin(angles_e)
                ])  # (n_q, 2)
                pot_e = (query_dirs_e @ endpoints_rel.T).max(axis=1)  # (n_q,) degrees
                pot_e = np.maximum(pot_e, 0.0)
                ax_polar.plot(angles_e, pot_e, lw=1.2, color='orange',
                              ls='--', label='Endpoint cloud (deg)')

            ax_polar.set_title(
                'Migration potential\n'
                f'(velocity hull support fn, V_ref={V_ref_lbl:.1f} m/s)',
                pad=14,
            )
            ax_polar.legend(loc='upper right', fontsize=7)
        else:
            # Fallback: endpoint support function only
            angles  = migration_pot['angles']
            pot     = migration_pot['potential']
            ax_polar.plot(angles, pot, lw=1.5, color='steelblue')
            ax_polar.fill(angles, pot, alpha=0.25, color='steelblue')
            ax_polar.set_title('Migration potential\n(endpoint cloud support fn)', pad=14)

        return fig, (ax_map, ax_polar)


# ─── __main__ quick test ──────────────────────────────────────────────────────

if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from albatross.macroscale import VelocityHull
    from albatross.macroscale.simulation import configure

    DATA_DIR = Path(__file__).resolve().parents[3] / 'data'

    hull = VelocityHull.from_npz(DATA_DIR / 'velocity_hulls.npz')

    # Without ERA5, we can only test the non-wind-dependent parts.
    # For a real run, load ERA5 and set RECOMPUTE=True in the notebook.
    print("Hull loaded:", hull.v_ref_levels)
    print("n_headings test with n_angles=36 would require ERA5 data.")
    print("Set RECOMPUTE=True in notebooks/migration.ipynb to run the full test.")
