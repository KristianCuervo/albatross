"""
path_optimizer.py
=================
CasADI NLP for coarse-grained routing: find the sequence of N soaring-cycle
headings that minimises the terminal distance to a goal position.

Problem structure
-----------------
  Decision variables
    (x_k, y_k)   position after cycle k,   k = 0 … N   (x_0 fixed = start)
    phi_k         heading relative to wind, k = 0 … N-1  [rad, 0…2π]

  Dynamics (one soaring cycle per step):
    V_ref_k = wind_speed(x_k, y_k)          [from 2-D CasADI interpolant]
    alpha_k = wind_from_direction(x_k, y_k) [from 2-D CasADI interpolant]
    u_k, v_k, T_k = tacking_lookup(phi_k, V_ref_k)   [2-D bspline/linear]
    vx_k = v_k·cos(alpha_k) − u_k·sin(alpha_k)
    vy_k = v_k·sin(alpha_k) + u_k·cos(alpha_k)
    x_{k+1} = x_k + vx_k · T_k
    y_{k+1} = y_k + vy_k · T_k

  Objective:  minimise (x_N − goal_x)² + (y_N − goal_y)²

Tacking diagram pre-processing
-------------------------------
The NPZ stores 808 data points on an irregular (theta ∈ [0,π], V_ref ∈ [9,25])
grid (original half only).  We reconstruct a regular 2-D grid over
phi_rel ∈ [0, 2π] by mirroring u → −u for phi > π, then build CasADI
'linear' interpolants (which accept non-uniform axes — however we use a
uniform phi grid and uniform V_ref integers).

Wind field pre-processing
-------------------------
We reconstruct the raw velocity components ux = −V_ref·cos(α) and
uy = −V_ref·sin(α) and build two separate 2-D linear interpolants over
(x, y).  This avoids the atan2 discontinuity that would arise from
interpolating α directly.  V_ref and α are then recovered symbolically:
  V_ref = sqrt(ux² + uy²)
  alpha = atan2(−uy, −ux)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


# --------------------------------------------------------------------------- #
# Step 1 — build regular tacking grid                                         #
# --------------------------------------------------------------------------- #

def build_tacking_grid(npz_path: str, N_phi: int = 120):
    """
    Construct a regular (N_phi+1) × 17 grid for (u_wind, v_wind, T_cycle)
    as functions of (phi_rel, V_ref).

    phi_rel ∈ [0, 2π] — heading relative to wind direction:
      phi_rel ∈ [0, π]  → original data  (u_avg ≥ 0, crosswind to the right)
      phi_rel ∈ (π, 2π] → mirrored data  (u_avg ≤ 0, crosswind to the left)
    V_ref ∈ {9, 10, …, 25} m/s.

    An extra wrap-around column is appended at phi = 2π (same values as φ=0)
    so that the CasADI interpolant handles φ ≈ 2π gracefully.

    Returns
    -------
    phi_axis  : (N_phi+1,)  uniformly spaced 0 … 2π
    vref_axis : (17,)       [9, 10, …, 25]
    u_grid    : (N_phi+1, 17)
    v_grid    : (N_phi+1, 17)
    T_grid    : (N_phi+1, 17)
    """
    data = np.load(npz_path)

    unique_thetas = np.unique(data['theta'])          # 58 values in [0, π]
    vref_axis     = np.arange(9.0, 26.0)              # 17 integer levels

    # Group by theta → sorted V_ref tables
    theta_tables: dict = {}
    for theta in unique_thetas:
        mask  = data['theta'] == theta
        order = np.argsort(data['V_ref'][mask])
        theta_tables[theta] = {
            'V_ref': data['V_ref'][mask][order],
            'u':     data['u_avg'][mask][order],
            'v':     data['v_avg'][mask][order],
            'T':     data['T_cycle'][mask][order],
        }

    # Regular phi grid (excluding the 2π endpoint for now)
    phi_core = np.linspace(0.0, 2.0 * np.pi, N_phi, endpoint=False)
    u_core   = np.zeros((N_phi, 17))
    v_core   = np.zeros((N_phi, 17))
    T_core   = np.zeros((N_phi, 17))

    for i, phi in enumerate(phi_core):
        # Map to [0, π] + mirror flag
        if phi <= np.pi:
            theta_q  = phi
            mirrored = False
        else:
            theta_q  = 2.0 * np.pi - phi
            mirrored = True

        for j, vref in enumerate(vref_axis):
            u_val, v_val, T_val = _interp_tacking(
                theta_tables, unique_thetas, theta_q, vref
            )
            if mirrored:
                u_val = -u_val
            u_core[i, j] = u_val
            v_core[i, j] = v_val
            T_core[i, j] = T_val

    # Append wrap-around row at phi = 2π (= phi = 0)
    phi_axis = np.append(phi_core, 2.0 * np.pi)
    u_grid   = np.vstack([u_core, u_core[0:1, :]])
    v_grid   = np.vstack([v_core, v_core[0:1, :]])
    T_grid   = np.vstack([T_core, T_core[0:1, :]])

    return phi_axis, vref_axis, u_grid, v_grid, T_grid


def _interp_tacking(theta_tables, theta_orig, theta_q, V_ref_q):
    """Bilinear interpolation in (theta, V_ref) from the original half-plane."""
    # Bracket theta
    idx  = int(np.searchsorted(theta_orig, theta_q))
    idx  = int(np.clip(idx, 1, len(theta_orig) - 1))
    t_lo = theta_orig[idx - 1]
    t_hi = theta_orig[idx]
    w_th = (theta_q - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0.0

    def lookup_vref(d, vref):
        vr    = d['V_ref']
        V_eff = float(np.clip(vref, vr[0], vr[-1]))
        if V_eff <= vr[0]:
            return d['u'][0], d['v'][0], d['T'][0]
        if V_eff >= vr[-1]:
            return d['u'][-1], d['v'][-1], d['T'][-1]
        j = int(np.searchsorted(vr, V_eff))
        w = (V_eff - vr[j - 1]) / (vr[j] - vr[j - 1])
        return (
            (1.0 - w) * d['u'][j - 1] + w * d['u'][j],
            (1.0 - w) * d['v'][j - 1] + w * d['v'][j],
            (1.0 - w) * d['T'][j - 1] + w * d['T'][j],
        )

    u_lo, v_lo, T_lo = lookup_vref(theta_tables[t_lo], V_ref_q)
    u_hi, v_hi, T_hi = lookup_vref(theta_tables[t_hi], V_ref_q)
    u = (1.0 - w_th) * u_lo + w_th * u_hi
    v = (1.0 - w_th) * v_lo + w_th * v_hi
    T = (1.0 - w_th) * T_lo + w_th * T_hi
    return u, v, T


# --------------------------------------------------------------------------- #
# Step 2 — PathOptimizer                                                      #
# --------------------------------------------------------------------------- #

class PathOptimizer:
    """
    CasADI NLP for coarse-grained routing over N soaring cycles.

    Parameters
    ----------
    npz_path : str
        Path to refactor/data/tacking_diagram.npz.
    wind_field : dict
        Output of any wind_field constructor (x_centers, y_centers, V_ref, alpha).
    N : int
        Number of soaring cycles (waypoints = N+1).
    start, goal : (float, float)
        Start and goal positions [m].
    """

    def __init__(self, npz_path: str, wind_field: dict,
                 N: int = 11,
                 start: tuple = (50.0, 50.0),
                 goal:  tuple = (950.0, 950.0)) -> None:

        import casadi as ca
        self._ca = ca

        self.N     = N
        self.start = start
        self.goal  = goal

        # ---- tacking interpolants ----------------------------------------
        print("  Building tacking interpolants …", end=' ', flush=True)
        phi_ax, vref_ax, u_g, v_g, T_g = build_tacking_grid(npz_path)
        # CasADI interpolant: grids = [phi_axis, vref_axis],
        # values in column-major (Fortran) order: values[i + n_phi*j]
        n_phi = len(phi_ax)
        self._iu = ca.interpolant('u_w', 'linear', [phi_ax, vref_ax],
                                   u_g.flatten('F'))
        self._iv = ca.interpolant('v_w', 'linear', [phi_ax, vref_ax],
                                   v_g.flatten('F'))
        self._iT = ca.interpolant('T_c', 'linear', [phi_ax, vref_ax],
                                   T_g.flatten('F'))
        print("done")

        # ---- wind field interpolants -------------------------------------
        # Reconstruct velocity components to avoid atan2 discontinuity.
        xs    = wind_field['x_centers']
        ys    = wind_field['y_centers']
        alpha = wind_field['alpha']
        V_ref = wind_field['V_ref']
        ux_w  = -V_ref * np.cos(alpha)   # where wind blows TO
        uy_w  = -V_ref * np.sin(alpha)

        print("  Building wind interpolants …", end=' ', flush=True)
        # CasADI expects [x_axis, y_axis] with values[i + nx*j] = f(xs[i], ys[j]).
        # Use 'bspline' (C2 smooth) so IPOPT gets consistent gradients as
        # positions cross grid-cell boundaries — critical when step size ≈ cell size.
        self._iux = ca.interpolant('ux', 'bspline', [xs, ys],
                                    ux_w.flatten('F'))
        self._iuy = ca.interpolant('uy', 'bspline', [xs, ys],
                                    uy_w.flatten('F'))
        print("done")

        # Domain bounds derived from the grid (used for variable bounds in NLP)
        self._x_min = float(xs[0])
        self._x_max = float(xs[-1])
        self._y_min = float(ys[0])
        self._y_max = float(ys[-1])

    # ---------------------------------------------------------------------- #

    def _step(self, x, y, phi):
        """
        CasADI expression for one-cycle position update.

        Returns (x_new, y_new, T_cycle).
        """
        ca = self._ca

        # Wind velocity components at (x, y)
        ux = self._iux(ca.vertcat(x, y))
        uy = self._iuy(ca.vertcat(x, y))

        # Derived wind quantities (symbolic)
        spd   = ca.sqrt(ux ** 2 + uy ** 2 + 1e-6)   # avoid sqrt(0)
        V_ref = ca.fmax(9.0, ca.fmin(25.0, spd))
        alpha = ca.atan2(-uy, -ux)                    # direction wind FROM

        # Tacking lookup
        u_w = self._iu(ca.vertcat(phi, V_ref))
        v_w = self._iv(ca.vertcat(phi, V_ref))
        T_c = self._iT(ca.vertcat(phi, V_ref))

        # Wind-frame → global rotation
        vx = v_w * ca.cos(alpha) - u_w * ca.sin(alpha)
        vy = v_w * ca.sin(alpha) + u_w * ca.cos(alpha)

        return x + vx * T_c, y + vy * T_c, T_c

    # ---------------------------------------------------------------------- #

    def optimise(
        self,
        x0_traj:    np.ndarray | None = None,
        phi0:       np.ndarray | None = None,
        ipopt_opts: dict | None       = None,
        obj_scale:  float             = 1.0,
    ) -> dict:
        """
        Run the NLP.

        Parameters
        ----------
        x0_traj : array of shape (M, 2), optional
            Warm-start positions.  Any length M is accepted — the trajectory is
            resampled to N+1 points via linear interpolation.  Defaults to a
            straight line start → goal.
        phi0 : (N,) array, optional
            Warm-start headings [rad].  Defaults to pointing toward goal in wind
            frame at each warm-start position.
        ipopt_opts : dict, optional
            Extra IPOPT options.
        obj_scale : float, optional
            Divide the objective (squared distance) by this value for better
            numerical scaling.  A typical choice is ``domain ** 2``.

        Returns
        -------
        dict
          'positions'  : (N+1, 2)  optimised position sequence
          'phi_rel'    : (N,)      optimised heading sequence [rad]
          'T_cycles'   : (N,)      cycle durations at each step [s]
          'success'    : bool
        """
        ca = self._ca
        N  = self.N
        sx, sy = self.start
        gx, gy = self.goal

        # ---- resample / default initial trajectory ----------------------
        if x0_traj is None:
            ts      = np.linspace(0.0, 1.0, N + 1)
            x0_traj = np.column_stack([sx + ts * (gx - sx),
                                       sy + ts * (gy - sy)])
        else:
            # Resample any-length path to exactly N+1 waypoints
            M   = len(x0_traj)
            if M != N + 1:
                t_in  = np.linspace(0.0, 1.0, M)
                t_out = np.linspace(0.0, 1.0, N + 1)
                x0_traj = np.column_stack([
                    np.interp(t_out, t_in, x0_traj[:, 0]),
                    np.interp(t_out, t_in, x0_traj[:, 1]),
                ])

        # ---- default phi0: heading toward goal in wind frame ----------
        if phi0 is None:
            phi0 = np.zeros(N)
            for k in range(N):
                xk, yk   = x0_traj[k]
                ux_k     = float(self._iux(ca.DM([xk, yk])))
                uy_k     = float(self._iuy(ca.DM([xk, yk])))
                alpha_k  = np.arctan2(-uy_k, -ux_k)
                phi_goal = np.arctan2(gy - yk, gx - xk)
                # Heading relative to wind ≈ global heading − wind FROM direction + π
                # (since phi_rel=0 means flying INTO wind = against wind FROM direction)
                phi0[k] = (phi_goal - alpha_k) % (2.0 * np.pi)

        # ---- CasADI Opti -----------------------------------------------
        opti = ca.Opti()

        X   = opti.variable(N + 1)   # x positions
        Y   = opti.variable(N + 1)   # y positions
        Phi = opti.variable(N)       # headings relative to wind

        # Fixed start
        opti.subject_to(X[0] == sx)
        opti.subject_to(Y[0] == sy)

        # Dynamics
        T_sym = []
        for k in range(N):
            x_new, y_new, T_k = self._step(X[k], Y[k], Phi[k])
            opti.subject_to(X[k + 1] == x_new)
            opti.subject_to(Y[k + 1] == y_new)
            T_sym.append(T_k)

        # Bounds (derived from wind-field grid extents)
        opti.subject_to(opti.bounded(self._x_min, X, self._x_max))
        opti.subject_to(opti.bounded(self._y_min, Y, self._y_max))
        opti.subject_to(opti.bounded(0.0,  Phi, 2.0 * np.pi))

        # Objective: minimise squared terminal distance to goal, normalised
        opti.minimize(((X[N] - gx) ** 2 + (Y[N] - gy) ** 2) / obj_scale)

        # Initial guesses
        opti.set_initial(X,   x0_traj[:, 0])
        opti.set_initial(Y,   x0_traj[:, 1])
        opti.set_initial(Phi, phi0)

        # Solver settings
        base_opts = {
            'ipopt.print_level':    3,
            'ipopt.max_iter':       1200,
            'ipopt.tol':            1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_iter': 5,
            'print_time':           False,
        }
        if ipopt_opts:
            base_opts.update(ipopt_opts)
        opti.solver('ipopt', base_opts)

        # ---- solve -------------------------------------------------------
        success = True
        try:
            sol     = opti.solve()
            X_val   = np.array(sol.value(X)).flatten()
            Y_val   = np.array(sol.value(Y)).flatten()
            Phi_val = np.array(sol.value(Phi)).flatten()
            T_val   = np.array([float(sol.value(t)) for t in T_sym])
        except RuntimeError as exc:
            print(f"\n  IPOPT did not converge: {exc}")
            print("  Returning last iterate.")
            success = False
            X_val   = np.array(opti.debug.value(X)).flatten()
            Y_val   = np.array(opti.debug.value(Y)).flatten()
            Phi_val = np.array(opti.debug.value(Phi)).flatten()
            T_val   = np.zeros(N)

        positions = np.column_stack([X_val, Y_val])
        dist_final = float(np.hypot(X_val[-1] - gx, Y_val[-1] - gy))
        print(f"  Final distance to goal: {dist_final:.1f} m  "
              f"({'converged' if success else 'NOT converged'})")

        return {
            'positions': positions,
            'phi_rel':   Phi_val,
            'T_cycles':  T_val,
            'success':   success,
        }
