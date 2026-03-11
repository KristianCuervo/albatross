"""
tacking_lookup.py
=================
Loads the pre-computed tacking diagram NPZ and provides interpolated velocity
lookup for an arbitrary local wind speed V_ref.

Data convention (from refactor/tacking_diagram.py):
  - NPZ stores the *original* half-plane results (theta in [0, π], u_avg ≥ 0).
  - The mirrored half (u → −u) is always derivable and is reconstructed here,
    giving full 360° heading coverage (116 velocity options per V_ref query).
  - u_avg : cycle-averaged crosswind velocity  (+x in wind frame)
  - v_avg : cycle-averaged upwind velocity     (+y = into wind)
  - theta  : optimisation heading in [0, π] — the *desired* direction, NOT the
             realised travel angle.

Wind-frame → global-frame conversion (caller's responsibility):
  vx = v_avg * cos(alpha) - u_avg * sin(alpha)
  vy = v_avg * sin(alpha) + u_avg * cos(alpha)
where alpha = direction the wind blows FROM, CCW from East [rad].
"""

import numpy as np


class TackingLookup:
    """
    Provides interpolated velocity lookups from the tacking diagram NPZ.

    For a query V_ref_local, returns all feasible (u_wind, v_wind, T_cycle)
    pairs — original + u-mirrored — interpolated linearly between the two
    bracketing integer V_ref levels.  Headings whose minimum feasible V_ref
    exceeds V_ref_local are silently omitted.
    """

    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)

        unique_thetas = np.unique(data['theta'])   # 58 headings in [0, π]

        # For each unique theta, store V_ref-sorted arrays of optimisation data.
        self._theta_data: list[dict] = []
        for theta in unique_thetas:
            mask = data['theta'] == theta
            order = np.argsort(data['V_ref'][mask])
            self._theta_data.append({
                'V_ref':   data['V_ref'][mask][order],     # ascending integers
                'u':       data['u_avg'][mask][order],
                'v':       data['v_avg'][mask][order],
                'T':       data['T_cycle'][mask][order],
            })

        self.n_thetas = len(unique_thetas)

    # ------------------------------------------------------------------ #

    def get_velocities(self, V_ref_local: float) -> np.ndarray:
        """
        Return velocity options for the given local wind speed.

        Parameters
        ----------
        V_ref_local : float
            Local wind reference speed [m/s].

        Returns
        -------
        ndarray, shape (N, 3)
            Columns: [u_wind_frame, v_wind_frame, T_cycle].
            Includes original and u-mirrored pairs for every feasible heading.
            Returns empty array (shape (0, 3)) if no heading is feasible.
        """
        results: list[list[float]] = []

        for d in self._theta_data:
            vr    = d['V_ref']
            v_min = vr[0]
            v_max = vr[-1]

            # Skip headings that require more wind than available.
            if V_ref_local < v_min:
                continue

            # Clamp to available upper end (no extrapolation above V_REF_MAX).
            V_eff = min(float(V_ref_local), float(v_max))

            if V_eff >= v_max:
                u_int = float(d['u'][-1])
                v_int = float(d['v'][-1])
                T_int = float(d['T'][-1])
            else:
                idx_hi = int(np.searchsorted(vr, V_eff))
                idx_lo = idx_hi - 1
                w      = (V_eff - vr[idx_lo]) / (vr[idx_hi] - vr[idx_lo])
                u_int  = (1.0 - w) * d['u'][idx_lo] + w * d['u'][idx_hi]
                v_int  = (1.0 - w) * d['v'][idx_lo] + w * d['v'][idx_hi]
                T_int  = (1.0 - w) * d['T'][idx_lo] + w * d['T'][idx_hi]

            results.append([ u_int, v_int, T_int])   # original
            results.append([-u_int, v_int, T_int])   # u-mirrored

        if not results:
            return np.empty((0, 3))

        return np.array(results, dtype=float)   # shape (2 * N_feasible, 3)
