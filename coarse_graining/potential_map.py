"""
potential_map.py
================
Computes the "potential" at every grid cell: the maximum achievable rate of
progress toward a fixed goal point, given the local wind (V_ref, alpha) and
the full set of velocity options from the tacking diagram.

Also provides a greedy path planner: at each position take the velocity that
maximises instantaneous progress toward the goal, advance one soaring cycle,
repeat.

No external dependencies beyond NumPy.
"""

import numpy as np
from tacking_lookup import TackingLookup


# --------------------------------------------------------------------------- #
# Potential map                                                                #
# --------------------------------------------------------------------------- #

def compute_potential(
    wind_field:  dict,
    lookup:      TackingLookup,
    goal:        tuple,
    grid_size:   int   = 100,
    domain:      float = 1000.0,
) -> dict:
    """
    For every grid cell compute the best available velocity toward the goal.

    Wind-frame → global-frame rotation:
        vx = v_wind * cos(alpha) - u_wind * sin(alpha)
        vy = v_wind * sin(alpha) + u_wind * cos(alpha)
    where alpha is the direction the wind blows FROM (CCW from East).

    Parameters
    ----------
    wind_field : dict
        Output of wind_field.make_rotational_wind_field().
    lookup : TackingLookup
        Loaded tacking diagram lookup object.
    goal : (float, float)
        (x, y) position of the target [m].
    grid_size, domain : int, float
        Must match the wind_field dimensions.

    Returns
    -------
    dict with keys:
      'potential' : (grid_size, grid_size)  max progress toward goal [m/s]
      'best_vx'   : (grid_size, grid_size)  optimal global-frame x velocity
      'best_vy'   : (grid_size, grid_size)  optimal global-frame y velocity
      'best_T'    : (grid_size, grid_size)  cycle period for optimal heading [s]
    """
    xs       = wind_field['x_centers']
    ys       = wind_field['y_centers']
    V_ref_g  = wind_field['V_ref']
    alpha_g  = wind_field['alpha']

    goal_xy  = np.array(goal, dtype=float)

    potential = np.full((grid_size, grid_size), np.nan)
    best_vx   = np.zeros((grid_size, grid_size))
    best_vy   = np.zeros((grid_size, grid_size))
    best_T    = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            pos     = np.array([xs[i], ys[j]])
            to_goal = goal_xy - pos
            dist    = np.hypot(to_goal[0], to_goal[1])

            if dist < 1.0:
                potential[i, j] = 0.0
                continue

            goal_dir = to_goal / dist
            V_ref_loc = float(V_ref_g[i, j])
            alpha_loc  = float(alpha_g[i, j])

            vels = lookup.get_velocities(V_ref_loc)   # shape (N, 3)
            if len(vels) == 0:
                continue

            u_wind = vels[:, 0]
            v_wind = vels[:, 1]
            T_cyc  = vels[:, 2]

            # Rotate wind-frame velocities to global frame
            ca, sa = np.cos(alpha_loc), np.sin(alpha_loc)
            vx = v_wind * ca - u_wind * sa
            vy = v_wind * sa + u_wind * ca

            # Project onto direction toward goal
            progress = vx * goal_dir[0] + vy * goal_dir[1]

            best_k = int(np.argmax(progress))
            potential[i, j] = progress[best_k]
            best_vx[i, j]   = vx[best_k]
            best_vy[i, j]   = vy[best_k]
            best_T[i, j]    = T_cyc[best_k]

    return {
        'potential': potential,
        'best_vx':   best_vx,
        'best_vy':   best_vy,
        'best_T':    best_T,
    }


# --------------------------------------------------------------------------- #
# Bilinear interpolation helper (no scipy)                                    #
# --------------------------------------------------------------------------- #

def _bilinear(grid_arr: np.ndarray, xs: np.ndarray, ys: np.ndarray,
              x: float, y: float) -> float:
    """Bilinear interpolation on a regular grid indexed as [i_x, i_y]."""
    x = float(np.clip(x, xs[0], xs[-1]))
    y = float(np.clip(y, ys[0], ys[-1]))

    i1 = int(np.searchsorted(xs, x)) - 1
    j1 = int(np.searchsorted(ys, y)) - 1
    i1 = int(np.clip(i1, 0, len(xs) - 2))
    j1 = int(np.clip(j1, 0, len(ys) - 2))
    i2, j2 = i1 + 1, j1 + 1

    wx = (x - xs[i1]) / (xs[i2] - xs[i1])
    wy = (y - ys[j1]) / (ys[j2] - ys[j1])

    return (
        (1 - wx) * (1 - wy) * grid_arr[i1, j1]
        + wx       * (1 - wy) * grid_arr[i2, j1]
        + (1 - wx) * wy       * grid_arr[i1, j2]
        + wx       * wy       * grid_arr[i2, j2]
    )


# --------------------------------------------------------------------------- #
# Greedy path planner                                                          #
# --------------------------------------------------------------------------- #

def plan_path(
    wind_field:       dict,
    potential_result: dict,
    goal:             tuple,
    start:            tuple,
    goal_radius:      float = 60.0,   # [m] stop when within this distance
    max_steps:        int   = 300,
) -> np.ndarray:
    """
    Greedy cycle-by-cycle path from *start* to *goal*.

    At each position the locally optimal velocity vector (from the potential
    map) is interpolated bilinearly from the grid and the bird advances by
    that velocity × T_cycle (one soaring cycle).  Continues until within
    *goal_radius* of the goal or *max_steps* cycles.

    Returns
    -------
    path : ndarray, shape (N_steps, 2)
        Sequence of (x, y) positions [m].
    """
    xs = wind_field['x_centers']
    ys = wind_field['y_centers']

    vx_grid = potential_result['best_vx']
    vy_grid = potential_result['best_vy']
    T_grid  = potential_result['best_T']

    pos  = np.array(start, dtype=float)
    goal = np.array(goal,  dtype=float)
    path = [pos.copy()]

    domain = float(xs[-1] + (xs[1] - xs[0]) / 2.0)   # reconstruct from grid

    for _ in range(max_steps):
        dist = float(np.hypot(*(pos - goal)))
        if dist < goal_radius:
            path.append(goal.copy())
            break

        # Clamp lookup position to grid interior (handles slight overshoot)
        pos_q = np.clip(pos, [xs[0], ys[0]], [xs[-1], ys[-1]])
        vx = _bilinear(vx_grid, xs, ys, pos_q[0], pos_q[1])
        vy = _bilinear(vy_grid, xs, ys, pos_q[0], pos_q[1])
        T  = _bilinear(T_grid,  xs, ys, pos_q[0], pos_q[1])

        # Check whether this step would cross the goal; if so, snap to it
        step = np.array([vx, vy]) * T
        new_pos = pos + step
        if np.hypot(*(new_pos - goal)) < goal_radius:
            path.append(goal.copy())
            break

        pos = new_pos

        # Stop if the bird exits the domain
        if not (0 <= pos[0] <= domain and 0 <= pos[1] <= domain):
            break

        path.append(pos.copy())

    return np.array(path)
