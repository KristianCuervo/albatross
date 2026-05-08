"""
Mesoscale Analysis: Symmetry-Breaking — Straight Upwind/Downwind Trajectories

Forces no net crosswind displacement per plotted segment (sum u = 0):

  Free BC       : periodic cycle + sum(u) = 0.
  Reflective BC : half-cycle (u[-1] = -u[0], h[-1] = h[0]) + sum(u) = 0,
                  so the plotted half-cycle itself has zero crosswind drift.

Solving in three steps to guarantee single-cycle convergence:

  Step 1  Unconstrained reflective BC  (no sum constraint; always single-cycle).
  Step 2  Free BC + sum(u) = 0, warm-started from step 1.
  Step 3  Reflective BC + sum(u) = 0, warm-started from step 2.
          The free-BC result has h_min→peak→h_min shape and antisymmetric u
          (sum≈0, u[-1]≈−u[0]), making it the ideal seed for step 3.

A retry loop with T_cycle_max reduction acts as a safety net: if the middle
half of h dips back to h_min, a multi-cycle was found and T_max is reduced.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from microscale.bird import Albatross
from microscale import Solver

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / 'data'
FIGURES_DIR = ROOT / 'figures' / 'mesoscale'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

H_MIN = 0.1   # must match Solver default h_min
V_REF = 15.0
N_SOL = 128

parser = argparse.ArgumentParser(description='Straight upwind/downwind DS analysis')
for flag in ('--T-max-up-free', '--T-max-dn-free',
             '--T-max-up-refl', '--T-max-dn-refl'):
    parser.add_argument(
        flag, type=float, default=None, metavar='SECONDS',
        help=f'Fix T_cycle_max for the {flag[7:]} case and skip its retry loop. '
             'Omit to use auto-retry (starts at 15 s, reduces until single-cycle).',
    )
args = parser.parse_args()
# None → auto-retry for that case; float → fixed T_max, no retry
T_MAX = {
    'upwind free':   args.T_max_up_free,
    'downwind free': args.T_max_dn_free,
    'upwind refl':   args.T_max_up_refl,
    'downwind refl': args.T_max_dn_refl,
}

bird = Albatross.from_toml(DATA_DIR / 'albatross.toml')


# ── Single-cycle guard ────────────────────────────────────────────────────────

def is_single_cycle(sol, tol=2.0):
    """
    Return True if the solution is a genuine single dynamic-soaring cycle.

    Checks that the middle half of the altitude profile never dips back to
    h_min.  A multi-cycle solution reaches h_min again near the midpoint,
    so np.min(h[N//4 : 3N//4]) ≤ H_MIN + tol catches it cleanly without
    being fooled by the endpoints (which are pinned to h_min by construction).
    """
    h = sol.h
    N = len(h)
    mid_min = float(np.min(h[N // 4 : 3 * N // 4]))
    return mid_min > H_MIN + tol


# ── Solve helpers ─────────────────────────────────────────────────────────────

def _set_warm_start(solver, ics, T_cycle_max):
    """Override all ICs from an ics dict; clamp dt so T_ic ≤ T_cycle_max."""
    dt_ic = min(float(ics['dt']), T_cycle_max / len(ics['h']))
    solver.opti.set_initial(solver.h,  ics['h'])
    solver.opti.set_initial(solver.v,  ics['v'])
    solver.opti.set_initial(solver.w,  ics['w'])
    solver.opti.set_initial(solver.mu, ics['mu'])
    solver.opti.set_initial(solver.cl, ics['cl'])
    solver.opti.set_initial(solver.dt, dt_ic)


def _add_straight_constraint(solver, u_ic):
    """Centre the u IC and add sum(u) = 0."""
    solver.opti.set_initial(solver.u, u_ic - u_ic.mean())
    solver.opti.subject_to(cas.sum1(solver.u) == 0.0)


def solve_unconstrained_reflective(theta, bird, V_ref, N):
    """Plain reflective half-cycle, no sum(u) constraint.  Always single-cycle."""
    return Solver(bird=bird, theta=theta, N=N, V_ref=V_ref, mode='max_vmg',
                  reflective_bc=True).optimise()


def solve_straight_free(theta, bird, V_ref, N, warm_start, T_cycle_max):
    """Free periodic BC + sum(u) = 0, warm-started from warm_start."""
    solver = Solver(bird=bird, theta=theta, N=N, V_ref=V_ref, mode='max_vmg',
                    reflective_bc=False, T_cycle_max=T_cycle_max)
    ics = warm_start.as_ics(N_target=N)
    _set_warm_start(solver, ics, T_cycle_max)
    _add_straight_constraint(solver, ics['u'])
    return solver.optimise()


def solve_straight_reflective(theta, bird, V_ref, N, warm_start, T_cycle_max):
    """Reflective half-cycle + sum(u) = 0, warm-started from warm_start."""
    solver = Solver(bird=bird, theta=theta, N=N, V_ref=V_ref, mode='max_vmg',
                    reflective_bc=True, T_cycle_max=T_cycle_max)
    ics = warm_start.as_ics(N_target=N)
    _set_warm_start(solver, ics, T_cycle_max)
    _add_straight_constraint(solver, ics['u'])
    return solver.optimise()


def solve_with_retry(solver_fn, label, T_max_init, T_reduction=0.70, max_retries=7):
    """
    Retry solver_fn(T_max) with a decreasing T_max budget until is_single_cycle
    passes.  T_max is reduced by T_reduction each failed attempt.

    If T_MAX_OVERRIDE is set (via --T-max), a single solve is run at that value
    with no retry, and the single-cycle check is reported but not enforced.
    """
    override = T_MAX.get(label)
    if override is not None:
        print(f'  [{label}] T_max fixed to {override:.2f} s (no retry)')
        sol = solver_fn(override)
        if not is_single_cycle(sol):
            mid_min = float(np.min(sol.h[len(sol.h) // 4 : 3 * len(sol.h) // 4]))
            print(f'  [{label}] WARNING: multi-cycle detected (h_mid_min={mid_min:.2f} m)')
        return sol

    T_max = T_max_init
    for attempt in range(max_retries):
        sol = solver_fn(T_max)
        if is_single_cycle(sol):
            if attempt > 0:
                print(f'  [{label}] single cycle at T_max={T_max:.2f} s')
            return sol
        mid_min = float(np.min(sol.h[len(sol.h) // 4 : 3 * len(sol.h) // 4]))
        T_max *= T_reduction
        print(f'  [{label}] multi-cycle (h_mid_min={mid_min:.2f} m) → '
              f'retrying T_max={T_max:.2f} s')
    raise RuntimeError(f'[{label}] single-cycle not found after {max_retries} retries')


# ── Run all four cases ────────────────────────────────────────────────────────
# Three-step warm-start chain:
#   Step 1 — unconstrained reflective BC (no sum constraint, always single-cycle)
#   Step 2 — free BC + sum(u)=0, seeded from step 1
#   Step 3 — reflective BC + sum(u)=0, seeded from step 2
#             (free BC result has h_min→peak→h_min shape and antisymmetric u,
#              making it the natural seed for the constrained reflective problem)

print(f'Solving 4 straight-flight cases at V_ref={V_REF} m/s, N={N_SOL} ...\n')

# Step 1
print('Step 1: unconstrained reflective BC ...')
sol_up_seed = solve_unconstrained_reflective(0.0,   bird, V_REF, N_SOL)
sol_dn_seed = solve_unconstrained_reflective(np.pi, bird, V_REF, N_SOL)

# Step 2
print('Step 2: free BC + sum(u)=0 ...')
sol_up_free = solve_with_retry(
    lambda T: solve_straight_free(0.0,   bird, V_REF, N_SOL, sol_up_seed, T),
    'upwind free', T_max_init=15.0,
)
sol_dn_free = solve_with_retry(
    lambda T: solve_straight_free(np.pi, bird, V_REF, N_SOL, sol_dn_seed, T),
    'downwind free', T_max_init=15.0,
)

# Step 3
print('Step 3: reflective BC + sum(u)=0 ...')
sol_up_refl = solve_with_retry(
    lambda T: solve_straight_reflective(0.0,   bird, V_REF, N_SOL, sol_up_free, T),
    'upwind refl', T_max_init=15.0,
)
sol_dn_refl = solve_with_retry(
    lambda T: solve_straight_reflective(np.pi, bird, V_REF, N_SOL, sol_dn_free, T),
    'downwind refl', T_max_init=15.0,
)

# ── Tacking reference solutions ───────────────────────────────────────────────
# Tacking at ±30° from upwind: bird alternates between +30° and −30°.
# Crosswind components cancel exactly, so effective upwind VMG = mean(v) at θ=30°.
# Tacking at ±30° from downwind (θ=150° / 210°): same logic, downwind VMG = mean(v) at θ=150°.

TACK_ANGLE_DEG = 30.0
theta_tack_up = np.radians(TACK_ANGLE_DEG)         # 30° off upwind
theta_tack_dn = np.pi - np.radians(TACK_ANGLE_DEG) # 150° (30° off downwind)

print(f'\nTacking reference: solving at ±{TACK_ANGLE_DEG:.0f}° (free BC, no sum constraint) ...')
sol_tack_up = Solver(bird=bird, theta=theta_tack_up, N=N_SOL, V_ref=V_REF,
                     mode='max_vmg', reflective_bc=False).optimise()
sol_tack_dn = Solver(bird=bird, theta=theta_tack_dn, N=N_SOL, V_ref=V_REF,
                     mode='max_vmg', reflective_bc=False).optimise()

# Effective VMG when tacking: crosswind components cancel, only v survives
vmg_tack_up = float(np.mean(sol_tack_up.v))  # > 0: upwind progress
vmg_tack_dn = float(np.mean(sol_tack_dn.v))  # < 0: downwind progress


# ── Print summaries ───────────────────────────────────────────────────────────

def crosswind_drift(sol):
    return float(np.sum(sol.u) * sol.dt)

print(
    f'[1/4] Upwind   free BC:  '
    f'VMG_v = {np.mean(sol_up_free.v):+.3f} m/s  '
    f'T = {sol_up_free.T_cycle:.2f} s  '
    f'crosswind drift = {crosswind_drift(sol_up_free):+.4f} m'
)
print(
    f'[2/4] Downwind free BC:  '
    f'VMG_v = {np.mean(sol_dn_free.v):+.3f} m/s  '
    f'T = {sol_dn_free.T_cycle:.2f} s  '
    f'crosswind drift = {crosswind_drift(sol_dn_free):+.4f} m'
)
print(
    f'[3/4] Upwind   refl. BC: '
    f'VMG_v = {np.mean(sol_up_refl.v):+.3f} m/s  '
    f'T = {sol_up_refl.T_cycle:.2f} s  '
    f'crosswind drift = {crosswind_drift(sol_up_refl):+.4f} m'
)
print(
    f'[4/4] Downwind refl. BC: '
    f'VMG_v = {np.mean(sol_dn_refl.v):+.3f} m/s  '
    f'T = {sol_dn_refl.T_cycle:.2f} s  '
    f'crosswind drift = {crosswind_drift(sol_dn_refl):+.4f} m'
)

# ── Tacking comparison ────────────────────────────────────────────────────────
vmg_up_free  = float(np.mean(sol_up_free.v))
vmg_dn_free  = float(np.mean(sol_dn_free.v))
vmg_up_refl  = float(np.mean(sol_up_refl.v))
vmg_dn_refl  = float(np.mean(sol_dn_refl.v))

def pct(value, reference):
    """Signed % change: (value - reference) / |reference| * 100."""
    return (value - reference) / abs(reference) * 100.0

sep = '-' * 76
print(f'\n{sep}')
print(f'  TACKING vs STRAIGHT-FLIGHT COMPARISON  (V_ref = {V_REF} m/s)')
print(sep)
print(f'  Tacking reference: ±{TACK_ANGLE_DEG:.0f}° tacks (free BC, unconstrained crosswind)')
print(f'    Upwind  tacking VMG : {vmg_tack_up:+.3f} m/s   (θ = +{TACK_ANGLE_DEG:.0f}°, T = {sol_tack_up.T_cycle:.2f} s)')
print(f'    Downwind tacking VMG: {vmg_tack_dn:+.3f} m/s  (θ = {np.degrees(theta_tack_dn):.0f}°, T = {sol_tack_dn.T_cycle:.2f} s)')
print()
print(f'  {"Case":<30}  {"VMG_v (m/s)":>12}  {"vs tacking":>12}  {"T_cycle (s)":>12}')
print(f'  {"-"*30}  {"-"*12}  {"-"*12}  {"-"*12}')
print(f'  {"Tacking  ±30°  (upwind ref)":<30}  {vmg_tack_up:>+12.3f}  {"—":>12}  {sol_tack_up.T_cycle:>12.2f}')
print(f'  {"Straight upwind — free BC":<30}  {vmg_up_free:>+12.3f}  {pct(vmg_up_free,  vmg_tack_up):>+11.1f}%  {sol_up_free.T_cycle:>12.2f}')
print(f'  {"Straight upwind — refl. BC":<30}  {vmg_up_refl:>+12.3f}  {pct(vmg_up_refl,  vmg_tack_up):>+11.1f}%  {sol_up_refl.T_cycle:>12.2f}')
print()
print(f'  {"Tacking  ±30° (downwind ref)":<30}  {vmg_tack_dn:>+12.3f}  {"—":>12}  {sol_tack_dn.T_cycle:>12.2f}')
print(f'  {"Straight downwind — free BC":<30}  {vmg_dn_free:>+12.3f}  {pct(vmg_dn_free,  vmg_tack_dn):>+11.1f}%  {sol_dn_free.T_cycle:>12.2f}')
print(f'  {"Straight downwind — refl. BC":<30}  {vmg_dn_refl:>+12.3f}  {pct(vmg_dn_refl,  vmg_tack_dn):>+11.1f}%  {sol_dn_refl.T_cycle:>12.2f}')
print(sep)
print('  NOTE: downwind "straight" solutions with suspiciously high |VMG| may')
print('        be degenerate (optimizer exploits repeated steep dives with no')
print('        net crosswind constraint but uncapped cycle count in v-direction).')
print(sep)

# ── 3-D trajectory plotting ───────────────────────────────────────────────────

def flat_arrow_3d(ax, tip, direction, width_dir, size, color='black', alpha=0.9):
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    w = np.asarray(width_dir,  float); w /= np.linalg.norm(w)
    t = np.asarray(tip, float)
    bl = t - size * d + size * 0.45 * w
    br = t - size * d - size * 0.45 * w
    ax.add_collection3d(Poly3DCollection([[t, bl, br]],
                        facecolor=color, edgecolor=color, alpha=alpha))


def add_curtain(ax, sol):
    x, y, h = sol.x, sol.y, sol.h
    verts = []
    for i in range(len(x) - 1):
        verts.append([(x[i], y[i], h[i]), (x[i+1], y[i+1], h[i+1]),
                      (x[i+1], y[i+1], 0.0), (x[i], y[i], 0.0)])
    ax.add_collection3d(Poly3DCollection(
        verts, facecolor='gray', edgecolor='none', alpha=0.25, zorder=4))


def add_bird_crosses(ax, sol, span, n_birds=5):
    arm   = span * 0.07
    front = arm * 0.4
    x, y, h = sol.x, sol.y, sol.h

    def bird_frame(i):
        fwd = np.array([sol.u[i], sol.v[i], -sol.w[i]])
        fwd /= np.linalg.norm(fwd); fwd /= 2
        up = np.array([0., 0., 1.])
        if abs(np.dot(fwd, up)) > 0.95: up = np.array([0., 1., 0.])
        right0 = np.cross(fwd, up);     right0 /= np.linalg.norm(right0)
        tilt0  = np.cross(right0, fwd); tilt0  /= np.linalg.norm(tilt0)
        wing = np.cos(sol.mu[i]) * right0 + np.sin(sol.mu[i]) * tilt0
        return np.array([x[i], y[i], h[i]]), fwd, wing

    idxs = np.linspace(0, len(x) - 1, n_birds, dtype=int)
    ckw  = dict(color='k', lw=1.8, solid_capstyle='round', zorder=10)
    for i in idxs:
        pos, fwd, wing = bird_frame(i)
        ax.plot([pos[0]-arm*fwd[0],  pos[0]+front*fwd[0]],
                [pos[1]-arm*fwd[1],  pos[1]+front*fwd[1]],
                [pos[2]-arm*fwd[2],  pos[2]+front*fwd[2]], **ckw)
        ax.plot([pos[0]-arm*wing[0], pos[0]+arm*wing[0]],
                [pos[1]-arm*wing[1], pos[1]+arm*wing[1]],
                [pos[2]-arm*wing[2], pos[2]+arm*wing[2]], **ckw)
        ax.scatter(*pos, s=12, color='k', zorder=11)


def add_flight_arrows(ax, sol, span, n_arrows=6, two_color=True):
    aw = span * 0.028
    x, y, h = sol.x, sol.y, sol.h
    N_pts = len(x); half = N_pts // 2
    idxs = np.linspace(0, N_pts - 1, n_arrows + 2, dtype=int)[1:-1]
    for i in idxs:
        color = ('C0' if i < half else 'C1') if two_color else 'C0'
        fwd = np.array([sol.u[i], sol.v[i], -sol.w[i]], dtype=float)
        if np.linalg.norm(fwd) < 1e-10: continue
        fwd /= np.linalg.norm(fwd)
        up = np.array([0., 0., 1.])
        if abs(np.dot(fwd, up)) > 0.95: up = np.array([0., 1., 0.])
        width = np.cross(fwd, up); width /= np.linalg.norm(width)
        flat_arrow_3d(ax, np.array([x[i], y[i], h[i]]),
                      tuple(fwd), tuple(width), aw, color=color, alpha=0.95)


def add_wind_shear_profile(ax, sol, span, xl, yl):
    h_max = float(sol.h.max()); V_ref = float(sol.V_ref)
    n = 200
    h_vals = np.linspace(0.0, h_max, n)
    V_w    = np.where(h_vals > 0.0, V_ref * (h_vals / 10.0) ** 0.143, 0.0)
    V_top  = max(V_ref * (h_max / 10.0) ** 0.143 if h_max > 1e-3 else V_ref, 1e-6)
    pw     = span * 0.18
    x_wall = xl[0]; y_sp = yl[0]
    y_curve = y_sp - (V_w / V_top) * pw
    aw = pw * 0.05
    for h_a in np.linspace(h_max / 8, h_max, 7):
        V_a   = V_ref * (h_a / 10.0) ** 0.143
        y_tip = y_sp - (V_a / V_top) * pw
        if y_sp - y_tip < aw * 1.5: continue
        ax.plot([x_wall]*2, [y_sp, y_tip + aw], [h_a]*2,
                color='red', lw=0.7, alpha=0.6, zorder=6)
        flat_arrow_3d(ax, (x_wall, y_tip, h_a), (0, -1, 0), (0, 0, 1), aw,
                      color='red', alpha=0.75)
    ax.plot([x_wall]*n, y_curve, h_vals, color='red', lw=0.7, zorder=7)
    ax.plot([x_wall]*2, [y_sp]*2, [0.0, h_max], color='red', lw=0.9, alpha=0.5)


def add_dimension_arrows(ax, sol, span, xl, yl):
    x, y, h = sol.x, sol.y, sol.h
    aw = span * 0.022
    dlkw = dict(color='black', lw=0.9, alpha=0.8)
    tlkw = dict(fontsize=11, color='black', alpha=0.8)
    x_lo, x_hi = float(x.min()), float(x.max())
    y_lo, y_hi = float(y.min()), float(y.max())
    h_hi = float(h.max())
    mg = span * 0.05
    yp = yl[0] - mg
    ax.plot([x_lo, x_hi], [yp]*2, [0]*2, **dlkw)
    flat_arrow_3d(ax, (x_lo, yp, 0), (-1, 0, 0), (0, 1, 0), aw)
    flat_arrow_3d(ax, (x_hi, yp, 0), ( 1, 0, 0), (0, 1, 0), aw)
    ax.text((x_lo+x_hi)/2, yp - 2*mg, aw,
            f'{x_hi-x_lo:.1f} m [x]', ha='center', va='bottom', **tlkw)
    xp = xl[1] + mg
    ax.plot([xp]*2, [y_lo, y_hi], [0]*2, **dlkw)
    flat_arrow_3d(ax, (xp, y_lo, 0), (0, -1, 0), (1, 0, 0), aw)
    flat_arrow_3d(ax, (xp, y_hi, 0), (0,  1, 0), (1, 0, 0), aw)
    ax.text(xp+mg, (y_lo+y_hi)/2, aw,
            f'{y_hi-y_lo:.1f} m [y]', ha='left', va='bottom', **tlkw)
    ax.plot([xp]*2, [yl[1]]*2, [0, h_hi], **dlkw)
    flat_arrow_3d(ax, (xp, yl[1], 0),    (0, 0, -1), (1, 0, 0), aw)
    flat_arrow_3d(ax, (xp, yl[1], h_hi), (0, 0,  1), (1, 0, 0), aw)
    ax.text(xp+mg*0.4, yl[1], h_hi/2,
            f'{h_hi:.1f} m [h]', ha='left', va='center', **tlkw)


def plot_trajectory_3d(sol, two_color=False, title=''):
    x, y, h = sol.x, sol.y, sol.h
    N_pts = len(x); half = N_pts // 2
    fig = plt.figure(figsize=(13, 12))
    ax  = fig.add_subplot(111, projection='3d')
    span = max(x.max()-x.min(), y.max()-y.min(), 1.0)
    mid_x = (x.max()+x.min())/2; mid_y = (y.max()+y.min())/2
    xl = (mid_x-span/2, mid_x+span/2); yl = (mid_y-span/2, mid_y+span/2)
    zl = (0.0, span)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
    ax.set_box_aspect([1, 1, 1])
    add_curtain(ax, sol)
    if two_color:
        ax.plot(x[:half+1], y[:half+1], h[:half+1], lw=2, color='C0', zorder=5)
        ax.plot(x[half:],   y[half:],   h[half:],   lw=2, color='C1', zorder=5)
    else:
        ax.plot(x, y, h, lw=2, color='C0', zorder=5)
    add_flight_arrows(ax, sol, span, two_color=two_color)
    pkw = dict(lw=0.8, color='black', alpha=0.6)
    ax.plot(x, y, np.zeros_like(h), **pkw)
    ax.plot(x, yl[1]*np.ones_like(y), h, **pkw)
    ax.plot(xl[0]*np.ones_like(x), y, h, **pkw)
    bkw = dict(lw=0.6, color='black', alpha=0.5)
    ax.plot([xl[0],xl[1],xl[1],xl[0],xl[0]], [yl[1]]*5, [0,0,zl[1],zl[1],0], **bkw)
    ax.plot([xl[0]]*5, [yl[0],yl[1],yl[1],yl[0],yl[0]], [0,0,zl[1],zl[1],0], **bkw)
    add_bird_crosses(ax, sol, span)
    add_wind_shear_profile(ax, sol, span, xl, yl)
    add_dimension_arrows(ax, sol, span, xl, yl)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor('lightgray')
    if title:
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return fig, ax


# ── Save figures ──────────────────────────────────────────────────────────────

trajectory_cases = [
    (sol_up_free,  False, 'straight_upwind_free.png',        'Upwind free BC  (sum u = 0 constrained)'),
    (sol_dn_free,  False, 'straight_downwind_free.png',      'Downwind free BC  (sum u = 0 constrained)'),
    (sol_up_refl,  False, 'straight_upwind_reflective.png',  'Upwind reflective BC  (half-cycle)'),
    (sol_dn_refl,  False, 'straight_downwind_reflective.png','Downwind reflective BC  (half-cycle)'),
]

for sol, tc, fname, title in trajectory_cases:
    fig, ax = plot_trajectory_3d(sol, two_color=tc, title=title)
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)
    print(f'Saved {fname}')

print(f'\nAll figures saved to {FIGURES_DIR.resolve()}')
