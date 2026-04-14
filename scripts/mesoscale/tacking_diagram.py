"""
Mesoscale Analysis: Tacking Diagram & Trajectory Comparison

Produces two groups of results:

1. Polar sweeps (RECOMPUTE=False by default — loads precomputed data)
   - Minimum wind speed V_ref_min(θ) polar curve
   - VMG tacking diagram (max ground-speed vs heading at fixed V_ref levels)
   - Objective polar and cycle-time polar
   - Convex velocity hull

2. Trajectory comparisons at V_ref=15 m/s
   - Upwind / downwind free-BC cycles
   - Upwind / downwind reflective-BC (half-cycle) trajectories
   - Two-cycle (mirrored) reconstructions of both headings

RECOMPUTE=True runs the full polar sweeps (~1-4 h).
Figures are saved to figures/mesoscale/.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from albatross import Albatross
from albatross.microscale import Ensemble, Solver, sweep_min_vref, sweep_max_vmg
from albatross.macroscale import VelocityHull

RECOMPUTE   = False
N_PROCS     = 4
ROOT        = Path(__file__).parent.parent.parent
DATA_DIR    = ROOT / 'data'
FIGURES_DIR = ROOT / 'figures' / 'mesoscale'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Bird & sweep parameters ───────────────────────────────────────────────────
bird   = Albatross.from_toml(DATA_DIR / 'albatross.toml')
thetas = np.linspace(0, 2 * np.pi, 72, endpoint=False)
V_refs = [10.0, 12.0, 15.0, 18.0, 20.0, 25.0]
print(bird)

# ── Load or recompute polar sweeps ────────────────────────────────────────────
TACK_NPZ = DATA_DIR / 'mesoscale' / 'tacking_diagram.npz'
HULL_NPZ = DATA_DIR / 'mesoscale' / 'velocity_hulls.npz'

if RECOMPUTE:
    print('Sweep 1/2: min_vref polar curve ...')
    ens_minvref = sweep_min_vref(
        bird=bird, thetas=thetas, N=40,
        add_progress_constraint=True, n_procs=N_PROCS,
    )
    print(f'  {len(ens_minvref)} converged solutions')

    print('\nSweep 2/2: max_vmg tacking diagram ...')
    ens_vmg = sweep_max_vmg(
        bird=bird, thetas=thetas, V_refs=V_refs, N=40, n_procs=N_PROCS,
    )
    print(f'  {len(ens_vmg)} converged solutions')

    ens_combined = Ensemble()
    for c in ens_minvref: ens_combined.add_container(c)
    for c in ens_vmg:     ens_combined.add_container(c)
    ens_combined.save(TACK_NPZ)

    print('\nBuilding velocity hull ...')
    hull = ens_vmg.to_hull()
    hull.save(HULL_NPZ)

else:
    print('Loading precomputed data ...')
    ens_combined = Ensemble.load(TACK_NPZ)
    hull         = VelocityHull.from_npz(HULL_NPZ)

    ens_minvref = Ensemble()
    ens_vmg     = Ensemble()
    for c in ens_combined:
        if c.mode == 'min_vref':
            ens_minvref.add_container(c)
        else:
            ens_vmg.add_container(c)

print(f'min_vref containers : {len(ens_minvref)}')
print(f'max_vmg  containers : {len(ens_vmg)}')
print(f'V_ref levels in hull: {hull.v_ref_levels}')

# ── Polar plots ───────────────────────────────────────────────────────────────

# Min-vref polar curve
if len(ens_minvref) > 0:
    cs = ens_minvref.sorted_by_theta()
    th = np.array([c.theta for c in cs])
    vr = np.array([c.V_ref for c in cs])
    th = np.append(th, th[0])
    vr = np.append(vr, vr[0])

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))
    ax.plot(th, vr, lw=2)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Minimum wind speed $V_{ref,min}(\\theta)$ [m/s]\n'
                 'Global minimum ≈ 8.6 m/s at θ=0 (downwind)', pad=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'vref_min_polar.png', dpi=150)
    plt.close(fig)
    print('Saved vref_min_polar.png')

    min_idx = np.argmin(vr[:-1])
    print(f'Global minimum: {vr[min_idx]:.3f} m/s at θ={np.degrees(th[min_idx]):.1f}°')

# VMG tacking diagram
if len(ens_vmg) > 0:
    fig, ax = ens_vmg.plot_tacking_diagram()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'tacking_diagram.png', dpi=150)
    plt.close(fig)
    print('Saved tacking_diagram.png')

# Objective polar
fig, ax = ens_vmg.plot_obj_polar()
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'objective_polar.png', dpi=150)
plt.close(fig)
print('Saved objective_polar.png')

# Cycle-time polar
if len(ens_vmg) > 0:
    fig, ax = ens_vmg.plot_cycle_times()
    ax.set_title('Cycle time $T_{cycle}$ [s] vs heading (coloured by $V_{ref}$)', pad=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'cycle_times.png', dpi=150)
    plt.close(fig)
    print('Saved cycle_times.png')

# Velocity hull
fig, ax = hull.plot()
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'velocity_hull.png', dpi=150)
plt.close(fig)
print('Saved velocity_hull.png')

# ── 3-D trajectory helpers ────────────────────────────────────────────────────

def flat_arrow_3d(ax, tip, direction, width_dir, size, color='black', alpha=0.9):
    d = np.asarray(direction, float);  d /= np.linalg.norm(d)
    w = np.asarray(width_dir,  float);  w /= np.linalg.norm(w)
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
    N = len(x); half = N // 2
    idxs = np.linspace(0, N - 1, n_arrows + 2, dtype=int)[1:-1]
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
        flat_arrow_3d(ax, (x_wall, y_tip, h_a), (0,-1,0), (0,0,1), aw,
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
    flat_arrow_3d(ax, (x_lo, yp, 0), (-1,0,0), (0,1,0), aw)
    flat_arrow_3d(ax, (x_hi, yp, 0), ( 1,0,0), (0,1,0), aw)
    ax.text((x_lo+x_hi)/2, yp - 2*mg, aw,
            f'{x_hi-x_lo:.1f} m [x]', ha='center', va='bottom', **tlkw)
    xp = xl[1] + mg
    ax.plot([xp]*2, [y_lo, y_hi], [0]*2, **dlkw)
    flat_arrow_3d(ax, (xp, y_lo, 0), (0,-1,0), (1,0,0), aw)
    flat_arrow_3d(ax, (xp, y_hi, 0), (0, 1,0), (1,0,0), aw)
    ax.text(xp+mg, (y_lo+y_hi)/2, aw,
            f'{y_hi-y_lo:.1f} m [y]', ha='left', va='bottom', **tlkw)
    ax.plot([xp]*2, [yl[1]]*2, [0, h_hi], **dlkw)
    flat_arrow_3d(ax, (xp, yl[1], 0),    (0,0,-1), (1,0,0), aw)
    flat_arrow_3d(ax, (xp, yl[1], h_hi), (0,0, 1), (1,0,0), aw)
    ax.text(xp+mg*0.4, yl[1], h_hi/2,
            f'{h_hi:.1f} m [h]', ha='left', va='center', **tlkw)


def plot_trajectory_3d(sol, two_color=False, title=''):
    x, y, h = sol.x, sol.y, sol.h
    N = len(x); half = N // 2
    fig = plt.figure(figsize=(13, 12))
    ax  = fig.add_subplot(111, projection='3d')
    span = max(x.max()-x.min(), y.max()-y.min())
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
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def two_cycle(s):
    """Reconstruct a full two-cycle trajectory from a reflective half-cycle."""
    u2 = np.concatenate([s.u, -s.u]); v2 = np.concatenate([s.v,  s.v])
    w2 = np.concatenate([s.w,  s.w]); h2 = np.concatenate([s.h,  s.h])
    return types.SimpleNamespace(
        u=u2, v=v2, w=w2, h=h2,
        x=np.cumsum(u2) * s.dt, y=np.cumsum(v2) * s.dt,
        mu=np.concatenate([s.mu, -s.mu]),
        V_ref=s.V_ref, T_cycle=2*s.T_cycle, theta=s.theta,
    )


# ── Trajectory comparison at V_ref = 15 m/s ──────────────────────────────────
V_REF = 15.0
N_SOL = 128
print(f'\nSolving trajectory comparisons at V_ref={V_REF} m/s, N={N_SOL} ...')

sol_up   = Solver(bird=bird, theta=0.0,   N=N_SOL, V_ref=V_REF, mode='max_vmg').optimise()
sol_dn   = Solver(bird=bird, theta=np.pi, N=N_SOL, V_ref=V_REF, mode='max_vmg',
                  ics=sol_up.as_ics()).optimise()
sol_r_up = Solver(bird=bird, theta=0.0,   N=N_SOL, V_ref=V_REF, mode='max_vmg',
                  reflective_bc=True).optimise()
sol_r_dn = Solver(bird=bird, theta=np.pi, N=N_SOL, V_ref=V_REF, mode='max_vmg',
                  reflective_bc=True).optimise()

full_up = two_cycle(sol_r_up)
full_dn = two_cycle(sol_r_dn)

print(f'  up   free: VMG={np.mean(sol_up.v):+.3f} m/s  T={sol_up.T_cycle:.2f} s')
print(f'  dn   free: VMG={np.mean(sol_dn.v):+.3f} m/s  T={sol_dn.T_cycle:.2f} s')
print(f'  up   refl: VMG={np.mean(sol_r_up.v):+.3f} m/s  T={sol_r_up.T_cycle:.2f} s')
print(f'  dn   refl: VMG={np.mean(sol_r_dn.v):+.3f} m/s  T={sol_r_dn.T_cycle:.2f} s')

trajectory_cases = [
    (sol_up,   False, 'trajectory_upwind.png',             'Upwind free cycle'),
    (sol_dn,   False, 'trajectory_downwind.png',           'Downwind free cycle'),
    (sol_r_up, False, 'trajectory_upwind_reflective.png',  'Upwind reflective (half-cycle)'),
    (sol_r_dn, False, 'trajectory_downwind_reflective.png','Downwind reflective (half-cycle)'),
    (full_up,  True,  'trajectory_upwind_two_cycle.png',   'Upwind two-cycle'),
    (full_dn,  True,  'trajectory_downwind_two_cycle.png', 'Downwind two-cycle'),
]

for sol, tc, fname, title in trajectory_cases:
    fig, ax = plot_trajectory_3d(sol, two_color=tc, title=title)
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)
    print(f'Saved {fname}')

print(f'\nAll figures saved to {FIGURES_DIR.resolve()}')
