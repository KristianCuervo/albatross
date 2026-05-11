"""
Minimum Turn: minimum wind speed for a half-cycle with u(0) = -u(T_cycle).

Finds the minimum V_ref to sustain a periodic half-cycle where the crosswind
velocity component reverses sign at the boundary:

    u[N-1] = -u[0]   (crosswind reflects — tacking half-cycle)
    v[N-1] =  v[0]   (upwind speed preserved)
    h[N-1] =  h[0]   (altitude preserved)
    w[N-1] =  w[0]   (vertical velocity preserved)

Tiling two such half-cycles (mirrored in u) gives a full periodic tacking orbit.

RECOMPUTE = False (default): load precomputed result from data/minimum_turn_result.npz
RECOMPUTE = True : run the optimisation live (~30-60 s on a modern laptop)

Output figures are saved to figures/.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from microscale.bird import Albatross
from microscale import Solver, Container

RECOMPUTE   = True
_HERE       = Path(__file__).parent
NPZ_PATH    = _HERE / 'data' / 'minimum_turn_result.npz'
FIGURES_DIR = _HERE / 'figures'

# ── Bird parameters ───────────────────────────────────────────────────────────
bird = Albatross.from_toml(_HERE / 'albatross.toml')
print(bird)

# ── Solve or load ─────────────────────────────────────────────────────────────
if RECOMPUTE or not NPZ_PATH.exists():
    print('Running min_vref optimisation with reflective u BC ...')
    sol = Solver(
        bird    = bird,
        theta   = 0.0,
        N       = 64,
        mode    = 'min_vref',
        add_progress_constraint = False,
        h_min        = 0.5,
        T_cycle_min  = 5.0,
        T_cycle_max  = 15.0,
        reflective_bc = True,
    ).optimise()
    print(f'V_ref = {sol.V_ref:.3f} m/s')

    NPZ_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        NPZ_PATH,
        u=sol.u, v=sol.v, w=sol.w, h=sol.h, cl=sol.cl, mu=sol.mu,
        dt=sol.dt, T_cycle=sol.T_cycle, V_ref=sol.V_ref, N=sol.N,
    )
    print(f'Saved -> {NPZ_PATH}')
else:
    print(f'Loading precomputed result from {NPZ_PATH}')
    d = np.load(NPZ_PATH)
    sol = Container.from_solution_dict({
        'u': d['u'], 'v': d['v'], 'w': d['w'], 'h': d['h'],
        'cl': d['cl'], 'mu': d['mu'],
        'dt': float(d['dt']), 'T_cycle': float(d['T_cycle']),
        'V_ref': float(d['V_ref']), 'N': int(d['N']),
        'theta': 0.0, 'obj': float(np.mean(d['v'])),
    })

print(f'\nResult: V_ref = {sol.V_ref:.3f} m/s  |  T_cycle = {sol.T_cycle:.2f} s')
print(f'Boundary check: u[0] = {sol.u[0]:.3f}  u[-1] = {sol.u[-1]:.3f}  (should be negated)')

x, y, h = sol.x, sol.y, sol.h
print(f'\nTrajectory extents:')
print(f'  x : {x.min():+.2f} -> {x.max():+.2f}  (range {x.max()-x.min():.2f} m)')
print(f'  y : {y.min():+.2f} -> {y.max():+.2f}  (range {y.max()-y.min():.2f} m)')
print(f'  h : {h.min():+.2f} -> {h.max():+.2f}  (range {h.max()-h.min():.2f} m)')

# ── Time-series plots ─────────────────────────────────────────────────────────
t_norm = np.arange(sol.N) * sol.dt / sol.T_cycle

V_wy = sol.V_ref * (sol.h / 10.0) ** 0.143
V_a  = np.sqrt(sol.u**2 + (sol.v + V_wy)**2 + sol.w**2)
V_K  = np.sqrt(sol.u**2 + sol.v**2 + sol.w**2)

xlabel = '$t \\,/\\, T_{\\mathrm{half}}$'

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_norm, sol.h, 'C0')
ax.set_xlabel(xlabel)
ax.set_ylabel('$h$  [m]')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'min_turn_altitude.png', dpi=150)
plt.close(fig)
print('Saved min_turn_altitude.png')

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_norm, V_a, 'C1',   label='airspeed $V_a$')
ax.plot(t_norm, V_K, 'C2--', label='inertial speed $V_K$')
ax.set_xlabel(xlabel)
ax.set_ylabel('speed  [m/s]')
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'min_turn_speeds.png', dpi=150)
plt.close(fig)
print('Saved min_turn_speeds.png')

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_norm, sol.u, 'C3')
ax.axhline(0, color='k', lw=0.7, ls='--')
ax.set_xlabel(xlabel)
ax.set_ylabel('$u$  [m/s]')
ax.set_title(f'u[0]={sol.u[0]:.2f}  →  u[-1]={sol.u[-1]:.2f} m/s')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'min_turn_u.png', dpi=150)
plt.close(fig)
print('Saved min_turn_u.png')

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_norm, sol.cl, 'C4')
ax.set_xlabel(xlabel)
ax.set_ylabel('$C_L$')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'min_turn_lift_coefficient.png', dpi=150)
plt.close(fig)
print('Saved min_turn_lift_coefficient.png')

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_norm, np.degrees(sol.mu), 'C5')
ax.set_xlabel(xlabel)
ax.set_ylabel('$\\mu$  [°]')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'min_turn_bank_angle.png', dpi=150)
plt.close(fig)
print('Saved min_turn_bank_angle.png')

# ── 3-D trajectory ────────────────────────────────────────────────────────────
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

N_BIRDS = 6


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
        verts.append([
            (x[i],   y[i],   h[i]),
            (x[i+1], y[i+1], h[i+1]),
            (x[i+1], y[i+1], 0.0),
            (x[i],   y[i],   0.0),
        ])
    ax.add_collection3d(Poly3DCollection(
        verts, facecolor='gray', edgecolor='none', alpha=0.25, zorder=4))


def add_bird_crosses(ax, sol, span, n_birds=5):
    arm   = span * 0.07
    front = arm * 0.4
    x, y, h = sol.x, sol.y, sol.h

    def bird_frame(i):
        fwd = np.array([sol.u[i], sol.v[i], -sol.w[i]])
        fwd /= np.linalg.norm(fwd)
        fwd /= 2
        up = np.array([0., 0., 1.])
        if abs(np.dot(fwd, up)) > 0.95:
            up = np.array([0., 1., 0.])
        right0 = np.cross(fwd, up);    right0 /= np.linalg.norm(right0)
        tilt0  = np.cross(right0, fwd); tilt0  /= np.linalg.norm(tilt0)
        wing = np.cos(sol.mu[i]) * right0 + np.sin(sol.mu[i]) * tilt0
        pos  = np.array([x[i], y[i], h[i]])
        return pos, fwd, wing

    idxs = np.linspace(0, len(x) - 1, n_birds, dtype=int)
    ckw  = dict(color='k', lw=1.8, solid_capstyle='round', zorder=10)
    for i in idxs:
        pos, fwd, wing = bird_frame(i)
        ax.plot([pos[0]-arm*fwd[0],   pos[0]+front*fwd[0]],
                [pos[1]-arm*fwd[1],   pos[1]+front*fwd[1]],
                [pos[2]-arm*fwd[2],   pos[2]+front*fwd[2]], **ckw)
        ax.plot([pos[0]-arm*wing[0],  pos[0]+arm*wing[0]],
                [pos[1]-arm*wing[1],  pos[1]+arm*wing[1]],
                [pos[2]-arm*wing[2],  pos[2]+arm*wing[2]], **ckw)
        ax.scatter(*pos, s=12, color='k', zorder=11)


def add_flight_arrows(ax, sol, span, n_arrows=6):
    aw = span * 0.028
    x, y, h = sol.x, sol.y, sol.h
    N = len(x)
    idxs = np.linspace(0, N - 1, n_arrows + 2, dtype=int)[1:-1]
    for i in idxs:
        fwd = np.array([sol.u[i], sol.v[i], -sol.w[i]], dtype=float)
        if np.linalg.norm(fwd) < 1e-10:
            continue
        fwd /= np.linalg.norm(fwd)
        up = np.array([0., 0., 1.])
        if abs(np.dot(fwd, up)) > 0.95:
            up = np.array([0., 1., 0.])
        width = np.cross(fwd, up); width /= np.linalg.norm(width)
        tip = np.array([x[i], y[i], h[i]])
        flat_arrow_3d(ax, tip, tuple(fwd), tuple(width), aw, color='C0', alpha=0.95)


def add_wind_shear_profile(ax, sol, span, xl, yl):
    h_max = float(sol.h.max())
    V_ref = float(sol.V_ref)

    n_profile = 200
    h_vals = np.linspace(0.0, h_max, n_profile)
    V_w = np.where(h_vals > 0.0, V_ref * (h_vals / 10.0) ** 0.143, 0.0)

    V_top = V_ref * (h_max / 10.0) ** 0.143 if h_max > 1e-3 else V_ref
    V_top = max(V_top, 1e-6)

    profile_w = span * 0.18
    x_wall  = xl[0]
    y_spine = yl[0]
    y_curve = y_spine - (V_w / V_top) * profile_w

    aw = profile_w * 0.05
    n_arrows = 7
    h_arrows = np.linspace(h_max / (n_arrows + 1), h_max, n_arrows)
    for h_a in h_arrows:
        V_a   = V_ref * (h_a / 10.0) ** 0.143 if h_a > 0 else 0.0
        y_tip = y_spine - (V_a / V_top) * profile_w
        if y_spine - y_tip < aw * 1.5:
            continue
        ax.plot([x_wall, x_wall], [y_spine, y_tip + aw], [h_a, h_a],
                color='red', lw=0.7, alpha=0.6, zorder=6)
        flat_arrow_3d(ax, (x_wall, y_tip, h_a), (0, -1, 0), (0, 0, 1),
                      aw, color='red', alpha=0.75)

    ax.plot([x_wall] * n_profile, y_curve, h_vals,
            color='red', lw=0.7, zorder=7)
    ax.plot([x_wall, x_wall], [y_spine, y_spine], [0.0, h_max],
            color='red', lw=0.9, alpha=0.5, zorder=6)


def add_dimension_arrows(ax, sol, span, xl, yl):
    x, y, h = sol.x, sol.y, sol.h
    aw   = span * 0.022
    dlkw = dict(color='black', lw=0.9, alpha=0.8)
    tlkw = dict(fontsize=11, color='black', alpha=0.8)

    x_lo, x_hi = float(x.min()), float(x.max())
    y_lo, y_hi = float(y.min()), float(y.max())
    h_hi       = float(h.max())

    margin = span * 0.05

    yp = yl[0] - margin
    ax.plot([x_lo, x_hi], [yp]*2, [0]*2, **dlkw)
    flat_arrow_3d(ax, (x_lo, yp, 0), (-1, 0, 0), (0, 1, 0), aw)
    flat_arrow_3d(ax, (x_hi, yp, 0), ( 1, 0, 0), (0, 1, 0), aw)
    ax.text((x_lo+x_hi)/2, yp - 2 * margin, aw,
            f'{x_hi-x_lo:.1f} m [x]', ha='center', va='bottom', **tlkw)

    xp = xl[1] + margin
    ax.plot([xp]*2, [y_lo, y_hi], [0]*2, **dlkw)
    flat_arrow_3d(ax, (xp, y_lo, 0), (0, -1, 0), (1, 0, 0), aw)
    flat_arrow_3d(ax, (xp, y_hi, 0), (0,  1, 0), (1, 0, 0), aw)
    ax.text(xp + margin, (y_lo+y_hi)/2, aw,
            f'{y_hi-y_lo:.1f} m [y]', ha='left', va='bottom', **tlkw)

    ax.plot([xp]*2, [yl[1]]*2, [0, h_hi], **dlkw)
    flat_arrow_3d(ax, (xp, yl[1], 0),    (0, 0, -1), (1, 0, 0), aw)
    flat_arrow_3d(ax, (xp, yl[1], h_hi), (0, 0,  1), (1, 0, 0), aw)
    ax.text(xp + margin * 0.4, yl[1], h_hi/2,
            f'{h_hi:.1f} m [h]', ha='left', va='center', **tlkw)


def plot_trajectory_3d(sol, n_birds=5):
    x, y, h = sol.x, sol.y, sol.h

    fig = plt.figure(figsize=(13, 12))
    ax  = fig.add_subplot(111, projection='3d')

    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    span  = max(x.max() - x.min(), y.max() - y.min())

    xl = (mid_x - span / 2, mid_x + span / 2)
    yl = (mid_y - span / 2, mid_y + span / 2)
    zl = (0.0, span)

    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
    ax.set_box_aspect([1, 1, 1])

    add_curtain(ax, sol)
    ax.plot(x, y, h, lw=2, color='C0', zorder=5)
    add_flight_arrows(ax, sol, span)

    pkw = dict(lw=0.8, color='black', alpha=0.6)
    ax.plot(x, y,                         np.zeros_like(h),  **pkw)
    ax.plot(x, yl[1] * np.ones_like(y),   h,                 **pkw)
    ax.plot(xl[0] * np.ones_like(x), y,   h,                 **pkw)

    bkw = dict(lw=0.6, color='black', alpha=0.5)
    ax.plot([xl[0], xl[1], xl[1], xl[0], xl[0]], [yl[1]]*5,
            [0, 0, zl[1], zl[1], 0], **bkw)
    ax.plot([xl[0]]*5, [yl[0], yl[1], yl[1], yl[0], yl[0]],
            [0, 0, zl[1], zl[1], 0], **bkw)

    add_bird_crosses(ax, sol, span, n_birds=n_birds)
    add_wind_shear_profile(ax, sol, span, xl, yl)
    add_dimension_arrows(ax, sol, span, xl, yl)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('lightgray')

    ax.set_title(f'Minimum turn half-cycle — $V_{{ref}}$ = {sol.V_ref:.2f} m/s  |  '
                 f'$T_{{half}}$ = {sol.T_cycle:.2f} s')
    plt.tight_layout()
    return fig, ax


fig, ax = plot_trajectory_3d(sol, n_birds=N_BIRDS)
fig.savefig(FIGURES_DIR / 'min_turn_trajectory_3d.png', dpi=150)
plt.close(fig)
print('Saved min_turn_trajectory_3d.png')

# ── 2-D projections ───────────────────────────────────────────────────────────
s    = 0.18
pad  = 0.65
fs   = 12
lpad = 4


def _draw_arrow_2d(ax, cx, cy, L, direction, color, label):
    akw = dict(arrowstyle='->', color=color, lw=1.5, mutation_scale=12)
    if direction == 'down':
        ax.annotate('', xy=(cx, cy - L / 2), xytext=(cx, cy + L / 2), arrowprops=akw)
    elif direction == 'left':
        ax.annotate('', xy=(cx - L / 2, cy), xytext=(cx + L / 2, cy), arrowprops=akw)
    elif direction == 'in':
        r = L * 0.2
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, color=color, lw=1.2))
        d = r / np.sqrt(2)
        ax.plot([cx - d, cx + d], [cy - d, cy + d], '-', color=color, lw=1.2)
        ax.plot([cx - d, cx + d], [cy + d, cy - d], '-', color=color, lw=1.2)


def make_proj(a, b, a_name, b_name, title, wind_dir, save_name):
    a_r = a.max() - a.min()
    b_r = b.max() - b.min()
    fig, ax = plt.subplots(figsize=(a_r * s + pad, b_r * s + pad))
    ax.plot(a, b, 'k', lw=1)
    ax.set_aspect('equal')
    ax.set_title(title, pad=4, fontsize=fs)
    ax.set_xlim(a.min(), a.max())
    ax.set_ylim(b.min(), b.max())
    ax.set_xticks([]); ax.set_yticks([])
    cx = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    cy = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
    L  = 0.15 * min(a_r, b_r)
    _draw_arrow_2d(ax, cx, cy, L, wind_dir, 'red', '')
    ax.set_xlabel(f'{a_r:.1f} m [{a_name}]', fontsize=fs, labelpad=lpad)
    ax.set_ylabel(f'{b_r:.1f} m [{b_name}]', fontsize=fs, labelpad=lpad)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES_DIR / save_name, dpi=150)
    plt.close(fig)
    print(f'Saved {save_name}')


make_proj(x, y, 'x', 'y', 'x-y plane',  wind_dir='down', save_name='min_turn_projection_xy.png')
make_proj(x, h, 'x', 'h', 'x-h plane',  wind_dir='in',   save_name='min_turn_projection_xh.png')
make_proj(y, h, 'y', 'h', 'y-h plane',  wind_dir='left', save_name='min_turn_projection_yh.png')

print(f'\nAll figures saved to {FIGURES_DIR.resolve()}')
