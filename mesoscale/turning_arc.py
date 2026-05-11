"""
Turning arc: compares max-VMG polar arcs at V_ref = 15 m/s under two boundary conditions.

  Periodic BC   (reflective_bc=False): optimal closed orbit — upper bound on VMG.
  Reflective BC (reflective_bc=True):  half-cycle where u[-1] = −u[0], modelling
                                        the constraint that crosswind direction reverses
                                        each half-period (tacking maneuver).

The gap between the two arcs in polar (u, v) velocity space quantifies the turning loss.

RECOMPUTE=True  : sweep 36 headings (θ ∈ [0, π]) with reflective BC (~5–20 min).
RECOMPUTE=False : load precomputed result from data/turning_arc.npz.

Figures saved to figures/.
"""

import sys
import multiprocessing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from microscale.bird import Albatross
from microscale import Solver

RECOMPUTE   = False
N_PROCS     = 4
V_REF       = 15.0
N_NODES     = 40
N_HEADINGS  = 36

_HERE       = Path(__file__).parent
DATA_DIR    = _HERE / 'data'
FIGURES_DIR = _HERE / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

REFL_NPZ = DATA_DIR / 'turning_arc.npz'
TACK_NPZ = DATA_DIR / 'tacking_diagram.npz'

bird = Albatross.from_toml(Path(__file__).parent.parent / 'microscale' / 'albatross.toml')
print(bird)


# ── Module-level worker (picklable for multiprocessing) ───────────────────────

def _solve_one(args):
    bird, theta, V_ref, N = args
    try:
        sol = Solver(bird=bird, theta=theta, N=N, V_ref=V_ref,
                     mode='max_vmg', reflective_bc=True).optimise()
        return (theta, float(np.mean(sol.u)), float(np.mean(sol.v)),
                float(sol.obj), float(sol.T_cycle))
    except Exception:
        return None


# ── Compute or load reflective BC sweep ───────────────────────────────────────

if RECOMPUTE or not REFL_NPZ.exists():
    print(f'Sweeping {N_HEADINGS} headings with reflective BC at V_ref={V_REF} m/s ...')
    thetas   = np.linspace(0, np.pi, N_HEADINGS, endpoint=False)
    job_args = [(bird, th, V_REF, N_NODES) for th in thetas]

    with multiprocessing.Pool(N_PROCS) as pool:
        raw = pool.map(_solve_one, job_args)

    converged = [r for r in raw if r is not None]
    print(f'  {len(converged)}/{N_HEADINGS} converged')

    theta_r, u_r, v_r, obj_r, Tc_r = map(np.array, zip(*converged))
    order = np.argsort(theta_r)
    theta_r, u_r, v_r, obj_r, Tc_r = (
        theta_r[order], u_r[order], v_r[order], obj_r[order], Tc_r[order]
    )
    np.savez_compressed(REFL_NPZ,
        theta=theta_r, u_avg=u_r, v_avg=v_r,
        obj=obj_r, T_cycle=Tc_r, V_ref=V_REF)
    print(f'  Saved → {REFL_NPZ}')
else:
    print(f'Loading reflective BC sweep from {REFL_NPZ}')

d_r    = np.load(REFL_NPZ)
u_refl = d_r['u_avg']
v_refl = d_r['v_avg']


# ── Load normal BC arc (V_ref = 15 m/s) from existing tacking diagram ────────

d_n    = np.load(TACK_NPZ)
mask_n = d_n['V_ref'] == V_REF
order_n = np.argsort(d_n['theta'][mask_n])
u_norm  = d_n['u_avg'][mask_n][order_n]
v_norm  = d_n['v_avg'][mask_n][order_n]

print(f'\nNormal BC:    {len(u_norm)} containers at V_ref={V_REF} m/s')
print(f'Reflective BC: {len(u_refl)} converged at V_ref={V_REF} m/s')


# ── Arc construction ──────────────────────────────────────────────────────────

def _right_arc_sorted(u_arr, v_arr):
    """Right-half arc (u > 0), sorted by polar angle."""
    mask   = u_arr > 0
    u_, v_ = u_arr[mask], v_arr[mask]
    angles = np.arctan2(u_, v_)          # ∈ (0, π) since u > 0
    radii  = np.hypot(u_, v_)
    order  = np.argsort(angles)
    return angles[order], radii[order]


def _two_arcs(u_arr, v_arr):
    """Right arc + mirrored left arc (ensemble.py convention)."""
    angles_r = np.arctan2( u_arr,         v_arr        ) % (2 * np.pi)
    angles_l = np.arctan2(-u_arr[::-1],   v_arr[::-1]  ) % (2 * np.pi)
    radii_r  = np.hypot(u_arr, v_arr)
    radii_l  = radii_r[::-1]
    return angles_r, radii_r, angles_l, radii_l


# ── Polar comparison plot ─────────────────────────────────────────────────────

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))

# Normal BC
ar_n, rr_n, al_n, rl_n = _two_arcs(u_norm, v_norm)
ax.plot(ar_n, rr_n, color='C0', lw=2.0, label=f'Periodic BC  ($V_{{ref}}={V_REF:.0f}$ m/s)')
ax.plot(al_n, rl_n, color='C0', lw=2.0)

# Reflective BC
ar_r, rr_r, al_r, rl_r = _two_arcs(u_refl, v_refl)
ax.plot(ar_r, rr_r, color='C1', lw=2.0, linestyle='--',
        label=f'Reflective BC  ($V_{{ref}}={V_REF:.0f}$ m/s, tacking)')
ax.plot(al_r, rl_r, color='C1', lw=2.0, linestyle='--')

# Shaded turning-loss region (right half, then mirrored left half)
ang_n_rh, rad_n_rh = _right_arc_sorted(u_norm, v_norm)
ang_r_rh = np.arctan2(u_refl, v_refl)         # all u_refl > 0 → ∈ (0, π)
rad_r_rh = np.hypot(u_refl, v_refl)
order_rh = np.argsort(ang_r_rh)
ang_r_rh, rad_r_rh = ang_r_rh[order_rh], rad_r_rh[order_rh]

ag_lo = max(ang_n_rh[0], ang_r_rh[0])
ag_hi = min(ang_n_rh[-1], ang_r_rh[-1])

if ag_hi > ag_lo:
    ag   = np.linspace(ag_lo, ag_hi, 300)
    rn_i = np.interp(ag, ang_n_rh, rad_n_rh)
    rr_i = np.interp(ag, ang_r_rh, rad_r_rh)

    # Right half: trace normal BC upwind→downwind, then reflective BC downwind→upwind
    ax.fill(np.concatenate([ag,            ag[::-1]]),
            np.concatenate([rn_i,          rr_i[::-1]]),
            alpha=0.18, color='grey', zorder=0, label='Turning loss')

    # Left half: same radii, mirrored angles (2π − angle)
    ax.fill(np.concatenate([2*np.pi - ag,       2*np.pi - ag[::-1]]),
            np.concatenate([rn_i,               rr_i[::-1]]),
            alpha=0.18, color='grey', zorder=0)

# Wind arrow at θ = 0 (top), same style as ensemble.py
r_max = max(rr_n.max(), rr_r.max() if len(rr_r) else 0)
ax.annotate('', xy=(0, r_max * 0.65), xytext=(0, r_max * 0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=2, mutation_scale=20))
ax.text(0, r_max * 0.97, 'Wind', ha='center', va='bottom',
        fontsize=11, fontweight='bold')

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_thetagrids(range(0, 360, 30), labels=[''] * 12)
ax.legend(loc='lower right', bbox_to_anchor=(1.38, -0.05), fontsize=11)
ax.set_title(
    f'Turning loss at $V_{{ref}} = {V_REF:.0f}$ m/s\n'
    r'Shaded region = VMG sacrificed by tacking constraint',
    pad=20, fontsize=12,
)

fig.tight_layout()
out = FIGURES_DIR / 'turning_arc.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nFigure saved → {out}')
