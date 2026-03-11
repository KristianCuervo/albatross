"""
tacking_diagram.py
==================
Polar scatter diagram of dynamic soaring ground speed vs flight heading,
coloured by V_ref (wind-shear reference speed).

Phase 1: For each heading theta (0…π), find the minimum V_ref that allows
         sustained soaring (using MinVRefSolver from polar_sweep.py).
Phase 2: For each heading, sweep V_ref from ceil(V_ref_min) to V_REF_MAX
         (integer steps) and record the ground speed achieved (using Solver).
Phase 3: Mirror by u → −u symmetry to obtain the full polar (0…2π).
Phase 4: Plot scatter coloured by V_ref + V_ref_min boundary curve.

Coordinate system (consistent with refactor throughout):
  x   crosswind
  y   upwind (+y = headwind, wind blows in −y)
  theta/psi = 0 → flying in +y (headwind) — same convention as Solver and polar_sweep

Polar plot: theta=0 at top (N), increasing clockwise.

Usage
-----
    cd /home/kristiancuervo/albatross/refactor
    /home/kristiancuervo/albatross/.venv/bin/python3 tacking_diagram.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.albatross import Albatross
from src.solver import Solver
from polar_sweep import MinVRefSolver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N           = 64                            # collocation nodes (Phase 1)
THETA_HALF  = np.linspace(0, np.pi, 58)    # 0°…180° every 15°
V_REF_MAX   = 25                            # ceiling for V_ref sweep [m/s]
V_REF_STEP  = 1                             # step size for V_ref sweep (integer)
OUTPUT      = os.path.join(os.path.dirname(__file__), "tacking_diagram.png")
OUTPUT_TCYC = os.path.join(os.path.dirname(__file__), "tacking_diagram_tcycle.png")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Load bird parameters
# ---------------------------------------------------------------------------
bird = Albatross.from_toml()

print(f"Tacking diagram: N={N}, {len(THETA_HALF)} headings from "
      f"{np.degrees(THETA_HALF[0]):.0f}° to {np.degrees(THETA_HALF[-1]):.0f}°, "
      f"V_ref up to {V_REF_MAX} m/s\n")

# ===========================================================================
# Phase 1 — V_ref_min sweep
# ===========================================================================
print("=" * 60)
print("Phase 1: minimising V_ref per heading")
print("=" * 60)

phase1 = {}   # theta -> {'V_ref_min': float | None, 'ics': dict | None}
warm_ics = None

for theta in THETA_HALF:
    solver = MinVRefSolver(bird=bird, N=N, psi=theta, ics=warm_ics)
    v_min, ics = solver.optimise()
    phase1[theta] = {'V_ref_min': v_min, 'ics': ics}
    if ics is not None:
        warm_ics = ics   # chain warm start; propagate last good value on failure

# ===========================================================================
# Phase 2 — Speed polar sweep
# ===========================================================================
print("\n" + "=" * 60)
print("Phase 2: sweeping V_ref for max ground speed per heading")
print("=" * 60)


def adaptive_N(v_ref: float) -> int:
    """Scale N linearly with V_ref: N=50 at V_ref=10, N=75 at V_ref=15, N=100 at V_ref=20."""
    return int(np.clip(round(7.0 * v_ref), 64, 150))


results = []   # list of dicts: {theta, V_ref, N_v, u_avg, v_avg, speed, angle}

for theta in THETA_HALF:
    entry = phase1[theta]
    if entry['V_ref_min'] is None:
        print(f"  theta = {np.degrees(theta):5.1f}°  SKIPPED (no V_ref_min)")
        continue

    v_start = math.ceil(entry['V_ref_min'])
    prev_container = None   # cold start for first V_ref at each theta
    prev_speed     = None

    for v_ref in range(v_start, V_REF_MAX + 1, V_REF_STEP):
        N_v = adaptive_N(v_ref)
        warm_ics = prev_container.as_ics(N_target=N_v) if prev_container is not None else None
        try:
            container = Solver(
                bird=bird, theta=theta, N=N_v, V_ref=float(v_ref), ics=warm_ics
            ).optimise()
            speed = float(np.hypot(np.mean(container.u), np.mean(container.v)))

            # If warm start caused a regression, retry cold and take the better result
            if warm_ics is not None and prev_speed is not None and speed < prev_speed * 0.95:
                try:
                    container_cold = Solver(
                        bird=bird, theta=theta, N=N_v, V_ref=float(v_ref), ics=None
                    ).optimise()
                    speed_cold = float(np.hypot(np.mean(container_cold.u), np.mean(container_cold.v)))
                    if speed_cold > speed:
                        print(f"  [retry cold] {speed:.3f} → {speed_cold:.3f} m/s")
                        container = container_cold
                        speed = speed_cold
                except RuntimeError:
                    pass   # keep the warm-started result

            u_avg = float(np.mean(container.u))
            v_avg = float(np.mean(container.v))
            speed = float(np.hypot(u_avg, v_avg))
            angle = np.arctan2(u_avg, v_avg) % (2 * np.pi)
            print(f"  theta = {np.degrees(theta):5.1f}°, V_ref = {v_ref:2d}, N = {N_v:3d} → speed = {speed:.3f} m/s, T = {container.T_cycle:.2f} s")
            results.append({
                'theta':   theta,
                'V_ref':   v_ref,
                'N_v':     N_v,
                'u_avg':   u_avg,
                'v_avg':   v_avg,
                'speed':   speed,
                'angle':   angle,
                'T_cycle': container.T_cycle,
            })
            prev_container = container
            prev_speed     = speed
        except RuntimeError as exc:
            print(f"  theta = {np.degrees(theta):5.1f}°, V_ref = {v_ref:2d}, N = {N_v:3d} → FAILED ({exc})")
            prev_container = None
            prev_speed     = None

# ===========================================================================
# Phase 3 — Mirror by u → −u symmetry
# ===========================================================================
print(f"\nPhase 3: mirroring {len(results)} primary results")

mirrored = []
for r in results:
    mirrored.append({
        'theta':   r['theta'],
        'V_ref':   r['V_ref'],
        'N_v':     r['N_v'],
        'u_avg':  -r['u_avg'],
        'v_avg':   r['v_avg'],
        'speed':   r['speed'],
        'angle':   np.arctan2(-r['u_avg'], r['v_avg']) % (2 * np.pi),
        'T_cycle': r['T_cycle'],
    })

all_results = results + mirrored

# Build V_ref_min boundary from phase-1 ICS mean velocities
boundary_pts = []
for theta, entry in phase1.items():
    if entry['ics'] is None:
        continue
    ics = entry['ics']
    u_avg = float(np.mean(ics['u']))
    v_avg = float(np.mean(ics['v']))
    speed = np.hypot(u_avg, v_avg)
    # Original point
    angle = np.arctan2(u_avg, v_avg) % (2 * np.pi)
    boundary_pts.append((angle, speed))
    # Mirrored point (u → −u)
    angle_m = np.arctan2(-u_avg, v_avg) % (2 * np.pi)
    boundary_pts.append((angle_m, speed))

# Separate boundary into original (angles ~0..π) and mirror (angles ~π..2π)
boundary_orig = sorted(
    [(float(np.arctan2(float(np.mean(e['ics']['u'])), float(np.mean(e['ics']['v']))) % (2 * np.pi)),
      float(np.hypot(np.mean(e['ics']['u']), np.mean(e['ics']['v']))))
     for e in phase1.values() if e['ics'] is not None],
    key=lambda p: p[0]
)
boundary_mirr = sorted(
    [(float(np.arctan2(-float(np.mean(e['ics']['u'])), float(np.mean(e['ics']['v']))) % (2 * np.pi)),
      float(np.hypot(np.mean(e['ics']['u']), np.mean(e['ics']['v']))))
     for e in phase1.values() if e['ics'] is not None],
    key=lambda p: p[0]
)

# ===========================================================================
# Save data
# ===========================================================================
os.makedirs(DATA_DIR, exist_ok=True)

# Phase 1: V_ref_min per heading
p1_theta   = np.array(list(phase1.keys()))
p1_vrefmin = np.array([e['V_ref_min'] if e['V_ref_min'] is not None else np.nan
                        for e in phase1.values()])

# Phase 2+3: full VMP polar data (original + mirrored)
def _col(key, src=all_results):
    return np.array([r[key] for r in src])

npz_path = os.path.join(DATA_DIR, "tacking_diagram.npz")
np.savez(
    npz_path,
    # Phase 1
    p1_theta   = p1_theta,
    p1_vrefmin = p1_vrefmin,
    # Phase 2 (original half only — mirror is always derivable)
    theta   = _col('theta',   results),
    V_ref   = _col('V_ref',   results),
    N_v     = _col('N_v',     results),
    u_avg   = _col('u_avg',   results),
    v_avg   = _col('v_avg',   results),
    speed   = _col('speed',   results),
    angle   = _col('angle',   results),
    T_cycle = _col('T_cycle', results),
)
print(f"\nData saved: {npz_path}")
print(f"  Phase 1: {len(p1_theta)} headings")
print(f"  Phase 2: {len(results)} primary results  ({len(all_results)} incl. mirror)")

# ===========================================================================
# Phase 4 — Plotting
# ===========================================================================
print("\nPhase 4: plotting")

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

v_ref_set = sorted(set(r['V_ref'] for r in results))
norm = Normalize(vmin=min(v_ref_set), vmax=max(v_ref_set))
cmap = plt.cm.plasma

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# One pair of arcs per V_ref (original half + mirrored half, same colour)
for v_ref in v_ref_set:
    color = cmap(norm(v_ref))

    # Original arc: actual travel angles, sorted
    pts_o = sorted(
        [(r['angle'], r['speed']) for r in results if r['V_ref'] == v_ref],
        key=lambda p: p[0]
    )
    # Mirrored arc: reflected travel angles, sorted
    pts_m = sorted(
        [(r['angle'], r['speed']) for r in mirrored if r['V_ref'] == v_ref],
        key=lambda p: p[0]
    )

    if len(pts_o) >= 2:
        ao, so = zip(*pts_o)
        ax.plot(ao, so, color=color, lw=1.2, alpha=0.85)
    if len(pts_m) >= 2:
        am, sm = zip(*pts_m)
        ax.plot(am, sm, color=color, lw=1.2, alpha=0.85)
# Colorbar
sm_cb = ScalarMappable(cmap=cmap, norm=norm)
sm_cb.set_array([])
plt.colorbar(sm_cb, ax=ax, label='$V_{ref}$ [m/s]', pad=0.08)

ax.set_theta_zero_location("N")   # theta=0 (headwind) at top
ax.set_theta_direction(-1)        # clockwise
ax.set_title(r"Dynamic soaring VMP polar — iso-$V_{ref}$ curves", pad=20)
ax.legend(loc='lower right')
fig.savefig(OUTPUT, dpi=150)
print(f"Saved: {OUTPUT}")

# ---------------------------------------------------------------------------
# T_cycle polar — same iso-V_ref structure, radius = cycle period
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

for v_ref in v_ref_set:
    color = cmap(norm(v_ref))

    # Use optimisation heading theta as polar angle (not actual travel angle)
    pts_o = sorted(
        [(r['theta'], r['T_cycle']) for r in results if r['V_ref'] == v_ref],
        key=lambda p: p[0]
    )
    # Mirror half: reflected heading = 2π − theta
    pts_m = sorted(
        [((2 * np.pi - r['theta']) % (2 * np.pi), r['T_cycle'])
         for r in results if r['V_ref'] == v_ref],
        key=lambda p: p[0]
    )

    if len(pts_o) >= 2:
        ao, to = zip(*pts_o)
        ax2.plot(ao, to, color=color, lw=1.2, alpha=0.85)
    if len(pts_m) >= 2:
        am, tm = zip(*pts_m)
        ax2.plot(am, tm, color=color, lw=1.2, alpha=0.85)

sm_cb2 = ScalarMappable(cmap=cmap, norm=norm)
sm_cb2.set_array([])
plt.colorbar(sm_cb2, ax=ax2, label='$V_{ref}$ [m/s]', pad=0.08)

ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)
ax2.set_title(r"Soaring cycle period $T_{cycle}$ — iso-$V_{ref}$ curves", pad=20)
fig2.savefig(OUTPUT_TCYC, dpi=150)
print(f"Saved: {OUTPUT_TCYC}")
