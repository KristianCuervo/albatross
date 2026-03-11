"""
min_vref_paths.py
=================
VMP polar diagram showing, for each heading psi, the velocity hodograph of
the minimum-V_ref orbit — i.e. the closed loop that the bird's instantaneous
ground-speed vector traces during one complete soaring cycle.

On a standard VMG polar:
  - Radial axis = instantaneous ground speed  ||(u, v)||
  - Angular axis = instantaneous travel direction  arctan2(u, v)
    (0 = headwind at top, clockwise)

Each orbit is periodic in velocity space, so it forms a closed loop.
The loops are coloured by V_ref_min: bright = cheap direction (tailwind),
dark = expensive (headwind).  The mirrored orbit (u → −u) is overlaid to
give the full left-right symmetric picture.

A dashed boundary curve connects the mean velocities of each orbit —
this is the "minimum-V_ref achievable ground speed" envelope.

Usage
-----
    cd /home/kristiancuervo/albatross/refactor
    /home/kristiancuervo/albatross/.venv/bin/python3 min_vref_paths.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from src.albatross import Albatross
from polar_sweep import MinVRefSolver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N          = 64
THETA_HALF = np.linspace(0, np.pi, 30)   # 30 headings, 0°…180°
OUTPUT     = os.path.join(os.path.dirname(__file__), "min_vref_paths.png")

# ---------------------------------------------------------------------------
bird = Albatross.from_toml()
print(f"min_vref_paths: N={N}, {len(THETA_HALF)} headings\n")

# ---------------------------------------------------------------------------
# Phase 1 — MinVRefSolver for each heading
# ---------------------------------------------------------------------------
phase1 = {}
warm_ics = None
for theta in THETA_HALF:
    solver = MinVRefSolver(bird=bird, N=N, psi=theta, ics=warm_ics)
    v_min, ics = solver.optimise()
    phase1[theta] = {'V_ref_min': v_min, 'ics': ics}
    if ics is not None:
        warm_ics = ics

# ---------------------------------------------------------------------------
# Compute net displacement per orbit (final location after one soaring cycle)
# ---------------------------------------------------------------------------
points = []   # list of (polar_angle, distance, theta_deg, mirror)
for theta, entry in phase1.items():
    if entry['ics'] is None:
        continue
    u  = np.asarray(entry['ics']['u'])
    v  = np.asarray(entry['ics']['v'])
    dt = float(entry['ics']['dt'])

    x_end = float(np.sum(u) * dt)
    y_end = float(np.sum(v) * dt)
    dist  = np.hypot(x_end, y_end)

    # Original orbit
    points.append({
        'angle': np.arctan2(x_end, y_end) % (2 * np.pi),
        'dist':  dist,
        'theta_deg': np.degrees(theta),
    })
    # Mirrored orbit (u → −u  ⟹  x_end → −x_end)
    points.append({
        'angle': np.arctan2(-x_end, y_end) % (2 * np.pi),
        'dist':  dist,
        'theta_deg': np.degrees(theta),
    })

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
angles    = np.array([p['angle']     for p in points])
dists     = np.array([p['dist']      for p in points])
theta_deg = np.array([p['theta_deg'] for p in points])

norm = Normalize(vmin=0, vmax=180)
cmap = plt.cm.plasma

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8),
                       layout='constrained')

sc = ax.scatter(angles, dists, c=theta_deg, cmap=cmap, norm=norm,
                s=40, zorder=3)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_title(
    r"Net displacement per soaring cycle at minimum $V_{ref}$",
    pad=20,
)

fig.colorbar(sc, ax=ax, label=r'Optimised heading $\psi$ [°]  (0 = headwind, 180 = tailwind)',
             pad=0.08)

fig.savefig(OUTPUT, dpi=150)
print(f"\nSaved: {OUTPUT}")
