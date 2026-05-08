"""
Minimum wind speed polar diagram V_ref_min(θ).

Loads precomputed data from data/mesoscale/tacking_diagram.npz (legacy
p1_theta / p1_vrefmin keys, 58 points covering 0 → π).  Mirrors to the
full circle via the left-right symmetry of the problem and saves a polar
figure to figures/mesoscale/vref_min_polar.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

ROOT        = Path(__file__).parent.parent
DATA_NPZ    = ROOT / 'data' / 'mesoscale' / 'tacking_diagram.npz'
FIGURES_DIR = ROOT / 'figures' / 'mesoscale'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

d  = np.load(DATA_NPZ)
th = d['p1_theta']   # 0 → π
vr = d['p1_vrefmin']

# Mirror: right half (0→π) + left half (2π→π, same V_ref by symmetry)
th_full = np.concatenate([th, 2 * np.pi - th[-2:0:-1]])
vr_full = np.concatenate([vr, vr[-2:0:-1]])
# Close the curve
th_plot = np.append(th_full, th_full[0])
vr_plot = np.append(vr_full, vr_full[0])

min_idx = np.argmin(vr_full)
min_vr  = vr_full[min_idx]
min_th  = th_full[min_idx]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))

r_outer = vr_plot.max() * 1.05
ax.fill_between(th_plot, vr_plot, r_outer, color='lightgray', alpha=0.6, lw=0)
ax.plot(th_plot, vr_plot, lw=2, color='#4B0082')

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

fig.tight_layout()
out = FIGURES_DIR / 'vref_min_polar.png'
fig.savefig(out, dpi=150)
plt.close(fig)
print(f'Saved {out}')
print(f'Global minimum: {min_vr:.3f} m/s at θ = {np.degrees(min_th):.1f}°')
