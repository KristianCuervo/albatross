"""
main.py
=======
Runs the coarse-graining navigation analysis for multiple wind field types.

For each configuration, produces a 4-panel figure plus — when run with
--opt flag — overlays the CasADI-optimised path against the greedy path.

Wind fields
-----------
  1. single_vortex     — single CCW vortex (baseline)
  2. multi_vortex      — two CCW + one CW Rankine vortex
  3. hills             — stream-function hills and troughs + background flow
  4. noisy_vortex      — multi-vortex + additive smooth noise

Usage
-----
    # Just greedy paths (fast):
    /home/kristiancuervo/albatross/.venv/bin/python3 main.py

    # Greedy + CasADI optimised paths:
    /home/kristiancuervo/albatross/.venv/bin/python3 main.py --opt
"""

import os
import sys
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from wind_field      import (make_rotational_wind_field, make_multi_vortex_field,
                              make_hill_field, add_noise)
from tacking_lookup  import TackingLookup
from potential_map   import compute_potential, plan_path
from path_optimizer  import PathOptimizer

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
GRID_SIZE   = 100
DOMAIN      = 10_000.0          # 10 km × 10 km
START       = (500.0,   500.0)
GOAL        = (9500.0, 9500.0)
N_STEPS     = 150               # soaring cycles for the NLP (~100–200 m per step)
QUIVER_STEP = 6

NPZ_PATH = os.path.join(os.path.dirname(__file__),
                         '..', 'refactor', 'data', 'tacking_diagram.npz')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'figures')


# ---------------------------------------------------------------------------
# Wind field configurations
# ---------------------------------------------------------------------------
def build_configs() -> list[dict]:
    # All spatial lengths scaled ×10 vs the 1 km case.
    # Wind speeds (V_ref, bg_speed, noise amplitude) are unchanged — they are
    # intrinsic to the physics (albatross + shear profile), not the domain size.
    # Circulations Γ and stream-function amplitudes A scale ×10 to preserve
    # the same peak wind speeds:
    #   Rankine: v_tan(r_core) = Γ/(2π r_core)  →  ×10/×10 = constant  ✓
    #   Hill:    v(r=σ)        = A/(σ √e)        →  ×10/×10 = constant  ✓

    # 1. Single CCW vortex — centre explicitly set to domain centre
    wf1 = make_rotational_wind_field(
        grid_size=GRID_SIZE, domain=DOMAIN,
        center=(5000.0, 5000.0),
        V_min=10.0, V_max=20.0,
    )

    # 2. Multi-vortex: positions, r_core, and circulations all ×10
    wf2 = make_multi_vortex_field(
        vortices=[
            (2500.0, 7200.0,  90000.0),   # CCW, upper-left
            (7500.0, 2800.0,  70000.0),   # CCW, lower-right
            (5600.0, 6200.0, -60000.0),   # CW,  centre-right
        ],
        grid_size=GRID_SIZE, domain=DOMAIN,
        bg_speed=12.0, bg_toward=0.0, r_core=1500.0,
    )

    # 3. Hills & troughs: positions, σ, and amplitudes all ×10
    wf3 = make_hill_field(
        hills=[
            (2800.0, 7500.0,  45000.0, 2000.0),
            (7200.0, 2500.0,  35000.0, 1700.0),
            (4800.0, 5200.0, -40000.0, 1800.0),
            (2000.0, 3000.0,  25000.0, 1400.0),
        ],
        grid_size=GRID_SIZE, domain=DOMAIN,
        bg_speed=13.0, bg_toward=0.0,
    )

    # 4. Noisy multi-vortex — noise amplitude is a wind speed [m/s], unchanged
    wf4 = add_noise(wf2, amplitude=4.0, seed=42)

    return [
        {'name': 'single_vortex', 'label': 'Single CCW vortex',             'wf': wf1},
        {'name': 'multi_vortex',  'label': 'Multi-vortex (2 CCW + 1 CW)',   'wf': wf2},
        {'name': 'hills',         'label': 'Hills & troughs (stream func.)', 'wf': wf3},
        {'name': 'noisy_vortex',  'label': 'Noisy multi-vortex',            'wf': wf4},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', action='store_true',
                        help='Also run CasADI path optimiser')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading tacking diagram …")
    lookup = TackingLookup(NPZ_PATH)
    print(f"  {lookup.n_thetas} headings loaded\n")

    configs = build_configs()

    for cfg in configs:
        name  = cfg['name']
        label = cfg['label']
        wf    = cfg['wf']
        print(f"{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        # --- potential map + greedy path ---
        t0 = time.time()
        pm = compute_potential(wf, lookup, GOAL, GRID_SIZE, DOMAIN)
        print(f"  Potential: {time.time()-t0:.1f} s  "
              f"(range {np.nanmin(pm['potential']):.1f} … "
              f"{np.nanmax(pm['potential']):.1f} m/s)")

        greedy = plan_path(wf, pm, GOAL, START, max_steps=600)
        print(f"  Greedy: {len(greedy)-1} cycles, "
              f"final = ({greedy[-1,0]:.0f}, {greedy[-1,1]:.0f}) m")

        # --- CasADI optimised path (optional) ---
        opt_result = None
        if args.opt:
            print(f"  CasADI optimiser (N={N_STEPS}) …")
            t1 = time.time()
            opt = PathOptimizer(NPZ_PATH, wf, N=N_STEPS, start=START, goal=GOAL)
            # Use greedy path as warm start (resampled to N+1 points inside
            # optimise()).  Normalise objective by domain² for good scaling.
            opt_result = opt.optimise(x0_traj=greedy,
                                      obj_scale=DOMAIN ** 2)
            print(f"  Optimiser: {time.time()-t1:.1f} s")

        out = os.path.join(OUT_DIR, f'{name}.png')
        _plot(wf, pm, greedy, opt_result, label, out)
        print(f"  Saved: {out}\n")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _plot(wf, pm, greedy, opt_result, title, out_path) -> None:
    xs, ys = wf['x_centers'], wf['y_centers']
    XX, YY = wf['XX'], wf['YY']
    s      = QUIVER_STEP

    pot   = pm['potential']
    vx_g  = pm['best_vx']
    vy_g  = pm['best_vy']
    speed = np.hypot(vx_g, vy_g)

    vmin_p = np.nanpercentile(pot, 2)
    vmax_p = np.nanpercentile(pot, 98)

    with np.errstate(invalid='ignore'):
        vx_n = np.where(speed > 0, vx_g / speed, 0.0)
        vy_n = np.where(speed > 0, vy_g / speed, 0.0)

    alpha  = wf['alpha']
    blow_x = -np.cos(alpha)
    blow_y = -np.sin(alpha)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ---- panel 1: wind field ----
    ax = axes[0, 0]
    im = ax.pcolormesh(xs, ys, wf['V_ref'].T, cmap='viridis',
                       shading='nearest', vmin=9, vmax=25)
    fig.colorbar(im, ax=ax, label='$V_{ref}$ [m/s]')
    ax.quiver(XX[::s, ::s], YY[::s, ::s],
              blow_x[::s, ::s], blow_y[::s, ::s],
              color='white', alpha=0.65, scale=28, width=0.003)
    ax.set_title('Wind field')
    _ax_fmt(ax)

    # ---- panel 2: potential map ----
    ax = axes[0, 1]
    im2 = ax.pcolormesh(xs, ys, pot.T, cmap='RdYlGn',
                        shading='nearest', vmin=vmin_p, vmax=vmax_p)
    fig.colorbar(im2, ax=ax, label='Max progress toward goal [m/s]')
    ax.contour(xs, ys, pot.T, levels=12, colors='k',
               linewidths=0.35, alpha=0.35)
    ax.set_title('Potential map')
    _ax_fmt(ax)

    # ---- panel 3: optimal directions ----
    ax = axes[1, 0]
    im3 = ax.pcolormesh(xs, ys, pot.T, cmap='RdYlGn',
                        shading='nearest', vmin=vmin_p, vmax=vmax_p)
    fig.colorbar(im3, ax=ax, label='Max progress toward goal [m/s]')
    ax.quiver(XX[::s, ::s], YY[::s, ::s],
              vx_n[::s, ::s], vy_n[::s, ::s],
              color='k', alpha=0.55, scale=38, width=0.002)
    ax.set_title('Optimal heading directions')
    _ax_fmt(ax)

    # ---- panel 4: paths ----
    ax = axes[1, 1]
    im4 = ax.pcolormesh(xs, ys, pot.T, cmap='RdYlGn',
                        shading='nearest', vmin=vmin_p, vmax=vmax_p)
    fig.colorbar(im4, ax=ax, label='Max progress toward goal [m/s]')

    # Greedy path
    if len(greedy) >= 2:
        ax.plot(greedy[:, 0], greedy[:, 1], 'b-o', markersize=4,
                linewidth=1.8, label=f'Greedy ({len(greedy)-1} cycles)', zorder=5)
        step = max(1, (len(greedy) - 1) // 8)
        for k in range(0, len(greedy), step):
            ax.annotate(str(k), greedy[k], fontsize=7, color='navy',
                        ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points')

    # CasADI optimised path (if computed)
    if opt_result is not None:
        op = opt_result['positions']
        lbl = (f"CasADI opt ({len(op)-1} cycles)"
               + (" ✓" if opt_result['success'] else " [non-conv]"))
        ax.plot(op[:, 0], op[:, 1], 'r-s', markersize=5,
                linewidth=2.0, label=lbl, zorder=6)
        for k, pt in enumerate(op):
            ax.annotate(str(k), pt, fontsize=7, color='darkred',
                        ha='center', va='top',
                        xytext=(0, -6), textcoords='offset points')

    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Path comparison')
    _ax_fmt(ax)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _ax_fmt(ax) -> None:
    ax.plot(*START, 'g^', markersize=10, zorder=6)
    ax.plot(*GOAL,  'r*', markersize=12, zorder=6)
    ax.set_xlim(0, DOMAIN)
    ax.set_ylim(0, DOMAIN)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
