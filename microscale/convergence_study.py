"""
Convergence study: V_ref_min and T_cycle vs N_steps.

Runs the min_vref Sachs solver for N = 8, 16, 32, 64, 128, 256
and plots both quantities on a log-log scale.  The N=128 result is
treated as the reference (final converged value).

Each solve warm-starts from the previous N solution via Container.as_ics().
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from microscale.bird import Albatross
from microscale import Solver

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / 'data'
FIGURES_DIR = ROOT / 'figures' / 'microscale'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

bird = Albatross.from_toml(DATA_DIR / 'albatross.toml')

N_VALUES = [8, 16, 32, 64, 128, 256]

v_ref_results = []
t_cycle_results = []

prev_sol = None

for N in N_VALUES:
    print(f'Solving N={N} ...', flush=True)
    ics = prev_sol.as_ics(N) if prev_sol is not None else None
    sol = Solver(
        bird=bird,
        theta=0.0,
        N=N,
        mode='min_vref',
        add_progress_constraint=False,
        h_min=0.5,
        T_cycle_min=5.0,
        T_cycle_max=15.0,
        ics=ics,
    ).optimise()
    print(f'  V_ref = {sol.V_ref:.4f} m/s   T_cycle = {sol.T_cycle:.4f} s')
    v_ref_results.append(sol.V_ref)
    t_cycle_results.append(sol.T_cycle)
    prev_sol = sol

N_arr     = np.array(N_VALUES)
vref_arr  = np.array(v_ref_results)
tcyc_arr  = np.array(t_cycle_results)

# ── Plot ──────────────────────────────────────────────────────────────────────
cases = [
    (vref_arr, r'$V_{\mathrm{ref,min}}$', 'm/s', 'convergence_study_vref.png'),
    (tcyc_arr, r'$T_{\mathrm{cycle}}$',   's',   'convergence_study_tcycle.png'),
]

for vals, label, unit, fname in cases:
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xticks(N_arr)
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.minorticks_off()

    ax.grid(True, which='major', color='grey', alpha=0.35, lw=0.8, zorder=0)

    ax.loglog(N_arr, vals, 'o-', color='C0', lw=1.5, ms=6, zorder=3)

    ax.set_xlabel('$N$ (collocation nodes)', fontsize=12)
    ax.set_ylabel(f'{label}  [{unit}]', fontsize=12)

    fig.tight_layout()
    out = FIGURES_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved -> {out}')
