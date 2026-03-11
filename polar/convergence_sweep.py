"""
convergence_sweep.py

Runs polar sweeps for N = 16, 32, 64 simulation steps and several V_ref values,
plots v_avg and T_cycle as polar diagrams with one line per N value.

Usage:
    python convergence_sweep.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from albatross import Albatross
from optimiser import Optimiser

# ── Configuration ────────────────────────────────────────────────────────────

V_REF_VALUES = [20.0]   # wind reference speeds [m/s]
N_VALUES     = [64]          # simulation step counts to compare
N_THETA      = 60                   # number of theta points in the sweep
N_WORKERS    = 8                     # parallel processes

THETAS = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)

# ── Worker ───────────────────────────────────────────────────────────────────

def _run_single(args):
    """Worker: run one (V_ref, N, theta) optimisation. Returns (V_ref, N, i, v_avg, t_cycle)."""
    V_ref, N, i, theta = args
    bird = Albatross()
    opt  = Optimiser(bird=bird, theta=theta, N=N, V_ref=V_ref)
    opt.optimise()
    v_avg   = float(opt.sol.value(opt.v_avg))
    t_cycle = float(opt.sol.value(opt.T_cycle))
    print(f"  V_ref={V_ref:5.1f}  N={N:3d}  theta={np.degrees(theta):6.1f}°"
          f"  v_avg={v_avg:.3f}  T={t_cycle:.3f}", flush=True)
    return V_ref, N, i, v_avg, t_cycle

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Build task list: all combinations of (V_ref, N, theta_index)
    tasks = [
        (V_ref, N, i, theta)
        for V_ref in V_REF_VALUES
        for N in N_VALUES
        for i, theta in enumerate(THETAS)
    ]

    print(f"Dispatching {len(tasks)} tasks across {N_WORKERS} workers …")

    with Pool(N_WORKERS) as pool:
        raw = pool.map(_run_single, tasks)

    # Assemble results: results[(V_ref, N)] = (v_avgs_array, t_cycles_array)
    results = {}
    for V_ref, N, i, v_avg, t_cycle in raw:
        key = (V_ref, N)
        if key not in results:
            results[key] = (np.full(N_THETA, np.nan), np.full(N_THETA, np.nan))
        results[key][0][i] = v_avg
        results[key][1][i] = t_cycle

    # ── Plot ─────────────────────────────────────────────────────────────────

    thetas_closed = np.append(THETAS, THETAS[0])

    fig, axes = plt.subplots(
        1, 2,
        subplot_kw={"projection": "polar"},
        figsize=(14, 6),
    )

    colors = plt.cm.tab10.colors  # distinct colors per V_ref
    linestyles = ["-", "--", ":", "-."]  # distinct styles per N

    for ci, V_ref in enumerate(V_REF_VALUES):
        for li, N in enumerate(N_VALUES):
            v_avgs, t_cycles = results[(V_ref, N)]
            v_closed = np.append(v_avgs, v_avgs[0])
            t_closed = np.append(t_cycles, t_cycles[0])
            label = f"$V_{{ref}}={V_ref:.0f}$, N={N}"
            kw = dict(color=colors[ci], linestyle=linestyles[li], label=label)
            axes[0].plot(thetas_closed, v_closed, **kw)
            axes[1].plot(thetas_closed, t_closed, **kw)

    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    axes[0].set_title(r"$v_\mathrm{avg}$ [m/s]", va="bottom", fontsize=13)
    axes[1].set_title(r"$T_\mathrm{cycle}$ [s]", va="bottom", fontsize=13)

    fig.suptitle("Polar convergence study", fontsize=14)
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "convergence_polar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.show()
