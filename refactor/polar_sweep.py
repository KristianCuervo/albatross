"""
polar_sweep.py
==============
Polar diagram of minimum V_ref vs. flight heading for dynamic soaring.

Reproduces the sachs/script.ipynb optimisation (minimise V_ref — the wind-shear
reference speed needed for sustained soaring) but parameterises it by the
flight heading psi, so we can ask:
  "What is the weakest wind shear the bird can exploit while flying in
   direction psi relative to the wind?"

Coordinate system (matches refactor/src/solver.py):
  x   crosswind direction
  y   upwind direction  (+y = directly into the wind; wind blows in −y)
  z   up  (altitude h = z)

  Wind shear: V_wy = V_ref*(h/h_ref)^p  (added to v component)
  Airspeed:   V_a  = sqrt(u² + (v + V_wy)² + w²)

psi convention — azimuth angle from +y (wind axis), rotating toward +x:
  psi = 0      → directly into the wind  (headwind, +y) — same as theta=0 in refactor
  psi = π/2    → crosswind (+x)
  psi = π      → directly downwind       (tailwind, −y)

Direction constraint for a given psi:
  The mean inertial velocity (u_avg, v_avg) is constrained to point in the
  direction (sin psi, cos psi) with at least V_GROUND_MIN ground speed.

By u → −u symmetry, V_ref(psi) = V_ref(2π − psi), so we compute 0 → π and
mirror for the full polar diagram.

Usage
-----
    cd /home/kristiancuervo/albatross/refactor
    python polar_sweep.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import casadi as cas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from contextlib import redirect_stdout, redirect_stderr

from src.albatross import Albatross

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N             = 128                             # collocation nodes
PSI_VALUES    = np.linspace(0, np.pi, 60)       # 0° … 180° in 22.5° steps
V_GROUND_MIN  = 0.5                            # min ground speed in psi direction [m/s]
OUTPUT        = os.path.join(os.path.dirname(__file__), "polar_sweep.png")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class MinVRefSolver:
    """
    Minimises V_ref for sustained dynamic soaring with mean flight direction
    fixed to angle psi (radians from the upwind axis, rotating toward crosswind).

    Aerodynamic model matches refactor/src/solver.py:
      - Wind blows in -y direction; shear profile V_wy(h) = V_ref*(h/h_ref)^p
      - Airspeed: V_a = sqrt(u^2 + (v + V_wy)^2 + w^2)
      - Trapezoidal collocation on periodic orbit
      - Objective: minimise V_ref
    """

    def __init__(self,
                 bird: Albatross,
                 N: int,
                 psi: float,
                 max_iters: int = 3000,
                 ics: dict | None = None,
                 V_ground_min: float = V_GROUND_MIN):
        self.bird         = bird
        self.N            = N
        self.psi          = psi
        self.max_iters    = max_iters
        self.ics          = ics
        self.V_ground_min = V_ground_min

        self.opti = cas.Opti()
        self._setup_variables()
        self._setup_aerodynamics()
        self._define_constraints()
        self._create_initial_condition()
        self.opti.minimize(self.V_ref)

    # ------------------------------------------------------------------
    def _setup_variables(self):
        self.dt    = self.opti.variable(1)
        self.h     = self.opti.variable(self.N)
        self.u     = self.opti.variable(self.N)
        self.v     = self.opti.variable(self.N)
        self.w     = self.opti.variable(self.N)
        self.mu    = self.opti.variable(self.N)
        self.cl    = self.opti.variable(self.N)
        self.V_ref = self.opti.variable(1)      # <-- the objective!

    # ------------------------------------------------------------------
    def _setup_aerodynamics(self):
        h_ref = 10
        p     = 0.143
        rho   = 1.225
        g     = 9.80665

        # Wind blows in -y; bird speed relative to air = (u, v + V_wy, w)
        V_wy      = self.V_ref * (self.h / h_ref) ** p
        self.V_a  = cas.sqrt(self.u**2 + (self.v + V_wy)**2 + self.w**2)

        gamma = cas.arctan2(-self.w, cas.sqrt(self.u**2 + (self.v + V_wy)**2))  # flight-path angle
        xi    = cas.arctan2(self.v + V_wy, self.u)                              # heading angle

        cd = self.bird.cd_0 + self.bird.k * self.cl**2
        L  = lambda Va: 0.5 * rho * Va**2 * self.bird.S * self.cl
        D  = lambda Va: 0.5 * rho * Va**2 * self.bird.S * cd

        a_u1 = cas.cos(gamma)*cas.cos(xi)
        a_u2 = cas.cos(self.mu)*cas.sin(gamma)*cas.cos(xi) + cas.sin(self.mu)*cas.sin(xi)
        a_v1 = cas.cos(gamma)*cas.sin(xi)
        a_v2 = cas.cos(self.mu)*cas.sin(gamma)*cas.sin(xi) - cas.sin(self.mu)*cas.cos(xi)
        a_w1 = -cas.sin(gamma)
        a_w2 = cas.cos(self.mu)*cas.cos(gamma)

        m = self.bird.m
        self.dudt = -a_u1*(D(self.V_a)/m) - a_u2*(L(self.V_a)/m)
        self.dvdt = -a_v1*(D(self.V_a)/m) - a_v2*(L(self.V_a)/m)
        self.dwdt = -a_w1*(D(self.V_a)/m) - a_w2*(L(self.V_a)/m) + g
        self.dhdt = -self.w

    # ------------------------------------------------------------------
    def _define_constraints(self):
        # --- Trapezoidal collocation (periodic wrap) ---
        du = cas.diff(cas.vertcat(self.u[-1], self.u))
        dv = cas.diff(cas.vertcat(self.v[-1], self.v))
        dw = cas.diff(cas.vertcat(self.w[-1], self.w))
        dh = cas.diff(cas.vertcat(self.h[-1], self.h))

        dudt_prev = cas.vertcat(self.dudt[-1], self.dudt[:-1])
        dvdt_prev = cas.vertcat(self.dvdt[-1], self.dvdt[:-1])
        dwdt_prev = cas.vertcat(self.dwdt[-1], self.dwdt[:-1])
        dhdt_prev = cas.vertcat(self.dhdt[-1], self.dhdt[:-1])

        self.opti.subject_to(du == 0.5 * (self.dudt + dudt_prev) * self.dt)
        self.opti.subject_to(dv == 0.5 * (self.dvdt + dvdt_prev) * self.dt)
        self.opti.subject_to(dw == 0.5 * (self.dwdt + dwdt_prev) * self.dt)
        self.opti.subject_to(dh == 0.5 * (self.dhdt + dhdt_prev) * self.dt)

        # --- Direction constraint ---
        # psi is an azimuth from +y (wind axis) rotating toward +x, so the
        # unit vector in the psi direction is (sin psi, cos psi) in (u, v).
        # Cross product d × v_avg = 0  →  direction aligned
        # Dot product   d · v_avg >= V_ground_min  →  positive forward speed
        cp = float(np.cos(self.psi))
        sp = float(np.sin(self.psi))
        u_sum = cas.sum1(self.u)
        v_sum = cas.sum1(self.v)

        # d × v_avg = sp*v_avg - cp*u_avg = 0
        self.opti.subject_to(v_sum * sp - u_sum * cp == 0)
        # d · v_avg = sp*u_avg + cp*v_avg >= min
        self.opti.subject_to(u_sum * sp + v_sum * cp >= self.N * self.V_ground_min)

        # --- Bounds (matching sachs/script.ipynb) ---
        self.opti.subject_to(self.h >= 0.5)
        self.opti.subject_to(self.h[0] == 0.5)

        self.opti.subject_to(self.cl >= 0.1)
        self.opti.subject_to(self.cl <= 1.5)

        self.opti.subject_to(-np.pi/2 < self.mu)
        self.opti.subject_to(self.mu < np.pi/2)

        self.opti.subject_to(self.V_a >= 5.0)

        self.opti.subject_to(self.dt >= 0.02)
        self.opti.subject_to(self.dt <= 1.0)

        self.T_cycle = self.dt * self.N
        self.opti.subject_to(self.T_cycle >= 5.0)
        self.opti.subject_to(self.T_cycle <= 15.0)

        self.opti.subject_to(self.V_ref >= 1.0)
        self.opti.subject_to(self.V_ref <= 30.0)

    # ------------------------------------------------------------------
    def _create_initial_condition(self):
        if self.ics is not None:
            ic = self.ics
            self.opti.set_initial(self.h,     ic['h'])
            self.opti.set_initial(self.u,     ic['u'])
            self.opti.set_initial(self.v,     ic['v'])
            self.opti.set_initial(self.w,     ic['w'])
            self.opti.set_initial(self.mu,    ic['mu'])
            self.opti.set_initial(self.cl,    ic['cl'])
            self.opti.set_initial(self.dt,    ic['dt'])
            self.opti.set_initial(self.V_ref, ic.get('V_ref', 10.0))
            return

        # Default: notebook-style ICs, velocity biased toward psi direction
        T  = 7.5
        tv = np.linspace(0, T, self.N, endpoint=False)
        l  = 2 * np.pi * tv / T

        h0  = 1.0 + 9.0 * (1 - np.cos(l))
        w0  = -9.0 * np.sin(l) * (2 * np.pi / T)
        mu0 = 0.7 * np.sin(l)
        cl0 = 0.8 + 0.3 * np.cos(l)

        # Mean velocity in the psi direction + orbital oscillation
        # Unit vector for azimuth psi: (sin psi, cos psi) in (u, v)
        spd = 8.0
        u0  = spd * np.sin(self.psi) + 5.0 * np.cos(l)
        v0  = spd * np.cos(self.psi) + 5.0 * np.sin(l)

        self.opti.set_initial(self.h,     h0)
        self.opti.set_initial(self.u,     u0)
        self.opti.set_initial(self.v,     v0)
        self.opti.set_initial(self.w,     w0)
        self.opti.set_initial(self.mu,    mu0)
        self.opti.set_initial(self.cl,    cl0)
        self.opti.set_initial(self.dt,    T / self.N)
        self.opti.set_initial(self.V_ref, 10.0)

    # ------------------------------------------------------------------
    def optimise(self) -> tuple[float | None, dict | None]:
        opts = {
            'ipopt.max_iter':    self.max_iters,
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.tol':         1e-6,
            'ipopt.print_level': 0,
        }
        self.opti.solver('ipopt', opts)
        deg = np.degrees(self.psi)
        print(f"  psi = {deg:6.1f}°  ...", end=" ", flush=True)
        try:
            with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
                sol = self.opti.solve()
            V_ref_opt = float(sol.value(self.V_ref))
            print(f"V_ref_min = {V_ref_opt:.3f} m/s")
            ics = {
                'h':     np.array(sol.value(self.h)),
                'u':     np.array(sol.value(self.u)),
                'v':     np.array(sol.value(self.v)),
                'w':     np.array(sol.value(self.w)),
                'mu':    np.array(sol.value(self.mu)),
                'cl':    np.array(sol.value(self.cl)),
                'dt':    float(sol.value(self.dt)),
                'V_ref': V_ref_opt,
            }
            return V_ref_opt, ics
        except RuntimeError as exc:
            print(f"FAILED  ({exc})")
            return None, None


# ---------------------------------------------------------------------------
# Polar sweep
# ---------------------------------------------------------------------------
def polar_sweep(bird: Albatross,
                psi_values: np.ndarray,
                N: int = N) -> tuple[np.ndarray, list[float | None]]:
    """
    Run MinVRefSolver for each psi in psi_values (radians, 0..π).
    Warm-starts each solve from the previous successful solution.
    Returns (psi_values, V_ref_list).
    """
    V_ref_results: list[float | None] = []
    warm_ics: dict | None = None

    for psi in psi_values:
        solver = MinVRefSolver(bird=bird, N=N, psi=psi, ics=warm_ics)
        V_ref_opt, ics = solver.optimise()
        V_ref_results.append(V_ref_opt)
        if ics is not None:
            warm_ics = ics   # carry forward only on success

    return psi_values, V_ref_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_polar(psi_half: np.ndarray,
               V_ref_half: list[float | None],
               output_path: str = OUTPUT) -> None:
    """
    Build a full polar diagram from the half-sweep (0..π) by applying the
    v → −v symmetry: V_ref(psi) = V_ref(2π − psi).
    """
    # Build full circle (0 .. 2π)
    psi_mirror = 2 * np.pi - psi_half[1:-1][::-1]   # π..0 reflected, skip duplicates at 0 and π
    V_ref_mirror = V_ref_half[1:-1][::-1]

    psi_full   = np.concatenate([psi_half,  psi_mirror])
    V_ref_full = V_ref_half + V_ref_mirror

    # Separate valid from failed
    valid_mask = np.array([v is not None for v in V_ref_full])
    psi_valid  = psi_full[valid_mask]
    V_valid    = np.array([v for v in V_ref_full if v is not None], dtype=float)

    # Closed curve: append first point
    psi_closed   = np.append(psi_valid,  psi_valid[0])
    V_closed     = np.append(V_valid, V_valid[0])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    # Main curve
    ax.plot(psi_closed, V_closed, "o-", color="C0", linewidth=2, markersize=6,
            label=r"$V_{\rm ref,\,min}(\psi)$")

    # Wind blows in −y (from top toward bottom in this plot).
    # Draw an arrow from psi=0 (headwind, top) toward psi=π (tailwind, bottom).
    r_max = max(v for v in V_valid)
    ax.annotate("", xy=(np.pi, r_max * 1.18), xytext=(0, r_max * 1.18),
                arrowprops=dict(arrowstyle="->", color="dodgerblue", lw=2.5))
    ax.text(np.pi, r_max * 1.27, "wind", ha="center", va="center",
            color="dodgerblue", fontsize=10)

    # Labels for cardinal directions
    for ang, lbl in [(0, "headwind\n(+y, θ=0°)"), (np.pi/2, "crosswind\n(+x)"),
                     (np.pi, "tailwind\n(−y)"),    (3*np.pi/2, "crosswind\n(−x)")]:
        ax.text(ang, r_max * 1.38, lbl, ha="center", va="center",
                fontsize=7.5, color="grey")

    ax.set_theta_zero_location("N")   # psi=0 at top = headwind (+y, theta=0)
    ax.set_theta_direction(-1)        # clockwise: psi=π/2 → +x crosswind (East)
    ax.set_title(r"Minimum $V_{\rm ref}$ for dynamic soaring vs flight heading $\psi$",
                 pad=25, fontsize=12)
    ax.legend(loc="lower left", bbox_to_anchor=(0.85, -0.05), fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    # Also print a summary table
    print(f"\n{'psi (deg)':>12}  {'V_ref_min (m/s)':>16}")
    print("-" * 32)
    for p, v in zip(psi_half, V_ref_half):
        status = f"{v:.3f}" if v is not None else "FAILED"
        print(f"{np.degrees(p):>12.1f}  {status:>16}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bird = Albatross.from_toml()

    print(f"Polar sweep: N={N}, {len(PSI_VALUES)} headings from "
          f"{np.degrees(PSI_VALUES[0]):.0f}° to {np.degrees(PSI_VALUES[-1]):.0f}°\n")

    psi_vals, V_ref_vals = polar_sweep(bird, PSI_VALUES, N=N)
    plot_polar(psi_vals, V_ref_vals, output_path=OUTPUT)
