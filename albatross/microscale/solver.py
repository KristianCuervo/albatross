"""
albatross.microscale.solver — CasADi Opti-based dynamic soaring optimiser.

Two solver modes
----------------
max_vmg  (default)
    V_ref is a fixed constructor parameter.
    Objective: maximise  sin(θ)·Σu + cos(θ)·Σv  (ground-speed along heading θ).
    Use for: tacking diagram VMG curves at each wind speed level.

min_vref
    V_ref is an optimisation variable (lower-bounded at 0).
    Objective: minimise V_ref.
    When theta is provided, optionally adds sin(θ)·ū + cos(θ)·v̄ ≥ 0 to ensure
    the cycle makes net progress in that direction (needed for polar V_ref_min curve).
    Use for: the 8.6 m/s Sachs minimum wind result; V_ref_min(θ) polar curve.

Discretisation schemes
----------------------
euler          : 1st-order forward Euler
trapezoidal    : 2nd-order Crank-Nicolson  (default)
hermite_simpson: 4th-order Hermite-Simpson
"""

from .container import Container

import casadi as cas
import numpy as np
import os
from contextlib import redirect_stdout, redirect_stderr
from ..bird import Albatross


class Solver:
    """
    Dynamic soaring cycle optimiser.

    Parameters
    ----------
    bird : Albatross
    theta : float
        Optimisation heading [rad].  θ=0 → upwind, θ=π/2 → crosswind.
        In max_vmg mode: objective direction.
        In min_vref mode: optional progress-direction constraint.
    N : int
        Number of collocation nodes.
    V_ref : float
        Reference wind speed at 10 m [m/s].
        max_vmg: fixed.  min_vref: used only as initial guess (ignored if ≤ 0).
    mode : str
        "max_vmg" or "min_vref".
    add_progress_constraint : bool
        Only relevant for min_vref mode.  When True (default), adds the constraint
        sin(θ)·ū + cos(θ)·v̄ ≥ 0  so the cycle makes progress in direction θ.
        Set False to replicate the original Sachs single-point result (θ=0, no
        directional constraint — the solver just finds the minimum wind to sustain
        any periodic DS cycle regardless of net direction).
    max_iters : int
    ics : dict or None
        Warm-start initial conditions (from Container.as_ics()).
    scheme : str
        "euler", "trapezoidal", or "hermite_simpson".
    h_min : float
        Minimum altitude constraint [m].  Default 0.1 m (refactor convention).
        Use 0.5 m to match the original Sachs notebook exactly.
    T_cycle_min : float
        Minimum cycle period [s].  Default 1.0 s.
        Use 5.0 s to match the original Sachs notebook and prevent degenerate
        fast-cycle local minima in min_vref mode.
    T_cycle_max : float
        Maximum cycle period [s].  Default 15.0 s.
    """

    def __init__(
        self,
        bird: Albatross,
        theta: float,
        N: int,
        V_ref: float = 10.0,
        mode: str = "max_vmg",
        add_progress_constraint: bool = True,
        max_iters: int = 2500,
        ics: dict | None = None,
        scheme: str = "trapezoidal",
        h_min: float = 0.1,
        T_cycle_min: float = 1.0,
        T_cycle_max: float = 15.0,
        reflective_bc: bool = False,
    ):
        if mode not in ("max_vmg", "min_vref"):
            raise ValueError(f"mode must be 'max_vmg' or 'min_vref', got {mode!r}")

        self.bird = bird
        self.theta = theta
        self.N = N
        self._V_ref_init = V_ref   # initial guess / fixed value
        self.mode = mode
        self.add_progress_constraint = add_progress_constraint
        self.max_iters = max_iters
        self.ics = ics
        self.scheme = scheme
        self.h_min = h_min
        self.T_cycle_min = T_cycle_min
        self.T_cycle_max = T_cycle_max
        self.reflective_bc = reflective_bc
        self.opti = cas.Opti()
        self._setup_variables()
        self._setup_aerodynamics()
        self._define_constraints()
        self._create_initial_condition()
        self._define_objective()

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    def _setup_variables(self):
        self.dt = self.opti.variable(1)

        # In min_vref mode V_ref becomes a decision variable
        if self.mode == "min_vref":
            self.V_ref = self.opti.variable(1)
            self.opti.subject_to(self.V_ref >= 0.0)
            self.opti.set_initial(self.V_ref, max(self._V_ref_init, 8.0))
        else:
            self.V_ref = self._V_ref_init   # plain float

        self.h  = self.opti.variable(self.N)
        self.u  = self.opti.variable(self.N)
        self.v  = self.opti.variable(self.N)
        self.w  = self.opti.variable(self.N)
        self.mu = self.opti.variable(self.N)
        self.cl = self.opti.variable(self.N)

    # ------------------------------------------------------------------
    # Aerodynamics (wind in y-direction only — refactor convention)
    # ------------------------------------------------------------------

    def _setup_aerodynamics(self):
        h_ref = 10.0
        p     = 0.143
        rho   = 1.225
        g     = 9.80665

        self.V_wy = self.V_ref * (self.h / h_ref) ** p

        self.V_a = cas.sqrt(self.u**2 + (self.v + self.V_wy)**2 + self.w**2)

        self.gamma = cas.arctan2(-self.w,
                                 cas.sqrt(self.u**2 + (self.v + self.V_wy)**2))
        self.xi    = cas.arctan2(self.v + self.V_wy, self.u)

        self.cd = self.bird.cd_0 + self.bird.k * self.cl**2

        def L(Va):
            return 0.5 * rho * Va**2 * self.bird.S * self.cl

        def D(Va):
            return 0.5 * rho * Va**2 * self.bird.S * self.cd

        a_u1 = cas.cos(self.gamma) * cas.cos(self.xi)
        a_u2 = (cas.cos(self.mu) * cas.sin(self.gamma) * cas.cos(self.xi)
                + cas.sin(self.mu) * cas.sin(self.xi))
        a_v1 = cas.cos(self.gamma) * cas.sin(self.xi)
        a_v2 = (cas.cos(self.mu) * cas.sin(self.gamma) * cas.sin(self.xi)
                - cas.sin(self.mu) * cas.cos(self.xi))
        a_w1 = -cas.sin(self.gamma)
        a_w2 = cas.cos(self.mu) * cas.cos(self.gamma)

        self.dudt = -a_u1 * (D(self.V_a) / self.bird.m) - a_u2 * (L(self.V_a) / self.bird.m)
        self.dvdt = -a_v1 * (D(self.V_a) / self.bird.m) - a_v2 * (L(self.V_a) / self.bird.m)
        self.dwdt = (-a_w1 * (D(self.V_a) / self.bird.m)
                     - a_w2 * (L(self.V_a) / self.bird.m) + g)
        self.dhdt = -self.w

    def _midpoint_aerodynamics(self, h_c, u_c, v_c, w_c, cl_c, mu_c):
        """Evaluate aerodynamic derivatives at symbolic midpoint states/controls."""
        h_ref = 10.0
        p     = 0.143
        rho   = 1.225
        g     = 9.80665

        V_wy_c = self.V_ref * (h_c / h_ref) ** p
        V_a_c  = cas.sqrt(u_c**2 + (v_c + V_wy_c)**2 + w_c**2)
        gamma_c = cas.arctan2(-w_c, cas.sqrt(u_c**2 + (v_c + V_wy_c)**2))
        xi_c    = cas.arctan2(v_c + V_wy_c, u_c)
        cd_c    = self.bird.cd_0 + self.bird.k * cl_c**2

        L_c = 0.5 * rho * V_a_c**2 * self.bird.S * cl_c
        D_c = 0.5 * rho * V_a_c**2 * self.bird.S * cd_c

        a_u1 = cas.cos(gamma_c) * cas.cos(xi_c)
        a_u2 = (cas.cos(mu_c) * cas.sin(gamma_c) * cas.cos(xi_c)
                + cas.sin(mu_c) * cas.sin(xi_c))
        a_v1 = cas.cos(gamma_c) * cas.sin(xi_c)
        a_v2 = (cas.cos(mu_c) * cas.sin(gamma_c) * cas.sin(xi_c)
                - cas.sin(mu_c) * cas.cos(xi_c))
        a_w1 = -cas.sin(gamma_c)
        a_w2 = cas.cos(mu_c) * cas.cos(gamma_c)

        dudt_c = -a_u1 * (D_c / self.bird.m) - a_u2 * (L_c / self.bird.m)
        dvdt_c = -a_v1 * (D_c / self.bird.m) - a_v2 * (L_c / self.bird.m)
        dwdt_c = -a_w1 * (D_c / self.bird.m) - a_w2 * (L_c / self.bird.m) + g
        dhdt_c = -w_c

        return dudt_c, dvdt_c, dwdt_c, dhdt_c

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _define_constraints(self):
        if self.reflective_bc:
            if self.scheme == "euler":
                self._apply_euler_open()
            elif self.scheme == "trapezoidal":
                self._apply_trapezoidal_open()
            elif self.scheme == "hermite_simpson":
                self._apply_hermite_simpson_open()
            else:
                raise ValueError(f"Unknown scheme: {self.scheme!r}")
            self._apply_bounds()
            self._apply_reflective_bcs()
        else:
            if self.scheme == "euler":
                self._apply_euler()
            elif self.scheme == "trapezoidal":
                self._apply_trapezoidal()
            elif self.scheme == "hermite_simpson":
                self._apply_hermite_simpson()
            else:
                raise ValueError(f"Unknown scheme: {self.scheme!r}")
            self._apply_bounds()

    def _apply_euler(self):
        du = cas.diff(cas.vertcat(self.u[-1], self.u))
        dv = cas.diff(cas.vertcat(self.v[-1], self.v))
        dw = cas.diff(cas.vertcat(self.w[-1], self.w))
        dh = cas.diff(cas.vertcat(self.h[-1], self.h))

        dudt_fwd = cas.vertcat(self.dudt[-1], self.dudt[:-1])
        dvdt_fwd = cas.vertcat(self.dvdt[-1], self.dvdt[:-1])
        dwdt_fwd = cas.vertcat(self.dwdt[-1], self.dwdt[:-1])
        dhdt_fwd = cas.vertcat(self.dhdt[-1], self.dhdt[:-1])

        self.opti.subject_to(du == dudt_fwd * self.dt)
        self.opti.subject_to(dv == dvdt_fwd * self.dt)
        self.opti.subject_to(dw == dwdt_fwd * self.dt)
        self.opti.subject_to(dh == dhdt_fwd * self.dt)

    def _apply_trapezoidal(self):
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

    def _apply_hermite_simpson(self):
        dudt_prev = cas.vertcat(self.dudt[-1], self.dudt[:-1])
        dvdt_prev = cas.vertcat(self.dvdt[-1], self.dvdt[:-1])
        dwdt_prev = cas.vertcat(self.dwdt[-1], self.dwdt[:-1])
        dhdt_prev = cas.vertcat(self.dhdt[-1], self.dhdt[:-1])

        u_prev = cas.vertcat(self.u[-1], self.u[:-1])
        v_prev = cas.vertcat(self.v[-1], self.v[:-1])
        w_prev = cas.vertcat(self.w[-1], self.w[:-1])
        h_prev = cas.vertcat(self.h[-1], self.h[:-1])

        h_c = 0.5*(h_prev + self.h)  + self.dt/8 * (dhdt_prev - self.dhdt)
        u_c = 0.5*(u_prev + self.u)  + self.dt/8 * (dudt_prev - self.dudt)
        v_c = 0.5*(v_prev + self.v)  + self.dt/8 * (dvdt_prev - self.dvdt)
        w_c = 0.5*(w_prev + self.w)  + self.dt/8 * (dwdt_prev - self.dwdt)

        cl_c = 0.5*(cas.vertcat(self.cl[-1], self.cl[:-1]) + self.cl)
        mu_c = 0.5*(cas.vertcat(self.mu[-1], self.mu[:-1]) + self.mu)

        dudt_c, dvdt_c, dwdt_c, dhdt_c = self._midpoint_aerodynamics(
            h_c, u_c, v_c, w_c, cl_c, mu_c
        )

        du = cas.diff(cas.vertcat(self.u[-1], self.u))
        dv = cas.diff(cas.vertcat(self.v[-1], self.v))
        dw = cas.diff(cas.vertcat(self.w[-1], self.w))
        dh = cas.diff(cas.vertcat(self.h[-1], self.h))

        self.opti.subject_to(du == self.dt/6 * (dudt_prev + 4*dudt_c + self.dudt))
        self.opti.subject_to(dv == self.dt/6 * (dvdt_prev + 4*dvdt_c + self.dvdt))
        self.opti.subject_to(dw == self.dt/6 * (dwdt_prev + 4*dwdt_c + self.dwdt))
        self.opti.subject_to(dh == self.dt/6 * (dhdt_prev + 4*dhdt_c + self.dhdt))

    # ------------------------------------------------------------------
    # Open-trajectory variants (for reflective_bc mode)
    # Dynamics connect node i → i+1 for i = 0..N-2 (N-1 constraints).
    # The wrap-around step is replaced by explicit reflective BCs.
    # ------------------------------------------------------------------

    def _apply_euler_open(self):
        du = cas.diff(self.u)
        dv = cas.diff(self.v)
        dw = cas.diff(self.w)
        dh = cas.diff(self.h)
        self.opti.subject_to(du == self.dudt[:-1] * self.dt)
        self.opti.subject_to(dv == self.dvdt[:-1] * self.dt)
        self.opti.subject_to(dw == self.dwdt[:-1] * self.dt)
        self.opti.subject_to(dh == self.dhdt[:-1] * self.dt)

    def _apply_trapezoidal_open(self):
        du = cas.diff(self.u)
        dv = cas.diff(self.v)
        dw = cas.diff(self.w)
        dh = cas.diff(self.h)
        self.opti.subject_to(du == 0.5 * (self.dudt[1:] + self.dudt[:-1]) * self.dt)
        self.opti.subject_to(dv == 0.5 * (self.dvdt[1:] + self.dvdt[:-1]) * self.dt)
        self.opti.subject_to(dw == 0.5 * (self.dwdt[1:] + self.dwdt[:-1]) * self.dt)
        self.opti.subject_to(dh == 0.5 * (self.dhdt[1:] + self.dhdt[:-1]) * self.dt)

    def _apply_hermite_simpson_open(self):
        u_p = self.u[:-1]; u_n = self.u[1:]
        v_p = self.v[:-1]; v_n = self.v[1:]
        w_p = self.w[:-1]; w_n = self.w[1:]
        h_p = self.h[:-1]; h_n = self.h[1:]
        dudt_p = self.dudt[:-1]; dudt_n = self.dudt[1:]
        dvdt_p = self.dvdt[:-1]; dvdt_n = self.dvdt[1:]
        dwdt_p = self.dwdt[:-1]; dwdt_n = self.dwdt[1:]
        dhdt_p = self.dhdt[:-1]; dhdt_n = self.dhdt[1:]

        h_c = 0.5*(h_p + h_n) + self.dt/8 * (dhdt_p - dhdt_n)
        u_c = 0.5*(u_p + u_n) + self.dt/8 * (dudt_p - dudt_n)
        v_c = 0.5*(v_p + v_n) + self.dt/8 * (dvdt_p - dvdt_n)
        w_c = 0.5*(w_p + w_n) + self.dt/8 * (dwdt_p - dwdt_n)
        cl_c = 0.5*(self.cl[:-1] + self.cl[1:])
        mu_c = 0.5*(self.mu[:-1] + self.mu[1:])

        dudt_c, dvdt_c, dwdt_c, dhdt_c = self._midpoint_aerodynamics(
            h_c, u_c, v_c, w_c, cl_c, mu_c
        )
        du = u_n - u_p
        dv = v_n - v_p
        dw = w_n - w_p
        dh = h_n - h_p
        self.opti.subject_to(du == self.dt/6 * (dudt_p + 4*dudt_c + dudt_n))
        self.opti.subject_to(dv == self.dt/6 * (dvdt_p + 4*dvdt_c + dvdt_n))
        self.opti.subject_to(dw == self.dt/6 * (dwdt_p + 4*dwdt_c + dwdt_n))
        self.opti.subject_to(dh == self.dt/6 * (dhdt_p + 4*dhdt_c + dhdt_n))

    def _apply_reflective_bcs(self):
        """
        Reflective tacking boundary condition.

        The trajectory represents one half-cycle.  At the tack the bird
        reverses crosswind velocity u while h, v, w are continuous:

            u[N-1] = -u[0]   (crosswind reflects)
            v[N-1] =  v[0]   (upwind speed preserved)
            w[N-1] =  w[0]   (vertical velocity preserved)
            h[N-1] =  h[0]   (altitude preserved)
        """
        self.opti.subject_to(self.u[-1] == -self.u[0])
        self.opti.subject_to(self.v[-1] ==  self.v[0])
        self.opti.subject_to(self.w[-1] ==  self.w[0])
        self.opti.subject_to(self.h[-1] ==  self.h[0])

    def _apply_bounds(self):
        self.opti.subject_to(self.h >= self.h_min)
        self.opti.subject_to(self.h <= 300)
        self.opti.subject_to(self.h[0] == self.h_min)

        self.opti.subject_to(self.cl >= 0.1)
        self.opti.subject_to(self.cl <= 1.5)

        self.opti.subject_to(-np.pi/2 < self.mu)
        self.opti.subject_to(self.mu < np.pi/2)

        self.opti.subject_to(self.V_a >= 5.0)

        self.opti.subject_to(self.dt >= 0.01)
        self.opti.subject_to(self.dt <= 1.0)

        self.T_cycle = self.dt * self.N
        self.opti.subject_to(self.T_cycle >= self.T_cycle_min)
        self.opti.subject_to(self.T_cycle <= self.T_cycle_max)

        # min_vref: optional directional progress constraint
        if self.mode == "min_vref" and self.add_progress_constraint:
            cx = np.sin(self.theta)
            cy = np.cos(self.theta)
            u_mean = cas.sum1(self.u) / self.N
            v_mean = cas.sum1(self.v) / self.N
            self.opti.subject_to(cx * u_mean + cy * v_mean >= 0.0)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def _create_initial_condition(self):
        if self.ics is not None:
            ic = self.ics
            self.opti.set_initial(self.h,  ic['h'])
            self.opti.set_initial(self.u,  ic['u'])
            self.opti.set_initial(self.v,  ic['v'])
            self.opti.set_initial(self.w,  ic['w'])
            self.opti.set_initial(self.mu, ic['mu'])
            self.opti.set_initial(self.cl, ic['cl'])
            self.opti.set_initial(self.dt, ic['dt'])
            self._ic = ic
            return

        cl_opt = np.sqrt(self.bird.cd_0 / self.bird.k)

        T  = max(5.0, self.T_cycle_min)
        tv = np.linspace(0, T, self.N, endpoint=False)

        h_low  = self.h_min
        h_high = max(20.0, h_low + 10.0)

        if self.reflective_bc:
            # Half-period: goes from trough → peak → trough.
            # Satisfies the reflective BCs at the IC level:
            #   h[0] = h[-1] = h_low,  w[0] = w[-1] = 0
            #   u[-1] = -u[0]  (cosine: A→-A over half period)
            lh    = np.pi * tv / T          # 0 → π
            h_ic  = h_low + (h_high - h_low) * np.sin(lh)**2
            w_ic  = -(h_high - h_low) * (np.pi / T) * np.sin(2 * lh)
        else:
            l    = 2 * np.pi * tv / T
            h_ic = h_low + (h_high - h_low) * (1 - np.cos(l)) / 2
            w_ic = -(h_high - h_low) * np.pi / T * np.sin(l)

        # Physical reflection symmetry: θ ∈ (π, 2π) mirrors θ_refl = 2π − θ ∈ (0, π)
        # under (u, mu) → (−u, −mu); v, w, h unchanged.
        theta_ic    = self.theta if self.theta <= np.pi else 2 * np.pi - self.theta
        flip        =  1.0 if self.theta <= np.pi else -1.0
        flipquarter = -1.0 if np.pi/2 <= self.theta <= 3*np.pi/2 else 1.0

        if self.reflective_bc:
            u0  = flip        * 10 * np.cos(lh)
            v0  = flipquarter * (5 + 10 * np.sin(theta_ic) * np.sin(lh))
            mu0 = flip * np.pi / 4 * np.cos(lh)
        else:
            u0  = flip        * (10 + 10 * np.cos(theta_ic) * np.sin(l))
            v0  = flipquarter * (5  + 10 * np.sin(theta_ic) * np.sin(l))
            mu0 = flip * np.pi / 4 * np.sin(l)
        cl0 = cl_opt * np.ones(self.N)

        self.opti.set_initial(self.h,  h_ic)
        self.opti.set_initial(self.u,  u0)
        self.opti.set_initial(self.v,  v0)
        self.opti.set_initial(self.w,  w_ic)
        self.opti.set_initial(self.mu, mu0)
        self.opti.set_initial(self.cl, cl0)
        self.opti.set_initial(self.dt, T / self.N)
        self._ic = {'h': h_ic, 'u': u0, 'v': v0, 'w': w_ic,
                    'mu': mu0, 'cl': cl0, 'dt': T / self.N}

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _define_objective(self):
        if self.mode == "max_vmg":
            cx = np.sin(self.theta)
            cy = np.cos(self.theta)
            self.obj = (cx * cas.sum1(self.u) + cy * cas.sum1(self.v)) / self.N
            self.opti.minimize(-self.obj)
        else:
            # min_vref: minimise V_ref; obj stored as the VMG for Container compat
            cx = np.sin(self.theta)
            cy = np.cos(self.theta)
            self.obj = (cx * cas.sum1(self.u) + cy * cas.sum1(self.v)) / self.N
            self.opti.minimize(self.V_ref)

    # ------------------------------------------------------------------
    # Solution extraction
    # ------------------------------------------------------------------

    def _get_solution_dict(self, sol) -> dict:
        V_ref_val = (float(sol.value(self.V_ref))
                     if self.mode == "min_vref"
                     else float(self._V_ref_init))
        return {
            'dt':          float(sol.value(self.dt)),
            'h':           np.array(sol.value(self.h)),
            'u':           np.array(sol.value(self.u)),
            'v':           np.array(sol.value(self.v)),
            'w':           np.array(sol.value(self.w)),
            'mu':          np.array(sol.value(self.mu)),
            'cl':          np.array(sol.value(self.cl)),
            'theta':       self.theta,
            'V_ref':       V_ref_val,
            'N':           self.N,
            'obj':         float(sol.value(self.obj)),
            'T_cycle':      float(sol.value(self.T_cycle)),
            'scheme':       self.scheme,
            'mode':         self.mode,
            'reflective_bc': self.reflective_bc,
        }

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def optimise(self) -> Container:
        opts = {
            'ipopt.max_iter':    self.max_iters,
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.tol':         1e-6,
            'ipopt.print_level': 5,
        }
        self.opti.solver('ipopt', opts)
        print(f"Optimising: theta={self.theta:.4f}  V_ref={self._V_ref_init}  mode={self.mode}")
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            sol = self.opti.solve()
        return Container.from_solution_dict(self._get_solution_dict(sol))
