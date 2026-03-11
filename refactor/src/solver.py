from .container import Container

import casadi as cas
import numpy as np
import os
from contextlib import redirect_stdout, redirect_stderr
from .albatross import Albatross

class Solver:
    """
    Uses the simulation data to then optimise a run and create a container for the solution.
    """

    def __init__(self,
                 bird: Albatross,
                 theta: float,
                 N: int,
                 V_ref: float,
                 max_iters: int = 2500,
                 ics: dict | None = None,
                 scheme: str = "trapezoidal"):
            self.bird = bird
            self.theta = theta
            self.N = N
            self.V_ref = V_ref
            self.max_iters = max_iters
            self.ics = ics
            self.scheme = scheme   # "euler" | "trapezoidal" | "hermite_simpson"
            self.opti = cas.Opti()
            self._setup_variables()
            self._setup_aerodynamics()
            self._define_constraints()
            self._create_initial_condition()
            self._define_objective()

    def _setup_variables(self):
        self.dt = self.opti.variable(1)

        # Altitude
        self.h = self.opti.variable(self.N)

        # Inertial (ground-frame) velocities
        self.u = self.opti.variable(self.N)
        self.v = self.opti.variable(self.N)
        self.w = self.opti.variable(self.N)

        # Control variables
        self.mu = self.opti.variable(self.N)
        self.cl = self.opti.variable(self.N)

    def _setup_aerodynamics(self):
        h_ref = 10
        p = 0.143
        rho = 1.225
        g = 9.80665

        # Polar wind speeds
        self.V_wy = self.V_ref * (self.h/h_ref)**p

        # Airspeed - now 2d impact on wind direction
        self.V_a = cas.sqrt((self.u)**2 + (self.v + self.V_wy)**2 + self.w**2)

        # Resulting flight angles
        self.gamma = cas.arctan2(-self.w, cas.sqrt(self.u**2 + (self.v + self.V_wy)**2))  # flight path angle
        self.xi = cas.arctan2(self.v + self.V_wy, self.u)    # heading angle

        # Drag characteristics
        self.cd = self.bird.cd_0 + self.bird.k * self.cl**2

        # Aerodynamic forces
        L = lambda V_a : 0.5 * rho * V_a**2 * self.bird.S * self.cl
        D = lambda V_a : 0.5 * rho * V_a**2 * self.bird.S * self.cd
        a_u1 = cas.cos(self.gamma)*cas.cos(self.xi)
        a_u2 = cas.cos(self.mu)*cas.sin(self.gamma)*cas.cos(self.xi) + cas.sin(self.mu)*cas.sin(self.xi)
        a_v1 = cas.cos(self.gamma)*cas.sin(self.xi)
        a_v2 = cas.cos(self.mu)*cas.sin(self.gamma)*cas.sin(self.xi) - cas.sin(self.mu)*cas.cos(self.xi)
        a_w1 = -cas.sin(self.gamma)
        a_w2 = cas.cos(self.mu)*cas.cos(self.gamma)

        # Equations of motion (inertial frame — wind enters via V_a, gamma, xi)

        self.dudt = -a_u1*(D(self.V_a)/self.bird.m) - a_u2*(L(self.V_a)/self.bird.m)
        self.dvdt = -a_v1*(D(self.V_a)/self.bird.m) - a_v2*(L(self.V_a)/self.bird.m)
        self.dwdt = -a_w1*(D(self.V_a)/self.bird.m) - a_w2*(L(self.V_a)/self.bird.m) + g

        self.dhdt = -self.w

    def _midpoint_aerodynamics(self, h_c, u_c, v_c, w_c, cl_c, mu_c):
        """Evaluate aerodynamic derivatives at symbolic midpoint states/controls."""
        h_ref = 10
        p = 0.143
        rho = 1.225
        g = 9.80665

        V_wy_c = self.V_ref * (h_c / h_ref) ** p
        V_a_c = cas.sqrt(u_c**2 + (v_c + V_wy_c)**2 + w_c**2)
        gamma_c = cas.arctan2(-w_c, cas.sqrt(u_c**2 + (v_c + V_wy_c)**2))
        xi_c = cas.arctan2(v_c + V_wy_c, u_c)
        cd_c = self.bird.cd_0 + self.bird.k * cl_c**2

        L_c = 0.5 * rho * V_a_c**2 * self.bird.S * cl_c
        D_c = 0.5 * rho * V_a_c**2 * self.bird.S * cd_c

        a_u1 = cas.cos(gamma_c)*cas.cos(xi_c)
        a_u2 = cas.cos(mu_c)*cas.sin(gamma_c)*cas.cos(xi_c) + cas.sin(mu_c)*cas.sin(xi_c)
        a_v1 = cas.cos(gamma_c)*cas.sin(xi_c)
        a_v2 = cas.cos(mu_c)*cas.sin(gamma_c)*cas.sin(xi_c) - cas.sin(mu_c)*cas.cos(xi_c)
        a_w1 = -cas.sin(gamma_c)
        a_w2 = cas.cos(mu_c)*cas.cos(gamma_c)

        dudt_c = -a_u1*(D_c/self.bird.m) - a_u2*(L_c/self.bird.m)
        dvdt_c = -a_v1*(D_c/self.bird.m) - a_v2*(L_c/self.bird.m)
        dwdt_c = -a_w1*(D_c/self.bird.m) - a_w2*(L_c/self.bird.m) + g
        dhdt_c = -w_c

        return dudt_c, dvdt_c, dwdt_c, dhdt_c

    def _define_constraints(self):
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
        """1st-order forward Euler: state[i] - state[i-1] = f[i-1] * dt."""
        du = cas.diff(cas.vertcat(self.u[-1], self.u))
        dv = cas.diff(cas.vertcat(self.v[-1], self.v))
        dw = cas.diff(cas.vertcat(self.w[-1], self.w))
        dh = cas.diff(cas.vertcat(self.h[-1], self.h))

        # Forward difference: use derivative at i-1
        dudt_fwd = cas.vertcat(self.dudt[-1], self.dudt[:-1])
        dvdt_fwd = cas.vertcat(self.dvdt[-1], self.dvdt[:-1])
        dwdt_fwd = cas.vertcat(self.dwdt[-1], self.dwdt[:-1])
        dhdt_fwd = cas.vertcat(self.dhdt[-1], self.dhdt[:-1])

        self.opti.subject_to(du == dudt_fwd * self.dt)
        self.opti.subject_to(dv == dvdt_fwd * self.dt)
        self.opti.subject_to(dw == dwdt_fwd * self.dt)
        self.opti.subject_to(dh == dhdt_fwd * self.dt)

    def _apply_trapezoidal(self):
        """2nd-order trapezoidal (Crank-Nicolson): average of derivatives at endpoints."""
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
        """4th-order Hermite-Simpson composite rule."""
        # Derivatives at i-1 (periodic wrap)
        dudt_prev = cas.vertcat(self.dudt[-1], self.dudt[:-1])
        dvdt_prev = cas.vertcat(self.dvdt[-1], self.dvdt[:-1])
        dwdt_prev = cas.vertcat(self.dwdt[-1], self.dwdt[:-1])
        dhdt_prev = cas.vertcat(self.dhdt[-1], self.dhdt[:-1])

        u_prev = cas.vertcat(self.u[-1], self.u[:-1])
        v_prev = cas.vertcat(self.v[-1], self.v[:-1])
        w_prev = cas.vertcat(self.w[-1], self.w[:-1])
        h_prev = cas.vertcat(self.h[-1], self.h[:-1])

        # Hermite midpoint states (algebraic — no new decision variables)
        h_c  = 0.5*(h_prev + self.h)  + self.dt/8 * (dhdt_prev - self.dhdt)
        u_c  = 0.5*(u_prev + self.u)  + self.dt/8 * (dudt_prev - self.dudt)
        v_c  = 0.5*(v_prev + self.v)  + self.dt/8 * (dvdt_prev - self.dvdt)
        w_c  = 0.5*(w_prev + self.w)  + self.dt/8 * (dwdt_prev - self.dwdt)

        # Controls at midpoints: simple average
        cl_c = 0.5*(cas.vertcat(self.cl[-1], self.cl[:-1]) + self.cl)
        mu_c = 0.5*(cas.vertcat(self.mu[-1], self.mu[:-1]) + self.mu)

        # Evaluate dynamics at midpoints
        dudt_c, dvdt_c, dwdt_c, dhdt_c = self._midpoint_aerodynamics(
            h_c, u_c, v_c, w_c, cl_c, mu_c
        )

        # Finite differences (periodic)
        du = cas.diff(cas.vertcat(self.u[-1], self.u))
        dv = cas.diff(cas.vertcat(self.v[-1], self.v))
        dw = cas.diff(cas.vertcat(self.w[-1], self.w))
        dh = cas.diff(cas.vertcat(self.h[-1], self.h))

        # Simpson defect: dt/6 * (f[i-1] + 4*f_c + f[i])
        self.opti.subject_to(du == self.dt/6 * (dudt_prev + 4*dudt_c + self.dudt))
        self.opti.subject_to(dv == self.dt/6 * (dvdt_prev + 4*dvdt_c + self.dvdt))
        self.opti.subject_to(dw == self.dt/6 * (dwdt_prev + 4*dwdt_c + self.dwdt))
        self.opti.subject_to(dh == self.dt/6 * (dhdt_prev + 4*dhdt_c + self.dhdt))

    def _apply_bounds(self):
        # Bounds
        ## Altitude
        self.opti.subject_to(self.h >= 0.1)
        self.opti.subject_to(self.h <= 300)
        self.opti.subject_to(self.h[0] == 0.1)

        ## Control bounds
        ### Lift coefficient
        self.opti.subject_to(self.cl >= 0.1)
        self.opti.subject_to(self.cl <= 1.5)

        ### Bank angle
        self.opti.subject_to(-np.pi/2 < self.mu)
        self.opti.subject_to(self.mu < np.pi/2)

        ## Airspeed
        self.opti.subject_to(self.V_a >= 5.0)

        ## Time step
        self.opti.subject_to(self.dt >= 0.01)
        self.opti.subject_to(self.dt <= 1.0)

        ## Total cycle period
        self.T_cycle = self.dt * self.N
        self.opti.subject_to(self.T_cycle >= 1.0)
        self.opti.subject_to(self.T_cycle <= 15.0)

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

        # Aerodynamically optimal quantities for this bird
        cl_opt = np.sqrt(self.bird.cd_0 / self.bird.k)

        T  = 5.0
        tv = np.linspace(0, T, self.N, endpoint=False)
        l  = 2 * np.pi * tv / T

        # Altitude: realistic dynamic soaring range
        h_low, h_high = 0.5, 20.0
        h0 = h_low + (h_high - h_low) * (1 - np.cos(l)) / 2

        # Vertical velocity consistent with altitude rate (dh/dt = -w)
        w0 = -(h_high - h_low) * np.pi / T * np.sin(l)

        # Physical reflection symmetry: θ ∈ (π, 2π) mirrors θ_refl = 2π − θ ∈ (0, π)
        # under (u, mu) → (−u, −mu); v, w, h are unchanged.
        # Build the IC in the first-hemisphere frame then apply the flip.
        theta_ic = self.theta if self.theta <= np.pi else 2 * np.pi - self.theta
        flip     = 1.0   if self.theta <= np.pi else -1.0
        flipquarter = -1.0 if np.pi/2 <= self.theta <= 3*np.pi/2 else 1.0

        u0  = flip * (10 + 10 * np.cos(theta_ic) * np.sin(l))
        v0  = flipquarter * (5 + 10 * np.sin(theta_ic) * np.sin(l))
        mu0 = flip * np.pi / 4 * np.sin(l)

        self.opti.set_initial(self.h,  h0)
        self.opti.set_initial(self.u,  u0)
        self.opti.set_initial(self.v,  v0)
        self.opti.set_initial(self.w,  w0)
        self.opti.set_initial(self.mu, mu0)
        self.opti.set_initial(self.cl, cl_opt * np.ones(self.N))
        self.opti.set_initial(self.dt, T / self.N)
        self._ic = {'h': h0, 'u': u0, 'v': v0, 'w': w0,
                    'mu': mu0, 'cl': cl_opt * np.ones(self.N), 'dt': T / self.N}

    def _define_objective(self):
        cx = np.sin(self.theta)
        cy = np.cos(self.theta)

        self.obj = (cx * cas.sum1(self.u) + cy * cas.sum1(self.v))/self.N
        self.opti.minimize(-self.obj)

    def _get_solution_dict(self, sol) -> dict:
        return {
            'dt':      float(sol.value(self.dt)),
            'h':       np.array(sol.value(self.h)),
            'u':       np.array(sol.value(self.u)),
            'v':       np.array(sol.value(self.v)),
            'w':       np.array(sol.value(self.w)),
            'mu':      np.array(sol.value(self.mu)),
            'cl':      np.array(sol.value(self.cl)),
            'theta':   self.theta,
            'V_ref':   self.V_ref,
            'N':       self.N,
            'obj':     float(sol.value(self.obj)),
            'T_cycle': float(sol.value(self.T_cycle)),
            'scheme':  self.scheme,
        }

    def optimise(self) -> Container:
        opts = {
                'ipopt.max_iter': self.max_iters,
                'ipopt.mu_strategy': 'adaptive',
                'ipopt.tol': 1e-6,
                'ipopt.print_level': 5,
                }
        self.opti.solver('ipopt', opts)
        print(f"Optimising: theta = {self.theta}, v_ref = {self.V_ref}")
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            sol = self.opti.solve()
        return Container.from_solution_dict(self._get_solution_dict(sol))
