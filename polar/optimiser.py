import casadi as cas
import numpy as np
import tomllib
from pathlib import Path
from multiprocessing import Pool
from albatross import Albatross
from trajectories import PathFamily

class Optimiser:
    def __init__(self, bird: Albatross, theta: float, N: int, V_ref: float,
                 initial_conditions: dict | None = None):
        self.max_iters = 1000
        self.bird = bird
        self.theta = theta
        self.N = N
        self.V_ref = V_ref
        self.initial_conditions = initial_conditions
        self.opti = cas.Opti()
        self._setup_variables()
        self._setup_aerodynamics()
        self._define_constraints()
        self._create_initial_condition()

        #v_avg = cas.sum1(self.v) * self.dt / self.T_cycle
        self.v_tot = cas.sum1(self.v)
        self.v_avg = self.v_tot / self.N
        self.opti.minimize(-self.v_avg)

    def _setup_variables(self):
        N = self.N

        self.dt = self.opti.variable(1)

        # Altitude
        self.h = self.opti.variable(N)

        # Inertial (ground-frame) velocities
        self.u = self.opti.variable(N)
        self.v = self.opti.variable(N)
        self.w = self.opti.variable(N)

        # Control variables
        self.mu = self.opti.variable(N)
        self.cl = self.opti.variable(N)

    def _setup_aerodynamics(self):
        h_ref = 10
        p = 0.143
        rho = 1.225 # Sea-level pressure
        g = 9.80665
        # Polar wind speeds
        self.V_wx = np.sin(self.theta)*self.V_ref * (self.h/h_ref)**p
        self.V_wy = np.cos(self.theta)*self.V_ref * (self.h/h_ref)**p

        # Airspeed - now 2d impact on wind direction
        self.V_a = cas.sqrt((self.u + self.V_wx)**2 + (self.v + self.V_wy)**2 + self.w**2)

        # Resulting flight angles
        self.gamma = cas.arcsin(-self.w/self.V_a)           # flight path angle
        self.xi = cas.arctan2(self.v + self.V_wy, self.u + self.V_wx)    # heading angle

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


    def _define_constraints(self):
        # Periodic boundary conditions

        # Periodic finite differences (wrap last → first)
        du = cas.diff(cas.vertcat(self.u[-1], self.u))
        dv = cas.diff(cas.vertcat(self.v[-1], self.v))
        dw = cas.diff(cas.vertcat(self.w[-1], self.w))
        dh = cas.diff(cas.vertcat(self.h[-1], self.h))

        # Shifted derivatives for trapezoidal collocation
        dudt_prev = cas.vertcat(self.dudt[-1], self.dudt[:-1])
        dvdt_prev = cas.vertcat(self.dvdt[-1], self.dvdt[:-1])
        dwdt_prev = cas.vertcat(self.dwdt[-1], self.dwdt[:-1])
        dhdt_prev = cas.vertcat(self.dhdt[-1], self.dhdt[:-1])

        # Trapezoidal dynamics constraints (2nd-order accurate)
        self.opti.subject_to(du == 0.5 * (self.dudt + dudt_prev) * self.dt)
        self.opti.subject_to(dv == 0.5 * (self.dvdt + dvdt_prev) * self.dt)
        self.opti.subject_to(dw == 0.5 * (self.dwdt + dwdt_prev) * self.dt)
        self.opti.subject_to(dh == 0.5 * (self.dhdt + dhdt_prev) * self.dt)

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

        #self.opti.subject_to(-np.pi < self.mu)
        #self.opti.subject_to(self.mu < np.pi)
        ## Airspeed
        self.opti.subject_to(self.V_a >= 5.0)       # stall speed [m/s]

        ## Time step
        self.opti.subject_to(self.dt >= 0.01)
        self.opti.subject_to(self.dt <= 1.0)

        ## Total cycle period — prevent degenerate zero-period solution
        self.T_cycle = self.dt * self.N
        self.opti.subject_to(self.T_cycle >= 1.0)
        self.opti.subject_to(self.T_cycle <= 30.0)


    def _create_initial_condition(self):
        if self.initial_conditions is not None:
            ic = self.initial_conditions
            self.opti.set_initial(self.h,  ic['h'])
            self.opti.set_initial(self.u,  ic['u'])
            self.opti.set_initial(self.v,  ic['v'])
            self.opti.set_initial(self.w,  ic['w'])
            self.opti.set_initial(self.mu, ic['mu'])
            self.opti.set_initial(self.cl, ic['cl'])
            self.opti.set_initial(self.dt, ic['dt'])
            return

        T = 7.5
        tv = np.linspace(0, T, self.N)
        l = 2 * np.pi * tv / T

        h0 = 1 + 9 * (1 - np.cos(l))
        u0 =  -5 * np.cos(l) * np.sin(self.theta)
        v0 =  5 * np.cos(l)
        w0 = -9 * np.sin(l) * (2 * np.pi / T)

        self.opti.set_initial(self.h, h0)
        self.opti.set_initial(self.u, u0)
        self.opti.set_initial(self.v, v0)
        self.opti.set_initial(self.w, w0)


        #self.opti.set_initial(self.mu, 0.7 * np.sin(l)*np.sin(self.theta))

        if self.theta < np.pi:
            self.opti.set_initial(self.mu, 0.7 * np.sin(l))
        else:
            self.opti.set_initial(self.mu, -0.7 * np.sin(l))
        self.opti.set_initial(self.cl, 0.8 + 0.3 * np.cos(l))

        self.opti.set_initial(self.dt, T / self.N)

    def get_solution_dict(self) -> dict:
        return {
            'dt':      float(self.sol.value(self.dt)),
            'h':       np.array(self.sol.value(self.h)),
            'u':       np.array(self.sol.value(self.u)),
            'v':       np.array(self.sol.value(self.v)),
            'w':       np.array(self.sol.value(self.w)),
            'mu':      np.array(self.sol.value(self.mu)),
            'cl':      np.array(self.sol.value(self.cl)),
            'theta':   self.theta,
            'V_ref':   self.V_ref,
            'v_avg':   float(self.sol.value(self.v_avg)),
            'T_cycle': float(self.sol.value(self.T_cycle)),
        }

    def optimise(self):
        import os
        from contextlib import redirect_stdout, redirect_stderr
        opts = {
                'ipopt.max_iter': self.max_iters,
                'ipopt.mu_strategy': 'adaptive',
                'ipopt.tol': 1e-6,
                'ipopt.print_level': 5,
                }
        self.opti.solver('ipopt', opts)
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            print(f"Optimising: theta = {self.theta}, v_ref = {self.V_ref}")
            self.sol = self.opti.solve()
