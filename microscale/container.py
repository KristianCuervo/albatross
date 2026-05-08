"""
albatross.microscale.container — single-run result container.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Container:
    """
    Stores the result of a single optimisation run.

    All array fields have length N (number of collocation nodes).
    x, y are reconstructed by cumulative trapezoidal integration of u, v.
    """
    # States
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray

    x: np.ndarray
    y: np.ndarray
    h: np.ndarray

    # Controls
    cl: np.ndarray
    mu: np.ndarray

    # Scalars
    obj: float
    dt: float
    T_cycle: float

    # Run parameters
    theta: float
    V_ref: float
    N: int
    scheme: str = "trapezoidal"
    mode: str   = "max_vmg"

    # ------------------------------------------------------------------

    def as_ics(self, N_target: int | None = None) -> dict:
        """
        Return an ICs dict compatible with Solver._create_initial_condition.

        If N_target differs from self.N the arrays are resampled via periodic
        linear interpolation so the solution can seed a finer/coarser grid.
        """
        arrays = {'h': self.h, 'u': self.u, 'v': self.v,
                  'w': self.w, 'mu': self.mu, 'cl': self.cl}
        dt = self.dt

        if N_target is not None and N_target != self.N:
            t_old = np.arange(self.N + 1)
            t_new = np.linspace(0, self.N, N_target, endpoint=False)
            arrays = {k: np.interp(t_new, t_old, np.append(arr, arr[0]))
                      for k, arr in arrays.items()}
            dt = self.dt * self.N / N_target   # preserve T_cycle = dt * N

        return {**arrays, 'dt': dt}

    @classmethod
    def from_solution_dict(cls, d: dict) -> "Container":
        dt = d['dt']
        u  = d['u']
        v  = d['v']
        return cls(
            u        = u,
            v        = v,
            w        = d['w'],
            x        = np.cumsum(u) * dt,
            y        = np.cumsum(v) * dt,
            h        = d['h'],
            cl       = d['cl'],
            mu       = d['mu'],
            obj      = d['obj'],
            dt       = dt,
            T_cycle  = d['T_cycle'],
            theta    = d['theta'],
            V_ref    = d['V_ref'],
            N        = d['N'],
            scheme  = d.get('scheme', 'trapezoidal'),
            mode    = d.get('mode', 'max_vmg'),
        )
