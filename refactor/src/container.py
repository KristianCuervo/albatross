from dataclasses import dataclass
import numpy as np

@dataclass
class Container:
    """
    Contains the results from a single optimisation run.
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

    # Scalar results
    obj: float
    dt: float
    T_cycle: float

    # Run parameters
    theta: float
    V_ref: float
    N: int
    scheme: str = "trapezoidal"

    def as_ics(self, N_target: int | None = None) -> dict:
        """
        Return an ICs dict compatible with Solver._create_initial_condition.
        If N_target differs from self.N the arrays are resampled via periodic
        linear interpolation so the same solution can seed a finer/coarser grid.
        """
        arrays = {'h': self.h, 'u': self.u, 'v': self.v,
                  'w': self.w, 'mu': self.mu, 'cl': self.cl}
        dt = self.dt

        if N_target is not None and N_target != self.N:
            # Periodic interpolation: append first element to close the loop
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
            u       = u,
            v       = v,
            w       = d['w'],
            x       = np.cumsum(u) * dt,
            y       = np.cumsum(v) * dt,
            h       = d['h'],
            cl      = d['cl'],
            mu      = d['mu'],
            obj     = d['obj'],
            dt      = dt,
            T_cycle = d['T_cycle'],
            theta   = d['theta'],
            V_ref   = d['V_ref'],
            N       = d['N'],
            scheme  = d.get('scheme', 'trapezoidal'),
        )
