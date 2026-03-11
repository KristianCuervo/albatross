from .solver import Solver
from .container import Container
from .ensemble import Ensemble
from .albatross import Albatross

import numpy as np
import itertools
from multiprocessing import Pool
from typing import Sequence


class Factory:
    """
    Creates a polar sweep of Solver instances for each (N, V_ref) combination,
    collects the resulting Containers into an Ensemble per combination.
    """

    def __init__(self,
                 bird: Albatross,
                 N: Sequence[int],
                 V_ref: Sequence[float],
                 theta: np.ndarray,
                 n_procs: int = 1,
                 warm_starts: list[Container] | None = None,
                 scheme: str = "trapezoidal"):
        self.bird = bird
        self.N = N
        self.V_ref = V_ref
        self.theta = theta
        self.n_procs = n_procs
        self.warm_starts = warm_starts
        self.scheme = scheme

    def _find_closest_ics(self, theta: float, n: int) -> dict | None:
        """Return the ics dict from the warm start whose theta is closest to the given theta.
        Handles wrap-around (theta=0 ≈ theta=2π) and resamples arrays to length n if needed."""
        if self.warm_starts is None:
            return None
        thetas = np.array([ws.theta for ws in self.warm_starts])
        diff   = np.abs(thetas - theta)
        diff   = np.minimum(diff, 2 * np.pi - diff)   # wrap-around at 0/2π boundary
        return self.warm_starts[np.argmin(diff)].as_ics(n)

    @staticmethod
    def _solve_one(args: tuple) -> Container | None:
        """Create and run a single Solver. Staticmethod for multiprocessing pickle compatibility."""
        bird, theta, N, V_ref, ics, scheme = args
        try:
            return Solver(bird=bird, theta=theta, N=N, V_ref=V_ref, ics=ics, scheme=scheme).optimise()
        except RuntimeError as e:
            print(f"[WARNING] Solver failed for theta={theta:.4f} rad: {e}")
            return None

    def run(self) -> Ensemble:
        """
        Sweeps all (N, V_ref, theta) combinations and collects all results
        into a single Ensemble.
        """
        ens = Ensemble()

        for n, v in itertools.product(self.N, self.V_ref):
            jobs = [(self.bird, t, n, v, self._find_closest_ics(t, n), self.scheme) for t in self.theta]

            if self.n_procs == 1:
                containers = [self._solve_one(job) for job in jobs]
            else:
                with Pool(self.n_procs) as pool:
                    containers = pool.map(self._solve_one, jobs)

            for c in containers:
                if c is not None:
                    ens.add_container(c)

        return ens
