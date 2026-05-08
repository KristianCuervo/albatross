"""
albatross.microscale.factory — batch parallel sweep + warm-starting.

Provides:
  Factory           : general batch solver for any mode
  sweep_min_vref    : convenience wrapper — V_ref_min(θ) polar sweep
  sweep_max_vmg     : convenience wrapper — VMG polar sweep at fixed V_ref levels
"""

from .solver import Solver
from .container import Container
from .ensemble import Ensemble
from .bird import Albatross

import numpy as np
import itertools
from multiprocessing import Pool
from typing import Sequence


class Factory:
    """
    Sweeps all (N, V_ref, theta) combinations for a fixed mode and scheme.

    Parameters
    ----------
    bird : Albatross
    N    : int or Sequence[int]
    V_ref : float or Sequence[float]
        Ignored (but still accepted for API compatibility) in min_vref mode.
    theta : array-like
        Heading angles [rad] to sweep.
    mode  : str  "max_vmg" or "min_vref"
    add_progress_constraint : bool
        Passed to Solver in min_vref mode.
    n_procs : int
        Number of parallel worker processes.
    warm_starts : list[Container] or None
        Containers used for warm-starting; nearest theta is chosen per job.
    scheme : str  Collocation scheme.
    """

    def __init__(
        self,
        bird: Albatross,
        N: int | Sequence[int],
        V_ref: float | Sequence[float],
        theta: np.ndarray,
        mode: str = "max_vmg",
        add_progress_constraint: bool = True,
        n_procs: int = 1,
        warm_starts: list[Container] | None = None,
        scheme: str = "trapezoidal",
    ):
        self.bird = bird
        self.N    = [N]     if isinstance(N, int)   else list(N)
        self.V_ref = [V_ref] if isinstance(V_ref, float) else list(V_ref)
        self.theta = np.asarray(theta)
        self.mode  = mode
        self.add_progress_constraint = add_progress_constraint
        self.n_procs = n_procs
        self.warm_starts = warm_starts
        self.scheme = scheme

    def _find_closest_ics(self, theta: float, n: int) -> dict | None:
        """Return ICs from the closest warm-start theta (wrap-around aware)."""
        if self.warm_starts is None:
            return None
        thetas = np.array([ws.theta for ws in self.warm_starts])
        diff   = np.abs(thetas - theta)
        diff   = np.minimum(diff, 2 * np.pi - diff)
        return self.warm_starts[int(np.argmin(diff))].as_ics(n)

    @staticmethod
    def _solve_one(args: tuple) -> Container | None:
        """Staticmethod for multiprocessing pickle compatibility."""
        bird, theta, N, V_ref, ics, scheme, mode, apc = args
        try:
            return Solver(
                bird=bird, theta=theta, N=N, V_ref=V_ref,
                mode=mode, add_progress_constraint=apc,
                ics=ics, scheme=scheme,
            ).optimise()
        except RuntimeError as e:
            print(f"[WARNING] Solver failed for theta={theta:.4f} rad: {e}")
            return None

    def run(self) -> Ensemble:
        """Sweep all (N, V_ref, theta) combinations and return an Ensemble."""
        ens = Ensemble()

        for n, v in itertools.product(self.N, self.V_ref):
            jobs = [
                (self.bird, t, n, v,
                 self._find_closest_ics(t, n),
                 self.scheme, self.mode, self.add_progress_constraint)
                for t in self.theta
            ]

            if self.n_procs == 1:
                containers = [self._solve_one(job) for job in jobs]
            else:
                with Pool(self.n_procs) as pool:
                    containers = pool.map(self._solve_one, jobs)

            for c in containers:
                if c is not None:
                    ens.add_container(c)

        return ens


# ------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------

def sweep_min_vref(
    bird: Albatross,
    thetas: np.ndarray,
    N: int = 40,
    add_progress_constraint: bool = True,
    n_procs: int = 1,
    scheme: str = "trapezoidal",
    V_ref_init: float = 10.0,
) -> Ensemble:
    """
    Polar sweep in min_vref mode: finds V_ref_min(θ) for each heading.

    Returns
    -------
    Ensemble where each container has V_ref = the minimum wind speed found.
    """
    return Factory(
        bird=bird,
        N=N,
        V_ref=V_ref_init,
        theta=thetas,
        mode="min_vref",
        add_progress_constraint=add_progress_constraint,
        n_procs=n_procs,
        scheme=scheme,
    ).run()


def sweep_max_vmg(
    bird: Albatross,
    thetas: np.ndarray,
    V_refs: Sequence[float],
    N: int = 40,
    n_procs: int = 1,
    warm_starts: list[Container] | None = None,
    scheme: str = "trapezoidal",
) -> Ensemble:
    """
    Polar sweep in max_vmg mode: finds maximum VMG for each (θ, V_ref) pair.

    Returns
    -------
    Ensemble of VMG-optimised containers; groups by V_ref for tacking diagram.
    """
    return Factory(
        bird=bird,
        N=N,
        V_ref=V_refs,
        theta=thetas,
        mode="max_vmg",
        n_procs=n_procs,
        warm_starts=warm_starts,
        scheme=scheme,
    ).run()
