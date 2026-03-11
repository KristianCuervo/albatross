from trajectories import PathFamily
from optimiser import Optimiser
from albatross import Albatross
from multiprocessing.pool import Pool


def _sweep_worker(args):
    """Worker for parallel_sweep — creates and solves a fresh Optimiser."""
    bird, theta, N, V_ref = args
    opt = Optimiser(bird=bird, theta=theta, N=N, V_ref=V_ref)
    opt.optimise()
    print(f"t = {theta}, v_ref = {V_ref}")
    return opt.get_solution_dict()


def serial_sweep(opti: Optimiser, thetas) -> PathFamily:
    """Optimise for each theta in sequence; return a PathFamily."""
    paths = []
    for theta in thetas:
        opt = Optimiser(bird=opti.bird, theta=float(theta),
                        N=opti.N, V_ref=opti.V_ref)
        opt.optimise()
        paths.append(opt.get_solution_dict())
    return PathFamily(paths)

def parallel_sweep(opti: Optimiser, thetas, n_workers=8) -> PathFamily:
    """Optimise across thetas using multiprocessing (8 workers by default)."""
    args = [(opti.bird, float(t), opti.N, opti.V_ref) for t in thetas]
    with Pool(n_workers) as pool:
        paths = pool.map(_sweep_worker, args)
    return PathFamily(paths)
