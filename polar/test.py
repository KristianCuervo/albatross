from utils import parallel_sweep
from optimiser import Optimiser
from albatross import Albatross
import numpy as np

THETAS = np.linspace(0, 2 * np.pi, 6, endpoint=False)

if __name__ == "__main__":
    bird  = Albatross()
    proto = Optimiser(bird=bird, theta=0.0, N=64, V_ref=15.0)

    fam = parallel_sweep(opti=proto, thetas=THETAS)

    fam.plot_polar_attribute(attr='v_avg')
    fam.plot_polar_attribute(attr='T_cycle')
    fam.plot_attribute(attr='h')
    fam.plot_attribute(attr='cl')

    import matplotlib.pyplot as plt
    plt.show()
