import tomllib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class Albatross:
    """
    Container for toml data. Change parameters by creating a new toml file
    and using the optional keyword config_path.

    Also visualises aerodynamic data.
    """


    def __init__(self, config_path: str = None):
        self._load_data(config_path)

    def _load_data(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "parameters" / "sachs_data.toml"
        
        with open(config_path, "rb") as f:
            data = tomllib.load(f)["albatross"]
        
        self.m = data["m"]              # mass [kg]
        self.S = data["S"]              # wing area [m²]
        self.b = data["b"]              # wingspan [m]
        self.cd_0 = data["C_D0"]        # zero-lift drag coefficient
        self.k = data["k"]              # induced drag factor
        self.LD_max = data["LD_max"]    # maximum lift-to-drag ratio
        self.cl_max = data["cl_max"]    # maximum lift coefficient
        self.cl_min = data["cl_min"]    # minimum lift coefficient

    def graph_drag_polar(self):
        cl_vals = np.linspace(self.cl_min, self.cl_max, 100)
        cd_vals = self.cd_0 + self.k * cl_vals**2

        plt.figure(figsize=(6, 4))
        plt.plot(cd_vals, cl_vals, lw=2)
        plt.axvline(self.cd_0, linestyle=':', color='black', label=r'$C_{d,0}$')
        plt.axhline(1.5, linestyle=':', color='red', label=r'$C_{L, max}$')
        plt.xlabel(r'$C_D$')
        plt.ylabel(r'$C_L$')
        plt.title('Drag Polar')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

