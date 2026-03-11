import tomllib
from pathlib import Path
import casadi as ca

rho = 1.225
g = 9.80665


class Albatross:

    def __init__(self, config_path: str = None):
        self._load_data(config_path)

    def _load_data(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "data.toml"
        
        with open(config_path, "rb") as f:
            data = tomllib.load(f)["albatross"]
        
        self.m = data["m"]              # mass [kg]
        self.S = data["S"]              # wing area [m²]
        self.b = data["b"]              # wingspan [m]
        self.cd_0 = data["C_D0"]        # zero-lift drag coefficient
        self.k = data["k"]              # induced drag factor
        self.LD_max = data["LD_max"]    # maximum lift-to-drag ratio
