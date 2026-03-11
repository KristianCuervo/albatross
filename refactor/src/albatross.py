from dataclasses import dataclass
import tomllib
from pathlib import Path
import numpy as np

@dataclass
class Albatross:
    m: float        # mass [kg]
    S: float        # wing area [m²]
    b: float        # wingspan [m]
    cd_0: float     # zero-lift drag coefficient
    k: float        # induced drag factor
    LD_max: float   # maximum lift-to-drag ratio
    cl_max: float   # maximum lift coefficient
    cl_min: float   # minimum lift coefficient

    @classmethod
    def from_toml(cls, config_path: str = None) -> "Albatross":
        if config_path is None:
            config_path = Path(__file__).parent / "parameters" / "sachs_data.toml"
        with open(config_path, "rb") as f:
            data = tomllib.load(f)["albatross"]
        return cls(**data)

    @classmethod
    def from_nominal(cls, sigma: float = 0.05) -> "Albatross":
        """Create a stochastic variation of the nominal Sachs albatross."""
        nominal = cls.from_toml()
        return cls(
            m      = nominal.m      * np.random.normal(1, sigma),
            S      = nominal.S      * np.random.normal(1, sigma),
            b      = nominal.b      * np.random.normal(1, sigma),
            cd_0   = nominal.cd_0   * np.random.normal(1, sigma),
            k      = nominal.k      * np.random.normal(1, sigma),
            LD_max = nominal.LD_max * np.random.normal(1, sigma),
            cl_max = nominal.cl_max * np.random.normal(1, sigma),
            cl_min = nominal.cl_min * np.random.normal(1, sigma),
        )