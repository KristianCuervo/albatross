"""
albatross.bird — Albatross aerodynamic parameter dataclass.

Loads Sachs (2005) reference parameters from data/albatross.toml by default.
"""

from dataclasses import dataclass
import tomllib
from pathlib import Path
import numpy as np

# Default TOML path: data/albatross.toml at the repo root
_DEFAULT_TOML = "albaross.toml"


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
    def from_toml(cls, config_path: str | Path | None = None) -> "Albatross":
        """Load from TOML file. Defaults to data/albatross.toml."""
        if config_path is None:
            config_path = _DEFAULT_TOML
        with open(config_path, "rb") as f:
            data = tomllib.load(f)["albatross"]
        return cls(**data)

    @classmethod
    def from_nominal(cls, sigma: float = 0.05) -> "Albatross":
        """Create a stochastic variation of the nominal Sachs albatross."""
        nominal = cls.from_toml()
        rng = np.random.default_rng()
        return cls(
            m      = nominal.m      * rng.normal(1, sigma),
            S      = nominal.S      * rng.normal(1, sigma),
            b      = nominal.b      * rng.normal(1, sigma),
            cd_0   = nominal.cd_0   * rng.normal(1, sigma),
            k      = nominal.k      * rng.normal(1, sigma),
            LD_max = nominal.LD_max * rng.normal(1, sigma),
            cl_max = nominal.cl_max * rng.normal(1, sigma),
            cl_min = nominal.cl_min * rng.normal(1, sigma),
        )
